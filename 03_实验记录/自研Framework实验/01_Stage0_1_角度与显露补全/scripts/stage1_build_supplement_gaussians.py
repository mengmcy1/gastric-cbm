from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from stage01_common import (
    camera_info_for_angle,
    camera_reference_depth,
    create_camera_context,
    endpoint_angle_deg,
    file_hash,
    prepare_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将补全后的 RGB-D-Alpha 端点显露区域反投影为补充高斯，并可与基础 PLY 融合。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="从单个端点生成补充高斯 PLY")
    build.add_argument("--base-ply", type=Path, required=True)
    build.add_argument("--rgb", type=Path, required=True)
    build.add_argument("--depth", type=Path, required=True)
    build.add_argument("--mask", type=Path, required=True)
    build.add_argument("--angle-total", type=float, required=True)
    build.add_argument("--side", choices=["left", "right"], required=True)
    build.add_argument(
        "--trajectory-mode", choices=["legacy_lateral", "true_arc"], required=True
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--stride", type=int, default=4)
    build.add_argument("--scale-factor", type=float, default=0.85)
    build.add_argument("--thickness-ratio", type=float, default=0.25)
    build.add_argument("--opacity-min", type=float, default=0.55)
    build.add_argument("--opacity-max", type=float, default=0.90)
    build.add_argument("--fade-distance-px", type=float, default=24.0)

    merge = subparsers.add_parser("merge", help="将一个或多个补充 PLY 与同一基础 PLY 拼接")
    merge.add_argument("--base-ply", type=Path, required=True)
    merge.add_argument("--supplement-ply", type=Path, action="append", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def concatenate(items):
    from sharp.utils.gaussians import Gaussians3D

    return Gaussians3D(
        mean_vectors=torch.cat([item.mean_vectors for item in items], dim=1),
        singular_values=torch.cat([item.singular_values for item in items], dim=1),
        quaternions=torch.cat([item.quaternions for item in items], dim=1),
        colors=torch.cat([item.colors for item in items], dim=1),
        opacities=torch.cat([item.opacities for item in items], dim=1),
    )


def run_build(args: argparse.Namespace) -> None:
    from sharp.utils import color_space as cs_utils
    from sharp.utils import linalg
    from sharp.utils.gaussians import Gaussians3D, save_ply

    for path in (args.base_ply, args.rgb, args.depth, args.mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.stride < 1:
        raise ValueError("stride 至少为 1。")
    output_dir = prepare_output_dir(args.output_dir)
    _, metadata, camera_model = create_camera_context(args.base_ply)
    angle_deg = endpoint_angle_deg(args.angle_total, args.side)
    eye, camera_info = camera_info_for_angle(camera_model, angle_deg, args.trajectory_mode)

    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    depth = np.load(args.depth).astype(np.float32)
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError("RGB、深度与掩码分辨率不一致。")
    if (rgb.shape[1], rgb.shape[0]) != (camera_info.width, camera_info.height):
        raise ValueError(
            f"输入分辨率 {rgb.shape[1]}x{rgb.shape[0]} 与渲染相机 "
            f"{camera_info.width}x{camera_info.height} 不一致。"
        )

    sample_mask = np.zeros_like(mask)
    sample_mask[:: args.stride, :: args.stride] = True
    selected = mask & sample_mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(selected)
    intrinsics = camera_info.intrinsics.float().cpu()
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if ys.size == 0:
        reference_depth = camera_reference_depth(camera_model)
        # SHARP 的 PLY 写入器不能序列化 0 个顶点（内部会计算深度分位数）。
        # 因此保存一个几何极小、近零不透明度的存储哨兵；统计中的有效补充数量仍为 0。
        supplement = Gaussians3D(
            mean_vectors=torch.tensor([[[0.0, 0.0, reference_depth]]], dtype=torch.float32),
            singular_values=torch.full((1, 1, 3), 1e-6, dtype=torch.float32),
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.zeros((1, 1, 3), dtype=torch.float32),
            opacities=torch.full((1, 1), 1e-8, dtype=torch.float32),
        )
        depth_stats = {"min": None, "median": None, "max": None}
    else:
        z = torch.from_numpy(depth[ys, xs]).float()
        x_cam = (torch.from_numpy(xs).float() - cx) * z / fx
        y_cam = (torch.from_numpy(ys).float() - cy) * z / fy
        points_camera = torch.stack([x_cam, y_cam, z], dim=1)

        extrinsics = camera_info.extrinsics.float().cpu()
        rotation_world_to_camera = extrinsics[:3, :3]
        translation = extrinsics[:3, 3]
        points_world = (points_camera - translation) @ rotation_world_to_camera

        scale_x = (z / fx) * args.stride * args.scale_factor
        scale_y = (z / fy) * args.stride * args.scale_factor
        scale_z = torch.minimum(scale_x, scale_y) * args.thickness_ratio
        scales = torch.stack([scale_x, scale_y, scale_z], dim=1).clamp_min(1e-6)

        local_to_world = rotation_world_to_camera.T
        rotations = local_to_world[None].expand(ys.size, -1, -1).contiguous()
        quaternions = linalg.quaternions_from_rotation_matrices(rotations)

        colors_srgb = torch.from_numpy(rgb[ys, xs].copy()).float() / 255.0
        colors_linear = cs_utils.sRGB2linearRGB(colors_srgb)
        distance = ndi.distance_transform_edt(mask)[ys, xs]
        confidence = np.clip(distance / max(args.fade_distance_px, 1e-6), 0.0, 1.0)
        opacity = args.opacity_min + (args.opacity_max - args.opacity_min) * confidence
        opacities = torch.from_numpy(opacity.astype(np.float32))

        supplement = Gaussians3D(
            mean_vectors=points_world[None],
            singular_values=scales[None],
            quaternions=quaternions[None],
            colors=colors_linear[None],
            opacities=opacities[None],
        )
        depth_stats = {
            "min": float(z.min()),
            "median": float(z.median()),
            "max": float(z.max()),
        }
    output_ply = output_dir / "supplement.ply"
    base_w, base_h = map(int, metadata.resolution_px)
    save_ply(supplement, float(metadata.focal_length_px), (base_h, base_w), output_ply)
    write_json(
        output_dir / "supplement_stats.json",
        {
            "schema_version": "1.0-rgbd-supplement-gaussians",
            "base_ply": str(args.base_ply.resolve()),
            "base_ply_sha256": file_hash(args.base_ply),
            "inputs": {
                "rgb": str(args.rgb.resolve()),
                "depth": str(args.depth.resolve()),
                "mask": str(args.mask.resolve()),
            },
            "camera": {
                "trajectory_mode": args.trajectory_mode,
                "angle_total_deg": args.angle_total,
                "side": args.side,
                "angle_deg": angle_deg,
                "eye_xyz": [float(value) for value in eye],
                "render_resolution_wh": [int(camera_info.width), int(camera_info.height)],
                "focal_xy_px": [fx, fy],
            },
            "parameters": {
                "stride": args.stride,
                "scale_factor": args.scale_factor,
                "thickness_ratio": args.thickness_ratio,
                "opacity_min": args.opacity_min,
                "opacity_max": args.opacity_max,
                "fade_distance_px": args.fade_distance_px,
            },
            "mask_pixels": int(mask.sum()),
            "source_identity": args.side,
            "supplement_gaussian_count": int(ys.size),
            "storage_sentinel_count": 1 if ys.size == 0 else 0,
            "storage_sentinel_opacity": 1e-8 if ys.size == 0 else None,
            "depth_m": depth_stats,
            "status": "no_op_empty_supplement" if ys.size == 0 else "generated",
            "limitations": (
                "第一版按规则从单端点 RGB-D 生成各向异性表面高斯；"
                "来源身份已保留，但跨端点冲突消解属于 Stage 2。"
            ),
        },
    )


def run_merge(args: argparse.Namespace) -> None:
    from sharp.utils.gaussians import load_ply, save_ply

    for path in (args.base_ply, *args.supplement_ply):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = prepare_output_dir(args.output_dir)
    base, metadata = load_ply(args.base_ply)
    supplements = []
    for path in args.supplement_ply:
        supplement, _ = load_ply(path)
        supplements.append(supplement)
    merged = concatenate([base, *supplements])
    next_index = int(base.mean_vectors.shape[1])
    ranges = {"base": [0, next_index]}
    supplement_records = []
    for supplement_index, (path, item) in enumerate(zip(args.supplement_ply, supplements)):
        count = int(item.mean_vectors.shape[1])
        stats_path = path.parent / "supplement_stats.json"
        source = f"supplement_{supplement_index}"
        if stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as handle:
                source = str(json.load(handle).get("source_identity", source))
        start, end = next_index, next_index + count
        ranges[source] = [start, end]
        supplement_records.append({
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "source_identity": source,
            "count": count,
            "index_range_start_inclusive_end_exclusive": [start, end],
        })
        next_index = end
    base_w, base_h = map(int, metadata.resolution_px)
    output_ply = output_dir / "scene_base_plus_supplement.ply"
    save_ply(merged, float(metadata.focal_length_px), (base_h, base_w), output_ply)
    write_json(
        output_dir / "merge_stats.json",
        {
            "schema_version": "1.0-append-supplements",
            "base_ply": str(args.base_ply.resolve()),
            "base_count": int(base.mean_vectors.shape[1]),
            "supplements": supplement_records,
            "index_ranges_start_inclusive_end_exclusive": ranges,
            "merged_count": int(merged.mean_vectors.shape[1]),
            "merge_method": "append_only_no_deduplication",
            "note": "Stage 1 仅追加并保留 base/left/right 索引范围；未完成跨端点冲突消解，后者属于 Stage 2。",
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "build":
        run_build(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
