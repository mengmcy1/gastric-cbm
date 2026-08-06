"""SHARP 角度标注横移基线：量化 SHARP 在 5°/15°/30° 下的显露压力。

v1（2026-08-06）：
- 轨迹为 z=0 平面横移近似（legacy_lateral），不是严格水平圆弧。
- 横移量按 x = focus_depth * tan(angle) 计算，总扫视角由 --angle-total 指定（5/15/30）。
- 相机始终看向场景中心（lookat_mode="point"，与官方一致）。
- 30° 下高斯大量移出视锥产生空洞是预期行为，正是本实验要量化的目标。
- 独立于官方源码和已有实验脚本，不修改任何既有文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))

# 当前项目将 CUDA Toolkit 安装在 Conda 的 sharp 环境内。
CONDA_PREFIX = Path(sys.prefix)
CUDA_TOOLKIT = CONDA_PREFIX / "Library"
CUDA_BIN = CUDA_TOOLKIT / "bin"
if (CUDA_BIN / "nvcc.exe").is_file():
    os.environ.setdefault("CUDA_HOME", str(CUDA_TOOLKIT))
    os.environ.setdefault("CUDA_PATH", str(CUDA_TOOLKIT))
    os.environ["PATH"] = str(CUDA_BIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(CUDA_BIN))

from sharp.utils import camera, gsplat, vis  # noqa: E402
from sharp.utils.gaussians import load_ply  # noqa: E402

_SSIM_MAX_SIDE = 512
_ANGLE_CHOICES = (5.0, 15.0, 30.0)

VIDEO_MANUAL_TEMPLATE = {
    "schema_version": "3.0-angle",
    "angle_total_deg": None,
    "review_status": "",
    "overall_quality_score": None,
    "hole_severity": None,
    "stretching_severity": None,
    "flicker_severity": None,
    "paper_feel_severity": None,
    "occlusion_error_severity": None,
    "reflection_deformation_severity": None,
    "first_artifact_frame_left": None,
    "first_artifact_frame_right": None,
    "worst_frame": None,
    "overall_pass": None,
    "notes": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-total", type=float, choices=_ANGLE_CHOICES, required=True,
                        help="水平总扫视角（度），5/15/30")
    parser.add_argument("--num-steps", type=int, default=61)
    parser.add_argument("--radius-ratio", type=float, default=1.0,
                        help="兼容旧命令保留；legacy_lateral 轨迹不使用该参数")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def project_points(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 3D 点投影到像素坐标，并返回未截断的相机空间深度。"""
    extrinsics = extrinsics.to(points.device)
    intrinsics = intrinsics.to(points.device)
    points_camera = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    depth = points_camera[:, 2]
    safe_depth = depth.clamp_min(1e-8)
    x = intrinsics[0, 0] * points_camera[:, 0] / safe_depth + intrinsics[0, 2]
    y = intrinsics[1, 1] * points_camera[:, 1] / safe_depth + intrinsics[1, 2]
    return torch.stack((x, y), dim=-1), depth


def angle_swipe_eye_positions(focus_depth: float, angle_total_deg: float, num_steps: int) -> list[torch.Tensor]:
    """生成角度 swipe 轨迹的相机位置（世界坐标）。

    相机在照片平面（z=0）横向滑动，x = focus_depth * tan(a)，
    a 从 -angle/2 扫到 +angle/2。相机始终看向场景中心 (0,0,focus_depth)。

    这样：
    - 中心帧（a=0）相机位于 (0,0,0)，等于原始视角，Alpha 可与静止基线直接对照；
    - 两端视线方向偏转 ±angle/2，总视角严格等于 angle_total_deg；
    - 与已有 Benchmark 的 swipe 轨迹同族，只是把 max_disparity 换成显式角度。
    """
    half_rad = math.radians(angle_total_deg) / 2.0
    angles = np.linspace(-half_rad, half_rad, num_steps)
    return [
        torch.tensor([focus_depth * math.tan(a), 0.0, 0.0], dtype=torch.float32)
        for a in angles
    ]


def save_diagnostics(output_dir: Path, label: str, color: torch.Tensor, depth: torch.Tensor, alpha: torch.Tensor) -> None:
    color_u8 = (color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    alpha_u16 = (alpha[0, 0].clamp(0, 1) * 65535).to(torch.uint16).cpu().numpy()
    depth_np = depth[0, 0].detach().float().cpu().numpy()
    write_png(output_dir / f"frame_{label}.png", color_u8)
    write_png(output_dir / f"alpha_{label}_u16.png", alpha_u16)
    np.save(output_dir / f"depth_{label}_float32.npy", depth_np)
    depth_color = vis.colorize_depth(depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
    write_png(output_dir / f"depth_{label}.png", depth_color)


def _downsample_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / max(h, w)
    new_size = (round(w * scale), round(h * scale))
    return np.asarray(Image.fromarray(rgb).resize(new_size, Image.Resampling.BICUBIC))


def _compute_adjacent_ssim(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    prev_small = _downsample_rgb(prev_rgb, _SSIM_MAX_SIDE)
    curr_small = _downsample_rgb(curr_rgb, _SSIM_MAX_SIDE)
    return float(ssim(prev_small, curr_small, channel_axis=2, data_range=255))


def _alpha_coverage(alpha: torch.Tensor) -> dict:
    a = alpha[0, 0].float().cpu().numpy()
    return {
        "alpha_ge_099": round(float((a >= 0.99).mean()), 6),
        "alpha_ge_095": round(float((a >= 0.95).mean()), 6),
        "alpha_ge_090": round(float((a >= 0.90).mean()), 6),
        "alpha_ge_050": round(float((a >= 0.50).mean()), 6),
    }


_METRICS_COLUMNS = [
    "frame", "angle_deg", "eye_x", "eye_y", "eye_z",
    "render_ms",
    "alpha_ge_099", "alpha_ge_095", "alpha_ge_090", "alpha_ge_050",
    "valid_gaussian_count",
    "rejected_nonpositive_depth_count",
    "adjacent_ssim",
]

_MANUAL_COLUMNS = [
    "frame", "hole", "stretching", "flicker", "thin_structure_break",
    "reflection_error", "severity", "notes",
]


def main() -> None:
    args = parse_args()
    if args.num_steps < 2:
        raise ValueError("--num-steps 至少为 2，否则无法形成扫视轨迹")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本实验入口要求可用 CUDA；请检查 CUDA_VISIBLE_DEVICES 与 sharp 环境。")
    if not args.ply.is_file():
        raise FileNotFoundError(args.ply)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"输出目录非空，按正式实验规则停止以防覆盖：{output_dir}\n"
            f"已有 {len(existing)} 个文件/子目录。"
        )

    device = torch.device("cuda")
    gaussians_cpu, metadata = load_ply(args.ply)
    ply_w, ply_h = int(metadata.resolution_px[0]), int(metadata.resolution_px[1])
    f_px = float(metadata.focal_length_px)
    intrinsics = torch.tensor(
        [[f_px, 0, (ply_w - 1) / 2, 0],
         [0, f_px, (ply_h - 1) / 2, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    camera_model = camera.create_camera_model(gaussians_cpu, intrinsics, resolution_px=metadata.resolution_px)

    # 角度 swipe：相机在 z=0 平面横向滑动，x = focus_depth * tan(±half_angle)
    focus_depth = float(camera_model.depth_quantiles.focus)
    eye_positions = angle_swipe_eye_positions(focus_depth, args.angle_total, args.num_steps)

    # ---- 输出目录 config 记录 ----
    config = {
        "script": "sharp_angle_protocol_render.py",
        "script_version": "v1",
        "ply": str(args.ply.resolve()),
        "trajectory": "legacy_lateral",
        "angle_total_deg": args.angle_total,
        "angle_half_deg": args.angle_total / 2,
        "num_steps": len(eye_positions),
        "radius_ratio_requested": args.radius_ratio,
        "radius_ratio_applied": False,
        "focus_depth": round(focus_depth, 6),
        "max_lateral_offset_m": round(focus_depth * math.tan(math.radians(args.angle_total) / 2), 6),
        "fps": args.fps,
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "ply_metadata_resolution": [ply_w, ply_h],
        "device": str(device),
        "gaussian_count": int(gaussians_cpu.mean_vectors.shape[1]),
        "angle_definition": (
            "Angle swipe: camera slides laterally in the photo plane (z=0), "
            "x = focus_depth * tan(a). "
            f"Total sweep angle = {args.angle_total} deg (+/- {args.angle_total/2} deg). "
            "Center frame equals the original viewpoint. "
            "Camera always looks at scene center (lookat_mode=point). "
            "Expected large holes at 30 deg are the measured artifact, not a bug."
        ),
        "note": (
            "Legacy lateral angle-labelled baseline; not a true circular arc and not directly "
            "comparable to max_disparity proxy or true_arc results."
        )
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 渲染 ----
    gaussians = gaussians_cpu.to(device)
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    center_info = camera_model.compute(torch.zeros(3, dtype=torch.float32))
    render_w = center_info.width
    render_h = center_info.height

    if render_w % 2 or render_h % 2:
        raise ValueError(f"yuv420p 要求宽高为偶数，当前渲染分辨率为 {render_w}×{render_h}")
    video_writer_kwargs = {
        "fps": args.fps,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 2,
    }
    color_writer = iio.get_writer(output_dir / "color.mp4", **video_writer_kwargs)
    depth_writer = iio.get_writer(output_dir / "depth.mp4", **video_writer_kwargs)

    n = len(eye_positions)
    selected = {
        0: "left",
        n - 1: "right",
        min(range(n), key=lambda i: abs(i - (n - 1) / 2)): "center",
    }

    half_rad = math.radians(args.angle_total) / 2.0
    angles_deg = np.degrees(np.linspace(-half_rad, half_rad, n))

    per_frame_json = []
    per_frame_metrics = []
    prev_rgb = None
    start = time.perf_counter()
    try:
        for index, eye in enumerate(eye_positions):
            camera_info = camera_model.compute(eye)
            render_start = time.perf_counter()
            with torch.inference_mode():
                result = renderer(
                    gaussians,
                    extrinsics=camera_info.extrinsics[None].to(device),
                    intrinsics=camera_info.intrinsics[None].to(device),
                    image_width=camera_info.width,
                    image_height=camera_info.height,
                )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - render_start) * 1000

            color_u8 = (result.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            depth_color = vis.colorize_depth(result.depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
            color_writer.append_data(color_u8)
            depth_writer.append_data(depth_color)

            alpha_cov = _alpha_coverage(result.alpha)

            if prev_rgb is not None:
                adj_ssim = _compute_adjacent_ssim(prev_rgb, color_u8)
            else:
                adj_ssim = None
            prev_rgb = color_u8

            row = {
                "frame": index,
                "angle_deg": round(float(angles_deg[index]), 6),
                "eye_x": round(float(eye[0]), 6),
                "eye_y": round(float(eye[1]), 6),
                "eye_z": round(float(eye[2]), 6),
                "render_ms": round(elapsed_ms, 3),
                **alpha_cov,
                "adjacent_ssim": round(adj_ssim, 6) if adj_ssim is not None else None,
            }
            per_frame_json.append(row)
            per_frame_metrics.append([
                index,
                round(float(angles_deg[index]), 6),
                round(float(eye[0]), 6),
                round(float(eye[1]), 6),
                round(float(eye[2]), 6),
                round(elapsed_ms, 3),
                alpha_cov["alpha_ge_099"],
                alpha_cov["alpha_ge_095"],
                alpha_cov["alpha_ge_090"],
                alpha_cov["alpha_ge_050"],
                "",  # valid_gaussian_count 暂不统计（角度协议聚焦 Alpha 空洞）
                "",
                round(adj_ssim, 6) if adj_ssim is not None else "",
            ])

            if index in selected:
                save_diagnostics(output_dir, selected[index], result.color, result.depth, result.alpha)
    finally:
        color_writer.close()
        depth_writer.close()

    # ---- 逐帧指标 CSV ----
    with (output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_METRICS_COLUMNS)
        writer.writerows(per_frame_metrics)

    # ---- 人工标记 CSV 模板 ----
    with (output_dir / "per_frame_manual.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_MANUAL_COLUMNS)
        for i in range(len(eye_positions)):
            writer.writerow([i, "", "", "", "", "", "", ""])

    # ---- 人工评分模板 ----
    vm = dict(VIDEO_MANUAL_TEMPLATE)
    vm["angle_total_deg"] = args.angle_total
    (output_dir / "video_manual.json").write_text(
        json.dumps(vm, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- projection_proxy.json（保留逐帧机器记录） ----
    (output_dir / "projection_proxy.json").write_text(
        json.dumps(per_frame_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- config.json 追加渲染结果 ----
    config["render_resolution"] = [int(render_w), int(render_h)]
    config["total_seconds"] = round(time.perf_counter() - start, 3)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总关键 Alpha 指标
    left = per_frame_json[0]
    right = per_frame_json[-1]
    center = per_frame_json[min(range(n), key=lambda i: abs(i - (n - 1) / 2))]
    print(json.dumps({
        "angle_total_deg": args.angle_total,
        "trajectory": "legacy_lateral",
        "radius_m": None,
        "max_lateral_offset_m": round(
            focus_depth * math.tan(math.radians(args.angle_total) / 2), 4
        ),
        "left_alpha_ge_099": left["alpha_ge_099"],
        "left_alpha_ge_050": left["alpha_ge_050"],
        "center_alpha_ge_099": center["alpha_ge_099"],
        "right_alpha_ge_099": right["alpha_ge_099"],
        "right_alpha_ge_050": right["alpha_ge_050"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
