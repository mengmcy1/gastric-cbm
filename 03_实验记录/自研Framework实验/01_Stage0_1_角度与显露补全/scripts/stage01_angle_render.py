from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch

from stage01_common import (
    angle_sequence,
    camera_info_for_angle,
    configure_sharp_cuda_toolkit,
    create_camera_context,
    file_hash,
    prepare_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自研 Stage 0 角度渲染：显式区分旧横移协议与真圆弧协议。"
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-total", type=float, choices=[5.0, 15.0, 30.0], required=True)
    parser.add_argument("--num-steps", type=int, default=61)
    parser.add_argument(
        "--trajectory-mode", choices=["legacy_lateral", "true_arc"], required=True
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def project_points(points: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor):
    points_camera = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    depth = points_camera[:, 2]
    safe = depth.clamp_min(1e-8)
    x = intrinsics[0, 0] * points_camera[:, 0] / safe + intrinsics[0, 2]
    y = intrinsics[1, 1] * points_camera[:, 1] / safe + intrinsics[1, 2]
    return torch.stack([x, y], dim=1), depth


def percentile_nearest(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q, interpolation="nearest"))


def main() -> None:
    args = parse_args()
    if not args.ply.is_file():
        raise FileNotFoundError(args.ply)
    if not torch.cuda.is_available():
        raise RuntimeError("角度渲染要求 CUDA。")
    output_dir = prepare_output_dir(args.output_dir)
    configure_sharp_cuda_toolkit()
    from sharp.utils import gsplat, vis

    gaussians_cpu, metadata, camera_model = create_camera_context(args.ply)
    gaussians = gaussians_cpu.to(torch.device("cuda"))
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    angles = angle_sequence(args.angle_total, args.num_steps)
    selected_indices = {0: "left", len(angles) // 2: "center", len(angles) - 1: "right"}

    center_eye, center_info = camera_info_for_angle(camera_model, 0.0, args.trajectory_mode)
    width, height = int(center_info.width), int(center_info.height)
    if width % 2 or height % 2:
        raise ValueError(f"H.264 yuv420p 要求偶数尺寸，当前 {width}x{height}")
    points = gaussians.mean_vectors[0]
    center_uv, center_z = project_points(
        points, center_info.extrinsics.to(points.device), center_info.intrinsics.to(points.device)
    )

    writer = iio.get_writer(
        output_dir / "color.mp4",
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
    )
    rows = []
    start = time.perf_counter()
    try:
        for index, angle_deg in enumerate(angles):
            eye, camera_info = camera_info_for_angle(
                camera_model, float(angle_deg), args.trajectory_mode
            )
            render_start = time.perf_counter()
            with torch.inference_mode():
                result = renderer(
                    gaussians,
                    extrinsics=camera_info.extrinsics[None].to("cuda"),
                    intrinsics=camera_info.intrinsics[None].to("cuda"),
                    image_width=camera_info.width,
                    image_height=camera_info.height,
                )
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - render_start) * 1000.0
            rgb = (
                result.color[0].permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            )
            alpha = result.alpha[0, 0].float().cpu().numpy()
            writer.append_data(rgb)

            uv, z = project_points(
                points,
                camera_info.extrinsics.to(points.device),
                camera_info.intrinsics.to(points.device),
            )
            valid = (
                (center_z > 0)
                & (z > 0)
                & torch.isfinite(uv).all(dim=1)
                & torch.isfinite(center_uv).all(dim=1)
                & (center_uv[:, 0] >= 0)
                & (center_uv[:, 0] < width)
                & (center_uv[:, 1] >= 0)
                & (center_uv[:, 1] < height)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            displacement = torch.linalg.norm(uv[valid] - center_uv[valid], dim=1)
            p95 = percentile_nearest(displacement, 0.95)
            rows.append(
                {
                    "frame": index,
                    "angle_deg": float(angle_deg),
                    "eye_x": float(eye[0]),
                    "eye_y": float(eye[1]),
                    "eye_z": float(eye[2]),
                    "render_ms": render_ms,
                    "alpha_ge_099": float((alpha >= 0.99).mean()),
                    "alpha_ge_095": float((alpha >= 0.95).mean()),
                    "alpha_ge_050": float((alpha >= 0.50).mean()),
                    "projection_p95_px": p95,
                    "projection_p95_width_fraction": p95 / width,
                    "valid_gaussian_count": int(valid.sum()),
                }
            )

            if index in selected_indices:
                label = selected_indices[index]
                iio.imwrite(output_dir / f"frame_{label}.png", rgb)
                iio.imwrite(
                    output_dir / f"alpha_{label}_u16.png",
                    np.round(np.clip(alpha, 0, 1) * 65535).astype(np.uint16),
                )
                depth = result.depth[0, 0].float().cpu().numpy()
                np.save(output_dir / f"depth_{label}_float32.npy", depth)
                depth_color = vis.colorize_depth(result.depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
                iio.imwrite(output_dir / f"depth_{label}.png", depth_color)
    finally:
        writer.close()

    with (output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    write_json(
        output_dir / "config.json",
        {
            "schema_version": "1.1-angle-protocol",
            "script": "stage01_angle_render.py",
            "ply": str(args.ply.resolve()),
            "ply_sha256": file_hash(args.ply),
            "angle_total_deg": args.angle_total,
            "angle_half_deg": args.angle_total / 2.0,
            "num_steps": args.num_steps,
            "trajectory_mode": args.trajectory_mode,
            "trajectory_definition": (
                "legacy_lateral: eye=(focus*tan(a),0,0), reproduces the existing P01 baseline; "
                "true_arc: eye=(focus*sin(a),0,focus*(1-cos(a))), circle center/look-at=(0,0,focus)."
            ),
            "focus_depth": float(camera_model.depth_quantiles.focus),
            "center_eye_xyz": [float(value) for value in center_eye],
            "render_resolution_wh": [width, height],
            "gaussian_count": int(points.shape[0]),
            "fps": args.fps,
            "total_seconds": time.perf_counter() - start,
            "endpoints": {"left": rows[0], "right": rows[-1]},
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "frames": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
