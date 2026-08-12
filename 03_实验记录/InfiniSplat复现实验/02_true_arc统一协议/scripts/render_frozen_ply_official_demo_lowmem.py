#!/usr/bin/env python3
"""Reproduce the official demo trajectory from an official PLY, one frame at a time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_frozen_ply_true_arc import (  # noqa: E402
    REPO_ROOT,
    linearRGB2sRGB,
    load_infinisplat_ply,
    load_json,
    look_at_c2w,
    rasterization,
    resolve_repo_path,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="低显存复现 InfiniSplat 官方 60 帧演示轨迹。"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("trajectory_mode") != "official_demo":
        raise ValueError("该入口只允许 trajectory_mode=official_demo。")

    ply_path = resolve_repo_path(config["ply_path"])
    output_dir = resolve_repo_path(config["output_dir"])
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_hash = sha256(ply_path)
    if actual_hash != config["ply_sha256"]:
        raise RuntimeError(f"PLY SHA256 不匹配：{actual_hash}")

    device = torch.device(config["device"])
    gaussians_cpu, _, _ = load_infinisplat_ply(ply_path)
    gaussians = gaussians_cpu.to(device)
    height, width = (int(value) for value in config["render_image_shape_hw"])
    intrinsics = torch.tensor(config["render_intrinsics_px"], dtype=torch.float32, device=device)
    focal = float(intrinsics[0, 0])

    points = gaussians.mean_vectors[0]
    valid = points[torch.isfinite(points).all(dim=-1) & (points[:, 2] > 1e-4)]
    if valid.numel() == 0:
        raise ValueError("没有可用的正深度高斯。")
    look_at = valid.median(dim=0).values
    min_depth = torch.quantile(valid[:, 2], 0.1).clamp_min(1e-3)
    diagonal = math.sqrt((width / focal) ** 2 + (height / focal) ** 2)
    lateral = 0.08 * diagonal * float(min_depth)
    medial = 0.15 * float(min_depth)

    frame_count = 60
    phases = torch.linspace(0.0, 1.0, frame_count, device=device)
    writer = imageio.get_writer(
        output_dir / "color.mp4",
        format="FFMPEG",
        mode="I",
        fps=10,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "21"],
    )
    selected = {0: "start", 15: "quarter", 30: "middle", 45: "three_quarter", 59: "end"}
    rows = []
    started = time.perf_counter()
    try:
        for index, phase in enumerate(phases):
            eye = torch.stack(
                [
                    lateral * torch.sin(2.0 * torch.pi * phase),
                    torch.zeros((), device=device),
                    medial * (1.0 - torch.cos(2.0 * torch.pi * phase)) / 2.0,
                ]
            ).float()
            w2c = torch.linalg.inv(look_at_c2w(eye, look_at))
            render_started = time.perf_counter()
            rendering, alpha, _ = rasterization(
                gaussians.mean_vectors.float(),
                gaussians.quaternions.float(),
                gaussians.singular_values.float(),
                gaussians.opacities.float(),
                gaussians.colors.float(),
                w2c.view(1, 1, 4, 4),
                intrinsics.view(1, 1, 3, 3),
                width,
                height,
                sh_degree=None,
                render_mode="RGB",
                packed=True,
                eps2d=1e-8,
            )
            torch.cuda.synchronize(device)
            render_ms = (time.perf_counter() - render_started) * 1000.0
            rgb = linearRGB2sRGB(rendering).clamp(0.0, 1.0)[0, 0]
            alpha_frame = alpha[0, 0, :, :, 0].clamp(0.0, 1.0)
            rgb_u8 = rgb.mul(255).round().to(torch.uint8).cpu().numpy()
            alpha_np = alpha_frame.float().cpu().numpy()
            writer.append_data(rgb_u8)
            rows.append(
                {
                    "frame": index,
                    "phase": float(phase),
                    "eye_x": float(eye[0]),
                    "eye_y": float(eye[1]),
                    "eye_z": float(eye[2]),
                    "render_ms": render_ms,
                    "alpha_ge_095": float((alpha_np >= 0.95).mean()),
                    "alpha_hard_hole_lt_050": float((alpha_np < 0.50).mean()),
                    "top_10pct_hard_hole_lt_050": float(
                        (alpha_np[: max(1, height // 10)] < 0.50).mean()
                    ),
                }
            )
            if index in selected:
                label = selected[index]
                imageio.imwrite(output_dir / f"frame_{label}.png", rgb_u8)
                imageio.imwrite(
                    output_dir / f"alpha_{label}_u16.png",
                    np.round(alpha_np * 65535).astype(np.uint16),
                )
    finally:
        writer.close()

    with (output_dir / "per_frame_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    result = {
        "schema_version": "1.0-infinisplat-official-demo-lowmem",
        "status": "success",
        "scope": "Official PLY + official camera formula + official render geometry; one-frame chunks replace upstream eight-frame chunks only for peak-memory reduction.",
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "ply_path": config["ply_path"],
        "ply_sha256": actual_hash,
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "render_resolution_wh": [width, height],
        "render_intrinsics_px": config["render_intrinsics_px"],
        "look_at_xyz": [float(value) for value in look_at],
        "min_depth_q10": float(min_depth),
        "max_lateral_offset": lateral,
        "max_medial_offset": medial,
        "frames": frame_count,
        "fps": 10,
        "duration_seconds": time.perf_counter() - started,
        "sampled_metrics": {label: rows[index] for index, label in selected.items()},
    }
    with (output_dir / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
