#!/usr/bin/env python3
"""Render a smooth camera path between a real source and target pose."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_frozen_ply_real_target_pose import (  # noqa: E402
    REPO_ROOT,
    linearRGB2sRGB,
    load_infinisplat_ply,
    load_json,
    render,
    resolve_repo_path,
    save_resized_reference,
    scaled_intrinsics,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Append an ASCII-safe suffix to the configured output directory.",
    )
    return parser.parse_args()


def so3_axis_angle(rotation: torch.Tensor) -> tuple[torch.Tensor, float]:
    cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = float(torch.acos(cosine).item())
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0], device=rotation.device), 0.0
    axis = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis / axis.norm().clamp_min(1e-8), angle


def rotation_from_axis_angle(axis: torch.Tensor, angle: float) -> torch.Tensor:
    x, y, z = axis
    skew = torch.stack(
        [
            torch.stack([x * 0.0, -z, y]),
            torch.stack([z, y * 0.0, -x]),
            torch.stack([-y, x, z * 0.0]),
        ]
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def c2w_to_w2c(rotation_c2w: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    w2c = torch.eye(4, dtype=rotation_c2w.dtype, device=rotation_c2w.device)
    w2c[:3, :3] = rotation_c2w.T
    w2c[:3, 3] = -(rotation_c2w.T @ center)
    return w2c


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    pair_config_path = resolve_repo_path(config["pair_config"])
    pair = load_json(pair_config_path)
    ply_path = resolve_repo_path(pair["ply_path"])
    source_image_path = resolve_repo_path(pair["source_image_path"])
    target_image_path = resolve_repo_path(pair["target_image_path"])
    output_dir = resolve_repo_path(config["output_dir"])
    if args.run_suffix:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_suffix):
            raise ValueError("--run-suffix must match [A-Za-z0-9._-]+")
        output_dir = output_dir.with_name(f"{output_dir.name}_{args.run_suffix}")
    for path, expected_hash in (
        (ply_path, pair["ply_sha256"]),
        (source_image_path, pair["source_image_sha256"]),
        (target_image_path, pair["target_image_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(config["device"])
    gaussians_cpu, _, _ = load_infinisplat_ply(ply_path)
    gaussians = gaussians_cpu.to(device)
    width, height = (int(value) for value in pair["render_resolution_wh"])
    source_intrinsics = scaled_intrinsics(
        pair["source_intrinsics_px"],
        pair["source_original_shape_wh"],
        pair["render_resolution_wh"],
    ).to(device)
    target_intrinsics = scaled_intrinsics(
        pair["target_intrinsics_px"],
        pair["target_original_shape_wh"],
        pair["render_resolution_wh"],
    ).to(device)

    target_w2c = torch.tensor(
        pair["source_camera_to_target_camera_w2c"], dtype=torch.float32, device=device
    )
    target_c2w = torch.linalg.inv(target_w2c)
    target_rotation_c2w = target_c2w[:3, :3]
    target_center = target_c2w[:3, 3]
    axis, total_rotation_rad = so3_axis_angle(target_rotation_c2w)
    total_baseline = float(target_center.norm().item())
    frame_count = int(config["trajectory"]["frame_count"])
    fps = int(config["trajectory"]["fps"])
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")

    save_resized_reference(
        source_image_path, output_dir / "source_input_2016x1344.jpg", pair["render_resolution_wh"]
    )
    save_resized_reference(
        target_image_path, output_dir / "target_ground_truth_2016x1344.jpg", pair["render_resolution_wh"]
    )
    black_writer = imageio.get_writer(
        output_dir / "real_pose_interpolation_black.mp4",
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18"],
    )
    white_writer = imageio.get_writer(
        output_dir / "real_pose_interpolation_white.mp4",
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18"],
    )
    selected = {
        0: "start",
        frame_count // 4: "quarter",
        frame_count // 2: "middle",
        (3 * frame_count) // 4: "three_quarter",
        frame_count - 1: "end",
    }
    rows: list[dict[str, float | int]] = []
    endpoint_w2c = None
    started = time.perf_counter()
    try:
        for frame_index in range(frame_count):
            progress = frame_index / (frame_count - 1)
            rotation_c2w = rotation_from_axis_angle(axis, progress * total_rotation_rad)
            center = progress * target_center
            w2c = c2w_to_w2c(rotation_c2w, center)
            intrinsics = (1.0 - progress) * source_intrinsics + progress * target_intrinsics
            if frame_index == frame_count - 1:
                endpoint_w2c = w2c.detach().cpu()

            render_started = time.perf_counter()
            rgb_linear, alpha = render(
                gaussians,
                w2c,
                intrinsics,
                width,
                height,
                gaussians.colors,
            )
            torch.cuda.synchronize(device)
            render_ms = (time.perf_counter() - render_started) * 1000.0
            rgb = linearRGB2sRGB(rgb_linear).clamp(0.0, 1.0)
            alpha = alpha.clamp(0.0, 1.0)
            white = rgb * alpha[..., None] + (1.0 - alpha[..., None])
            rgb_u8 = rgb.mul(255).round().to(torch.uint8).cpu().numpy()
            white_u8 = white.mul(255).round().to(torch.uint8).cpu().numpy()
            alpha_np = alpha.float().cpu().numpy()
            black_writer.append_data(rgb_u8)
            white_writer.append_data(white_u8)

            row = {
                "frame": frame_index,
                "progress": progress,
                "baseline_m": progress * total_baseline,
                "relative_rotation_deg": math.degrees(progress * total_rotation_rad),
                "hard_hole_alpha_lt_0_01": float((alpha_np < 0.01).mean()),
                "soft_hole_alpha_lt_0_5": float((alpha_np < 0.5).mean()),
                "mean_alpha": float(alpha_np.mean()),
                "render_ms": render_ms,
            }
            rows.append(row)
            if frame_index in selected:
                label = selected[frame_index]
                imageio.imwrite(output_dir / f"frame_{label}_black.png", rgb_u8)
                imageio.imwrite(output_dir / f"frame_{label}_white.png", white_u8)
                imageio.imwrite(
                    output_dir / f"alpha_{label}_u16.png",
                    np.rint(alpha_np * 65535.0).astype(np.uint16),
                )
    finally:
        black_writer.close()
        white_writer.close()

    assert endpoint_w2c is not None
    endpoint_error = float(
        (endpoint_w2c - target_w2c.detach().cpu()).abs().max().item()
    )
    with (output_dir / "per_frame_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "1.0-infinisplat-real-pose-interpolation-result",
        "status": "success",
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "pair_config_path": config["pair_config"],
        "scope": config["scope"],
        "trajectory": config["trajectory"],
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "render_resolution_wh": [width, height],
        "total_metric_baseline_m": total_baseline,
        "total_relative_rotation_deg": math.degrees(total_rotation_rad),
        "endpoint_w2c_max_abs_error": endpoint_error,
        "duration_seconds": time.perf_counter() - started,
        "sampled_metrics": {label: rows[index] for index, label in selected.items()},
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
