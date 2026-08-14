#!/usr/bin/env python3
"""Render a frozen InfiniSplat PLY at a real dataset target camera pose."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TRUE_ARC_SCRIPT_DIR = SCRIPT_DIR.parents[1] / "02_true_arc统一协议" / "scripts"
if str(TRUE_ARC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRUE_ARC_SCRIPT_DIR))

from render_frozen_ply_true_arc import (  # noqa: E402
    REPO_ROOT,
    linearRGB2sRGB,
    load_infinisplat_ply,
    load_json,
    rasterization,
    resolve_repo_path,
    sha256,
)
from gsplat.utils import normalized_quat_to_rotmat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Append a portable suffix to the configured output directory.",
    )
    return parser.parse_args()


def scaled_intrinsics(
    matrix: list[list[float]],
    original_shape_wh: list[int],
    render_shape_wh: list[int],
) -> torch.Tensor:
    original_width, original_height = (float(value) for value in original_shape_wh)
    render_width, render_height = (float(value) for value in render_shape_wh)
    intrinsics = torch.tensor(matrix, dtype=torch.float32)
    intrinsics[0, :] *= render_width / original_width
    intrinsics[1, :] *= render_height / original_height
    intrinsics[2, :] = torch.tensor([0.0, 0.0, 1.0])
    return intrinsics


def save_resized_reference(source: Path, destination: Path, shape_wh: list[int]) -> None:
    width, height = (int(value) for value in shape_wh)
    with Image.open(source) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        image.save(destination, quality=95, subsampling=0)


def render(
    gaussians,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    colors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendering, alpha, _ = rasterization(
        gaussians.mean_vectors.float(),
        gaussians.quaternions.float(),
        gaussians.singular_values.float(),
        gaussians.opacities.float(),
        colors.float(),
        w2c.view(1, 1, 4, 4),
        intrinsics.view(1, 1, 3, 3),
        width,
        height,
        sh_degree=None,
        render_mode="RGB",
        packed=True,
        eps2d=1e-8,
    )
    return rendering[0, 0], alpha[0, 0, :, :, 0]


def gaussian_surface_normals(gaussians, target_w2c: torch.Tensor) -> torch.Tensor:
    quaternions = gaussians.quaternions[0]
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rotations = normalized_quat_to_rotmat(quaternions)
    smallest_axis = gaussians.singular_values[0].argmin(dim=-1)
    gather_index = smallest_axis[:, None, None].expand(-1, 3, 1)
    normals_source = rotations.gather(2, gather_index).squeeze(2)
    normals_target = normals_source @ target_w2c[:3, :3].T
    positions_target = (
        gaussians.mean_vectors[0] @ target_w2c[:3, :3].T + target_w2c[:3, 3]
    )
    flip = (normals_target * positions_target).sum(dim=-1, keepdim=True) > 0
    normals_target = torch.where(flip, -normals_target, normals_target)
    normals_target = normals_target / normals_target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normals_target * 0.5 + 0.5).unsqueeze(0)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    ply_path = resolve_repo_path(config["ply_path"])
    source_image_path = resolve_repo_path(config["source_image_path"])
    target_image_path = resolve_repo_path(config["target_image_path"])
    output_dir = resolve_repo_path(config["output_dir"])
    if args.run_suffix:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_suffix):
            raise ValueError("--run-suffix contains unsupported characters")
        output_dir = output_dir.with_name(f"{output_dir.name}_{args.run_suffix}")
    for path, expected_hash in (
        (ply_path, config["ply_sha256"]),
        (source_image_path, config["source_image_sha256"]),
        (target_image_path, config["target_image_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch: {path} expected={expected_hash} actual={actual_hash}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(config["device"])
    gaussians_cpu, _, _ = load_infinisplat_ply(ply_path)
    gaussians = gaussians_cpu.to(device)
    width, height = (int(value) for value in config["render_resolution_wh"])
    source_intrinsics = scaled_intrinsics(
        config["source_intrinsics_px"],
        config["source_original_shape_wh"],
        config["render_resolution_wh"],
    ).to(device)
    target_intrinsics = scaled_intrinsics(
        config["target_intrinsics_px"],
        config["target_original_shape_wh"],
        config["render_resolution_wh"],
    ).to(device)
    source_w2c = torch.eye(4, dtype=torch.float32, device=device)
    target_w2c = torch.tensor(
        config["source_camera_to_target_camera_w2c"], dtype=torch.float32, device=device
    )

    save_resized_reference(
        source_image_path, output_dir / "source_input_2016x1344.jpg", config["render_resolution_wh"]
    )
    save_resized_reference(
        target_image_path, output_dir / "target_ground_truth_2016x1344.jpg", config["render_resolution_wh"]
    )

    started = time.perf_counter()
    source_rgb_linear, source_alpha = render(
        gaussians,
        source_w2c,
        source_intrinsics,
        width,
        height,
        gaussians.colors,
    )
    target_rgb_linear, target_alpha = render(
        gaussians,
        target_w2c,
        target_intrinsics,
        width,
        height,
        gaussians.colors,
    )
    normal_colors = gaussian_surface_normals(gaussians, target_w2c)
    target_normal, _ = render(
        gaussians,
        target_w2c,
        target_intrinsics,
        width,
        height,
        normal_colors,
    )
    torch.cuda.synchronize(device)

    source_rgb = linearRGB2sRGB(source_rgb_linear).clamp(0.0, 1.0)
    target_rgb = linearRGB2sRGB(target_rgb_linear).clamp(0.0, 1.0)
    target_normal = target_normal.clamp(0.0, 1.0)
    source_alpha = source_alpha.clamp(0.0, 1.0)
    target_alpha = target_alpha.clamp(0.0, 1.0)

    def save_u8(name: str, tensor: torch.Tensor) -> None:
        array = tensor.mul(255).round().to(torch.uint8).cpu().numpy()
        imageio.imwrite(output_dir / name, array)

    save_u8("source_center_render.png", source_rgb)
    save_u8("target_real_pose_render_black.png", target_rgb)
    save_u8("target_real_pose_normal_derived.png", target_normal)
    target_white = target_rgb * target_alpha[..., None] + (1.0 - target_alpha[..., None])
    save_u8("target_real_pose_render_white.png", target_white)
    imageio.imwrite(
        output_dir / "source_center_alpha_u16.png",
        source_alpha.mul(65535).round().to(torch.uint16).cpu().numpy(),
    )
    imageio.imwrite(
        output_dir / "target_real_pose_alpha_u16.png",
        target_alpha.mul(65535).round().to(torch.uint16).cpu().numpy(),
    )

    source_hard_hole = float((source_alpha < 0.01).float().mean().item())
    target_hard_hole = float((target_alpha < 0.01).float().mean().item())
    target_soft_hole = float((target_alpha < 0.5).float().mean().item())
    result = {
        "schema_version": "1.0-infinisplat-real-target-pose-render",
        "status": "success",
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "scope": config["scope"],
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "render_resolution_wh": [width, height],
        "metric_baseline_m": config["metric_baseline_m"],
        "paper_baseline_bin": config["paper_baseline_bin"],
        "relative_rotation_deg": config["relative_rotation_deg"],
        "covisibility_proxy": config["covisibility_proxy"],
        "source_scaled_intrinsics_px": source_intrinsics.cpu().tolist(),
        "target_scaled_intrinsics_px": target_intrinsics.cpu().tolist(),
        "source_camera_to_target_camera_w2c": config["source_camera_to_target_camera_w2c"],
        "metrics": {
            "source_hard_hole_alpha_lt_0_01": source_hard_hole,
            "target_hard_hole_alpha_lt_0_01": target_hard_hole,
            "target_soft_hole_alpha_lt_0_5": target_soft_hole,
        },
        "normal_note": "Derived diagnostic: smallest Gaussian scale axis, transformed to target camera and alpha-composited; not an author-released normal renderer.",
        "duration_seconds": time.perf_counter() - started,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
