#!/usr/bin/env python3
"""Render an exported InfiniSplat PLY with the project's frozen true_arc camera model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from plyfile import PlyData


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("无法定位 2Dto3D 仓库根目录。")


REPO_ROOT = find_repo_root(Path(__file__))
INFINISPLAT_SOURCE = REPO_ROOT / "源码" / "InfiniSplat"
if str(INFINISPLAT_SOURCE) not in sys.path:
    sys.path.insert(0, str(INFINISPLAT_SOURCE))

from gsplat import rasterization  # noqa: E402
from src.utils.color_space import linearRGB2sRGB, sRGB2linearRGB  # noqa: E402
from src.utils.gaussians import Gaussians3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将冻结 InfiniSplat PLY 按项目 true_arc 总视角协议渲染。"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_repo_path(path_text: str) -> Path:
    path = (REPO_ROOT / path_text).resolve()
    path.relative_to(REPO_ROOT)
    return path


def ply_element_array(ply: PlyData, element_name: str, property_name: str) -> np.ndarray:
    element = next((item for item in ply.elements if item.name == element_name), None)
    if element is None or property_name not in element:
        raise KeyError(f"PLY 缺少 {element_name}.{property_name}")
    return np.asarray(element[property_name])


def load_infinisplat_ply(path: Path) -> tuple[Gaussians3D, np.ndarray, tuple[int, int]]:
    """Load the SuperSplat-compatible PLY emitted by official InfiniSplat."""
    ply = PlyData.read(path)
    vertex = next((item for item in ply.elements if item.name == "vertex"), None)
    if vertex is None:
        raise KeyError("PLY 缺少 vertex 元素。")

    means = np.stack([np.asarray(vertex[key]) for key in ("x", "y", "z")], axis=-1)
    sh0 = np.stack([np.asarray(vertex[f"f_dc_{index}"]) for index in range(3)], axis=-1)
    scales = np.stack([np.asarray(vertex[f"scale_{index}"]) for index in range(3)], axis=-1)
    quaternions = np.stack(
        [np.asarray(vertex[f"rot_{index}"]) for index in range(4)], axis=-1
    )
    opacity = np.asarray(vertex["opacity"])

    sh_coefficient = math.sqrt(1.0 / (4.0 * math.pi))
    colors_srgb = torch.from_numpy(sh0.copy()).float() * sh_coefficient + 0.5
    colors_linear = sRGB2linearRGB(colors_srgb.clamp(0.0, 1.0))
    gaussians = Gaussians3D(
        mean_vectors=torch.from_numpy(means.copy()).float().unsqueeze(0),
        singular_values=torch.from_numpy(scales.copy()).float().exp().unsqueeze(0),
        quaternions=torch.from_numpy(quaternions.copy()).float().unsqueeze(0),
        colors=colors_linear.unsqueeze(0),
        opacities=torch.from_numpy(opacity.copy()).float().sigmoid().unsqueeze(0),
    )

    intrinsics = ply_element_array(ply, "intrinsic", "intrinsic").reshape(3, 3).astype(np.float32)
    image_size = ply_element_array(ply, "image_size", "image_size")
    width, height = int(image_size[0]), int(image_size[1])
    return gaussians, intrinsics, (width, height)


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def look_at_c2w(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Create an OpenCV camera-to-world matrix; center eye/target yields identity."""
    forward = normalize(target - eye)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=eye.device)
    right = normalize(torch.cross(forward, world_up, dim=-1))
    down = normalize(torch.cross(forward, right, dim=-1))
    matrix = torch.eye(4, dtype=torch.float32, device=eye.device)
    matrix[:3, 0] = right
    matrix[:3, 1] = down
    matrix[:3, 2] = forward
    matrix[:3, 3] = eye
    return matrix


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("trajectory_mode") != "true_arc":
        raise ValueError("该入口只允许 trajectory_mode=true_arc。")

    ply_path = resolve_repo_path(config["ply_path"])
    output_dir = resolve_repo_path(config["output_dir"])
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_ply_sha256 = sha256(ply_path)
    if actual_ply_sha256 != config["ply_sha256"]:
        raise RuntimeError(
            f"PLY SHA256 不匹配：expected={config['ply_sha256']} actual={actual_ply_sha256}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("true_arc 渲染需要 CUDA。")
    device = torch.device(config["device"])
    gaussians_cpu, intrinsics_np, (ply_width, ply_height) = load_infinisplat_ply(ply_path)
    width, height = (int(value) for value in config["render_resolution_wh"])
    if (width, height) != (ply_width, ply_height):
        raise ValueError(
            "诊断 v1 要求按 PLY 推理分辨率原尺寸渲染："
            f"config={(width, height)} ply={(ply_width, ply_height)}"
        )

    gaussians = gaussians_cpu.to(device)
    valid_depth = gaussians.mean_vectors[0, :, 2]
    valid_depth = valid_depth[torch.isfinite(valid_depth) & (valid_depth > 1e-4)]
    focus_quantile = float(config["focus_depth_quantile"])
    focus_depth = max(
        float(config["minimum_focus_depth"]),
        float(torch.quantile(valid_depth.float(), focus_quantile).item()),
    )

    intrinsics_px = torch.from_numpy(intrinsics_np).float().to(device)
    total_angle = float(config["angle_total_deg"])
    num_steps = int(config["num_steps"])
    angles = np.linspace(-total_angle / 2.0, total_angle / 2.0, num_steps)
    target = torch.tensor([0.0, 0.0, focus_depth], dtype=torch.float32, device=device)

    writer = imageio.get_writer(
        output_dir / "color.mp4",
        format="FFMPEG",
        mode="I",
        fps=int(config["fps"]),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "21"],
    )
    rows: list[dict[str, Any]] = []
    selected = {0: "left", num_steps // 2: "center", num_steps - 1: "right"}
    center_w2c = None
    started = time.perf_counter()
    try:
        for frame_index, angle_deg in enumerate(angles):
            angle_rad = math.radians(float(angle_deg))
            eye = torch.tensor(
                [
                    focus_depth * math.sin(angle_rad),
                    0.0,
                    focus_depth * (1.0 - math.cos(angle_rad)),
                ],
                dtype=torch.float32,
                device=device,
            )
            w2c = torch.linalg.inv(look_at_c2w(eye, target))
            if frame_index == num_steps // 2:
                center_w2c = w2c.detach().cpu()

            render_started = time.perf_counter()
            rendering, alpha, _ = rasterization(
                gaussians.mean_vectors.float(),
                gaussians.quaternions.float(),
                gaussians.singular_values.float(),
                gaussians.opacities.float(),
                gaussians.colors.float(),
                w2c.view(1, 1, 4, 4),
                intrinsics_px.view(1, 1, 3, 3),
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
                    "frame": frame_index,
                    "angle_deg": float(angle_deg),
                    "eye_x": float(eye[0]),
                    "eye_y": float(eye[1]),
                    "eye_z": float(eye[2]),
                    "render_ms": render_ms,
                    "alpha_ge_099": float((alpha_np >= 0.99).mean()),
                    "alpha_ge_095": float((alpha_np >= 0.95).mean()),
                    "alpha_hard_hole_lt_050": float((alpha_np < 0.50).mean()),
                    "top_10pct_hard_hole_lt_050": float(
                        (alpha_np[: max(1, height // 10)] < 0.50).mean()
                    ),
                }
            )
            if frame_index in selected:
                label = selected[frame_index]
                imageio.imwrite(output_dir / f"frame_{label}.png", rgb_u8)
                imageio.imwrite(
                    output_dir / f"alpha_{label}_u16.png",
                    np.round(alpha_np * 65535).astype(np.uint16),
                )
    finally:
        writer.close()

    if center_w2c is None:
        raise RuntimeError("中心帧未生成。")
    with (output_dir / "per_frame_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    result = {
        "schema_version": "1.0-infinisplat-true-arc-diagnostic",
        "status": "success",
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "ply_path": config["ply_path"],
        "ply_sha256": actual_ply_sha256,
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "trajectory_mode": "true_arc",
        "angle_total_deg": total_angle,
        "angle_half_deg": total_angle / 2.0,
        "num_steps": num_steps,
        "focus_depth_quantile": focus_quantile,
        "focus_depth": focus_depth,
        "arc_radius": focus_depth,
        "look_at_xyz": [0.0, 0.0, focus_depth],
        "render_resolution_wh": [width, height],
        "center_w2c_max_abs_diff_from_identity": float(
            (center_w2c - torch.eye(4)).abs().max()
        ),
        "fps": int(config["fps"]),
        "duration_seconds": time.perf_counter() - started,
        "endpoints": {"left": rows[0], "center": rows[num_steps // 2], "right": rows[-1]},
    }
    with (output_dir / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
