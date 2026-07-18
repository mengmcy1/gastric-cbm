"""独立的 SHARP 浅 3D 实验渲染入口；不修改官方源码。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))

# 当前项目将 CUDA Toolkit 安装在 Conda 的 sharp 环境内，而不是系统全局 PATH。
# 在导入 gsplat 前显式暴露该 Toolkit，避免触碰系统环境变量或其他 Conda 环境。
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory", choices=["swipe", "rotate_forward"], default="swipe")
    parser.add_argument("--max-disparity", type=float, choices=[0.0, 0.02, 0.04, 0.08], required=True)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--endpoint", choices=["all", "center", "left", "right"], default="all")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def project_points(points: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    extrinsics = extrinsics.to(points.device)
    intrinsics = intrinsics.to(points.device)
    points_camera = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    z = points_camera[:, 2].clamp_min(1e-8)
    x = intrinsics[0, 0] * points_camera[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * points_camera[:, 1] / z + intrinsics[1, 2]
    return torch.stack((x, y), dim=-1)


def save_diagnostics(output_dir: Path, label: str, color: torch.Tensor, depth: torch.Tensor, alpha: torch.Tensor) -> None:
    color_u8 = (color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    alpha_u16 = (alpha[0, 0].clamp(0, 1) * 65535).to(torch.uint16).cpu().numpy()
    depth_np = depth[0, 0].detach().float().cpu().numpy()
    write_png(output_dir / f"frame_{label}.png", color_u8)
    write_png(output_dir / f"alpha_{label}_u16.png", alpha_u16)
    np.save(output_dir / f"depth_{label}_float32.npy", depth_np)
    depth_color = vis.colorize_depth(depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
    write_png(output_dir / f"depth_{label}.png", depth_color)


def main() -> None:
    args = parse_args()
    if args.num_steps < 1:
        raise ValueError("--num-steps must be at least 1")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本实验入口要求可用 CUDA；请检查 CUDA_VISIBLE_DEVICES 与 sharp 环境。")
    if not args.ply.is_file():
        raise FileNotFoundError(args.ply)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    gaussians_cpu, metadata = load_ply(args.ply)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    intrinsics = torch.tensor(
        [[f_px, 0, (width - 1) / 2, 0], [0, f_px, (height - 1) / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    params = camera.TrajectoryParams(type=args.trajectory, max_disparity=args.max_disparity, num_steps=args.num_steps)
    camera_model = camera.create_camera_model(gaussians_cpu, intrinsics, resolution_px=metadata.resolution_px)
    if args.max_disparity == 0 or args.endpoint == "center":
        eye_positions = [torch.zeros(3, dtype=torch.float32)]
    else:
        trajectory = camera.create_eye_trajectory(gaussians_cpu, params, metadata.resolution_px, f_px)
        if args.endpoint == "left":
            eye_positions = [trajectory[0]]
        elif args.endpoint == "right":
            eye_positions = [trajectory[-1]]
        else:
            eye_positions = trajectory

    # 官方 CLI 每帧搬运高斯；实验工具只搬运一次，以避免把搬运成本混入纯渲染。
    gaussians = gaussians_cpu.to(device)
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    center_info = camera_model.compute(torch.zeros(3, dtype=torch.float32))
    center_uv = project_points(gaussians.mean_vectors[0], center_info.extrinsics, center_info.intrinsics)
    color_writer = iio.get_writer(output_dir / "color.mp4", fps=args.fps, codec="libx264", pixelformat="yuv420p")
    depth_writer = iio.get_writer(output_dir / "depth.mp4", fps=args.fps, codec="libx264", pixelformat="yuv420p")
    selected = {0: "left", len(eye_positions) - 1: "right", min(range(len(eye_positions)), key=lambda i: abs(i - (len(eye_positions) - 1) / 2)): "center"}
    per_frame = []
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
            color = (result.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            depth_color = vis.colorize_depth(result.depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
            color_writer.append_data(color)
            depth_writer.append_data(depth_color)
            current_uv = project_points(gaussians.mean_vectors[0], camera_info.extrinsics, camera_info.intrinsics)
            displacement = torch.linalg.vector_norm(current_uv - center_uv, dim=-1)
            visible = torch.isfinite(displacement) & (current_uv[:, 0] >= 0) & (current_uv[:, 0] < width) & (current_uv[:, 1] >= 0) & (current_uv[:, 1] < height)
            values = displacement[visible]
            per_frame.append({"frame": index, "render_ms": round(elapsed_ms, 3), "proxy_p95_px": float(torch.quantile(values, 0.95).cpu()), "proxy_p99_px": float(torch.quantile(values, 0.99).cpu()), "proxy_max_px": float(values.max().cpu())})
            if index in selected:
                save_diagnostics(output_dir, selected[index], result.color, result.depth, result.alpha)
    finally:
        color_writer.close()
        depth_writer.close()
    config = {"ply": str(args.ply.resolve()), "trajectory": args.trajectory, "max_disparity": args.max_disparity, "num_steps": len(eye_positions), "fps": args.fps, "codec": "libx264", "pix_fmt": "yuv420p", "ply_metadata_resolution": [int(width), int(height)], "render_resolution": [int(center_info.width), int(center_info.height)], "device": str(device), "gaussian_count": int(gaussians.mean_vectors.shape[1]), "total_seconds": round(time.perf_counter() - start, 3), "projection_displacement_definition": "Gaussian-center projection displacement relative to center camera; proxy only, not optical flow."}
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "projection_proxy.json").write_text(json.dumps(per_frame, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
