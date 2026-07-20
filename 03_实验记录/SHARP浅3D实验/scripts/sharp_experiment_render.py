"""独立的 SHARP 浅 3D 实验渲染入口；不修改官方源码。

v2.1 变更（2026-07-20）：
- 修复可见性判断使用 PLY 原始分辨率而非实际渲染分辨率的 bug。
- 有效高斯约束：中心帧与当前帧投影均在渲染画布内、深度>0、位移有限。
- 新增位移占画宽百分比（P95/P99/max）。
- max_disparity=0 时遵守 --num-steps，用于运动矩阵零档生成 N 张相同视角帧。
- 新增逐帧自动指标 CSV 与人工标记 CSV 模板。
- 新增相邻帧 SSIM（编码前 RGB，降采样加速）。
- 新增逐帧 Alpha 覆盖率。
- 输出目录默认防覆盖；需 --allow-overwrite 才可写入非空目录。
- 修复投影代理统计未实际应用正深度条件的问题。
- 视频写入按 2 像素宏块对齐，避免 810×1080 被静默补边，并记录编码后分辨率。
"""

from __future__ import annotations

import argparse
import csv
import json
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

# 相邻帧 SSIM 降采样最大边长（加速，不影响异常检测目的）
_SSIM_MAX_SIDE = 512


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
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="允许写入非空输出目录（默认拒绝覆盖已有结果）")
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
    """降采样 RGB uint8 图像用于快速 SSIM 计算。"""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / max(h, w)
    new_size = (round(w * scale), round(h * scale))
    return np.asarray(Image.fromarray(rgb).resize(new_size, Image.Resampling.BICUBIC))


def _compute_adjacent_ssim(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    """计算相邻帧 SSIM（降采样后，在编码前 RGB 上计算）。"""
    prev_small = _downsample_rgb(prev_rgb, _SSIM_MAX_SIDE)
    curr_small = _downsample_rgb(curr_rgb, _SSIM_MAX_SIDE)
    return float(ssim(prev_small, curr_small, channel_axis=2, data_range=255))


def _compute_proxy_stats(
    gaussian_means: torch.Tensor,
    camera_info,
    center_uv: torch.Tensor,
    center_depth: torch.Tensor,
    render_w: int,
    render_h: int,
) -> dict:
    """计算单帧的高斯投影位移统计（使用实际渲染分辨率）。

    有效高斯条件：中心投影在中心画布内、当前投影在当前画布内、
    深度>0、位移有限。
    """
    current_uv, current_depth = project_points(
        gaussian_means, camera_info.extrinsics, camera_info.intrinsics
    )
    displacement = torch.linalg.vector_norm(current_uv - center_uv, dim=-1)

    # 中心帧可见性（使用实际渲染分辨率）
    center_in_bounds = (
        torch.isfinite(center_uv).all(dim=-1)
        & (center_uv[:, 0] >= 0) & (center_uv[:, 0] < render_w)
        & (center_uv[:, 1] >= 0) & (center_uv[:, 1] < render_h)
    )
    # 当前帧可见性（使用实际渲染分辨率）
    current_in_bounds = (
        torch.isfinite(current_uv).all(dim=-1)
        & (current_uv[:, 0] >= 0) & (current_uv[:, 0] < render_w)
        & (current_uv[:, 1] >= 0) & (current_uv[:, 1] < render_h)
    )
    # 位移有效
    disp_finite = torch.isfinite(displacement)
    positive_depth = (
        torch.isfinite(center_depth)
        & torch.isfinite(current_depth)
        & (center_depth > 0)
        & (current_depth > 0)
    )

    valid = center_in_bounds & current_in_bounds & positive_depth & disp_finite
    pre_depth_valid = center_in_bounds & current_in_bounds & disp_finite
    rejected_nonpositive_depth_count = int((pre_depth_valid & ~positive_depth).sum().cpu())
    values = displacement[valid]
    valid_count = int(valid.sum().cpu())

    if valid_count == 0:
        return {
            "proxy_p95_px": 0.0,
            "proxy_p95_width_percent": 0.0,
            "proxy_p99_px": 0.0,
            "proxy_p99_width_percent": 0.0,
            "proxy_max_px": 0.0,
            "proxy_max_width_percent": 0.0,
            "valid_gaussian_count": 0,
            "rejected_nonpositive_depth_count": rejected_nonpositive_depth_count,
        }

    p95 = float(torch.quantile(values, 0.95).cpu())
    p99 = float(torch.quantile(values, 0.99).cpu())
    pmax = float(values.max().cpu())

    return {
        "proxy_p95_px": p95,
        "proxy_p95_width_percent": round(p95 / render_w * 100, 4),
        "proxy_p99_px": p99,
        "proxy_p99_width_percent": round(p99 / render_w * 100, 4),
        "proxy_max_px": pmax,
        "proxy_max_width_percent": round(pmax / render_w * 100, 4),
        "valid_gaussian_count": valid_count,
        "rejected_nonpositive_depth_count": rejected_nonpositive_depth_count,
    }


def _alpha_coverage(alpha: torch.Tensor) -> dict:
    """计算逐帧 Alpha 覆盖率。"""
    a = alpha[0, 0].float().cpu().numpy()
    return {
        "alpha_ge_099": round(float((a >= 0.99).mean()), 6),
        "alpha_ge_095": round(float((a >= 0.95).mean()), 6),
    }


def _probe_video(path: Path) -> dict:
    """读取编码后视频元数据，记录实际尺寸和帧数。"""
    reader = iio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        size = meta.get("size") or meta.get("source_size")
        frame_count = reader.count_frames()
    finally:
        reader.close()
    return {
        "resolution": [int(size[0]), int(size[1])] if size else None,
        "frame_count": int(frame_count),
    }


# ---- per_frame_manual.csv 列定义 ----
_MANUAL_COLUMNS = [
    "frame", "hole", "stretching", "flicker", "thin_structure_break",
    "reflection_error", "severity", "notes",
]

# ---- per_frame_metrics.csv 列定义 ----
_METRICS_COLUMNS = [
    "frame", "eye_x", "eye_y", "eye_z",
    "render_ms",
    "proxy_p95_px", "proxy_p95_width_percent",
    "proxy_p99_px", "proxy_p99_width_percent",
    "proxy_max_px", "proxy_max_width_percent",
    "valid_gaussian_count",
    "rejected_nonpositive_depth_count",
    "alpha_ge_099", "alpha_ge_095",
    "adjacent_ssim",
]


def main() -> None:
    args = parse_args()
    if args.num_steps < 1:
        raise ValueError("--num-steps must be at least 1")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本实验入口要求可用 CUDA；请检查 CUDA_VISIBLE_DEVICES 与 sharp 环境。")
    if not args.ply.is_file():
        raise FileNotFoundError(args.ply)

    output_dir = args.output_dir.resolve()

    # ---- 防覆盖检查 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            f"输出目录非空且未指定 --allow-overwrite：{output_dir}\n"
            f"已有 {len(existing)} 个文件/子目录。"
            f"请使用新的输出目录或添加 --allow-overwrite 确认覆盖。"
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

    # ---- 确定相机轨迹 ----
    if args.max_disparity == 0:
        # 零运动：--num-steps 张相同中心视角帧（运动矩阵零档可用 60 帧）
        eye_positions = [torch.zeros(3, dtype=torch.float32)] * args.num_steps
    elif args.endpoint == "center":
        eye_positions = [torch.zeros(3, dtype=torch.float32)]
    elif args.endpoint in ("left", "right"):
        trajectory = camera.create_eye_trajectory(gaussians_cpu,
                                                  camera.TrajectoryParams(type=args.trajectory,
                                                                          max_disparity=args.max_disparity,
                                                                          num_steps=args.num_steps),
                                                  metadata.resolution_px, f_px)
        eye_positions = [trajectory[0]] if args.endpoint == "left" else [trajectory[-1]]
    else:
        trajectory = camera.create_eye_trajectory(gaussians_cpu,
                                                  camera.TrajectoryParams(type=args.trajectory,
                                                                          max_disparity=args.max_disparity,
                                                                          num_steps=args.num_steps),
                                                  metadata.resolution_px, f_px)
        eye_positions = trajectory

    # 官方 CLI 每帧搬运高斯；实验工具只搬运一次，以避免把搬运成本混入纯渲染。
    gaussians = gaussians_cpu.to(device)
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    center_info = camera_model.compute(torch.zeros(3, dtype=torch.float32))
    render_w = center_info.width
    render_h = center_info.height

    # 中心帧高斯投影（使用实际渲染内参）
    center_uv, center_depth = project_points(
        gaussians.mean_vectors[0], center_info.extrinsics, center_info.intrinsics
    )

    if render_w % 2 or render_h % 2:
        raise ValueError(
            f"yuv420p 要求宽高为偶数，当前渲染分辨率为 {render_w}×{render_h}"
        )
    video_writer_kwargs = {
        "fps": args.fps,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 2,
    }
    color_writer = iio.get_writer(output_dir / "color.mp4", **video_writer_kwargs)
    depth_writer = iio.get_writer(output_dir / "depth.mp4", **video_writer_kwargs)

    # 三视图诊断帧选择
    n = len(eye_positions)
    selected = {
        0: "left",
        n - 1: "right",
        min(range(n), key=lambda i: abs(i - (n - 1) / 2)): "center",
    }

    per_frame_json = []
    per_frame_metrics = []
    prev_rgb = None  # 用于相邻帧 SSIM

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

            # 编码前 RGB uint8（用于相邻帧 SSIM，避免视频编码误差）
            color_u8 = (result.color[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            depth_color = vis.colorize_depth(result.depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
            color_writer.append_data(color_u8)
            depth_writer.append_data(depth_color)

            # ---- 投影位移统计 ----
            proxy = _compute_proxy_stats(
                gaussians.mean_vectors[0], camera_info, center_uv, center_depth, render_w, render_h
            )

            # ---- Alpha 覆盖率 ----
            alpha_cov = _alpha_coverage(result.alpha)

            # ---- 相邻帧 SSIM ----
            if prev_rgb is not None:
                adj_ssim = _compute_adjacent_ssim(prev_rgb, color_u8)
            else:
                adj_ssim = None
            prev_rgb = color_u8

            # 写入 per_frame_json（保留完整机器记录）
            per_frame_json.append({
                "frame": index,
                "eye_x": round(float(eye[0]), 6),
                "eye_y": round(float(eye[1]), 6),
                "eye_z": round(float(eye[2]), 6),
                "render_ms": round(elapsed_ms, 3),
                **proxy,
                **alpha_cov,
                "adjacent_ssim": round(adj_ssim, 6) if adj_ssim is not None else None,
            })

            # 写入 per_frame_metrics_csv 行
            per_frame_metrics.append([
                index,
                round(float(eye[0]), 6),
                round(float(eye[1]), 6),
                round(float(eye[2]), 6),
                round(elapsed_ms, 3),
                proxy["proxy_p95_px"],
                proxy["proxy_p95_width_percent"],
                proxy["proxy_p99_px"],
                proxy["proxy_p99_width_percent"],
                proxy["proxy_max_px"],
                proxy["proxy_max_width_percent"],
                proxy["valid_gaussian_count"],
                proxy["rejected_nonpositive_depth_count"],
                alpha_cov["alpha_ge_099"],
                alpha_cov["alpha_ge_095"],
                round(adj_ssim, 6) if adj_ssim is not None else "",
            ])

            # 三视图诊断帧
            if index in selected:
                save_diagnostics(output_dir, selected[index], result.color, result.depth, result.alpha)

    finally:
        color_writer.close()
        depth_writer.close()

    encoded_color = _probe_video(output_dir / "color.mp4")
    encoded_depth = _probe_video(output_dir / "depth.mp4")

    # ---- 写入逐帧指标 CSV ----
    metrics_csv_path = output_dir / "per_frame_metrics.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_METRICS_COLUMNS)
        writer.writerows(per_frame_metrics)

    # ---- 写入人工标记 CSV 模板 ----
    manual_csv_path = output_dir / "per_frame_manual.csv"
    with open(manual_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_MANUAL_COLUMNS)
        for i in range(len(eye_positions)):
            writer.writerow([i, "", "", "", "", "", "", ""])

    # ---- 视频级人工评分模板 ----
    video_meta = {
        "schema_version": "2.0",
        "review_status": "",
        "overall_quality_score": "",
        "hole_severity": "",
        "stretching_severity": "",
        "flicker_severity": "",
        "paper_feel_severity": "",
        "occlusion_error_severity": "",
        "reflection_deformation_severity": "",
        "first_artifact_frame_left": "",
        "first_artifact_frame_right": "",
        "worst_frame": "",
        "overall_pass": "",
        "notes": "",
    }
    (output_dir / "video_manual.json").write_text(
        json.dumps(video_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- config.json ----
    config = {
        "script_version": "v2.1",
        "ply": str(args.ply.resolve()),
        "trajectory": args.trajectory,
        "max_disparity": args.max_disparity,
        "num_steps": len(eye_positions),
        "fps": args.fps,
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "ply_metadata_resolution": [ply_w, ply_h],
        "render_resolution": [int(render_w), int(render_h)],
        "encoded_color_video": encoded_color,
        "encoded_depth_video": encoded_depth,
        "device": str(device),
        "gaussian_count": int(gaussians.mean_vectors.shape[1]),
        "total_seconds": round(time.perf_counter() - start, 3),
        "projection_displacement_definition": (
            "Gaussian-center projection displacement relative to center camera, "
            "computed using actual render intrinsics. "
            "Valid gaussians: center UV in render bounds AND current UV in render bounds "
            "AND depth>0 AND finite displacement. "
            "Width ratio = px / render_width * 100. "
            "Proxy only, not optical flow."
        ),
        "adjacent_ssim_definition": (
            "SSIM between consecutive frames computed on pre-encoding RGB uint8, "
            f"downsampled to max {_SSIM_MAX_SIDE}px side for speed. "
            "Purpose: anomaly detection (sudden drops), not absolute quality."
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- projection_proxy.json（保留兼容） ----
    (output_dir / "projection_proxy.json").write_text(
        json.dumps(per_frame_json, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
