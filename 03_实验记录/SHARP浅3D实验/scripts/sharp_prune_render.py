"""SHARP 高斯保留率实验（含3%裁剪）：按不透明度剪枝 + 裁剪 + 静止指标。

变量控制：max_disparity=0.04，crop=3%（与阶段3冻结值一致），只改变 keep_percent。

静止指标使用严格零位移中心帧（frame_center_zero.png），不使用轨迹第29帧。
视频中每帧在编码前统一完成四边裁剪并放大回原始显示尺寸。
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
import lpips as lpips_lib
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARP_SRC = REPO_ROOT / "源码" / "SHARP_APPLE注释" / "src"
if str(SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(SHARP_SRC))

CONDA_PREFIX = Path(sys.prefix)
CUDA_BIN = CONDA_PREFIX / "Library" / "bin"
if (CUDA_BIN / "nvcc.exe").is_file():
    os.environ.setdefault("CUDA_HOME", str(CONDA_PREFIX / "Library"))
    os.environ["PATH"] = str(CUDA_BIN) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(CUDA_BIN))

from sharp.utils import camera, gsplat, vis          # noqa: E402
from sharp.utils.gaussians import Gaussians3D, load_ply  # noqa: E402

_SSIM_MAX_SIDE = 512
_LPIPS_MAX_SIDE = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--input-image", type=Path, required=True,
                   help="原始输入图（用于静止质量指标对比）")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--keep-percent", type=int, required=True, choices=[100, 75, 50, 25],
                   help="高斯保留率（按不透明度从高到低保留）")
    p.add_argument("--crop-single-side-percent", type=int, default=3, choices=[0, 3, 5, 10],
                   help="四边单边裁剪百分比，默认3，与阶段3冻结值对齐")
    p.add_argument("--max-disparity", type=float, choices=[0.0, 0.02, 0.04, 0.08],
                   default=0.04)
    p.add_argument("--num-steps", type=int, default=60)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--manual-schema",
        choices=["prune", "validation"],
        default="prune",
        help="人工评分模板类型；独立验证使用 validation",
    )
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def _apply_crop(frame: np.ndarray, crop_percent: int) -> np.ndarray:
    """四边各裁剪 crop_percent%，然后以LANCZOS放大回原始显示尺寸。"""
    if crop_percent == 0:
        return frame
    h, w = frame.shape[:2]
    p = crop_percent / 100.0
    left, top = round(w * p), round(h * p)
    right, bottom = round(w * (1 - p)), round(h * (1 - p))
    cropped = frame[top:bottom, left:right]
    return np.asarray(Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS))


def prune_gaussians(g: Gaussians3D, keep_percent: int) -> tuple[Gaussians3D, dict]:
    """按不透明度从高到低排序，保留前 keep_percent%。"""
    op = g.opacities[0]
    n_total = int(op.shape[0])
    n_keep = int(round(n_total * keep_percent / 100))
    if keep_percent == 100:
        idx = torch.arange(n_total)
    else:
        idx = torch.argsort(op, descending=True)[:n_keep]
    g_pruned = Gaussians3D(
        mean_vectors=g.mean_vectors[:, idx, :],
        opacities=g.opacities[:, idx],
        colors=g.colors[:, idx, :],
        quaternions=g.quaternions[:, idx, :],
        singular_values=g.singular_values[:, idx, :],
    )
    stats = {
        "pruning_method": "opacity_descending",
        "n_original": n_total,
        "n_kept": n_keep,
        "keep_percent": keep_percent,
        "opacity_mean_original": round(float(op.mean().cpu()), 6),
        "opacity_min_kept": round(float(op[idx].min().cpu()), 6),
        "opacity_mean_kept": round(float(op[idx].mean().cpu()), 6),
    }
    return g_pruned, stats


def project_points(pts, extrinsics, intrinsics):
    extrinsics = extrinsics.to(pts.device)
    intrinsics = intrinsics.to(pts.device)
    pts_cam = pts @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    depth = pts_cam[:, 2]
    safe = depth.clamp_min(1e-8)
    x = intrinsics[0, 0] * pts_cam[:, 0] / safe + intrinsics[0, 2]
    y = intrinsics[1, 1] * pts_cam[:, 1] / safe + intrinsics[1, 2]
    return torch.stack((x, y), dim=-1), depth


def _resize_rgb(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img
    size = (round(w * scale), round(h * scale))
    return np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BICUBIC))


def compute_static_metrics(rendered: np.ndarray, reference: np.ndarray,
                            lpips_model, lpips_device) -> dict:
    """计算渲染帧相对原始输入图的PSNR/SSIM/LPIPS。rendered与reference均为RGB uint8。"""
    if rendered.shape != reference.shape:
        ref_pil = Image.fromarray(reference).resize(
            (rendered.shape[1], rendered.shape[0]), Image.Resampling.LANCZOS)
        reference = np.asarray(ref_pil)
    psnr_val = float(peak_signal_noise_ratio(reference, rendered, data_range=255))
    ssim_val = float(structural_similarity(reference, rendered, channel_axis=2, data_range=255))
    r_lp = _resize_rgb(rendered.copy(), _LPIPS_MAX_SIDE)
    ref_lp = _resize_rgb(reference.copy(), _LPIPS_MAX_SIDE)

    def to_t(a):
        return torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device) / 127.5 - 1

    with torch.inference_mode():
        lpips_val = float(lpips_model(to_t(r_lp), to_t(ref_lp)).item())
    return {
        "psnr_db": None if not np.isfinite(psnr_val) else round(psnr_val, 6),
        "psnr_is_infinite": bool(np.isinf(psnr_val)),
        "ssim": round(ssim_val, 6),
        "lpips_alex": round(lpips_val, 6),
        "lpips_resolution": [int(r_lp.shape[1]), int(r_lp.shape[0])],
        "comparison": "cropped zero-eye render vs original input image",
    }


def _probe_video(path: Path) -> dict:
    reader = iio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        size = meta.get("size") or meta.get("source_size")
        fc = reader.count_frames()
    finally:
        reader.close()
    return {"resolution": [int(size[0]), int(size[1])] if size else None, "frame_count": int(fc)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境。")
    for f in (args.ply, args.input_image):
        if not f.is_file():
            raise FileNotFoundError(f)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}\n已有{len(existing)}个文件；"
                              "请用新目录或添加 --allow-overwrite。")

    device = torch.device("cuda")
    crop = args.crop_single_side_percent
    print(f"[{args.keep_percent}%/crop{crop}%] 加载 PLY：{args.ply.name}")
    t0 = time.perf_counter()

    gaussians_cpu, metadata = load_ply(args.ply)
    print(f"  原始高斯数：{gaussians_cpu.mean_vectors.shape[1]:,}")

    gaussians_pruned_cpu, prune_stats = prune_gaussians(gaussians_cpu, args.keep_percent)
    print(f"  保留 {prune_stats['n_kept']:,} 个（min_opacity_kept={prune_stats['opacity_min_kept']:.4f}）")

    ply_w = int(metadata.resolution_px[0])
    ply_h = int(metadata.resolution_px[1])
    f_px = float(metadata.focal_length_px)
    intrinsics = torch.tensor(
        [[f_px, 0, (ply_w - 1) / 2, 0],
         [0, f_px, (ply_h - 1) / 2, 0],
         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)

    camera_model = camera.create_camera_model(
        gaussians_pruned_cpu, intrinsics, resolution_px=metadata.resolution_px)
    trajectory = camera.create_eye_trajectory(
        gaussians_pruned_cpu,
        camera.TrajectoryParams(type="swipe", max_disparity=args.max_disparity,
                                num_steps=args.num_steps),
        metadata.resolution_px, f_px) if args.max_disparity > 0 else \
        [torch.zeros(3, dtype=torch.float32)] * args.num_steps

    gaussians = gaussians_pruned_cpu.to(device)
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
    center_info = camera_model.compute(torch.zeros(3, dtype=torch.float32))
    render_w, render_h = center_info.width, center_info.height
    if render_w % 2 or render_h % 2:
        raise ValueError(f"yuv420p 要求宽高为偶数：{render_w}×{render_h}")

    lpips_model = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()

    # ── 严格零位移中心帧（用于静止质量指标，不进入视频）──────────────────
    print("  渲染零位移中心帧...")
    with torch.inference_mode():
        result_zero = renderer(gaussians,
                               extrinsics=center_info.extrinsics[None].to(device),
                               intrinsics=center_info.intrinsics[None].to(device),
                               image_width=render_w, image_height=render_h)
    color_zero_raw = (result_zero.color[0].permute(1, 2, 0).clamp(0, 1) * 255
                      ).to(torch.uint8).cpu().numpy()
    center_rgb = _apply_crop(color_zero_raw, crop)   # 裁剪后用于指标
    iio.imwrite(output_dir / "frame_center_zero.png", center_rgb)

    # ── 60帧主渲染循环（每帧先裁剪再编码）──────────────────────────────────
    video_kwargs = {"fps": args.fps, "codec": "libx264",
                    "pixelformat": "yuv420p", "macro_block_size": 2}
    color_writer = iio.get_writer(output_dir / "color.mp4", **video_kwargs)
    depth_writer = iio.get_writer(output_dir / "depth.mp4", **video_kwargs)

    n = len(trajectory)
    diag_idx = {0: "left", n - 1: "right",
                min(range(n), key=lambda i: abs(i - (n - 1) / 2)): "mid_video"}

    per_frame_metrics = []
    prev_rgb: np.ndarray | None = None

    try:
        for idx, eye in enumerate(trajectory):
            cam = camera_model.compute(eye)
            t_render = time.perf_counter()
            with torch.inference_mode():
                result = renderer(gaussians,
                                  extrinsics=cam.extrinsics[None].to(device),
                                  intrinsics=cam.intrinsics[None].to(device),
                                  image_width=cam.width, image_height=cam.height)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_render) * 1000

            # 裁剪后再编码
            color_raw = (result.color[0].permute(1, 2, 0).clamp(0, 1) * 255
                         ).to(torch.uint8).cpu().numpy()
            depth_raw = vis.colorize_depth(result.depth[0]).squeeze(0).permute(1, 2, 0).cpu().numpy()
            color_u8 = _apply_crop(color_raw, crop)
            depth_color = _apply_crop(depth_raw, crop)
            color_writer.append_data(color_u8)
            depth_writer.append_data(depth_color)

            # 相邻帧SSIM
            adj_ssim = None
            if prev_rgb is not None:
                p_s = np.asarray(Image.fromarray(prev_rgb).resize(
                    (min(prev_rgb.shape[1], _SSIM_MAX_SIDE),
                     min(prev_rgb.shape[0], _SSIM_MAX_SIDE)), Image.Resampling.BICUBIC))
                c_s = np.asarray(Image.fromarray(color_u8).resize(
                    (min(color_u8.shape[1], _SSIM_MAX_SIDE),
                     min(color_u8.shape[0], _SSIM_MAX_SIDE)), Image.Resampling.BICUBIC))
                adj_ssim = float(structural_similarity(p_s, c_s, channel_axis=2, data_range=255))
            prev_rgb = color_u8

            # 诊断帧（裁剪后，与视频一致）
            if idx in diag_idx:
                iio.imwrite(output_dir / f"frame_{diag_idx[idx]}.png", color_u8)

            per_frame_metrics.append({
                "frame": idx, "render_ms": round(elapsed_ms, 3),
                "adjacent_ssim": round(adj_ssim, 6) if adj_ssim is not None else None,
            })
    finally:
        color_writer.close()
        depth_writer.close()

    total_sec = time.perf_counter() - t0

    # ── 静止质量指标（零位移裁剪帧 vs 原始输入图）─────────────────────────
    reference_img = np.asarray(Image.open(args.input_image).convert("RGB"))
    static_metrics = compute_static_metrics(center_rgb, reference_img, lpips_model, device)
    static_metrics["script_version"] = "v2.0-prune-crop"
    static_metrics["frame_used"] = "frame_center_zero.png (strict zero-eye, cropped)"
    (output_dir / "static_metrics.json").write_text(
        json.dumps(static_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 逐帧指标CSV ────────────────────────────────────────────────────────
    with open(output_dir / "per_frame_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "render_ms", "adjacent_ssim"])
        w.writeheader()
        w.writerows(per_frame_metrics)

    # ── 剪枝统计 ──────────────────────────────────────────────────────────
    (output_dir / "prune_stats.json").write_text(
        json.dumps(prune_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 人工评分模板 ──────────────────────────────────────────────────────
    if args.manual_schema == "validation":
        video_manual = {
            "schema_version": "1.0-validation",
            "score_interpretation": "absolute_quality",
            "review_status": "",
            "overall_quality_score": None,
            "overall_pass": None,
            "depth_error_severity": None,
            "disocclusion_hole_severity": None,
            "coverage_edge_failure_severity": None,
            "thin_structure_failure_severity": None,
            "reflection_deformation_severity": None,
            "temporal_flicker_severity": None,
            "floating_gaussian_severity": None,
            "stretching_severity": None,
            "paper_feel_severity": None,
            "occlusion_error_severity": None,
            "first_artifact_frame_left": None,
            "first_artifact_frame_right": None,
            "worst_frame": None,
            "notes": "",
            "severity_scale": "0=无，1=轻微，2=明显，3=严重",
            "pass_rule": "overall_quality_score >= 2 and no unacceptable blocking artifact",
        }
    else:
        video_manual = {
            "schema_version": "2.1-prune",
            "score_interpretation": "absolute_quality",
            "review_status": "",
            "overall_quality_score": None,
            "hole_severity": None,
            "floating_gaussian_severity": None,
            "depth_layering_loss_severity": None,
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
            "prune_field_note": (
                "floating_gaussian_severity: 漂浮高斯/孤立斑块 (0=无 1=轻微 2=明显 3=严重)。"
                "depth_layering_loss_severity: 纵深层次损失 (0=无 1=轻微 2=明显 3=严重)。"
            ),
        }
    (output_dir / "video_manual.json").write_text(
        json.dumps(video_manual, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── config.json ───────────────────────────────────────────────────────
    ec = _probe_video(output_dir / "color.mp4")
    ed = _probe_video(output_dir / "depth.mp4")
    config = {
        "script_version": "v2.0-prune-crop",
        "ply_source": str(args.ply.resolve()),
        "input_image": str(args.input_image.resolve()),
        "keep_percent": args.keep_percent,
        "crop_single_side_percent": crop,
        "pruning_method": "opacity_descending",
        "n_gaussians_original": prune_stats["n_original"],
        "n_gaussians_kept": prune_stats["n_kept"],
        "opacity_min_kept": prune_stats["opacity_min_kept"],
        "max_disparity": args.max_disparity,
        "trajectory": "swipe",
        "num_steps": len(trajectory),
        "fps": args.fps,
        "render_resolution": [int(render_w), int(render_h)],
        "encoded_color_video": ec,
        "encoded_depth_video": ed,
        "static_metric_pose": "exact_zero_eye",
        "manual_schema": args.manual_schema,
        "total_seconds": round(total_sec, 3),
        "device": str(device),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    psnr_str = "inf" if static_metrics.get("psnr_is_infinite") else f"{static_metrics['psnr_db']:.2f}"
    print(f"  [完成] keep={args.keep_percent}% crop={crop}%  高斯={prune_stats['n_kept']:,}"
          f"  PSNR={psnr_str}dB  SSIM={static_metrics['ssim']:.4f}"
          f"  LPIPS={static_metrics['lpips_alex']:.4f}  耗时={round(total_sec,1)}s")


if __name__ == "__main__":
    main()
