"""SHARP 浅 3D 裁剪实验工具：对已渲染视频做四边单边后裁剪并放大回原尺寸。

裁剪是纯后处理，不重新渲染。输入为阶段 B 的 md004 视频。
输出目录：outputs/04_裁剪实验/{样本}/crop{00,03,05,10}/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as iio
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "outputs" / "03_运动实验"
DEFAULT_DST = REPO_ROOT / "outputs" / "04_裁剪实验"
SAMPLES = ("P01", "P02", "P03", "P04", "P05")
SRC_CONFIG = "md004_swipe60_keep100_crop00"
CROP_LEVELS = (0, 3, 5, 10)

# v2.0 人工评分模板
VIDEO_MANUAL_TEMPLATE = {
    "schema_version": "2.1-crop",
    "review_status": "",
    "overall_quality_score": "",
    "hole_severity": "",
    "stretching_severity": "",
    "flicker_severity": "",
    "paper_feel_severity": "",
    "occlusion_error_severity": "",
    "reflection_deformation_severity": "",
    "edge_artifact_reduction_score": "",
    "sharpness_loss_severity": "",
    "framing_loss_severity": "",
    "first_artifact_frame_left": "",
    "first_artifact_frame_right": "",
    "worst_frame": "",
    "overall_pass": "",
    "notes": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST)
    parser.add_argument("--samples", nargs="*", default=list(SAMPLES),
                        help=f"要处理的样本（默认全部：{' '.join(SAMPLES)}）")
    parser.add_argument("--levels", type=int, nargs="*", default=list(CROP_LEVELS),
                        help=f"裁剪比例（默认：{' '.join(str(l) for l in CROP_LEVELS)}）")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg",
                        help="ffmpeg 可执行文件路径或命令名")
    parser.add_argument("--lpips-max-side", type=int, default=1024,
                        help="LPIPS 输入图像最长边（默认 1024，与静止基线一致）")
    parser.add_argument("--lpips-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="LPIPS 计算设备（默认 auto：CUDA 可用时使用 CUDA）")
    parser.add_argument("--crf", type=int, default=18,
                        help="FFmpeg 视频编码 CRF（默认 18）")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="允许覆盖已有裁剪输出")
    return parser.parse_args()


def _check_ffmpeg(ffmpeg_cmd: str) -> None:
    try:
        subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"找不到 {ffmpeg_cmd}；请确认 FFmpeg 已安装并在 PATH 中")
        sys.exit(1)


def _crop_video(
    src: Path,
    dst: Path,
    crop_percent: int,
    ffmpeg_cmd: str,
    crf: int,
    target_width: int,
    target_height: int,
) -> None:
    """用 FFmpeg 对视频做四边单边裁剪并放大回原尺寸。"""
    # 单边 X% → 裁后宽度 = 原宽 × (1 - 2X/100)，然后放大回原宽
    # 例：3% → crop=iw*0.94:ih*0.94:iw*0.03:ih*0.03
    p = crop_percent / 100.0
    keep = 1.0 - 2 * p
    crop_w = f"iw*{keep}"
    crop_h = f"ih*{keep}"
    crop_x = f"iw*{p}"
    crop_y = f"ih*{p}"

    vf = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={target_width}:{target_height}"
    )

    cmd = [
        ffmpeg_cmd, "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",  # 不需要音频
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _crop_frame(src: Path, dst: Path, crop_percent: int) -> tuple[int, int]:
    """裁剪单帧 PNG，返回 (宽, 高) 用于后续 PSNR/SSIM。"""
    img = Image.open(src)
    w, h = img.size
    p = crop_percent / 100.0
    left = round(w * p)
    upper = round(h * p)
    right = round(w * (1 - p))
    lower = round(h * (1 - p))
    cropped = img.crop((left, upper, right, lower))
    # 放大回原尺寸
    restored = cropped.resize((w, h), Image.Resampling.LANCZOS)
    restored.save(dst)
    return w, h


def _resize_rgb(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    size = (round(width * scale), round(height * scale))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))


def _compute_crop_metrics(
    original_frame: Path,
    cropped_frame: Path,
    lpips_model: torch.nn.Module,
    lpips_device: torch.device,
    lpips_max_side: int,
) -> dict:
    """计算裁剪帧相对未裁剪渲染帧的 PSNR、SSIM 和 LPIPS。"""
    orig = np.asarray(Image.open(original_frame).convert("RGB"))
    crop = np.asarray(Image.open(cropped_frame).convert("RGB"))
    if orig.shape != crop.shape:
        crop_img = Image.fromarray(crop).resize((orig.shape[1], orig.shape[0]), Image.Resampling.LANCZOS)
        crop = np.asarray(crop_img)
    orig_lpips = _resize_rgb(orig, lpips_max_side).copy()
    crop_lpips = _resize_rgb(crop, lpips_max_side).copy()
    orig_tensor = (
        torch.from_numpy(orig_lpips).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device)
        / 127.5 - 1
    )
    crop_tensor = (
        torch.from_numpy(crop_lpips).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device)
        / 127.5 - 1
    )
    with torch.inference_mode():
        lpips_value = float(lpips_model(orig_tensor, crop_tensor).item())
    psnr_value = float(peak_signal_noise_ratio(orig, crop, data_range=255))
    return {
        "metric_version": "crop_v2.1",
        "comparison": "cropped-and-rescaled center render vs uncropped center render",
        "psnr_db": None if not np.isfinite(psnr_value) else round(psnr_value, 6),
        "psnr_is_infinite": bool(np.isinf(psnr_value)),
        "ssim": round(float(structural_similarity(orig, crop, channel_axis=2, data_range=255)), 6),
        "lpips_alex": round(lpips_value, 6),
        "lpips_resolution": [int(orig_lpips.shape[1]), int(orig_lpips.shape[0])],
    }


def main() -> None:
    args = parse_args()
    _check_ffmpeg(args.ffmpeg)
    if args.lpips_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --lpips-device cuda，但当前环境 CUDA 不可用")
    if args.lpips_device == "auto":
        lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        lpips_device = torch.device(args.lpips_device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()
    print(f"LPIPS 设备：{lpips_device}")

    for sample in args.samples:
        if sample not in SAMPLES:
            print(f"警告：未知样本 {sample}，跳过")
            continue

        src_dir = args.src_root / sample / SRC_CONFIG
        if not src_dir.is_dir():
            print(f"缺少运动实验源目录：{src_dir}，跳过 {sample}")
            continue

        color_mp4 = src_dir / "color.mp4"
        center_png = src_dir / "frame_center.png"
        if not color_mp4.is_file() or not center_png.is_file():
            print(f"缺少 color.mp4 或 frame_center.png：{src_dir}，跳过 {sample}")
            continue
        with Image.open(center_png) as center_image:
            target_width, target_height = center_image.size

        for level in args.levels:
            dst_dir = args.dst_root / sample / f"crop{level:02d}"
            if level == 0:
                # crop00：直接复用 md004 视频（不做裁剪，仅建立对应目录）
                dst_dir.mkdir(parents=True, exist_ok=True)
                # 复制 color.mp4（不重新编码）
                dst_video = dst_dir / "color.mp4"
                dst_frame = dst_dir / "frame_center.png"
                existing = list(dst_dir.iterdir())
                if existing and not args.allow_overwrite:
                    print(f"  {sample}/crop00 已存在，跳过（已有 {len(existing)} 个文件）")
                else:
                    shutil.copy2(color_mp4, dst_video)
                    shutil.copy2(center_png, dst_frame)
                    metrics = _compute_crop_metrics(
                        center_png, center_png, lpips_model, lpips_device, args.lpips_max_side
                    )
                    metrics["crop_level"] = 0
                    metrics["note"] = "crop00 直接复用 md004 原始视频，无裁剪、无放大损失"
                    (dst_dir / "crop_metrics.json").write_text(
                        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    (dst_dir / "video_manual.json").write_text(
                        json.dumps(VIDEO_MANUAL_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"  {sample}/crop00 完成（复用原始视频）")
                continue

            # crop03/05/10
            dst_dir.mkdir(parents=True, exist_ok=True)
            existing = list(dst_dir.iterdir())
            if existing and not args.allow_overwrite:
                print(f"  {sample}/crop{level:02d} 已存在，跳过（已有 {len(existing)} 个文件）")
                continue

            # 裁剪视频
            dst_video = dst_dir / "color.mp4"
            _crop_video(
                color_mp4,
                dst_video,
                level,
                args.ffmpeg,
                args.crf,
                target_width,
                target_height,
            )

            # 裁剪中心帧
            dst_frame = dst_dir / "frame_center.png"
            _crop_frame(center_png, dst_frame, level)

            # 计算 PSNR/SSIM/LPIPS
            metrics = _compute_crop_metrics(
                center_png, dst_frame, lpips_model, lpips_device, args.lpips_max_side
            )
            metrics["crop_level"] = level
            metrics["crop_definition"] = (
                f"单边裁剪 {level}%，四边各裁掉 {level}% 后放大回原显示尺寸。"
                f"剩余宽度 = 原宽 × {100 - 2 * level}%，放大倍数 = {100 / (100 - 2 * level):.4f}×"
            )
            (dst_dir / "crop_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

            # 空白评分模板
            (dst_dir / "video_manual.json").write_text(
                json.dumps(VIDEO_MANUAL_TEMPLATE, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"  {sample}/crop{level:02d} 完成  PSNR={metrics['psnr_db']} dB  SSIM={metrics['ssim']}")

    print("\n全部裁剪任务完成。")


if __name__ == "__main__":
    main()
