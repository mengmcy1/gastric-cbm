"""计算 SHARP 零运动重渲染与冻结输入图之间的基础指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def resize_rgb(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    size = (round(width * scale), round(height * scale))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--alpha", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lpips-max-side", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(args.input).convert("RGB"))
    render = np.asarray(Image.open(args.render).convert("RGB"))
    source_original_shape = source.shape
    if source.shape != render.shape:
        source = np.asarray(
            Image.fromarray(source).resize((render.shape[1], render.shape[0]), Image.Resampling.LANCZOS)
        )
    iio.imwrite(args.output_dir / "input_aligned.png", source)
    alpha_u16 = iio.imread(args.alpha)
    if alpha_u16.shape != source.shape[:2]:
        raise ValueError("Alpha 尺寸与 RGB 不一致")
    diff = np.abs(source.astype(np.int16) - render.astype(np.int16)).astype(np.uint8)
    iio.imwrite(args.output_dir / "difference_abs.png", diff)
    psnr = float(peak_signal_noise_ratio(source, render, data_range=255))
    ssim = float(structural_similarity(source, render, channel_axis=2, data_range=255))
    source_lpips = resize_rgb(source, args.lpips_max_side)
    render_lpips = resize_rgb(render, args.lpips_max_side)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    source_tensor = torch.from_numpy(source_lpips).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    render_tensor = torch.from_numpy(render_lpips).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    with torch.inference_mode():
        lpips_value = float(model(source_tensor, render_tensor).item())
    alpha = alpha_u16.astype(np.float32) / 65535.0
    metrics = {
        "comparison": "frozen input RGB vs center re-render RGB",
        "input_original_resolution": [int(source_original_shape[1]), int(source_original_shape[0])],
        "resolution": [int(source.shape[1]), int(source.shape[0])],
        "psnr_db": psnr,
        "ssim": ssim,
        "lpips_alex": lpips_value,
        "lpips_resolution": [int(source_lpips.shape[1]), int(source_lpips.shape[0])],
        "alpha_coverage_ge_099": float((alpha >= 0.99).mean()),
        "alpha_coverage_ge_095": float((alpha >= 0.95).mean()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
