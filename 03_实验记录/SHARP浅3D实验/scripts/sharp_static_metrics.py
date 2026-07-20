"""计算 SHARP 零运动重渲染与冻结输入图之间的基础指标。

v2.1 变更（2026-07-20）：
- 新增 Alpha 掩码 PSNR / SSIM（alpha≥0.99 和 alpha≥0.95 区域）。
- SSIM 掩码指标：先计算局部 SSIM 图，再在掩码内取平均。
- 支持 --output-name 指定输出文件名，避免覆盖旧指标文件。
- 指标 JSON 与辅助图片默认均拒绝覆盖；非默认指标名自动生成同版本辅助图片名。
"""

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


def _psnr_masked(source: np.ndarray, render: np.ndarray, mask: np.ndarray) -> float:
    """计算掩码区域内的 PSNR (dB)。

    source, render: uint8 RGB [H,W,3]
    mask: float [H,W]，True 表示参与计算
    """
    if not mask.any():
        return float("nan")
    s = source[mask].astype(np.float64)
    r = render[mask].astype(np.float64)
    mse = np.mean((s - r) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_masked(source: np.ndarray, render: np.ndarray, mask: np.ndarray) -> float:
    """计算掩码区域内的 SSIM（先生成局部 SSIM 图，再在掩码内取平均）。

    source, render: uint8 RGB [H,W,3]
    mask: float [H,W]，True 表示参与计算
    """
    if not mask.any():
        return float("nan")
    ssim_full, ssim_map = structural_similarity(
        source, render, channel_axis=2, data_range=255, full=True
    )
    # ssim_map: [H,W]，每个像素的局部 SSIM
    return float(ssim_map[mask].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="冻结原始图片路径")
    parser.add_argument("--render", type=Path, required=True, help="中心帧渲染 PNG 路径")
    parser.add_argument("--alpha", type=Path, required=True, help="中心帧 16-bit Alpha PNG 路径")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lpips-max-side", type=int, default=1024)
    parser.add_argument("--output-name", type=str, default="metrics.json",
                        help="输出 JSON 文件名（默认 metrics.json；可设为 metrics_v2.json 避免覆盖）")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="允许覆盖本次将生成的指标 JSON 和辅助图片（默认拒绝）")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".json":
        raise ValueError("--output-name 必须是当前输出目录下的 .json 文件名")
    if output_name.stem == "metrics":
        suffix = ""
    elif output_name.stem.startswith("metrics"):
        suffix = output_name.stem[len("metrics"):]
    else:
        suffix = f"_{output_name.stem}"
    aligned_path = args.output_dir / f"input_aligned{suffix}.png"
    diff_path = args.output_dir / f"difference_abs{suffix}.png"
    out_path = args.output_dir / output_name
    targets = [aligned_path, diff_path, out_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.allow_overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"拒绝覆盖已有输出：{names}")

    # ---- 加载图像 ----
    source = np.asarray(Image.open(args.input).convert("RGB"))
    render = np.asarray(Image.open(args.render).convert("RGB"))
    source_original_shape = source.shape

    if source.shape != render.shape:
        source = np.asarray(
            Image.fromarray(source).resize((render.shape[1], render.shape[0]), Image.Resampling.LANCZOS)
        )
    iio.imwrite(aligned_path, source)

    # ---- Alpha 掩码 ----
    alpha_u16 = iio.imread(args.alpha)
    if alpha_u16.shape != source.shape[:2]:
        raise ValueError(f"Alpha 尺寸 {alpha_u16.shape} 与 RGB {source.shape[:2]} 不一致")
    alpha = alpha_u16.astype(np.float32) / 65535.0
    mask_099 = alpha >= 0.99
    mask_095 = alpha >= 0.95

    # ---- 差异图 ----
    diff = np.abs(source.astype(np.int16) - render.astype(np.int16)).astype(np.uint8)
    iio.imwrite(diff_path, diff)

    # ---- 全图 PSNR / SSIM ----
    psnr_full = float(peak_signal_noise_ratio(source, render, data_range=255))
    ssim_full = float(structural_similarity(source, render, channel_axis=2, data_range=255))

    # ---- Alpha 掩码 PSNR / SSIM ----
    psnr_099 = _psnr_masked(source, render, mask_099)
    psnr_095 = _psnr_masked(source, render, mask_095)
    ssim_099 = _ssim_masked(source, render, mask_099)
    ssim_095 = _ssim_masked(source, render, mask_095)

    # ---- LPIPS ----
    source_lpips = resize_rgb(source, args.lpips_max_side).copy()
    render_lpips = resize_rgb(render, args.lpips_max_side).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    source_tensor = torch.from_numpy(source_lpips).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    render_tensor = torch.from_numpy(render_lpips).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    with torch.inference_mode():
        lpips_value = float(model(source_tensor, render_tensor).item())

    # ---- Alpha 覆盖率 ----
    alpha_cov_099 = float(mask_099.mean())
    alpha_cov_095 = float(mask_095.mean())

    # ---- 组装指标 ----
    metrics = {
        "script_version": "v2.1",
        "comparison": "frozen input RGB vs center re-render RGB",
        "input_original_resolution": [int(source_original_shape[1]), int(source_original_shape[0])],
        "resolution": [int(source.shape[1]), int(source.shape[0])],
        "psnr_db": round(psnr_full, 6),
        "ssim": round(ssim_full, 6),
        "psnr_alpha_ge_099": round(psnr_099, 6),
        "psnr_alpha_ge_095": round(psnr_095, 6),
        "ssim_alpha_ge_099": round(ssim_099, 6),
        "ssim_alpha_ge_095": round(ssim_095, 6),
        "ssim_masked_method": (
            "Full SSIM map computed via skimage structural_similarity(full=True), "
            "then averaged within alpha mask region. "
            "Not equivalent to per-pixel PSNR masking."
        ),
        "lpips_alex": round(lpips_value, 6),
        "lpips_resolution": [int(source_lpips.shape[1]), int(source_lpips.shape[0])],
        "alpha_coverage_ge_099": round(alpha_cov_099, 6),
        "alpha_coverage_ge_095": round(alpha_cov_095, 6),
    }

    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
