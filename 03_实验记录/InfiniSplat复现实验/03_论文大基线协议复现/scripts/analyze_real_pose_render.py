#!/usr/bin/env python3
"""Compute transparent, non-official diagnostics for one real-pose render."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_alpha(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float32)
    return array / 65535.0


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else -10.0 * math.log10(mse)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    gt = load_rgb(output_dir / "target_ground_truth_2016x1344.jpg")
    black = load_rgb(output_dir / "target_real_pose_render_black.png")
    white = load_rgb(output_dir / "target_real_pose_render_white.png")
    alpha = load_alpha(output_dir / "target_real_pose_alpha_u16.png")
    valid = alpha >= 0.5

    black_error = np.square(black - gt).mean(axis=-1)
    white_error = np.square(white - gt).mean(axis=-1)
    _, black_ssim_map = structural_similarity(
        gt, black, channel_axis=2, data_range=1.0, full=True
    )
    _, white_ssim_map = structural_similarity(
        gt, white, channel_axis=2, data_range=1.0, full=True
    )
    black_ssim_scalar_map = black_ssim_map.mean(axis=-1)
    diagnostics = {
        "schema_version": "1.0-real-pose-render-diagnostics",
        "status": "success",
        "warning": (
            "These are local diagnostics, not paper-comparable metrics: the authors' "
            "exact frustum-visible evaluation mask/pair list is unavailable, and ETH3D "
            "color alignment is not reproduced here."
        ),
        "coverage": {
            "alpha_ge_0_5_fraction": float(valid.mean()),
            "alpha_ge_0_01_fraction": float((alpha >= 0.01).mean()),
            "mean_alpha": float(alpha.mean()),
        },
        "full_frame_black_background": {
            "psnr_db": psnr_from_mse(float(black_error.mean())),
            "ssim": float(structural_similarity(gt, black, channel_axis=2, data_range=1.0)),
        },
        "full_frame_white_background": {
            "psnr_db": psnr_from_mse(float(white_error.mean())),
            "ssim": float(structural_similarity(gt, white, channel_axis=2, data_range=1.0)),
        },
        "covered_region_alpha_ge_0_5": {
            "psnr_db": psnr_from_mse(float(black_error[valid].mean())),
            "ssim_map_mean": float(black_ssim_scalar_map[valid].mean()),
            "pixel_fraction": float(valid.mean()),
        },
        "alpha_weighted_black_background": {
            "psnr_db": psnr_from_mse(float((black_error * alpha).sum() / alpha.sum())),
            "ssim_map_mean": float(
                (black_ssim_scalar_map * alpha).sum() / alpha.sum()
            ),
        },
        "interpretation": (
            "Use coverage to describe extrapolation/disocclusion failure and covered-region "
            "metrics only as a sanity check for visible-surface alignment."
        ),
    }
    destination = output_dir / "diagnostic_metrics.json"
    destination.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
