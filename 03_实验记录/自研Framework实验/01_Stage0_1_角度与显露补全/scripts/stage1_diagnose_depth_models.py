from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from stage01_common import prepare_output_dir, write_json
from stage1_estimate_align_depth import robust_affine_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐连通区域比较深度对齐模型，仅作诊断，不生成高斯。")
    parser.add_argument("--predicted-depth", type=Path, required=True)
    parser.add_argument("--reference-depth", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--ring-px", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def robust_rmse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    residual = target - prediction
    finite = np.isfinite(residual) & np.isfinite(target) & (target > 0) & (prediction > 0)
    residual = residual[finite]
    if residual.size < 128:
        return float("nan"), int(residual.size)
    median = np.median(residual)
    mad = np.median(np.abs(residual - median)) + 1e-8
    keep = np.abs(residual - median) <= 3.5 * 1.4826 * mad
    return float(np.sqrt(np.mean(residual[keep] ** 2))), int(keep.sum())


def robust_affine_fit_any(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 128:
        raise ValueError(f"拟合样本不足：{x.size}")
    keep = np.ones(x.size, dtype=bool)
    coef = np.array([1.0, 0.0])
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        coef, *_ = np.linalg.lstsq(design, y[keep], rcond=None)
        residual = y - (coef[0] * x + coef[1])
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - median)) + 1e-8
        next_keep = np.abs(residual - median) <= 3.5 * 1.4826 * mad
        if next_keep.sum() < 128 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return float(coef[0]), float(coef[1])


def main() -> None:
    args = parse_args()
    for path in (args.predicted_depth, args.reference_depth, args.mask):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = prepare_output_dir(args.output_dir)
    predicted = np.load(args.predicted_depth).astype(np.float64)
    reference = np.load(args.reference_depth).astype(np.float64)
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    if predicted.shape != reference.shape or mask.shape != reference.shape:
        raise ValueError("输入分辨率不一致。")
    labels, count = ndi.label(mask)
    components = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        ring = ndi.binary_dilation(component, iterations=max(args.ring_px, 1)) & ~mask
        valid = ring & np.isfinite(predicted) & (predicted > 0) & np.isfinite(reference) & (reference > 0)
        x, y = predicted[valid], reference[valid]
        reference_median = float(np.median(y))
        models = {}

        scale, offset, _, _ = robust_affine_fit(x, y)
        rmse, inliers = robust_rmse(scale * x + offset, y)
        models["depth_affine"] = {
            "parameters": {"scale": scale, "offset": offset},
            "rmse_m": rmse,
            "normalized_rmse": rmse / reference_median,
            "inlier_count": inliers,
        }

        scale_only = float(np.median(y / x))
        rmse, inliers = robust_rmse(scale_only * x, y)
        models["depth_scale_only_median_ratio"] = {
            "parameters": {"scale": scale_only},
            "rmse_m": rmse,
            "normalized_rmse": rmse / reference_median,
            "inlier_count": inliers,
        }

        log_scale, log_offset = robust_affine_fit_any(np.log(x), np.log(y))
        log_prediction = np.exp(log_scale * np.log(x) + log_offset)
        rmse, inliers = robust_rmse(log_prediction, y)
        models["log_depth_affine"] = {
            "parameters": {"scale": log_scale, "offset": log_offset},
            "rmse_m": rmse,
            "normalized_rmse": rmse / reference_median,
            "inlier_count": inliers,
        }

        inv_scale, inv_offset, _, _ = robust_affine_fit(1.0 / x, 1.0 / y)
        denominator = inv_scale / x + inv_offset
        inverse_prediction = np.where(denominator > 1e-8, 1.0 / denominator, np.nan)
        rmse, inliers = robust_rmse(inverse_prediction, y)
        models["inverse_depth_affine"] = {
            "parameters": {"scale": inv_scale, "offset": inv_offset},
            "rmse_m": rmse,
            "normalized_rmse": rmse / reference_median,
            "inlier_count": inliers,
        }

        best = min(models, key=lambda name: models[name]["normalized_rmse"])
        components.append({
            "component_id": component_id,
            "component_pixels": int(component.sum()),
            "ring_samples": int(x.size),
            "reference_median_m": reference_median,
            "models": models,
            "best_normalized_rmse_model": best,
            "best_normalized_rmse": models[best]["normalized_rmse"],
        })
    write_json(
        output_dir / "depth_model_diagnostic.json",
        {
            "schema_version": "1.0-depth-alignment-model-diagnostic",
            "component_count": count,
            "ring_px": args.ring_px,
            "components": components,
            "boundary": "诊断结果不改变正式质量门，也不会生成补充高斯。",
        },
    )


if __name__ == "__main__":
    main()
