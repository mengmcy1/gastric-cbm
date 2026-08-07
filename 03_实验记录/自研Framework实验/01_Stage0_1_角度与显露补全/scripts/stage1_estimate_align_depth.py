from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

from stage01_common import (
    SHARP_SRC,
    configure_sharp_cuda_toolkit,
    file_hash,
    prepare_output_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 SHARP 冻结单目深度骨干重估补全图深度，并对齐端点已有深度。"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--reference-depth", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--focal-px", type=float, required=True, help="当前 RGB 分辨率下的焦距像素值")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-layer", type=int, default=0)
    parser.add_argument("--ring-px", type=int, default=48)
    parser.add_argument("--max-alignment-samples", type=int, default=200000)
    parser.add_argument(
        "--alignment-model",
        choices=["depth_affine", "inverse_depth_affine", "depth_affine_with_inverse_fallback"],
        default="depth_affine",
    )
    parser.add_argument(
        "--max-normalized-rmse",
        type=float,
        default=0.25,
        help="深度对齐 RMSE / 参考环带中位深度的最大允许值",
    )
    parser.add_argument(
        "--allow-poor-alignment",
        action="store_true",
        help="仅供工具链冒烟：记录失败质量门但继续输出，正式实验不得启用",
    )
    return parser.parse_args()


def robust_affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    if x.size < 128:
        raise ValueError(f"深度对齐样本不足：{x.size}")
    keep = np.ones(x.size, dtype=bool)
    coef = np.array([1.0, 0.0], dtype=np.float64)
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
    rmse = float(np.sqrt(np.mean((y[keep] - (coef[0] * x[keep] + coef[1])) ** 2)))
    return float(coef[0]), float(coef[1]), rmse, int(keep.sum())


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return preview
    lo, hi = np.percentile(depth[valid], [2, 98])
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    # 不引入额外绘图库：近处暖色，远处冷色的简单诊断图。
    preview[..., 0] = ((1.0 - normalized) * 255).astype(np.uint8)
    preview[..., 1] = (4.0 * normalized * (1.0 - normalized) * 255).astype(np.uint8)
    preview[..., 2] = (normalized * 255).astype(np.uint8)
    preview[~valid] = 0
    return preview


def robust_depth_rmse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(prediction) & (prediction > 0) & np.isfinite(target) & (target > 0)
    prediction, target = prediction[finite], target[finite]
    if prediction.size < 128:
        raise ValueError(f"深度误差评估样本不足：{prediction.size}")
    residual = target - prediction
    median = np.median(residual)
    mad = np.median(np.abs(residual - median)) + 1e-8
    keep = np.abs(residual - median) <= 3.5 * 1.4826 * mad
    return float(np.sqrt(np.mean(residual[keep] ** 2))), int(keep.sum())


def main() -> None:
    args = parse_args()
    for path in (args.image, args.mask, args.reference_depth, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("当前正式深度后端要求可用 CUDA。")
    configure_sharp_cuda_toolkit()
    from sharp.models import PredictorParams, create_predictor

    output_dir = prepare_output_dir(args.output_dir)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    reference = np.load(args.reference_depth).astype(np.float32)
    if image.shape[:2] != mask.shape or mask.shape != reference.shape:
        raise ValueError("RGB、掩码与参考深度的分辨率必须一致。")

    device = torch.device("cuda")
    load_start = time.perf_counter()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    monodepth = predictor.monodepth_model.eval()
    del predictor, state_dict
    gc.collect()
    monodepth.to(device)
    torch.cuda.empty_cache()
    model_load_seconds = time.perf_counter() - load_start

    image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)[None] / 255.0
    image_tensor = F.interpolate(
        image_tensor, size=(1536, 1536), mode="bilinear", align_corners=True
    ).to(device)
    disparity_factor = float(args.focal_px) / float(image.shape[1])
    torch.cuda.reset_peak_memory_stats(device)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        output = monodepth(image_tensor)
        disparity = output.disparity
        if args.depth_layer < 0 or args.depth_layer >= disparity.shape[1]:
            raise ValueError(
                f"depth_layer={args.depth_layer} 超出模型输出通道数 {disparity.shape[1]}。"
            )
        predicted = disparity_factor / disparity[:, args.depth_layer : args.depth_layer + 1].clamp(
            1e-4, 1e4
        )
        predicted = F.interpolate(
            predicted, size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    peak_cuda = int(torch.cuda.max_memory_allocated(device))
    predicted_np = predicted.float().cpu().numpy()

    valid_reference = np.isfinite(reference) & (reference > 0)
    labels, component_count = ndi.label(mask)
    accepted_mask = np.zeros_like(mask, dtype=bool)
    rejected_mask = np.zeros_like(mask, dtype=bool)
    aligned = np.full_like(predicted_np, np.nan, dtype=np.float32)
    component_records = []
    rng = np.random.default_rng(0)
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        ring = ndi.binary_dilation(component, iterations=max(args.ring_px, 1)) & ~mask
        alignment_region = (
            ring & valid_reference & np.isfinite(predicted_np) & (predicted_np > 0)
        )
        ys, xs = np.where(alignment_region)
        raw_sample_count = int(ys.size)
        record = {
            "component_id": component_id,
            "component_pixels": int(component.sum()),
            "ring_sample_count": raw_sample_count,
        }
        try:
            if ys.size > args.max_alignment_samples:
                chosen = rng.choice(ys.size, args.max_alignment_samples, replace=False)
                ys, xs = ys[chosen], xs[chosen]
            x_samples, y_samples = predicted_np[ys, xs], reference[ys, xs]
            reference_median = float(np.median(reference[alignment_region]))
            candidates = {}
            depth_scale, depth_offset, depth_rmse, depth_inliers = robust_affine_fit(
                x_samples, y_samples
            )
            depth_nrmse = depth_rmse / max(reference_median, 1e-6)
            depth_valid = bool(0.1 <= depth_scale <= 10.0 and depth_nrmse <= args.max_normalized_rmse)
            candidates["depth_affine"] = {
                "scale": depth_scale, "offset": depth_offset, "rmse_m": depth_rmse,
                "normalized_rmse": depth_nrmse, "inlier_count": depth_inliers,
                "scale_in_range_0p1_to_10": 0.1 <= depth_scale <= 10.0,
                "quality_gate_status": "passed" if depth_valid else "failed",
            }
            if args.alignment_model == "depth_affine":
                selected_model = "depth_affine"
                scale, offset, rmse, inlier_count = depth_scale, depth_offset, depth_rmse, depth_inliers
                component_aligned = scale * predicted_np + offset
            elif args.alignment_model == "inverse_depth_affine" or not depth_valid:
                scale, offset, _, _ = robust_affine_fit(1.0 / x_samples, 1.0 / y_samples)
                denominator = scale / predicted_np + offset
                inverse_aligned = np.where(denominator > 1e-8, 1.0 / denominator, np.nan)
                inverse_rmse, inverse_inliers = robust_depth_rmse(
                    inverse_aligned[ys, xs], y_samples
                )
                inverse_nrmse = inverse_rmse / max(reference_median, 1e-6)
                inverse_valid = bool(0.1 <= scale <= 10.0 and inverse_nrmse <= args.max_normalized_rmse)
                candidates["inverse_depth_affine"] = {
                    "scale": scale, "offset": offset, "rmse_m": inverse_rmse,
                    "normalized_rmse": inverse_nrmse, "inlier_count": inverse_inliers,
                    "scale_in_range_0p1_to_10": 0.1 <= scale <= 10.0,
                    "quality_gate_status": "passed" if inverse_valid else "failed",
                }
                selected_model = "inverse_depth_affine"
                rmse, inlier_count = inverse_rmse, inverse_inliers
                component_aligned = inverse_aligned
            else:
                selected_model = "depth_affine"
                scale, offset, rmse, inlier_count = depth_scale, depth_offset, depth_rmse, depth_inliers
                component_aligned = scale * predicted_np + offset
            normalized_rmse = rmse / max(reference_median, 1e-6)
            scale_in_range = 0.1 <= scale <= 10.0
            accepted = bool(scale_in_range and normalized_rmse <= args.max_normalized_rmse)
            ring_values = reference[alignment_region]
            clamp_lo, clamp_hi = np.percentile(ring_values, [1, 99])
            component_aligned = np.clip(
                component_aligned,
                max(float(clamp_lo) * 0.5, 1e-4),
                float(clamp_hi) * 2.0,
            )
            aligned[component] = component_aligned[component]
            record.update({
                "scale": scale,
                "offset": offset,
                "rmse_m": rmse,
                "reference_median_m": reference_median,
                "normalized_rmse": normalized_rmse,
                "inlier_count": inlier_count,
                "scale_in_range_0p1_to_10": scale_in_range,
                "clamp_reference_p01_p99_m": [float(clamp_lo), float(clamp_hi)],
                "quality_gate_status": "passed" if accepted else "failed",
                "reason": "accepted" if accepted else "scale_or_normalized_rmse_failed",
                "selected_model": selected_model,
                "candidate_models": candidates,
            })
        except ValueError as error:
            accepted = False
            record.update({
                "scale": None,
                "offset": None,
                "rmse_m": None,
                "reference_median_m": None,
                "normalized_rmse": None,
                "inlier_count": 0,
                "scale_in_range_0p1_to_10": False,
                "clamp_reference_p01_p99_m": None,
                "quality_gate_status": "failed",
                "reason": str(error),
            })
        if accepted:
            accepted_mask |= component
        else:
            rejected_mask |= component
        component_records.append(record)

    no_op = component_count == 0
    alignment_acceptable = bool(no_op or accepted_mask.any())
    gate_status = (
        "not_applicable_no_mask" if no_op
        else "passed" if alignment_acceptable and not rejected_mask.any()
        else "partial_pass" if alignment_acceptable
        else "failed"
    )
    filled = reference.copy()
    filled[accepted_mask] = aligned[accepted_mask]

    np.save(output_dir / "depth_predicted_raw_float32.npy", predicted_np.astype(np.float32))
    np.save(output_dir / "depth_aligned_float32.npy", aligned.astype(np.float32))
    np.save(output_dir / "depth_filled_float32.npy", filled.astype(np.float32))
    iio.imwrite(output_dir / "mask_depth_accepted.png", accepted_mask.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_depth_rejected.png", rejected_mask.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "depth_predicted_preview.png", depth_preview(predicted_np))
    iio.imwrite(output_dir / "depth_filled_preview.png", depth_preview(filled))
    run_record = {
            "schema_version": "2.0-sharp-monodepth-component-align",
            "backend": "frozen_sharp_monodepth",
            "sharp_source": str(SHARP_SRC),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_hash(args.checkpoint),
            "inputs": {
                "image": str(args.image.resolve()),
                "mask": str(args.mask.resolve()),
                "reference_depth": str(args.reference_depth.resolve()),
            },
            "resolution_hw": [int(image.shape[0]), int(image.shape[1])],
            "focal_px_render": args.focal_px,
            "depth_layer": args.depth_layer,
            "alignment": {
                "model": args.alignment_model,
                "ring_px": args.ring_px,
                "component_count": int(component_count),
                "accepted_component_count": int(sum(r["quality_gate_status"] == "passed" for r in component_records)),
                "rejected_component_count": int(sum(r["quality_gate_status"] != "passed" for r in component_records)),
                "accepted_pixels": int(accepted_mask.sum()),
                "rejected_pixels": int(rejected_mask.sum()),
                "components": component_records,
            },
            "quality_gate": {
                "status": gate_status,
                "max_normalized_rmse": args.max_normalized_rmse,
                "allow_poor_alignment": args.allow_poor_alignment,
                "policy": "failed components are excluded; an empty accepted mask is a valid baseline-unchanged no-op",
                "no_op_means_baseline_unchanged": no_op,
            },
            "timing_seconds": {"model_load": model_load_seconds, "inference": inference_seconds},
            "peak_cuda_allocated_bytes": peak_cuda,
            "limitations": (
                "复用 SHARP 冻结单目深度头，仅作 Stage 1 可行性深度；"
                "两层深度未硬排序，默认第 0 层不代表真实隐藏表面。"
            ),
        }
    write_json(
        output_dir / "depth_run.json",
        run_record,
    )
    if not alignment_acceptable and not args.allow_poor_alignment:
        raise RuntimeError(
            "深度对齐质量门未通过："
            f"accepted={int(accepted_mask.sum())}px, rejected={int(rejected_mask.sum())}px；"
            f"诊断结果已保存在 {output_dir}。"
        )


if __name__ == "__main__":
    main()
