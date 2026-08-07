from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects

from stage01_common import file_hash, prepare_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从端点 Alpha、中心 Alpha 和端点深度生成显露区域掩码。"
    )
    parser.add_argument("--endpoint-alpha", type=Path, required=True)
    parser.add_argument("--center-alpha", type=Path, required=True)
    parser.add_argument("--endpoint-depth", type=Path, required=True)
    parser.add_argument("--endpoint-rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=0.95)
    parser.add_argument("--hard-hole-threshold", type=float, default=0.50)
    parser.add_argument("--depth-edge-percentile", type=float, default=97.5)
    parser.add_argument("--depth-edge-alpha-ceiling", type=float, default=0.985)
    parser.add_argument("--dilate-px", type=int, default=5)
    parser.add_argument("--min-component-px", type=int, default=64)
    parser.add_argument(
        "--boundary-band-px", type=int, default=32,
        help="第一轮只接受触及该宽度画面边界带的连通区域",
    )
    return parser.parse_args()


def load_alpha(path: Path) -> np.ndarray:
    alpha = np.asarray(Image.open(path))
    maximum = np.iinfo(alpha.dtype).max if np.issubdtype(alpha.dtype, np.integer) else 1.0
    return alpha.astype(np.float32) / float(maximum)


def main() -> None:
    args = parse_args()
    for path in (args.endpoint_alpha, args.center_alpha, args.endpoint_depth, args.endpoint_rgb):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = prepare_output_dir(args.output_dir)

    endpoint_alpha = load_alpha(args.endpoint_alpha)
    center_alpha = load_alpha(args.center_alpha)
    depth = np.load(args.endpoint_depth).astype(np.float32)
    rgb = np.asarray(Image.open(args.endpoint_rgb).convert("RGB"))
    if endpoint_alpha.shape != center_alpha.shape or endpoint_alpha.shape != depth.shape:
        raise ValueError("Alpha 与深度分辨率不一致。")
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB 与深度分辨率不一致。")

    newly_low_alpha = (
        (endpoint_alpha < args.alpha_threshold)
        & (center_alpha >= args.alpha_threshold)
    )
    hard_hole = endpoint_alpha < args.hard_hole_threshold
    invalid_depth = ~np.isfinite(depth) | (depth <= 0)

    valid_depth = np.isfinite(depth) & (depth > 0)
    log_depth = np.zeros_like(depth, dtype=np.float32)
    log_depth[valid_depth] = np.log(depth[valid_depth])
    grad_y, grad_x = np.gradient(log_depth)
    gradient = np.hypot(grad_x, grad_y)
    finite_gradient = gradient[valid_depth]
    threshold = (
        float(np.percentile(finite_gradient, args.depth_edge_percentile))
        if finite_gradient.size
        else float("inf")
    )
    depth_conflict = (
        (gradient >= threshold)
        & (endpoint_alpha < args.depth_edge_alpha_ceiling)
        & ndi.binary_dilation(newly_low_alpha | hard_hole, iterations=max(args.dilate_px, 1) * 2)
    )

    raw_mask = newly_low_alpha | hard_hole | invalid_depth | depth_conflict
    # 低 Alpha 并不总是“新显露”：树叶、栏杆等半透明边缘也会低于阈值。
    # 仅保留与硬空洞/无效深度种子连通的候选域，避免在原本可见内容上过度补全。
    raw_labels, raw_component_count = ndi.label(raw_mask)
    seed_mask = hard_hole | invalid_depth
    seeded_labels = np.unique(raw_labels[seed_mask])
    seeded_labels = seeded_labels[seeded_labels != 0]
    seeded_mask = np.isin(raw_labels, seeded_labels)
    height, width = raw_mask.shape
    band = max(1, min(args.boundary_band_px, height // 2, width // 2))
    boundary_band = np.zeros_like(raw_mask, dtype=bool)
    boundary_band[:band] = True
    boundary_band[-band:] = True
    boundary_band[:, :band] = True
    boundary_band[:, -band:] = True
    accepted_raw = np.zeros_like(raw_mask, dtype=bool)
    rejected_internal = np.zeros_like(raw_mask, dtype=bool)
    rejected_small = np.zeros_like(raw_mask, dtype=bool)
    component_records = []
    for component_id in range(1, raw_component_count + 1):
        component = raw_labels == component_id
        size_px = int(component.sum())
        touches_boundary = bool(np.any(component & boundary_band))
        contains_seed = bool(np.any(component & seed_mask))
        accepted = size_px >= args.min_component_px and touches_boundary and contains_seed
        if accepted:
            accepted_raw |= component
            reason = "accepted_boundary_connected_seeded"
        elif size_px < args.min_component_px:
            rejected_small |= component
            reason = "rejected_too_small"
        else:
            rejected_internal |= component
            reason = "rejected_internal_or_unseeded"
        component_records.append({
            "component_id": component_id,
            "size_px": size_px,
            "touches_boundary_band": touches_boundary,
            "contains_hard_hole_or_invalid_depth_seed": contains_seed,
            "accepted": accepted,
            "reason": reason,
        })
    clean_mask = ndi.binary_closing(accepted_raw, iterations=2)
    # scikit-image 0.26 起 min_size 已弃用；max_size=N 会移除大小 <=N 的连通域。
    clean_mask = remove_small_objects(clean_mask, max_size=max(args.min_component_px - 1, 0))
    if args.dilate_px > 0:
        clean_mask = ndi.binary_dilation(clean_mask, iterations=args.dilate_px)
    clean_mask = ndi.binary_fill_holes(clean_mask)

    labels, component_count = ndi.label(clean_mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    mask_u8 = clean_mask.astype(np.uint8) * 255
    iio.imwrite(output_dir / "mask_final.png", mask_u8)
    iio.imwrite(output_dir / "mask_accepted.png", mask_u8)
    rejected_mask = raw_mask & ~clean_mask
    iio.imwrite(output_dir / "mask_rejected.png", rejected_mask.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_rejected_internal.png", rejected_internal.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_rejected_small.png", rejected_small.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_raw.png", raw_mask.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_boundary_band.png", boundary_band.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_alpha_new.png", newly_low_alpha.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_depth_conflict.png", depth_conflict.astype(np.uint8) * 255)
    iio.imwrite(output_dir / "mask_unseeded_rejected.png", (raw_mask & ~seeded_mask).astype(np.uint8) * 255)

    overlay = rgb.copy()
    overlay[clean_mask] = (
        0.35 * overlay[clean_mask].astype(np.float32)
        + 0.65 * np.array([255, 32, 32], dtype=np.float32)
    ).astype(np.uint8)
    iio.imwrite(output_dir / "mask_overlay.png", overlay)

    write_json(
        output_dir / "mask_stats.json",
        {
            "schema_version": "1.0-disocclusion-mask",
            "inputs": {
                "endpoint_alpha": str(args.endpoint_alpha.resolve()),
                "endpoint_alpha_sha256": file_hash(args.endpoint_alpha),
                "center_alpha": str(args.center_alpha.resolve()),
                "endpoint_depth": str(args.endpoint_depth.resolve()),
                "endpoint_rgb": str(args.endpoint_rgb.resolve()),
            },
            "resolution_hw": [int(depth.shape[0]), int(depth.shape[1])],
            "parameters": {
                "alpha_threshold": args.alpha_threshold,
                "hard_hole_threshold": args.hard_hole_threshold,
                "depth_edge_percentile": args.depth_edge_percentile,
                "depth_edge_threshold_observed": threshold,
                "depth_edge_alpha_ceiling": args.depth_edge_alpha_ceiling,
                "dilate_px": args.dilate_px,
                "min_component_px": args.min_component_px,
                "boundary_band_px": band,
                "policy": "boundary_connected_large_seeded_components_only",
            },
            "fractions": {
                "newly_low_alpha": float(newly_low_alpha.mean()),
                "hard_hole": float(hard_hole.mean()),
                "invalid_depth": float(invalid_depth.mean()),
                "depth_conflict": float(depth_conflict.mean()),
                "seeded_before_cleanup": float(seeded_mask.mean()),
                "unseeded_rejected": float((raw_mask & ~seeded_mask).mean()),
                "final_mask": float(clean_mask.mean()),
                "rejected_total": float(rejected_mask.mean()),
                "rejected_internal": float(rejected_internal.mean()),
                "rejected_small": float(rejected_small.mean()),
            },
            "components": {
                "count": int(component_count),
                "largest_px": int(component_sizes.max()) if component_sizes.size else 0,
                "median_px": float(np.median(component_sizes)) if component_sizes.size else 0.0,
                "raw_count": int(raw_component_count),
                "details": component_records,
            },
            "interpretation": (
                "该掩码是 Alpha/深度代理显露检测，不是真值遮挡掩码；"
                "正式结论需结合端点画面和人工检查。"
            ),
        },
    )


if __name__ == "__main__":
    main()
