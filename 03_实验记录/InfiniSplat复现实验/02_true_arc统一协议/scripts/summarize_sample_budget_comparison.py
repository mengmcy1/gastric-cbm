#!/usr/bin/env python3
"""Compare center renders from multiple InfiniSplat sample budgets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from analyze_center_gaussian_density import grayscale_gradient, load_infinisplat_ply, tile_mean


def psnr(mae_source: np.ndarray, rendered: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = (mae_source.astype(np.float32) - rendered.astype(np.float32)) / 255.0
    if mask is not None:
        error = error[mask]
    mse = float(np.mean(error * error))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def enhanced_difference(source: np.ndarray, rendered: np.ndarray) -> Image.Image:
    difference = np.abs(source.astype(np.int16) - rendered.astype(np.int16)).astype(np.float32)
    difference = np.clip(difference * 4.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(difference, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    vertices, _, intrinsic, (width, height) = load_infinisplat_ply(Path(config["baseline_ply"]))
    xyz = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1)
    z = xyz[:, 2]
    u = intrinsic[0, 0] * xyz[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * xyz[:, 1] / z + intrinsic[1, 2]
    valid = np.isfinite(u) & np.isfinite(v) & (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    tile = int(config["tile_size_px"])
    tile_w, tile_h = width // tile, height // tile
    tx = np.clip((u[valid] // tile).astype(np.int64), 0, tile_w - 1)
    ty = np.clip((v[valid] // tile).astype(np.int64), 0, tile_h - 1)
    flat = ty * tile_w + tx
    count = np.bincount(flat, minlength=tile_h * tile_w)
    depth_sum = np.bincount(flat, weights=z[valid], minlength=tile_h * tile_w)
    mean_depth = np.divide(depth_sum, count, out=np.full_like(depth_sum, np.nan), where=count > 0).reshape(tile_h, tile_w)

    source_pil = ImageOps.exif_transpose(Image.open(config["source_rgb"])).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    source = np.asarray(source_pil)
    source_detail = tile_mean(grayscale_gradient(source), tile)
    valid_tiles = np.isfinite(mean_depth) & (source_detail > np.quantile(source_detail, 0.20))
    near_tiles = valid_tiles & (mean_depth <= np.nanquantile(mean_depth[valid_tiles], 0.25))
    far_tiles = valid_tiles & (mean_depth >= np.nanquantile(mean_depth[valid_tiles], 0.75))
    near_pixels = np.repeat(np.repeat(near_tiles, tile, axis=0), tile, axis=1)
    far_pixels = np.repeat(np.repeat(far_tiles, tile, axis=0), tile, axis=1)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    metrics = []
    frames = []
    diffs = []
    for run in config["runs"]:
        frame = Image.open(run["frame"]).convert("RGB")
        if frame.size != (width, height):
            raise ValueError(f"Unexpected frame size for {run['label']}: {frame.size}")
        rendered = np.asarray(frame)
        render_detail = tile_mean(grayscale_gradient(rendered), tile)
        retention = render_detail / np.maximum(source_detail, 1e-4)
        result = json.loads(Path(run["render_result"]).read_text(encoding="utf-8"))
        center_result = result["endpoints"]["center"]
        entry = {
            "label": run["label"],
            "requested_sample_point_num": int(run["requested_sample_point_num"]),
            "gaussian_count_after_filter": int(run["gaussian_count_after_filter"]),
            "frame": run["frame"],
            "overall": {
                "psnr_db": psnr(source, rendered),
                "rgb_mae": float(np.abs(source.astype(np.float32) - rendered.astype(np.float32)).mean() / 255.0),
                "detail_retention_median": float(np.median(retention[valid_tiles])),
            },
            "near_tiles": {
                "psnr_db": psnr(source, rendered, near_pixels),
                "rgb_mae": float(np.abs(source.astype(np.float32) - rendered.astype(np.float32))[near_pixels].mean() / 255.0),
                "detail_retention_median": float(np.median(retention[near_tiles])),
            },
            "far_tiles": {
                "psnr_db": psnr(source, rendered, far_pixels),
                "rgb_mae": float(np.abs(source.astype(np.float32) - rendered.astype(np.float32))[far_pixels].mean() / 255.0),
                "detail_retention_median": float(np.median(retention[far_tiles])),
            },
            "coverage": {
                "alpha_ge_095": float(center_result["alpha_ge_095"]),
                "hard_hole_lt_050": float(center_result["alpha_hard_hole_lt_050"]),
            },
        }
        metrics.append(entry)
        frames.append(frame)
        diffs.append(enhanced_difference(source, rendered))

    review = Image.new("RGB", (width * 4, height * 2), "black")
    review.paste(source_pil, (0, 0))
    far_overlay = source.copy()
    overlay_mask = np.repeat(np.repeat(far_tiles, tile, axis=0), tile, axis=1)
    far_overlay[overlay_mask] = (0.45 * far_overlay[overlay_mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    review.paste(Image.fromarray(far_overlay, "RGB"), (0, height))
    for index, (frame, difference) in enumerate(zip(frames, diffs), start=1):
        review.paste(frame, (index * width, 0))
        review.paste(difference, (index * width, height))
    review_path = output / "review_source_0750k_1500k_2000k_and_differences.png"
    review.save(review_path)

    baseline = next(item for item in metrics if item["label"] == "1500k_official")
    high = next(item for item in metrics if item["label"] == "2000k")
    low = next(item for item in metrics if item["label"] == "0750k")
    high_vs_baseline = {
        "overall_psnr_delta_db": high["overall"]["psnr_db"] - baseline["overall"]["psnr_db"],
        "far_psnr_delta_db": high["far_tiles"]["psnr_db"] - baseline["far_tiles"]["psnr_db"],
        "far_detail_retention_delta": high["far_tiles"]["detail_retention_median"] - baseline["far_tiles"]["detail_retention_median"],
        "hard_hole_relative_change": high["coverage"]["hard_hole_lt_050"] / baseline["coverage"]["hard_hole_lt_050"] - 1.0,
    }
    low_vs_baseline = {
        "overall_psnr_delta_db": low["overall"]["psnr_db"] - baseline["overall"]["psnr_db"],
        "far_psnr_delta_db": low["far_tiles"]["psnr_db"] - baseline["far_tiles"]["psnr_db"],
        "far_detail_retention_delta": low["far_tiles"]["detail_retention_median"] - baseline["far_tiles"]["detail_retention_median"],
        "hard_hole_relative_change": low["coverage"]["hard_hole_lt_050"] / baseline["coverage"]["hard_hole_lt_050"] - 1.0,
    }
    if high_vs_baseline["far_psnr_delta_db"] >= 0.5 and high_vs_baseline["far_detail_retention_delta"] >= 0.05:
        interpretation = "sample_budget_materially_improves_far_detail"
    elif high_vs_baseline["far_psnr_delta_db"] < 0.25 and high_vs_baseline["far_detail_retention_delta"] < 0.03:
        interpretation = "sample_budget_improves_coverage_but_not_far_detail"
    else:
        interpretation = "sample_budget_has_mixed_far_detail_effect"

    result = {
        "schema_version": "1.0-infinisplat-sample-budget-comparison",
        "status": "success",
        "experiment_id": config["experiment_id"],
        "producer_machine_id": config["producer_machine_id"],
        "scope": config["scope"],
        "far_tile_definition": "upper quartile of baseline projected mean Gaussian depth among tiles with source detail above its 20th percentile",
        "near_tile_definition": "lower quartile under the same validity rule",
        "metrics": metrics,
        "high_2000k_vs_official_1500k": high_vs_baseline,
        "low_0750k_vs_official_1500k": low_vs_baseline,
        "automatic_interpretation": interpretation,
        "review_layout": "top: source, 0750k, 1500k, 2000k; bottom: red far-tile mask, then 4x enhanced absolute differences",
        "review_path": str(review_path),
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "automatic_interpretation": interpretation, "high_2000k_vs_official_1500k": high_vs_baseline, "low_0750k_vs_official_1500k": low_vs_baseline}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
