#!/usr/bin/env python3
"""Build the MG0b size-matched teacher-ROI sensitivity manifest.

The locked MG0b manifest uses cancer GT boxes and non-cancer detector boxes.
Its geometry audit found that crop size alone can partially predict the label.
This sensitivity variant leaves every cancer crop unchanged and uses each
non-cancer ROI center as the placement target, but deterministically assigns its
crop side length from the train-cancer distribution in the closest source/size
stratum. Large crops near an edge are shifted only as needed to stay in-frame.
It does not replace the locked MG0b manifest or read test/external data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_mage_teacher_roi_manifest import (
    PROJECT_ROOT,
    file_sha256,
    patient_mean_auc,
    stratified_qc,
)


DEFAULT_INPUT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_教师ROI清单与审计_20260817/"
    "mage_teacher_roi_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0b_教师ROI尺寸匹配敏感性_20260817"
)
GEOMETRY_GROUPS = {
    "center_only": ["base_center_x", "base_center_y"],
    "normalized_size_only": ["base_width", "base_height", "base_area"],
    "pixel_square_size_only": ["base_side_min_dim_fraction"],
    "center_plus_pixel_size": [
        "base_center_x", "base_center_y", "base_side_min_dim_fraction"
    ],
    "full_without_aspect": [
        "base_center_x", "base_center_y", "base_width", "base_height", "base_area"
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse the locked input manifest, output directory and self-test mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def hash_uniform(value: str) -> float:
    """Map a stable identifier to a deterministic value in the closed unit interval."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def quantile_from_sorted(values: np.ndarray, quantile: float) -> float:
    """Linearly interpolate one quantile from a non-empty sorted array."""
    if not len(values):
        raise ValueError("尺寸匹配池为空")
    position = quantile * (len(values) - 1)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def add_pixel_side_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add square crop side as a fraction of the image's shorter pixel dimension."""
    output = frame.copy()
    side_pixels = output.base_width * output.width
    output["base_side_min_dim_fraction"] = side_pixels / output[["width", "height"]].min(axis=1)
    return output


def build_cancer_size_pools(
    frame: pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    """Fit source-size, source-only and global crop-size pools on train cancer only."""
    cancer = frame.loc[frame.split.eq("train") & frame.label.eq(1)].copy()
    pools: dict[tuple[str, str], np.ndarray] = {}
    for keys, subset in cancer.groupby(["source", "size_group"], dropna=False):
        pools[(str(keys[0]), str(keys[1]))] = np.sort(
            subset.base_side_min_dim_fraction.to_numpy(dtype=float)
        )
    for source, subset in cancer.groupby("source", dropna=False):
        pools[(str(source), "*")] = np.sort(
            subset.base_side_min_dim_fraction.to_numpy(dtype=float)
        )
    pools[("*", "*")] = np.sort(cancer.base_side_min_dim_fraction.to_numpy(dtype=float))
    return pools


def resolve_size_pool(
    pools: dict[tuple[str, str], np.ndarray], source: str, size_group: str
) -> tuple[np.ndarray, str]:
    """Resolve the closest train-cancer size pool without using validation labels."""
    for key, level in (
        ((source, size_group), "source_size"),
        ((source, "*"), "source"),
        (("*", "*"), "global"),
    ):
        if key in pools:
            return pools[key], level
    raise RuntimeError("无法解析train癌框尺寸匹配池")


def place_pixel_square(
    center_x: float, center_y: float, side_fraction: float, width: int, height: int
) -> np.ndarray:
    """Place a pixel-square crop at a normalized center and clamp it inside the image."""
    side = min(max(side_fraction * min(width, height), 1.0), width, height)
    cx, cy = center_x * width, center_y * height
    cx = min(max(cx, side / 2.0), width - side / 2.0)
    cy = min(max(cy, side / 2.0), height - side / 2.0)
    return np.array([
        (cx - side / 2.0) / width,
        (cy - side / 2.0) / height,
        (cx + side / 2.0) / width,
        (cy + side / 2.0) / height,
    ])


def build_size_matched_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve cancer crops and size-match non-cancer crops around target centers."""
    frame = add_pixel_side_feature(frame)
    pools = build_cancer_size_pools(frame)
    records = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        for column in (
            "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
            "base_center_x", "base_center_y", "base_width", "base_height",
            "base_area", "base_aspect_ratio", "base_side_min_dim_fraction",
        ):
            record[f"original_{column}"] = record[column]

        if int(row.label) == 1:
            box = np.array([
                row.base_crop_x1, row.base_crop_y1, row.base_crop_x2, row.base_crop_y2
            ])
            level, target_quantile = "unchanged_cancer_gt", np.nan
        else:
            pool, level = resolve_size_pool(pools, str(row.source), str(row.size_group))
            target_quantile = hash_uniform(str(row.sha256))
            side_fraction = quantile_from_sorted(pool, target_quantile)
            box = place_pixel_square(
                float(row.base_center_x), float(row.base_center_y), side_fraction,
                int(row.width), int(row.height),
            )

        base_width, base_height = float(box[2] - box[0]), float(box[3] - box[1])
        record.update({
            "base_crop_x1": float(box[0]), "base_crop_y1": float(box[1]),
            "base_crop_x2": float(box[2]), "base_crop_y2": float(box[3]),
            "base_center_x": float((box[0] + box[2]) / 2.0),
            "base_center_y": float((box[1] + box[3]) / 2.0),
            "base_width": base_width, "base_height": base_height,
            "base_area": base_width * base_height,
            "base_aspect_ratio": base_width / base_height,
            "base_side_min_dim_fraction": base_width * int(row.width) / min(
                int(row.width), int(row.height)
            ),
            "size_match_pool_level": level,
            "size_match_quantile": target_quantile,
            "roi_geometry_variant": "size_matched_sensitivity",
        })
        records.append(record)

    output = pd.DataFrame(records)
    boxes = output[[
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
    ]].to_numpy(dtype=float)
    if not np.all(np.isfinite(boxes)) or not np.all((boxes >= 0.0) & (boxes <= 1.0)):
        raise ValueError("尺寸匹配crop box存在非有限值或越界")
    if not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
        raise ValueError("尺寸匹配crop box宽高非正")
    return output


def geometry_ablation_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit each geometry feature group on train and evaluate it once on validation."""
    train = frame.loc[frame.split.eq("train")].copy()
    val = frame.loc[frame.split.eq("val")].copy()
    predictions = val[["relative_path", "patient_id", "label", "roi_source"]].copy()
    metrics = {}
    for name, features in GEOMETRY_GROUPS.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=42
            ),
        )
        model.fit(train[features], train.label)
        probability_column = f"probability_{name}"
        predictions[probability_column] = model.predict_proba(val[features])[:, 1]
        val[probability_column] = predictions[probability_column].to_numpy()
        metrics[name] = {
            "features": features,
            "val_image_auc": float(roc_auc_score(val.label, val[probability_column])),
            "val_patient_auc": patient_mean_auc(val, probability_column),
        }
    return metrics, predictions


def run_self_test() -> None:
    """Check stable hashing, quantiles and square placement."""
    assert hash_uniform("sample") == hash_uniform("sample")
    assert quantile_from_sorted(np.array([0.0, 1.0]), 0.25) == 0.25
    box = place_pixel_square(0.0, 1.0, 0.5, 800, 600)
    pixels = box * np.array([800, 600, 800, 600])
    assert np.all((box >= 0.0) & (box <= 1.0))
    assert abs((pixels[2] - pixels[0]) - (pixels[3] - pixels[1])) < 1e-9
    print("MG0b尺寸匹配self-test通过: 哈希、分位数和像素方框有效")


def main() -> None:
    """Create a drop-in sensitivity manifest, geometry ablations and QC sheets."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    input_path = args.input_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"尺寸匹配敏感性输出已存在: {output_dir}")
    frame = pd.read_csv(
        input_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    if len(frame) != 2847 or not set(frame.split).issubset({"train", "val"}):
        raise ValueError("输入不是冻结MG0b train/val教师ROI清单")
    original_with_side = add_pixel_side_feature(frame)
    original_geometry_metrics, original_predictions = geometry_ablation_audit(
        original_with_side
    )
    matched = build_size_matched_manifest(frame)
    cancer = matched.label.eq(1)
    original = matched.loc[cancer, [
        "original_base_crop_x1", "original_base_crop_y1",
        "original_base_crop_x2", "original_base_crop_y2",
    ]].to_numpy()
    current = matched.loc[cancer, [
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
    ]].to_numpy()
    if not np.array_equal(original, current):
        raise ValueError("尺寸匹配错误修改了癌图GT裁图")

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "mage_teacher_roi_manifest_size_matched_sensitivity.csv"
    matched.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    geometry_metrics, predictions = geometry_ablation_audit(matched)
    predictions.to_csv(
        output_dir / "geometry_ablation_val_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    original_predictions.to_csv(
        output_dir / "geometry_ablation_original_val_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    qc_paths = stratified_qc(matched, output_dir)
    summary = (
        matched.groupby(["split", "label", "size_match_pool_level"], dropna=False)
        .agg(
            images=("relative_path", "size"), patients=("patient_id", "nunique"),
            side_p10=("base_side_min_dim_fraction", lambda x: float(x.quantile(0.10))),
            side_p50=("base_side_min_dim_fraction", "median"),
            side_p90=("base_side_min_dim_fraction", lambda x: float(x.quantile(0.90))),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "size_match_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "stage": "MG0b_teacher_roi_size_matched_sensitivity",
        "created_on": "2026-08-17",
        "input_manifest": str(input_path),
        "input_manifest_sha256": file_sha256(input_path),
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": file_sha256(manifest_path),
        "rule": {
            "cancer": "unchanged locked GT-derived base crop",
            "noncancer_center": (
                "locked detector/fallback center is the placement target; large crops "
                "near an edge shift only enough to remain in-frame"
            ),
            "noncancer_side": (
                "deterministic SHA quantile from train-cancer source+size, "
                "then source-only, then global pool"
            ),
        },
        "original_geometry_ablation": original_geometry_metrics,
        "size_matched_geometry_ablation": geometry_metrics,
        "qc_sheets": qc_paths,
        "test_rows_read": 0,
        "internal_test_read": False,
        "external_read": False,
        "status": "sensitivity_candidate_not_primary_manifest",
    }
    noncancer = matched.loc[matched.label.eq(0)]
    center_shift = np.sqrt(
        (noncancer.base_center_x - noncancer.original_base_center_x) ** 2
        + (noncancer.base_center_y - noncancer.original_base_center_y) ** 2
    )
    shifted = center_shift.gt(1e-12)
    config["center_clamping"] = {
        "noncancer_images": int(len(noncancer)),
        "shifted_images": int(shifted.sum()),
        "shifted_fraction": float(shifted.mean()),
        "median_shift_among_shifted": float(center_shift.loc[shifted].median()),
        "maximum_shift": float(center_shift.max()),
    }
    (output_dir / "mg0b_size_matched_sensitivity_audit.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    for name, metric in geometry_metrics.items():
        original_metric = original_geometry_metrics[name]
        print(
            f"{name}: original patient AUC={original_metric['val_patient_auc']:.4f} -> "
            f"matched patient AUC={metric['val_patient_auc']:.4f}"
        )
    print(f"尺寸匹配敏感性清单: {manifest_path}")
    print("癌图裁图未改；未读取internal test或external。")


if __name__ == "__main__":
    main()
