#!/usr/bin/env python3
"""Build and audit MG0b dynamic-margin teacher ROI v2 candidates.

Unlike the legacy pixel-square crop, v2 uses a crop with the same pixel aspect
ratio as the source image. In normalized coordinates its width and height are
equal, so every source box can remain fully enclosed without padding. Small
boxes retain the 20% per-side context margin; larger boxes are capped at a
target scale of 0.85 unless the source box itself is larger. A second manifest
size-matches non-cancer crops to the train-cancer v2 scale distribution.

Both outputs are MG1 candidates. They do not overwrite legacy manifests or read
internal test/external data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_mage_size_matched_roi_sensitivity import hash_uniform, quantile_from_sorted
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
    "数据整理记录/MAGE/MG0b_动态扩边ROI_v2_20260817"
)
MARGIN = 0.20
SCALE_CAP = 0.85
GEOMETRY_GROUPS = {
    "center_only": ["base_center_x", "base_center_y"],
    "size_only": ["base_width", "base_height", "base_area"],
    "center_plus_size": [
        "base_center_x", "base_center_y", "base_width", "base_height", "base_area"
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse the locked source manifest, output directory and self-test mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def dynamic_target_scale(
    source_scale: float, margin: float = MARGIN, scale_cap: float = SCALE_CAP
) -> float:
    """Return an expanded normalized scale without shrinking the source box."""
    if not np.isfinite(source_scale) or not 0.0 < source_scale <= 1.0:
        raise ValueError(f"无效源框尺度: {source_scale}")
    expanded = source_scale * (1.0 + 2.0 * margin)
    return float(min(expanded, max(source_scale, scale_cap), 1.0))


def place_same_aspect_crop(source_box: np.ndarray, target_scale: float) -> np.ndarray:
    """Place a normalized-square crop that contains the source box and stays in-frame."""
    source_box = np.asarray(source_box, dtype=float)
    x1, y1, x2, y2 = source_box
    source_width, source_height = x2 - x1, y2 - y1
    if not np.all(np.isfinite(source_box)) or source_width <= 0 or source_height <= 0:
        raise ValueError(f"无效源框: {source_box}")
    if target_scale + 1e-12 < max(source_width, source_height) or target_scale > 1.0:
        raise ValueError("目标裁图无法完整包含源框")
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(center_x - target_scale / 2.0, 0.0), 1.0 - target_scale)
    top = min(max(center_y - target_scale / 2.0, 0.0), 1.0 - target_scale)
    crop = np.array([left, top, left + target_scale, top + target_scale])
    if crop[0] > x1 + 1e-10 or crop[1] > y1 + 1e-10:
        raise RuntimeError("左/上边界未包含源框")
    if crop[2] < x2 - 1e-10 or crop[3] < y2 - 1e-10:
        raise RuntimeError("右/下边界未包含源框")
    return crop


def update_base_crop(record: dict, crop: np.ndarray, variant: str) -> None:
    """Write one v2 crop and its reproducible geometry fields into a record."""
    width, height = float(crop[2] - crop[0]), float(crop[3] - crop[1])
    record.update({
        "base_crop_x1": float(crop[0]), "base_crop_y1": float(crop[1]),
        "base_crop_x2": float(crop[2]), "base_crop_y2": float(crop[3]),
        "base_center_x": float((crop[0] + crop[2]) / 2.0),
        "base_center_y": float((crop[1] + crop[3]) / 2.0),
        "base_width": width, "base_height": height,
        "base_area": width * height,
        "base_aspect_ratio": width / height,
        "base_pixel_aspect_ratio": float(record["width"] / record["height"]),
        "base_scale": width,
        "roi_geometry_variant": variant,
    })


def build_dynamic_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy base crops with source-containing dynamic v2 crops."""
    records = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        for column in (
            "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
            "base_center_x", "base_center_y", "base_width", "base_height",
            "base_area", "base_aspect_ratio",
        ):
            record[f"legacy_{column}"] = record[column]
        source_box = np.array([
            row.source_box_x1, row.source_box_y1, row.source_box_x2, row.source_box_y2
        ])
        source_scale = float(max(source_box[2] - source_box[0], source_box[3] - source_box[1]))
        target_scale = dynamic_target_scale(source_scale)
        crop = place_same_aspect_crop(source_box, target_scale)
        update_base_crop(record, crop, "dynamic_margin_same_image_aspect_v2")
        record.update({
            "source_scale": source_scale,
            "dynamic_target_scale": target_scale,
            "dynamic_effective_margin_per_side": (target_scale / source_scale - 1.0) / 2.0,
            "dynamic_cap_active": bool(target_scale + 1e-12 < source_scale * 1.4),
            "size_match_pool_level": "not_applicable",
            "size_match_quantile": np.nan,
            "size_match_source_floor_active": False,
        })
        records.append(record)
    return validate_manifest(pd.DataFrame(records), "动态扩边")


def build_scale_pools(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Fit dynamic crop-scale pools using train cancer images only."""
    cancer = frame.loc[frame.split.eq("train") & frame.label.eq(1)]
    pools: dict[tuple[str, str], np.ndarray] = {}
    for keys, subset in cancer.groupby(["source", "size_group"], dropna=False):
        pools[(str(keys[0]), str(keys[1]))] = np.sort(subset.base_scale.to_numpy(dtype=float))
    for source, subset in cancer.groupby("source", dropna=False):
        pools[(str(source), "*")] = np.sort(subset.base_scale.to_numpy(dtype=float))
    pools[("*", "*")] = np.sort(cancer.base_scale.to_numpy(dtype=float))
    return pools


def resolve_scale_pool(
    pools: dict[tuple[str, str], np.ndarray], source: str, size_group: str
) -> tuple[np.ndarray, str]:
    """Resolve source-size, source-only, then global train-cancer scale pool."""
    for key, level in (
        ((source, size_group), "source_size"),
        ((source, "*"), "source"),
        (("*", "*"), "global"),
    ):
        if key in pools:
            return pools[key], level
    raise RuntimeError("无法解析动态尺寸匹配池")


def build_dynamic_size_matched_manifest(dynamic: pd.DataFrame) -> pd.DataFrame:
    """Size-match non-cancer v2 crops while retaining source-box containment."""
    pools = build_scale_pools(dynamic)
    records = []
    for row in dynamic.itertuples(index=False):
        record = row._asdict()
        if int(row.label) == 1:
            record["size_match_pool_level"] = "unchanged_cancer_dynamic"
            record["roi_geometry_variant"] = "dynamic_margin_size_matched_v2"
            records.append(record)
            continue
        pool, level = resolve_scale_pool(pools, str(row.source), str(row.size_group))
        quantile = hash_uniform(str(row.sha256))
        sampled_scale = quantile_from_sorted(pool, quantile)
        target_scale = max(float(row.source_scale), sampled_scale)
        source_floor_active = bool(target_scale > sampled_scale + 1e-12)
        source_box = np.array([
            row.source_box_x1, row.source_box_y1, row.source_box_x2, row.source_box_y2
        ])
        crop = place_same_aspect_crop(source_box, target_scale)
        update_base_crop(record, crop, "dynamic_margin_size_matched_v2")
        record.update({
            "size_match_pool_level": level,
            "size_match_quantile": quantile,
            "size_match_sampled_scale": sampled_scale,
            "size_match_source_floor_active": source_floor_active,
        })
        records.append(record)
    return validate_manifest(pd.DataFrame(records), "动态扩边+尺寸匹配")


def source_coverage(frame: pd.DataFrame) -> pd.Series:
    """Return the fraction of each source box covered by its base crop."""
    intersection_width = np.maximum(
        0.0,
        np.minimum(frame.source_box_x2, frame.base_crop_x2)
        - np.maximum(frame.source_box_x1, frame.base_crop_x1),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(frame.source_box_y2, frame.base_crop_y2)
        - np.maximum(frame.source_box_y1, frame.base_crop_y1),
    )
    source_area = (
        (frame.source_box_x2 - frame.source_box_x1)
        * (frame.source_box_y2 - frame.source_box_y1)
    )
    return intersection_width * intersection_height / source_area


def validate_manifest(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate one candidate's coverage, bounds and locked-data boundary."""
    boxes = frame[[
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
    ]].to_numpy(dtype=float)
    if len(frame) != 2847 or frame.relative_path.nunique() != 2847:
        raise ValueError(f"{name}未唯一覆盖2847张train/val图像")
    if not np.all(np.isfinite(boxes)) or not np.all((boxes >= 0.0) & (boxes <= 1.0)):
        raise ValueError(f"{name}存在非有限值或越界")
    coverage = source_coverage(frame)
    if float(coverage.min()) < 1.0 - 1e-8:
        raise ValueError(f"{name}未完整覆盖源框，最小覆盖率={coverage.min()}")
    if frame[["internal_test_read", "external_read"]].astype(bool).any().any():
        raise ValueError(f"{name}混入锁定数据")
    return frame


def geometry_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit geometry-only controls on train and evaluate image/patient AUC on val."""
    train = frame.loc[frame.split.eq("train")]
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


def candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize crop scale, containment and cap behavior by split and label."""
    audit = frame.copy()
    audit["source_coverage"] = source_coverage(audit)
    return (
        audit.groupby(["split", "label"], dropna=False)
        .agg(
            images=("relative_path", "size"), patients=("patient_id", "nunique"),
            scale_p50=("base_scale", "median"),
            scale_p75=("base_scale", lambda x: float(x.quantile(0.75))),
            scale_p90=("base_scale", lambda x: float(x.quantile(0.90))),
            at_or_above_cap=("base_scale", lambda x: int((x >= SCALE_CAP - 1e-12).sum())),
            full_frame=("base_scale", lambda x: int((x >= 1.0 - 1e-12).sum())),
            minimum_source_coverage=("source_coverage", "min"),
        )
        .reset_index()
    )


def save_candidate(
    frame: pd.DataFrame, output_dir: Path, manifest_name: str
) -> tuple[Path, dict, pd.DataFrame, list[str]]:
    """Save one candidate manifest, geometry predictions, summary and QC sheets."""
    output_dir.mkdir()
    manifest_path = output_dir / manifest_name
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    metrics, predictions = geometry_audit(frame)
    predictions.to_csv(
        output_dir / "geometry_val_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary = candidate_summary(frame)
    summary.to_csv(output_dir / "crop_summary.csv", index=False, encoding="utf-8-sig")
    qc_paths = stratified_qc(frame, output_dir)
    return manifest_path, metrics, summary, qc_paths


def run_self_test() -> None:
    """Exercise dynamic scaling, edge clamping and source containment."""
    expected = {0.20: 0.28, 0.50: 0.70, 0.70: 0.85, 0.80: 0.85, 0.90: 0.90}
    for source_scale, target_scale in expected.items():
        assert abs(dynamic_target_scale(source_scale) - target_scale) < 1e-12
    source = np.array([0.0, 0.30, 0.95, 0.55])
    crop = place_same_aspect_crop(source, dynamic_target_scale(0.95))
    assert np.allclose(crop, [0.0, 0.0, 0.95, 0.95])
    assert crop[0] <= source[0] and crop[2] >= source[2]
    print("MG0b动态扩边v2 self-test通过: 尺度、边缘钳制与源框覆盖有效")


def main() -> None:
    """Build both v2 candidates and persist lineage, audits and QC products."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    input_path = args.input_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"MG0b动态扩边v2输出已存在: {output_dir}")
    frame = pd.read_csv(
        input_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    if not set(frame.split).issubset({"train", "val"}):
        raise ValueError("输入清单混入test数据")

    dynamic = build_dynamic_manifest(frame)
    matched = build_dynamic_size_matched_manifest(dynamic)
    output_dir.mkdir(parents=True)
    dynamic_path, dynamic_metrics, dynamic_summary, dynamic_qc = save_candidate(
        dynamic, output_dir / "dynamic_margin",
        "mage_teacher_roi_manifest_dynamic_margin_v2.csv",
    )
    matched_path, matched_metrics, matched_summary, matched_qc = save_candidate(
        matched, output_dir / "dynamic_margin_size_matched",
        "mage_teacher_roi_manifest_dynamic_margin_size_matched_v2.csv",
    )
    cancer = dynamic.label.eq(1)
    if not np.array_equal(
        dynamic.loc[cancer, [
            "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
        ]].to_numpy(),
        matched.loc[cancer, [
            "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
        ]].to_numpy(),
    ):
        raise ValueError("尺寸匹配分支错误修改了癌图动态裁图")

    config = {
        "stage": "MG0b_dynamic_margin_roi_v2",
        "created_on": "2026-08-17",
        "input_manifest": str(input_path),
        "input_manifest_sha256": file_sha256(input_path),
        "crop_protocol": {
            "pixel_aspect": "same as source image; normalized crop width equals height",
            "margin_per_side": MARGIN,
            "normalized_scale_cap": SCALE_CAP,
            "target_scale": "min(source_scale*1.4, max(source_scale, 0.85))",
            "source_scale": "max(source_box normalized width, normalized height)",
            "source_containment": "mandatory; no padding and no source-box shrink",
        },
        "dynamic_margin": {
            "manifest": str(dynamic_path),
            "manifest_sha256": file_sha256(dynamic_path),
            "geometry_audit": dynamic_metrics,
            "summary": dynamic_summary.to_dict(orient="records"),
            "qc_sheets": dynamic_qc,
        },
        "dynamic_margin_size_matched": {
            "manifest": str(matched_path),
            "manifest_sha256": file_sha256(matched_path),
            "geometry_audit": matched_metrics,
            "summary": matched_summary.to_dict(orient="records"),
            "qc_sheets": matched_qc,
            "noncancer_source_floor_count": int(
                matched.loc[matched.label.eq(0), "size_match_source_floor_active"].sum()
            ),
        },
        "test_rows_read": 0,
        "internal_test_read": False,
        "external_read": False,
        "status": "MG1_candidates_pending_manual_qc",
    }
    (output_dir / "mg0b_dynamic_roi_v2_audit.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n动态扩边v2:")
    print(dynamic_summary.to_string(index=False))
    print("\n动态扩边+尺寸匹配v2:")
    print(matched_summary.to_string(index=False))
    print("\n几何患者AUC:")
    for name in GEOMETRY_GROUPS:
        print(
            f"{name}: dynamic={dynamic_metrics[name]['val_patient_auc']:.4f}, "
            f"matched={matched_metrics[name]['val_patient_auc']:.4f}"
        )
    print(f"输出目录: {output_dir}")
    print("两套候选源框覆盖率均为100%；未读取internal test或external。")


if __name__ == "__main__":
    main()
