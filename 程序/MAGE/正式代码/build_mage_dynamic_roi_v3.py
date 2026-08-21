#!/usr/bin/env python3
"""构建MG0b独立轴动态教师ROI v3候选。

V2强制所有裁图遵循固定形状规则，长条来源框会在短轴带入大量无关背景。
V3分别扩展宽和高：每个方向尽量保留20%边距，并以0.85为上限，但不缩小
已经更大的来源框。下游教师数据集再将矩形ROI统一调整到固定输入尺寸。

本脚本输出独立轴主清单和尺寸匹配敏感性清单，只读取冻结的MG0b训练/验证
队列，内部测试集和外部数据保持锁定。
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

from build_mage_size_matched_roi_sensitivity import hash_uniform
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
    "数据整理记录/MAGE/MG0b_独立轴动态扩边ROI_v3_20260817"
)
MARGIN = 0.20
AXIS_CAP = 0.85
GEOMETRY_GROUPS = {
    "center_only": ["base_center_x", "base_center_y"],
    "size_shape_only": [
        "base_width", "base_height", "base_area",
        "base_aspect_ratio", "base_pixel_aspect_ratio",
    ],
    "center_plus_size_shape": [
        "base_center_x", "base_center_y", "base_width", "base_height",
        "base_area", "base_aspect_ratio", "base_pixel_aspect_ratio",
    ],
}


def parse_args() -> argparse.Namespace:
    """解析冻结输入清单、输出目录和自测模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def dynamic_target_extent(
    source_extent: float, margin: float = MARGIN, axis_cap: float = AXIS_CAP
) -> float:
    """扩展一个归一化方向，同时不缩小来源范围。"""
    if not np.isfinite(source_extent) or not 0.0 < source_extent <= 1.0:
        raise ValueError(f"无效源框轴长: {source_extent}")
    expanded = source_extent * (1.0 + 2.0 * margin)
    return float(min(expanded, max(source_extent, axis_cap), 1.0))


def place_rectangular_crop(
    source_box: np.ndarray, target_width: float, target_height: float
) -> np.ndarray:
    """在图像内放置完整包含来源框的矩形裁剪框。"""
    source_box = np.asarray(source_box, dtype=float)
    x1, y1, x2, y2 = source_box
    source_width, source_height = x2 - x1, y2 - y1
    if not np.all(np.isfinite(source_box)) or source_width <= 0 or source_height <= 0:
        raise ValueError(f"无效源框: {source_box}")
    if target_width + 1e-12 < source_width or target_height + 1e-12 < source_height:
        raise ValueError("目标裁图无法完整包含源框")
    if target_width > 1.0 or target_height > 1.0:
        raise ValueError("目标裁图越界")

    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = min(max(center_x - target_width / 2.0, 0.0), 1.0 - target_width)
    top = min(max(center_y - target_height / 2.0, 0.0), 1.0 - target_height)
    crop = np.array([left, top, left + target_width, top + target_height])
    if crop[0] > x1 + 1e-10 or crop[1] > y1 + 1e-10:
        raise RuntimeError("左/上边界未包含源框")
    if crop[2] < x2 - 1e-10 or crop[3] < y2 - 1e-10:
        raise RuntimeError("右/下边界未包含源框")
    return crop


def source_coverage(frame: pd.DataFrame) -> pd.Series:
    """计算候选裁剪框对每个来源框的覆盖比例。"""
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


def update_crop_fields(record: dict, crop: np.ndarray, variant: str) -> None:
    """将裁剪几何和相对来源框的紧致度写入记录。"""
    width, height = float(crop[2] - crop[0]), float(crop[3] - crop[1])
    source_width = float(record["source_box_x2"] - record["source_box_x1"])
    source_height = float(record["source_box_y2"] - record["source_box_y1"])
    record.update({
        "base_crop_x1": float(crop[0]), "base_crop_y1": float(crop[1]),
        "base_crop_x2": float(crop[2]), "base_crop_y2": float(crop[3]),
        "base_center_x": float((crop[0] + crop[2]) / 2.0),
        "base_center_y": float((crop[1] + crop[3]) / 2.0),
        "base_width": width, "base_height": height,
        "base_area": width * height,
        "base_aspect_ratio": width / height,
        "base_pixel_aspect_ratio": width * float(record["width"]) / (
            height * float(record["height"])
        ),
        "base_scale": max(width, height),
        "source_width": source_width,
        "source_height": source_height,
        "source_area": source_width * source_height,
        "crop_to_source_area_ratio": width * height / (source_width * source_height),
        "roi_geometry_variant": variant,
    })


def preserve_legacy_fields(record: dict) -> None:
    """保留原像素正方形裁剪字段，用于血缘追踪和对照。"""
    for column in (
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2",
        "base_center_x", "base_center_y", "base_width", "base_height",
        "base_area", "base_aspect_ratio",
    ):
        record[f"legacy_{column}"] = record[column]


def validate_manifest(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """校验覆盖率、队列唯一性和锁定数据边界。"""
    boxes = frame[[
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
    ]].to_numpy(dtype=float)
    if len(frame) != 2847 or frame.relative_path.nunique() != 2847:
        raise ValueError(f"{name}未唯一覆盖2847张train/val图像")
    if not np.all(np.isfinite(boxes)) or not np.all((boxes >= 0.0) & (boxes <= 1.0)):
        raise ValueError(f"{name}存在非有限值或越界")
    if not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
        raise ValueError(f"{name}存在非正裁图尺寸")
    coverage = source_coverage(frame)
    if float(coverage.min()) < 1.0 - 1e-8:
        raise ValueError(f"{name}未完整覆盖源框，最小覆盖率={coverage.min()}")
    if frame[["internal_test_read", "external_read"]].astype(bool).any().any():
        raise ValueError(f"{name}混入锁定数据")
    return frame


def build_independent_axis_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """分别扩展宽和高，构建v3主裁剪清单。"""
    records = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        preserve_legacy_fields(record)
        source_box = np.array([
            row.source_box_x1, row.source_box_y1, row.source_box_x2, row.source_box_y2
        ], dtype=float)
        source_width = float(source_box[2] - source_box[0])
        source_height = float(source_box[3] - source_box[1])
        target_width = dynamic_target_extent(source_width)
        target_height = dynamic_target_extent(source_height)
        crop = place_rectangular_crop(source_box, target_width, target_height)
        update_crop_fields(record, crop, "independent_axis_dynamic_margin_v3")
        record.update({
            "dynamic_target_width": target_width,
            "dynamic_target_height": target_height,
            "dynamic_effective_margin_x": (target_width / source_width - 1.0) / 2.0,
            "dynamic_effective_margin_y": (target_height / source_height - 1.0) / 2.0,
            "dynamic_cap_active_x": bool(target_width + 1e-12 < source_width * 1.4),
            "dynamic_cap_active_y": bool(target_height + 1e-12 < source_height * 1.4),
            "size_match_pool_level": "not_applicable",
            "size_match_quantile": np.nan,
            "size_match_source_floor_x": False,
            "size_match_source_floor_y": False,
        })
        records.append(record)
    return validate_manifest(pd.DataFrame(records), "独立轴动态扩边")


def build_dimension_pools(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """只使用训练癌图裁剪框建立宽高配对供体池。"""
    cancer = frame.loc[frame.split.eq("train") & frame.label.eq(1)]
    pools: dict[tuple[str, str], np.ndarray] = {}

    def paired_dimensions(subset: pd.DataFrame) -> np.ndarray:
        """按稳定顺序整理同组癌图尺寸，供非癌ROI尺寸匹配使用。"""
        values = subset[["base_width", "base_height"]].to_numpy(dtype=float)
        order = np.lexsort((values[:, 0] / values[:, 1], values[:, 0] * values[:, 1]))
        return values[order]

    for keys, subset in cancer.groupby(["source", "size_group"], dropna=False):
        pools[(str(keys[0]), str(keys[1]))] = paired_dimensions(subset)
    for source, subset in cancer.groupby("source", dropna=False):
        pools[(str(source), "*")] = paired_dimensions(subset)
    pools[("*", "*")] = paired_dimensions(cancer)
    return pools


def resolve_dimension_pool(
    pools: dict[tuple[str, str], np.ndarray], source: str, size_group: str
) -> tuple[np.ndarray, str]:
    """依次按来源加尺寸、仅来源、全局训练癌图选择供体池。"""
    for key, level in (
        ((source, size_group), "source_size"),
        ((source, "*"), "source"),
        (("*", "*"), "global"),
    ):
        if key in pools:
            return pools[key], level
    raise RuntimeError("无法解析独立轴尺寸匹配池")


def choose_paired_dimensions(pool: np.ndarray, quantile: float) -> tuple[float, float]:
    """从供体池中确定性选择一组真实出现过的宽高。"""
    if not len(pool):
        raise ValueError("尺寸匹配池为空")
    index = min(int(np.floor(quantile * len(pool))), len(pool) - 1)
    return float(pool[index, 0]), float(pool[index, 1])


def build_size_matched_manifest(dynamic: pd.DataFrame) -> pd.DataFrame:
    """为非癌图匹配宽高，同时确保完整保留来源框。"""
    pools = build_dimension_pools(dynamic)
    records = []
    for row in dynamic.itertuples(index=False):
        record = row._asdict()
        if int(row.label) == 1:
            record["size_match_pool_level"] = "unchanged_cancer_dynamic"
            record["roi_geometry_variant"] = "independent_axis_size_matched_v3"
            records.append(record)
            continue

        pool, level = resolve_dimension_pool(
            pools, str(row.source), str(row.size_group)
        )
        quantile = hash_uniform(str(row.sha256))
        donor_width, donor_height = choose_paired_dimensions(pool, quantile)
        target_width = max(float(row.source_width), donor_width)
        target_height = max(float(row.source_height), donor_height)
        source_box = np.array([
            row.source_box_x1, row.source_box_y1, row.source_box_x2, row.source_box_y2
        ], dtype=float)
        crop = place_rectangular_crop(source_box, target_width, target_height)
        update_crop_fields(record, crop, "independent_axis_size_matched_v3")
        record.update({
            "size_match_pool_level": level,
            "size_match_quantile": quantile,
            "size_match_donor_width": donor_width,
            "size_match_donor_height": donor_height,
            "size_match_source_floor_x": bool(target_width > donor_width + 1e-12),
            "size_match_source_floor_y": bool(target_height > donor_height + 1e-12),
        })
        records.append(record)
    return validate_manifest(pd.DataFrame(records), "独立轴动态扩边+尺寸匹配")


def geometry_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """在训练集拟合纯几何对照，并在验证集评估一次。"""
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
    """按数据划分和标签汇总裁剪尺寸、紧致度和覆盖率。"""
    audit = frame.copy()
    audit["source_coverage"] = source_coverage(audit)
    audit["full_frame"] = audit.base_width.ge(1.0 - 1e-12) & audit.base_height.ge(
        1.0 - 1e-12
    )
    return (
        audit.groupby(["split", "label"], dropna=False)
        .agg(
            images=("relative_path", "size"), patients=("patient_id", "nunique"),
            width_p50=("base_width", "median"),
            height_p50=("base_height", "median"),
            area_p50=("base_area", "median"),
            crop_source_ratio_p50=("crop_to_source_area_ratio", "median"),
            crop_source_ratio_p90=(
                "crop_to_source_area_ratio", lambda x: float(x.quantile(0.90))
            ),
            full_frame=("full_frame", "sum"),
            minimum_source_coverage=("source_coverage", "min"),
        )
        .reset_index()
    )


def save_candidate(
    frame: pd.DataFrame, output_dir: Path, manifest_name: str
) -> tuple[Path, dict, pd.DataFrame, list[str]]:
    """保存一套候选清单、几何审计、统计汇总和质控图。"""
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
    """自测独立缩放、长条框紧致度和来源框完整包含。"""
    expected = {0.20: 0.28, 0.50: 0.70, 0.70: 0.85, 0.80: 0.85, 0.90: 0.90}
    for source_extent, target_extent in expected.items():
        assert abs(dynamic_target_extent(source_extent) - target_extent) < 1e-12

    long_box = np.array([0.10, 0.05, 0.25, 0.95])
    target_width = dynamic_target_extent(0.15)
    target_height = dynamic_target_extent(0.90)
    crop = place_rectangular_crop(long_box, target_width, target_height)
    assert abs((crop[2] - crop[0]) - 0.21) < 1e-12
    assert abs((crop[3] - crop[1]) - 0.90) < 1e-12
    assert crop[0] <= long_box[0] and crop[2] >= long_box[2]
    assert crop[1] <= long_box[1] and crop[3] >= long_box[3]
    print("MG0b独立轴动态扩边v3 self-test通过: 分轴扩边、长框紧致性与覆盖有效")


def main() -> None:
    """构建两套v3候选，并保存血缘、审计和可视化质控。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    input_path = args.input_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"MG0b独立轴动态扩边v3输出已存在: {output_dir}")
    frame = pd.read_csv(
        input_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    if not set(frame.split).issubset({"train", "val"}):
        raise ValueError("输入清单混入test数据")

    dynamic = build_independent_axis_manifest(frame)
    matched = build_size_matched_manifest(dynamic)
    output_dir.mkdir(parents=True)
    dynamic_path, dynamic_metrics, dynamic_summary, dynamic_qc = save_candidate(
        dynamic,
        output_dir / "independent_axis_dynamic",
        "mage_teacher_roi_manifest_independent_axis_dynamic_v3.csv",
    )
    matched_path, matched_metrics, matched_summary, matched_qc = save_candidate(
        matched,
        output_dir / "independent_axis_dynamic_size_matched",
        "mage_teacher_roi_manifest_independent_axis_dynamic_size_matched_v3.csv",
    )

    cancer = dynamic.label.eq(1)
    crop_columns = [
        "base_crop_x1", "base_crop_y1", "base_crop_x2", "base_crop_y2"
    ]
    if not np.array_equal(
        dynamic.loc[cancer, crop_columns].to_numpy(),
        matched.loc[cancer, crop_columns].to_numpy(),
    ):
        raise ValueError("尺寸匹配分支错误修改了癌图独立轴裁图")

    config = {
        "stage": "MG0b_independent_axis_dynamic_roi_v3",
        "created_on": "2026-08-17",
        "input_manifest": str(input_path),
        "input_manifest_sha256": file_sha256(input_path),
        "crop_protocol": {
            "shape": "rectangular; normalized width and height expanded independently",
            "margin_per_side_each_axis": MARGIN,
            "normalized_axis_cap": AXIS_CAP,
            "target_extent_each_axis": (
                "min(source_extent*1.4, max(source_extent, 0.85), 1.0)"
            ),
            "source_containment": "mandatory; no padding and no source-box shrink",
            "teacher_resize": (
                "rectangular crop resized directly to 224x224; deliberate ROIAlign-style "
                "fixed-size representation"
            ),
        },
        "independent_axis_dynamic": {
            "manifest": str(dynamic_path),
            "manifest_sha256": file_sha256(dynamic_path),
            "geometry_audit": dynamic_metrics,
            "summary": dynamic_summary.to_dict(orient="records"),
            "qc_sheets": dynamic_qc,
        },
        "independent_axis_dynamic_size_matched": {
            "manifest": str(matched_path),
            "manifest_sha256": file_sha256(matched_path),
            "geometry_audit": matched_metrics,
            "summary": matched_summary.to_dict(orient="records"),
            "qc_sheets": matched_qc,
            "noncancer_source_floor_x": int(
                matched.loc[matched.label.eq(0), "size_match_source_floor_x"].sum()
            ),
            "noncancer_source_floor_y": int(
                matched.loc[matched.label.eq(0), "size_match_source_floor_y"].sum()
            ),
        },
        "test_rows_read": 0,
        "internal_test_read": False,
        "external_read": False,
        "status": "MG1_candidates_pending_manual_qc",
    }
    audit_path = output_dir / "mg0b_independent_axis_dynamic_roi_v3_audit.json"
    audit_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n独立轴动态扩边v3:")
    print(dynamic_summary.to_string(index=False))
    print("\n独立轴动态扩边+尺寸匹配v3:")
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
