#!/usr/bin/env python3
"""CD0 指标的合成数据回归测试。"""

from __future__ import annotations

import math

import pandas as pd

from cade_cd0_metrics import (
    froc_curve,
    lesion_size_groups,
    match_predictions,
    summarize_detection,
)


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造两张阳性图、一张阴性图及重复候选框。"""
    images = pd.DataFrame({
        "image_key": ["a", "b", "c"],
        "patient_id": ["p1", "p2", "p3"],
        "label": [1, 1, 0],
    })
    ground_truth = pd.DataFrame({
        "image_key": ["a", "b"],
        "gt_index": [0, 0],
        "x1": [0.1, 0.5], "y1": [0.1, 0.5],
        "x2": [0.4, 0.8], "y2": [0.4, 0.8],
    })
    predictions = pd.DataFrame([
        {"image_key": "a", "prediction_index": 0, "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4},
        {"image_key": "a", "prediction_index": 1, "confidence": 0.8, "x1": 0.12, "y1": 0.12, "x2": 0.4, "y2": 0.4},
        {"image_key": "b", "prediction_index": 0, "confidence": 0.7, "x1": 0.5, "y1": 0.5, "x2": 0.8, "y2": 0.8},
        {"image_key": "c", "prediction_index": 0, "confidence": 0.6, "x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.4},
    ])
    return images, predictions, ground_truth


def main() -> None:
    """验证一对一匹配、额外框计FP、FROC和部署点口径。"""
    images, predictions, ground_truth = fixture()
    matched = match_predictions(predictions, ground_truth, 0.65, 0.30)
    assert matched.is_tp.tolist() == [True, False, True]
    assert matched.matched_gt_index.dropna().astype(int).tolist() == [0, 0]

    curve = froc_curve(images, predictions, ground_truth)
    point = curve[curve.threshold.eq(0.7)].iloc[0]
    assert point.tp_lesions == 2 and point.fp_boxes == 1
    assert math.isclose(point.lesion_sensitivity, 1.0)
    assert math.isclose(point.fp_per_image, 1 / 3)

    summary, _, deployed = summarize_detection(
        images, predictions, ground_truth, deployment_threshold=0.7
    )
    assert summary["deployment"]["lesion_sensitivity"] == 1.0
    assert math.isclose(summary["deployment"]["fp_per_image"], 1 / 3)
    assert summary["deployment"]["negative_fp_per_image"] == 0.0
    assert summary["deployment"]["iou_ge_0p5_all_lesions"] == 1.0
    assert len(deployed) == 3
    empty = match_predictions(predictions, ground_truth, 2.0, 0.30)
    assert empty.empty and {"is_tp", "matched_gt_index", "matched_iou"}.issubset(empty.columns)
    groups = lesion_size_groups(pd.Series([0.10, 0.20, 0.40]), (0.18, 0.34))
    assert groups.astype(str).tolist() == ["lesion_small", "lesion_medium", "lesion_large"]
    boundary_groups = lesion_size_groups(pd.Series([0.18, 0.34]), (0.18, 0.34))
    assert boundary_groups.astype(str).tolist() == ["lesion_small", "lesion_medium"]
    val_group = lesion_size_groups(pd.Series([0.20]), (0.18, 0.34)).astype(str).iloc[0]
    external_group = lesion_size_groups(
        pd.Series([0.01, 0.20, 0.90]), (0.18, 0.34)
    ).astype(str).iloc[1]
    assert val_group == external_group == "lesion_medium"
    print("CD0 metrics tests: 12/12 passed")


if __name__ == "__main__":
    main()
