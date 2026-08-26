#!/usr/bin/env python3
"""CD1 External逐病灶四象限与跨seed稳定性测试。"""

from __future__ import annotations

import math

import pandas as pd

from evaluate_cd1_external import (
    build_seed_quadrants,
    gt_candidate_evidence,
    image_trigger_summary,
    summarize_rescue_stability,
)


def prediction(image_key: str, confidence: float, x1: float = 0.1) -> dict:
    """生成单个合成候选框。"""
    return {
        "image_key": image_key, "prediction_index": 0, "confidence": confidence,
        "x1": x1, "y1": 0.1, "x2": x1 + 0.3, "y2": 0.4,
    }


def main() -> None:
    """验证触发率、最佳候选、四象限和3/3 rescue统计。"""
    images = pd.DataFrame({
        "image_key": ["a", "b", "n"],
        "patient_id": ["p1", "p2", "p3"],
        "label": [1, 1, 0],
        "source": ["x", "x", "x"],
        "original_width": [640, 640, 640],
        "original_height": [480, 480, 480],
    })
    ground_truth = pd.DataFrame({
        "image_key": ["a", "b"], "gt_index": [0, 0],
        "x1": [0.1, 0.1], "y1": [0.1, 0.1],
        "x2": [0.4, 0.4], "y2": [0.4, 0.4],
        "bbox_area_fraction": [0.09, 0.09],
    })
    rgb = pd.DataFrame([prediction("a", 0.9), prediction("n", 0.8)])
    gray = pd.DataFrame([prediction("b", 0.7), prediction("n", 0.2)])

    trigger = image_trigger_summary(images, rgb, 0.5)
    assert math.isclose(trigger["cancer_images"], 0.5)
    assert math.isclose(trigger["noncancer_images"], 1.0)
    evidence = gt_candidate_evidence(ground_truth, rgb)
    assert evidence[("a", 0)] == (1.0, 0.9)
    assert evidence[("b", 0)] == (0.0, 0.0)

    tables = []
    for seed in (42, 202, 503):
        table = build_seed_quadrants(
            seed, images, ground_truth, rgb, gray,
            0.5, 0.5, (0.1, 0.2), "external_primary",
        )
        assert table.set_index("image_key").loc["a", "quadrant"] == "rgb_only"
        assert table.set_index("image_key").loc["b", "quadrant"] == "gray_only_rescue"
        tables.append(table)
    stable, summary = summarize_rescue_stability(pd.concat(tables, ignore_index=True))
    assert len(stable) == 2
    assert summary["3/3"] == 1
    assert summary["0/3"] == 1
    assert summary["stable_3_of_3"] == 1
    print("CD1 External evaluator tests: 14/14 passed")


if __name__ == "__main__":
    main()
