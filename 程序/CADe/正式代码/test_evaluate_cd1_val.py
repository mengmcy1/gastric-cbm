#!/usr/bin/env python3
"""CD1 Development Val评价入口的阈值和配对汇总测试。"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd

from evaluate_cd1_val import (
    bootstrap_interval,
    build_inference_view,
    fp_overlap_by_image_label,
    image_detection_sensitivity,
    lock_image_recall_threshold,
    validate_rgb_reproduction,
)


def main() -> None:
    """验证图像级阈值、FP分层和bootstrap区间。"""
    images = pd.DataFrame({
        "image_key": [f"p{i}" for i in range(10)] + ["n0"],
        "patient_id": [f"u{i}" for i in range(11)],
        "label": [1] * 10 + [0],
    })
    predictions = pd.DataFrame([
        {
            "image_key": key, "prediction_index": 0, "confidence": score,
            "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4,
        }
        for key, score in zip(images.image_key, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.7])
        if score > 0
    ])
    threshold = lock_image_recall_threshold(images, predictions, 0.90)
    assert math.isclose(threshold, 0.1)
    assert math.isclose(image_detection_sensitivity(images, predictions, threshold), 0.9)

    gt = pd.DataFrame({
        "image_key": [f"p{i}" for i in range(10)], "gt_index": list(range(10)),
        "x1": [0.1] * 10, "y1": [0.1] * 10,
        "x2": [0.4] * 10, "y2": [0.4] * 10,
    })
    shifted = predictions.copy()
    shifted[["x1", "y1", "x2", "y2"]] += 0.01
    overlap = fp_overlap_by_image_label(
        images, predictions, shifted, gt, 0.5, 0.5
    )
    assert overlap["negative_images"]["matched_fp"] == 1
    assert overlap["positive_images_auxiliary"]["rgb_fp"] == 0

    bootstrap = pd.DataFrame({
        "sensitivity_delta": [-0.1, 0.0, 0.1],
        "rgb_fp_per_image": [0.2, 0.3, 0.4],
        "gray_fp_per_image": [0.1, 0.2, 0.3],
    })
    interval = bootstrap_interval(bootstrap)
    assert set(interval) == {
        "sensitivity_delta", "rgb_fp_per_image", "gray_fp_per_image"
    }
    assert interval["sensitivity_delta"][0] < 0 < interval["sensitivity_delta"][1]

    summary = {
        "primary_froc_at_most_0p5_fp_per_image": {
            "lesion_sensitivity": 0.8, "fp_per_image": 0.5,
        },
        "deployment": {"lesion_sensitivity": 0.9, "fp_per_image": 0.6},
    }
    validate_rgb_reproduction(42, summary, summary)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.png"
        source.write_bytes(b"png")
        view_images = pd.DataFrame({"image_path": [str(source)]})
        view = build_inference_view(view_images, root / "view")
        assert len(list(view.iterdir())) == 1
        assert next(view.iterdir()).resolve() == source
    print("CD1 val evaluator tests: 11/11 passed")


if __name__ == "__main__":
    main()
