#!/usr/bin/env python3
"""CD1 Gray数据、错误互补、二分图匹配和四态决策回归测试。"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cade_cd1_metrics import (
    classify_cd1,
    detection_errors,
    fn_complementarity,
    fp_overlap,
    maximum_cardinality_iou_pairs,
    paired_patient_bootstrap,
)
from prepare_cd1_gray_data import file_sha256, prepare_gray_dataset


def detection_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造两病灶、一个非癌图以及互补FN/FP。"""
    images = pd.DataFrame({
        "image_key": ["a", "b", "c"],
        "patient_id": ["p1", "p2", "p1"],
        "label": [1, 1, 0],
    })
    gt = pd.DataFrame({
        "image_key": ["a", "b"], "gt_index": [3, 7],
        "x1": [0.1, 0.6], "y1": [0.1, 0.6],
        "x2": [0.4, 0.9], "y2": [0.4, 0.9],
    })
    rgb = pd.DataFrame([
        {"image_key": "a", "prediction_index": 0, "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4},
        {"image_key": "c", "prediction_index": 0, "confidence": 0.8, "x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4},
    ])
    gray = pd.DataFrame([
        {"image_key": "b", "prediction_index": 0, "confidence": 0.9, "x1": 0.6, "y1": 0.6, "x2": 0.9, "y2": 0.9},
        {"image_key": "c", "prediction_index": 0, "confidence": 0.8, "x1": 0.12, "y1": 0.12, "x2": 0.42, "y2": 0.42},
    ])
    return images, gt, rgb, gray


def decision_rows(deltas, rescues, jaccards) -> list[dict]:
    """生成三seed判定输入。"""
    return [
        {"seed": seed, "primary_delta": delta, "gray_rescue": rescue, "fn_jaccard": jaccard}
        for seed, delta, rescue, jaccard in zip((42, 202, 503), deltas, rescues, jaccards)
    ]


def test_gray_data() -> None:
    """验证PNG三通道、标签、split和一一映射。"""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "source"
        output = Path(temporary) / "gray"
        rows = []
        for index, split in enumerate(("train", "val", "test")):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            image[:, :, index % 3] = 50 + index
            image_path = image_dir / f"sample_{index}.jpg"
            cv2.imwrite(str(image_path), image)
            label_path = label_dir / f"sample_{index}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n" if index else "", encoding="utf-8")
            rows.append({
                "image_relpath": f"original/{split}/sample_{index}.jpg",
                "yolo_image_relpath": image_path.relative_to(root).as_posix(),
                "yolo_label_relpath": label_path.relative_to(root).as_posix(),
                "patient_id": f"p{index}", "label": int(index > 0), "split": split,
                "sha256": file_sha256(image_path), "bbox_xyxy": "0,0,1,1",
            })
        pd.DataFrame(rows).to_csv(root / "y0f_mapping.csv", index=False, encoding="utf-8-sig")
        config = prepare_gray_dataset(root, output, strict_counts=False)
        manifest = pd.read_csv(output / "gray_manifest.csv", encoding="utf-8-sig")
        assert config["images"] == {"train": 1, "val": 1}
        assert len(manifest) == 2 and set(manifest.split) == {"train", "val"}
        assert not (output / "images/test").exists()
        for row in manifest.itertuples(index=False):
            image = cv2.imread(str(output / row.gray_relpath))
            assert image is not None and np.array_equal(image[:, :, 0], image[:, :, 1])
            assert np.array_equal(image[:, :, 1], image[:, :, 2])
            assert file_sha256(output / row.gray_label_relpath) == file_sha256(
                root / row.source_label_relpath
            )


def main() -> None:
    """运行20项有明确协议含义的断言。"""
    test_gray_data()
    images, gt, rgb, gray = detection_fixture()

    detected, missed, rgb_fp = detection_errors(rgb, gt, 0.5)
    assert detected == {("a", 3)}
    assert missed == {("b", 7)}
    assert len(rgb_fp) == 1

    complement = fn_complementarity(rgb, gray, gt, 0.5, 0.5)
    assert complement["gray_rescue"] == 1.0
    assert complement["rgb_reverse_rescue"] == 1.0
    assert complement["fn_jaccard"] == 0.0

    _, _, gray_fp = detection_errors(gray, gt, 0.5)
    overlap = fp_overlap(rgb_fp, gray_fp)
    assert overlap["matched_fp"] == 1
    assert overlap["fp_jaccard"] == 1.0

    # 最高IoU贪心会选(0,0)后只得到1对；正式匹配应得到2对。
    matrix = np.array([[0.90, 0.80], [0.70, 0.10]])
    pairs = maximum_cardinality_iou_pairs(matrix, 0.30)
    assert {(row, column) for row, column, _ in pairs} == {(0, 1), (1, 0)}
    assert math.isclose(sum(iou for _, _, iou in pairs), 1.5)
    assert maximum_cardinality_iou_pairs(np.empty((0, 2))) == []

    pass_result = classify_cd1(decision_rows(
        [-1 / 240, 0.02, 0.02], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]
    ))
    assert pass_result["decision"] == "PASS"
    below_pass = classify_cd1(decision_rows(
        [-1 / 240 - 1e-6, 0.02, 0.02], [0.25, 0.25, 0.25], [0.7, 0.7, 0.7]
    ))
    assert below_pass["decision"] == "COMPLEMENTARY_PASS"
    complementary = classify_cd1(decision_rows(
        [-0.03, -0.01, 0.0], [0.20, 0.21, 0.19], [0.75, 0.70, 0.76]
    ))
    assert complementary["decision"] == "COMPLEMENTARY_PASS"
    inconclusive = classify_cd1(decision_rows(
        [-0.04, -0.01, 0.0], [0.20, 0.20, 0.20], [0.75, 0.75, 0.75]
    ))
    assert inconclusive["decision"] == "INCONCLUSIVE"
    failed = classify_cd1(decision_rows(
        [-0.03, -0.03, -0.01], [0.10, 0.10, 0.10], [0.90, 0.90, 0.90]
    ))
    assert failed["decision"] == "FAIL"
    only_three_fail_checks = classify_cd1(decision_rows(
        [-0.03, -0.03, -0.01], [0.10, 0.10, 0.10], [0.78, 0.80, 0.81]
    ))
    assert only_three_fail_checks["decision"] == "INCONCLUSIVE"

    bootstrap = paired_patient_bootstrap(
        images, gt, rgb, gray, 0.5, 0.5, repeats=20, seed=7
    )
    assert len(bootstrap) == 20
    assert set(bootstrap.columns) == {
        "repeat", "rgb_sensitivity", "gray_sensitivity", "sensitivity_delta",
        "rgb_fp_per_image", "gray_fp_per_image",
    }
    assert np.isfinite(bootstrap.drop(columns="repeat").to_numpy()).all()
    print("CD1 tests: 20/20 passed")


if __name__ == "__main__":
    main()
