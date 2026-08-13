#!/usr/bin/env python3
"""Second-stage audit for Y0-F train/validation pHash candidates.

The broad pHash screen is intentionally sensitive to low-texture endoscopy
frames. This audit keeps only cross-patient train/validation pairs, computes
224-pixel grayscale SSIM, and freezes the sortable review table. It does not
remove images or alter either split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[3]
Y0F_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)


def parse_args() -> argparse.Namespace:
    """Parse the frozen Y0-F candidate table and isolated audit outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Y0F_ROOT / "audit/near_duplicate_candidates.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Y0F_ROOT / "audit/train_val_near_duplicate_secondary_audit.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Y0F_ROOT / "audit/train_val_near_duplicate_secondary_audit.json",
    )
    parser.add_argument("--review-threshold", type=float, default=0.85)
    return parser.parse_args()


def ssim224(left_path: Path, right_path: Path) -> float:
    """Return grayscale SSIM after deterministic 224x224 area resizing."""
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError(f"Cannot decode candidate pair: {left_path}, {right_path}")
    left = cv2.resize(left, (224, 224), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (224, 224), interpolation=cv2.INTER_AREA)
    return float(structural_similarity(left, right, data_range=255))


def main() -> None:
    """Freeze SSIM values for every cross-patient train/validation candidate."""
    args = parse_args()
    if args.output_csv.exists() or args.output_json.exists():
        raise FileExistsError("Y0-F secondary duplicate-audit output already exists")
    candidates = pd.read_csv(args.candidates)
    train_val = candidates.loc[
        (candidates["left_split"].eq("train") & candidates["right_split"].eq("val"))
        | (candidates["left_split"].eq("val") & candidates["right_split"].eq("train"))
    ].copy()
    train_val["ssim224"] = [
        ssim224(
            PROJECT_ROOT / str(row.left_image_relpath),
            PROJECT_ROOT / str(row.right_image_relpath),
        )
        for row in train_val.itertuples(index=False)
    ]
    train_val = train_val.sort_values(
        ["ssim224", "hamming_distance"], ascending=[False, True]
    ).reset_index(drop=True)
    train_val.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "stage": "Y0-F train-val near-duplicate secondary audit",
        "pairs": int(len(train_val)),
        "ssim224_mean": float(train_val["ssim224"].mean()),
        "ssim224_median": float(train_val["ssim224"].median()),
        "ssim224_max": float(train_val["ssim224"].max()),
        "review_threshold": args.review_threshold,
        "pairs_at_or_above_review_threshold": int(
            train_val["ssim224"].ge(args.review_threshold).sum()
        ),
        "data_modified": False,
        "full_original_test_used_for_training_decision": False,
    }
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
