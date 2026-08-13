#!/usr/bin/env python3
"""Render Y0 cross-split pHash candidates as side-by-side review sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
Y0_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "10_Y0_YOLO26检测数据_20260813"
)


def parse_args() -> argparse.Namespace:
    """Return input/output settings for the deterministic review export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Y0_ROOT / "audit/near_duplicate_candidates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Y0_ROOT / "audit/cross_split_near_duplicate_review",
    )
    parser.add_argument("--pairs-per-page", type=int, default=3)
    parser.add_argument(
        "--include-same-split",
        action="store_true",
        help="Render every cross-patient candidate instead of cross-split pairs only.",
    )
    return parser.parse_args()


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image into a fixed dark canvas without changing aspect ratio."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def render_pair(row: pd.Series, pair_id: int) -> np.ndarray:
    """Render one candidate pair with split, label and Hamming-distance metadata."""
    left = cv2.imread(str(PROJECT_ROOT / row["left_image_relpath"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(PROJECT_ROOT / row["right_image_relpath"]), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise ValueError(f"候选对图像读取失败: pair {pair_id}")
    left = fit_image(left, 560, 360)
    right = fit_image(right, 560, 360)
    panel = np.full((430, 1140, 3), 245, dtype=np.uint8)
    panel[55:415, 5:565] = left
    panel[55:415, 575:1135] = right
    title = f"pair {pair_id:03d} | pHash distance={int(row['hamming_distance'])}"
    left_meta = f"LEFT  {row['left_split']} label={int(row['left_label'])}"
    right_meta = f"RIGHT {row['right_split']} label={int(row['right_label'])}"
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, left_meta, (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, right_meta, (578, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    """Create paginated JPEG sheets and a blank decision table for manual review."""
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {args.output_dir}")
    candidates = pd.read_csv(args.candidates)
    if not args.include_same_split:
        candidates = candidates.loc[candidates["cross_split"].astype(bool)]
    candidates = candidates.reset_index(drop=True)
    if candidates.empty:
        raise ValueError("没有跨split近重复候选")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    panels = [render_pair(row, index + 1) for index, row in candidates.iterrows()]
    for start in range(0, len(panels), args.pairs_per_page):
        page_panels = panels[start : start + args.pairs_per_page]
        if len(page_panels) < args.pairs_per_page:
            blank = np.full_like(page_panels[0], 245)
            page_panels.extend([blank] * (args.pairs_per_page - len(page_panels)))
        page = np.vstack(page_panels)
        page_number = start // args.pairs_per_page + 1
        cv2.imwrite(str(args.output_dir / f"cross_split_review_{page_number:02d}.jpg"), page)

    review = candidates.copy()
    review.insert(0, "pair_id", np.arange(1, len(review) + 1))
    review["review_decision"] = ""
    review["review_notes"] = ""
    review.to_csv(args.output_dir / "cross_split_review_decisions.csv", index=False, encoding="utf-8-sig")
    scope = "全部跨患者" if args.include_same_split else "跨split"
    print(f"{scope}候选: {len(review)}组; 对照页: {(len(panels) - 1) // args.pairs_per_page + 1}页")
    print(f"输出: {args.output_dir}")


if __name__ == "__main__":
    main()
