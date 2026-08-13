#!/usr/bin/env python3
"""Build the locked Y0-F full-data YOLO diagnostic view.

The input is the frozen 3348-image M1 full Keep manifest. All rows are audited,
but only its original train/val splits are exported to YOLO. The original
501-image full-data test split is checksum-locked and excluded; Y4 will instead
use the already frozen 198-image balanced internal-test queue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_y0_yolo_dataset import (
    BBOX_COLUMNS,
    PROJECT_ROOT,
    dataframe_sha256,
    export_yolo_view,
    file_sha256,
    inspect_images,
    patient_frequency_table,
    validate_manifest,
    write_manual_checklist,
)


DEFAULT_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "09_M1全量诊断清单_20260810/m1_full_keep_split_seed42.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)
BALANCED_TEST_QUEUE = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "10_Y0_YOLO26检测数据_20260813/locked_internal_test_queue.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse the frozen source, isolated output and pHash review distance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """Build the full diagnostic YOLO view without exposing either test queue.

    The frozen 3348-row manifest is decoded and audited in full. Only its
    train/val rows are symlinked and converted to YOLO labels; the original
    501-row test split is recorded separately and remains unavailable to Y3.
    """
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Y0-F output already exists: {output_dir}")
    if not BALANCED_TEST_QUEUE.is_file():
        raise FileNotFoundError(f"Missing frozen Y4 queue: {BALANCED_TEST_QUEUE}")

    frame = pd.read_csv(manifest)
    validate_manifest(frame)
    expected = {"rows": 3348, "patients": 1732, "cancer": 1670, "control": 1678}
    actual = {
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "cancer": int(frame["label"].eq(1).sum()),
        "control": int(frame["label"].eq(0).sum()),
    }
    if actual != expected:
        raise ValueError(f"Y0-F frozen cohort mismatch: expected={expected}, actual={actual}")

    image_audit, near_pairs = inspect_images(frame, args.near_duplicate_distance)
    exact_duplicate_rows = int(image_audit["sha256"].duplicated(keep=False).sum())
    if exact_duplicate_rows:
        raise ValueError(f"Y0-F has {exact_duplicate_rows} exact-duplicate rows")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=False)
    (output_dir / "audit").mkdir(parents=True, exist_ok=False)

    mapping = export_yolo_view(frame, output_dir)
    mapping.to_csv(output_dir / "y0f_mapping.csv", index=False, encoding="utf-8-sig")
    original_test = frame.loc[frame["split"].eq("test")].copy()
    original_test.to_csv(
        output_dir / "excluded_full_original_test_queue.csv", index=False, encoding="utf-8-sig"
    )
    patient_frequency_table(frame).to_csv(
        output_dir / "audit/patient_image_frequency.csv", index=False
    )
    near_pairs.to_csv(
        output_dir / "audit/near_duplicate_candidates.csv", index=False, encoding="utf-8-sig"
    )
    bbox_summary = (
        frame.loc[frame["label"].eq(1)]
        .groupby("split")["bbox_area_fraction"]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
    )
    bbox_summary.to_csv(output_dir / "audit/bbox_area_summary.csv", index=False)
    split_summary = (
        frame.groupby(["split", "label"], as_index=False)
        .agg(images=("image_relpath", "size"), patients=("patient_id", "nunique"))
    )
    split_summary.to_csv(output_dir / "audit/split_class_summary.csv", index=False)

    data_yaml = (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: early_cancer_or_HGD\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    write_manual_checklist(output_dir)

    cross_split_near = 0 if near_pairs.empty else int(near_pairs["cross_split"].sum())
    config = {
        "stage": "Y0-F",
        "role": "full_data_diagnostic_only",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "frozen_queue_sha256": dataframe_sha256(
            frame,
            ["image_relpath", "patient_id", "label", "split", "sha256", *BBOX_COLUMNS],
        ),
        "class_name": "early_cancer_or_HGD",
        "single_box_schema": True,
        "clinical_multilesion_completeness_confirmed": False,
        "no_known_multilesion_images_reported": True,
        "near_duplicate_hash": "64-bit DCT pHash",
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "exact_duplicate_rows": exact_duplicate_rows,
        "near_duplicate_candidates_different_patient": int(len(near_pairs)),
        "near_duplicate_candidates_cross_split": cross_split_near,
        "full_original_test_exported_to_yolo": False,
        "full_original_test_evaluated": False,
        "full_original_test_rows": int(len(original_test)),
        "y4_balanced_test_queue": str(BALANCED_TEST_QUEUE),
        "y4_balanced_test_queue_sha256": file_sha256(BALANCED_TEST_QUEUE),
        "internal_test_read": False,
        "external_read": False,
        "counts": split_summary.to_dict("records"),
    }
    (output_dir / "y0f_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y0-F output: {output_dir}")
    print(split_summary.to_string(index=False))
    print(f"Near-duplicate candidates: {len(near_pairs)}; cross-split: {cross_split_near}")
    print("The full-data original test was checksum-locked but not exported or evaluated.")


if __name__ == "__main__":
    main()
