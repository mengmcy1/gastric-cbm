#!/usr/bin/env python3
"""Build frozen M0 full-diagnostic Keep/Notch manifests for the 0804 data."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRED_ROOT = (
    PROJECT_ROOT
    / "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1"
    / "05_Keep_Notch配对manifest_20260805"
)
DEFAULT_KEEP_METADATA = (
    PROJECT_ROOT
    / "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1"
    / "最终Keep集_应用保守回退_20260804/keep_final_manifest_v2.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1"
    / "06_M0全量诊断清单_20260805"
)
CROSS_LABEL_PATIENTS = {
    "01.0000000129422",
    "01.0000000166648",
}
SPLIT_SEED = 42
TRAINING_SEEDS = [42, 202, 503]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-manifest",
        type=Path,
        default=DEFAULT_PAIRED_ROOT / "keep_manifest.csv",
    )
    parser.add_argument(
        "--notch-manifest",
        type=Path,
        default=DEFAULT_PAIRED_ROOT / "notch_manifest.csv",
    )
    parser.add_argument(
        "--keep-metadata",
        type=Path,
        default=DEFAULT_KEEP_METADATA,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    return parser.parse_args()


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_from_path(relative_path):
    if "/外院数据/" in relative_path:
        return "外院"
    if "/武大省人民数据/" in relative_path:
        return "武大省人民"
    if "/第一届早癌大赛/" in relative_path:
        return "第一届早癌大赛"
    if "/第二届早癌大赛/" in relative_path:
        return "第二届早癌大赛"
    raise ValueError(f"无法识别来源：{relative_path}")


def label_from_path(relative_path):
    if relative_path.startswith("早癌与高级别/"):
        return 1
    if relative_path.startswith("非癌/"):
        return 0
    raise ValueError(f"无法识别标签：{relative_path}")


def patient_leaf(patient_id):
    return Path(patient_id).name


def aspect_group(width, height):
    ratio = width / height
    if ratio < 0.90:
        return "portrait"
    if ratio <= 1.10:
        return "square"
    if ratio < 1.50:
        return "landscape"
    return "wide"


def size_group(width, height):
    longest = max(width, height)
    if longest <= 640:
        return "small_le640"
    if longest <= 1024:
        return "medium_641_1024"
    if longest <= 1600:
        return "large_1025_1600"
    return "xlarge_gt1600"


def extract_year(text):
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)]
    plausible = [value for value in years if 2000 <= value <= 2030]
    return min(plausible) if plausible else ""


def allocate_split(patient_frame, seed):
    rng = np.random.default_rng(seed)
    assignments = []
    for (source, label), group in patient_frame.groupby(
        ["source", "label"], sort=True
    ):
        patient_ids = sorted(group["patient_id"].tolist())
        patient_ids = list(rng.permutation(patient_ids))
        count = len(patient_ids)
        test_count = max(1, int(round(count * 0.15)))
        val_count = max(1, int(round(count * 0.15)))
        if test_count + val_count >= count:
            raise ValueError(f"分层患者数过少：{source}/{label}/{count}")
        split_by_index = (
            ["test"] * test_count
            + ["val"] * val_count
            + ["train"] * (count - test_count - val_count)
        )
        assignments.extend(
            {
                "patient_id": patient_id,
                "source": source,
                "label": int(label),
                "split": split,
            }
            for patient_id, split in zip(patient_ids, split_by_index)
        )
    result = pd.DataFrame(assignments)
    if result["patient_id"].duplicated().any():
        raise ValueError("患者划分中出现重复 patient_id")
    return result


def prepare_branch(branch, eligible_paths, split_by_patient, metadata):
    frame = branch.loc[branch["relative_path"].isin(eligible_paths)].copy()
    frame["label"] = frame["relative_path"].map(label_from_path)
    frame["source"] = frame["relative_path"].map(source_from_path)
    frame["center"] = frame["source"]
    frame["patient_leaf"] = frame["patient_id"].map(patient_leaf)
    frame["image_relpath"] = frame["input_path"]
    frame["year"] = frame["relative_path"].map(extract_year)
    frame["aspect_group"] = [
        aspect_group(width, height)
        for width, height in zip(frame["width"], frame["height"])
    ]
    frame["size_group"] = [
        size_group(width, height)
        for width, height in zip(frame["width"], frame["height"])
    ]
    crop_source = frame["relative_path"].map(metadata["crop_source"])
    frame["frame_profile"] = crop_source.fillna("unknown")
    frame["style_group"] = (
        frame["size_group"]
        + "|"
        + frame["aspect_group"]
        + "|"
        + frame["frame_profile"]
    )
    frame["split"] = frame["patient_id"].map(split_by_patient)
    if frame["split"].isna().any():
        raise ValueError("存在未分配split的图片")
    columns = [
        "relative_path",
        "image_relpath",
        "patient_id",
        "patient_leaf",
        "label",
        "split",
        "source",
        "center",
        "year",
        "width",
        "height",
        "size_group",
        "aspect_group",
        "frame_profile",
        "style_group",
        "geometry_id",
        "sha256",
        "pip_present_original",
        "analysis_group",
        "branch",
    ]
    return frame[columns].sort_values(
        ["split", "source", "label", "patient_id", "relative_path"]
    ).reset_index(drop=True)


def audit_branch(frame):
    if len(frame) != frame["relative_path"].nunique():
        raise ValueError("训练清单存在重复 relative_path")
    if frame.groupby("patient_id")["split"].nunique().gt(1).any():
        raise ValueError("患者跨split")
    if frame.groupby("sha256")["split"].nunique().gt(1).any():
        raise ValueError("完全重复图跨split")
    for split, group in frame.groupby("split"):
        if set(group["label"]) != {0, 1}:
            raise ValueError(f"{split}缺少某一类标签")
        if set(group["source"]) != {
            "武大省人民",
            "第一届早癌大赛",
            "第二届早癌大赛",
        }:
            raise ValueError(f"{split}来源不完整")
    missing = [
        path for path in frame["image_relpath"]
        if not (PROJECT_ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺失图片{len(missing)}张，示例：{missing[0]}")


def summary_rows(frame):
    rows = []
    for (split, source, label), group in frame.groupby(
        ["split", "source", "label"], sort=True
    ):
        rows.append({
            "split": split,
            "source": source,
            "label": int(label),
            "images": len(group),
            "patients": group["patient_id"].nunique(),
        })
    return rows


def main():
    args = parse_args()
    for path in [args.keep_manifest, args.notch_manifest, args.keep_metadata]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")

    keep = pd.read_csv(args.keep_manifest, encoding="utf-8-sig")
    notch = pd.read_csv(args.notch_manifest, encoding="utf-8-sig")
    keep_meta = pd.read_csv(args.keep_metadata, encoding="utf-8-sig")
    if set(keep["relative_path"]) != set(notch["relative_path"]):
        raise ValueError("Keep/Notch图片集不一致")
    metadata = keep_meta.set_index("processed_relative_path", verify_integrity=True)

    audit = keep.copy()
    audit["label"] = audit["relative_path"].map(label_from_path)
    audit["source"] = audit["relative_path"].map(source_from_path)
    audit["patient_leaf"] = audit["patient_id"].map(patient_leaf)
    reasons = {relative_path: [] for relative_path in audit["relative_path"]}
    for row in audit.itertuples():
        if row.source == "外院":
            reasons[row.relative_path].append("external_held_out")
        if row.patient_leaf in CROSS_LABEL_PATIENTS:
            reasons[row.relative_path].append("cross_label_patient")

    development = audit.loc[
        audit["source"].ne("外院")
        & ~audit["patient_leaf"].isin(CROSS_LABEL_PATIENTS)
    ].copy()
    development = development.sort_values(["sha256", "relative_path"])
    duplicate_mask = development.duplicated("sha256", keep="first")
    for relative_path in development.loc[duplicate_mask, "relative_path"]:
        reasons[relative_path].append("duplicate_sha256")
    eligible = development.loc[~duplicate_mask].copy()
    eligible_paths = set(eligible["relative_path"])

    patient_frame = eligible[["patient_id", "source", "label"]].drop_duplicates()
    if patient_frame.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("排除后仍存在跨标签患者")
    assignments = allocate_split(patient_frame, args.split_seed)
    split_by_patient = assignments.set_index("patient_id")["split"]

    keep_ready = prepare_branch(
        keep, eligible_paths, split_by_patient, metadata
    )
    notch_ready = prepare_branch(
        notch, eligible_paths, split_by_patient, metadata
    )
    audit_branch(keep_ready)
    audit_branch(notch_ready)
    compare_columns = [
        "relative_path", "patient_id", "label", "split", "geometry_id"
    ]
    if not keep_ready[compare_columns].equals(notch_ready[compare_columns]):
        raise ValueError("Keep/Notch的图片、患者或split不一致")

    args.output.mkdir(parents=True, exist_ok=False)
    keep_path = args.output / "m0_full_keep_split_seed42.csv"
    notch_path = args.output / "m0_full_notch_split_seed42.csv"
    assignment_path = args.output / "patient_split_seed42.csv"
    exclusion_path = args.output / "excluded_rows.csv"
    summary_path = args.output / "split_summary.csv"
    keep_ready.to_csv(keep_path, index=False, encoding="utf-8-sig")
    notch_ready.to_csv(notch_path, index=False, encoding="utf-8-sig")
    assignments.sort_values(["split", "source", "label", "patient_id"]).to_csv(
        assignment_path, index=False, encoding="utf-8-sig"
    )
    excluded = audit.loc[
        audit["relative_path"].map(lambda value: bool(reasons[value]))
    ].copy()
    excluded["exclusion_reasons"] = excluded["relative_path"].map(
        lambda value: "|".join(reasons[value])
    )
    excluded.to_csv(exclusion_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows(keep_ready)).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    pre_registered = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "M0 full diagnostic baseline",
        "scope": "unbalanced eligible development data; external held out",
        "split_seed": args.split_seed,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "training_seeds": TRAINING_SEEDS,
        "augmentation": {
            "random_resized_crop_scale": [0.85, 1.0],
            "random_resized_crop_ratio": [0.90, 1.10],
        },
        "preprocess_delta_auc_noninferiority": 0.005,
        "backbone_delta_auc_noninferiority": 0.005,
        "delta_sensitivity_noninferiority": 0.02,
        "delta_specificity_noninferiority": 0.02,
        "threshold_rule": "max specificity subject to val patient sensitivity >= 0.90",
        "minimum_val_sensitivity": 0.90,
        "minimum_val_specificity": 0.50,
        "paired_bootstrap_iterations": 2000,
        "model_matrix": [
            "resnet50_keep", "resnet50_notch",
            "efficientnet_b0_keep", "efficientnet_b0_notch",
        ],
        "selection_data": "validation only",
        "internal_test_role": "locked evaluation after selection",
    }
    with (args.output / "m0_pre_registered_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(pre_registered, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "frozen_ready_for_debug",
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "inputs": {
            "keep_manifest": str(args.keep_manifest),
            "keep_manifest_sha256": file_sha256(args.keep_manifest),
            "notch_manifest": str(args.notch_manifest),
            "notch_manifest_sha256": file_sha256(args.notch_manifest),
            "keep_metadata": str(args.keep_metadata),
            "keep_metadata_sha256": file_sha256(args.keep_metadata),
        },
        "eligible_images": len(keep_ready),
        "eligible_patients": keep_ready["patient_id"].nunique(),
        "label_images": {
            str(key): int(value)
            for key, value in keep_ready["label"].value_counts().sort_index().items()
        },
        "label_patients": {
            str(key): int(value)
            for key, value in patient_frame["label"].value_counts().sort_index().items()
        },
        "excluded_images": len(excluded),
        "exclusion_reason_counts": dict(Counter(
            reason
            for value in excluded["exclusion_reasons"]
            for reason in value.split("|")
        )),
        "split_summary": summary_rows(keep_ready),
        "keep_manifest": str(keep_path),
        "notch_manifest": str(notch_path),
        "patient_split": str(assignment_path),
        "excluded_rows": str(exclusion_path),
        "pre_registered_config": str(
            args.output / "m0_pre_registered_config.json"
        ),
    }
    with (args.output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
