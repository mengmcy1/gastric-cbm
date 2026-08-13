#!/usr/bin/env python3
"""Build the locked Y0 YOLO dataset view and its leakage/QC audit.

The script exports only train/val images and labels for model development. The
internal test split is recorded as a checksum-locked queue, but is deliberately
excluded from ``data.yaml`` and from the YOLO image/label directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "08_M1辅助定位清单_20260810/"
    "m1_balanced_keep_primary_1to1p3_split_seed42.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "10_Y0_YOLO26检测数据_20260813"
)
BBOX_COLUMNS = [
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
]


def parse_args() -> argparse.Namespace:
    """Return command-line settings for the deterministic Y0 export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=6,
        help="Maximum 64-bit perceptual-hash Hamming distance for a review candidate.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Calculate a file SHA-256 without loading the complete image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash selected dataframe columns in their frozen row order."""
    payload = frame[columns].fillna("").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """Return a 64-bit DCT perceptual hash used only to nominate manual reviews."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.dct(np.float32(resized))[:8, :8]
    threshold = float(np.median(low_frequency.reshape(-1)[1:]))
    bits = low_frequency > threshold
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def validate_manifest(frame: pd.DataFrame) -> None:
    """Validate split, label and single-box conventions before writing outputs."""
    required = {
        "image_relpath",
        "patient_id",
        "label",
        "split",
        "sha256",
        "width",
        "height",
        "localization_supervision",
        "bbox_valid",
        *BBOX_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"输入清单缺少字段: {missing}")
    if not set(frame["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("split仅允许train/val/test")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label仅允许0/1")
    if frame.groupby("patient_id")["split"].nunique().max() != 1:
        raise ValueError("发现患者跨split，停止Y0构建")
    if frame["image_relpath"].duplicated().any():
        raise ValueError("发现重复image_relpath，停止Y0构建")

    positive = frame["label"].eq(1)
    bbox = frame[BBOX_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_positive = (
        bbox.notna().all(axis=1)
        & bbox["bbox_x1_norm"].ge(0)
        & bbox["bbox_y1_norm"].ge(0)
        & bbox["bbox_x2_norm"].le(1)
        & bbox["bbox_y2_norm"].le(1)
        & bbox["bbox_x2_norm"].gt(bbox["bbox_x1_norm"])
        & bbox["bbox_y2_norm"].gt(bbox["bbox_y1_norm"])
    )
    if not valid_positive[positive].all():
        raise ValueError("存在癌图缺失或越界bbox，停止Y0构建")
    if not frame.loc[positive, "localization_supervision"].eq(1).all():
        raise ValueError("存在癌图未开启定位监督，停止Y0构建")
    if frame.loc[~positive, "localization_supervision"].ne(0).any():
        raise ValueError("存在非癌图开启定位监督，停止Y0构建")


def inspect_images(
    frame: pd.DataFrame, near_duplicate_distance: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode every image, verify SHA/dimensions and return suspicious pHash pairs."""
    records: list[dict] = []
    failures: list[dict] = []
    for row_index, row in frame.iterrows():
        path = PROJECT_ROOT / str(row["image_relpath"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"row_index": row_index, "reason": "decode_error", "path": str(path)})
            continue
        actual_sha = file_sha256(path)
        actual_height, actual_width = image.shape[:2]
        if actual_sha != str(row["sha256"]):
            failures.append({"row_index": row_index, "reason": "sha256_mismatch", "path": str(path)})
        if actual_width != int(row["width"]) or actual_height != int(row["height"]):
            failures.append({"row_index": row_index, "reason": "dimension_mismatch", "path": str(path)})
        records.append(
            {
                "row_index": row_index,
                "split": row["split"],
                "patient_id": row["patient_id"],
                "label": int(row["label"]),
                "sha256": actual_sha,
                "phash": perceptual_hash(image),
                "image_relpath": row["image_relpath"],
            }
        )
    if failures:
        preview = pd.DataFrame(failures).head(10).to_dict("records")
        raise ValueError(f"图像完整性检查失败，共{len(failures)}项，示例: {preview}")

    image_audit = pd.DataFrame(records)
    pairs: list[dict] = []
    values = image_audit.to_dict("records")
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            if left["patient_id"] == right["patient_id"]:
                continue
            distance = (int(left["phash"]) ^ int(right["phash"])).bit_count()
            if distance <= near_duplicate_distance:
                pairs.append(
                    {
                        "hamming_distance": distance,
                        "cross_split": left["split"] != right["split"],
                        "left_split": left["split"],
                        "right_split": right["split"],
                        "left_patient_id": left["patient_id"],
                        "right_patient_id": right["patient_id"],
                        "left_label": left["label"],
                        "right_label": right["label"],
                        "left_image_relpath": left["image_relpath"],
                        "right_image_relpath": right["image_relpath"],
                    }
                )
    near_pairs = pd.DataFrame(pairs)
    if not near_pairs.empty:
        near_pairs = near_pairs.sort_values(
            ["cross_split", "hamming_distance"], ascending=[False, True]
        )
    return image_audit, near_pairs


def patient_frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-patient image frequency for each split and class."""
    counts = (
        frame.groupby(["split", "label", "patient_id"], as_index=False)
        .size()
        .rename(columns={"size": "images_per_patient"})
    )
    rows = []
    for (split, label), group in counts.groupby(["split", "label"]):
        values = group["images_per_patient"].to_numpy()
        cutoff = np.quantile(values, 0.9)
        rows.append(
            {
                "split": split,
                "label": int(label),
                "patients": len(values),
                "images": int(values.sum()),
                "min": int(values.min()),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "p90": float(cutoff),
                "max": int(values.max()),
                "images_from_top_frequency_patients": int(values[values >= cutoff].sum()),
            }
        )
    return pd.DataFrame(rows)


def export_yolo_view(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create symlinked train/val images and YOLO labels; leave test unexported."""
    mapping_rows = []
    for row_index, row in frame.iterrows():
        mapping = row.to_dict()
        mapping["yolo_image_relpath"] = ""
        mapping["yolo_label_relpath"] = ""
        if row["split"] in {"train", "val"}:
            source = PROJECT_ROOT / str(row["image_relpath"])
            suffix = source.suffix.lower() or ".jpg"
            stem = f"{row['split']}_{row_index:06d}_{str(row['sha256'])[:12]}"
            image_path = output_dir / "images" / str(row["split"]) / f"{stem}{suffix}"
            label_path = output_dir / "labels" / str(row["split"]) / f"{stem}.txt"
            image_path.symlink_to(source)
            if int(row["label"]) == 1:
                width = float(row["bbox_x2_norm"] - row["bbox_x1_norm"])
                height = float(row["bbox_y2_norm"] - row["bbox_y1_norm"])
                center_x = float(row["bbox_x1_norm"] + width / 2)
                center_y = float(row["bbox_y1_norm"] + height / 2)
                label_path.write_text(
                    f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n",
                    encoding="ascii",
                )
            else:
                label_path.write_text("", encoding="ascii")
            mapping["yolo_image_relpath"] = str(image_path.relative_to(output_dir))
            mapping["yolo_label_relpath"] = str(label_path.relative_to(output_dir))
        mapping_rows.append(mapping)
    return pd.DataFrame(mapping_rows)


def write_manual_checklist(output_dir: Path) -> None:
    """Write the clinical checks that cannot be inferred from coordinates alone."""
    text = """# Y0人工确认清单

- [ ] 所有癌图中的目标病灶均已标注；当前CSV结构每图只能表达一个框。
- [ ] 若同图存在多个分离的早癌/HGD病灶，需改为一图多框YOLO标签后再进入Y2正式训练。
- [ ] 非癌空标签仅表示“无早癌/HGD目标框”，不表示图中没有炎症、息肉、溃疡等异常。
- [ ] 抽查近重复候选，确认不存在跨患者或跨split的同帧/相邻帧泄漏。
- [ ] 抽查YOLO框转换后的位置与原始结构化bbox一致。

自动审计只能验证文件、坐标和重复候选，不能替代临床病灶完整性确认。
"""
    (output_dir / "Y0_人工确认清单.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run Y0 validation, audit and leakage-safe train/val YOLO export."""
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")

    frame = pd.read_csv(manifest)
    validate_manifest(frame)
    image_audit, near_pairs = inspect_images(frame, args.near_duplicate_distance)
    exact_duplicate_rows = int(image_audit["sha256"].duplicated(keep=False).sum())
    if exact_duplicate_rows:
        raise ValueError(f"发现{exact_duplicate_rows}行精确重复图像，停止Y0构建")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=False)
    (output_dir / "audit").mkdir(parents=True, exist_ok=False)

    mapping = export_yolo_view(frame, output_dir)
    mapping.to_csv(output_dir / "y0_mapping.csv", index=False, encoding="utf-8-sig")
    test_queue = frame.loc[frame["split"].eq("test")].copy()
    test_queue.to_csv(output_dir / "locked_internal_test_queue.csv", index=False, encoding="utf-8-sig")
    patient_frequency = patient_frequency_table(frame)
    patient_frequency.to_csv(output_dir / "audit/patient_image_frequency.csv", index=False)
    near_pairs.to_csv(output_dir / "audit/near_duplicate_candidates.csv", index=False, encoding="utf-8-sig")

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
        "stage": "Y0",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "frozen_queue_sha256": dataframe_sha256(
            frame,
            ["image_relpath", "patient_id", "label", "split", "sha256", *BBOX_COLUMNS],
        ),
        "class_name": "early_cancer_or_HGD",
        "single_box_schema": True,
        "clinical_multilesion_completeness_confirmed": False,
        "near_duplicate_hash": "64-bit DCT pHash",
        "near_duplicate_hamming_threshold": args.near_duplicate_distance,
        "exact_duplicate_rows": exact_duplicate_rows,
        "near_duplicate_candidates_different_patient": int(len(near_pairs)),
        "near_duplicate_candidates_cross_split": cross_split_near,
        "test_exported_to_yolo": False,
        "test_evaluated": False,
        "counts": split_summary.to_dict("records"),
    }
    (output_dir / "y0_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Y0输出: {output_dir}")
    print(split_summary.to_string(index=False))
    print(f"跨患者近重复候选: {len(near_pairs)}; 其中跨split: {cross_split_near}")
    print("internal test仅冻结为队列，未导出到data.yaml，未计算test指标。")


if __name__ == "__main__":
    args = parse_args()
    main()
