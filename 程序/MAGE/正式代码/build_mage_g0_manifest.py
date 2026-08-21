#!/usr/bin/env python3
"""构建冻结的MG0a训练/验证数据血缘和患者级OOF分折。

本脚本读取冻结的Y0-F清单，排除原始测试集，并将每位训练患者分配到
五个分层留出折之一；不训练YOLO、不生成ROI，也不读取内外部测试数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MAPPING = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813/y0f_mapping.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0a_患者级OOF分折_20260817"
)
EXPECTED_MAPPING_SHA256 = "b329d8d3b0033e84124fb7be6db05a70d6bb46fb014a3653ef20cadac07407cd"
N_FOLDS = 5
FOLD_SEED = 42


def parse_args() -> argparse.Namespace:
    """解析冻结清单、独立输出目录和自测开关。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """以流式读取方式计算单个文件的SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    """稳定排序指定列后计算哈希，用于数据血缘校验。"""
    stable = frame[columns].astype(str).replace("<NA>", "")
    stable = stable.sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(stable.to_csv(index=False).encode("utf-8")).hexdigest()


def validate_source(frame: pd.DataFrame) -> None:
    """校验Y0-F队列的规模、患者隔离、标签、重复项和癌图框监督。"""
    required = {
        "image_relpath", "yolo_image_relpath", "patient_id", "label", "split",
        "source", "center", "size_group", "sha256", "bbox_valid",
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "bbox_area_fraction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Y0-F mapping缺少字段: {sorted(missing)}")
    expected_images = {
        ("train", 0): 1169, ("train", 1): 1181,
        ("val", 0): 257, ("val", 1): 240,
        ("test", 0): 252, ("test", 1): 249,
    }
    if len(frame) != 3348 or frame.groupby(["split", "label"]).size().to_dict() != expected_images:
        raise ValueError("Y0-F规模或split/label计数与冻结记录不一致")
    if frame.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Y0-F存在患者跨split")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("Y0-F存在患者跨标签")
    if frame.sha256.duplicated().any():
        raise ValueError("Y0-F存在重复SHA行")
    cancer = frame.label.eq(1)
    if not frame.loc[cancer, "bbox_valid"].astype(bool).all():
        raise ValueError("Y0-F癌图存在无效GT bbox")
    boxes = frame.loc[cancer, [
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"
    ]].to_numpy(float)
    if not np.all((boxes >= 0) & (boxes <= 1)):
        raise ValueError("Y0-F癌图bbox超出归一化坐标范围")
    if not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
        raise ValueError("Y0-F癌图bbox宽高非正")


def assign_patient_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """按标签分层，将每位训练患者分配到一个OOF留出折。"""
    train_patients = (
        frame.loc[frame.split.eq("train"), ["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    patient_to_fold: dict[str, int] = {}
    for fold, (_, holdout_indices) in enumerate(
        splitter.split(train_patients.patient_id, train_patients.label)
    ):
        for patient_id in train_patients.iloc[holdout_indices].patient_id:
            patient_to_fold[str(patient_id)] = fold
    if len(patient_to_fold) != len(train_patients):
        raise RuntimeError("MG0a未给每位train患者分配唯一OOF fold")

    output = frame.copy()
    output["oof_fold"] = output.patient_id.astype(str).map(patient_to_fold).astype("Int64")
    if output.loc[output.split.eq("train"), "oof_fold"].isna().any():
        raise RuntimeError("MG0a train存在未分折患者")
    if output.loc[output.split.eq("val"), "oof_fold"].notna().any():
        raise RuntimeError("MG0a val不应分配OOF fold")
    return output


def annotate_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    """预留后续ROI来源字段，但不伪造任何预测结果。"""
    output = frame.copy()
    train = output.split.eq("train")
    cancer = output.label.eq(1)
    output["privileged_roi_source"] = np.select(
        [cancer, train & ~cancer, ~train & ~cancer],
        ["gt_bbox", "pending_oof_yolo_top1", "pending_frozen_y3f_seed42_top1"],
        default="invalid",
    )
    output["teacher_input_protocol"] = "square_roi_margin0.20_luma_3ch_224"
    output["student_input_protocol"] = "full_rgb_224"
    output["internal_test_read"] = False
    output["external_read"] = False
    return output


def fold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """按折和标签汇总留出图像数与患者数。"""
    train = frame.loc[frame.split.eq("train")].copy()
    return (
        train.groupby(["oof_fold", "label"], as_index=False)
        .agg(images=("image_relpath", "size"), patients=("patient_id", "nunique"))
        .sort_values(["oof_fold", "label"])
    )


def run_self_test() -> None:
    """自测患者分组和确定性五折分配。"""
    rows = []
    for label in (0, 1):
        for patient in range(10):
            for image in range(2):
                rows.append({
                    "patient_id": f"{label}-{patient}", "label": label,
                    "split": "train", "image_relpath": f"{label}-{patient}-{image}.jpg",
                })
    synthetic = pd.DataFrame(rows)
    first = assign_patient_folds(synthetic)
    second = assign_patient_folds(synthetic)
    assert first.oof_fold.equals(second.oof_fold)
    assert first.groupby("patient_id").oof_fold.nunique().eq(1).all()
    assert set(first.oof_fold.astype(int)) == set(range(N_FOLDS))
    print("MG0a self-test通过: 分折确定、患者不跨fold、5折齐全")


def main() -> None:
    """构建MG0a清单及机器可读的审计产物。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    mapping = args.mapping.resolve()
    output_dir = args.output_dir.resolve()
    if file_sha256(mapping) != EXPECTED_MAPPING_SHA256:
        raise ValueError("Y0-F mapping SHA与冻结记录不一致")
    if output_dir.exists():
        raise FileExistsError(f"MG0a输出目录已存在: {output_dir}")

    full = pd.read_csv(mapping, encoding="utf-8-sig", dtype={"patient_id": str})
    validate_source(full)
    frame = full.loc[full.split.isin(["train", "val"])].copy().reset_index(drop=True)
    frame = annotate_protocol(assign_patient_folds(frame))
    if len(frame) != 2847 or frame.patient_id.nunique() != 1472:
        raise ValueError("MG0a train/val规模与冻结记录不一致")

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "mage_g0_manifest.csv"
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    folds = fold_summary(frame)
    folds.to_csv(output_dir / "oof_fold_summary.csv", index=False, encoding="utf-8-sig")
    split_summary = (
        frame.groupby(["split", "label"], as_index=False)
        .agg(images=("image_relpath", "size"), patients=("patient_id", "nunique"))
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    lineage_columns = ["image_relpath", "patient_id", "label", "split", "sha256", "oof_fold"]
    config = {
        "stage": "MG0a",
        "created_on": "2026-08-17",
        "mapping": str(mapping),
        "mapping_sha256": file_sha256(mapping),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "lineage_dataframe_sha256": dataframe_sha256(frame, lineage_columns),
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        "train_images": int(frame.split.eq("train").sum()),
        "train_patients": int(frame.loc[frame.split.eq("train"), "patient_id"].nunique()),
        "val_images": int(frame.split.eq("val").sum()),
        "val_patients": int(frame.loc[frame.split.eq("val"), "patient_id"].nunique()),
        "test_rows_in_source": int(full.split.eq("test").sum()),
        "test_rows_exported": 0,
        "internal_test_read": False,
        "external_read": False,
        "teacher_roi_protocol": "square ROI, margin=0.20, luma replicated to 3 channels, 224x224",
        "student_input_protocol": "full RGB WLI, 224x224",
        "train_noncancer_roi_status": "pending patient-level OOF YOLO Top-1",
        "val_noncancer_roi_status": "pending frozen Y3-F seed42 Top-1",
        "fold_summary": folds.to_dict("records"),
        "split_summary": split_summary.to_dict("records"),
    }
    (output_dir / "mg0a_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MG0a输出: {output_dir}")
    print(split_summary.to_string(index=False))
    print(folds.to_string(index=False))
    print("原Y0-F test未导出；internal test/external未读取。")


if __name__ == "__main__":
    main()
