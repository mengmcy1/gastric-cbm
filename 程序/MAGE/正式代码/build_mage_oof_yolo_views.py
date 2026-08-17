#!/usr/bin/env python3
"""Build five patient-disjoint YOLO views for MG0b OOF ROI generation.

For each outer fold, the other four train folds are split by patient into fit
and monitor subsets. The outer holdout patients are exported only for later
prediction. Project val, original test and external data are never used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from build_mage_g0_manifest import file_sha256


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MG0_MANIFEST = PROJECT_ROOT / (
    "数据整理记录/MAGE/MG0a_患者级OOF分折_20260817/mage_g0_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "数据整理记录/MAGE/MG0b_OOF_YOLO数据_20260817"
Y0F_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)
EXPECTED_MG0_MANIFEST_SHA256 = "8a840fbcf02cbe7518bdcd395d39eced03591bbfed327c3d8fc755dad792d7c4"
N_FOLDS = 5
MONITOR_FRACTION = 0.15
MONITOR_SEED_BASE = 4200


def parse_args() -> argparse.Namespace:
    """Parse the frozen MG0a manifest, isolated output and self-test mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MG0_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def split_development_patients(
    patients: pd.DataFrame, fold: int
) -> tuple[set[str], set[str]]:
    """Split non-holdout patients into fit and early-stop monitor sets.

    Args:
        patients: Unique ``patient_id`` and binary ``label`` rows.
        fold: Outer OOF fold, used to derive a deterministic monitor seed.

    Returns:
        Two disjoint patient-id sets: fit and monitor.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=MONITOR_FRACTION,
        random_state=MONITOR_SEED_BASE + fold,
    )
    fit_indices, monitor_indices = next(
        splitter.split(patients.patient_id, patients.label)
    )
    fit = set(patients.iloc[fit_indices].patient_id.astype(str))
    monitor = set(patients.iloc[monitor_indices].patient_id.astype(str))
    if fit & monitor or len(fit | monitor) != len(patients):
        raise RuntimeError(f"fold{fold}内部fit/monitor患者划分异常")
    return fit, monitor


def link_row(row: pd.Series, fold_root: Path, role: str) -> tuple[str, str]:
    """Create image/label symlinks for one fit, monitor or holdout row.

    Args:
        row: MG0a image row containing Y0-F relative image and label paths.
        fold_root: Current outer-fold dataset root.
        role: ``fit``, ``monitor`` or ``holdout``.

    Returns:
        Relative image and label paths inside the fold dataset.
    """
    source_image = Y0F_ROOT / str(row.yolo_image_relpath)
    source_label = Y0F_ROOT / str(row.yolo_label_relpath)
    if not source_image.is_file() or not source_label.is_file():
        raise FileNotFoundError(f"Y0-F YOLO文件缺失: {source_image} / {source_label}")
    stem = source_image.stem
    image_target = fold_root / "images" / role / source_image.name
    label_target = fold_root / "labels" / role / f"{stem}.txt"
    image_target.symlink_to(source_image.resolve())
    label_target.symlink_to(source_label.resolve())
    return str(image_target.relative_to(fold_root)), str(label_target.relative_to(fold_root))


def build_fold(frame: pd.DataFrame, output_dir: Path, fold: int) -> dict:
    """Build one outer-fold fit/monitor/holdout view and audit metadata."""
    train = frame.loc[frame.split.eq("train")].copy()
    holdout = train.loc[train.oof_fold.eq(fold)].copy()
    development = train.loc[~train.oof_fold.eq(fold)].copy()
    patients = (
        development[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    fit_ids, monitor_ids = split_development_patients(patients, fold)
    fit = development.loc[development.patient_id.isin(fit_ids)].copy()
    monitor = development.loc[development.patient_id.isin(monitor_ids)].copy()
    patient_sets = [set(part.patient_id.astype(str)) for part in (fit, monitor, holdout)]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise RuntimeError(f"fold{fold}存在患者跨fit/monitor/holdout")
    if len(pd.concat([fit, monitor, holdout])) != len(train):
        raise RuntimeError(f"fold{fold}未完整覆盖train图片")

    fold_root = output_dir / f"fold_{fold}"
    for role in ("fit", "monitor", "holdout"):
        (fold_root / "images" / role).mkdir(parents=True)
        (fold_root / "labels" / role).mkdir(parents=True)

    records = []
    for role, part in (("fit", fit), ("monitor", monitor), ("holdout", holdout)):
        for _, row in part.iterrows():
            image_path, label_path = link_row(row, fold_root, role)
            record = row.to_dict()
            record.update({
                "outer_fold": fold,
                "oof_role": role,
                "fold_image_relpath": image_path,
                "fold_label_relpath": label_path,
            })
            records.append(record)
    mapping = pd.DataFrame(records)
    mapping_path = fold_root / "fold_mapping.csv"
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    (fold_root / "data.yaml").write_text(
        f"path: {fold_root}\n"
        "train: images/fit\n"
        "val: images/monitor\n"
        "names:\n"
        "  0: early_cancer_or_HGD\n",
        encoding="utf-8",
    )
    summary = (
        mapping.groupby(["oof_role", "label"], as_index=False)
        .agg(images=("image_relpath", "size"), patients=("patient_id", "nunique"))
    )
    summary.to_csv(fold_root / "split_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "fold": fold,
        "mapping": str(mapping_path),
        "mapping_sha256": file_sha256(mapping_path),
        "data_yaml": str(fold_root / "data.yaml"),
        "monitor_fraction": MONITOR_FRACTION,
        "monitor_seed": MONITOR_SEED_BASE + fold,
        "counts": summary.to_dict("records"),
    }


def run_self_test() -> None:
    """Check deterministic, stratified and disjoint fit/monitor splitting."""
    patients = pd.DataFrame({
        "patient_id": [f"p{i:03d}" for i in range(100)],
        "label": [0] * 70 + [1] * 30,
    })
    fit_a, monitor_a = split_development_patients(patients, fold=2)
    fit_b, monitor_b = split_development_patients(patients, fold=2)
    assert fit_a == fit_b and monitor_a == monitor_b
    assert not fit_a & monitor_a and fit_a | monitor_a == set(patients.patient_id)
    labels = patients.set_index("patient_id").label
    assert labels.loc[list(monitor_a)].nunique() == 2
    print("MG0b数据视图self-test通过: 内部分层确定且患者互斥")


def main() -> None:
    """Build all five OOF YOLO views without training or inference."""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"MG0b数据视图已存在: {output_dir}")
    if EXPECTED_MG0_MANIFEST_SHA256 and file_sha256(manifest) != EXPECTED_MG0_MANIFEST_SHA256:
        raise ValueError("MG0a manifest SHA与冻结记录不一致")
    frame = pd.read_csv(manifest, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "image_relpath", "yolo_image_relpath", "yolo_label_relpath", "patient_id",
        "label", "split", "oof_fold", "sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MG0a manifest缺少字段: {sorted(missing)}")
    if set(frame.split) != {"train", "val"} or len(frame) != 2847:
        raise ValueError("MG0a manifest数据边界异常")
    train = frame.loc[frame.split.eq("train")]
    if train.groupby("patient_id").oof_fold.nunique().ne(1).any():
        raise ValueError("MG0a患者跨OOF fold")

    output_dir.mkdir(parents=True)
    folds = [build_fold(frame, output_dir, fold) for fold in range(N_FOLDS)]
    config = {
        "stage": "MG0b_data_views",
        "created_on": "2026-08-17",
        "mg0a_manifest": str(manifest),
        "mg0a_manifest_sha256": file_sha256(manifest),
        "n_outer_folds": N_FOLDS,
        "monitor_fraction": MONITOR_FRACTION,
        "original_project_val_used_for_training_or_early_stop": False,
        "project_test_read": False,
        "internal_test_read": False,
        "external_read": False,
        "folds": folds,
    }
    (output_dir / "mg0b_data_views_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MG0b五折YOLO数据视图: {output_dir}")
    for item in folds:
        print(f"fold {item['fold']}: {item['counts']}")
    print("仅建立软链接；尚未训练OOF检测器，项目val/test/external未进入训练视图。")


if __name__ == "__main__":
    main()
