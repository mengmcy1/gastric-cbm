#!/usr/bin/env python3
"""Summarize the paired MG1 real/shuffle experiment against frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_mage_teacher_roi_manifest import PROJECT_ROOT, file_sha256


DEFAULT_ROOT = PROJECT_ROOT / "结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选"
MIN_REAL_PATIENT_AUC = 0.80
MIN_GEOMETRY_DELTA = 0.10
MIN_SHUFFLE_DELTA = 0.05


def parse_args() -> argparse.Namespace:
    """Parse the paired formal result root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_formal_config(path: Path, expected_mode: str) -> dict:
    """Load one complete formal MG1 config and verify its mode and product hash."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("stage") != "MG1_grayscale_local_teacher":
        raise ValueError(f"不是MG1正式配置: {path}")
    if config.get("debug") is not False or config.get("input_mode") != expected_mode:
        raise ValueError(f"MG1角色或debug标记错误: {path}")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("MG1正式配对只接受seed42")
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file() or file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError(f"MG1 checkpoint不完整或SHA不一致: {checkpoint}")
    return config


def main() -> None:
    """Apply all preregistered gates and save a single MG1 decision record."""
    args = parse_args()
    root = args.result_root.resolve()
    real_path = root / "mg1_real_efficientnet_b0_seed42/config.json"
    shuffle_path = root / "mg1_patch_shuffle_efficientnet_b0_seed42/config.json"
    real = load_formal_config(real_path, "real")
    shuffle = load_formal_config(shuffle_path, "patch_shuffle")
    if real["manifest_sha256"] != shuffle["manifest_sha256"]:
        raise ValueError("MG1真实与shuffle未使用同一份v3清单")
    if real["training"] != shuffle["training"]:
        raise ValueError("MG1真实与shuffle训练协议不一致")

    real_patient_auc = float(real["metrics"]["val_patient_auc"])
    shuffle_patient_auc = float(shuffle["metrics"]["val_patient_auc"])
    geometry_patient_auc = float(real["geometry_patient_auc"])
    gates = {
        "real_patient_auc_at_least_0p80": real_patient_auc >= MIN_REAL_PATIENT_AUC,
        "real_minus_geometry_at_least_0p10": (
            real_patient_auc - geometry_patient_auc >= MIN_GEOMETRY_DELTA
        ),
        "real_minus_patch_shuffle_at_least_0p05": (
            real_patient_auc - shuffle_patient_auc >= MIN_SHUFFLE_DELTA
        ),
    }
    passed = bool(all(gates.values()))
    summary = {
        "stage": "MG1_paired_decision",
        "seed": 42,
        "manifest_sha256": real["manifest_sha256"],
        "thresholds": {
            "minimum_real_patient_auc": MIN_REAL_PATIENT_AUC,
            "minimum_real_minus_geometry": MIN_GEOMETRY_DELTA,
            "minimum_real_minus_patch_shuffle": MIN_SHUFFLE_DELTA,
        },
        "real": {
            "config": str(real_path),
            "checkpoint": real["checkpoint"],
            "val_image_auc": float(real["metrics"]["val_image_auc"]),
            "val_patient_auc": real_patient_auc,
            "image_threshold_metrics": real["metrics"]["image_threshold_metrics"],
            "patient_threshold_metrics": real["metrics"]["patient_threshold_metrics"],
        },
        "patch_shuffle": {
            "config": str(shuffle_path),
            "val_image_auc": float(shuffle["metrics"]["val_image_auc"]),
            "val_patient_auc": shuffle_patient_auc,
            "image_threshold_metrics": shuffle["metrics"]["image_threshold_metrics"],
            "patient_threshold_metrics": shuffle["metrics"]["patient_threshold_metrics"],
        },
        "geometry_only": {
            "val_patient_auc": geometry_patient_auc,
            "source": "v3 center_plus_size_shape train-fit/val-eval audit",
        },
        "deltas": {
            "real_minus_geometry_patient_auc": real_patient_auc - geometry_patient_auc,
            "real_minus_patch_shuffle_patient_auc": real_patient_auc - shuffle_patient_auc,
        },
        "gates": gates,
        "passed_mg1": passed,
        "decision": "proceed_to_MG2" if passed else "stop_before_MG2",
        "test_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    output_path = root / "mg1_pair_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {
            "mode": "real",
            "val_image_auc": summary["real"]["val_image_auc"],
            "val_patient_auc": real_patient_auc,
        },
        {
            "mode": "patch_shuffle",
            "val_image_auc": summary["patch_shuffle"]["val_image_auc"],
            "val_patient_auc": shuffle_patient_auc,
        },
        {
            "mode": "geometry_only",
            "val_image_auc": None,
            "val_patient_auc": geometry_patient_auc,
        },
    ]).to_csv(root / "mg1_pair_summary.csv", index=False, encoding="utf-8-sig")

    print("=== MG1冻结门槛汇总 ===")
    print(
        f"真实教师: image AUC={summary['real']['val_image_auc']:.4f}, "
        f"patient AUC={real_patient_auc:.4f}"
    )
    print(f"geometry-only patient AUC={geometry_patient_auc:.4f}")
    print(f"patch-shuffle patient AUC={shuffle_patient_auc:.4f}")
    print(
        f"真实-geometry={summary['deltas']['real_minus_geometry_patient_auc']:+.4f}; "
        f"真实-shuffle={summary['deltas']['real_minus_patch_shuffle_patient_auc']:+.4f}"
    )
    print(f"门槛={gates}")
    print(f"MG1结论: {'通过，可进入MG2' if passed else '失败，停止在MG1'}")
    print(f"汇总: {output_path}")


if __name__ == "__main__":
    main()
