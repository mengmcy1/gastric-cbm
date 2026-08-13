#!/usr/bin/env python3
"""Evaluate one Y3 replicate against the same-seed frozen M1 product.

The evaluator reuses the Y2 Top-1 protocol at a val-locked 90% cancer-image
sensitivity. It additionally computes paired M1 box geometry and patient-cluster
bootstrap intervals for sensitivity, IoU>=0.5 and mean IoU differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from evaluate_y2_yolo26 import (
    BOOTSTRAP_REPETITIONS,
    GEOMETRY_CONFIDENCE,
    GEOMETRY_MAX_DET,
    INFERENCE_BATCH_SIZE,
    SAFETY_MARGIN,
    TARGET_SENSITIVITY,
    box_metrics,
    lock_recall_threshold,
    predict_val,
    summarize_geometry,
)
from run_y1_yolo26_smoke import PROJECT_ROOT, assert_locked_library_behavior, file_sha256
from train_y3_yolo26 import (
    ALLOWED_SEEDS,
    DEFAULT_OUTPUT_ROOT,
    Y0_FULL_ROOT,
    expected_run_name,
)
from run_y1_yolo26_smoke import Y0_ROOT


M1_ROOT = PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"


def parse_args() -> argparse.Namespace:
    """Parse role, seed, CUDA device and isolated debug flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("balanced", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def m1_run_dir(role: str, seed: int) -> Path:
    """Return the same-seed frozen warmup-only M1 product directory."""
    prefix = "m1_balanced_keep" if role == "balanced" else "m1_full_keep"
    return M1_ROOT / f"{prefix}_efficientnet_b0_seed{seed}_warmup_product"


def pair_m1_geometry(predictions: pd.DataFrame, role: str, seed: int) -> tuple[pd.DataFrame, dict]:
    """Pair YOLO rows with same-seed M1 predictions and compute M1 geometry.

    Args:
        predictions: One row per validation image with normalized YOLO Top-1
            box coordinates and the frozen binary label.
        role: Balanced or full diagnostic data role.
        seed: Replicate seed used to locate the matching M1 product.

    Returns:
        The one-to-one merged image table and M1 metrics at its own val-locked
        90% cancer-image-sensitivity threshold.
    """
    run_dir = m1_run_dir(role, seed)
    path = run_dir / "val_image_predictions.csv"
    checkpoint = run_dir / "m1_best_warmup_localization.pth"
    if not path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing same-seed M1 product: {run_dir}")
    columns = [
        "image_relpath",
        "localization_confidence",
        "valid_box",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
    ]
    m1 = pd.read_csv(path, usecols=columns, float_precision="round_trip")
    if m1["image_relpath"].duplicated().any():
        raise ValueError("M1 val predictions contain duplicate image paths")
    merged = predictions.merge(m1, on="image_relpath", how="left", validate="one_to_one")
    if merged["localization_confidence"].isna().any():
        raise ValueError("Y3 val images cannot be fully paired with M1")

    cancer = merged["label"].eq(1)
    cancer_scores = merged.loc[cancer, "localization_confidence"].to_numpy()
    threshold = lock_recall_threshold(cancer_scores, TARGET_SENSITIVITY)
    if threshold is None:
        raise RuntimeError("Same-seed M1 cannot reach the frozen 90% sensitivity target")
    merged["m1_detected"] = merged["localization_confidence"].ge(threshold)

    m1_ious = np.full(len(merged), np.nan, dtype=float)
    m1_center_hits = np.full(len(merged), np.nan, dtype=float)
    for index in np.flatnonzero(cancer.to_numpy()):
        row = merged.iloc[index]
        gt = row[["gt_x1", "gt_y1", "gt_x2", "gt_y2"]].to_numpy(dtype=float)
        pred = None
        if bool(row["valid_box"]):
            pred = row[["pred_x1", "pred_y1", "pred_x2", "pred_y2"]].to_numpy(dtype=float)
        iou, _, center_hit, _ = box_metrics(pred, gt)
        m1_ious[index] = iou
        m1_center_hits[index] = center_hit
    merged["m1_iou"] = m1_ious
    merged["m1_center_hit"] = m1_center_hits
    cancer_frame = merged.loc[cancer]
    metrics = {
        "threshold": float(threshold),
        "cancer_images": int(cancer.sum()),
        "cancer_detected": int(cancer_frame["m1_detected"].sum()),
        "sensitivity": float(cancer_frame["m1_detected"].mean()),
        "mean_iou": float(cancer_frame["m1_iou"].mean()),
        "iou_ge_0p5": float(cancer_frame["m1_iou"].ge(0.5).mean()),
        "center_hit_rate": float(cancer_frame["m1_center_hit"].mean()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    return merged, metrics


def paired_patient_bootstrap(frame: pd.DataFrame, seed: int) -> dict:
    """Return paired patient-cluster CIs for YOLO-minus-M1 cancer metrics.

    Whole cancer patients are sampled with replacement so repeated patients
    retain their full image contribution. Returned intervals cover sensitivity,
    IoU>=0.5 and mean-IoU differences.
    """
    cancer = frame[frame["label"].eq(1)].copy()
    grouped = {patient: group for patient, group in cancer.groupby("patient_id", sort=False)}
    patients = np.array(list(grouped), dtype=object)
    rng = np.random.default_rng(20260813 + seed)
    values = {"sensitivity": [], "iou_ge_0p5": [], "mean_iou": []}
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        bootstrap = pd.concat([grouped[patient] for patient in sampled], ignore_index=True)
        values["sensitivity"].append(
            float(bootstrap["yolo_detected"].mean() - bootstrap["m1_detected"].mean())
        )
        values["iou_ge_0p5"].append(
            float(bootstrap["top1_iou"].ge(0.5).mean() - bootstrap["m1_iou"].ge(0.5).mean())
        )
        values["mean_iou"].append(float(bootstrap["top1_iou"].mean() - bootstrap["m1_iou"].mean()))
    return {
        name: [float(value) for value in np.quantile(samples, [0.025, 0.975])]
        for name, samples in values.items()
    }


def main() -> None:
    """Evaluate one complete Y3 run using only its role-specific train/val view."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing Y3 evaluation")
    if args.role == "balanced" and args.seed == 42 and not args.debug:
        raise ValueError("Balanced seed42 is the frozen Y2-B reference, not a Y3 run")
    run_dir = args.output_root.resolve() / expected_run_name(args.role, args.seed, args.debug)
    train_config_path = run_dir / "y3_train_config.json"
    output_path = run_dir / "y3_geometry_config.json"
    checkpoint = run_dir / "weights/best.pt"
    if not train_config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Y3 training products are incomplete: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"Y3 geometry result already exists: {output_path}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if (
        train_config["role"] != args.role
        or int(train_config["seed"]) != args.seed
        or bool(train_config["debug"]) != args.debug
    ):
        raise ValueError("Y3 train config role, seed or debug mismatch")
    if file_sha256(checkpoint) != train_config["products"]["best_pt_sha256"]:
        raise ValueError("Y3 best checkpoint SHA mismatch")

    data_root = Y0_ROOT if args.role == "balanced" else Y0_FULL_ROOT
    mapping_name = "y0_mapping.csv" if args.role == "balanced" else "y0f_mapping.csv"
    mapping = pd.read_csv(data_root / mapping_name)
    val = mapping[mapping["split"].eq("val")].copy().reset_index(drop=True)
    train = mapping[mapping["split"].eq("train")].copy()
    model = YOLO(str(checkpoint))
    behavior = assert_locked_library_behavior(model)
    predictions = predict_val(
        model,
        val,
        640,
        args.device,
        source_dir=data_root / "images/val",
    )
    cancer_scores = predictions.loc[predictions["label"].eq(1), "top1_confidence"].to_numpy()
    threshold = lock_recall_threshold(cancer_scores, TARGET_SENSITIVITY)
    target_recall_reachable = threshold is not None
    if threshold is None:
        threshold = GEOMETRY_CONFIDENCE
    predictions["yolo_detected"] = predictions["top1_confidence"].ge(threshold)
    predictions, m1_metrics = pair_m1_geometry(predictions, args.role, args.seed)
    geometry = summarize_geometry(predictions, train, threshold)
    differences = {
        "sensitivity": float(geometry["sensitivity"] - m1_metrics["sensitivity"]),
        "iou_ge_0p5": float(geometry["iou_ge_0p5"] - m1_metrics["iou_ge_0p5"]),
        "mean_iou": float(geometry["mean_iou"] - m1_metrics["mean_iou"]),
    }
    bootstrap_ci = paired_patient_bootstrap(predictions, args.seed)
    safety = {
        "point_gate_margin": SAFETY_MARGIN,
        "passed_sensitivity_engineering_gate": bool(differences["sensitivity"] >= SAFETY_MARGIN),
        "paired_differences_yolo_minus_m1": differences,
        "patient_cluster_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_percentile_95_ci": bootstrap_ci,
    }

    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    best_index = int(history["metrics/mAP50-95(B)"].idxmax())
    standard_validation = {
        "best_epoch": int(history.loc[best_index, "epoch"]),
        "mAP50": float(history.loc[best_index, "metrics/mAP50(B)"]),
        "mAP50_95": float(history.loc[best_index, "metrics/mAP50-95(B)"]),
    }
    predictions.to_csv(run_dir / "y3_val_top1_predictions.csv", index=False, encoding="utf-8-sig")
    config = {
        "stage": "Y3-B" if args.role == "balanced" else "Y3-F",
        "role": args.role,
        "seed": args.seed,
        "debug": args.debug,
        "internal_test_read": False,
        "external_read": False,
        "full_original_test_read": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_behavior": behavior,
        "geometry_protocol": {
            "imgsz": 640,
            "confidence": GEOMETRY_CONFIDENCE,
            "max_det": GEOMETRY_MAX_DET,
            "nms": False,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "target_sensitivity": TARGET_SENSITIVITY,
            "target_sensitivity_reachable": target_recall_reachable,
        },
        "m1_paired_baseline": m1_metrics,
        "standard_validation": standard_validation,
        "geometry": geometry,
        "safety": safety,
        "replicate_success": bool(
            not args.debug
            and target_recall_reachable
            and safety["passed_sensitivity_engineering_gate"]
            and differences["iou_ge_0p5"] > 0
            and differences["mean_iou"] > 0
        ),
    }
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Y3 {args.role} seed{args.seed}: Sens={geometry['sensitivity']:.4f}, "
        f"IoU50={geometry['iou_ge_0p5']:.4f} ({differences['iou_ge_0p5']:+.4f} vs M1), "
        f"meanIoU={geometry['mean_iou']:.4f} ({differences['mean_iou']:+.4f} vs M1)"
    )
    print(f"Replicate success={config['replicate_success']}; output={run_dir}")


if __name__ == "__main__":
    main()
