#!/usr/bin/env python3
"""Evaluate one Y2-S checkpoint with the unchanged Y2 Top-1 protocol."""

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
    lock_recall_threshold,
    paired_m1_baseline,
    patient_bootstrap_sensitivity_difference,
    predict_val,
    summarize_geometry,
)
from run_y1_yolo26_smoke import PROJECT_ROOT, Y0_ROOT, assert_locked_library_behavior, file_sha256
from train_y2_resolution_sensitivity import DEFAULT_OUTPUT_ROOT, expected_run_name
from train_y2_yolo26 import DEFAULT_OUTPUT_ROOT as Y2_ROOT
from train_y2_yolo26 import expected_run_name as y2_run_name


def parse_args() -> argparse.Namespace:
    """Parse one frozen sensitivity candidate and its CUDA device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, choices=(640, 960), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def metric_delta(current: dict, original: dict) -> dict:
    """Return the within-resolution change caused by no-Mosaic fine-tuning."""
    keys = (
        "sensitivity",
        "noncancer_positive_trigger_rate",
        "noncancer_boxes_per_image",
        "mean_iou",
        "median_iou",
        "iou_ge_0p5",
        "mean_lesion_coverage",
        "center_hit_rate",
    )
    return {key: float(current[key] - original[key]) for key in keys}


def main() -> None:
    """Evaluate one complete Y2-S run without reading test or external data."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing Y2-S evaluation")
    run_dir = args.output_root.resolve() / expected_run_name(args.imgsz, args.seed, args.debug)
    train_config_path = run_dir / "y2s_train_config.json"
    output_path = run_dir / "y2s_geometry_config.json"
    checkpoint = run_dir / "weights/best.pt"
    if not train_config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Y2-S training products are incomplete: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"Y2-S geometry result already exists: {output_path}")

    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    protocol = train_config["training_protocol"]
    if bool(train_config["debug"]) != args.debug or int(protocol["imgsz"]) != args.imgsz:
        raise ValueError("Y2-S training config role mismatch")
    if file_sha256(checkpoint) != train_config["products"]["best_pt_sha256"]:
        raise ValueError("Y2-S best checkpoint SHA mismatch")

    original_dir = Y2_ROOT / y2_run_name(args.imgsz, args.seed, False)
    original_path = original_dir / "y2_geometry_config.json"
    if not original_path.is_file():
        raise FileNotFoundError(f"Missing original Y2 geometry result: {original_path}")
    original_config = json.loads(original_path.read_text(encoding="utf-8"))
    original_geometry = original_config["geometry"]

    mapping = pd.read_csv(Y0_ROOT / "y0_mapping.csv")
    val = mapping[mapping["split"].eq("val")].copy().reset_index(drop=True)
    train = mapping[mapping["split"].eq("train")].copy()
    model = YOLO(str(checkpoint))
    behavior = assert_locked_library_behavior(model)
    predictions = predict_val(model, val, args.imgsz, args.device)
    cancer_scores = predictions.loc[predictions["label"].eq(1), "top1_confidence"].to_numpy()
    threshold = lock_recall_threshold(cancer_scores, TARGET_SENSITIVITY)
    target_recall_reachable = threshold is not None
    if threshold is None:
        threshold = GEOMETRY_CONFIDENCE
    predictions["yolo_detected"] = predictions["top1_confidence"].ge(threshold)
    predictions, m1_metrics = paired_m1_baseline(predictions)
    geometry = summarize_geometry(predictions, train, threshold)

    sensitivity_difference = geometry["sensitivity"] - m1_metrics["sensitivity"]
    bootstrap = patient_bootstrap_sensitivity_difference(predictions)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    safety = {
        "point_difference_yolo_minus_m1": float(sensitivity_difference),
        "point_gate_margin": SAFETY_MARGIN,
        "passed_engineering_gate": bool(sensitivity_difference >= SAFETY_MARGIN),
        "patient_cluster_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_percentile_95_ci": [float(ci[0]), float(ci[1])],
        "statistical_noninferiority": bool(ci[0] > SAFETY_MARGIN),
    }

    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    best_index = int(history["metrics/mAP50-95(B)"].idxmax())
    standard_validation = {
        "checkpoint_selection": "maximum val mAP50-95 fitness during 20-epoch fine-tuning",
        "best_epoch": int(history.loc[best_index, "epoch"]),
        "mAP50": float(history.loc[best_index, "metrics/mAP50(B)"]),
        "mAP50_95": float(history.loc[best_index, "metrics/mAP50-95(B)"]),
        "standard_val_max_det": 300,
    }

    predictions.to_csv(run_dir / "y2s_val_top1_predictions.csv", index=False, encoding="utf-8-sig")
    config = {
        "stage": "Y2-S",
        "role": "supplementary_no_mosaic_convergence_sensitivity",
        "debug": args.debug,
        "primary_y2_replacement_allowed": False,
        "internal_test_read": False,
        "external_read": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_behavior": behavior,
        "geometry_protocol": {
            "imgsz": args.imgsz,
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
        "original_y2_geometry": original_geometry,
        "delta_from_original_y2": metric_delta(geometry, original_geometry),
        "safety": safety,
        "comparison_eligible": bool(
            not args.debug and target_recall_reachable and safety["passed_engineering_gate"]
        ),
    }
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    delta = config["delta_from_original_y2"]
    print(
        f"Y2-S {args.imgsz}: Sens={geometry['sensitivity']:.4f}, "
        f"IoU50={geometry['iou_ge_0p5']:.4f} ({delta['iou_ge_0p5']:+.4f}), "
        f"meanIoU={geometry['mean_iou']:.4f} ({delta['mean_iou']:+.4f})"
    )
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
