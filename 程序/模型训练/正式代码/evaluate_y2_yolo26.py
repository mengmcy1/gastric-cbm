#!/usr/bin/env python3
"""Evaluate one Y2-B checkpoint with the frozen Top-1 geometry protocol.

Ultralytics mAP remains a standard detection metric. This script separately
extracts the highest-confidence end-to-end box at confidence 0.001, locks a
90%-sensitivity deployment threshold on val cancer images, and compares image
detection with the paired M1 seed-42 val predictions. Internal test and external
images are never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from run_y1_yolo26_smoke import PROJECT_ROOT, Y0_ROOT, assert_locked_library_behavior, file_sha256
from train_y2_yolo26 import DEFAULT_OUTPUT_ROOT, expected_run_name


M1_RUN = PROJECT_ROOT / (
    "结果/M1辅助定位_0804/正式验证集筛选/"
    "m1_balanced_keep_efficientnet_b0_seed42_warmup_product"
)
M1_FIXED_THRESHOLD = 0.23004847764968872
GEOMETRY_CONFIDENCE = 0.001
GEOMETRY_MAX_DET = 100
TARGET_SENSITIVITY = 0.90
SAFETY_MARGIN = -0.02
BOOTSTRAP_REPETITIONS = 5000
INFERENCE_BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    """Parse a frozen resolution candidate and its selected CUDA device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, choices=(640, 960), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def box_metrics(pred: np.ndarray | None, gt: np.ndarray) -> tuple[float, float, float, float]:
    """Return IoU, lesion coverage, center-hit and normalized predicted area."""
    if pred is None:
        return 0.0, 0.0, 0.0, 0.0
    ix1, iy1 = np.maximum(pred[:2], gt[:2])
    ix2, iy2 = np.minimum(pred[2:], gt[2:])
    intersection = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
    pred_area = max(0.0, float(pred[2] - pred[0])) * max(0.0, float(pred[3] - pred[1]))
    gt_area = max(0.0, float(gt[2] - gt[0])) * max(0.0, float(gt[3] - gt[1]))
    union = pred_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0.0
    coverage = intersection / gt_area if gt_area > 0 else 0.0
    center_x = float((pred[0] + pred[2]) / 2)
    center_y = float((pred[1] + pred[3]) / 2)
    center_hit = float(gt[0] <= center_x <= gt[2] and gt[1] <= center_y <= gt[3])
    return iou, coverage, center_hit, pred_area


def lock_recall_threshold(values: np.ndarray, recall: float) -> float | None:
    """Return the highest observed positive score reaching the requested recall."""
    candidates = np.sort(np.unique(values[values >= GEOMETRY_CONFIDENCE]))[::-1]
    for threshold in candidates:
        if float(np.mean(values >= threshold)) >= recall:
            return float(threshold)
    return None


def predict_val(
    model: YOLO,
    frame: pd.DataFrame,
    imgsz: int,
    device: str,
    source_dir: Path | None = None,
) -> pd.DataFrame:
    """Run deterministic inference and attach one Top-1 box per val image.

    ``source_dir`` defaults to the frozen Y2 balanced validation directory.
    Later stages must pass their role-specific directory explicitly so the
    yielded paths remain in one-to-one correspondence with ``frame``.
    """
    rows_by_path = {}
    for order, (_, row) in enumerate(frame.iterrows()):
        resolved = (PROJECT_ROOT / str(row["image_relpath"])).resolve()
        if resolved in rows_by_path:
            raise ValueError(f"val存在重复解析路径: {resolved}")
        rows_by_path[resolved] = (order, row)
    results = model.predict(
        # A list[str] is treated as one in-memory batch in this locked version.
        # Passing the directory activates LoadImagesAndVideos and honors batch=4.
        source=str(source_dir or (Y0_ROOT / "images/val")),
        imgsz=imgsz,
        conf=GEOMETRY_CONFIDENCE,
        max_det=GEOMETRY_MAX_DET,
        device=device,
        batch=INFERENCE_BATCH_SIZE,
        stream=True,
        augment=False,
        nms=False,
        verbose=False,
    )
    records = []
    for result in results:
        result_path = Path(result.path).resolve()
        if result_path not in rows_by_path:
            raise RuntimeError(f"YOLO预测路径不在冻结val清单: {result.path}")
        order, row = rows_by_path.pop(result_path)
        confidences = result.boxes.conf.detach().cpu().numpy()
        normalized_boxes = result.boxes.xyxyn.detach().cpu().numpy()
        if len(confidences):
            top_index = int(np.argmax(confidences))
            top_confidence = float(confidences[top_index])
            top_box = normalized_boxes[top_index].astype(float)
        else:
            top_confidence = 0.0
            top_box = None

        gt = np.array(
            [row["bbox_x1_norm"], row["bbox_y1_norm"], row["bbox_x2_norm"], row["bbox_y2_norm"]],
            dtype=float,
        )
        if int(row["label"]) == 1:
            iou, coverage, center_hit, pred_area = box_metrics(top_box, gt)
        else:
            iou, coverage, center_hit = math.nan, math.nan, math.nan
            pred_area = 0.0 if top_box is None else float(
                max(0.0, top_box[2] - top_box[0]) * max(0.0, top_box[3] - top_box[1])
            )
        record = row.to_dict()
        record.update(
            {
                "_val_order": order,
                "top1_confidence": top_confidence,
                "candidate_count_at_0p001": int(len(confidences)),
                "top1_x1": math.nan if top_box is None else float(top_box[0]),
                "top1_y1": math.nan if top_box is None else float(top_box[1]),
                "top1_x2": math.nan if top_box is None else float(top_box[2]),
                "top1_y2": math.nan if top_box is None else float(top_box[3]),
                "top1_iou": iou,
                "lesion_coverage": coverage,
                "center_hit": center_hit,
                "predicted_area_fraction": pred_area,
                "all_confidences": ";".join(f"{float(value):.8f}" for value in confidences),
            }
        )
        records.append(record)
    if rows_by_path or len(records) != len(frame):
        raise RuntimeError(f"YOLO预测数量不完整: {len(records)} != {len(frame)}")
    return pd.DataFrame(records).sort_values("_val_order").drop(columns="_val_order").reset_index(drop=True)


def paired_m1_baseline(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Align frozen M1 val confidence to the same images used by YOLO."""
    path = M1_RUN / "val_image_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少M1配对预测: {path}")
    # Preserve the checkpoint-era float exactly at the frozen threshold boundary.
    m1 = pd.read_csv(
        path,
        usecols=["image_relpath", "localization_confidence"],
        float_precision="round_trip",
    )
    if m1["image_relpath"].duplicated().any():
        raise ValueError("M1 val预测存在重复image_relpath")
    merged = predictions.merge(m1, on="image_relpath", how="left", validate="one_to_one")
    if merged["localization_confidence"].isna().any():
        raise ValueError("YOLO val图像无法与M1预测完整配对")
    cancer = merged["label"].eq(1)
    m1_detected = merged["localization_confidence"].ge(M1_FIXED_THRESHOLD)
    metrics = {
        "threshold": M1_FIXED_THRESHOLD,
        "cancer_images": int(cancer.sum()),
        "cancer_detected": int((cancer & m1_detected).sum()),
        "sensitivity": float(m1_detected[cancer].mean()),
        "checkpoint_sha256": file_sha256(M1_RUN / "m1_best_warmup_localization.pth"),
    }
    merged["m1_detected"] = m1_detected
    return merged, metrics


def patient_bootstrap_sensitivity_difference(frame: pd.DataFrame) -> list[float]:
    """Bootstrap YOLO-minus-M1 sensitivity while retaining all images per patient."""
    cancer = frame[frame["label"].eq(1)].copy()
    grouped = {patient: group for patient, group in cancer.groupby("patient_id", sort=False)}
    patients = np.array(list(grouped), dtype=object)
    rng = np.random.default_rng(20260813)
    differences = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        sampled_frames = [grouped[patient] for patient in sampled]
        bootstrap = pd.concat(sampled_frames, ignore_index=True)
        differences.append(
            float(bootstrap["yolo_detected"].mean() - bootstrap["m1_detected"].mean())
        )
    return differences


def summarize_geometry(frame: pd.DataFrame, train_mapping: pd.DataFrame, threshold: float) -> dict:
    """Compute the pre-registered overall, false-positive and lesion-size metrics."""
    cancer = frame[frame["label"].eq(1)]
    noncancer = frame[frame["label"].eq(0)]
    train_areas = train_mapping.loc[train_mapping["label"].eq(1), "bbox_area_fraction"].to_numpy()
    terciles = np.quantile(train_areas, [1 / 3, 2 / 3])
    groups = pd.cut(
        cancer["bbox_area_fraction"],
        bins=[-np.inf, terciles[0], terciles[1], np.inf],
        labels=["small", "medium", "large"],
        include_lowest=True,
    )
    stratified = {}
    for name in ("small", "medium", "large"):
        subset = cancer[groups.eq(name)]
        stratified[name] = {
            "images": int(len(subset)),
            "patients": int(subset["patient_id"].nunique()),
            "sensitivity": float(subset["yolo_detected"].mean()),
            "mean_iou": float(subset["top1_iou"].mean()),
            "iou_ge_0p5": float(subset["top1_iou"].ge(0.5).mean()),
        }
    noncancer_counts = noncancer["all_confidences"].map(
        lambda text: sum(float(value) >= threshold for value in str(text).split(";") if value)
    )
    return {
        "cancer_images": int(len(cancer)),
        "noncancer_images": int(len(noncancer)),
        "deployment_threshold": threshold,
        "sensitivity": float(cancer["yolo_detected"].mean()),
        "noncancer_positive_trigger_rate": float(noncancer["yolo_detected"].mean()),
        "noncancer_boxes_per_image": float(noncancer_counts.mean()),
        "mean_iou": float(cancer["top1_iou"].mean()),
        "median_iou": float(cancer["top1_iou"].median()),
        "iou_ge_0p5": float(cancer["top1_iou"].ge(0.5).mean()),
        "mean_lesion_coverage": float(cancer["lesion_coverage"].mean()),
        "center_hit_rate": float(cancer["center_hit"].mean()),
        "missing_candidate_cancer_images": int(cancer["top1_confidence"].lt(GEOMETRY_CONFIDENCE).sum()),
        "train_lesion_area_terciles": [float(value) for value in terciles],
        "lesion_size_stratified": stratified,
    }


def main() -> None:
    """Evaluate one completed training run and save predictions plus a frozen summary."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝运行Y2几何评估")
    run_dir = args.output_root.resolve() / expected_run_name(args.imgsz, args.seed, args.debug)
    train_config_path = run_dir / "y2_train_config.json"
    output_path = run_dir / "y2_geometry_config.json"
    if not train_config_path.is_file() or not (run_dir / "weights/best.pt").is_file():
        raise FileNotFoundError(f"Y2训练产物不完整: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"Y2几何结果已存在，拒绝覆盖: {output_path}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    if bool(train_config["debug"]) != args.debug or int(train_config["training_protocol"]["imgsz"]) != args.imgsz:
        raise ValueError("Y2训练config与待评估角色不一致")

    mapping = pd.read_csv(Y0_ROOT / "y0_mapping.csv")
    val = mapping[mapping["split"].eq("val")].copy().reset_index(drop=True)
    train = mapping[mapping["split"].eq("train")].copy()
    model = YOLO(str(run_dir / "weights/best.pt"))
    behavior = assert_locked_library_behavior(model)
    predictions = predict_val(model, val, args.imgsz, args.device)
    cancer_scores = predictions.loc[predictions["label"].eq(1), "top1_confidence"].to_numpy()
    threshold = lock_recall_threshold(cancer_scores, TARGET_SENSITIVITY)
    target_recall_reachable = threshold is not None
    if threshold is None:
        # Keep a complete failure record instead of aborting the matrix. Images
        # without a candidate remain undetected at the lowest allowed threshold.
        threshold = GEOMETRY_CONFIDENCE
    predictions["yolo_detected"] = predictions["top1_confidence"].ge(threshold)
    predictions, m1_metrics = paired_m1_baseline(predictions)
    geometry = summarize_geometry(predictions, train, threshold)
    difference = geometry["sensitivity"] - m1_metrics["sensitivity"]
    bootstrap = patient_bootstrap_sensitivity_difference(predictions)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    safety = {
        "point_difference_yolo_minus_m1": float(difference),
        "point_gate_margin": SAFETY_MARGIN,
        "passed_engineering_gate": bool(difference >= SAFETY_MARGIN),
        "patient_cluster_bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_percentile_95_ci": [float(ci[0]), float(ci[1])],
        "statistical_noninferiority": bool(ci[0] > SAFETY_MARGIN),
    }
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    best_index = int(history["metrics/mAP50-95(B)"].idxmax())
    standard_validation = {
        "checkpoint_selection": "maximum val mAP50-95 fitness",
        "best_epoch": int(history.loc[best_index, "epoch"]),
        "mAP50": float(history.loc[best_index, "metrics/mAP50(B)"]),
        "mAP50_95": float(history.loc[best_index, "metrics/mAP50-95(B)"]),
        "standard_val_max_det": 300,
    }

    predictions.to_csv(run_dir / "y2_val_top1_predictions.csv", index=False, encoding="utf-8-sig")
    config = {
        "stage": "Y2-B",
        "role": "balanced_primary_resolution_screen",
        "debug": args.debug,
        "internal_test_read": False,
        "external_read": False,
        "checkpoint": str(run_dir / "weights/best.pt"),
        "checkpoint_sha256": file_sha256(run_dir / "weights/best.pt"),
        "model_behavior": behavior,
        "geometry_protocol": {
            "imgsz": args.imgsz,
            "confidence": GEOMETRY_CONFIDENCE,
            "max_det": GEOMETRY_MAX_DET,
            "nms": False,
            "top1": "highest final end-to-end confidence",
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "target_sensitivity": TARGET_SENSITIVITY,
            "target_sensitivity_reachable": target_recall_reachable,
            "missing_candidate_geometry": 0,
        },
        "m1_paired_baseline": m1_metrics,
        "standard_validation": standard_validation,
        "geometry": geometry,
        "safety": safety,
        "selection_eligible": bool(
            not args.debug and target_recall_reachable and safety["passed_engineering_gate"]
        ),
    }
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Y2-B {args.imgsz}: Sens={geometry['sensitivity']:.4f}, "
        f"FP={geometry['noncancer_positive_trigger_rate']:.4f}, "
        f"IoU50={geometry['iou_ge_0p5']:.4f}, meanIoU={geometry['mean_iou']:.4f}"
    )
    print(
        f"相对M1 Sens差={difference:+.4f}, 95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}], "
        f"工程门槛={safety['passed_engineering_gate']}"
    )
    print(f"输出: {run_dir}")


if __name__ == "__main__":
    main()
