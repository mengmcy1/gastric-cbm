#!/usr/bin/env python3
"""CD0 病灶检测指标与错误归因的纯计算函数。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def lesion_size_groups(
    area_fraction: pd.Series,
    bounds: tuple[float, float],
) -> pd.Series:
    """使用训练集预先冻结的面积边界划分病灶大小。"""
    lower, upper = bounds
    return pd.cut(
        area_fraction,
        bins=[-math.inf, lower, upper, math.inf],
        labels=["lesion_small", "lesion_medium", "lesion_large"],
        include_lowest=True,
    )


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算一个 xyxy 框与多个 xyxy 框的 IoU。"""
    if len(boxes) == 0:
        return np.empty(0, dtype=float)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    sizes = np.maximum(0.0, bottom_right - top_left)
    intersection = sizes[:, 0] * sizes[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def match_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    iou_threshold: float,
) -> pd.DataFrame:
    """按置信度降序将预测框与同图未匹配真值框一对一匹配。"""
    selected = predictions[predictions.confidence.ge(threshold)].copy()
    selected = selected.sort_values(
        ["confidence", "image_key", "prediction_index"],
        ascending=[False, True, True],
    )
    gt_groups = {
        key: group.reset_index(drop=True)
        for key, group in ground_truth.groupby("image_key", sort=False)
    }
    matched = {key: set() for key in gt_groups}
    records = []
    for row in selected.itertuples(index=False):
        group = gt_groups.get(row.image_key)
        matched_index = None
        matched_iou = 0.0
        if group is not None:
            available = [index for index in range(len(group)) if index not in matched[row.image_key]]
            if available:
                boxes = group.loc[available, ["x1", "y1", "x2", "y2"]].to_numpy(float)
                ious = box_iou(np.array([row.x1, row.y1, row.x2, row.y2]), boxes)
                best = int(np.argmax(ious))
                if float(ious[best]) >= iou_threshold:
                    matched_index = available[best]
                    matched_iou = float(ious[best])
                    matched[row.image_key].add(matched_index)
        record = row._asdict()
        record.update({
            "is_tp": matched_index is not None,
            "matched_gt_index": math.nan if matched_index is None else int(matched_index),
            "matched_iou": matched_iou,
        })
        records.append(record)
    if not records:
        empty = selected.iloc[0:0].copy()
        empty["is_tp"] = pd.Series(dtype=bool)
        empty["matched_gt_index"] = pd.Series(dtype=float)
        empty["matched_iou"] = pd.Series(dtype=float)
        return empty
    return pd.DataFrame(records)


def froc_curve(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.30,
) -> pd.DataFrame:
    """在所有实际候选置信度上计算非插值 FROC 曲线。"""
    total_images = len(images)
    total_negative_images = int(images.label.eq(0).sum())
    total_lesions = len(ground_truth)
    ordered = predictions.sort_values(
        ["confidence", "image_key", "prediction_index"],
        ascending=[False, True, True],
    )
    gt_groups = {
        key: group.reset_index(drop=True)
        for key, group in ground_truth.groupby("image_key", sort=False)
    }
    matched_gt = {key: set() for key in gt_groups}
    image_labels = images.set_index("image_key").label.to_dict()
    tp = fp = negative_fp = 0

    def record(threshold: float) -> dict:
        return {
            "threshold": threshold,
            "tp_lesions": tp,
            "fn_lesions": total_lesions - tp,
            "fp_boxes": fp,
            "lesion_sensitivity": tp / total_lesions if total_lesions else math.nan,
            "fp_per_image": fp / total_images,
            "negative_fp_per_image": (
                negative_fp / total_negative_images if total_negative_images else math.nan
            ),
        }

    records = [record(math.inf)]
    for threshold, threshold_group in ordered.groupby("confidence", sort=False):
        for row in threshold_group.itertuples(index=False):
            group = gt_groups.get(row.image_key)
            found = False
            if group is not None:
                available = [
                    index for index in range(len(group))
                    if index not in matched_gt[row.image_key]
                ]
                if available:
                    boxes = group.loc[available, ["x1", "y1", "x2", "y2"]].to_numpy(float)
                    ious = box_iou(
                        np.array([row.x1, row.y1, row.x2, row.y2]), boxes
                    )
                    best = int(np.argmax(ious))
                    if float(ious[best]) >= iou_threshold:
                        matched_gt[row.image_key].add(available[best])
                        found = True
            if found:
                tp += 1
            else:
                fp += 1
                if int(image_labels[row.image_key]) == 0:
                    negative_fp += 1
        records.append(record(float(threshold)))
    return pd.DataFrame(records)


def primary_froc_point(curve: pd.DataFrame, fp_limit: float = 0.5) -> dict:
    """返回 FP/image 不超过上限时实际可达的最高病灶敏感度点。"""
    eligible = curve[curve.fp_per_image.le(fp_limit)].sort_values(
        ["lesion_sensitivity", "fp_per_image", "threshold"],
        ascending=[False, True, False],
    )
    return eligible.iloc[0].to_dict()


def interpolated_froc(curve: pd.DataFrame, targets: tuple[float, ...]) -> dict[str, float]:
    """对单调 FROC 包络线做线性插值，仅作辅助报告。"""
    envelope = (
        curve.groupby("fp_per_image", as_index=False).lesion_sensitivity.max()
        .sort_values("fp_per_image")
    )
    envelope["lesion_sensitivity"] = np.maximum.accumulate(envelope.lesion_sensitivity)
    maximum = float(envelope.fp_per_image.max())
    return {
        f"sensitivity_at_fp_per_image_{target:g}": (
            float(np.interp(target, envelope.fp_per_image, envelope.lesion_sensitivity))
            if target <= maximum else math.nan
        )
        for target in targets
    }


def average_precision(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float,
) -> float:
    """使用 101 点插值计算单类检测 AP。"""
    matched = match_predictions(predictions, ground_truth, -math.inf, iou_threshold)
    if len(ground_truth) == 0 or len(matched) == 0:
        return 0.0
    true_positive = matched.is_tp.to_numpy(float)
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(1.0 - true_positive)
    recall = cumulative_tp / len(ground_truth)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(precision[recall >= point].max()) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def geometry_summary(
    ground_truth: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict:
    """将未检出病灶按几何得分 0 纳入整体定位质量。"""
    ious, coverages, center_hits = [], [], []
    tp = matches[matches.is_tp] if len(matches) else matches
    for image_key, group in ground_truth.groupby("image_key", sort=False):
        image_matches = tp[tp.image_key.eq(image_key)] if len(tp) else tp
        matched_indices = {
            int(row.matched_gt_index): row for row in image_matches.itertuples(index=False)
        }
        for index, gt in group.reset_index(drop=True).iterrows():
            row = matched_indices.get(index)
            if row is None:
                ious.append(0.0)
                coverages.append(0.0)
                center_hits.append(0.0)
                continue
            pred = np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)
            truth = gt[["x1", "y1", "x2", "y2"]].to_numpy(float)
            intersection_size = np.maximum(0.0, np.minimum(pred[2:], truth[2:]) - np.maximum(pred[:2], truth[:2]))
            intersection = float(intersection_size[0] * intersection_size[1])
            gt_area = float((truth[2] - truth[0]) * (truth[3] - truth[1]))
            center = (pred[:2] + pred[2:]) / 2
            ious.append(float(row.matched_iou))
            coverages.append(intersection / gt_area if gt_area > 0 else 0.0)
            center_hits.append(float(np.all(center >= truth[:2]) and np.all(center <= truth[2:])))
    return {
        "mean_iou_all_lesions": float(np.mean(ious)),
        "median_iou_all_lesions": float(np.median(ious)),
        "iou_ge_0p5_all_lesions": float(np.mean(np.asarray(ious) >= 0.5)),
        "mean_lesion_coverage_all_lesions": float(np.mean(coverages)),
        "center_hit_rate_all_lesions": float(np.mean(center_hits)),
    }


def summarize_detection(
    images: pd.DataFrame,
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    deployment_threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """统一产生 CD0 主指标、FROC 和冻结部署点匹配表。"""
    curve = froc_curve(images, predictions, ground_truth, iou_threshold=0.30)
    deployed = match_predictions(predictions, ground_truth, deployment_threshold, 0.30)
    tp = int(deployed.is_tp.sum()) if len(deployed) else 0
    fp = int((~deployed.is_tp).sum()) if len(deployed) else 0
    negative_fp = 0
    if len(deployed):
        labelled = deployed.merge(images[["image_key", "label"]], on="image_key", how="left")
        negative_fp = int((labelled.label.eq(0) & ~labelled.is_tp).sum())
    aps = [average_precision(images, predictions, ground_truth, threshold) for threshold in np.arange(0.5, 1.0, 0.05)]
    summary = {
        "images": int(len(images)),
        "patients": int(images.patient_id.nunique()),
        "lesions": int(len(ground_truth)),
        "deployment_threshold": float(deployment_threshold),
        "deployment": {
            "lesion_sensitivity": tp / len(ground_truth),
            "fp_per_image": fp / len(images),
            "negative_fp_per_image": negative_fp / int(images.label.eq(0).sum()),
            **geometry_summary(ground_truth, deployed),
        },
        "primary_froc_at_most_0p5_fp_per_image": primary_froc_point(curve, 0.5),
        "interpolated_froc_auxiliary": interpolated_froc(curve, (0.1, 0.25, 0.5, 1.0)),
        "ap50": aps[0],
        "map50_95": float(np.mean(aps)),
    }
    return summary, curve, deployed
