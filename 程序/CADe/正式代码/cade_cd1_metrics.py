#!/usr/bin/env python3
"""CD1 Gray Qualification 的错误互补、决策与配对统计函数。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cade_cd0_metrics import box_iou, match_predictions


def detection_errors(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    threshold: float,
    iou_threshold: float = 0.30,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]], pd.DataFrame]:
    """返回冻结工作点的已检出GT、漏检GT和未匹配预测框。"""
    matches = match_predictions(predictions, ground_truth, threshold, iou_threshold)
    gt_groups = {
        key: group.reset_index(drop=True)
        for key, group in ground_truth.groupby("image_key", sort=False)
    }
    detected: set[tuple[str, int]] = set()
    for row in matches[matches.is_tp].itertuples(index=False):
        local_index = int(row.matched_gt_index)
        gt_index = int(gt_groups[row.image_key].iloc[local_index].gt_index)
        detected.add((str(row.image_key), gt_index))
    all_instances = {
        (str(row.image_key), int(row.gt_index))
        for row in ground_truth.itertuples(index=False)
    }
    false_positives = matches[~matches.is_tp].copy().reset_index(drop=True)
    return detected, all_instances - detected, false_positives


def fn_complementarity(
    rgb_predictions: pd.DataFrame,
    gray_predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rgb_threshold: float,
    gray_threshold: float,
) -> dict:
    """计算两个检测器在各自冻结工作点的FN互补关系。"""
    rgb_detected, rgb_fn, _ = detection_errors(
        rgb_predictions, ground_truth, rgb_threshold
    )
    gray_detected, gray_fn, _ = detection_errors(
        gray_predictions, ground_truth, gray_threshold
    )
    shared_fn = rgb_fn & gray_fn
    fn_union = rgb_fn | gray_fn
    return {
        "rgb_fn": len(rgb_fn),
        "gray_fn": len(gray_fn),
        "gray_rescued_rgb_fn": len(rgb_fn & gray_detected),
        "rgb_rescued_gray_fn": len(gray_fn & rgb_detected),
        "gray_rescue": len(rgb_fn & gray_detected) / len(rgb_fn) if rgb_fn else math.nan,
        "rgb_reverse_rescue": (
            len(gray_fn & rgb_detected) / len(gray_fn) if gray_fn else math.nan
        ),
        "shared_fn": len(shared_fn),
        "fn_union": len(fn_union),
        "fn_jaccard": len(shared_fn) / len(fn_union) if fn_union else math.nan,
    }


def maximum_cardinality_iou_pairs(
    iou_matrix: np.ndarray,
    iou_threshold: float = 0.30,
) -> list[tuple[int, int, float]]:
    """先最大化匹配数量，再在同基数解中最大化总IoU。"""
    matrix = np.asarray(iou_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("IoU矩阵必须是二维数组")
    rgb_count, gray_count = matrix.shape
    if rgb_count == 0 or gray_count == 0:
        return []

    size = rgb_count + gray_count
    cardinality_bonus = min(rgb_count, gray_count) + 1.0
    forbidden = -cardinality_bonus * (size + 1)
    weights = np.full((size, size), forbidden, dtype=float)
    valid = matrix >= iou_threshold
    weights[:rgb_count, :gray_count][valid] = cardinality_bonus + matrix[valid]

    # 每个真实框都有自己的“不匹配”槽位；右下角用于补齐完整指派。
    for rgb_index in range(rgb_count):
        weights[rgb_index, gray_count + rgb_index] = 0.0
    for gray_index in range(gray_count):
        weights[rgb_count + gray_index, gray_index] = 0.0
    weights[rgb_count:, gray_count:] = 0.0

    rows, columns = linear_sum_assignment(weights, maximize=True)
    pairs = []
    for row, column in zip(rows, columns):
        if row < rgb_count and column < gray_count and valid[row, column]:
            pairs.append((int(row), int(column), float(matrix[row, column])))
    return pairs


def fp_overlap(
    rgb_false_positives: pd.DataFrame,
    gray_false_positives: pd.DataFrame,
    iou_threshold: float = 0.30,
) -> dict:
    """按图进行严格最大基数二分图匹配并汇总FP重合。"""
    image_keys = sorted(
        set(rgb_false_positives.image_key.astype(str))
        | set(gray_false_positives.image_key.astype(str))
    )
    matched = 0
    matched_iou = 0.0
    for image_key in image_keys:
        rgb = rgb_false_positives[
            rgb_false_positives.image_key.astype(str).eq(image_key)
        ][["x1", "y1", "x2", "y2"]].to_numpy(float)
        gray = gray_false_positives[
            gray_false_positives.image_key.astype(str).eq(image_key)
        ][["x1", "y1", "x2", "y2"]].to_numpy(float)
        if len(rgb) == 0 or len(gray) == 0:
            continue
        ious = np.vstack([box_iou(box, gray) for box in rgb])
        pairs = maximum_cardinality_iou_pairs(ious, iou_threshold)
        matched += len(pairs)
        matched_iou += sum(pair[2] for pair in pairs)

    rgb_count = len(rgb_false_positives)
    gray_count = len(gray_false_positives)
    union = rgb_count + gray_count - matched
    return {
        "rgb_fp": rgb_count,
        "gray_fp": gray_count,
        "matched_fp": matched,
        "rgb_unique_fp": rgb_count - matched,
        "gray_unique_fp": gray_count - matched,
        "matched_total_iou": matched_iou,
        "fp_jaccard": matched / union if union else math.nan,
    }


def classify_cd1(seed_metrics: list[dict]) -> dict:
    """按冻结顺序将三seed Development Val结果归入四种结论。"""
    if len(seed_metrics) != 3:
        raise ValueError("CD1正式判定必须恰好包含三个seed")
    deltas = np.asarray([row["primary_delta"] for row in seed_metrics], dtype=float)
    rescues = np.asarray([row["gray_rescue"] for row in seed_metrics], dtype=float)
    jaccards = np.asarray([row["fn_jaccard"] for row in seed_metrics], dtype=float)
    if not np.isfinite(np.concatenate([deltas, rescues, jaccards])).all():
        raise ValueError("CD1正式判定指标必须全部为有限数")

    pass_checks = {
        "all_seed_primary_delta_ge_minus_1_over_240": bool(
            np.all(deltas >= -(1.0 / 240.0))
        ),
        "mean_primary_delta_ge_0p01": bool(np.mean(deltas) >= 0.01),
    }
    complementary_checks = {
        "mean_primary_delta_ge_minus_0p02": bool(np.mean(deltas) >= -0.02),
        "all_seed_primary_delta_ge_minus_0p03": bool(np.all(deltas >= -0.03)),
        "mean_gray_rescue_ge_0p20": bool(np.mean(rescues) >= 0.20),
        "at_least_two_gray_rescue_ge_0p20": bool(np.sum(rescues >= 0.20) >= 2),
        "mean_fn_jaccard_le_0p75": bool(np.mean(jaccards) <= 0.75),
        "at_least_two_fn_jaccard_le_0p75": bool(np.sum(jaccards <= 0.75) >= 2),
    }
    fail_checks = {
        "mean_primary_delta_le_minus_0p02": bool(np.mean(deltas) <= -0.02),
        "at_least_two_primary_delta_le_minus_0p02": bool(
            np.sum(deltas <= -0.02) >= 2
        ),
        "mean_gray_rescue_lt_0p15": bool(np.mean(rescues) < 0.15),
        "mean_fn_jaccard_gt_0p80": bool(np.mean(jaccards) > 0.80),
    }
    if all(pass_checks.values()):
        decision = "PASS"
    elif all(complementary_checks.values()):
        decision = "COMPLEMENTARY_PASS"
    elif all(fail_checks.values()):
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "mean_primary_delta": float(np.mean(deltas)),
        "mean_gray_rescue": float(np.mean(rescues)),
        "mean_fn_jaccard": float(np.mean(jaccards)),
        "pass_checks": pass_checks,
        "complementary_checks": complementary_checks,
        "fail_checks": fail_checks,
        "seed_metrics": seed_metrics,
    }


def paired_patient_bootstrap(
    images: pd.DataFrame,
    ground_truth: pd.DataFrame,
    rgb_predictions: pd.DataFrame,
    gray_predictions: pd.DataFrame,
    rgb_threshold: float,
    gray_threshold: float,
    repeats: int = 5000,
    seed: int = 20260825,
) -> pd.DataFrame:
    """在固定工作阈值下按患者整簇有放回估计配对Sensitivity差。"""
    rgb_detected, _, rgb_fp = detection_errors(
        rgb_predictions, ground_truth, rgb_threshold
    )
    gray_detected, _, gray_fp = detection_errors(
        gray_predictions, ground_truth, gray_threshold
    )
    patient_ids = images.patient_id.astype(str).drop_duplicates().to_numpy()
    per_patient = {}
    for patient_id in patient_ids:
        keys = set(images.loc[images.patient_id.astype(str).eq(patient_id), "image_key"])
        gt_keys = {
            (str(row.image_key), int(row.gt_index))
            for row in ground_truth[ground_truth.image_key.isin(keys)].itertuples(index=False)
        }
        per_patient[patient_id] = {
            "images": len(keys),
            "gt": len(gt_keys),
            "rgb_tp": len(gt_keys & rgb_detected),
            "gray_tp": len(gt_keys & gray_detected),
            "rgb_fp": int(rgb_fp.image_key.isin(keys).sum()),
            "gray_fp": int(gray_fp.image_key.isin(keys).sum()),
        }

    rng = np.random.default_rng(seed)
    records = []
    for repeat in range(repeats):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        totals = {
            key: sum(per_patient[patient_id][key] for patient_id in sampled)
            for key in ("images", "gt", "rgb_tp", "gray_tp", "rgb_fp", "gray_fp")
        }
        records.append({
            "repeat": repeat,
            "rgb_sensitivity": totals["rgb_tp"] / totals["gt"],
            "gray_sensitivity": totals["gray_tp"] / totals["gt"],
            "sensitivity_delta": (
                totals["gray_tp"] - totals["rgb_tp"]
            ) / totals["gt"],
            "rgb_fp_per_image": totals["rgb_fp"] / totals["images"],
            "gray_fp_per_image": totals["gray_fp"] / totals["images"],
        })
    return pd.DataFrame(records)
