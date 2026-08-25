#!/usr/bin/env python3
"""RP-C2中间剂量、matched-reference与fixed-attention纯函数。"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata


ALPHAS = np.asarray([1.0, 0.75, 0.50, 0.25, 0.0], dtype=np.float64)
INTERMEDIATE_INDICES = np.asarray([1, 2, 3], dtype=np.int64)


def direction_code(values: np.ndarray) -> tuple[str, int, str]:
    """用三seed中位数冻结方向，并返回同向seed数和原始符号串。"""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("direction values必须是三个有限seed值")
    signs = np.where(array > 0, "positive", np.where(array < 0, "negative", "zero"))
    median = float(np.median(array))
    expected = "positive" if median > 0 else "negative" if median < 0 else "zero"
    return expected, int(np.count_nonzero(signs == expected)), ";".join(signs.tolist())


def intermediate_curve(dose_values: np.ndarray) -> np.ndarray:
    """只平均alpha=.75/.50/.25三个真正新增剂量。"""
    array = np.asarray(dose_values, dtype=np.float64)
    if array.shape[-1] != 5 or not np.isfinite(array).all():
        raise ValueError("dose values末维必须对应五档有限数值")
    return array[..., INTERMEDIATE_INDICES].mean(axis=-1)


def aligned_dose_spearman(dose_values: np.ndarray, expected_direction: str) -> float | None:
    """计算五档方向对齐Spearman；零方向或常数曲线返回None。"""
    if expected_direction == "zero":
        return None
    if expected_direction not in {"positive", "negative"}:
        raise ValueError("expected_direction非法")
    values = np.asarray(dose_values, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("dose curve必须为五个有限值")
    aligned = values * (1.0 if expected_direction == "positive" else -1.0)
    if np.ptp(aligned) == 0:
        return None
    removal = 1.0 - ALPHAS
    return float(np.corrcoef(rankdata(removal), rankdata(aligned))[0, 1])


def matched_reference_summary(target: float, controls: np.ndarray) -> dict[str, object]:
    """汇总确定性matched reference位置，不把tail fraction称为p值。"""
    reference = np.asarray(controls, dtype=np.float64)
    if reference.ndim != 1 or len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("matched controls必须是一维非空有限数值")
    value = float(target)
    if not np.isfinite(value) or value < 0 or np.any(reference < 0):
        raise ValueError("overall matched statistic必须非负且有限")
    n_control = len(reference)
    median = float(np.median(reference))
    ratio = value / median if median > 0 else None
    return {
        "n_control": int(n_control),
        "matched_midrank_percentile": float(
            (np.count_nonzero(reference < value) + 0.5 * np.count_nonzero(reference == value))
            / n_control
        ),
        "matched_plus_one_tail_fraction": float(
            (1 + np.count_nonzero(reference >= value)) / (n_control + 1)
        ),
        "effect_ratio_to_control_median": ratio,
        "effect_ratio_status": "ok" if median > 0 else "control_median_zero",
        "control_median": median,
        "control_q90_lower": float(np.quantile(reference, 0.90, method="lower")),
        "control_q95_lower": float(np.quantile(reference, 0.95, method="lower")),
        "percentile_resolution": float(1 / n_control),
        "tail_fraction_resolution": float(1 / (n_control + 1)),
        "conditional_exploratory": True,
        "same_train_selection": True,
    }


@torch.no_grad()
def fixed_attention_delta_margin(
    original_attention: torch.Tensor,
    hidden: torch.Tensor,
    decoder: torch.Tensor,
    margin_weight: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """在冻结原attention下计算多个Feature的margin变化[B,A]。"""
    if original_attention.ndim != 2 or hidden.ndim != 3:
        raise ValueError("attention/hidden必须是[B,49]和[B,49,A]")
    margin_direction = decoder @ margin_weight
    weighted_activation = (original_attention.unsqueeze(2) * hidden).sum(1)
    return (float(alpha) - 1.0) * weighted_activation * margin_direction.unsqueeze(0)
