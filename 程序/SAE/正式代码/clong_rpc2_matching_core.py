#!/usr/bin/env python3
"""RP-C2 outcome-blind matching geometry 的纯函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd


CALIPERS = (0.025, 0.05, 0.075, 0.10, 0.15)
REQUIRED_CONTROLS = 100


def empirical_midrank(values: np.ndarray) -> np.ndarray:
    """把一维有限数值映射为固定 average-midrank percentile。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("matching covariate必须是一维有限数值")
    ranks = pd.Series(array).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks - 0.5) / len(array)


def patient_equal_covariate(values: np.ndarray, eligible_ids: np.ndarray) -> np.ndarray:
    """对患者等权平均，再按冻结 eligible Feature ID 取值。"""
    matrix = np.asarray(values)
    ids = np.asarray(eligible_ids, dtype=np.int64)
    if matrix.ndim != 2 or len(ids) == 0:
        raise ValueError("患者Feature矩阵或eligible IDs形状错误")
    return matrix[:, ids].mean(axis=0, dtype=np.float64)


def audit_target_geometry(
    target_feature_id: int,
    eligible_ids: np.ndarray,
    percentile_covariates: np.ndarray,
    excluded_feature_ids: set[int],
) -> dict[str, float | int]:
    """返回一个 target 到 outcome-blind control universe 的匹配几何。"""
    ids = np.asarray(eligible_ids, dtype=np.int64)
    covariates = np.asarray(percentile_covariates, dtype=np.float64)
    if covariates.shape != (len(ids), 3):
        raise ValueError("percentile covariates必须为[N_eligible,3]")
    positions = np.flatnonzero(ids == int(target_feature_id))
    if len(positions) != 1:
        raise ValueError("target Feature不在冻结eligible universe中或不唯一")
    keep = ~np.isin(ids, np.fromiter(sorted(excluded_feature_ids), dtype=np.int64))
    control_ids = ids[keep]
    distances = np.max(np.abs(covariates[keep] - covariates[positions[0]]), axis=1)
    order = np.lexsort((control_ids, distances))
    sorted_distances = distances[order]
    record: dict[str, float | int] = {
        "eligible_control_universe_n": int(len(control_ids)),
    }
    for caliper in CALIPERS:
        suffix = f"c{int(round(caliper * 1000)):04d}"
        record[f"pool_n_{suffix}"] = int(np.count_nonzero(distances <= caliper))
    for rank in (1, 25, 50, 100):
        record[f"nearest_{rank}_distance"] = float(sorted_distances[rank - 1])
    return record


def smallest_global_feasible_caliper(frame: pd.DataFrame) -> float | None:
    """按冻结候选顺序返回所有 target-seed 均有100个controls的最小caliper。"""
    for caliper in CALIPERS:
        column = f"pool_n_c{int(round(caliper * 1000)):04d}"
        if frame[column].ge(REQUIRED_CONTROLS).all():
            return float(caliper)
    return None
