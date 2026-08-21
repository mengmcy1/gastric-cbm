#!/usr/bin/env python3
"""RP-A跨seed null/FDR的纯函数核心。

本模块不训练SAE、不读取患者数据。它实现拟冻结的目标Feature
strata、train经验百分位、val到冻结train CDF的映射、完整候选搜索的
精确条件置换p值、BH-FDR和reciprocal nearest neighbour边构建。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR / "rpa_null_fdr_protocol_v1.json"


@dataclass(frozen=True)
class DirectedHypothesis:
    """一个有向source Feature的best-candidate假设。"""

    source_seed: int
    target_seed: int
    source_feature_id: int
    target_feature_id: int
    p_value: float


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """读取null/FDR静态协议。"""
    return json.loads(path.read_text(encoding="utf-8"))


def assign_target_strata(
    patient_coverage: np.ndarray,
    active_frequency: np.ndarray,
    coverage_cutpoints: Iterable[float],
    frequency_cutpoints: Iterable[float],
) -> np.ndarray:
    """按冻结绝对切点将Feature分到4×4个目标strata。

    边界值用``searchsorted(..., side='right')``进入较高bin；返回编号
    ``coverage_bin * 4 + frequency_bin``。
    """
    coverage = np.asarray(patient_coverage, dtype=np.float64)
    frequency = np.asarray(active_frequency, dtype=np.float64)
    if coverage.shape != frequency.shape or coverage.ndim != 1:
        raise ValueError("coverage/frequency必须是同shape一维数组")
    if not np.isfinite(coverage).all() or not np.isfinite(frequency).all():
        raise ValueError("strata输入不得包含NaN/Inf")
    c = np.searchsorted(np.asarray(tuple(coverage_cutpoints)), coverage, side="right")
    f = np.searchsorted(np.asarray(tuple(frequency_cutpoints)), frequency, side="right")
    return (4 * c + f).astype(np.int16)


def validate_target_strata(strata: np.ndarray, minimum_size: int = 32) -> dict[int, int]:
    """检查每个非空目标stratum的最小规模，不自动合并。"""
    values, counts = np.unique(np.asarray(strata), return_counts=True)
    result = {int(v): int(n) for v, n in zip(values, counts)}
    small = {v: n for v, n in result.items() if n < minimum_size}
    if small:
        raise RuntimeError(f"null_stratification_infeasible: {small}")
    return result


def train_midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """在单个source行内将train原始指标转为经验中秩百分位。"""
    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if x.shape != mask.shape or x.ndim != 1:
        raise ValueError("values/valid必须是同shape一维数组")
    output = np.full(x.shape, np.nan, dtype=np.float64)
    selected = x[mask]
    if selected.size == 0 or not np.isfinite(selected).all():
        return output
    output[mask] = rankdata(selected, method="average") / selected.size
    return output


def frozen_train_cdf_percentile(
    train_values: np.ndarray, val_values: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """将val指标映射到冻结train empirical CDF。

    对每个val值``x``返回``(#train<x + 0.5*#train==x)/N_train``；不在val
    重排、重估CDF或改变Feature universe。
    """
    train = np.asarray(train_values, dtype=np.float64)
    val = np.asarray(val_values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if train.shape != val.shape or train.shape != mask.shape or train.ndim != 1:
        raise ValueError("train/val/valid必须是同shape一维数组")
    reference = np.sort(train[mask])
    output = np.full(train.shape, np.nan, dtype=np.float64)
    if reference.size == 0 or not np.isfinite(reference).all():
        return output
    query = val[mask]
    left = np.searchsorted(reference, query, side="left")
    right = np.searchsorted(reference, query, side="right")
    output[mask] = (left + 0.5 * (right - left)) / reference.size
    return output


def edge_scores(
    decoder_percentile: np.ndarray,
    spearman_percentile: np.ndarray,
    jaccard_percentile: np.ndarray,
    spatial_percentile: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算``U_behavior=median(3项)``和``S_edge=min(U_decoder,U_behavior)``。"""
    arrays = [np.asarray(x, dtype=np.float64) for x in (
        decoder_percentile, spearman_percentile, jaccard_percentile, spatial_percentile
    )]
    mask = np.asarray(valid, dtype=bool)
    if any(x.shape != mask.shape for x in arrays):
        raise ValueError("百分位数组与valid shape不一致")
    behavior = np.full(mask.shape, np.nan, dtype=np.float64)
    score = np.full(mask.shape, np.nan, dtype=np.float64)
    stacked = np.stack(arrays[1:], axis=0)
    behavior[mask] = np.median(stacked[:, mask], axis=0)
    score[mask] = np.minimum(arrays[0][mask], behavior[mask])
    return behavior, score


def choose_best_candidate(
    edge_score: np.ndarray,
    signed_decoder_cosine: np.ndarray,
    target_feature_ids: np.ndarray,
    valid: np.ndarray,
) -> int | None:
    """按score、raw decoder cosine、target ID的冻结顺序选唯一best target索引。"""
    score = np.asarray(edge_score, dtype=np.float64)
    cosine = np.asarray(signed_decoder_cosine, dtype=np.float64)
    ids = np.asarray(target_feature_ids, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(score) & np.isfinite(cosine)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    ordered = sorted(indices, key=lambda i: (-score[i], -cosine[i], int(ids[i])))
    return int(ordered[0])


def _log_choose(n: int, k: int) -> float:
    """返回log(C(n,k))；非法k返回负无穷。"""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_conditional_search_p(
    decoder_percentile: np.ndarray,
    behavior_score: np.ndarray,
    target_strata: np.ndarray,
    valid: np.ndarray,
    observed_score: float,
) -> float:
    """计算完整候选搜索的精确条件置换右尾p值。

    null在每个target stratum内将整个行为证据块随机挂到decoder候选上。
    对阈值``s``，令``a=#(U_decoder>=s)``、``b=#(U_behavior>=s)``，则该层
    无交集的概率为``C(n-a,b)/C(n,b)``。各层独立置换，乘积后取
    ``1-P(no overlap)``得到``P(max S_edge>=s)``。
    """
    decoder = np.asarray(decoder_percentile, dtype=np.float64)
    behavior = np.asarray(behavior_score, dtype=np.float64)
    strata = np.asarray(target_strata)
    mask = np.asarray(valid, dtype=bool)
    if not np.isfinite(observed_score):
        return 1.0
    if not (decoder.shape == behavior.shape == strata.shape == mask.shape):
        raise ValueError("exact null的输入shape不一致")
    log_no_overlap = 0.0
    any_valid = False
    for stratum in np.unique(strata[mask]):
        local = mask & (strata == stratum)
        d = decoder[local]
        b_values = behavior[local]
        finite = np.isfinite(d) & np.isfinite(b_values)
        d, b_values = d[finite], b_values[finite]
        n = int(d.size)
        if n == 0:
            continue
        any_valid = True
        a = int(np.sum(d >= observed_score))
        b = int(np.sum(b_values >= observed_score))
        if b > n - a:
            return 1.0
        log_no_overlap += _log_choose(n - a, b) - _log_choose(n, b)
    if not any_valid:
        return 1.0
    p_value = -math.expm1(log_no_overlap)
    return float(min(1.0, max(0.0, p_value)))


def raw_direction_gates_pass(
    decoder_cosine: float, spearman: float, jaccard: float, spatial: float
) -> bool:
    """所有原始方向指标必须为有限正值。"""
    values = np.asarray([decoder_cosine, spearman, jaccard, spatial], dtype=float)
    return bool(np.isfinite(values).all() and np.all(values > 0))


def benjamini_hochberg(
    hypotheses: list[DirectedHypothesis], q: float = 0.05
) -> tuple[set[tuple[int, int, int, int]], float | None]:
    """对一个无序seed-pair的双方向best-candidate假设执行BH-FDR。"""
    if not 0 < q < 1:
        raise ValueError("BH q必须在(0,1)")
    if not hypotheses:
        return set(), None
    for item in hypotheses:
        if not math.isfinite(item.p_value) or not 0 <= item.p_value <= 1:
            raise ValueError("p_value必须是[0,1]内有限值")
    ordered = sorted(hypotheses, key=lambda h: (
        h.p_value, h.source_seed, h.source_feature_id, h.target_feature_id
    ))
    m = len(ordered)
    k_max = 0
    for rank, item in enumerate(ordered, start=1):
        if item.p_value <= rank * q / m:
            k_max = rank
    if k_max == 0:
        return set(), None
    cutoff = float(ordered[k_max - 1].p_value)
    rejected = {
        (h.source_seed, h.target_seed, h.source_feature_id, h.target_feature_id)
        for h in hypotheses if h.p_value <= cutoff
    }
    return rejected, cutoff


def reciprocal_edges(
    hypotheses: list[DirectedHypothesis],
    rejected: set[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    """在BH后仅保留双方向均拒绝且best target互为彼此的一一边。"""
    by_source = {
        (h.source_seed, h.target_seed, h.source_feature_id): h.target_feature_id
        for h in hypotheses
    }
    edges: set[tuple[int, int]] = set()
    for h in hypotheses:
        key = (h.source_seed, h.target_seed, h.source_feature_id, h.target_feature_id)
        reverse = (h.target_seed, h.source_seed, h.target_feature_id, h.source_feature_id)
        if key not in rejected or reverse not in rejected:
            continue
        if by_source.get((h.target_seed, h.source_seed, h.target_feature_id)) != h.source_feature_id:
            continue
        if h.source_seed < h.target_seed:
            edges.add((h.source_feature_id, h.target_feature_id))
    left = [a for a, _ in edges]
    right = [b for _, b in edges]
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise RuntimeError("RNN后出现一对多冲突，属于实现错误")
    return sorted(edges)
