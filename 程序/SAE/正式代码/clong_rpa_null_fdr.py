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
import numpy as np

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
    feature_ids: np.ndarray,
    bins_per_dimension: int = 4,
) -> np.ndarray:
    """确定性构造患者覆盖率×激活频率的平衡层。

    先按``(coverage, frequency, feature_id)``稳定排序并等频分成4组，再在
    每个覆盖率组内按``(frequency, coverage, feature_id)``等频分4组。返回
    ``coverage_bin * 4 + frequency_bin``。该层只使用目标seed的train eligible
    Feature，不读取匹配指标或val；val沿用train分层。
    """
    coverage = np.asarray(patient_coverage, dtype=np.float64)
    frequency = np.asarray(active_frequency, dtype=np.float64)
    ids = np.asarray(feature_ids, dtype=np.int64)
    if not (coverage.shape == frequency.shape == ids.shape) or coverage.ndim != 1:
        raise ValueError("coverage/frequency/feature_ids必须是同shape一维数组")
    if not np.isfinite(coverage).all() or not np.isfinite(frequency).all():
        raise ValueError("strata输入不得包含NaN/Inf")
    if bins_per_dimension < 2 or coverage.size < bins_per_dimension ** 2:
        raise ValueError("strata分箱数无效或Feature不足")
    if np.unique(ids).size != ids.size:
        raise ValueError("feature_ids必须唯一")

    strata = np.empty(coverage.size, dtype=np.int16)
    coverage_order = np.lexsort((ids, frequency, coverage))
    for coverage_bin, coverage_indices in enumerate(
        np.array_split(coverage_order, bins_per_dimension)
    ):
        local_order = np.lexsort((
            ids[coverage_indices], coverage[coverage_indices], frequency[coverage_indices]
        ))
        frequency_order = coverage_indices[local_order]
        for frequency_bin, indices in enumerate(
            np.array_split(frequency_order, bins_per_dimension)
        ):
            strata[indices] = coverage_bin * bins_per_dimension + frequency_bin
    return strata


def validate_target_strata(
    strata: np.ndarray,
    minimum_size: int = 32,
    expected_strata: int = 16,
    minimum_total: int = 512,
) -> dict[int, int]:
    """检查目标strata完整性、总规模和每层最小规模，不自动合并。"""
    strata = np.asarray(strata)
    if strata.ndim != 1 or strata.size < minimum_total:
        raise RuntimeError(
            f"null_stratification_infeasible: total={strata.size}, minimum={minimum_total}"
        )
    if not np.issubdtype(strata.dtype, np.integer):
        raise ValueError("strata必须是整数编号")
    values, counts = np.unique(strata, return_counts=True)
    result = {int(v): int(n) for v, n in zip(values, counts)}
    expected = set(range(expected_strata))
    actual = set(result)
    if actual != expected:
        raise RuntimeError(
            "null_stratification_infeasible: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
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
    order = np.argsort(selected, kind="mergesort")
    sorted_values = selected[order]
    sorted_ranks = np.empty(selected.size, dtype=np.float64)
    start = 0
    while start < selected.size:
        end = start + 1
        while end < selected.size and sorted_values[end] == sorted_values[start]:
            end += 1
        # 1-based ranks start+1...end have average (start+1+end)/2.
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(selected.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    output[mask] = (ranks - 0.5) / selected.size
    return output


def frozen_train_cdf_percentile(
    train_values: np.ndarray,
    val_values: np.ndarray,
    train_valid: np.ndarray,
    val_valid: np.ndarray,
) -> np.ndarray:
    """将val指标映射到冻结train empirical CDF。

    对每个val值``x``返回``(#train<x + 0.5*#train==x)/N_train``；不在val
    重排、重估CDF或改变Feature universe。
    """
    train = np.asarray(train_values, dtype=np.float64)
    val = np.asarray(val_values, dtype=np.float64)
    train_mask = np.asarray(train_valid, dtype=bool)
    val_mask = np.asarray(val_valid, dtype=bool)
    if not (
        train.shape == val.shape == train_mask.shape == val_mask.shape
    ) or train.ndim != 1:
        raise ValueError("train/val/train_valid/val_valid必须是同shape一维数组")
    reference = np.sort(train[train_mask])
    output = np.full(val.shape, np.nan, dtype=np.float64)
    if reference.size == 0:
        return output
    if not np.isfinite(reference).all():
        raise ValueError("train_valid不得包含NaN/Inf")
    query = val[val_mask]
    if not np.isfinite(query).all():
        raise ValueError("val_valid不得包含NaN/Inf")
    left = np.searchsorted(reference, query, side="left")
    right = np.searchsorted(reference, query, side="right")
    output[val_mask] = (left + 0.5 * (right - left)) / reference.size
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
    by_source: dict[tuple[int, int, int], int] = {}
    for h in hypotheses:
        source_key = (h.source_seed, h.target_seed, h.source_feature_id)
        if source_key in by_source:
            raise RuntimeError(f"同一有向source出现多个best-candidate假设: {source_key}")
        by_source[source_key] = h.target_feature_id
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
