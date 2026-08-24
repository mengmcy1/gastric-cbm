#!/usr/bin/env python3
"""RP-A development bootstrap的确定性纯函数核心。

本模块不读取正式患者资产、不运行matching也不生成门槛。
它只实现拟冻结的分层有放回计划、重复患者等价加权、
Top-25 instance语义和最终结果归并。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from clong_s2b_core import ACTIVE_EPS


SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR / "rpa_bootstrap_protocol_v1.json"
STRATA_ORDER = (
    (0, "武大省人民"),
    (0, "第一届早癌大赛"),
    (0, "第二届早癌大赛"),
    (1, "武大省人民"),
    (1, "第一届早癌大赛"),
    (1, "第二届早癌大赛"),
)
FORMAL_STRATUM_COUNTS = (830, 32, 38, 261, 19, 32)
BOOTSTRAP_COUNT = 400
BOOTSTRAP_SEED = 20260824
TOP_PATIENT_COUNT = 25
P_MIN = 25
A_MIN = 0.25 / 49.0
METRIC_NAMES = (
    "R_anchor_recall",
    "R_confirm_coverage",
    "R_activation_reference",
    "R_activation_confirmation",
    "R_energy_reference",
    "R_energy_confirmation",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """读取bootstrap拟冻结协议。"""
    return json.loads(path.read_text(encoding="utf-8"))


def activation_presence(activation: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """按冻结``ACTIVE_EPS``返回Feature presence，禁止简化为``>0``。"""
    values = np.asarray(activation)
    return np.any(values > ACTIVE_EPS, axis=axis)


def validate_formal_patient_table(
    patient_ids: np.ndarray, labels: np.ndarray, sources: np.ndarray
) -> tuple[int, ...]:
    """核对正式train患者表的冻结六层数量。"""
    ids = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(labels)
    sources = np.asarray(sources, dtype=str)
    if not (ids.shape == labels.shape == sources.shape) or ids.ndim != 1:
        raise ValueError("正式患者表三列shape不一致")
    if np.unique(ids).size != ids.size:
        raise ValueError("正式患者表patient_id必须唯一")
    counts = tuple(
        int(np.sum((labels == label) & (sources == source)))
        for label, source in STRATA_ORDER
    )
    if counts != FORMAL_STRATUM_COUNTS or sum(counts) != ids.size:
        raise RuntimeError(f"正式bootstrap六层患者数不一致: {counts}")
    return counts


def generate_bootstrap_plans(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    replicate_count: int = BOOTSTRAP_COUNT,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """按冻结strata顺序生成``[replicate, patient]`` multiplicity。"""
    patient_ids = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(labels)
    sources = np.asarray(sources, dtype=str)
    if not (patient_ids.shape == labels.shape == sources.shape) or patient_ids.ndim != 1:
        raise ValueError("patient_ids/labels/sources必须是同shape一维数组")
    if np.unique(patient_ids).size != patient_ids.size:
        raise ValueError("patient_id必须一行一患者且唯一")
    strata: list[np.ndarray] = []
    for label, source in STRATA_ORDER:
        indices = np.flatnonzero((labels == label) & (sources == source))
        indices = indices[np.argsort(patient_ids[indices], kind="stable")]
        if indices.size == 0:
            raise ValueError(f"冻结stratum为空: {(label, source)}")
        strata.append(indices)
    if sum(len(indices) for indices in strata) != patient_ids.size:
        raise ValueError("患者包含冻结六层以外的label/source")

    rng = np.random.Generator(np.random.PCG64(seed))
    plans = np.zeros((replicate_count, patient_ids.size), dtype=np.int32)
    for replicate in range(replicate_count):
        for indices in strata:
            draws = rng.integers(0, len(indices), size=len(indices))
            np.add.at(plans[replicate], indices[draws], 1)
    return plans


def eligible_membership(
    presence: np.ndarray,
    active_frequency: np.ndarray,
    multiplicity: np.ndarray,
    full_train_nondead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """重算一个replicate的positive instance数、活跃频率和eligible。"""
    presence = np.asarray(presence, dtype=bool)
    frequency = np.asarray(active_frequency, dtype=np.float64)
    weights = np.asarray(multiplicity, dtype=np.int64)
    nondead = np.asarray(full_train_nondead, dtype=bool)
    if presence.shape != frequency.shape or presence.shape[0] != weights.size:
        raise ValueError("presence/frequency/multiplicity shape不一致")
    if nondead.shape != (presence.shape[1],):
        raise ValueError("full_train_nondead shape不一致")
    if np.any(weights < 0) or weights.sum() == 0:
        raise ValueError("multiplicity必须为非负且总数大于0")
    positive_count = weights @ presence.astype(np.int64)
    weighted_frequency = weights @ frequency / weights.sum()
    eligible = nondead & (positive_count >= P_MIN) & (weighted_frequency >= A_MIN)
    return positive_count, weighted_frequency, eligible


def top_patient_instances(
    ranking_score: np.ndarray,
    patient_ids: np.ndarray,
    multiplicity: np.ndarray,
    count: int = TOP_PATIENT_COUNT,
) -> list[tuple[str, int]]:
    """按score、patient ID、occurrence index取重复患者Top instances。"""
    scores = np.asarray(ranking_score, dtype=np.float64)
    ids = np.asarray(patient_ids, dtype=str)
    weights = np.asarray(multiplicity, dtype=np.int64)
    if not (scores.shape == ids.shape == weights.shape) or scores.ndim != 1:
        raise ValueError("ranking_score/patient_ids/multiplicity shape不一致")
    instances = [
        (float(scores[index]), str(ids[index]), occurrence)
        for index in range(ids.size)
        for occurrence in range(int(weights[index]))
    ]
    instances.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(patient_id, occurrence) for _, patient_id, occurrence in instances[:count]]


def weighted_sum(values: np.ndarray, multiplicity: np.ndarray) -> np.ndarray:
    """返回与显式复制患者instance等价的逐Feature加权和。"""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(multiplicity, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != weights.size:
        raise ValueError("values必须为[patient,feature]且与multiplicity对齐")
    return weights @ values


def weighted_mean(values: np.ndarray, multiplicity: np.ndarray) -> np.ndarray:
    """返回与显式复制患者instance等价的逐Feature均值。"""
    weights = np.asarray(multiplicity, dtype=np.int64)
    return weighted_sum(values, weights) / weights.sum()


def weighted_union_positive_spearman(
    source_score: np.ndarray,
    target_score: np.ndarray,
    multiplicity: np.ndarray,
) -> float:
    """按重复患者instance计算union-positive Spearman。"""
    source = np.asarray(source_score, dtype=np.float64)
    target = np.asarray(target_score, dtype=np.float64)
    weights = np.asarray(multiplicity, dtype=np.int64)
    if not (source.shape == target.shape == weights.shape) or source.ndim != 1:
        raise ValueError("Spearman输入shape不一致")
    expanded_source = np.repeat(source, weights)
    expanded_target = np.repeat(target, weights)
    union = (expanded_source > ACTIVE_EPS) | (expanded_target > ACTIVE_EPS)
    if union.sum() < 2:
        return float("nan")
    source_rank = rankdata(expanded_source[union], method="average")
    target_rank = rankdata(expanded_target[union], method="average")
    if np.ptp(source_rank) == 0 or np.ptp(target_rank) == 0:
        return float("nan")
    return float(np.corrcoef(source_rank, target_rank)[0, 1])


def weighted_spatial_patient_mean(
    patient_pair_similarity: np.ndarray,
    patient_valid: np.ndarray,
    multiplicity: np.ndarray,
) -> float:
    """对可评价患者做multiplicity加权的spatial均值。"""
    similarity = np.asarray(patient_pair_similarity, dtype=np.float64)
    valid = np.asarray(patient_valid, dtype=bool)
    weights = np.asarray(multiplicity, dtype=np.int64)
    if not (similarity.shape == valid.shape == weights.shape) or similarity.ndim != 1:
        raise ValueError("spatial输入shape不一致")
    effective = weights * valid
    if effective.sum() == 0:
        return float("nan")
    if not np.isfinite(similarity[valid & (weights > 0)]).all():
        raise ValueError("有效spatial值不得包含NaN/Inf")
    return float(np.sum(effective * np.where(valid, similarity, 0.0)) / effective.sum())


def worker_replicate_indices(worker_rank: int, worker_count: int) -> np.ndarray:
    """返回与GPU数量无关的静态replicate分配。"""
    if not 0 <= worker_rank < worker_count:
        raise ValueError("worker_rank必须在[0,worker_count)")
    return np.arange(worker_rank, BOOTSTRAP_COUNT, worker_count, dtype=np.int32)


def structural_failure_record(replicate_index: int, reason: str) -> dict:
    """构造已冻结null-strata结构失败的全零记录。"""
    if reason != "null_stratification_infeasible":
        raise ValueError("只有null_stratification_infeasible属于冻结结构失败")
    return {
        "replicate_index": int(replicate_index),
        "status": "replicate_structural_failure",
        "reason": str(reason),
        "fold_metrics": {
            str(fold): {name: 0.0 for name in METRIC_NAMES}
            for fold in (42, 43, 44)
        },
    }


def aggregate_thresholds(records: list[dict]) -> dict[str, float]:
    """校验400条记录并按最弱折Q0.05/lower生成六个门槛。"""
    indices = [int(record["replicate_index"]) for record in records]
    if sorted(indices) != list(range(BOOTSTRAP_COUNT)) or len(set(indices)) != len(indices):
        raise RuntimeError("replicate index必须完整且唯一覆盖0...399")
    ordered = sorted(records, key=lambda record: int(record["replicate_index"]))
    for record in ordered:
        status = record.get("status")
        if status not in {"completed", "replicate_structural_failure"}:
            raise RuntimeError(f"非法replicate status: {status}")
        if status == "replicate_structural_failure" and any(
            float(record["fold_metrics"][str(fold)][name]) != 0.0
            for fold in (42, 43, 44) for name in METRIC_NAMES
        ):
            raise RuntimeError("结构失败replicate的三折六指标必须全为0")
    thresholds: dict[str, float] = {}
    for name in METRIC_NAMES:
        weakest = np.asarray([
            min(float(record["fold_metrics"][str(fold)][name]) for fold in (42, 43, 44))
            for record in ordered
        ])
        if not np.isfinite(weakest).all():
            raise ValueError("bootstrap正式指标不得包含NaN/Inf")
        threshold = float(np.quantile(weakest, 0.05, method="lower"))
        if threshold <= 0:
            raise RuntimeError("bootstrap_calibration_infeasible: nonpositive threshold")
        thresholds[name] = threshold
    return thresholds


def verify_full_train_self_consistency(
    thresholds: dict[str, float], full_fold_metrics: dict[str, dict[str, float]]
) -> None:
    """检查未重采样full train的三折六指标均达到门槛。"""
    for fold in (42, 43, 44):
        for name in METRIC_NAMES:
            observed = float(full_fold_metrics[str(fold)][name])
            threshold = float(thresholds[name])
            if not np.isfinite(observed) or not np.isfinite(threshold):
                raise ValueError("full-train self-consistency输入不得包含NaN/Inf")
            if observed < threshold:
                raise RuntimeError("bootstrap_calibration_infeasible: full-train self-consistency")
