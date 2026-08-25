#!/usr/bin/env python3
"""RP-B sharedness、来源审计和技术家族的纯函数。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, kruskal


SEEDS = (42, 43, 44)
ACTIVE_EPS = 1e-8


def file_sha256(path: Path) -> str:
    """计算文件SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    """读取并核验冻结RP-B协议。"""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_2026-08-25":
        raise RuntimeError("RP-B协议尚未冻结")
    return protocol


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg校正，返回单调q值。"""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("BH输入必须是一维有限p值")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    output = np.empty_like(ranked)
    output[order] = ranked
    return output


def stratified_bootstrap_contrasts(
    presence: np.ndarray,
    mass: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """返回癌减非癌覆盖率差和归一化mass对比的bootstrap区间。"""
    cancer = np.flatnonzero(labels == 1)
    noncancer = np.flatnonzero(labels == 0)
    if presence.shape != mass.shape or presence.shape[0] != len(labels):
        raise ValueError("患者矩阵shape不一致")
    if not np.isfinite(presence).all() or not np.isfinite(mass).all():
        raise ValueError("患者统计不得包含NaN/Inf")
    n_features = presence.shape[1]
    coverage_samples = np.empty((replicates, n_features), dtype=np.float32)
    mass_samples = np.empty((replicates, n_features), dtype=np.float32)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        c_draw = rng.choice(cancer, size=(stop - start, len(cancer)), replace=True)
        n_draw = rng.choice(noncancer, size=(stop - start, len(noncancer)), replace=True)
        c_coverage = presence[c_draw].mean(axis=1)
        n_coverage = presence[n_draw].mean(axis=1)
        c_mass = mass[c_draw].mean(axis=1)
        n_mass = mass[n_draw].mean(axis=1)
        denominator = c_mass + n_mass
        coverage_samples[start:stop] = c_coverage - n_coverage
        mass_samples[start:stop] = np.divide(
            c_mass - n_mass,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
    quantiles = (0.025, 0.975)
    return {
        "coverage_low": np.quantile(coverage_samples, quantiles[0], axis=0),
        "coverage_high": np.quantile(coverage_samples, quantiles[1], axis=0),
        "mass_low": np.quantile(mass_samples, quantiles[0], axis=0),
        "mass_high": np.quantile(mass_samples, quantiles[1], axis=0),
    }


def label_source_bootstrap_contrasts(
    presence: np.ndarray,
    mass: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """按label×source保持原层样本数，用于sharedness敏感性诊断。"""
    if presence.shape != mass.shape or presence.shape[0] != len(labels) or len(labels) != len(sources):
        raise ValueError("患者矩阵shape不一致")
    strata = {
        (label, source): np.flatnonzero((labels == label) & (sources == source))
        for label in (0, 1)
        for source in sorted(np.unique(sources).tolist())
    }
    if any(len(indices) == 0 for indices in strata.values()):
        raise ValueError("label×source分层不完整")
    n_features = presence.shape[1]
    coverage_samples = np.empty((replicates, n_features), dtype=np.float32)
    mass_samples = np.empty((replicates, n_features), dtype=np.float32)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        size = stop - start
        draws = {
            key: rng.choice(indices, size=(size, len(indices)), replace=True)
            for key, indices in strata.items()
        }
        cancer_draw = np.concatenate([draws[key] for key in strata if key[0] == 1], axis=1)
        noncancer_draw = np.concatenate([draws[key] for key in strata if key[0] == 0], axis=1)
        cancer_coverage = presence[cancer_draw].mean(axis=1)
        noncancer_coverage = presence[noncancer_draw].mean(axis=1)
        cancer_mass = mass[cancer_draw].mean(axis=1)
        noncancer_mass = mass[noncancer_draw].mean(axis=1)
        denominator = cancer_mass + noncancer_mass
        coverage_samples[start:stop] = cancer_coverage - noncancer_coverage
        mass_samples[start:stop] = np.divide(
            cancer_mass - noncancer_mass,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
    return {
        "coverage_low": np.quantile(coverage_samples, 0.025, axis=0),
        "coverage_high": np.quantile(coverage_samples, 0.975, axis=0),
        "mass_low": np.quantile(mass_samples, 0.025, axis=0),
        "mass_high": np.quantile(mass_samples, 0.975, axis=0),
    }


def fast_delong_auc_ci(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """计算单预测器DeLong AUC及正态近似95%区间。"""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = (
        (positive[:, None] > negative[None, :]).astype(np.float64)
        + 0.5 * (positive[:, None] == negative[None, :])
    )
    v10 = comparisons.mean(axis=1)
    v01 = comparisons.mean(axis=0)
    auc = float(v10.mean())
    variance = float(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))
    standard_error = np.sqrt(max(variance, 0.0))
    return auc, max(0.0, auc - 1.96 * standard_error), min(1.0, auc + 1.96 * standard_error)


def classify_sharedness(seed_rows: list[dict], protocol: dict) -> str:
    """按冻结双指标CI和2/3 seed共识返回五类sharedness。"""
    settings = protocol["sharedness"]
    margin_c = float(settings["coverage_equivalence_margin_absolute"])
    margin_m = float(settings["mass_equivalence_margin_absolute"])
    minimum = int(settings["seed_consensus_minimum"])
    states = []
    for row in seed_rows:
        if row["coverage_ci_low"] > margin_c and row["mass_ci_low"] > margin_m:
            states.append("cancer")
        elif row["coverage_ci_high"] < -margin_c and row["mass_ci_high"] < -margin_m:
            states.append("noncancer")
        elif (
            row["coverage_ci_low"] >= -margin_c
            and row["coverage_ci_high"] <= margin_c
            and row["mass_ci_low"] >= -margin_m
            and row["mass_ci_high"] <= margin_m
        ):
            states.append("equivalent")
        else:
            states.append("mixed")
    if states.count("cancer") >= minimum and "noncancer" not in states:
        return "cancer_enriched"
    if states.count("noncancer") >= minimum and "cancer" not in states:
        return "noncancer_enriched"
    if states.count("equivalent") >= minimum and not ({"cancer", "noncancer"} & set(states)):
        cancer_coverage = float(np.median([row["cancer_coverage"] for row in seed_rows]))
        noncancer_coverage = float(np.median([row["noncancer_coverage"] for row in seed_rows]))
        high = float(settings["shared_high_minimum_coverage_each_label"])
        return "shared_high" if min(cancer_coverage, noncancer_coverage) >= high else "shared_low_rare"
    return "mixed_uncertain"


def diagnose_sharedness(seed_rows: list[dict], protocol: dict) -> tuple[str, str, str]:
    """返回冻结分类、不改变分类的原因码和三seed状态。"""
    settings = protocol["sharedness"]
    margin_c = float(settings["coverage_equivalence_margin_absolute"])
    margin_m = float(settings["mass_equivalence_margin_absolute"])
    states = []
    coverage_states = []
    mass_states = []
    for row in seed_rows:
        if row["coverage_ci_low"] > margin_c:
            coverage_state = "cancer"
        elif row["coverage_ci_high"] < -margin_c:
            coverage_state = "noncancer"
        elif row["coverage_ci_low"] >= -margin_c and row["coverage_ci_high"] <= margin_c:
            coverage_state = "equivalent"
        else:
            coverage_state = "unresolved"
        if row["mass_ci_low"] > margin_m:
            mass_state = "cancer"
        elif row["mass_ci_high"] < -margin_m:
            mass_state = "noncancer"
        elif row["mass_ci_low"] >= -margin_m and row["mass_ci_high"] <= margin_m:
            mass_state = "equivalent"
        else:
            mass_state = "unresolved"
        coverage_states.append(coverage_state)
        mass_states.append(mass_state)
        if row["coverage_ci_low"] > margin_c and row["mass_ci_low"] > margin_m:
            state = "cancer"
        elif row["coverage_ci_high"] < -margin_c and row["mass_ci_high"] < -margin_m:
            state = "noncancer"
        elif (
            row["coverage_ci_low"] >= -margin_c
            and row["coverage_ci_high"] <= margin_c
            and row["mass_ci_low"] >= -margin_m
            and row["mass_ci_high"] <= margin_m
        ):
            state = "equivalent"
        else:
            state = "mixed"
        states.append(state)
    classification = classify_sharedness(seed_rows, protocol)
    if classification != "mixed_uncertain":
        reason = f"{classification}_rule_satisfied"
    elif "cancer" in states and "noncancer" in states:
        reason = "opposite_enrichment_across_seeds"
    elif coverage_states.count("equivalent") >= 2 and mass_states.count("equivalent") < 2:
        reason = "coverage_equivalent_mass_not_equivalent"
    elif mass_states.count("equivalent") >= 2 and coverage_states.count("equivalent") < 2:
        reason = "mass_equivalent_coverage_not_equivalent"
    elif (
        coverage_states.count("equivalent") >= 2
        and mass_states.count("equivalent") >= 2
        and states.count("equivalent") < 2
    ):
        reason = "metric_equivalence_on_different_seeds"
    elif "equivalent" in states and ({"cancer", "noncancer"} & set(states)):
        reason = "equivalence_enrichment_disagreement"
    elif len(set(states)) == 3:
        reason = "three_way_seed_disagreement"
    elif states.count("mixed") >= 2:
        reason = "within_seed_joint_criteria_unresolved"
    else:
        reason = "insufficient_seed_consensus"
    return classification, reason, ";".join(states)


def source_tests(
    presence: np.ndarray,
    mass: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> list[dict]:
    """逐Feature、逐标签计算来源presence与mass关联。"""
    output = []
    source_levels = sorted(np.unique(sources).tolist())
    for label in (0, 1):
        selected = labels == label
        group_indices = [np.flatnonzero(selected & (sources == source)) for source in source_levels]
        for feature in range(presence.shape[1]):
            active = [int(presence[index, feature].sum()) for index in group_indices]
            totals = [len(index) for index in group_indices]
            table = np.asarray([active, np.asarray(totals) - active], dtype=np.int64)
            mass_groups = [mass[index, feature] for index in group_indices]
            if sum(active) in {0, sum(totals)}:
                presence_p = 1.0
            else:
                _, presence_p, _, _ = chi2_contingency(table, correction=False)
            if np.ptp(np.concatenate(mass_groups)) == 0:
                statistic, mass_p = 0.0, 1.0
            else:
                statistic, mass_p = kruskal(*mass_groups)
            coverage = np.asarray(active, dtype=np.float64) / np.asarray(totals)
            epsilon_sq = max(0.0, (float(statistic) - len(source_levels) + 1) / (sum(totals) - len(source_levels)))
            output.append({
                "label": label,
                "feature_column": feature,
                "presence_p": float(presence_p),
                "presence_range": float(coverage.max() - coverage.min()),
                "mass_p": float(mass_p),
                "mass_epsilon_squared": float(epsilon_sq),
                "source_coverages": json.dumps(dict(zip(source_levels, coverage.tolist())), ensure_ascii=False),
                "source_mass_means": json.dumps(
                    {source: float(values.mean()) for source, values in zip(source_levels, mass_groups)},
                    ensure_ascii=False,
                ),
            })
    return output


def complete_link_families(anchor_ids: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    """按固定字典序执行全跨边约束的确定性complete-link合并。"""
    edge_set = {tuple(sorted(edge)) for edge in edges}
    clusters = [[anchor] for anchor in sorted(anchor_ids)]
    while True:
        candidates = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if all(tuple(sorted((a, b))) in edge_set for a in clusters[left] for b in clusters[right]):
                    candidates.append((clusters[left][0], clusters[right][0], left, right))
        if not candidates:
            break
        _, _, left, right = min(candidates)
        merged = sorted(clusters[left] + clusters[right])
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left, right}]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0])
    return clusters


def choose_representative(rows: list[dict]) -> str:
    """按冻结词典序选择family技术代表，不生成加权总分。"""
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["stability_worst_edge_score"]),
            -float(row["overall_patient_coverage"]),
            -float(row["energy_percentile"]),
            str(row["anchor_id"]),
        ),
    )
    return str(ordered[0]["anchor_id"])
