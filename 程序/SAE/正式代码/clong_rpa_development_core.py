#!/usr/bin/env python3
"""RP-A development 3-clique 与三折伪确认指标的纯函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from clong_rpa_bootstrap import METRIC_NAMES


DEVELOPMENT_SEEDS = (42, 43, 44)
PAIR_ORDER = ((42, 43), (42, 44), (43, 44))


def edge_mapping(
    pair_edges: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
    source_seed: int,
    target_seed: int,
) -> dict[int, int]:
    """按请求方向返回一一 edge 映射。"""
    pair = tuple(sorted((source_seed, target_seed)))
    if pair not in pair_edges:
        raise ValueError(f"缺少seed pair: {pair}")
    rows = list(pair_edges[pair])
    left = [int(a) for a, _ in rows]
    right = [int(b) for _, b in rows]
    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValueError(f"seed pair {pair}不是一一edge")
    if source_seed == pair[0]:
        return dict(zip(left, right))
    return dict(zip(right, left))


def strict_three_cliques(
    pair_edges: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
) -> list[dict[str, int | str | bool]]:
    """由三组一一 pair edge 构造严格42/43/44三角形。"""
    map_42_43 = edge_mapping(pair_edges, 42, 43)
    map_42_44 = edge_mapping(pair_edges, 42, 44)
    map_43_44 = edge_mapping(pair_edges, 43, 44)
    anchors = []
    for feature_42 in sorted(set(map_42_43) & set(map_42_44)):
        feature_43 = map_42_43[feature_42]
        feature_44 = map_42_44[feature_42]
        if map_43_44.get(feature_43) != feature_44:
            continue
        anchors.append({
            "anchor_id": f"a{len(anchors):05d}",
            "feature_42": int(feature_42),
            "feature_43": int(feature_43),
            "feature_44": int(feature_44),
            "edge_42_43": True,
            "edge_42_44": True,
            "edge_43_44": True,
        })
    return anchors


def _ratio(numerator: float, denominator: float) -> float:
    """返回正式比例；零分母表示该折无可评价证据，记0。"""
    if denominator < 0 or numerator < 0 or numerator > denominator + 1e-12:
        raise ValueError("coverage numerator/denominator非法")
    return 0.0 if denominator == 0 else float(numerator / denominator)


def pseudo_confirmation_metrics(
    held_out_seed: int,
    pair_edges: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
    eligible_ids: Mapping[int, np.ndarray],
    activation_mass: Mapping[int, np.ndarray],
    representation_energy: Mapping[int, np.ndarray],
) -> dict[str, float]:
    """计算一个2/2 reference → held-out伪确认折的六项正式比例。

    reference anchor 是两个reference seed之间的一一edge；只有held-out
    Feature同时连接这两个成员才算 reproduced。reference coverage 先逐
    seed计算ratio再等权平均，禁止合并两个seed的raw mass/energy。
    """
    if held_out_seed not in DEVELOPMENT_SEEDS:
        raise ValueError("held_out_seed必须属于42/43/44")
    references = [seed for seed in DEVELOPMENT_SEEDS if seed != held_out_seed]
    left, right = references
    reference_edges = list(pair_edges[tuple(sorted((left, right)))])
    if left > right:
        reference_edges = [(b, a) for a, b in reference_edges]
    held_to_left = edge_mapping(pair_edges, held_out_seed, left)
    held_to_right = edge_mapping(pair_edges, held_out_seed, right)
    left_to_anchor = {int(a): index for index, (a, _b) in enumerate(reference_edges)}
    right_members = [int(b) for _a, b in reference_edges]
    reproduced_anchor_indices: set[int] = set()
    reproduced_held_features: set[int] = set()
    for held_feature, left_feature in held_to_left.items():
        right_feature = held_to_right.get(held_feature)
        anchor_index = left_to_anchor.get(left_feature)
        if anchor_index is None or right_feature != right_members[anchor_index]:
            continue
        reproduced_anchor_indices.add(anchor_index)
        reproduced_held_features.add(int(held_feature))

    eligible_sets = {seed: set(np.asarray(ids, dtype=int).tolist()) for seed, ids in eligible_ids.items()}
    if any(feature not in eligible_sets[left] or right_members[index] not in eligible_sets[right]
           for index, (feature, _other) in enumerate(reference_edges)):
        raise ValueError("reference edge含非eligible Feature")
    if not reproduced_held_features.issubset(eligible_sets[held_out_seed]):
        raise ValueError("reproduced held-out Feature不在eligible集合")

    anchor_count = len(reference_edges)
    anchor_recall = _ratio(len(reproduced_anchor_indices), anchor_count)
    confirm_coverage = _ratio(len(reproduced_held_features), len(eligible_sets[held_out_seed]))

    def reference_coverage(values: Mapping[int, np.ndarray]) -> float:
        ratios = []
        for position, seed in enumerate((left, right)):
            all_ids = np.asarray([edge[position] for edge in reference_edges], dtype=int)
            matched_ids = all_ids[np.asarray(sorted(reproduced_anchor_indices), dtype=int)] \
                if reproduced_anchor_indices else np.asarray([], dtype=int)
            array = np.asarray(values[seed], dtype=np.float64)
            ratios.append(_ratio(float(array[matched_ids].sum()), float(array[all_ids].sum())))
        return float(np.mean(ratios))

    def confirmation_coverage(values: Mapping[int, np.ndarray]) -> float:
        array = np.asarray(values[held_out_seed], dtype=np.float64)
        matched = np.asarray(sorted(reproduced_held_features), dtype=int)
        all_ids = np.asarray(sorted(eligible_sets[held_out_seed]), dtype=int)
        return _ratio(float(array[matched].sum()), float(array[all_ids].sum()))

    result = {
        "R_anchor_recall": anchor_recall,
        "R_confirm_coverage": confirm_coverage,
        "R_activation_reference": reference_coverage(activation_mass),
        "R_activation_confirmation": confirmation_coverage(activation_mass),
        "R_energy_reference": reference_coverage(representation_energy),
        "R_energy_confirmation": confirmation_coverage(representation_energy),
    }
    if set(result) != set(METRIC_NAMES):
        raise RuntimeError("伪确认没有产生冻结六指标")
    return result


def all_pseudo_fold_metrics(
    pair_edges: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
    eligible_ids: Mapping[int, np.ndarray],
    activation_mass: Mapping[int, np.ndarray],
    representation_energy: Mapping[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    """按42、43、44依次作为held-out生成三折指标。"""
    return {
        str(seed): pseudo_confirmation_metrics(
            seed, pair_edges, eligible_ids, activation_mass, representation_energy
        )
        for seed in DEVELOPMENT_SEEDS
    }
