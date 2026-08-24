#!/usr/bin/env python3
"""在val上复现RP-A development matching，不重估train冻结规则。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark_clong_rpa_matching import batch_best_indices, spearman_matrix
from clong_rpa_bootstrap import METRIC_NAMES
from clong_rpa_development_core import (
    PAIR_ORDER,
    all_pseudo_fold_metrics_with_frozen_references,
)
from clong_rpa_match_development import (
    FORMAL_SPATIAL_SUPPORT,
    build_positive_ranks,
    build_top_sets,
    cross_spatial_matrix,
    jaccard_matrix,
    load_seed,
    normalized_maps,
    validate_cross_seed_alignment,
)
from clong_rpa_null_fdr import (
    DirectedHypothesis,
    benjamini_hochberg,
    exact_conditional_search_p,
    frozen_train_cdf_percentile,
    raw_direction_gates_pass,
    reciprocal_edges,
)
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
from clong_s2c_matryoshka import file_sha256


OUTPUT_ROOT = DEVELOPMENT_ROOT / "val_reproduction"
MATCHING_ROOT = DEVELOPMENT_ROOT / "full_train_matching"
BOOTSTRAP_ROOT = DEVELOPMENT_ROOT / "bootstrap_calibration" / "formal"
FORMAL_VAL_TOP_COUNT = 6
FORMAL_VAL_SPATIAL_SUPPORT = 6


def parse_args() -> argparse.Namespace:
    """解析设备、分块和debug缓存版本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-block", type=int, default=32)
    parser.add_argument("--target-block", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-cache-version", default="debug_v6/p2_e1")
    return parser.parse_args()


def pair_edges_from_csv(path: Path) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """读取冻结train RNN edges并恢复三个无序seed pair。"""
    if not path.read_text(encoding="utf-8").strip():
        return {pair: [] for pair in PAIR_ORDER}
    frame = pd.read_csv(path)
    result = {pair: [] for pair in PAIR_ORDER}
    for row in frame.itertuples(index=False):
        pair = tuple(sorted((int(row.seed_a), int(row.seed_b))))
        if pair not in result:
            raise ValueError(f"train edge包含非法seed pair: {pair}")
        left, right = int(row.feature_a), int(row.feature_b)
        if int(row.seed_a) > int(row.seed_b):
            left, right = right, left
        result[pair].append((left, right))
    return {pair: sorted(rows) for pair, rows in result.items()}


def _static(seed: dict, device: torch.device, top_count: int) -> dict[str, torch.Tensor]:
    """构造一个split的ranking、正激活秩和Top患者集合。"""
    ranking = np.asarray(seed["ranking"])
    patients = seed["patients"].patient_id.to_numpy(str)
    return {
        "ranking": torch.from_numpy(ranking.copy()).to(device),
        "ranks": torch.from_numpy(build_positive_ranks(ranking)).to(device),
        "top": torch.from_numpy(build_top_sets(ranking, patients, top_count)).to(device),
    }


def _patient_index(seed: dict, device: torch.device) -> torch.Tensor:
    """把图像行映射到当前split的患者行。"""
    lookup = {
        str(patient): index
        for index, patient in enumerate(seed["patients"].patient_id.to_numpy(str))
    }
    return torch.as_tensor(
        [lookup[str(patient)] for patient in seed["images"].patient_id], device=device
    )


def _raw_behavior(
    source: dict, target: dict, source_ids: np.ndarray, target_ids: np.ndarray,
    source_static: dict[str, torch.Tensor], target_static: dict[str, torch.Tensor],
    source_maps: torch.Tensor, source_active: torch.Tensor,
    target_maps: torch.Tensor, target_active: torch.Tensor,
    image_patient_index: torch.Tensor, top_count: int, minimum_support: int,
    source_block: int, target_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算一个source block对全部target的三项患者依赖原始指标。"""
    device = source_static["ranking"].device
    source_index = torch.as_tensor(source_ids, device=device)
    target_index = torch.as_tensor(target_ids, device=device)
    jaccard = jaccard_matrix(
        source_static["top"].index_select(1, source_index),
        target_static["top"].index_select(1, target_index), top_count,
    )
    spearman = spearman_matrix(
        source_static["ranking"].index_select(1, source_index),
        target_static["ranking"].index_select(1, target_index),
        source_static["ranks"].index_select(1, source_index),
        target_static["ranks"].index_select(1, target_index),
        source_block, target_block,
    )
    spatial = cross_spatial_matrix(
        source_maps, source_active, target_maps, target_active,
        image_patient_index, len(source["patients"]), source_ids, target_ids,
        minimum_support, target_block,
    )
    return spearman, jaccard, spatial


def reduce_val_direction(
    source_seed: int, target_seed: int,
    train_source: dict, train_target: dict, val_source: dict, val_target: dict,
    decoder_cosine: np.ndarray,
    train_static_source: dict[str, torch.Tensor], train_static_target: dict[str, torch.Tensor],
    val_static_source: dict[str, torch.Tensor], val_static_target: dict[str, torch.Tensor],
    train_maps_source: torch.Tensor, train_active_source: torch.Tensor,
    train_maps_target: torch.Tensor, train_active_target: torch.Tensor,
    val_maps_source: torch.Tensor, val_active_source: torch.Tensor,
    val_maps_target: torch.Tensor, val_active_target: torch.Tensor,
    train_image_patient_index: torch.Tensor, val_image_patient_index: torch.Tensor,
    train_top_count: int, val_top_count: int,
    train_minimum_support: int, val_minimum_support: int,
    source_block: int, target_block: int,
) -> list[dict]:
    """用train CDF映射val指标，并只保留每个source的val唯一best。"""
    output = []
    source_ids = np.asarray(train_source["eligible_ids"], dtype=int)
    target_ids = np.asarray(train_target["eligible_ids"], dtype=int)
    for start in range(0, len(source_ids), source_block):
        stop = min(start + source_block, len(source_ids))
        block_ids = source_ids[start:stop]
        train_behavior = _raw_behavior(
            train_source, train_target, block_ids, target_ids,
            train_static_source, train_static_target,
            train_maps_source, train_active_source, train_maps_target, train_active_target,
            train_image_patient_index, train_top_count, train_minimum_support,
            source_block, target_block,
        )
        val_behavior = _raw_behavior(
            val_source, val_target, block_ids, target_ids,
            val_static_source, val_static_target,
            val_maps_source, val_active_source, val_maps_target, val_active_target,
            val_image_patient_index, val_top_count, val_minimum_support,
            source_block, target_block,
        )
        raw_train = [decoder_cosine[start:stop], *train_behavior]
        raw_val = [decoder_cosine[start:stop], *val_behavior]
        train_valid = np.logical_and.reduce([np.isfinite(values) for values in raw_train])
        val_valid = np.logical_and.reduce([np.isfinite(values) for values in raw_val])
        percentiles = [np.full(values.shape, np.nan) for values in raw_val]
        for row in range(len(block_ids)):
            for metric in range(4):
                percentiles[metric][row] = frozen_train_cdf_percentile(
                    raw_train[metric][row], raw_val[metric][row],
                    train_valid[row], val_valid[row],
                )
        behavior = np.full(val_valid.shape, np.nan, dtype=np.float64)
        score = np.full(val_valid.shape, np.nan, dtype=np.float64)
        stacked = np.stack(percentiles[1:])
        behavior[val_valid] = np.median(stacked[:, val_valid], axis=0)
        score[val_valid] = np.minimum(percentiles[0][val_valid], behavior[val_valid])
        best_indices = batch_best_indices(score, raw_val[0], target_ids, val_valid)
        for row, feature in enumerate(block_ids):
            best = int(best_indices[row])
            if best < 0:
                target_feature, p_value = int(target_ids[0]), 1.0
                best_raw, best_score = [None] * 4, None
            else:
                best_raw = [float(values[row, best]) for values in raw_val]
                target_feature = int(target_ids[best])
                p_value = 1.0
                if raw_direction_gates_pass(*best_raw):
                    p_value = exact_conditional_search_p(
                        percentiles[0][row], behavior[row], train_target["strata"],
                        val_valid[row], float(score[row, best]),
                    )
                best_score = float(score[row, best]) if np.isfinite(score[row, best]) else None
            output.append({
                "source_seed": source_seed,
                "source_feature_id": int(feature),
                "target_seed": target_seed,
                "target_feature_id": target_feature,
                "p_value": float(p_value),
                "bh_rejected": False,
                "edge_score": best_score,
                "signed_decoder_cosine": best_raw[0],
                "patient_spearman": best_raw[1],
                "top_patient_jaccard": best_raw[2],
                "spatial_similarity": best_raw[3],
            })
        print(f"val direction {source_seed}->{target_seed}: {stop}/{len(source_ids)}", flush=True)
    return output


def compute_val_matching(
    train: dict[int, dict], val: dict[int, dict], device: torch.device,
    source_block: int, target_block: int, train_top_count: int,
    val_top_count: int, train_minimum_support: int, val_minimum_support: int,
) -> tuple[list[dict], dict[tuple[int, int], list[tuple[int, int]]]]:
    """以train CDF和strata为冻结参照，独立构造val matching graph。"""
    validate_cross_seed_alignment(train)
    validate_cross_seed_alignment(val)
    train_static = {seed: _static(data, device, train_top_count) for seed, data in train.items()}
    val_static = {seed: _static(data, device, val_top_count) for seed, data in val.items()}
    train_patient_index = _patient_index(train[42], device)
    val_patient_index = _patient_index(val[42], device)
    all_rows, pair_edges = [], {}
    for left, right in PAIR_ORDER:
        train_left_maps, train_left_active = normalized_maps(train[left], device)
        train_right_maps, train_right_active = normalized_maps(train[right], device)
        val_left_maps, val_left_active = normalized_maps(val[left], device)
        val_right_maps, val_right_active = normalized_maps(val[right], device)
        left_ids = np.asarray(train[left]["eligible_ids"], dtype=int)
        right_ids = np.asarray(train[right]["eligible_ids"], dtype=int)
        left_decoder = torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(train[left]["decoder"])[left_ids]).to(device), dim=1
        )
        right_decoder = torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(train[right]["decoder"])[right_ids]).to(device), dim=1
        )
        cosine = (left_decoder @ right_decoder.T).cpu().numpy()
        forward = reduce_val_direction(
            left, right, train[left], train[right], val[left], val[right], cosine,
            train_static[left], train_static[right], val_static[left], val_static[right],
            train_left_maps, train_left_active, train_right_maps, train_right_active,
            val_left_maps, val_left_active, val_right_maps, val_right_active,
            train_patient_index, val_patient_index, train_top_count, val_top_count,
            train_minimum_support, val_minimum_support, source_block, target_block,
        )
        reverse = reduce_val_direction(
            right, left, train[right], train[left], val[right], val[left], cosine.T,
            train_static[right], train_static[left], val_static[right], val_static[left],
            train_right_maps, train_right_active, train_left_maps, train_left_active,
            val_right_maps, val_right_active, val_left_maps, val_left_active,
            train_patient_index, val_patient_index, train_top_count, val_top_count,
            train_minimum_support, val_minimum_support, source_block, target_block,
        )
        rows = forward + reverse
        hypotheses = [DirectedHypothesis(
            int(row["source_seed"]), int(row["target_seed"]),
            int(row["source_feature_id"]), int(row["target_feature_id"]),
            float(row["p_value"]),
        ) for row in rows]
        rejected, cutoff = benjamini_hochberg(hypotheses, q=0.05)
        for row in rows:
            key = (row["source_seed"], row["target_seed"],
                   row["source_feature_id"], row["target_feature_id"])
            row["bh_rejected"] = key in rejected
            row["bh_cutoff"] = cutoff
        pair_edges[(left, right)] = reciprocal_edges(hypotheses, rejected)
        all_rows.extend(rows)
        del train_left_maps, train_right_maps, val_left_maps, val_right_maps
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_rows, pair_edges


def minimum_fold_metrics(folds: dict[str, dict[str, float]]) -> dict[str, float]:
    """按冻结最弱折规则把三折归约成六项val正式指标。"""
    return {
        metric: float(min(folds[str(seed)][metric] for seed in (42, 43, 44)))
        for metric in METRIC_NAMES
    }


def main() -> None:
    """执行val独立复现；debug只验证链路，不生成正式PASS。"""
    args = parse_args()
    if not args.debug and args.device != "cuda":
        raise ValueError("正式val复现必须使用CUDA")
    target = OUTPUT_ROOT / ("debug" if args.debug else "formal")
    if target.exists():
        raise FileExistsError(f"val复现输出已存在: {target}")
    matching = MATCHING_ROOT / ("debug_v2" if args.debug else "formal")
    train_edges = pair_edges_from_csv(matching / "reciprocal_edges.csv")
    train = {seed: load_seed(seed, args, split="train") for seed in (42, 43, 44)}
    val = {seed: load_seed(seed, args, split="val") for seed in (42, 43, 44)}
    device = torch.device(args.device)
    train_top = min(25, len(train[42]["patients"])) if args.debug else 25
    val_top = min(FORMAL_VAL_TOP_COUNT, len(val[42]["patients"])) if args.debug else FORMAL_VAL_TOP_COUNT
    train_support = min(2, len(train[42]["patients"])) if args.debug else FORMAL_SPATIAL_SUPPORT
    val_support = min(2, len(val[42]["patients"])) if args.debug else FORMAL_VAL_SPATIAL_SUPPORT
    rows, val_edges = compute_val_matching(
        train, val, device, args.source_block, args.target_block,
        train_top, val_top, train_support, val_support,
    )
    eligible = {seed: np.asarray(data["eligible_ids"], dtype=int) for seed, data in train.items()}
    mass = {seed: np.asarray(data["mass"]).sum(0) for seed, data in val.items()}
    energy = {seed: np.asarray(data["energy"]).sum(0) for seed, data in val.items()}
    folds = all_pseudo_fold_metrics_with_frozen_references(
        train_edges, val_edges, eligible, mass, energy,
    )
    metrics = minimum_fold_metrics(folds)
    if args.debug:
        thresholds, passed, status = None, None, "debug_completed"
    else:
        threshold_payload = json.loads(
            (BOOTSTRAP_ROOT / "bootstrap_thresholds.json").read_text(encoding="utf-8")
        )
        thresholds = threshold_payload["thresholds"]
        passed = all(metrics[name] >= float(thresholds[name]) for name in METRIC_NAMES)
        status = "val_reproduction_passed" if passed else "val_reproduction_failure"
    target.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(target / "val_best_directed_hypotheses.csv", index=False)
    edge_rows = [{
        "seed_a": left, "feature_a": int(a), "seed_b": right, "feature_b": int(b),
        "both_directions_bh_rejected": True, "reciprocal": True,
    } for (left, right), edges in val_edges.items() for a, b in edges]
    pd.DataFrame(edge_rows, columns=[
        "seed_a", "feature_a", "seed_b", "feature_b",
        "both_directions_bh_rejected", "reciprocal",
    ]).to_csv(target / "val_reciprocal_edges.csv", index=False)
    result = {
        "debug": bool(args.debug),
        "status": status,
        "passed": passed,
        "fold_metrics": folds,
        "minimum_fold_metrics": metrics,
        "thresholds": thresholds,
        "train_feature_universe_frozen": True,
        "train_strata_frozen": True,
        "train_empirical_cdf_frozen": True,
        "val_null_reestimated": False,
        "val_eligible_rescreened": False,
        "val_reference_anchors_rebuilt": False,
        "train_matching_sha256": file_sha256(matching / "full_train_metrics.json"),
    }
    result_path = target / "val_reproduction.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    names = sorted(path.name for path in target.iterdir() if path.is_file())
    (target / "SHA256SUMS.txt").write_text(
        "\n".join(f"{file_sha256(target / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )
    print(f"RP-A val复现完成: {target}; status={status}")


if __name__ == "__main__":
    main()
