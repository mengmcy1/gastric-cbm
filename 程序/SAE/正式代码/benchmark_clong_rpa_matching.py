#!/usr/bin/env python3
"""RP-A完整matching链的纯计算benchmark。

本入口只使用seed42冻结SAE和train缓存，通过Feature列的确定性
双射构造3个逻辑seed。它执行一正式规模的3 pair matching和3折
pseudo-confirm plumbing，但不训练新SAE、不执行bootstrap循环，也不保存
任何边、anchor、p值或覆盖统计。
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from clong_rpa_eligible import (
    OUTPUT_ROOT as ELIGIBLE_ROOT,
    SPATIAL_CACHE,
    load_sae,
    validate_lineage,
)
from clong_rpa_null_fdr import (
    DirectedHypothesis,
    assign_target_strata,
    benjamini_hochberg,
    choose_best_candidate,
    exact_conditional_search_p,
    raw_direction_gates_pass,
    reciprocal_edges,
    train_midrank_percentile,
    validate_target_strata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "结果/SAE/RP_A_Matching计算基准_20260824/benchmark_only.json"
)
HIDDEN_DIM = 10240
TOP_Q_COUNT = 25
A_MIN = 0.25 / 49.0
LOGICAL_SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    """解析正式规模、分块大小和输出位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--encoding-batch-size", type=int, default=4)
    parser.add_argument("--source-block", type=int, default=32)
    parser.add_argument("--target-block", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def logical_permutations(hidden_dim: int) -> dict[int, np.ndarray]:
    """返回逻辑Feature ID到seed42基础Feature ID的确定性双射。"""
    ids = np.arange(hidden_dim, dtype=np.int64)
    return {
        42: ids,
        43: (3 * ids + 17) % hidden_dim,
        44: (7 * ids + 31) % hidden_dim,
    }


def build_top_sets(ranking: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    """按冻结秩次和patient ID并列顺序构造Top-25集合。"""
    result = np.zeros(ranking.shape, dtype=bool)
    for feature_id in range(ranking.shape[1]):
        order = np.lexsort((patient_ids, -ranking[:, feature_id]))
        result[order[:TOP_Q_COUNT], feature_id] = True
    return result


def build_positive_ranks(ranking: np.ndarray) -> np.ndarray:
    """只在正激活患者中计算并列平均秩，供union-positive Spearman使用。"""
    result = np.zeros_like(ranking, dtype=np.float32)
    for feature_id in range(ranking.shape[1]):
        positive = ranking[:, feature_id] > 0
        if positive.any():
            result[positive, feature_id] = rankdata(
                ranking[positive, feature_id], method="average"
            ).astype(np.float32)
    return result


@torch.no_grad()
def encode_spatial_activations(
    sae: torch.nn.Module,
    spatial: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """编码全部图像并就地归一化49位置激活图。

    返回``[image,49,feature]``归一化激活与``[image,feature]``
    非零标记。归一化只用于spatial cosine，患者ranking/presence
    仍读取eligible审计的冻结原值。
    """
    n_images = spatial.shape[0]
    activations = torch.empty(
        (n_images, spatial.shape[1], HIDDEN_DIM), device=device, dtype=torch.float32
    )
    for start in range(0, n_images, batch_size):
        stop = min(start + batch_size, n_images)
        features = torch.from_numpy(np.array(spatial[start:stop], copy=True)).to(device)
        activations[start:stop] = sae.encode(features, k=1024)
    norms = torch.linalg.vector_norm(activations, dim=1)
    active = norms > 0
    activations.div_(norms.clamp_min(1e-12).unsqueeze(1))
    return activations, active


def synchronize(device: torch.device) -> None:
    """等待当前CUDA设备完成，确保阶段计时不受异步执行影响。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stage_start(device: torch.device) -> float:
    """同步设备并开始准确计时。"""
    synchronize(device)
    return time.perf_counter()


def stage_stop(device: torch.device, start: float) -> float:
    """同步设备并返回阶段耗时。"""
    synchronize(device)
    return time.perf_counter() - start


def jaccard_matrix(
    source_top: torch.Tensor, target_top: torch.Tensor, source_block: int
) -> np.ndarray:
    """分块计算全候选Top-patient Jaccard。"""
    n_source, n_target = source_top.shape[1], target_top.shape[1]
    output = np.empty((n_source, n_target), dtype=np.float32)
    target_float = target_top.float()
    for start in range(0, n_source, source_block):
        stop = min(start + source_block, n_source)
        intersection = source_top[:, start:stop].T.float() @ target_float
        union = 2 * TOP_Q_COUNT - intersection
        output[start:stop] = (intersection / union).cpu().numpy()
    return output


def _union_spearman_chunk(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    source_ranks: torch.Tensor,
    target_ranks: torch.Tensor,
) -> torch.Tensor:
    """对一个Feature block精确计算union-positive Spearman。"""
    sx = source_values.gt(0).unsqueeze(2)
    ty = target_values.gt(0).unsqueeze(1)
    union = sx | ty
    n = union.sum(dim=0).float()
    y_only = ((~sx) & ty).sum(dim=0).float()
    x_only = (sx & (~ty)).sum(dim=0).float()
    rx = torch.where(
        sx,
        source_ranks.unsqueeze(2) + y_only.unsqueeze(0),
        ((y_only + 1.0) / 2.0).unsqueeze(0),
    )
    ry = torch.where(
        ty,
        target_ranks.unsqueeze(1) + x_only.unsqueeze(0),
        ((x_only + 1.0) / 2.0).unsqueeze(0),
    )
    mask = union.float()
    sum_x = (rx * mask).sum(0)
    sum_y = (ry * mask).sum(0)
    sum_x2 = (rx.square() * mask).sum(0)
    sum_y2 = (ry.square() * mask).sum(0)
    sum_xy = (rx * ry * mask).sum(0)
    covariance = sum_xy - sum_x * sum_y / n.clamp_min(1)
    variance_x = sum_x2 - sum_x.square() / n.clamp_min(1)
    variance_y = sum_y2 - sum_y.square() / n.clamp_min(1)
    denominator = torch.sqrt(variance_x.clamp_min(0) * variance_y.clamp_min(0))
    result = covariance / denominator.clamp_min(1e-12)
    result[(n < 2) | (denominator <= 0)] = torch.nan
    return result


def spearman_matrix(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    source_ranks: torch.Tensor,
    target_ranks: torch.Tensor,
    source_block: int,
    target_block: int,
) -> np.ndarray:
    """以二维分块计算全候选union-positive Spearman。"""
    n_source, n_target = source_values.shape[1], target_values.shape[1]
    output = np.empty((n_source, n_target), dtype=np.float32)
    for s0 in range(0, n_source, source_block):
        s1 = min(s0 + source_block, n_source)
        for t0 in range(0, n_target, target_block):
            t1 = min(t0 + target_block, n_target)
            output[s0:s1, t0:t1] = _union_spearman_chunk(
                source_values[:, s0:s1], target_values[:, t0:t1],
                source_ranks[:, s0:s1], target_ranks[:, t0:t1],
            ).cpu().numpy()
    return output


def spatial_matrix(
    normalized_maps: torch.Tensor,
    image_active: torch.Tensor,
    image_patient_index: torch.Tensor,
    source_base_ids: np.ndarray,
    target_base_ids: np.ndarray,
    patient_count: int,
    source_block: int,
    target_block: int,
) -> np.ndarray:
    """按图像计算49位置cosine，再按患者内、患者间等权聚合。"""
    n_source, n_target = len(source_base_ids), len(target_base_ids)
    output = np.empty((n_source, n_target), dtype=np.float32)
    device = normalized_maps.device
    for s0 in range(0, n_source, source_block):
        s1 = min(s0 + source_block, n_source)
        source_ids = torch.as_tensor(source_base_ids[s0:s1], device=device)
        source_maps = normalized_maps.index_select(2, source_ids).permute(0, 2, 1)
        source_active = image_active.index_select(1, source_ids)
        for t0 in range(0, n_target, target_block):
            t1 = min(t0 + target_block, n_target)
            target_ids = torch.as_tensor(target_base_ids[t0:t1], device=device)
            target_maps = normalized_maps.index_select(2, target_ids)
            target_active = image_active.index_select(1, target_ids)
            cosine = torch.bmm(source_maps, target_maps)
            union = source_active.unsqueeze(2) | target_active.unsqueeze(1)
            cosine.mul_(union)
            patient_sum = torch.zeros(
                (patient_count, s1 - s0, t1 - t0), device=device
            )
            patient_count_map = torch.zeros_like(patient_sum)
            patient_sum.index_add_(0, image_patient_index, cosine)
            patient_count_map.index_add_(0, image_patient_index, union.float())
            valid = patient_count_map > 0
            patient_value = patient_sum / patient_count_map.clamp_min(1)
            valid_patients = valid.sum(0)
            result = (patient_value * valid).sum(0) / valid_patients.clamp_min(1)
            result[valid_patients == 0] = torch.nan
            output[s0:s1, t0:t1] = result.cpu().numpy()
    return output


def batch_midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """批量计算与冻结逐行函数完全等价的经验中秩百分位。"""
    x = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if x.shape != mask.shape or x.ndim != 2:
        raise ValueError("values/valid必须是同shape二维数组")
    if not np.isfinite(x[mask]).all():
        raise ValueError("valid不得包含NaN/Inf")
    query = x.copy()
    query[~mask] = np.nan
    ranks = rankdata(query, axis=1, method="average", nan_policy="omit")
    counts = mask.sum(axis=1)
    output = np.full(x.shape, np.nan, dtype=np.float64)
    nonempty = counts > 0
    output[nonempty] = (
        ranks[nonempty] - 0.5
    ) / counts[nonempty, None]
    output[~mask] = np.nan
    return output


def batch_best_indices(
    edge_score: np.ndarray,
    signed_decoder_cosine: np.ndarray,
    target_feature_ids: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """按冻结的score→cosine→target ID顺序批量选择best索引。"""
    score = np.asarray(edge_score, dtype=np.float64)
    cosine = np.asarray(signed_decoder_cosine, dtype=np.float64)
    ids = np.asarray(target_feature_ids, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(score) & np.isfinite(cosine)
    if score.shape != cosine.shape or score.shape != mask.shape or score.ndim != 2:
        raise ValueError("best candidate的score/cosine/valid shape不一致")
    if ids.shape != (score.shape[1],):
        raise ValueError("target_feature_ids shape不一致")
    masked_score = np.where(mask, score, -np.inf)
    best_score = masked_score.max(axis=1)
    score_tie = mask & (score == best_score[:, None])
    masked_cosine = np.where(score_tie, cosine, -np.inf)
    best_cosine = masked_cosine.max(axis=1)
    final_tie = score_tie & (cosine == best_cosine[:, None])
    sentinel = np.iinfo(np.int64).max
    chosen_id = np.where(final_tie, ids[None, :], sentinel).min(axis=1)
    result = np.full(score.shape[0], -1, dtype=np.int64)
    for row in np.flatnonzero(chosen_id != sentinel):
        result[row] = int(np.flatnonzero(final_tie[row] & (ids == chosen_id[row]))[0])
    return result


def directional_hypotheses_streaming(
    source_seed: int,
    target_seed: int,
    source_info: dict,
    target_info: dict,
    decoder_cosine: np.ndarray,
    ranking_gpu: torch.Tensor,
    positive_ranks_gpu: torch.Tensor,
    top_gpu: torch.Tensor,
    normalized_maps: torch.Tensor,
    image_active: torch.Tensor,
    image_patient_index: torch.Tensor,
    patient_count: int,
    source_block: int,
    target_block: int,
    device: torch.device,
    timings: dict[str, float],
) -> list[DirectedHypothesis]:
    """按source block计算四项指标并立即归约为有向假设。

    每个block只保留``[source_block, all_target]``的临时指标；完成百分位、
    best candidate和exact-null后立即释放，不持久化完整行为矩阵。
    """
    hypotheses: list[DirectedHypothesis] = []
    target_base = torch.as_tensor(target_info["base_ids"], device=device)
    target_top = top_gpu.index_select(1, target_base)
    target_ranking = ranking_gpu.index_select(1, target_base)
    target_ranks = positive_ranks_gpu.index_select(1, target_base)
    print(f"[benchmark] START direction {source_seed}->{target_seed}", flush=True)
    direction_start = time.perf_counter()
    for start_row in range(0, len(source_info["ids"]), source_block):
        stop_row = min(start_row + source_block, len(source_info["ids"]))
        source_base_ids = source_info["base_ids"][start_row:stop_row]
        source_base = torch.as_tensor(source_base_ids, device=device)

        started = stage_start(device)
        jaccard = jaccard_matrix(
            top_gpu.index_select(1, source_base), target_top, source_block
        )
        timings["jaccard_seconds"] += stage_stop(device, started)

        started = stage_start(device)
        spearman = spearman_matrix(
            ranking_gpu.index_select(1, source_base), target_ranking,
            positive_ranks_gpu.index_select(1, source_base), target_ranks,
            source_block, target_block,
        )
        timings["spearman_seconds"] += stage_stop(device, started)

        started = stage_start(device)
        spatial_values = spatial_matrix(
            normalized_maps, image_active, image_patient_index,
            source_base_ids, target_info["base_ids"], patient_count,
            source_block, target_block,
        )
        timings["spatial_seconds"] += stage_stop(device, started)

        hypotheses.extend(directed_hypotheses(
            source_seed, target_seed,
            source_info["ids"][start_row:stop_row], target_info["ids"],
            decoder_cosine[start_row:stop_row], spearman, jaccard, spatial_values,
            target_info["strata"], timings,
        ))
        del jaccard, spearman, spatial_values
        print(
            f"[benchmark] direction {source_seed}->{target_seed} "
            f"rows {stop_row}/{len(source_info['ids'])}",
            flush=True,
        )
    synchronize(device)
    print(
        f"[benchmark] DONE direction {source_seed}->{target_seed} "
        f"{time.perf_counter() - direction_start:.2f}s",
        flush=True,
    )
    return hypotheses


def directed_hypotheses(
    source_seed: int,
    target_seed: int,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    decoder_cosine: np.ndarray,
    spearman: np.ndarray,
    jaccard: np.ndarray,
    spatial: np.ndarray,
    target_strata: np.ndarray,
    timings: dict[str, float],
) -> list[DirectedHypothesis]:
    """对一个source block执行百分位、best candidate和exact-null。"""
    hypotheses: list[DirectedHypothesis] = []
    raw_arrays = [
        np.asarray(decoder_cosine, dtype=np.float64),
        np.asarray(spearman, dtype=np.float64),
        np.asarray(jaccard, dtype=np.float64),
        np.asarray(spatial, dtype=np.float64),
    ]
    valid = np.logical_and.reduce([np.isfinite(values) for values in raw_arrays])
    start = time.perf_counter()
    percentiles = [batch_midrank_percentile(values, valid) for values in raw_arrays]
    behavior = np.full(valid.shape, np.nan, dtype=np.float64)
    score = np.full(valid.shape, np.nan, dtype=np.float64)
    stacked = np.stack(percentiles[1:], axis=0)
    behavior[valid] = np.median(stacked[:, valid], axis=0)
    score[valid] = np.minimum(percentiles[0][valid], behavior[valid])
    best_indices = batch_best_indices(
        score, raw_arrays[0], target_ids, valid
    )
    timings["percentile_seconds"] += time.perf_counter() - start
    for row, source_id in enumerate(source_ids):
        best = int(best_indices[row])
        raw = tuple(values[row] for values in raw_arrays)
        if best < 0 or not raw_direction_gates_pass(*(values[best] for values in raw)):
            p_value, target_id = 1.0, int(target_ids[0])
        else:
            start = time.perf_counter()
            p_value = exact_conditional_search_p(
                percentiles[0][row], behavior[row], target_strata,
                valid[row], float(score[row, best])
            )
            timings["exact_null_seconds"] += time.perf_counter() - start
            target_id = int(target_ids[best])
        hypotheses.append(DirectedHypothesis(
            source_seed, target_seed, int(source_id), target_id, p_value
        ))
    return hypotheses


def pseudo_fold_plumbing(
    pair_edges: dict[tuple[int, int], list[tuple[int, int]]]
) -> None:
    """构造3折2/2 anchor与held-out成员的完整关联，不返回统计量。"""
    def mapping(a: int, b: int) -> dict[int, int]:
        low, high = sorted((a, b))
        edges = pair_edges[(low, high)]
        return dict(edges) if a == low else {right: left for left, right in edges}

    for held_out in LOGICAL_SEEDS:
        references = [seed for seed in LOGICAL_SEEDS if seed != held_out]
        reference_map = mapping(references[0], references[1])
        held_to_first = mapping(held_out, references[0])
        held_to_second = mapping(held_out, references[1])
        # 遍历全部held-out边，证明2/2 anchor plumbing路径完整执行。
        for held_feature, first_feature in held_to_first.items():
            second_feature = held_to_second.get(held_feature)
            _ = second_feature is not None and reference_map.get(first_feature) == second_feature


def peak_rss_gb() -> float:
    """返回Linux进程峰值RSS（GiB）。"""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2))


def run_self_test() -> None:
    """用小矩阵核对Spearman、空间聚合和逻辑seed双射。"""
    permutations = logical_permutations(64)
    assert all(np.unique(values).size == 64 for values in permutations.values())
    values = torch.tensor([[0., 1.], [2., 0.], [3., 4.], [0., 5.]])
    ranks = torch.from_numpy(build_positive_ranks(values.numpy()))
    result = _union_spearman_chunk(values, values, ranks, ranks)
    assert torch.allclose(torch.diag(result), torch.ones(2), atol=1e-6)
    left = values[:, :1].numpy().ravel()
    right = values[:, 1:].numpy().ravel()
    union = (left > 0) | (right > 0)
    expected = np.corrcoef(
        rankdata(left[union], method="average"),
        rankdata(right[union], method="average"),
    )[0, 1]
    assert np.isclose(float(result[0, 1]), expected, atol=1e-6)

    maps = torch.tensor([
        [[1., 1.], [0., 0.]],
        [[0., 1.], [1., 0.]],
        [[1., 0.], [0., 1.]],
    ])
    norms = torch.linalg.vector_norm(maps, dim=1)
    active = norms > 0
    maps = maps / norms.clamp_min(1e-12).unsqueeze(1)
    spatial = spatial_matrix(
        maps, active, torch.tensor([0, 0, 1]),
        np.array([0]), np.array([1]), 2, 1, 1,
    )
    assert np.isclose(float(spatial[0, 0]), 0.25, atol=1e-6)

    rng = np.random.default_rng(20260824)
    metric = rng.integers(0, 5, size=(7, 19)).astype(np.float64)
    valid = rng.random((7, 19)) > 0.2
    batch_percentile = batch_midrank_percentile(metric, valid)
    for row in range(metric.shape[0]):
        expected_percentile = train_midrank_percentile(metric[row], valid[row])
        assert np.allclose(
            batch_percentile[row], expected_percentile, equal_nan=True, atol=0.0
        )

    score = rng.integers(0, 4, size=(7, 19)).astype(np.float64)
    cosine = rng.integers(-2, 3, size=(7, 19)).astype(np.float64)
    target_ids = rng.permutation(19)
    batch_best = batch_best_indices(score, cosine, target_ids, valid)
    for row in range(score.shape[0]):
        expected_best = choose_best_candidate(
            score[row], cosine[row], target_ids, valid[row]
        )
        assert batch_best[row] == (-1 if expected_best is None else expected_best)
    pair_edges = {(42, 43): [], (42, 44): [], (43, 44): []}
    pseudo_fold_plumbing(pair_edges)
    print("RP-A matching benchmark self-test passed")


def main() -> None:
    """执行一次正式规模surrogate matching并仅保存工程资源指标。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output.exists():
        raise FileExistsError(f"benchmark输出已存在: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    device = torch.device(args.device)
    validate_lineage()
    audit = pd.read_csv(ELIGIBLE_ROOT / "seed42_feature_aggregation_audit.csv")
    patients = pd.read_csv(ELIGIBLE_ROOT / "seed42_train_patients.csv", dtype={"patient_id": str})
    ranking = np.load(ELIGIBLE_ROOT / "seed42_patient_ranking.npy", mmap_mode="r")
    metadata = pd.read_csv(SPATIAL_CACHE / "train_metadata.csv", dtype={"patient_id": str})
    spatial = np.load(SPATIAL_CACHE / "train_spatial_features.npy", mmap_mode="r")

    timings = {
        "sae_encoding_seconds": 0.0,
        "decoder_cosine_seconds": 0.0,
        "eligible_seconds": 0.0,
        "spearman_seconds": 0.0,
        "jaccard_seconds": 0.0,
        "spatial_seconds": 0.0,
        "percentile_seconds": 0.0,
        "exact_null_seconds": 0.0,
        "bh_rnn_seconds": 0.0,
        "pseudo_fold_plumbing_seconds": 0.0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sae = load_sae(device)
    print("[benchmark] START SAE spatial encoding", flush=True)
    start = stage_start(device)
    normalized_maps, image_active = encode_spatial_activations(
        sae, spatial, device, args.encoding_batch_size
    )
    timings["sae_encoding_seconds"] = stage_stop(device, start)
    print(
        f"[benchmark] DONE SAE spatial encoding "
        f"{timings['sae_encoding_seconds']:.2f}s",
        flush=True,
    )

    patient_index = {patient_id: idx for idx, patient_id in enumerate(patients.patient_id)}
    image_patient_index = torch.as_tensor(
        [patient_index[patient_id] for patient_id in metadata.patient_id], device=device
    )
    permutations = logical_permutations(HIDDEN_DIM)
    nondead = audit["full_train_nondead"].to_numpy(bool)
    base_eligible = (
        nondead
        & (audit["positive_patient_count"].to_numpy() >= TOP_Q_COUNT)
        & (audit["active_position_frequency"].to_numpy() >= A_MIN)
    )
    start = time.perf_counter()
    logical = {}
    for seed, base_order in permutations.items():
        ids = np.flatnonzero(base_eligible[base_order])
        base_ids = base_order[ids]
        strata = assign_target_strata(
            audit["patient_coverage"].to_numpy()[base_ids],
            audit["active_position_frequency"].to_numpy()[base_ids], ids,
        )
        validate_target_strata(strata)
        logical[seed] = {"ids": ids, "base_ids": base_ids, "strata": strata}
    timings["eligible_seconds"] = time.perf_counter() - start

    start = stage_start(device)
    top_sets = build_top_sets(np.asarray(ranking), patients.patient_id.to_numpy(str))
    positive_ranks = build_positive_ranks(np.asarray(ranking))
    ranking_gpu = torch.from_numpy(np.array(ranking, copy=True)).to(device)
    positive_ranks_gpu = torch.from_numpy(positive_ranks).to(device)
    top_gpu = torch.from_numpy(top_sets).to(device)
    timings["static_index_seconds"] = stage_stop(device, start)

    decoder = torch.nn.functional.normalize(sae.decoder_weight.detach(), dim=1)
    decoder_cosines: dict[tuple[int, int], np.ndarray] = {}
    print("[benchmark] START decoder cosine precompute", flush=True)
    start = stage_start(device)
    for left, right in ((42, 43), (42, 44), (43, 44)):
        left_base = torch.as_tensor(logical[left]["base_ids"], device=device)
        right_base = torch.as_tensor(logical[right]["base_ids"], device=device)
        decoder_cosines[(left, right)] = (
            decoder.index_select(0, left_base) @ decoder.index_select(0, right_base).T
        ).cpu().numpy().astype(np.float32, copy=False)
    timings["decoder_cosine_seconds"] = stage_stop(device, start)
    print(
        f"[benchmark] DONE decoder cosine precompute "
        f"{timings['decoder_cosine_seconds']:.2f}s",
        flush=True,
    )

    pair_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for left, right in ((42, 43), (42, 44), (43, 44)):
        left_info, right_info = logical[left], logical[right]
        cosine = decoder_cosines[(left, right)]
        forward = directional_hypotheses_streaming(
            left, right, left_info, right_info, cosine,
            ranking_gpu, positive_ranks_gpu, top_gpu,
            normalized_maps, image_active, image_patient_index, len(patients),
            args.source_block, args.target_block, device, timings,
        )
        reverse = directional_hypotheses_streaming(
            right, left, right_info, left_info, cosine.T,
            ranking_gpu, positive_ranks_gpu, top_gpu,
            normalized_maps, image_active, image_patient_index, len(patients),
            args.source_block, args.target_block, device, timings,
        )
        start = time.perf_counter()
        hypotheses = forward + reverse
        rejected, _ = benjamini_hochberg(hypotheses, q=0.05)
        pair_edges[(left, right)] = reciprocal_edges(hypotheses, rejected)
        timings["bh_rnn_seconds"] += time.perf_counter() - start
        del cosine, forward, reverse, hypotheses, rejected

    start = time.perf_counter()
    pseudo_fold_plumbing(pair_edges)
    timings["pseudo_fold_plumbing_seconds"] = time.perf_counter() - start

    precompute_seconds = sum(
        timings[key] for key in (
            "sae_encoding_seconds", "decoder_cosine_seconds", "static_index_seconds"
        )
    )
    replicate_seconds = sum(
        value for key, value in timings.items()
        if key not in {
            "sae_encoding_seconds", "decoder_cosine_seconds", "static_index_seconds"
        }
    )
    payload = {
        "benchmark_only": True,
        "statistical_outputs_persisted": False,
        "no_threshold_calibration": True,
        "no_protocol_decision_based_on_metric_values": True,
        "surrogate_replicate": "full_train_unit_patient_weights",
        "implementation": {
            "encoding_batch_size": args.encoding_batch_size,
            "source_block": args.source_block,
            "target_block": args.target_block,
            "candidate_materialization": "source_block_by_all_target_then_discard",
            "percentile": "batched_exact_midrank_equivalent_to_frozen_core",
        },
        "completed": True,
        "device": str(device),
        "feature_count": int(sum(len(logical[s]["ids"]) for s in LOGICAL_SEEDS) // 3),
        "patient_count": int(len(patients)),
        "image_count": int(len(metadata)),
        "precompute": {
            "sae_encoding_seconds": timings["sae_encoding_seconds"],
            "decoder_cosine_seconds": timings["decoder_cosine_seconds"],
            "static_index_seconds": timings["static_index_seconds"],
            "total_seconds": precompute_seconds,
        },
        "replicate": {
            key: value for key, value in timings.items()
            if key not in {
                "sae_encoding_seconds", "decoder_cosine_seconds", "static_index_seconds"
            }
        } | {"total_seconds": replicate_seconds},
        "peak_rss_gb": peak_rss_gb(),
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
            if device.type == "cuda" else 0.0
        ),
        "scratch_bytes": 0,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"benchmark完成: {args.output}")


if __name__ == "__main__":
    main()
