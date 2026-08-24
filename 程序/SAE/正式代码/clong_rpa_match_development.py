#!/usr/bin/env python3
"""执行RP-A 42/43/44 full-train matching与严格anchor构建。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from benchmark_clong_rpa_matching import (
    batch_best_indices,
    batch_midrank_percentile,
    spearman_matrix,
)
from clong_rpa_artifacts import (
    validate_anchors,
    validate_best_hypotheses,
    validate_edges,
)
from clong_rpa_development_core import (
    PAIR_ORDER,
    all_pseudo_fold_metrics,
    strict_three_cliques,
)
from clong_rpa_null_fdr import (
    DirectedHypothesis,
    benjamini_hochberg,
    exact_conditional_search_p,
    raw_direction_gates_pass,
    reciprocal_edges,
)
from clong_rpa_prepare_seed import OUTPUT_ROOT as CACHE_ROOT
from clong_rpa_train_development import OUTPUT_ROOT as DEVELOPMENT_ROOT
from clong_s2c_matryoshka import file_sha256
from clong_s2b_core import ACTIVE_EPS


OUTPUT_ROOT = DEVELOPMENT_ROOT / "full_train_matching"
FORMAL_TOP_COUNT = 25
FORMAL_SPATIAL_SUPPORT = 25


def parse_args() -> argparse.Namespace:
    """解析设备、分块和debug缓存版本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-block", type=int, default=32)
    parser.add_argument("--target-block", type=int, default=256)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-cache-version", default="debug_v6/p2_e1")
    return parser.parse_args()


def cache_directory(seed: int, args: argparse.Namespace) -> Path:
    """返回正式或debug的单seed分析缓存目录。"""
    root = CACHE_ROOT / args.debug_cache_version if args.debug else CACHE_ROOT
    return root / f"seed{seed}"


def load_seed(seed: int, args: argparse.Namespace) -> dict:
    """加载并核验一个seed的train缓存、eligible和decoder。"""
    directory = cache_directory(seed, args)
    if not directory.is_dir():
        raise FileNotFoundError(f"缺少seed{seed}分析缓存: {directory}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if int(config["seed"]) != seed or bool(config["debug"]) != bool(args.debug):
        raise RuntimeError("分析缓存seed/debug标记不一致")
    patients = pd.read_csv(directory / "train_patients.csv", dtype={"patient_id": str})
    images = pd.read_csv(directory / "train_images.csv", dtype={"patient_id": str})
    return {
        "directory": directory,
        "config": config,
        "patients": patients,
        "images": images,
        "presence": np.load(directory / "train_presence.npy", mmap_mode="r"),
        "ranking": np.load(directory / "train_ranking.npy", mmap_mode="r"),
        "mass": np.load(directory / "train_mass.npy", mmap_mode="r"),
        "energy": np.load(directory / "train_energy.npy", mmap_mode="r"),
        "active_frequency": np.load(directory / "train_active_frequency.npy", mmap_mode="r"),
        "image_activations": np.load(directory / "train_image_activations.npy", mmap_mode="r"),
        "eligible_ids": np.load(directory / "train_eligible_ids.npy"),
        "strata": np.load(directory / "train_eligible_strata.npy"),
        "decoder": np.load(directory / "decoder_weight.npy", mmap_mode="r"),
    }


def validate_cross_seed_alignment(seeds: dict[int, dict]) -> None:
    """三个seed必须共享逐位相同的患者与图像顺序。"""
    reference = seeds[42]
    for seed in (43, 44):
        if reference["patients"].patient_id.tolist() != seeds[seed]["patients"].patient_id.tolist():
            raise RuntimeError(f"seed{seed}患者顺序不一致")
        if reference["images"].relative_path.tolist() != seeds[seed]["images"].relative_path.tolist():
            raise RuntimeError(f"seed{seed}图像顺序不一致")


def build_top_sets(ranking: np.ndarray, patient_ids: np.ndarray, count: int) -> np.ndarray:
    """按score降序、patient ID升序构造固定人数Top集合。"""
    if count > len(patient_ids):
        raise ValueError("Top人数超过患者数")
    result = np.zeros(ranking.shape, dtype=bool)
    for feature in range(ranking.shape[1]):
        order = np.lexsort((patient_ids, -ranking[:, feature]))
        result[order[:count], feature] = True
    return result


def build_positive_ranks(ranking: np.ndarray) -> np.ndarray:
    """仅对正患者计算并列平均秩，零值在union时另行处理。"""
    result = np.zeros_like(ranking, dtype=np.float32)
    for feature in range(ranking.shape[1]):
        positive = ranking[:, feature] > ACTIVE_EPS
        if positive.any():
            result[positive, feature] = rankdata(
                ranking[positive, feature], method="average"
            ).astype(np.float32)
    return result


def jaccard_matrix(
    source_top: torch.Tensor, target_top: torch.Tensor, count: int,
) -> np.ndarray:
    """计算一个source block对全部target的Top-patient Jaccard。"""
    intersection = source_top.T.float() @ target_top.float()
    return (intersection / (2 * count - intersection)).cpu().numpy()


def normalized_maps(seed: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """将图像激活移到设备并按49位置归一化，返回active标记。"""
    values = torch.from_numpy(np.asarray(seed["image_activations"]).copy()).to(device)
    norms = torch.linalg.vector_norm(values, dim=1)
    active = norms > ACTIVE_EPS
    values.div_(norms.clamp_min(1e-12).unsqueeze(1))
    return values, active


def cross_spatial_matrix(
    source_maps: torch.Tensor, source_active: torch.Tensor,
    target_maps: torch.Tensor, target_active: torch.Tensor,
    image_patient_index: torch.Tensor, patient_count: int,
    source_ids: np.ndarray, target_ids: np.ndarray, minimum_support: int,
) -> np.ndarray:
    """按同图位置cosine→患者内均值→患者间均值计算跨seed空间复现。"""
    device = source_maps.device
    source_index = torch.as_tensor(source_ids, device=device)
    target_index = torch.as_tensor(target_ids, device=device)
    source = source_maps.index_select(2, source_index).permute(0, 2, 1)
    target = target_maps.index_select(2, target_index)
    cosine = torch.bmm(source, target)
    union = source_active.index_select(1, source_index).unsqueeze(2) | \
        target_active.index_select(1, target_index).unsqueeze(1)
    cosine.mul_(union)
    shape = (patient_count, len(source_ids), len(target_ids))
    patient_sum = torch.zeros(shape, dtype=torch.float64, device=device)
    patient_n = torch.zeros(shape, dtype=torch.float64, device=device)
    patient_sum.index_add_(0, image_patient_index, cosine.double())
    patient_n.index_add_(0, image_patient_index, union.double())
    valid = patient_n > 0
    patient_values = patient_sum / patient_n.clamp_min(1)
    support = valid.sum(0)
    result = (patient_values * valid).sum(0) / support.clamp_min(1)
    result[support < minimum_support] = torch.nan
    return result.cpu().numpy()


def reduce_direction(
    source_seed: int, target_seed: int,
    source: dict, target: dict, decoder_cosine: np.ndarray,
    source_ranking: torch.Tensor, target_ranking: torch.Tensor,
    source_ranks: torch.Tensor, target_ranks: torch.Tensor,
    source_top: torch.Tensor, target_top: torch.Tensor,
    source_maps: torch.Tensor, source_active: torch.Tensor,
    target_maps: torch.Tensor, target_active: torch.Tensor,
    image_patient_index: torch.Tensor, top_count: int, minimum_support: int,
    source_block: int, target_block: int,
) -> list[dict]:
    """分块完整搜索并仅返回每个source的唯一best hypothesis。"""
    output = []
    source_ids = np.asarray(source["eligible_ids"], dtype=int)
    target_ids = np.asarray(target["eligible_ids"], dtype=int)
    for start in range(0, len(source_ids), source_block):
        stop = min(start + source_block, len(source_ids))
        block_ids = source_ids[start:stop]
        source_index = torch.as_tensor(block_ids, device=source_ranking.device)
        jaccard = jaccard_matrix(
            source_top.index_select(1, source_index),
            target_top.index_select(1, torch.as_tensor(target_ids, device=target_top.device)),
            top_count,
        )
        spearman = spearman_matrix(
            source_ranking.index_select(1, source_index),
            target_ranking.index_select(1, torch.as_tensor(target_ids, device=target_ranking.device)),
            source_ranks.index_select(1, source_index),
            target_ranks.index_select(1, torch.as_tensor(target_ids, device=target_ranks.device)),
            source_block, target_block,
        )
        spatial = cross_spatial_matrix(
            source_maps, source_active, target_maps, target_active,
            image_patient_index, len(source["patients"]), block_ids, target_ids,
            minimum_support,
        )
        raw = [decoder_cosine[start:stop], spearman, jaccard, spatial]
        valid = np.logical_and.reduce([np.isfinite(values) for values in raw])
        percentiles = [batch_midrank_percentile(values, valid) for values in raw]
        behavior = np.full(valid.shape, np.nan, dtype=np.float64)
        score = np.full(valid.shape, np.nan, dtype=np.float64)
        stacked = np.stack(percentiles[1:])
        behavior[valid] = np.median(stacked[:, valid], axis=0)
        score[valid] = np.minimum(percentiles[0][valid], behavior[valid])
        best_indices = batch_best_indices(score, raw[0], target_ids, valid)
        for row, feature in enumerate(block_ids):
            best = int(best_indices[row])
            if best < 0:
                target_feature, p_value = int(target_ids[0]), 1.0
                best_raw = [None] * 4
                best_score = None
            else:
                best_raw = [float(values[row, best]) for values in raw]
                target_feature = int(target_ids[best])
                if raw_direction_gates_pass(*best_raw):
                    p_value = exact_conditional_search_p(
                        percentiles[0][row], behavior[row], target["strata"],
                        valid[row], float(score[row, best]),
                    )
                else:
                    p_value = 1.0
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
        print(f"direction {source_seed}->{target_seed}: {stop}/{len(source_ids)}", flush=True)
    return output


def main() -> None:
    """执行三pair双向matching，保存最小证据和full-train指标。"""
    args = parse_args()
    if not args.debug and args.device != "cuda":
        raise ValueError("正式matching必须使用CUDA")
    target = OUTPUT_ROOT / ("debug_v2" if args.debug else "formal")
    if target.exists():
        raise FileExistsError(f"matching输出已存在，禁止覆盖: {target}")
    device = torch.device(args.device)
    seeds = {seed: load_seed(seed, args) for seed in (42, 43, 44)}
    validate_cross_seed_alignment(seeds)
    target.mkdir(parents=True)
    patient_ids = seeds[42]["patients"].patient_id.to_numpy(str)
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    image_patient_index = torch.as_tensor(
        [patient_index[str(patient)] for patient in seeds[42]["images"].patient_id],
        device=device,
    )
    top_count = min(FORMAL_TOP_COUNT, len(patient_ids)) if args.debug else FORMAL_TOP_COUNT
    support = min(2, len(patient_ids)) if args.debug else FORMAL_SPATIAL_SUPPORT
    static = {}
    for seed, data in seeds.items():
        ranking = np.asarray(data["ranking"])
        static[seed] = {
            "ranking": torch.from_numpy(ranking.copy()).to(device),
            "ranks": torch.from_numpy(build_positive_ranks(ranking)).to(device),
            "top": torch.from_numpy(build_top_sets(ranking, patient_ids, top_count)).to(device),
        }
    all_rows, pair_edges = [], {}
    for left, right in PAIR_ORDER:
        left_maps, left_active = normalized_maps(seeds[left], device)
        right_maps, right_active = normalized_maps(seeds[right], device)
        left_ids = np.asarray(seeds[left]["eligible_ids"], dtype=int)
        right_ids = np.asarray(seeds[right]["eligible_ids"], dtype=int)
        left_decoder = torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(seeds[left]["decoder"])[left_ids]).to(device), dim=1
        )
        right_decoder = torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(seeds[right]["decoder"])[right_ids]).to(device), dim=1
        )
        cosine = (left_decoder @ right_decoder.T).cpu().numpy()
        forward = reduce_direction(
            left, right, seeds[left], seeds[right], cosine,
            static[left]["ranking"], static[right]["ranking"],
            static[left]["ranks"], static[right]["ranks"],
            static[left]["top"], static[right]["top"],
            left_maps, left_active, right_maps, right_active,
            image_patient_index, top_count, support,
            args.source_block, args.target_block,
        )
        reverse = reduce_direction(
            right, left, seeds[right], seeds[left], cosine.T,
            static[right]["ranking"], static[left]["ranking"],
            static[right]["ranks"], static[left]["ranks"],
            static[right]["top"], static[left]["top"],
            right_maps, right_active, left_maps, left_active,
            image_patient_index, top_count, support,
            args.source_block, args.target_block,
        )
        hypotheses = forward + reverse
        frozen = [DirectedHypothesis(
            int(row["source_seed"]), int(row["target_seed"]),
            int(row["source_feature_id"]), int(row["target_feature_id"]),
            float(row["p_value"]),
        ) for row in hypotheses]
        rejected, cutoff = benjamini_hochberg(frozen, q=0.05)
        for row in hypotheses:
            key = (row["source_seed"], row["target_seed"],
                   row["source_feature_id"], row["target_feature_id"])
            row["bh_rejected"] = key in rejected
            row["bh_cutoff"] = cutoff
        edges = reciprocal_edges(frozen, rejected)
        pair_edges[(left, right)] = edges
        all_rows.extend(hypotheses)
        del left_maps, left_active, right_maps, right_active
        if device.type == "cuda":
            torch.cuda.empty_cache()

    anchors = strict_three_cliques(pair_edges)
    eligible = {seed: np.asarray(data["eligible_ids"], dtype=int) for seed, data in seeds.items()}
    mass = {seed: np.asarray(data["mass"]).sum(0) for seed, data in seeds.items()}
    energy = {seed: np.asarray(data["energy"]).sum(0) for seed, data in seeds.items()}
    metrics = all_pseudo_fold_metrics(pair_edges, eligible, mass, energy)
    edge_rows = [{
        "seed_a": left, "feature_a": int(a), "seed_b": right, "feature_b": int(b),
        "both_directions_bh_rejected": True, "reciprocal": True,
    } for (left, right), edges in pair_edges.items() for a, b in edges]
    validate_best_hypotheses(all_rows)
    validate_edges(edge_rows)
    validate_anchors(anchors)
    pd.DataFrame(all_rows).to_csv(target / "best_directed_hypotheses.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(target / "reciprocal_edges.csv", index=False)
    pd.DataFrame(anchors).to_csv(target / "development_anchors.csv", index=False)
    result = {
        "debug": bool(args.debug),
        "eligible_counts": {str(seed): int(len(ids)) for seed, ids in eligible.items()},
        "pair_edge_counts": {f"{a}_{b}": len(rows) for (a, b), rows in pair_edges.items()},
        "strict_anchor_count": len(anchors),
        "minimum_anchor_required": 100,
        "status": (
            "debug_completed" if args.debug else
            "full_train_matching_completed" if len(anchors) >= 100 else
            "development_calibration_infeasible"
        ),
        "fold_metrics": metrics,
        "top_count": top_count,
        "spatial_minimum_support": support,
        "complete_candidate_matrix_persisted": False,
    }
    (target / "full_train_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    names = sorted(path.name for path in target.iterdir() if path.is_file())
    (target / "SHA256SUMS.txt").write_text(
        "\n".join(f"{file_sha256(target / name)}  {name}" for name in names) + "\n",
        encoding="utf-8",
    )
    print(f"RP-A full-train matching完成: {target}; anchors={len(anchors)}")


if __name__ == "__main__":
    main()
