#!/usr/bin/env python3
"""按三seed技术证据构建RP-B complete-link Feature families。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from clong_rpb_core import (
    ACTIVE_EPS,
    SEEDS,
    choose_representative,
    complete_link_families,
    file_sha256,
    load_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
CACHE_ROOT = SOURCE_ROOT / "analysis_cache"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825"
MASTER_ROOT = OUTPUT_ROOT / "anchor_master"
PROTOCOL_PATH = Path(__file__).with_name("rpb_technical_protocol_v1.json")


def union_positive_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """在union-positive患者上计算Spearman，单侧零保留。"""
    selected = (left > ACTIVE_EPS) | (right > ACTIVE_EPS)
    if selected.sum() < 2:
        return float("nan")
    a = rankdata(left[selected], method="average")
    b = rankdata(right[selected], method="average")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_jaccard(left: np.ndarray, right: np.ndarray, patient_ids: np.ndarray, count: int) -> float:
    """计算固定Top患者集合Jaccard。"""
    left_order = np.lexsort((patient_ids, -left))[:count]
    right_order = np.lexsort((patient_ids, -right))[:count]
    intersection = len(set(left_order.tolist()) & set(right_order.tolist()))
    return intersection / (2 * count - intersection)


def spatial_for_pairs(
    activations: np.ndarray,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    image_patient_index: np.ndarray,
    patient_count: int,
    minimum_support: int,
) -> np.ndarray:
    """分块计算同seed候选pair的正式患者平衡空间相似度。"""
    output = np.full(len(left_ids), np.nan, dtype=np.float64)
    for start in range(0, len(left_ids), 64):
        stop = min(start + 64, len(left_ids))
        left = np.asarray(activations[:, :, left_ids[start:stop]], dtype=np.float32)
        right = np.asarray(activations[:, :, right_ids[start:stop]], dtype=np.float32)
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        left_active = left_norm > ACTIVE_EPS
        right_active = right_norm > ACTIVE_EPS
        both = left_active & right_active
        union = left_active | right_active
        cosine = np.divide(
            (left * right).sum(axis=1),
            left_norm * right_norm,
            out=np.zeros_like(left_norm),
            where=both,
        )
        patient_sum = np.zeros((patient_count, stop - start), dtype=np.float64)
        patient_n = np.zeros((patient_count, stop - start), dtype=np.float64)
        np.add.at(patient_sum, image_patient_index, cosine)
        np.add.at(patient_n, image_patient_index, union)
        valid = patient_n > 0
        patient_values = np.divide(patient_sum, patient_n, out=np.zeros_like(patient_sum), where=valid)
        support = valid.sum(axis=0)
        values = np.divide(
            (patient_values * valid).sum(axis=0),
            support,
            out=np.full(stop - start, np.nan),
            where=support > 0,
        )
        values[support < minimum_support] = np.nan
        output[start:stop] = values
    return output


def main() -> None:
    """计算候选边、complete-link families和冻结词典序代表。"""
    protocol = load_protocol(PROTOCOL_PATH)
    target = OUTPUT_ROOT / "technical_families"
    if target.exists():
        raise FileExistsError(f"RP-B family结果已存在: {target}")
    master_path = MASTER_ROOT / "anchor_master.csv"
    if not master_path.is_file():
        raise FileNotFoundError("必须先完成RP-B1 anchor master")
    master = pd.read_csv(master_path, dtype={"anchor_id": str})
    anchors = master.anchor_id.tolist()
    thresholds = protocol["technical_family"]["thresholds"]
    minimum_seed = int(protocol["technical_family"]["seed_consensus_minimum"])
    top_count = 25

    decoder_pass = np.zeros((len(master), len(master)), dtype=np.int8)
    decoder_values = {}
    seed_data = {}
    for seed in SEEDS:
        root = CACHE_ROOT / f"seed{seed}"
        feature_ids = master[f"feature_{seed}"].to_numpy(int)
        decoder = np.asarray(np.load(root / "decoder_weight.npy", mmap_mode="r")[feature_ids], dtype=np.float32)
        decoder /= np.linalg.norm(decoder, axis=1, keepdims=True).clip(min=1e-12)
        cosine = decoder @ decoder.T
        decoder_values[seed] = cosine
        decoder_pass += cosine >= float(thresholds["signed_decoder_cosine"])
        seed_data[seed] = {
            "root": root,
            "feature_ids": feature_ids,
            "ranking": np.asarray(np.load(root / "train_ranking.npy", mmap_mode="r")[:, feature_ids]),
        }
    candidate_left, candidate_right = np.where(np.triu(decoder_pass >= minimum_seed, k=1))
    decoder_candidate_count = len(candidate_left)

    patients = pd.read_csv(CACHE_ROOT / "seed42/train_patients.csv", dtype={"patient_id": str})
    patient_ids = patients.patient_id.to_numpy(str)
    metric_rows = []
    pre_spatial = np.zeros(len(candidate_left), dtype=np.int8)
    per_seed_metrics = {}
    for seed in SEEDS:
        ranking = seed_data[seed]["ranking"]
        spearman = np.empty(len(candidate_left), dtype=np.float64)
        jaccard = np.empty(len(candidate_left), dtype=np.float64)
        for index, (left, right) in enumerate(zip(candidate_left, candidate_right)):
            spearman[index] = union_positive_spearman(ranking[:, left], ranking[:, right])
            jaccard[index] = top_jaccard(ranking[:, left], ranking[:, right], patient_ids, top_count)
        cpu_pass = (
            (decoder_values[seed][candidate_left, candidate_right] >= float(thresholds["signed_decoder_cosine"]))
            & (spearman >= float(thresholds["patient_union_positive_spearman"]))
            & (jaccard >= float(thresholds["top25_patient_jaccard"]))
        )
        pre_spatial += cpu_pass
        per_seed_metrics[seed] = {"spearman": spearman, "jaccard": jaccard, "cpu_pass": cpu_pass}
    selected = pre_spatial >= minimum_seed
    candidate_left = candidate_left[selected]
    candidate_right = candidate_right[selected]

    final_pass = np.zeros(len(candidate_left), dtype=np.int8)
    image_frame = pd.read_csv(CACHE_ROOT / "seed42/train_images.csv", dtype={"patient_id": str})
    patient_lookup = {patient: index for index, patient in enumerate(patient_ids)}
    image_patient_index = np.asarray([patient_lookup[patient] for patient in image_frame.patient_id], dtype=np.int64)
    original_indices = np.flatnonzero(selected)
    for seed in SEEDS:
        data = seed_data[seed]
        spatial = spatial_for_pairs(
            np.load(data["root"] / "train_image_activations.npy", mmap_mode="r"),
            data["feature_ids"][candidate_left],
            data["feature_ids"][candidate_right],
            image_patient_index,
            len(patients),
            int(protocol["technical_family"]["spatial_minimum_evaluable_patients"]),
        )
        cpu_pass = per_seed_metrics[seed]["cpu_pass"][original_indices]
        seed_pass = cpu_pass & (spatial >= float(thresholds["patient_balanced_spatial_similarity"]))
        final_pass += seed_pass
        for index, (left, right) in enumerate(zip(candidate_left, candidate_right)):
            metric_rows.append({
                "anchor_left": anchors[left],
                "anchor_right": anchors[right],
                "seed": seed,
                "signed_decoder_cosine": float(decoder_values[seed][left, right]),
                "patient_spearman": float(per_seed_metrics[seed]["spearman"][original_indices[index]]),
                "top25_patient_jaccard": float(per_seed_metrics[seed]["jaccard"][original_indices[index]]),
                "spatial_similarity": float(spatial[index]),
                "all_four_pass": bool(seed_pass[index]),
            })
    edge_mask = final_pass >= minimum_seed
    edges = {
        tuple(sorted((anchors[left], anchors[right])))
        for left, right in zip(candidate_left[edge_mask], candidate_right[edge_mask])
    }
    families = complete_link_families(anchors, edges)

    assignments = []
    for index, family in enumerate(families, start=1):
        family_rows = master[master.anchor_id.isin(family)].to_dict("records")
        representative = choose_representative(family_rows)
        family_id = f"TF{index:04d}"
        for anchor in family:
            assignments.append({
                "anchor_id": anchor,
                "technical_family_id": family_id,
                "family_size": len(family),
                "family_representative": anchor == representative,
                "representative_anchor_id": representative,
            })
    assignments = pd.DataFrame(assignments).sort_values("anchor_id")
    edge_frame = pd.DataFrame([
        {"anchor_left": left, "anchor_right": right, "technical_family_edge": True}
        for left, right in sorted(edges)
    ], columns=["anchor_left", "anchor_right", "technical_family_edge"])
    metric_frame = pd.DataFrame(metric_rows, columns=[
        "anchor_left", "anchor_right", "seed", "signed_decoder_cosine",
        "patient_spearman", "top25_patient_jaccard", "spatial_similarity",
        "all_four_pass",
    ])

    target.mkdir(parents=True)
    assignments.to_csv(target / "anchor_family_assignments.csv", index=False)
    edge_frame.to_csv(target / "technical_family_edges.csv", index=False)
    metric_frame.to_csv(target / "technical_family_pair_metrics.csv", index=False)
    master.merge(assignments, on="anchor_id", validate="one_to_one").to_csv(
        target / "anchor_master_with_families.csv", index=False,
    )
    sizes = assignments.drop_duplicates("technical_family_id").family_size
    config = {
        "stage": "RP-B2 technical Feature families",
        "status": "completed",
        "is_medical_concept_family": False,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "anchor_count": len(assignments),
        "candidate_pair_count_after_decoder": int(decoder_candidate_count),
        "candidate_pair_count_before_spatial": int(len(candidate_left)),
        "technical_family_edge_count": len(edges),
        "technical_family_count": len(families),
        "nonsingleton_family_count": int((sizes > 1).sum()),
        "anchors_in_nonsingleton_families": int(sizes[sizes > 1].sum()),
        "largest_family_size": int(sizes.max()),
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "input_sha256": {"anchor_master.csv": file_sha256(master_path)},
    }
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RP-B2完成: {target}")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
