#!/usr/bin/env python3
"""汇总RP-B1/B2并生成无加权总分的多轨技术队列。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from clong_rpb_core import SEEDS, file_sha256, load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825"
MASTER_ROOT = OUTPUT_ROOT / "anchor_master"
FAMILY_ROOT = OUTPUT_ROOT / "technical_families"
PROTOCOL_PATH = Path(__file__).with_name("rpb_technical_protocol_v1.json")
PRIORITY_PATH = Path(__file__).with_name("rpb_priority_protocol_v1.json")


def main() -> None:
    """核验1150行闭环，生成技术轨道、诊断和SHA。"""
    load_protocol(PROTOCOL_PATH)
    priority = json.loads(PRIORITY_PATH.read_text(encoding="utf-8"))
    if priority.get("status") != "frozen_after_numeric_audit_before_images_2026-08-25":
        raise RuntimeError("RP-B priority协议未冻结")
    target = OUTPUT_ROOT / "summary"
    if target.exists():
        raise FileExistsError(f"RP-B汇总已存在: {target}")

    master_path = FAMILY_ROOT / "anchor_master_with_families.csv"
    master = pd.read_csv(master_path, dtype={"anchor_id": str})
    if len(master) != 1150 or master.anchor_id.nunique() != 1150:
        raise RuntimeError("RP-B汇总必须恰好包含1150个唯一anchor")
    rare_threshold = float(priority["extremely_rare_overall_coverage_below"])
    master["extremely_rare"] = master.overall_patient_coverage < rare_threshold
    master["review_track"] = "content_candidate"
    master.loc[master.source_risk, "review_track"] = "source_risk"
    master.loc[master.extremely_rare, "review_track"] = "rare"
    master.loc[~master.family_representative, "review_track"] = "technical_family_member"

    master["stability_rank"] = rankdata(-master.stability_worst_edge_score, method="min").astype(int)
    master["coverage_rank"] = rankdata(-master.overall_patient_coverage, method="min").astype(int)
    master["energy_rank"] = rankdata(-master.energy_percentile, method="min").astype(int)
    master["label_auc_distance_rank"] = rankdata(-np.abs(master.label_auc - 0.5), method="min").astype(int)
    track_order = {name: index for index, name in enumerate(priority["review_tracks_in_order"])}
    master["_track"] = master.review_track.map(track_order)
    master = master.sort_values([
        "_track", "stability_worst_edge_score", "overall_patient_coverage",
        "energy_percentile", "anchor_id",
    ], ascending=[True, False, False, False, True]).reset_index(drop=True)
    master["track_order"] = master.groupby("review_track").cumcount() + 1
    master = master.drop(columns="_track")

    second_best_values = []
    decoder_matrices = []
    for seed in SEEDS:
        feature_ids = master.set_index("anchor_id").loc[
            sorted(master.anchor_id), f"feature_{seed}"
        ].to_numpy(int)
        decoder = np.asarray(np.load(
            SOURCE_ROOT / f"analysis_cache/seed{seed}/decoder_weight.npy", mmap_mode="r"
        )[feature_ids], dtype=np.float32)
        decoder /= np.linalg.norm(decoder, axis=1, keepdims=True).clip(min=1e-12)
        decoder_matrices.append(decoder @ decoder.T)
    stack = np.stack(decoder_matrices)
    second = np.partition(stack, 1, axis=0)[1]
    triangle = np.triu_indices(1150, 1)
    second_best_values = second[triangle]

    target.mkdir(parents=True)
    queue_path = target / "anchor_technical_review_tracks.csv"
    master.to_csv(queue_path, index=False)
    master.groupby(["review_track", "sharedness_class"], observed=True).size().rename("count").reset_index().to_csv(
        target / "review_track_counts.csv", index=False,
    )
    result = {
        "stage": "RP-B technical summary",
        "status": "completed",
        "is_formal_rpa_result": False,
        "anchor_count": len(master),
        "sharedness_counts": master.sharedness_class.value_counts().sort_index().to_dict(),
        "source_risk_count": int(master.source_risk.sum()),
        "review_track_counts": master.review_track.value_counts().sort_index().to_dict(),
        "technical_family_count": int(master.technical_family_id.nunique()),
        "nonsingleton_family_count": int(master.loc[master.family_size > 1, "technical_family_id"].nunique()),
        "maximum_second_highest_cross_seed_decoder_cosine_between_distinct_anchors": float(second_best_values.max()),
        "second_highest_cross_seed_decoder_cosine_quantiles": {
            str(q): float(np.quantile(second_best_values, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        "coverage_quantiles": {
            str(q): float(master.overall_patient_coverage.quantile(q)) for q in (0.0, 0.01, 0.05, 0.5, 0.95, 1.0)
        },
        "train_only": True,
        "images_exported": False,
        "medical_naming_performed": False,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "priority_protocol_sha256": file_sha256(PRIORITY_PATH),
        "input_sha256": {
            "anchor_master_with_families.csv": file_sha256(master_path),
            "anchor_master_config.json": file_sha256(MASTER_ROOT / "config.json"),
            "technical_families_config.json": file_sha256(FAMILY_ROOT / "config.json"),
        },
    }
    summary_path = target / "technical_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    files = (queue_path, target / "review_track_counts.csv", summary_path)
    (target / "SHA256SUMS.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in files), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
