#!/usr/bin/env python3
"""构建RP-B 1150个development anchor的技术主表。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from clong_rpb_core import (
    SEEDS,
    bh_adjust,
    classify_sharedness,
    fast_delong_auc_ci,
    file_sha256,
    load_protocol,
    source_tests,
    stratified_bootstrap_contrasts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824"
CACHE_ROOT = SOURCE_ROOT / "analysis_cache"
MATCHING_ROOT = SOURCE_ROOT / "full_train_matching/formal"
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825"
PROTOCOL_PATH = Path(__file__).with_name("rpb_technical_protocol_v1.json")


def load_seed(seed: int, feature_ids: np.ndarray) -> dict:
    """加载一个seed在anchor Feature上的train患者统计。"""
    root = CACHE_ROOT / f"seed{seed}"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if int(config["seed"]) != seed or bool(config["debug"]):
        raise RuntimeError(f"seed{seed}缓存身份不一致")
    patients = pd.read_csv(root / "train_patients.csv", dtype={"patient_id": str})
    return {
        "root": root,
        "config": config,
        "patients": patients,
        "presence": np.asarray(np.load(root / "train_presence.npy", mmap_mode="r")[:, feature_ids]),
        "ranking": np.asarray(np.load(root / "train_ranking.npy", mmap_mode="r")[:, feature_ids]),
        "mass": np.asarray(np.load(root / "train_mass.npy", mmap_mode="r")[:, feature_ids]),
        "energy": np.asarray(np.load(root / "train_energy.npy", mmap_mode="r")[:, feature_ids]),
    }


def matching_evidence(anchors: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """提取每个anchor六个有向假设及其最弱edge score。"""
    hypotheses = pd.read_csv(MATCHING_ROOT / "best_directed_hypotheses.csv")
    lookup = {
        (int(row.source_seed), int(row.source_feature_id), int(row.target_seed), int(row.target_feature_id)): row
        for row in hypotheses.itertuples(index=False)
    }
    output = []
    worst = {}
    for anchor in anchors.itertuples(index=False):
        rows = []
        features = {seed: int(getattr(anchor, f"feature_{seed}")) for seed in SEEDS}
        for left, right in ((42, 43), (42, 44), (43, 44)):
            for source, target in ((left, right), (right, left)):
                key = (source, features[source], target, features[target])
                if key not in lookup:
                    raise RuntimeError(f"anchor {anchor.anchor_id}缺少有向matching证据: {key}")
                row = lookup[key]
                if not bool(row.bh_rejected):
                    raise RuntimeError(f"anchor {anchor.anchor_id}包含未通过BH的方向")
                record = {"anchor_id": anchor.anchor_id, **row._asdict()}
                rows.append(record)
                output.append(record)
        worst[str(anchor.anchor_id)] = min(float(row["edge_score"]) for row in rows)
    return pd.DataFrame(output), worst


def main() -> None:
    """只读冻结train缓存并落盘anchor主表、长表与来源审计。"""
    protocol = load_protocol(PROTOCOL_PATH)
    target = OUTPUT_ROOT / "anchor_master"
    if target.exists():
        raise FileExistsError(f"RP-B anchor master已存在: {target}")
    anchors = pd.read_csv(MATCHING_ROOT / "development_anchors.csv", dtype={"anchor_id": str})
    expected = int(protocol["candidate_pool"]["expected_anchor_count"])
    if len(anchors) != expected or anchors.anchor_id.tolist() != [f"a{i:05d}" for i in range(expected)]:
        raise RuntimeError("RP-B候选池不是冻结的1150个顺序anchor")
    evidence, worst_scores = matching_evidence(anchors)
    feature_ids = {seed: anchors[f"feature_{seed}"].to_numpy(int) for seed in SEEDS}
    seeds = {seed: load_seed(seed, feature_ids[seed]) for seed in SEEDS}
    reference_patients = seeds[42]["patients"]
    for seed in (43, 44):
        if not reference_patients.equals(seeds[seed]["patients"]):
            raise RuntimeError("三seed患者表不逐位一致")

    labels = reference_patients.label.to_numpy(np.int8)
    sources = reference_patients.source.to_numpy(str)
    rng = np.random.Generator(np.random.PCG64(20260825))
    bootstrap_count = int(protocol["sharedness"]["bootstrap_replicates"])
    seed_stats, source_frames = [], []
    energy_percentiles = {}
    ranking_percentiles = []

    for seed in SEEDS:
        data = seeds[seed]
        intervals = stratified_bootstrap_contrasts(
            data["presence"], data["mass"], labels, bootstrap_count, rng,
        )
        raw_source = pd.DataFrame(source_tests(
            data["presence"], data["mass"], labels, sources,
        ))
        raw_source["anchor_id"] = anchors.anchor_id.to_numpy()[raw_source.feature_column]
        raw_source["seed"] = seed
        for label in (0, 1):
            selected = raw_source.label == label
            raw_source.loc[selected, "presence_q"] = bh_adjust(raw_source.loc[selected, "presence_p"].to_numpy())
            raw_source.loc[selected, "mass_q"] = bh_adjust(raw_source.loc[selected, "mass_p"].to_numpy())
        settings = protocol["source_audit"]
        raw_source["presence_risk"] = (
            (raw_source.presence_q <= float(settings["bh_fdr_q"]))
            & (raw_source.presence_range >= float(settings["presence_practical_range"]))
        )
        raw_source["mass_risk"] = (
            (raw_source.mass_q <= float(settings["bh_fdr_q"]))
            & (raw_source.mass_epsilon_squared >= float(settings["mass_practical_effect"]))
        )
        source_frames.append(raw_source)

        cancer = labels == 1
        noncancer = labels == 0
        energy_mean = data["energy"].mean(axis=0)
        energy_percentiles[seed] = rankdata(energy_mean, method="average") / len(anchors)
        ranking_percentiles.append(rankdata(data["ranking"], axis=0, method="average") / len(labels))
        for column, anchor in enumerate(anchors.itertuples(index=False)):
            auc, auc_low, auc_high = fast_delong_auc_ci(labels, data["ranking"][:, column])
            cancer_mass = float(data["mass"][cancer, column].mean())
            noncancer_mass = float(data["mass"][noncancer, column].mean())
            denominator = cancer_mass + noncancer_mass
            seed_stats.append({
                "anchor_id": anchor.anchor_id,
                "seed": seed,
                "feature_id": int(feature_ids[seed][column]),
                "cancer_coverage": float(data["presence"][cancer, column].mean()),
                "noncancer_coverage": float(data["presence"][noncancer, column].mean()),
                "overall_coverage": float(data["presence"][:, column].mean()),
                "cancer_mass": cancer_mass,
                "noncancer_mass": noncancer_mass,
                "mass_contrast": (cancer_mass - noncancer_mass) / denominator if denominator > 0 else 0.0,
                "mean_energy": float(energy_mean[column]),
                "energy_percentile": float(energy_percentiles[seed][column]),
                "label_auc": auc,
                "label_auc_ci_low": auc_low,
                "label_auc_ci_high": auc_high,
                "label_effect_2auc_minus_1": 2 * auc - 1,
                "coverage_ci_low": float(intervals["coverage_low"][column]),
                "coverage_ci_high": float(intervals["coverage_high"][column]),
                "mass_ci_low": float(intervals["mass_low"][column]),
                "mass_ci_high": float(intervals["mass_high"][column]),
            })

    seed_frame = pd.DataFrame(seed_stats)
    source_frame = pd.concat(source_frames, ignore_index=True)
    source_seed_risk = source_frame.groupby(["anchor_id", "seed"])[["presence_risk", "mass_risk"]].any()
    source_consensus = source_seed_risk.any(axis=1).groupby("anchor_id").sum()
    source_minimum = int(protocol["source_audit"]["source_risk_seed_consensus_minimum"])
    median_ranking_percentile = np.median(np.stack(ranking_percentiles), axis=0)
    patient_ids = reference_patients.patient_id.to_numpy(str)

    master_rows = []
    for column, anchor in enumerate(anchors.itertuples(index=False)):
        rows = seed_frame[seed_frame.anchor_id == anchor.anchor_id].to_dict("records")
        top_ids = {}
        for label, name in ((1, "cancer"), (0, "noncancer")):
            selected = np.flatnonzero(labels == label)
            order = np.lexsort((patient_ids[selected], -median_ranking_percentile[selected, column]))
            top_ids[name] = ";".join(patient_ids[selected[order[:10]]])
        master_rows.append({
            "anchor_id": anchor.anchor_id,
            "feature_42": int(anchor.feature_42),
            "feature_43": int(anchor.feature_43),
            "feature_44": int(anchor.feature_44),
            "evidence_level": "development_3seed_exploratory",
            "sharedness_class": classify_sharedness(rows, protocol),
            "stability_worst_edge_score": float(worst_scores[anchor.anchor_id]),
            "overall_patient_coverage": float(np.median([row["overall_coverage"] for row in rows])),
            "cancer_coverage": float(np.median([row["cancer_coverage"] for row in rows])),
            "noncancer_coverage": float(np.median([row["noncancer_coverage"] for row in rows])),
            "cancer_mass": float(np.median([row["cancer_mass"] for row in rows])),
            "noncancer_mass": float(np.median([row["noncancer_mass"] for row in rows])),
            "label_auc": float(np.median([row["label_auc"] for row in rows])),
            "energy_percentile": float(np.median([row["energy_percentile"] for row in rows])),
            "source_risk": bool(source_consensus.get(anchor.anchor_id, 0) >= source_minimum),
            "source_risk_seed_count": int(source_consensus.get(anchor.anchor_id, 0)),
            "top10_cancer_patient_ids": top_ids["cancer"],
            "top10_noncancer_patient_ids": top_ids["noncancer"],
        })
    master = pd.DataFrame(master_rows)

    target.mkdir(parents=True)
    master.to_csv(target / "anchor_master.csv", index=False)
    seed_frame.to_csv(target / "anchor_seed_statistics.csv", index=False)
    source_frame.to_csv(target / "anchor_source_audit.csv", index=False)
    evidence.to_csv(target / "anchor_matching_evidence.csv", index=False)
    config = {
        "stage": "RP-B1 anchor master/sharedness/source audit",
        "status": "completed",
        "is_formal_rpa_result": False,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "anchor_count": len(master),
        "sharedness_counts": master.sharedness_class.value_counts().sort_index().to_dict(),
        "source_risk_count": int(master.source_risk.sum()),
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
        "input_sha256": {
            "anchors": file_sha256(MATCHING_ROOT / "development_anchors.csv"),
            "hypotheses": file_sha256(MATCHING_ROOT / "best_directed_hypotheses.csv"),
            **{f"seed{seed}_config": file_sha256(CACHE_ROOT / f"seed{seed}/config.json") for seed in SEEDS},
        },
    }
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RP-B1完成: {target}")
    print(f"sharedness={config['sharedness_counts']} source-risk={config['source_risk_count']}")


if __name__ == "__main__":
    main()
