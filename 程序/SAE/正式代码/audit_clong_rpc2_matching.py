#!/usr/bin/env python3
"""执行RP-C2 train-only、outcome-blind matching feasibility audit。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clong_rpb_core import file_sha256
from clong_rpc2_matching_core import (
    CALIPERS,
    audit_target_geometry,
    empirical_midrank,
    patient_equal_covariate,
    smallest_global_feasible_caliper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "结果/SAE/RP_C2_Matching_Feasibility_20260825"
STUDY_ROOT = ROOT / "study_objects"
CACHE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824/analysis_cache"
PROTOCOL_PATH = CODE_ROOT / "rpc2_matching_audit_protocol_v1.json"
SEEDS = (42, 43, 44)


def main() -> None:
    """审计447个target-seed的分位L∞匹配池，不抽controls、不计算干预。"""
    target = ROOT / "formal_audit"
    if target.exists():
        raise FileExistsError(f"matching audit已存在: {target}")
    study = pd.read_csv(STUDY_ROOT / "rpc2_study_objects.csv", dtype={"anchor_id": str})
    study_config = json.loads((STUDY_ROOT / "config.json").read_text(encoding="utf-8"))
    if file_sha256(STUDY_ROOT / "rpc2_study_objects.csv") != study_config[
        "study_object_manifest_sha256"
    ]:
        raise RuntimeError("study-object manifest SHA不一致")
    forbidden = {
        "delta_margin", "label_auc", "source_risk", "sharedness", "rpc1_effect_rank",
    }
    if forbidden.intersection(study.columns):
        raise RuntimeError("outcome-blind study manifest含禁止字段")

    rows: list[dict[str, object]] = []
    input_sha: dict[str, str] = {
        "study_manifest": file_sha256(STUDY_ROOT / "rpc2_study_objects.csv"),
        "study_config": file_sha256(STUDY_ROOT / "config.json"),
        "protocol": file_sha256(PROTOCOL_PATH),
    }
    for seed in SEEDS:
        cache = CACHE_ROOT / f"seed{seed}"
        eligible_ids = np.load(cache / "train_eligible_ids.npy").astype(np.int64)
        raw_columns = []
        for name, filename in (
            ("active_frequency", "train_active_frequency.npy"),
            ("activation_mass", "train_mass.npy"),
            ("representation_energy", "train_energy.npy"),
        ):
            path = cache / filename
            matrix = np.load(path, mmap_mode="r")
            if matrix.shape[0] != 1212:
                raise RuntimeError(f"seed{seed} {name}患者数不是1212")
            raw_columns.append(patient_equal_covariate(matrix, eligible_ids))
            input_sha[f"seed{seed}_{name}"] = file_sha256(path)
        input_sha[f"seed{seed}_eligible_ids"] = file_sha256(cache / "train_eligible_ids.npy")
        raw = np.column_stack(raw_columns)
        percentiles = np.column_stack([
            empirical_midrank(raw[:, column]) for column in range(raw.shape[1])
        ])
        study_feature_ids = set(study[f"feature_{seed}"].astype(int))
        if len(study_feature_ids) != len(study):
            raise RuntimeError(f"seed{seed} study-object Feature ID不唯一")
        lookup = {int(feature): index for index, feature in enumerate(eligible_ids)}
        for item in study.itertuples(index=False):
            feature_id = int(getattr(item, f"feature_{seed}"))
            position = lookup.get(feature_id)
            if position is None:
                raise RuntimeError(f"seed{seed} target Feature不在eligible universe: {feature_id}")
            record: dict[str, object] = {
                "anchor_id": str(item.anchor_id),
                "primary_candidate": bool(item.primary_candidate),
                "low_effect_control": bool(item.low_effect_control),
                "post_rpc1_secondary_exploratory": bool(
                    item.post_rpc1_secondary_exploratory
                ),
                "sae_seed": seed,
                "target_feature_id": feature_id,
                "active_frequency_raw": float(raw[position, 0]),
                "activation_mass_raw": float(raw[position, 1]),
                "representation_energy_raw": float(raw[position, 2]),
                "active_frequency_percentile": float(percentiles[position, 0]),
                "activation_mass_percentile": float(percentiles[position, 1]),
                "representation_energy_percentile": float(percentiles[position, 2]),
            }
            record.update(audit_target_geometry(
                feature_id, eligible_ids, percentiles, study_feature_ids,
            ))
            rows.append(record)

    audit = pd.DataFrame(rows).sort_values(["sae_seed", "anchor_id"]).reset_index(drop=True)
    if len(audit) != 447:
        raise RuntimeError(f"matching audit应为149x3=447行，实际{len(audit)}")
    recommended = smallest_global_feasible_caliper(audit)
    summary_rows = []
    for seed, group in audit.groupby("sae_seed", sort=True):
        for caliper in CALIPERS:
            column = f"pool_n_c{int(round(caliper * 1000)):04d}"
            summary_rows.append({
                "sae_seed": int(seed),
                "caliper": float(caliper),
                "pool_min": int(group[column].min()),
                "pool_p05_lower": int(group[column].quantile(0.05, interpolation="lower")),
                "pool_median": float(group[column].median()),
                "worst_anchor_id": str(group.loc[group[column].idxmin(), "anchor_id"]),
            })
    target.mkdir(parents=True)
    detail_path = target / "matching_geometry.csv"
    summary_path = target / "matching_geometry_summary.csv"
    audit.to_csv(detail_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    payload = {
        "status": (
            "matching_geometry_feasible_pending_caliper_freeze"
            if recommended is not None else "matching_calibration_infeasible"
        ),
        "unique_study_objects": int(study.anchor_id.nunique()),
        "target_seed_rows": int(len(audit)),
        "role_flag_counts": study_config["counts"],
        "recommended_smallest_global_feasible_caliper": recommended,
        "required_controls_per_target": 100,
        "input_sha256": input_sha,
        "output_sha256": {
            "matching_geometry": file_sha256(detail_path),
            "matching_geometry_summary": file_sha256(summary_path),
        },
        "intervention_or_model_forward_executed": False,
        "rpc1_effect_fields_read_by_audit": False,
        "control_ids_sampled": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (target / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
