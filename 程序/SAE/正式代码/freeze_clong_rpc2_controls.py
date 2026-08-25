#!/usr/bin/env python3
"""按冻结matching v2生成RP-C2 matched-control manifest。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clong_rpb_core import file_sha256
from clong_rpc2_matching_core import (
    empirical_midrank,
    patient_equal_covariate,
    select_nearest_controls,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "结果/SAE/RP_C2_Matching_Feasibility_20260825"
STUDY_ROOT = ROOT / "study_objects"
CACHE_ROOT = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824/analysis_cache"
AUDIT_ROOT = ROOT / "formal_audit"
OUTPUT_ROOT = ROOT / "matching_v2_freeze_rational_ties"
PROTOCOL_PATH = CODE_ROOT / "rpc2_matching_protocol_v2.json"
SEEDS = (42, 43, 44)


def main() -> None:
    """冻结每个target-seed的最近100个或caliper内全部controls。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"matching v2冻结目录已存在: {OUTPUT_ROOT}")
    audit_summary = json.loads((AUDIT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if audit_summary["status"] != "matching_calibration_infeasible":
        raise RuntimeError("matching v2只接续已完成且不可行的v1 audit")
    study = pd.read_csv(STUDY_ROOT / "rpc2_study_objects.csv", dtype={"anchor_id": str})
    study_config = json.loads((STUDY_ROOT / "config.json").read_text(encoding="utf-8"))
    if file_sha256(STUDY_ROOT / "rpc2_study_objects.csv") != study_config[
        "study_object_manifest_sha256"
    ]:
        raise RuntimeError("study-object manifest SHA不一致")

    controls: list[pd.DataFrame] = []
    target_rows: list[dict[str, object]] = []
    input_sha: dict[str, str] = {
        "matching_v2_protocol": file_sha256(PROTOCOL_PATH),
        "study_manifest": file_sha256(STUDY_ROOT / "rpc2_study_objects.csv"),
        "study_config": file_sha256(STUDY_ROOT / "config.json"),
        "v1_audit_summary": file_sha256(AUDIT_ROOT / "summary.json"),
        "v1_audit_geometry": file_sha256(AUDIT_ROOT / "matching_geometry.csv"),
        "preliminary_v2_manifest_status": "invalidated_before_intervention_due_float_tie_break",
    }
    for seed in SEEDS:
        cache = CACHE_ROOT / f"seed{seed}"
        eligible_path = cache / "train_eligible_ids.npy"
        eligible_ids = np.load(eligible_path).astype(np.int64)
        raw_columns = []
        for name, filename in (
            ("active_frequency", "train_active_frequency.npy"),
            ("activation_mass", "train_mass.npy"),
            ("representation_energy", "train_energy.npy"),
        ):
            path = cache / filename
            raw_columns.append(patient_equal_covariate(np.load(path, mmap_mode="r"), eligible_ids))
            input_sha[f"seed{seed}_{name}"] = file_sha256(path)
        input_sha[f"seed{seed}_eligible_ids"] = file_sha256(eligible_path)
        raw = np.column_stack(raw_columns)
        percentiles = np.column_stack([
            empirical_midrank(raw[:, column]) for column in range(raw.shape[1])
        ])
        lookup = {int(feature): index for index, feature in enumerate(eligible_ids)}
        excluded = set(study[f"feature_{seed}"].astype(int))
        for item in study.itertuples(index=False):
            feature_id = int(getattr(item, f"feature_{seed}"))
            position = lookup[feature_id]
            selected, status = select_nearest_controls(
                feature_id, eligible_ids, percentiles, excluded,
            )
            n_control = int(len(selected))
            target_rows.append({
                "anchor_id": str(item.anchor_id),
                "primary_candidate": bool(item.primary_candidate),
                "low_effect_control": bool(item.low_effect_control),
                "post_rpc1_secondary_exploratory": bool(item.post_rpc1_secondary_exploratory),
                "sae_seed": seed,
                "target_feature_id": feature_id,
                "matching_support_status": status,
                "n_control": n_control,
                "percentile_resolution": (1.0 / n_control if n_control else None),
                "conditional_p_resolution": (1.0 / (n_control + 1) if n_control else None),
                "target_active_frequency_percentile": float(percentiles[position, 0]),
                "target_activation_mass_percentile": float(percentiles[position, 1]),
                "target_representation_energy_percentile": float(percentiles[position, 2]),
                "max_selected_distance": (
                    float(selected.linf_distance.max()) if n_control else None
                ),
            })
            if status != "matched":
                continue
            selected.insert(0, "target_feature_id", feature_id)
            selected.insert(0, "sae_seed", seed)
            selected.insert(0, "post_rpc1_secondary_exploratory", bool(item.post_rpc1_secondary_exploratory))
            selected.insert(0, "low_effect_control", bool(item.low_effect_control))
            selected.insert(0, "primary_candidate", bool(item.primary_candidate))
            selected.insert(0, "anchor_id", str(item.anchor_id))
            selected["target_active_frequency_percentile"] = float(percentiles[position, 0])
            selected["target_activation_mass_percentile"] = float(percentiles[position, 1])
            selected["target_representation_energy_percentile"] = float(percentiles[position, 2])
            selected["n_control_for_target"] = n_control
            selected["matching_support_status"] = status
            controls.append(selected)

    target_summary = pd.DataFrame(target_rows).sort_values(["sae_seed", "anchor_id"])
    control_manifest = pd.concat(controls, ignore_index=True).sort_values(
        ["sae_seed", "anchor_id", "control_rank"], kind="stable",
    )
    if len(target_summary) != 447 or target_summary.matching_support_status.ne("matched").any():
        raise RuntimeError("matching v2存在未匹配target-seed")
    expected_counts = {21: 1, 27: 1, 32: 1, 66: 1, 78: 1, 100: 442}
    actual_counts = target_summary.n_control.value_counts().sort_index().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"matching v2 control数量与冻结audit不一致: {actual_counts}")
    for seed in SEEDS:
        seed_study = set(study[f"feature_{seed}"].astype(int))
        if control_manifest.loc[
            control_manifest.sae_seed.eq(seed), "control_feature_id"
        ].isin(seed_study).any():
            raise RuntimeError(f"seed{seed} control manifest含study-object Feature")

    OUTPUT_ROOT.mkdir(parents=True)
    manifest_path = OUTPUT_ROOT / "rpc2_matched_control_manifest_v2.csv"
    target_path = OUTPUT_ROOT / "rpc2_target_matching_summary_v2.csv"
    control_manifest.to_csv(manifest_path, index=False)
    target_summary.to_csv(target_path, index=False)
    payload = {
        "status": "matching_v2_frozen_before_rpc2_intervention",
        "unique_study_objects": 149,
        "target_seed_rows": 447,
        "matched_control_rows": int(len(control_manifest)),
        "n_control_distribution": {str(key): int(value) for key, value in actual_counts.items()},
        "matching_support_insufficient_count": 0,
        "input_sha256": input_sha,
        "output_sha256": {
            "matched_control_manifest": file_sha256(manifest_path),
            "target_matching_summary": file_sha256(target_path),
        },
        "intervention_or_model_forward_executed": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
