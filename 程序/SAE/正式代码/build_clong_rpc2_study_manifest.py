#!/usr/bin/env python3
"""从冻结RP-C1结果一次性导出不含效应值的RP-C2 study-object清单。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clong_rpb_core import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
RPC1_MASTER = (
    PROJECT_ROOT
    / "结果/SAE/RP_C1_Effect_Screen_20260825/formal/summary/rpc1_anchor_effect_master.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "结果/SAE/RP_C2_Matching_Feasibility_20260825/study_objects"
PROTOCOL_PATH = CODE_ROOT / "rpc2_matching_audit_protocol_v1.json"


def main() -> None:
    """冻结122主候选、20低效旗标和8个事后探索对象的纯ID血缘。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"RP-C2 study-object目录已存在: {OUTPUT_ROOT}")
    master = pd.read_csv(RPC1_MASTER, dtype={"anchor_id": str})
    bidirectional = (
        master.cancer_seed_consistency.str.startswith("3/3:negative")
        & master.noncancer_seed_consistency.str.startswith("3/3:positive")
    )
    secondary = bidirectional & ~master.rpc2_selected.astype(bool)
    selected = (
        master.rpc2_selected.astype(bool)
        | master.low_effect_control.astype(bool)
        | secondary
    )
    columns = ["anchor_id", "feature_42", "feature_43", "feature_44"]
    output = master.loc[selected, columns].copy()
    output["primary_candidate"] = master.loc[selected, "rpc2_selected"].to_numpy(dtype=bool)
    output["low_effect_control"] = master.loc[selected, "low_effect_control"].to_numpy(dtype=bool)
    output["post_rpc1_secondary_exploratory"] = secondary[selected].to_numpy(dtype=bool)
    output = output.sort_values("anchor_id").reset_index(drop=True)
    counts = {
        "unique_anchor_count": int(len(output)),
        "primary_candidate_flag_count": int(output.primary_candidate.sum()),
        "low_effect_control_flag_count": int(output.low_effect_control.sum()),
        "post_rpc1_secondary_exploratory_flag_count": int(
            output.post_rpc1_secondary_exploratory.sum()
        ),
        "multi_role_anchor_count": int(
            (output[[
                "primary_candidate", "low_effect_control",
                "post_rpc1_secondary_exploratory",
            ]].sum(axis=1) > 1).sum()
        ),
    }
    expected = {
        "unique_anchor_count": 149,
        "primary_candidate_flag_count": 122,
        "low_effect_control_flag_count": 20,
        "post_rpc1_secondary_exploratory_flag_count": 8,
        "multi_role_anchor_count": 1,
    }
    if counts != expected:
        raise RuntimeError(f"RP-C2 study-object角色与冻结结果不一致: {counts}")
    OUTPUT_ROOT.mkdir(parents=True)
    manifest = OUTPUT_ROOT / "rpc2_study_objects.csv"
    output.to_csv(manifest, index=False)
    config = {
        "stage": "rpc2_outcome_blind_study_object_freeze",
        "counts": counts,
        "known_overlap_anchor_ids": output.loc[
            output[["primary_candidate", "low_effect_control"]].all(axis=1), "anchor_id"
        ].tolist(),
        "rpc1_master_sha256": file_sha256(RPC1_MASTER),
        "matching_audit_protocol_sha256": file_sha256(PROTOCOL_PATH),
        "study_object_manifest_sha256": file_sha256(manifest),
        "manifest_contains_intervention_outcomes": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
