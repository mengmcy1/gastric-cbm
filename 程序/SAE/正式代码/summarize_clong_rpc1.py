#!/usr/bin/env python3
"""汇总RP-C1三seed效应并执行结果前冻结的RP-C2分轨。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clong_rpb_core import file_sha256
from clong_rpc_core import assign_rpc2_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "结果/SAE/RP_C1_Effect_Screen_20260825/formal"
RPB_ROOT = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825"
PROTOCOL_PATH = CODE_ROOT / "rpc_effect_screen_protocol_v1.json"
SEEDS = (42, 43, 44)


def direction_summary(values: list[float]) -> str:
    """记录三seed方向和最大同向数，不平均掉单seed。"""
    signs = ["positive" if value > 0 else "negative" if value < 0 else "zero" for value in values]
    same = max(signs.count(sign) for sign in set(signs))
    return f"{same}/3:{';'.join(signs)}"


def main() -> None:
    """构建RP-C1 Master Table、独立轨道和机器可读汇总。"""
    target = ROOT / "summary"
    if target.exists():
        raise FileExistsError(f"RP-C1汇总已存在: {target}")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if bool(config["debug"]) or int(config["anchor_count"]) != 1150:
        raise RuntimeError("RP-C1汇总只接受完整正式screen")
    metrics = pd.read_csv(ROOT / "seed_anchor_metrics.csv", dtype={"anchor_id": str})
    if len(metrics) != 3450 or metrics.groupby("anchor_id").seed.nunique().ne(3).any():
        raise RuntimeError("RP-C1三seed指标不完整")
    rpb = pd.read_csv(RPB_ROOT / "anchor_master/anchor_master.csv", dtype={"anchor_id": str})
    sensitivity = pd.read_csv(
        RPB_ROOT / "sharedness_label_source_sensitivity_v2/anchor_sharedness_sensitivity.csv",
        dtype={"anchor_id": str},
    )
    reason = sensitivity[[
        "anchor_id", "v1_reason_code", "class_changed",
        "label_source_sensitivity_class",
    ]].rename(columns={
        "v1_reason_code": "sharedness_reason",
        "class_changed": "sharedness_source_sensitivity_changed",
    })
    master = rpb.rename(columns={"sharedness_class": "sharedness_v1"}).merge(
        reason, on="anchor_id", how="left", validate="one_to_one",
    )

    summary_rows = []
    wide_metrics = [
        "cancer_mean_delta_margin", "noncancer_mean_delta_margin",
        "cancer_active_mean_delta_margin", "noncancer_active_mean_delta_margin",
        "overall_mean_abs_delta_margin", "class_separation_delta_D",
        "patient_flip_rate", "image_flip_rate",
        "cancer_mean_attention_l1", "noncancer_mean_attention_l1",
        "cancer_mean_attention_cosine", "noncancer_mean_attention_cosine",
    ]
    for anchor_id, group in metrics.groupby("anchor_id", sort=True):
        group = group.sort_values("seed")
        row: dict[str, object] = {"anchor_id": anchor_id}
        for item in group.itertuples(index=False):
            for metric in wide_metrics:
                row[f"seed{int(item.seed)}_{metric}"] = float(getattr(item, metric))
        row["overall_abs_effect"] = float(np.median(group.overall_mean_abs_delta_margin))
        row["class_separation_effect"] = float(np.median(group.class_separation_delta_D))
        row["median_delta_margin_cancer"] = float(np.median(group.cancer_mean_delta_margin))
        row["median_delta_margin_noncancer"] = float(np.median(group.noncancer_mean_delta_margin))
        row["median_active_delta_margin_cancer"] = float(np.median(group.cancer_active_mean_delta_margin))
        row["median_active_delta_margin_noncancer"] = float(np.median(group.noncancer_active_mean_delta_margin))
        row["cancer_seed_consistency"] = direction_summary(group.cancer_mean_delta_margin.tolist())
        row["noncancer_seed_consistency"] = direction_summary(group.noncancer_mean_delta_margin.tolist())
        row["median_attention_l1"] = float(np.median(
            (group.cancer_mean_attention_l1 + group.noncancer_mean_attention_l1) / 2,
        ))
        row["median_attention_cosine"] = float(np.median(
            (group.cancer_mean_attention_cosine + group.noncancer_mean_attention_cosine) / 2,
        ))
        summary_rows.append(row)
    effects = pd.DataFrame(summary_rows)
    master = master.merge(effects, on="anchor_id", how="left", validate="one_to_one")
    master = assign_rpc2_tracks(master)
    master["lightweight_effect_rank"] = master.overall_abs_effect.rank(
        method="first", ascending=False,
    ).astype(int)

    target.mkdir(parents=True)
    master.to_csv(target / "rpc1_anchor_effect_master.csv", index=False)
    track_columns = [column for column in master if column.startswith("rpc2_track_")]
    track_counts = {
        **{column: int(master[column].sum()) for column in track_columns},
        "rpc2_selected_union": int(master.rpc2_selected.sum()),
        "low_effect_controls": int(master.low_effect_control.sum()),
    }
    payload = {
        "status": "rpc1_complete_rpc2_tracks_frozen_and_applied",
        "anchor_count": int(len(master)),
        "seed_anchor_rows": int(len(metrics)),
        "track_counts": track_counts,
        "effect_distribution": {
            "overall_abs_effect_median": float(master.overall_abs_effect.median()),
            "overall_abs_effect_max": float(master.overall_abs_effect.max()),
            "class_separation_effect_median": float(master.class_separation_effect.median()),
            "class_separation_effect_max": float(master.class_separation_effect.max()),
        },
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "input_sha256": {
            "rpc1_config": file_sha256(ROOT / "config.json"),
            "seed_anchor_metrics": file_sha256(ROOT / "seed_anchor_metrics.csv"),
            "rpb_anchor_master": file_sha256(RPB_ROOT / "anchor_master/anchor_master.csv"),
            "rpb_sensitivity": file_sha256(
                RPB_ROOT / "sharedness_label_source_sensitivity_v2/anchor_sharedness_sensitivity.csv"
            ),
        },
        "p_values_or_pass_fail_generated": False,
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
