#!/usr/bin/env python3
"""复算label×source sharedness敏感性，不改写RP-B v1分类。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_clong_rpb_anchor_master import CACHE_ROOT, MATCHING_ROOT, OUTPUT_ROOT, PROTOCOL_PATH, load_seed
from clong_rpb_core import (
    SEEDS,
    diagnose_sharedness,
    file_sha256,
    label_source_bootstrap_contrasts,
    load_protocol,
)


def main() -> None:
    """生成原v1原因码和六层bootstrap敏感性转移表。"""
    protocol = load_protocol(PROTOCOL_PATH)
    target = OUTPUT_ROOT / "sharedness_label_source_sensitivity_v2"
    if target.exists():
        raise FileExistsError(f"敏感性诊断已存在: {target}")

    anchors = pd.read_csv(MATCHING_ROOT / "development_anchors.csv", dtype={"anchor_id": str})
    original_master = pd.read_csv(OUTPUT_ROOT / "anchor_master/anchor_master.csv", dtype={"anchor_id": str})
    original_stats = pd.read_csv(OUTPUT_ROOT / "anchor_master/anchor_seed_statistics.csv", dtype={"anchor_id": str})
    feature_ids = {seed: anchors[f"feature_{seed}"].to_numpy(int) for seed in SEEDS}
    seeds = {seed: load_seed(seed, feature_ids[seed]) for seed in SEEDS}
    patients = seeds[42]["patients"]
    for seed in (43, 44):
        if not patients.equals(seeds[seed]["patients"]):
            raise RuntimeError("三seed患者表不逐位一致")

    labels = patients.label.to_numpy(np.int8)
    sources = patients.source.to_numpy(str)
    rng = np.random.Generator(np.random.PCG64(20260825))
    sensitivity_rows = []
    intervals_by_seed = {}
    for seed in SEEDS:
        data = seeds[seed]
        intervals_by_seed[seed] = label_source_bootstrap_contrasts(
            data["presence"], data["mass"], labels, sources,
            int(protocol["sharedness"]["bootstrap_replicates"]), rng,
        )

    original_lookup = original_master.set_index("anchor_id").sharedness_class.to_dict()
    for column, anchor_id in enumerate(anchors.anchor_id):
        original_seed_rows = original_stats[original_stats.anchor_id == anchor_id].sort_values("seed").to_dict("records")
        original_class, original_reason, original_states = diagnose_sharedness(original_seed_rows, protocol)
        if original_class != original_lookup[anchor_id]:
            raise RuntimeError(f"{anchor_id}原v1分类无法复现")
        sensitivity_seed_rows = []
        for row in original_seed_rows:
            intervals = intervals_by_seed[int(row["seed"])]
            updated = dict(row)
            for prefix in ("coverage", "mass"):
                updated[f"{prefix}_ci_low"] = float(intervals[f"{prefix}_low"][column])
                updated[f"{prefix}_ci_high"] = float(intervals[f"{prefix}_high"][column])
            sensitivity_seed_rows.append(updated)
        sensitivity_class, sensitivity_reason, sensitivity_states = diagnose_sharedness(
            sensitivity_seed_rows, protocol,
        )
        sensitivity_rows.append({
            "anchor_id": anchor_id,
            "v1_label_only_class": original_class,
            "v1_reason_code": original_reason,
            "v1_seed_states": original_states,
            "label_source_sensitivity_class": sensitivity_class,
            "label_source_reason_code": sensitivity_reason,
            "label_source_seed_states": sensitivity_states,
            "class_changed": sensitivity_class != original_class,
        })

    frame = pd.DataFrame(sensitivity_rows)
    target.mkdir(parents=True)
    frame.to_csv(target / "anchor_sharedness_sensitivity.csv", index=False)
    transitions = (
        frame.groupby(["v1_label_only_class", "label_source_sensitivity_class"], observed=True)
        .size().rename("count").reset_index().to_dict("records")
    )
    summary = {
        "status": "completed_diagnostic_only",
        "role": "label_x_source_bootstrap_sensitivity_does_not_replace_frozen_v1_classes",
        "anchor_count": int(len(frame)),
        "class_changed_count": int(frame.class_changed.sum()),
        "v1_counts": frame.v1_label_only_class.value_counts().sort_index().to_dict(),
        "label_source_sensitivity_counts": frame.label_source_sensitivity_class.value_counts().sort_index().to_dict(),
        "transitions": transitions,
        "v1_reason_counts": frame.v1_reason_code.value_counts().sort_index().to_dict(),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "input_sha256": {
            "anchor_master": file_sha256(OUTPUT_ROOT / "anchor_master/anchor_master.csv"),
            "anchor_seed_statistics": file_sha256(OUTPUT_ROOT / "anchor_master/anchor_seed_statistics.csv"),
        },
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
