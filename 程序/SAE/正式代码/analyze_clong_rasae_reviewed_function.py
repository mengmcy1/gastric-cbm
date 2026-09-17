#!/usr/bin/env python3
"""核对已审核64个RA-SAE Feature–图片对的单Feature删除功能影响。

优先复用已有val98精确残差保留删除矩阵；对未覆盖的已审核配对，只补算
``F' = F - h_j d_j``并重新计算冻结注意力和分类头。不新增图片或医学标注。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rpc_core import remove_feature_group  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
BATCH_ROOT = BASE / "functional_first_medical_batch_20260915"
FEEDBACK_ROOT = BASE / "medical_feedback_strength_check_20260915"
DECISION_ROOT = BASE / "decision_alignment"
DURATION_ROOT = BASE / "duration100"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
K = 256


def reviewed_pairs() -> pd.DataFrame:
    """将首批选例与医学主要模式/疑似反例/不确定关系合并。

    Args:
        None.

    Returns:
        pd.DataFrame: 64行、每个Feature–图片对一行的既审核资料。
    """
    selected = pd.read_csv(BATCH_ROOT / "selected_images_internal.csv")
    feedback = pd.read_csv(FEEDBACK_ROOT / "medical_feedback_image_mapping.csv")
    keep = [
        "feature_id", "image_code", "relation_to_current_candidate_description",
        "medical_description", "interpretation_boundary",
    ]
    pairs = selected.merge(feedback[keep], on=["feature_id", "image_code"], validate="one_to_one")
    if len(pairs) != 64 or pairs.groupby("feature_id").size().ne(8).any():
        raise RuntimeError("已审核Feature–图片对不是8×8")
    return pairs


def attach_reused_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    """从现有val98单Feature删除矩阵复用可直接匹配的效应。

    Args:
        pairs (pd.DataFrame): 64个已审核配对。

    Returns:
        pd.DataFrame: 增加delta和provenance列；未匹配行保留NaN。
    """
    output = pairs.copy()
    output["delta_margin"] = np.nan
    output["delta_probability"] = np.nan
    output["effect_provenance"] = "missing_to_compute"
    subset = pd.read_csv(DURATION_ROOT / "val_subset.csv", usecols=["source_row"]).reset_index(
        names="existing_effect_row"
    )
    lookup = dict(zip(subset.source_row.astype(int), subset.existing_effect_row.astype(int)))
    delta_margin = np.load(DECISION_ROOT / "ra_val_delta_margin.npy", mmap_mode="r")
    delta_probability = np.load(DECISION_ROOT / "ra_val_delta_probability.npy", mmap_mode="r")
    for index, record in output.iterrows():
        if record.split != "val" or int(record.source_row) not in lookup:
            continue
        row = lookup[int(record.source_row)]
        feature = int(record.feature_id)
        output.loc[index, "delta_margin"] = float(delta_margin[row, feature])
        output.loc[index, "delta_probability"] = float(delta_probability[row, feature])
        output.loc[index, "effect_provenance"] = "reused_existing_val98_exact_deletion"
    return output


def compute_missing_effects(
    pairs: pd.DataFrame, device: torch.device, batch_size: int,
) -> pd.DataFrame:
    """仅为现有矩阵未覆盖的已审核配对补算精确单Feature删除。

    Args:
        pairs (pd.DataFrame): 已填入可复用效应的64行表。
        device (torch.device): 已核对的CUDA设备或CPU。
        batch_size (int): 唯一图片的投影批量。

    Returns:
        pd.DataFrame: 64行delta均已填入，补算行标记新精确foreward。
    """
    output = pairs.copy()
    missing = output[output.effect_provenance.eq("missing_to_compute")]
    frames = {
        split: pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(columns={"index": "source_row"})
        for split in ("train", "val")
    }
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight
    with torch.no_grad():
        for split in ("train", "val"):
            split_missing = missing[missing.split.eq(split)]
            indices = sorted(split_missing.source_row.astype(int).unique())
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                batch_frame = frames[split].iloc[current]
                data = read_subset(batch_frame, split, device)
                spatial = data["spatial"]
                hidden = sae.encode(spatial, K)
                original_attention = attention_from_features(spatial, head)
                original_pool = pooled_from_features(spatial, original_attention)
                original_logits = original_pool @ classifier_weight.T + classifier_bias
                original_margin = original_logits[:, 1] - original_logits[:, 0]
                original_probability = original_logits.softmax(1)[:, 1]
                local_lookup = {source_row: local for local, source_row in enumerate(current)}
                target_rows = split_missing[split_missing.source_row.isin(current)]
                for index, record in target_rows.iterrows():
                    local = local_lookup[int(record.source_row)]
                    feature = int(record.feature_id)
                    changed = remove_feature_group(
                        spatial[local:local + 1], hidden[local:local + 1], decoder, [feature]
                    )
                    changed_attention = attention_from_features(changed, head)
                    changed_pool = pooled_from_features(changed, changed_attention)
                    changed_logits = changed_pool @ classifier_weight.T + classifier_bias
                    changed_margin = changed_logits[:, 1] - changed_logits[:, 0]
                    changed_probability = changed_logits.softmax(1)[:, 1]
                    output.loc[index, "delta_margin"] = float(changed_margin[0] - original_margin[local])
                    output.loc[index, "delta_probability"] = float(
                        changed_probability[0] - original_probability[local]
                    )
                    output.loc[index, "effect_provenance"] = "new_exact_deletion_reviewed_pair_only"
    if output[["delta_margin", "delta_probability"]].isna().any().any():
        raise RuntimeError("存在未补齐的已审核配对")
    return output


def summarize_relations(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按Feature和医学反馈关系汇总激活与实际删除影响。

    Args:
        pairs (pd.DataFrame): 64行已审核配对及精确delta。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 分类汇总和逐Feature主要/反例对照。
    """
    frame = pairs.copy()
    frame["abs_delta_margin"] = frame.delta_margin.abs()
    frame["abs_delta_probability"] = frame.delta_probability.abs()
    rows = []
    for (feature, relation), group in frame.groupby(
        ["feature_id", "relation_to_current_candidate_description"], sort=False
    ):
        rows.append({
            "feature_id": int(feature), "feature": f"RA-F{int(feature):04d}",
            "relation": relation, "images": len(group),
            "mean_peak_q99": float(group.peak_q99.mean()),
            "median_peak_q99": float(group.peak_q99.median()),
            "mean_signed_delta_margin": float(group.delta_margin.mean()),
            "mean_abs_delta_margin": float(group.abs_delta_margin.mean()),
            "median_abs_delta_margin": float(group.abs_delta_margin.median()),
            "max_abs_delta_margin": float(group.abs_delta_margin.max()),
            "mean_abs_delta_probability": float(group.abs_delta_probability.mean()),
            "max_abs_delta_probability": float(group.abs_delta_probability.max()),
        })
    summary = pd.DataFrame(rows)
    comparisons = []
    for feature in sorted(frame.feature_id.unique()):
        group = frame[frame.feature_id.eq(feature)]
        main = group[group.relation_to_current_candidate_description.eq("main_candidate_description")]
        counter = group[group.relation_to_current_candidate_description.eq("possible_counterexample")]
        uncertain = group[group.relation_to_current_candidate_description.eq("other_or_uncertain")]
        record = {
            "feature_id": int(feature), "feature": f"RA-F{int(feature):04d}",
            "main_images": len(main), "counter_images": len(counter), "uncertain_images": len(uncertain),
            "main_mean_peak_q99": float(main.peak_q99.mean()) if len(main) else np.nan,
            "counter_mean_peak_q99": float(counter.peak_q99.mean()) if len(counter) else np.nan,
            "uncertain_mean_peak_q99": float(uncertain.peak_q99.mean()) if len(uncertain) else np.nan,
            "main_mean_abs_delta_margin": float(main.delta_margin.abs().mean()) if len(main) else np.nan,
            "counter_mean_abs_delta_margin": float(counter.delta_margin.abs().mean()) if len(counter) else np.nan,
            "uncertain_mean_abs_delta_margin": float(uncertain.delta_margin.abs().mean()) if len(uncertain) else np.nan,
            "main_max_abs_delta_margin": float(main.delta_margin.abs().max()) if len(main) else np.nan,
            "counter_max_abs_delta_margin": float(counter.delta_margin.abs().max()) if len(counter) else np.nan,
            "uncertain_max_abs_delta_margin": float(uncertain.delta_margin.abs().max()) if len(uncertain) else np.nan,
        }
        comparisons.append(record)
    return summary, pd.DataFrame(comparisons)


def main() -> None:
    """复用或补算已审核配对的功能影响，不增加医学材料。

    Args:
        None: 输出、设备和批量由CLI提供。

    Returns:
        None: 写入逐对结果、分类汇总与验证JSON。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    definition = {
        "scope": "64 previously medically reviewed Feature-image pairs only",
        "reuse": "existing val98 exact single-Feature residual-preserving deletion where source_row matches",
        "missing": "compute exact F_prime=F-h_j*d_j and recompute frozen attention/pooling/classifier",
        "relation_categories": [
            "main_candidate_description", "possible_counterexample", "other_or_uncertain"
        ],
        "causal_limit": "internal representation intervention, not medical causality",
        "new_images": False, "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pairs = attach_reused_effects(reviewed_pairs())
    reused = int(pairs.effect_provenance.eq("reused_existing_val98_exact_deletion").sum())
    pairs = compute_missing_effects(pairs, torch.device(args.device), args.batch_size)
    relation, comparison = summarize_relations(pairs)
    pairs.to_csv(args.output / "reviewed_pair_effects.csv", index=False)
    relation.to_csv(args.output / "effect_by_feedback_relation.csv", index=False)
    comparison.to_csv(args.output / "feature_main_counter_uncertain_comparison.csv", index=False)
    verification = {
        "reviewed_pairs": len(pairs), "existing_effects_reused": reused,
        "missing_pairs_newly_computed": int(len(pairs) - reused),
        "all_effects_finite": bool(np.isfinite(pairs[["delta_margin", "delta_probability"]]).all().all()),
        "new_images": False, "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
