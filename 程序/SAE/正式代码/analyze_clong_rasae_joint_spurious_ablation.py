#!/usr/bin/env python3
"""在完整验证集上联合删除四个医生提示的疑似干扰方向组。

联合条件一次性删除G0080、G0203、G2313和G0107的全部7项Feature，
随后重新计算冻结注意力和分类输出。四个单删条件只读取既有逐图CSV；
非加和量先在同一图片、同一原始参照下按带符号变化计算，再聚合到患者。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from analyze_clong_rasae_medical_feedback import BASE, CACHE, K, load_models, parse_members
from clong_rpc_core import remove_feature_group
from clong_s2b_core import attention_from_features, pooled_from_features
from clong_sae_discovery import FROZEN_PATIENT_THRESHOLD
from run_clong_rasae_pilot import read_subset


GROUPS = ("G0080", "G0203", "G2313", "G0107")
EXPECTED_MEMBERS = {
    "G0080": (407, 2003),
    "G0203": (1689, 1910),
    "G2313": (2555,),
    "G0107": (558, 2441),
}
JOINT_NAME = "四组联合删除"
SINGLE_RESULT = BASE / "medical_feedback_followup_20260909_v2/四组干预_逐图结果.csv"


def patient_effects(image_effects: pd.DataFrame) -> pd.DataFrame:
    """按冻结的患者图像等权规则聚合一个或多个删除条件。"""
    work = image_effects.copy()
    work["abs_image_delta_margin"] = work.delta_margin.abs()
    work["abs_image_delta_probability"] = work.delta_probability.abs()
    grouped = work.groupby(["condition", "patient_id"], sort=True)
    patients = grouped.agg(
        source=("source", "first"), label=("label", "first"),
        image_count=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        ablated_probability=("ablated_probability", "mean"),
        signed_mean_delta_margin=("delta_margin", "mean"),
        mean_image_abs_delta_margin=("abs_image_delta_margin", "mean"),
        signed_mean_delta_probability=("delta_probability", "mean"),
        mean_image_abs_delta_probability=("abs_image_delta_probability", "mean"),
        mean_attention_l1=("attention_l1", "mean"),
        mean_attention_cosine=("attention_cosine", "mean"),
        peak_change_fraction=("peak_changed", "mean"),
        active=("active", "max"),
    ).reset_index()
    source_counts = work.groupby(["condition", "patient_id"]).source.nunique()
    if source_counts.max() != 1:
        raise RuntimeError("同一条件内存在跨来源患者")
    return patients


def summarize_conditions(images: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """汇总全队列、类别、来源及来源×类别的单删和联合效应。"""
    scopes: list[tuple[str, str, int | None]] = [("all", "all", None)]
    scopes += [("label", "cancer", 1), ("label", "noncancer", 0)]
    for source in sorted(patients.source.unique()):
        scopes.append(("source", str(source), None))
        scopes.append(("source_label", f"{source}|cancer", 1))
        scopes.append(("source_label", f"{source}|noncancer", 0))
    rows = []
    for condition in (*GROUPS, JOINT_NAME):
        condition_patients = patients[patients.condition.eq(condition)]
        condition_images = images[images.condition.eq(condition)]
        for scope_type, scope_value, label in scopes:
            selected = condition_patients
            selected_images = condition_images
            if scope_type.startswith("source"):
                source = scope_value.split("|", 1)[0]
                selected = selected[selected.source.eq(source)]
                selected_images = selected_images[selected_images.source.eq(source)]
            if label is not None:
                selected = selected[selected.label.eq(label)]
                selected_images = selected_images[selected_images.label.eq(label)]
            if selected.empty:
                continue
            original_prediction = selected.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
            changed_prediction = selected.ablated_probability.ge(FROZEN_PATIENT_THRESHOLD)
            original_correct = original_prediction.eq(selected.label.astype(bool))
            changed_correct = changed_prediction.eq(selected.label.astype(bool))
            margin_abs = selected.mean_image_abs_delta_margin.to_numpy()
            probability_abs = selected.mean_image_abs_delta_probability.to_numpy()
            has_both_labels = selected.label.nunique() == 2
            rows.append({
                "condition": condition, "scope_type": scope_type,
                "scope_value": scope_value, "patients": len(selected),
                "images": len(selected_images), "active_patients": int(selected.active.sum()),
                "mean_patient_signed_delta_margin": float(selected.signed_mean_delta_margin.mean()),
                "mean_patient_image_abs_delta_margin": float(margin_abs.mean()),
                "median_patient_image_abs_delta_margin": float(np.median(margin_abs)),
                "p95_patient_image_abs_delta_margin": float(np.quantile(margin_abs, 0.95)),
                "max_patient_image_abs_delta_margin": float(margin_abs.max()),
                "mean_patient_signed_delta_probability": float(selected.signed_mean_delta_probability.mean()),
                "mean_patient_image_abs_delta_probability": float(probability_abs.mean()),
                "median_patient_image_abs_delta_probability": float(np.median(probability_abs)),
                "p95_patient_image_abs_delta_probability": float(np.quantile(probability_abs, 0.95)),
                "max_patient_image_abs_delta_probability": float(probability_abs.max()),
                "patient_flip_count": int((original_prediction != changed_prediction).sum()),
                "prediction_cancer_to_noncancer_count": int((original_prediction & ~changed_prediction).sum()),
                "prediction_noncancer_to_cancer_count": int((~original_prediction & changed_prediction).sum()),
                "incorrect_to_correct_count": int((~original_correct & changed_correct).sum()),
                "correct_to_incorrect_count": int((original_correct & ~changed_correct).sum()),
                "original_patient_auc": (
                    float(roc_auc_score(selected.label, selected.original_probability))
                    if has_both_labels else np.nan
                ),
                "ablated_patient_auc": (
                    float(roc_auc_score(selected.label, selected.ablated_probability))
                    if has_both_labels else np.nan
                ),
                "patient_threshold": FROZEN_PATIENT_THRESHOLD,
            })
    result = pd.DataFrame(rows)
    result["auc_change"] = result.ablated_patient_auc - result.original_patient_auc
    return result


def build_nonadditivity(joint: pd.DataFrame, singles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先逐图计算联合带符号变化减四个单删带符号变化之和，再聚合患者。"""
    margin = singles.pivot(index="source_row", columns="condition", values="delta_margin")
    probability = singles.pivot(index="source_row", columns="condition", values="delta_probability")
    if list(margin.columns) != sorted(GROUPS) or list(probability.columns) != sorted(GROUPS):
        raise RuntimeError("既有单删CSV未包含恰好四个目标组")
    result = joint.copy()
    result["single_delta_margin_sum"] = margin.loc[result.source_row, list(GROUPS)].sum(axis=1).to_numpy()
    result["single_delta_probability_sum"] = probability.loc[result.source_row, list(GROUPS)].sum(axis=1).to_numpy()
    result["nonadditive_delta_margin"] = result.delta_margin - result.single_delta_margin_sum
    result["nonadditive_delta_probability"] = result.delta_probability - result.single_delta_probability_sum
    for group in GROUPS:
        result[f"{group}_delta_margin"] = margin.loc[result.source_row, group].to_numpy()
        result[f"{group}_delta_probability"] = probability.loc[result.source_row, group].to_numpy()
    grouped = result.groupby("patient_id", sort=True)
    patients = grouped.agg(
        source=("source", "first"), label=("label", "first"), image_count=("source_row", "size"),
        joint_signed_mean_delta_margin=("delta_margin", "mean"),
        single_sum_signed_mean_delta_margin=("single_delta_margin_sum", "mean"),
        nonadditive_signed_mean_delta_margin=("nonadditive_delta_margin", "mean"),
        joint_signed_mean_delta_probability=("delta_probability", "mean"),
        single_sum_signed_mean_delta_probability=("single_delta_probability_sum", "mean"),
        nonadditive_signed_mean_delta_probability=("nonadditive_delta_probability", "mean"),
    ).reset_index()
    return result, patients


def summarize_nonadditivity(patients: pd.DataFrame) -> pd.DataFrame:
    """按全队列、类别、来源和来源×类别描述逐图先算得到的非加和量。"""
    selectors = [("all", "all", patients)]
    selectors += [("label", "cancer", patients[patients.label.eq(1)]),
                  ("label", "noncancer", patients[patients.label.eq(0)])]
    for source in sorted(patients.source.unique()):
        source_rows = patients[patients.source.eq(source)]
        selectors.append(("source", str(source), source_rows))
        selectors.append(("source_label", f"{source}|cancer", source_rows[source_rows.label.eq(1)]))
        selectors.append(("source_label", f"{source}|noncancer", source_rows[source_rows.label.eq(0)]))
    rows = []
    for scope_type, scope_value, selected in selectors:
        if selected.empty:
            continue
        margin = selected.nonadditive_signed_mean_delta_margin.to_numpy()
        probability = selected.nonadditive_signed_mean_delta_probability.to_numpy()
        rows.append({
            "scope_type": scope_type, "scope_value": scope_value,
            "patients": len(selected), "images": int(selected.image_count.sum()),
            "mean_joint_signed_delta_margin": float(selected.joint_signed_mean_delta_margin.mean()),
            "mean_single_sum_signed_delta_margin": float(selected.single_sum_signed_mean_delta_margin.mean()),
            "mean_nonadditive_delta_margin": float(margin.mean()),
            "median_nonadditive_delta_margin": float(np.median(margin)),
            "p95_abs_nonadditive_delta_margin": float(np.quantile(np.abs(margin), 0.95)),
            "max_abs_nonadditive_delta_margin": float(np.abs(margin).max()),
            "mean_joint_signed_delta_probability": float(selected.joint_signed_mean_delta_probability.mean()),
            "mean_single_sum_signed_delta_probability": float(selected.single_sum_signed_mean_delta_probability.mean()),
            "mean_nonadditive_delta_probability": float(probability.mean()),
            "median_nonadditive_delta_probability": float(np.median(probability)),
            "p95_abs_nonadditive_delta_probability": float(np.quantile(np.abs(probability), 0.95)),
            "max_abs_nonadditive_delta_probability": float(np.abs(probability).max()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """运行联合删除并写入稳定结果目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    group_table = pd.read_csv(BASE / "full_dictionary_groups_20260909/groups.csv").set_index("group")
    actual_members = {group: tuple(parse_members(group_table.loc[group, "members"])) for group in GROUPS}
    if actual_members != EXPECTED_MEMBERS:
        raise RuntimeError(f"四组成员与审核映射不一致: {actual_members}")
    joint_members = tuple(member for group in GROUPS for member in actual_members[group])
    if len(joint_members) != len(set(joint_members)):
        raise RuntimeError("四组之间存在重复Feature")

    frame = pd.read_csv(CACHE / "val_metadata.csv").reset_index().rename(columns={"index": "source_row"})
    if len(frame) != 497 or frame.patient_id.nunique() != 260:
        raise RuntimeError("完整验证集不是预期的497图/260人")
    cross_source = frame.groupby("patient_id").source.nunique()
    if cross_source.max() != 1:
        raise RuntimeError("验证集中存在跨来源患者，需先约定聚合方式")

    singles = pd.read_csv(SINGLE_RESULT)
    singles = singles[singles.split.eq("val") & singles.group.isin(GROUPS)].copy()
    singles = singles.rename(columns={"group": "condition"})
    if len(singles) != len(frame) * len(GROUPS):
        raise RuntimeError("既有验证集单删逐图记录不完整")
    metadata = frame[["source_row", "patient_id", "label", "source"]]
    singles = singles.drop(columns=["patient_id", "label"]).merge(
        metadata, on="source_row", how="left", validate="many_to_one"
    )

    device = torch.device(args.device)
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight
    rows = []
    with torch.no_grad():
        for start in range(0, len(frame), args.batch_size):
            stop = min(start + args.batch_size, len(frame))
            batch_frame = frame.iloc[start:stop]
            spatial = read_subset(batch_frame, "val", device)["spatial"]
            hidden = sae.encode(spatial, K)
            original_attention = attention_from_features(spatial, head)
            original_pool = pooled_from_features(spatial, original_attention)
            original_logits = original_pool @ classifier_weight.T + classifier_bias
            original_margin = original_logits[:, 1] - original_logits[:, 0]
            original_probability = original_logits.softmax(1)[:, 1]
            expected = torch.as_tensor(
                batch_frame.cancer_probability.to_numpy(dtype=np.float32, copy=True), device=device
            )
            torch.testing.assert_close(original_probability, expected, atol=1e-4, rtol=1e-4)

            changed = remove_feature_group(spatial, hidden, decoder, joint_members)
            changed_attention = attention_from_features(changed, head)
            changed_pool = pooled_from_features(changed, changed_attention)
            changed_logits = changed_pool @ classifier_weight.T + classifier_bias
            changed_margin = changed_logits[:, 1] - changed_logits[:, 0]
            changed_probability = changed_logits.softmax(1)[:, 1]
            attention_cosine = torch.nn.functional.cosine_similarity(
                original_attention, changed_attention, dim=1
            )
            active = hidden[:, :, joint_members].gt(1e-8).any((1, 2))
            for local, record in enumerate(batch_frame.itertuples()):
                rows.append({
                    "condition": JOINT_NAME, "source_row": int(record.source_row),
                    "patient_id": str(record.patient_id), "label": int(record.label),
                    "source": str(record.source), "members": ",".join(map(str, joint_members)),
                    "active": bool(active[local]), "original_margin": float(original_margin[local]),
                    "ablated_margin": float(changed_margin[local]),
                    "delta_margin": float(changed_margin[local] - original_margin[local]),
                    "original_probability": float(original_probability[local]),
                    "ablated_probability": float(changed_probability[local]),
                    "delta_probability": float(changed_probability[local] - original_probability[local]),
                    "attention_l1": float((changed_attention[local] - original_attention[local]).abs().sum()),
                    "attention_cosine": float(attention_cosine[local]),
                    "peak_changed": bool(changed_attention[local].argmax() != original_attention[local].argmax()),
                })
            print(f"val: {stop}/{len(frame)}", flush=True)

    joint = pd.DataFrame(rows)
    reference = singles[singles.condition.eq(GROUPS[0])].sort_values("source_row")
    current = joint.sort_values("source_row")
    original_probability_error = float(np.max(np.abs(
        reference.original_probability.to_numpy() - current.original_probability.to_numpy()
    )))
    if original_probability_error > 2e-6:
        raise RuntimeError("联合运行与既有单删的原始概率参照不一致")

    all_images = pd.concat([singles, joint], ignore_index=True, sort=False)
    all_patients = patient_effects(all_images)
    comparison = summarize_conditions(all_images, all_patients)
    paired_images, paired_patients = build_nonadditivity(joint, singles)
    nonadditive_summary = summarize_nonadditivity(paired_patients)

    joint.to_csv(args.output / "联合删除_逐图结果.csv", index=False)
    all_patients.to_csv(args.output / "联合与单删_逐患者结果.csv", index=False)
    comparison.to_csv(args.output / "联合与单删_分层比较.csv", index=False)
    paired_images.to_csv(args.output / "非加和_逐图配对结果.csv", index=False)
    paired_patients.to_csv(args.output / "非加和_逐患者配对结果.csv", index=False)
    nonadditive_summary.to_csv(args.output / "非加和_分层汇总.csv", index=False)
    definition = {
        "model": "fixed_RA_duration100_K256_width2560",
        "cohort": "complete validation, 497 images/260 patients",
        "groups": {group: list(actual_members[group]) for group in GROUPS},
        "joint_members": list(joint_members),
        "joint_removal": "one simultaneous removal from original spatial representation, then recompute attention/pooling/classifier",
        "single_removal_source": str(SINGLE_RESULT),
        "nonadditivity": "per-image signed joint delta minus sum of four signed single deltas under the same original reference; then patient mean",
        "nonadditivity_limit": "descriptive only; does not by itself identify compensation, redundancy, or mechanism",
        "patient_aggregation": "mean image cancer probability per patient",
        "source_field": "val_metadata.source; every patient belongs to exactly one source",
        "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "joint_members_unique": len(joint_members) == len(set(joint_members)),
        "joint_member_count": len(joint_members),
        "simultaneous_joint_removal": True,
        "existing_single_rows_reused": len(singles),
        "joint_rows": len(joint), "patients": frame.patient_id.nunique(),
        "cross_source_patient_count": int((cross_source > 1).sum()),
        "max_original_probability_difference_vs_existing_single": original_probability_error,
        "test_read": False, "external_read": False,
    }
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
