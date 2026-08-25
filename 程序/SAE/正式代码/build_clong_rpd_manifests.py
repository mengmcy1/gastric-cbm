#!/usr/bin/env python3
"""构建RP-D v1 Heavy成员和train-only确定性病例清单，不渲染图像。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from clong_rpd_core import build_patient_table, select_activation_bins, select_hard_negative


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = CODE_ROOT / "rpd_atlas_protocol_v1.json"
OUTPUT_ROOT = Path(os.environ.get(
    "RPD_MANIFEST_OUTPUT_ROOT",
    PROJECT_ROOT / "结果/SAE/RP_D_Technical_Atlas_20260825/manifest_dry_run_v1",
))
RPA_CACHE = PROJECT_ROOT / "结果/SAE/RP_A_Development_20260824/analysis_cache/seed42"
SPATIAL_CACHE = PROJECT_ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
RPB_MASTER = PROJECT_ROOT / "结果/SAE/RP_B_Technical_20260825/anchor_master/anchor_master.csv"
RPB_SENSITIVITY = (
    PROJECT_ROOT
    / "结果/SAE/RP_B_Technical_20260825/sharedness_label_source_sensitivity_v2/anchor_sharedness_sensitivity.csv"
)
RPC1_MASTER = (
    PROJECT_ROOT
    / "结果/SAE/RP_C1_Effect_Screen_20260825/formal/summary/rpc1_anchor_effect_master.csv"
)
RPC2_STUDY = (
    PROJECT_ROOT
    / "结果/SAE/RP_C2_Matching_Feasibility_20260825/study_objects/rpc2_study_objects.csv"
)
RPC2_EVIDENCE = (
    PROJECT_ROOT
    / "结果/SAE/RP_C2_Intervention_20260825/formal_retry1/rpc2_anchor_evidence_master.csv"
)


def file_sha256(path: Path) -> str:
    """计算文件SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dry_run_boundary_flags() -> dict[str, bool]:
    """返回RP-D清单dry-run固定的数据与结果边界。"""
    return {
        "images_read": False,
        "assets_rendered": False,
        "scientific_pass_fail_generated": False,
        "train_only": True,
        "val_evaluated": False,
        "internal_test_evaluated": False,
        "external_evaluated": False,
    }


def build_anchor_manifest() -> pd.DataFrame:
    """合并冻结RP-B/RP-C证据并生成Heavy布尔成员。"""
    study = pd.read_csv(RPC2_STUDY, dtype={"anchor_id": str})
    rpb = pd.read_csv(RPB_MASTER, dtype={"anchor_id": str}).rename(
        columns={"sharedness_class": "sharedness_v1"},
    )
    sensitivity = pd.read_csv(RPB_SENSITIVITY, dtype={"anchor_id": str})[[
        "anchor_id", "v1_reason_code", "class_changed",
    ]].rename(columns={
        "v1_reason_code": "sharedness_reason",
        "class_changed": "source_sensitivity_changed",
    })
    rpc1 = pd.read_csv(RPC1_MASTER, dtype={"anchor_id": str})
    rpc2 = pd.read_csv(RPC2_EVIDENCE, dtype={"anchor_id": str})
    anchor = study.merge(
        rpb[["anchor_id", "sharedness_v1", "source_risk"]],
        on="anchor_id", validate="one_to_one",
    ).merge(
        sensitivity, on="anchor_id", validate="one_to_one",
    ).merge(
        rpc1[["anchor_id", "rpc2_track_source"]],
        on="anchor_id", validate="one_to_one",
    ).merge(
        rpc2[[
            "anchor_id", "functional_pattern", "median_matched_midrank_percentile",
            "minimum_matched_midrank_percentile", "median_intermediate_curve_overall_effect",
        ]],
        on="anchor_id", validate="one_to_one",
    )
    anchor["heavy_cross_seed_high"] = (
        anchor.primary_candidate.astype(bool)
        & anchor.minimum_matched_midrank_percentile.ge(0.90)
    )
    anchor["heavy_bidirectional"] = anchor.functional_pattern.eq(
        "bidirectional_label_supporting_3of3",
    )
    anchor["heavy_source_risk"] = anchor.rpc2_track_source.astype(bool)
    anchor["heavy_correlation_dependence_sentinel"] = anchor.anchor_id.eq("a00987")
    anchor["heavy_low_effect_control"] = anchor.low_effect_control.astype(bool)
    heavy_columns = [column for column in anchor if column.startswith("heavy_")]
    anchor["heavy_atlas"] = anchor[heavy_columns].any(axis=1)
    if len(anchor) != 149 or anchor.anchor_id.nunique() != 149:
        raise RuntimeError("RP-D Anchor必须为冻结149个唯一对象")
    return anchor.sort_values("anchor_id").reset_index(drop=True)


def add_case_identity(frame: pd.DataFrame, anchor_id: str) -> pd.DataFrame:
    """为标准病例增加稳定case_id和Atlas层级。"""
    output = frame.copy()
    output["case_id"] = [
        f"{anchor_id}_s42_l{int(row.label)}_{row.case_role}_{int(row.selection_rank):02d}"
        for row in output.itertuples(index=False)
    ]
    return output


def main() -> None:
    """生成Heavy成员、病例清单和shortfall审计，不读取原图。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"RP-D dry-run目录已存在: {OUTPUT_ROOT}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_case_selection_2026-08-25":
        raise RuntimeError("RP-D v1协议未冻结")
    anchor = build_anchor_manifest()
    metadata = pd.read_csv(RPA_CACHE / "train_images.csv", dtype={"patient_id": str})
    spatial_metadata = pd.read_csv(
        SPATIAL_CACHE / "train_metadata.csv", dtype={"patient_id": str}, low_memory=False,
    )
    alignment_columns = ["relative_path", "patient_id", "label", "source"]
    if not metadata[alignment_columns].astype(str).equals(
        spatial_metadata[alignment_columns].astype(str),
    ):
        raise RuntimeError("RP-A activation与C-long pooled cache行顺序不一致")
    activations = np.load(RPA_CACHE / "train_image_activations.npy", mmap_mode="r")
    pooled = np.load(SPATIAL_CACHE / "train_pooled_features.npy", mmap_mode="r")
    feature_ids = anchor.feature_42.to_numpy(dtype=np.int64)
    selected_activations = np.take(activations, feature_ids, axis=2)
    if selected_activations.shape != (2350, 49, 149):
        raise RuntimeError(f"RP-D选择激活shape异常: {selected_activations.shape}")

    case_frames = []
    shortfall_rows = []
    for column, item in enumerate(anchor.itertuples(index=False)):
        image_activation = selected_activations[:, :, column]
        patient_table = build_patient_table(
            metadata,
            image_activation.max(axis=1),
            image_activation.mean(axis=1),
            str(item.anchor_id),
        )
        target_count = 5 if bool(item.heavy_atlas) else 3
        standard_parts = []
        for label in (0, 1):
            selected, shortfalls = select_activation_bins(
                patient_table, str(item.anchor_id), label, target_count,
            )
            standard_parts.append(selected)
            for bin_name, shortfall in shortfalls.items():
                shortfall_rows.append({
                    "anchor_id": item.anchor_id,
                    "label": label,
                    "activation_bin": bin_name,
                    "requested": target_count,
                    "selected": target_count - shortfall,
                    "shortfall": shortfall,
                })
        standard = add_case_identity(pd.concat(standard_parts, ignore_index=True), item.anchor_id)
        standard["feature_id"] = int(item.feature_42)
        standard["in_light_atlas"] = standard.selection_rank.le(3)
        standard["in_heavy_atlas"] = bool(item.heavy_atlas)
        standard["hard_negative_query_case_id"] = ""
        standard["hard_negative_cosine"] = np.nan
        standard["hard_negative_pool"] = ""
        case_frames.append(standard)

        if bool(item.heavy_atlas):
            hard_rows = []
            for _, query in standard[standard.case_role.eq("high")].iterrows():
                selected_low = standard[
                    standard.label.eq(int(query.label)) & standard.case_role.eq("low")
                ]
                negative, similarity, pool_status = select_hard_negative(
                    query, patient_table, selected_low, pooled,
                )
                if negative is None:
                    shortfall_rows.append({
                        "anchor_id": item.anchor_id, "label": int(query.label),
                        "activation_bin": "hard_negative", "requested": 1,
                        "selected": 0, "shortfall": 1,
                    })
                    continue
                record = negative.to_dict()
                record.update({
                    "anchor_id": item.anchor_id,
                    "canonical_seed": 42,
                    "feature_id": int(item.feature_42),
                    "case_role": "hard_negative",
                    "activation_bin": "hard_negative",
                    "selection_rank": int(query.selection_rank),
                    "selection_status": "selected",
                    "case_id": f"{item.anchor_id}_s42_l{int(query.label)}_hard_negative_{int(query.selection_rank):02d}",
                    "in_light_atlas": False,
                    "in_heavy_atlas": True,
                    "hard_negative_query_case_id": query.case_id,
                    "hard_negative_cosine": similarity,
                    "hard_negative_pool": pool_status,
                })
                hard_rows.append(record)
            if hard_rows:
                case_frames.append(pd.DataFrame(hard_rows))

        if bool(item.heavy_source_risk):
            source_rows = []
            for (label, source), group in patient_table.groupby(["label", "source"], sort=True):
                chosen = group.sort_values(
                    ["patient_peak_activation", "selection_hash"],
                    ascending=[False, True], kind="stable",
                ).head(3)
                shortfall_rows.append({
                    "anchor_id": item.anchor_id, "label": int(label),
                    "activation_bin": f"source_panel:{source}", "requested": 3,
                    "selected": int(len(chosen)), "shortfall": int(3 - len(chosen)),
                })
                source_code = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
                for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                    record = row.to_dict()
                    record.update({
                        "anchor_id": item.anchor_id,
                        "canonical_seed": 42,
                        "feature_id": int(item.feature_42),
                        "case_role": "source_panel",
                        "activation_bin": "source_panel",
                        "selection_rank": rank,
                        "selection_status": "selected",
                        "case_id": f"{item.anchor_id}_s42_l{int(label)}_source_{source_code}_{rank:02d}",
                        "in_light_atlas": False,
                        "in_heavy_atlas": True,
                        "hard_negative_query_case_id": "",
                        "hard_negative_cosine": np.nan,
                        "hard_negative_pool": "",
                    })
                    source_rows.append(record)
            if source_rows:
                case_frames.append(pd.DataFrame(source_rows))

    cases = pd.concat(case_frames, ignore_index=True)
    cases["image_relpath"] = spatial_metadata.iloc[
        cases.image_index.to_numpy(dtype=np.int64)
    ].image_relpath.to_numpy()
    cases = cases.sort_values(["anchor_id", "case_role", "label", "selection_rank", "case_id"])
    if cases.case_id.duplicated().any():
        raise RuntimeError("RP-D case_id不唯一")
    shortfalls = pd.DataFrame(shortfall_rows).sort_values(
        ["anchor_id", "label", "activation_bin"], kind="stable",
    )

    OUTPUT_ROOT.mkdir(parents=True)
    heavy_columns = [column for column in anchor if column.startswith("heavy_")]
    heavy = anchor[["anchor_id", *heavy_columns]].copy()
    anchor_path = OUTPUT_ROOT / "rpd_anchor_manifest.csv"
    heavy_path = OUTPUT_ROOT / "rpd_heavy_membership_v1.csv"
    case_path = OUTPUT_ROOT / "rpd_case_manifest.csv"
    shortfall_path = OUTPUT_ROOT / "rpd_case_shortfalls.csv"
    anchor.to_csv(anchor_path, index=False)
    heavy.to_csv(heavy_path, index=False)
    cases.to_csv(case_path, index=False)
    shortfalls.to_csv(shortfall_path, index=False)
    output_sha = {
        "anchor_manifest": file_sha256(anchor_path),
        "heavy_membership": file_sha256(heavy_path),
        "case_manifest": file_sha256(case_path),
        "case_shortfalls": file_sha256(shortfall_path),
    }
    summary = {
        "status": "rpd_v1_manifest_dry_run_complete_no_images_rendered",
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "code_sha256": file_sha256(Path(__file__)),
        "input_sha256": {
            "rpb_master": file_sha256(RPB_MASTER),
            "rpb_sensitivity": file_sha256(RPB_SENSITIVITY),
            "rpc1_master": file_sha256(RPC1_MASTER),
            "rpc2_study": file_sha256(RPC2_STUDY),
            "rpc2_evidence": file_sha256(RPC2_EVIDENCE),
            "rpa_cache_config": file_sha256(RPA_CACHE / "config.json"),
            "spatial_cache_config": file_sha256(SPATIAL_CACHE / "cache_config.json"),
        },
        "counts": {
            "anchor_count": int(len(anchor)),
            "heavy_unique_count": int(anchor.heavy_atlas.sum()),
            **{column: int(anchor[column].sum()) for column in heavy_columns},
            "case_rows": int(len(cases)),
            "light_case_rows": int(cases.in_light_atlas.sum()),
            "heavy_case_rows": int(cases.in_heavy_atlas.sum()),
            "hard_negative_rows": int(cases.case_role.eq("hard_negative").sum()),
            "source_panel_rows": int(cases.case_role.eq("source_panel").sum()),
            "shortfall_slots": int(shortfalls.shortfall.sum()),
            "shortfall_records": int(shortfalls.shortfall.gt(0).sum()),
        },
        "hard_negative_pool_counts": {
            str(key): int(value)
            for key, value in cases.loc[
                cases.case_role.eq("hard_negative"), "hard_negative_pool"
            ].value_counts().sort_index().items()
        },
        "output_sha256": output_sha,
        **dry_run_boundary_flags(),
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
