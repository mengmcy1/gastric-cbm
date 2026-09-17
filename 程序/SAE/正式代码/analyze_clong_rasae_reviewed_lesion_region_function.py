#!/usr/bin/env python3
"""对8个既有医学审核配对做病灶真值框内／框外Feature删除诊断。

本脚本不把病灶框当作医生文字中所有结构的精确区域，也不把7x7网格当作
像素级定位。框内、框外和整图条件都从同一原始表示独立构造并重新forward。
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
from clong_s2b_core import (  # noqa: E402
    attention_from_features, cell_overlap_map, pooled_from_features,
)
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
REVIEWED = BASE / "reviewed_response_function_20260915/reviewed_pair_effects.csv"
TARGET_FEATURES = (868, 2464)
EXPECTED_COUNTS = {868: 4, 2464: 4}
K = 256
LARGE_BOX_THRESHOLD = 0.75
REPRODUCE_TOLERANCE = 1e-5


def load_pairs() -> pd.DataFrame:
    """读取两项Feature的8个既有癌图审核配对及病灶真值框。"""
    reviewed = pd.read_csv(REVIEWED)
    reviewed = reviewed[
        reviewed.feature_id.isin(TARGET_FEATURES) & reviewed.label_internal.eq(1)
    ].copy()
    metadata = []
    for split in ("train", "val"):
        frame = pd.read_csv(CACHE / f"{split}_metadata.csv").reset_index().rename(
            columns={"index": "source_row"}
        )
        frame["split"] = split
        metadata.append(frame)
    columns = [
        "split", "source_row", "label", "patient_id", "cancer_probability",
        "lesion_bbox_available", "localization_supervision", "privileged_roi_source",
        "bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm",
        "bbox_area_fraction",
    ]
    pairs = reviewed.merge(
        pd.concat(metadata, ignore_index=True)[columns],
        on=["split", "source_row"], how="left", validate="many_to_one",
        suffixes=("", "_metadata"),
    ).sort_values(["feature_id", "split", "source_row"]).reset_index(drop=True)
    counts = pairs.groupby("feature_id").size().to_dict()
    if counts != EXPECTED_COUNTS or len(pairs) != 8:
        raise RuntimeError(f"目标配对不是预期的2x4: {counts}")
    if pairs.label.ne(1).any() or pairs.label_internal.ne(1).any():
        raise RuntimeError("目标配对中出现非癌图")
    if not pairs.lesion_bbox_available.all():
        raise RuntimeError("目标配对中存在无病灶框图片")
    if not pairs.localization_supervision.eq(1).all() or not pairs.privileged_roi_source.eq("gt_bbox").all():
        raise RuntimeError("目标区域不是既有病灶真值框")
    if pairs.duplicated(["feature_id", "split", "source_row"]).any():
        raise RuntimeError("存在重复Feature–图片配对")
    coordinates = pairs[
        ["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]
    ].to_numpy(float)
    if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
        raise RuntimeError("病灶框坐标非法")
    if (coordinates[:, 0] >= coordinates[:, 2]).any() or (coordinates[:, 1] >= coordinates[:, 3]).any():
        raise RuntimeError("病灶框宽高非正")
    pairs["region_discrimination_limited"] = pairs.bbox_area_fraction.ge(LARGE_BOX_THRESHOLD)
    if not bool(pairs.loc[pairs.image_code.eq("train_1222"), "region_discrimination_limited"].item()):
        raise RuntimeError("train_1222未按约定标记为区域区分能力受限")
    return pairs


def model_outputs(
    spatial: torch.Tensor, head: torch.nn.Module,
    classifier_weight: torch.Tensor, classifier_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """从给定空间表示重新计算冻结注意力、margin和癌概率。"""
    attention = attention_from_features(spatial, head)
    pooled = pooled_from_features(spatial, attention)
    logits = pooled @ classifier_weight.T + classifier_bias
    return {
        "attention": attention,
        "margin": logits[:, 1] - logits[:, 0],
        "probability": logits.softmax(1)[:, 1],
    }


@torch.no_grad()
def analyze(pairs: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, dict]:
    """分别从原表示构造框内、框外和整图删除并重新forward。"""
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight
    rows = []
    partition_errors = []
    area_errors = []
    original_probability_errors = []
    for split in ("train", "val"):
        frame = pairs[pairs.split.eq(split)].copy().reset_index(drop=True)
        data = read_subset(frame, split, device)
        spatial = data["spatial"]
        hidden = sae.encode(spatial, K)
        original = model_outputs(spatial, head, classifier_weight, classifier_bias)
        expected_probability = torch.as_tensor(
            frame.cancer_probability.to_numpy(dtype=np.float32, copy=True), device=device
        )
        original_probability_errors.append(
            float((original["probability"] - expected_probability).abs().max())
        )
        for local, record in enumerate(frame.itertuples(index=False)):
            box = np.array([
                record.bbox_x1_norm, record.bbox_y1_norm,
                record.bbox_x2_norm, record.bbox_y2_norm,
            ], dtype=np.float64)
            overlap_np = cell_overlap_map(box).reshape(-1)
            if not np.isfinite(overlap_np).all() or (overlap_np < 0).any() or (overlap_np > 1).any():
                raise RuntimeError("7x7病灶框面积权重非法")
            area_errors.append(abs(float(overlap_np.mean()) - float(record.bbox_area_fraction)))
            overlap = torch.as_tensor(overlap_np, device=device, dtype=spatial.dtype)
            feature = int(record.feature_id)
            activation = hidden[local, :, feature]
            if float(activation.sum()) <= 0:
                raise RuntimeError(f"{record.image_code}/RA-F{feature:04d}没有正激活")
            full_component = activation[:, None] * decoder[feature][None, :]
            inside_component = overlap[:, None] * full_component
            outside_component = (1.0 - overlap[:, None]) * full_component
            partition_errors.append(float(
                (inside_component + outside_component - full_component).abs().max()
            ))
            original_spatial = spatial[local:local + 1]
            conditions = {
                "inside": original_spatial - inside_component.unsqueeze(0),
                "outside": original_spatial - outside_component.unsqueeze(0),
                "full": original_spatial - full_component.unsqueeze(0),
            }
            original_margin = original["margin"][local]
            original_probability = original["probability"][local]
            row = {
                "feature_id": feature, "feature": f"RA-F{feature:04d}",
                "image_code": record.image_code, "split": split,
                "source_row": int(record.source_row), "patient_id": record.patient_id,
                "relation_to_current_candidate_description": record.relation_to_current_candidate_description,
                "medical_description": record.medical_description,
                "interpretation_boundary": record.interpretation_boundary,
                "bbox_x1_norm": float(record.bbox_x1_norm),
                "bbox_y1_norm": float(record.bbox_y1_norm),
                "bbox_x2_norm": float(record.bbox_x2_norm),
                "bbox_y2_norm": float(record.bbox_y2_norm),
                "lesion_box_area_fraction": float(record.bbox_area_fraction),
                "feature_activation_sum": float(activation.sum()),
                "feature_inside_activation_fraction": float((activation * overlap).sum() / activation.sum()),
                "region_discrimination_limited": bool(record.region_discrimination_limited),
                "original_margin": float(original_margin),
                "original_probability": float(original_probability),
                "existing_full_delta_margin": float(record.delta_margin),
                "existing_full_delta_probability": float(record.delta_probability),
            }
            for condition, changed_spatial in conditions.items():
                changed = model_outputs(changed_spatial, head, classifier_weight, classifier_bias)
                delta_margin = float(changed["margin"][0] - original_margin)
                delta_probability = float(changed["probability"][0] - original_probability)
                row.update({
                    f"{condition}_delta_margin": delta_margin,
                    f"{condition}_abs_delta_margin": abs(delta_margin),
                    f"{condition}_delta_probability": delta_probability,
                    f"{condition}_abs_delta_probability": abs(delta_probability),
                })
            row["nonadditivity_delta_margin"] = (
                row["full_delta_margin"] - row["inside_delta_margin"] - row["outside_delta_margin"]
            )
            row["nonadditivity_delta_probability"] = (
                row["full_delta_probability"]
                - row["inside_delta_probability"] - row["outside_delta_probability"]
            )
            row["full_margin_reproduction_error"] = (
                row["full_delta_margin"] - row["existing_full_delta_margin"]
            )
            row["full_probability_reproduction_error"] = (
                row["full_delta_probability"] - row["existing_full_delta_probability"]
            )
            rows.append(row)
    output = pd.DataFrame(rows).sort_values(
        ["feature_id", "split", "source_row"]
    ).reset_index(drop=True)
    margin_error = float(output.full_margin_reproduction_error.abs().max())
    probability_error = float(output.full_probability_reproduction_error.abs().max())
    verification = {
        "reviewed_pairs": len(output),
        "counts_by_feature": {
            str(key): int(value) for key, value in output.groupby("feature_id").size().items()
        },
        "all_regions_are_existing_gt_lesion_boxes": True,
        "large_box_threshold": LARGE_BOX_THRESHOLD,
        "limited_region_cases": output.loc[
            output.region_discrimination_limited, "image_code"
        ].tolist(),
        "train_1222_box_area_fraction": float(
            output.loc[output.image_code.eq("train_1222"), "lesion_box_area_fraction"].item()
        ),
        "max_grid_mean_vs_box_area_error": max(area_errors),
        "max_component_partition_abs_error": max(partition_errors),
        "max_original_probability_cache_error": max(original_probability_errors),
        "max_full_margin_reproduction_abs_error": margin_error,
        "max_full_probability_reproduction_abs_error": probability_error,
        "full_effect_reproduction_tolerance": REPRODUCE_TOLERANCE,
        "full_effect_reproduced": bool(
            margin_error <= REPRODUCE_TOLERANCE and probability_error <= REPRODUCE_TOLERANCE
        ),
        "representation_partition_identity_passed": max(partition_errors) <= 2e-6,
        "new_training": False, "new_annotation": False,
        "test_read": False, "external_read": False,
    }
    if not verification["representation_partition_identity_passed"]:
        raise RuntimeError("框内与框外被删除分量之和未复原整图被删除分量")
    if verification["max_grid_mean_vs_box_area_error"] > 2e-6:
        raise RuntimeError("7x7面积权重均值未复现病灶框面积占比")
    if verification["max_original_probability_cache_error"] > 1e-4:
        raise RuntimeError("原始概率未复现冻结缓存")
    if not verification["full_effect_reproduced"]:
        raise RuntimeError("整图删除未复现既有审核配对效应")
    return output, verification


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """按Feature与既有医学关系生成小样本描述性汇总。"""
    metrics = [
        "lesion_box_area_fraction", "feature_inside_activation_fraction",
        "inside_delta_margin", "outside_delta_margin", "full_delta_margin",
        "inside_abs_delta_margin", "outside_abs_delta_margin", "full_abs_delta_margin",
        "inside_delta_probability", "outside_delta_probability", "full_delta_probability",
        "nonadditivity_delta_margin", "nonadditivity_delta_probability",
    ]
    grouped = rows.groupby(
        ["feature", "relation_to_current_candidate_description"], sort=True
    )
    output = grouped.size().rename("images").reset_index()
    for metric in metrics:
        values = grouped[metric].median().rename(f"median_{metric}").reset_index()
        output = output.merge(
            values, on=["feature", "relation_to_current_candidate_description"],
            validate="one_to_one",
        )
    return output


def main() -> None:
    """运行8个固定配对并写出数值、口径与验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    rows, verification = analyze(pairs, torch.device(args.device))
    rows.to_csv(args.output / "逐配对结果.csv", index=False)
    summarize(rows).to_csv(args.output / "关系汇总.csv", index=False)
    definition = {
        "scope": "8 existing reviewed cancer Feature-image pairs: RA-F0868 and RA-F2464, four each",
        "region": "existing GT lesion bbox only; not a pixel mask for every structure named by the medical student",
        "inside": "F - m_p * h_pj * d_j",
        "outside": "F - (1-m_p) * h_pj * d_j",
        "full": "F - h_pj * d_j",
        "m": "fraction of each 7x7 grid cell area overlapped by the normalized GT lesion bbox",
        "construction": "inside/outside/full independently start from original F and original h, then recompute attention/pooling/classifier",
        "activation_fraction": "sum(m_p*h_pj)/sum(h_pj); interpreted alongside lesion bbox area fraction",
        "nonadditivity": "full signed effect - inside signed effect - outside signed effect; descriptive only",
        "key_limit": "7x7 positions have broad receptive fields; inside does not mean pixel content is confined to the bbox",
        "train_1222_limit": "GT lesion bbox covers about 81% of image; cannot distinguish doctor-described normal mucosa from lesion",
        "decision_limit": "spatial differences are clues only; they do not identify visual semantics or prove spatially conditioned extraction will improve mixing",
        "new_training": False, "new_annotation": False,
        "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
