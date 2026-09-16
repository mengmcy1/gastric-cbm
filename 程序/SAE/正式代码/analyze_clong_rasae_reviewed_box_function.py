#!/usr/bin/env python3
"""定位已医学审核RA-SAE Feature的框内与框外功能效应。

仅使用F0868、F1089和F2388中已审核且具有非癌矩形框的12个
Feature–图片配对。对同一原始表示和同一组原始SAE激活，独立构造框内删除、
框外删除和整图删除，然后重算冻结注意力、汇聚和分类头。
这是7x7粗网格的内部表示干预，不是像素级病灶移除或医学因果分解。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_functional


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from analyze_clong_rasae_noncancer_box_alignment import cell_overlap_map  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
REVIEWED = BASE / "reviewed_response_function_20260915/reviewed_pair_effects.csv"
BOX_ROOT = BASE / "noncancer_box_alignment_20260916_v3"
TARGET_FEATURES = (868, 1089, 2388)
EXPECTED_COUNTS = {868: 4, 1089: 4, 2388: 4}
K = 256
REPRODUCE_TOLERANCE = 1e-5


def load_reviewed_box_pairs() -> pd.DataFrame:
    """读取三项Feature中已审核且有非癌框的全部配对。

    Args:
        None.

    Returns:
        pd.DataFrame: 12行配对，包含train/val源行、医学描述和规范化框。
    """
    reviewed = pd.read_csv(REVIEWED)
    reviewed = reviewed[reviewed.feature_id.isin(TARGET_FEATURES)].copy()
    boxes = []
    for split in ("train", "val"):
        frame = pd.read_csv(BOX_ROOT / f"{split}_matched_noncancer_boxes.csv")
        frame["split"] = split
        boxes.append(frame)
    box_columns = [
        "split", "source_row", "box_x1_norm", "box_y1_norm", "box_x2_norm",
        "box_y2_norm", "box_area_fraction", "annotation_width", "annotation_height",
        "width_matches", "height_matches",
    ]
    matched = reviewed.merge(
        pd.concat(boxes, ignore_index=True)[box_columns],
        on=["split", "source_row"], how="inner", validate="many_to_one",
    ).sort_values(["feature_id", "split", "source_row"]).reset_index(drop=True)

    counts = matched.groupby("feature_id").size().to_dict()
    if counts != EXPECTED_COUNTS or len(matched) != 12:
        raise RuntimeError(f"已审核有框配对不是预期的3x4: {counts}")
    if matched.label_internal.ne(0).any():
        raise RuntimeError("非癌框配对中出现癌图")
    if matched.duplicated(["feature_id", "split", "source_row"]).any():
        raise RuntimeError("存在重复Feature–图片配对")
    if matched.patient_id.nunique() != 12:
        raise RuntimeError("预期12个配对对应12位不同患者")
    if not matched[["width_matches", "height_matches"]].all().all():
        raise RuntimeError("入选框的当前图片尺寸与标注尺寸不一致")
    coordinates = matched[
        ["box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm"]
    ].to_numpy(float)
    if (coordinates < 0).any() or (coordinates > 1).any():
        raise RuntimeError("框坐标超出[0,1]")
    if (coordinates[:, 0] >= coordinates[:, 2]).any() or (coordinates[:, 1] >= coordinates[:, 3]).any():
        raise RuntimeError("框坐标无效")
    return matched


def model_outputs(
    spatial: torch.Tensor, head: torch.nn.Module,
    classifier_weight: torch.Tensor, classifier_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """对给定空间表示重算冻结C-long输出。

    Args:
        spatial (torch.Tensor): ``[B,49,1280]``空间表示。
        head (torch.nn.Module): 冻结的1280到1注意力头。
        classifier_weight (torch.Tensor): ``[2,1280]``冻结分类权重。
        classifier_bias (torch.Tensor): ``[2]``冻结分类偏置。

    Returns:
        dict[str, torch.Tensor]: 注意力``[B,49]``、汇聚``[B,1280]``、
        margin``[B]``和癌概率``[B]``。
    """
    attention = attention_from_features(spatial, head)
    pooled = pooled_from_features(spatial, attention)
    logits = pooled @ classifier_weight.T + classifier_bias
    return {
        "attention": attention,
        "pooled": pooled,
        "margin": logits[:, 1] - logits[:, 0],
        "probability": logits.softmax(1)[:, 1],
    }


def summarize(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """按Feature或Feature与既有医学关系做小样本描述性汇总。

    Args:
        rows (pd.DataFrame): 12行逐配对干预结果。
        group_columns (list[str]): 分组列名。

    Returns:
        pd.DataFrame: 图数及各带符号/绝对效应的中位数，不含推断检验。
    """
    metrics = [
        "inside_delta_margin", "outside_delta_margin", "full_delta_margin",
        "inside_abs_delta_margin", "outside_abs_delta_margin", "full_abs_delta_margin",
        "inside_delta_probability", "outside_delta_probability", "full_delta_probability",
        "nonadditivity_delta_margin", "feature_box_response_fraction",
    ]
    grouped = rows.groupby(group_columns, sort=True, dropna=False)
    output = grouped.size().rename("images").reset_index()
    for metric in metrics:
        values = grouped[metric].median().rename(f"median_{metric}").reset_index()
        output = output.merge(values, on=group_columns, validate="one_to_one")
    return output


@torch.no_grad()
def analyze_pairs(pairs: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, dict]:
    """逐配对执行原始、框内、框外和整图四条独立路径。

    Args:
        pairs (pd.DataFrame): 12个已审核且有框的Feature–图片配对。
        device (torch.device): CPU或已核对的CUDA设备。

    Returns:
        tuple[pd.DataFrame, dict]: 逐配对结果和数值复现/分量恒等验证。
    """
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight
    output_rows = []
    component_errors = []
    cache_attention_errors = []
    cache_pool_errors = []

    for split in ("train", "val"):
        frame = pairs[pairs.split.eq(split)].copy().reset_index(drop=True)
        data = read_subset(frame, split, device)
        spatial = data["spatial"]
        hidden = sae.encode(spatial, K)
        original = model_outputs(spatial, head, classifier_weight, classifier_bias)
        cache_attention_errors.append(float((original["attention"] - data["attention"]).abs().max()))
        cache_pool_errors.append(float((original["pooled"] - data["pooled"]).abs().max()))

        for local, record in enumerate(frame.itertuples(index=False)):
            box = np.array([
                record.box_x1_norm, record.box_y1_norm,
                record.box_x2_norm, record.box_y2_norm,
            ], dtype=np.float64)
            overlap_np = cell_overlap_map(box)
            if not np.isfinite(overlap_np).all() or (overlap_np < 0).any() or (overlap_np > 1).any():
                raise RuntimeError("单元格框覆盖比例非法")
            overlap = torch.as_tensor(overlap_np, device=device, dtype=spatial.dtype)
            feature = int(record.feature_id)
            activation = hidden[local, :, feature]
            component = activation[:, None] * decoder[feature][None, :]
            inside_component = overlap[:, None] * component
            outside_component = (1.0 - overlap[:, None]) * component
            component_error = float((inside_component + outside_component - component).abs().max())
            component_errors.append(component_error)

            original_spatial = spatial[local:local + 1]
            conditions = {
                "inside": original_spatial - inside_component.unsqueeze(0),
                "outside": original_spatial - outside_component.unsqueeze(0),
                "full": original_spatial - component.unsqueeze(0),
            }
            original_one = {key: value[local:local + 1] for key, value in original.items()}
            original_attention = original_one["attention"][0]
            original_box_attention = float((original_attention * overlap).sum())
            total_activation = float(activation.sum())
            if not np.isfinite(total_activation) or total_activation <= 0:
                raise RuntimeError(f"{record.image_code}/F{feature:04d}没有正激活")

            row = {
                "feature_id": feature,
                "feature": f"RA-F{feature:04d}",
                "image_code": record.image_code,
                "split": split,
                "source_row": int(record.source_row),
                "patient_id": record.patient_id,
                "image_relpath": record.image_relpath,
                "relation_to_current_candidate_description": record.relation_to_current_candidate_description,
                "medical_description": record.medical_description,
                "interpretation_boundary": record.interpretation_boundary,
                "box_area_fraction": float(record.box_area_fraction),
                "feature_activation_sum": total_activation,
                "feature_box_response_fraction": float((activation * overlap).sum()) / total_activation,
                "original_margin": float(original_one["margin"][0]),
                "original_probability": float(original_one["probability"][0]),
                "original_attention_box_mass": original_box_attention,
                "existing_full_delta_margin": float(record.delta_margin),
                "existing_full_delta_probability": float(record.delta_probability),
            }
            for name, changed_spatial in conditions.items():
                changed = model_outputs(changed_spatial, head, classifier_weight, classifier_bias)
                delta_margin = float(changed["margin"][0] - original_one["margin"][0])
                delta_probability = float(changed["probability"][0] - original_one["probability"][0])
                changed_attention = changed["attention"][0]
                attention_box_mass = float((changed_attention * overlap).sum())
                row.update({
                    f"{name}_margin": float(changed["margin"][0]),
                    f"{name}_probability": float(changed["probability"][0]),
                    f"{name}_delta_margin": delta_margin,
                    f"{name}_abs_delta_margin": abs(delta_margin),
                    f"{name}_delta_probability": delta_probability,
                    f"{name}_abs_delta_probability": abs(delta_probability),
                    f"{name}_attention_box_mass": attention_box_mass,
                    f"{name}_attention_box_mass_change": attention_box_mass - original_box_attention,
                    f"{name}_attention_cosine": float(torch_functional.cosine_similarity(
                        original_attention[None, :], changed_attention[None, :], dim=1
                    )[0]),
                    f"{name}_attention_l1": float((changed_attention - original_attention).abs().sum()),
                    f"{name}_attention_peak_changed": bool(
                        changed_attention.argmax() != original_attention.argmax()
                    ),
                })
            row["nonadditivity_delta_margin"] = (
                row["full_delta_margin"] - row["inside_delta_margin"] - row["outside_delta_margin"]
            )
            row["full_margin_reproduction_error"] = (
                row["full_delta_margin"] - row["existing_full_delta_margin"]
            )
            row["full_probability_reproduction_error"] = (
                row["full_delta_probability"] - row["existing_full_delta_probability"]
            )
            output_rows.append(row)

    output = pd.DataFrame(output_rows).sort_values(
        ["feature_id", "split", "source_row"]
    ).reset_index(drop=True)
    numeric = output.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("输出中存在NaN或Inf")
    margin_error = float(output.full_margin_reproduction_error.abs().max())
    probability_error = float(output.full_probability_reproduction_error.abs().max())
    verification = {
        "reviewed_box_pairs": int(len(output)),
        "unique_patients": int(output.patient_id.nunique()),
        "counts_by_feature": {
            str(key): int(value) for key, value in output.groupby("feature_id").size().items()
        },
        "all_pairs_noncancer": True,
        "max_component_partition_abs_error": max(component_errors),
        "max_cached_attention_abs_error": max(cache_attention_errors),
        "max_cached_pooled_abs_error": max(cache_pool_errors),
        "max_full_margin_reproduction_abs_error": margin_error,
        "max_full_probability_reproduction_abs_error": probability_error,
        "full_effect_reproduction_tolerance": REPRODUCE_TOLERANCE,
        "full_effect_reproduced": bool(
            margin_error <= REPRODUCE_TOLERANCE and probability_error <= REPRODUCE_TOLERANCE
        ),
        "all_outputs_finite": True,
        "new_training": False,
        "new_medical_images_or_annotations": False,
        "test_read": False,
        "external_read": False,
    }
    if verification["max_component_partition_abs_error"] > 2e-6:
        raise RuntimeError("框内与框外分量之和不等于整图分量")
    if (verification["max_cached_attention_abs_error"] > 1e-4
            or verification["max_cached_pooled_abs_error"] > 1e-4):
        raise RuntimeError("原始空间特征重算的注意力或汇聚向量与冻结缓存不一致")
    if not verification["full_effect_reproduced"]:
        raise RuntimeError("整图删除未在容差内复现既有结果")
    return output, verification


def write_report(output_dir: Path, rows: pd.DataFrame, verification: dict) -> None:
    """写出逐例数值和不越过证据边界的简短说明。

    Args:
        output_dir (Path): 新建且独立的结果目录。
        rows (pd.DataFrame): 12行逐配对结果。
        verification (dict): 数值与数据边界验证结果。

    Returns:
        None: 写入CSV、JSON和Markdown。
    """
    rows.to_csv(output_dir / "reviewed_box_pair_effects.csv", index=False)
    summarize(rows, ["feature_id"]).to_csv(
        output_dir / "feature_summary.csv", index=False
    )
    summarize(rows, ["feature_id", "relation_to_current_candidate_description"]).to_csv(
        output_dir / "feature_relation_summary.csv", index=False
    )
    (output_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    definition = {
        "scope": "12 previously reviewed noncancer Feature-image pairs with verified boxes",
        "features": list(TARGET_FEATURES),
        "k": K,
        "inside": "F - w_p * h_pj * d_j",
        "outside": "F - (1-w_p) * h_pj * d_j",
        "full": "F - h_pj * d_j",
        "construction": "all edited representations independently constructed from original F and original h",
        "nonadditivity_delta_margin": "full delta margin - inside delta margin - outside delta margin",
        "nonadditivity_boundary": "descriptive warning against additive interpretation; not causal decomposition",
        "automatic_inside_outside_classification": False,
        "auc_or_generalization_analysis": False,
        "test_read": False,
        "external_read": False,
    }
    (output_dir / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    indexed = rows.set_index("image_code")
    nonadditivity_max = float(rows.nonadditivity_delta_margin.abs().max())
    lines = [
        "# 已审核Feature框内／框外功能定位",
        "",
        "本轮只分析F0868、F1089和F2388中已医学审核且有非癌矩形框的12个配对。",
        "框内、框外和整图删除都从同一原始表示独立构造，并重算冻结注意力和分类头。",
        "",
        "## 阅读边界",
        "",
        "- 逐例直接比较框内、框外和整图的带符号效应与绝对值；不设自动判定门槛，不把比值解释为贡献百分比。",
        "- 框内与框外的面积和被删除Feature分量大小不同；`Delta margin`更大只是本例总删除效应更大，不表示单位面积更重要。",
        "- `非加和差值 = Δmargin整图 - Δmargin框内 - Δmargin框外`只提示局部删除效应不能简单相加，不是额外的因果贡献分解。",
        "- 整图净效应小可能来自框内外方向相反或注意力重分配，不单独解释为总体作用弱。",
        "- 这是7x7粗网格内部表示干预，不能定位到某个具体医学结构，框外效应也不自动等于伪相关。",
        "- F0868不包含无框的强反例；F1089只比较幽门附近与无幽门隆起两类候选模式；F2388包含一个有框疑似反例。",
        "",
        "## 主要发现",
        "",
        "- F0868：`train_0057`、`train_0097`和`val_0456`的框内删除|Delta margin|大于框外；"
        "`val_0027`的Feature框内响应占比为0，框内删除也为0，整图效应来自框外分量。"
        "本轮仍不能回答无框正常黏膜强反例的空间来源。",
        "- F1089：两张幽门附近主要模式图和两张无幽门隆起候选图中，"
        "框外删除的|Delta margin|均大于框内删除。这只表示当前Feature功能效应更多随标注框外分量删除而改变，"
        "不把框外区域命名为伪相关。",
        "- F2388：有框疑似反例`train_0507`的框内响应和框内删除效应均近似0，"
        f"框外删除Delta margin={indexed.loc['train_0507', 'outside_delta_margin']:.6f}，与整图删除一致。"
        "正的删除变化表示该框外分量在本次内部干预定义下原本起降低癌margin的作用；"
        "它位于框外，不能直接证明是有害依赖。"
        "两张val主要模式图的框内删除变化更大；`train_0722`则出现框内为正、框外为负的反向效应（"
        f"{indexed.loc['train_0722', 'inside_delta_margin']:.6f} vs "
        f"{indexed.loc['train_0722', 'outside_delta_margin']:.6f}），整图净变化仅"
        f"{indexed.loc['train_0722', 'full_delta_margin']:.6f}，直接说明小的整图净效应不等于局部作用都弱。",
        f"- 12例的|非加和差值|最大为{nonadditivity_max:.6f}。F2388的注意力重分配最明显；"
        f"`train_0722`框内删除的attention L1变化为"
        f"{indexed.loc['train_0722', 'inside_attention_l1']:.6f}，注意力框内质量改变为"
        f"{indexed.loc['train_0722', 'inside_attention_box_mass_change']:.6f}。这些只是互作和非加和的描述性证据。",
        "",
        "## 验证",
        "",
        f"- 12个配对、12位患者；三项Feature各{verification['counts_by_feature']['868']}例。",
        f"- 整图删除margin最大复现误差：{verification['max_full_margin_reproduction_abs_error']:.3e}。",
        f"- 整图删除概率最大复现误差：{verification['max_full_probability_reproduction_abs_error']:.3e}。",
        f"- 框内分量+框外分量的最大误差：{verification['max_component_partition_abs_error']:.3e}。",
        "- 未读取internal test或external，未训练模型，未增加医学图片或标注。",
        "",
        "完整逐例数值见 `reviewed_box_pair_effects.csv`；汇总表只作描述，不计算AUC或推断统计。",
    ]
    (output_dir / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """运行12个已审核配对的框内／框外功能定位。

    Args:
        None: 输出目录和设备由CLI提供。

    Returns:
        None: 写入逐例、描述性汇总、分析口径和验证结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = load_reviewed_box_pairs()
    rows, verification = analyze_pairs(pairs, torch.device(args.device))
    write_report(args.output, rows, verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
