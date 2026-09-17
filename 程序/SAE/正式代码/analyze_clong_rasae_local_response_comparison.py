#!/usr/bin/env python3
"""比较16个已审核配对的原C-long局部响应与RA-SAE单Feature局部响应。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_s2b_core import ACTIVE_EPS, attention_from_features, pooled_from_features  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
REVIEWED = BASE / "reviewed_response_function_20260915/reviewed_pair_effects.csv"
Q99_PATH = BASE / "decision_alignment/ra_train_q99.npy"
TARGET_FEATURES = (868, 1089)
K = 256
DISPLAY_ALPHA = 0.42
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def safe_spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    """响应恒定时返回NaN并记录不可评价。"""
    left, right = np.asarray(left), np.asarray(right)
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan, "constant"
    value = float(spearmanr(left, right).statistic)
    return (value, "evaluable") if np.isfinite(value) else (np.nan, "nonfinite")


def fraction_in_mask(values: np.ndarray, mask: np.ndarray) -> float:
    """返回绝对质量落在mask内的比例；总质量近零时为NaN。"""
    total = float(np.abs(values).sum())
    return np.nan if total <= ACTIVE_EPS else float(np.abs(values[mask]).sum() / total)


def load_pairs() -> pd.DataFrame:
    """读取F0868/F1089全部16个既有审核配对。"""
    pairs = pd.read_csv(REVIEWED)
    pairs = pairs[pairs.feature_id.isin(TARGET_FEATURES)].copy()
    if pairs.groupby("feature_id").size().to_dict() != {868: 8, 1089: 8}:
        raise RuntimeError("目标配对不是F0868/F1089各8例")
    if pairs.duplicated(["feature_id", "image_code"]).any():
        raise RuntimeError("目标Feature-图片对重复")
    return pairs.sort_values(["feature_id", "split", "source_row"]).reset_index(drop=True)


@torch.no_grad()
def local_maps(
    spatial: torch.Tensor, hidden: torch.Tensor, decoder: torch.Tensor,
    head: torch.nn.Module, classifier_weight: torch.Tensor, classifier_bias: torch.Tensor,
    feature: int,
) -> dict[str, np.ndarray | float]:
    """计算注意力、原净贡献、固定注意力Feature项及49次单位置删除效应。"""
    attention = attention_from_features(spatial, head)[0]
    pooled = pooled_from_features(spatial, attention[None, :])
    logits = pooled @ classifier_weight.T + classifier_bias
    margin = logits[0, 1] - logits[0, 0]
    margin_weight = classifier_weight[1] - classifier_weight[0]
    margin_bias = classifier_bias[1] - classifier_bias[0]
    patch_margin = spatial[0] @ margin_weight
    original_contribution = attention * patch_margin
    activation = hidden[0, :, feature]
    direction_margin = decoder[feature] @ margin_weight
    feature_additive = attention * activation * direction_margin
    remainder = original_contribution - feature_additive

    repeated = spatial.repeat(49, 1, 1)
    indices = torch.arange(49, device=spatial.device)
    repeated[indices, indices] -= activation[:, None] * decoder[feature][None, :]
    changed_attention = attention_from_features(repeated, head)
    changed_pool = pooled_from_features(repeated, changed_attention)
    changed_logits = changed_pool @ classifier_weight.T + classifier_bias
    changed_margin = changed_logits[:, 1] - changed_logits[:, 0]
    local_delta = changed_margin - margin
    local_support = -local_delta
    attention_recompute_difference = local_support - feature_additive

    return {
        "attention": attention.cpu().numpy(),
        "original_contribution": original_contribution.cpu().numpy(),
        "activation": activation.cpu().numpy(),
        "feature_additive": feature_additive.cpu().numpy(),
        "remainder_contribution": remainder.cpu().numpy(),
        "local_delta_margin": local_delta.cpu().numpy(),
        "local_support": local_support.cpu().numpy(),
        "attention_recompute_difference": attention_recompute_difference.cpu().numpy(),
        "original_margin": float(margin), "margin_bias": float(margin_bias),
    }


@torch.no_grad()
def train_scales(
    sae: torch.nn.Module, head: torch.nn.Module, classifier_weight: torch.Tensor,
    device: torch.device,
) -> dict:
    """仅用完整train计算attention、C和每目标Feature B的固定显示尺度。"""
    frame = pd.read_csv(CACHE / "train_metadata.csv").reset_index().rename(
        columns={"index": "source_row"}
    )
    margin_weight = classifier_weight[1] - classifier_weight[0]
    attention_values, contribution_values = [], []
    feature_values = {feature: [] for feature in TARGET_FEATURES}
    for start in range(0, len(frame), 32):
        stop = min(start + 32, len(frame))
        spatial = read_subset(frame.iloc[start:stop], "train", device)["spatial"]
        attention = attention_from_features(spatial, head)
        patch_margin = spatial @ margin_weight
        contribution = attention * patch_margin
        hidden = sae.encode(spatial, K)
        attention_values.append(attention.cpu().numpy().reshape(-1))
        contribution_values.append(np.abs(contribution.cpu().numpy()).reshape(-1))
        for feature in TARGET_FEATURES:
            direction_margin = sae.decoder_weight[feature] @ margin_weight
            value = attention * hidden[:, :, feature] * direction_margin
            feature_values[feature].append(np.abs(value.cpu().numpy()).reshape(-1))
    scales = {
        "attention_q99": float(np.quantile(np.concatenate(attention_values), 0.99)),
        "original_contribution_abs_q99": float(
            np.quantile(np.concatenate(contribution_values), 0.99)
        ),
        "feature_additive_abs_q99": {
            str(feature): float(np.quantile(np.concatenate(feature_values[feature]), 0.99))
            for feature in TARGET_FEATURES
        },
    }
    if min(scales["attention_q99"], scales["original_contribution_abs_q99"],
           *scales["feature_additive_abs_q99"].values()) <= 0:
        raise RuntimeError("train-only显示尺度非正")
    return scales


def analyze_pairs(pairs: pd.DataFrame, device: torch.device) -> tuple[pd.DataFrame, dict, dict]:
    """计算16对局部图和描述性指标，并执行分解与整图删除复现核对。"""
    sae, head, classifier_weight, classifier_bias = load_models(device)
    q99 = np.load(Q99_PATH)
    maps, rows = {}, []
    decomposition_errors, margin_errors, full_errors = [], [], []
    for record in pairs.itertuples(index=False):
        frame = pd.DataFrame([record._asdict()])
        spatial = read_subset(frame, record.split, device)["spatial"]
        hidden = sae.encode(spatial, K)
        result = local_maps(
            spatial, hidden, sae.decoder_weight, head, classifier_weight,
            classifier_bias, int(record.feature_id),
        )
        feature = int(record.feature_id)
        activation_normalized = result["activation"] / q99[feature]
        high = activation_normalized >= 0.5
        contribution = result["original_contribution"]
        additive = result["feature_additive"]
        remainder = result["remainder_contribution"]
        support = result["local_support"]
        decomposition_errors.append(float(np.max(np.abs(contribution - additive - remainder))))
        margin_errors.append(abs(
            float(contribution.sum()) + result["margin_bias"] - result["original_margin"]
        ))
        full_changed = spatial - hidden[:, :, feature, None] * sae.decoder_weight[feature]
        full_attention = attention_from_features(full_changed, head)
        full_pool = pooled_from_features(full_changed, full_attention)
        full_logits = full_pool @ classifier_weight.T + classifier_bias
        full_delta = float(full_logits[0, 1] - full_logits[0, 0] - result["original_margin"])
        full_errors.append(abs(full_delta - float(record.delta_margin)))

        activation_attention, activation_attention_status = safe_spearman(
            activation_normalized, result["attention"]
        )
        activation_contribution, activation_contribution_status = safe_spearman(
            activation_normalized, np.abs(contribution)
        )
        support_contribution, support_contribution_status = safe_spearman(
            support, contribution
        )
        additive_support, additive_support_status = safe_spearman(additive, support)
        additive_contribution, additive_contribution_status = safe_spearman(
            additive, contribution
        )
        sign_mask = (np.abs(support) > ACTIVE_EPS) & (np.abs(contribution) > ACTIVE_EPS)
        sign_count = int(sign_mask.sum())
        sign_agreement = (
            float(np.mean(np.sign(support[sign_mask]) == np.sign(contribution[sign_mask])))
            if sign_count else np.nan
        )
        denom = float(np.abs(additive).sum() + np.abs(remainder).sum())
        cancellation = np.nan if denom <= ACTIVE_EPS else 1 - float(np.abs(contribution).sum()) / denom
        support_abs_sum = float(np.abs(support).sum())
        recompute_abs_sum = float(np.abs(result["attention_recompute_difference"]).sum())
        row = {
            "feature_id": feature, "feature": f"RA-F{feature:04d}",
            "image_code": record.image_code, "split": record.split,
            "source_row": int(record.source_row), "patient_id": record.patient_id,
            "label_internal": int(record.label_internal),
            "relation_to_current_candidate_description": record.relation_to_current_candidate_description,
            "medical_description": record.medical_description,
            "high_activation_position_count": int(high.sum()),
            "attention_mass_in_high_activation_positions": (
                float(result["attention"][high].sum()) if high.any() else np.nan
            ),
            "abs_original_contribution_fraction_in_high_activation_positions": (
                fraction_in_mask(contribution, high) if high.any() else np.nan
            ),
            "activation_attention_spearman": activation_attention,
            "activation_attention_status": activation_attention_status,
            "activation_abs_original_contribution_spearman": activation_contribution,
            "activation_abs_original_contribution_status": activation_contribution_status,
            "local_support_original_contribution_spearman": support_contribution,
            "local_support_original_contribution_status": support_contribution_status,
            "fixed_additive_local_support_spearman": additive_support,
            "fixed_additive_local_support_status": additive_support_status,
            "fixed_additive_original_contribution_spearman": additive_contribution,
            "fixed_additive_original_contribution_status": additive_contribution_status,
            "sign_comparison_position_count": sign_count,
            "local_support_original_contribution_sign_agreement": sign_agreement,
            "fixed_attention_feature_abs_mass": float(np.abs(additive).sum()),
            "fixed_attention_remainder_abs_mass": float(np.abs(remainder).sum()),
            "original_net_contribution_abs_mass": float(np.abs(contribution).sum()),
            "fixed_attention_cancellation_fraction": cancellation,
            "attention_recompute_abs_difference_sum": recompute_abs_sum,
            "attention_recompute_fraction_of_local_support": (
                np.nan if support_abs_sum <= ACTIVE_EPS else recompute_abs_sum / support_abs_sum
            ),
            "local_support_abs_sum": support_abs_sum,
            "full_delta_margin_recomputed": full_delta,
            "existing_full_delta_margin": float(record.delta_margin),
        }
        rows.append(row)
        maps[(feature, record.image_code)] = {
            **result, "activation_normalized": activation_normalized,
        }
    verification = {
        "pairs": len(rows), "active_eps_for_sign": ACTIVE_EPS,
        "max_C_equals_B_plus_remainder_error": max(decomposition_errors),
        "max_margin_additive_reconstruction_error": max(margin_errors),
        "max_full_deletion_reproduction_error": max(full_errors),
        "test_read": False, "external_read": False, "new_training": False,
        "new_medical_annotation": False,
    }
    if max(decomposition_errors + margin_errors + full_errors) > 1e-5:
        raise RuntimeError(f"局部分解或整图删除复现失败: {verification}")
    return pd.DataFrame(rows), maps, verification


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """按Feature和既有反馈关系做中位数描述；不可评价数量单独保留。"""
    metrics = [
        "high_activation_position_count", "attention_mass_in_high_activation_positions",
        "abs_original_contribution_fraction_in_high_activation_positions",
        "activation_attention_spearman", "activation_abs_original_contribution_spearman",
        "local_support_original_contribution_spearman",
        "fixed_additive_local_support_spearman",
        "fixed_additive_original_contribution_spearman",
        "local_support_original_contribution_sign_agreement",
        "fixed_attention_cancellation_fraction", "attention_recompute_abs_difference_sum",
        "attention_recompute_fraction_of_local_support",
        "local_support_abs_sum",
    ]
    output = []
    for keys, group in rows.groupby(
        ["feature", "relation_to_current_candidate_description"], sort=True
    ):
        row = {"feature": keys[0], "feedback_relation": keys[1], "pairs": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row[f"median_{metric}"] = float(values.median()) if len(values) else np.nan
            row[f"evaluable_{metric}"] = len(values)
        output.append(row)
    return pd.DataFrame(output)


def dense(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将7×7量双线性插值到图片尺寸。"""
    return torch.nn.functional.interpolate(
        torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, 1, 7, 7),
        size=shape, mode="bilinear", align_corners=False,
    )[0, 0].numpy()


def render(
    pairs: pd.DataFrame, maps: dict, scales: dict, output: Path,
) -> None:
    """每项Feature生成一张8行×6列内部技术对照图。"""
    for feature in TARGET_FEATURES:
        current = pairs[pairs.feature_id.eq(feature)].reset_index(drop=True)
        figure, axes = plt.subplots(8, 6, figsize=(24, 31), squeeze=False)
        signed_scale_c = scales["original_contribution_abs_q99"]
        signed_scale_b = scales["feature_additive_abs_q99"][str(feature)]
        for row_index, record in current.iterrows():
            image = Image.open(ROOT / record.image_relpath).convert("RGB")
            image.thumbnail((800, 800))
            rgb = np.asarray(image)
            item = maps[(feature, record.image_code)]
            panels = [
                ("原图", None, None, None),
                ("C-long注意力", item["attention"], "magma", scales["attention_q99"]),
                ("原空间净贡献 C", item["original_contribution"], "coolwarm", signed_scale_c),
                ("SAE激活 / Q99", item["activation_normalized"], "magma", 1.0),
                ("固定注意力Feature项 B", item["feature_additive"], "coolwarm", signed_scale_b),
                ("SAE局部功能支持", item["local_support"], "coolwarm", signed_scale_b),
            ]
            for column, (title, values, cmap, scale) in enumerate(panels):
                axis = axes[row_index, column]
                axis.imshow(rgb)
                if values is not None:
                    current_dense = dense(values, rgb.shape[:2])
                    if cmap == "coolwarm":
                        axis.imshow(
                            current_dense, cmap=cmap,
                            norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
                            alpha=DISPLAY_ALPHA,
                        )
                    else:
                        axis.imshow(
                            current_dense, cmap=cmap, vmin=0, vmax=scale,
                            alpha=DISPLAY_ALPHA,
                        )
                relation = str(record.relation_to_current_candidate_description)
                axis.set_title(
                    f"{record.image_code} | {title}" if column == 0 else title,
                    fontsize=9,
                )
                axis.axis("off")
            axes[row_index, 0].text(
                0, -0.08, relation, transform=axes[row_index, 0].transAxes, fontsize=8,
            )
        figure.suptitle(
            f"RA-F{feature:04d} 局部响应技术对照｜各量使用各自train-only固定尺度，颜色亮度不可跨列比较",
            fontsize=16,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        figure.savefig(output / f"RA-F{feature:04d}_局部响应对照.png", dpi=130, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """执行16配对数值分析和两张内部技术图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    device = torch.device(args.device)
    sae, head, classifier_weight, _classifier_bias = load_models(device)
    scales = train_scales(sae, head, classifier_weight, device)
    rows, maps, verification = analyze_pairs(pairs, device)
    rows.to_csv(args.output / "逐配对局部响应.csv", index=False)
    summarize(rows).to_csv(args.output / "按反馈关系汇总.csv", index=False)
    np.savez_compressed(
        args.output / "16对局部响应图.npz",
        **{
            f"F{feature:04d}_{code}_{name}": value
            for (feature, code), item in maps.items()
            for name, value in item.items() if isinstance(value, np.ndarray)
        },
    )
    render(pairs, maps, scales, args.output)
    definition = {
        "attention": "original C-long spatial softmax pooling weight; sums to one",
        "C_p": "A_p * (w_margin dot F_p); all-channel net additive term",
        "B_pj": "A_p * h_pj * (w_margin dot d_j); target Feature term at fixed original attention",
        "remainder": "C_p - B_pj; all other components and residual at fixed attention",
        "sae_local_support": "negative of margin change after deleting one Feature component at one position and recomputing attention",
        "attention_recompute_difference": "sae_local_support - B_pj",
        "sign_tolerance": ACTIVE_EPS,
        "display_scales": {**scales, "activation_q99": "existing per-Feature train positive Q99"},
        "display_alpha": DISPLAY_ALPHA,
        "display_warning": "each quantity has its own fixed train-only scale; color brightness is not comparable across columns",
        "medical_region_mapping": "none; textual feedback has no exact pixel/grid annotation",
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
