#!/usr/bin/env python3
"""在完整val上评价RA-SAE每图梯度排名Top-k Feature的residual保留增量解释价值。

固定原100轮RA-SAE和编码K=256。对每张图在原始表示处计算既有激活×梯度分数，
构造``residual + decoder_bias + Top-k components``，k为0/1/3/6/10/20；全分量用于no-op核对。
本评价不训练、不重排、不读取test/external，不评价医学语义。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from clong_sae_discovery import FROZEN_IMAGE_THRESHOLD, FROZEN_PATIENT_THRESHOLD  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
ENCODER_K = 256
RETAIN_COUNTS = (1, 3, 6, 10, 20)
LEVELS = ("k0", "k1", "k3", "k6", "k10", "k20", "all")


def analytic_margin_gradient(
    spatial: torch.Tensor,
    attention: torch.Tensor,
    attention_weight: torch.Tensor,
    margin_weight: torch.Tensor,
) -> torch.Tensor:
    """计算原C-long癌-非癌logit margin对每个空间特征的解析梯度。

    Args:
        spatial (torch.Tensor): ``[B,49,D]``原始空间表示。
        attention (torch.Tensor): ``[B,49]``原始softmax注意力。
        attention_weight (torch.Tensor): ``[D]``注意力1x1卷积方向。
        margin_weight (torch.Tensor): ``[D]``癌-非癌分类头方向。

    Returns:
        torch.Tensor: ``[B,49,D]``完整注意力链路的margin梯度。
    """
    patch_margin = spatial @ margin_weight
    pooled_margin = (attention * patch_margin).sum(1, keepdim=True)
    return attention.unsqueeze(2) * (
        margin_weight + (patch_margin - pooled_margin).unsqueeze(2) * attention_weight
    )


def deterministic_top_ids(scores: np.ndarray, count: int) -> np.ndarray:
    """按绝对分数降序、Feature ID升序确定性选取Top-k。

    Args:
        scores (np.ndarray): ``[B,H]``带符号梯度功能分数。
        count (int): 每张图保留的Feature数。

    Returns:
        np.ndarray: ``[B,count]`` Feature ID，每行按排名顺序。
    """
    if scores.ndim != 2 or not 0 < count <= scores.shape[1]:
        raise ValueError("scores/count维度异常")
    feature_ids = np.arange(scores.shape[1])
    return np.stack([
        np.lexsort((feature_ids, -np.abs(row)))[:count] for row in scores
    ])


def retained_representation(
    base: torch.Tensor, hidden: torch.Tensor, decoder: torch.Tensor, ids: torch.Tensor,
) -> torch.Tensor:
    """在residual+decoder bias基础上加回每图指定Feature的全49位分量。

    Args:
        base (torch.Tensor): ``[B,49,D]``，等于``F-hD``。
        hidden (torch.Tensor): ``[B,49,H]`` K=256编码激活。
        decoder (torch.Tensor): ``[H,D]`` decoder方向。
        ids (torch.Tensor): ``[B,k]``每图要保留的Feature ID。

    Returns:
        torch.Tensor: ``[B,49,D]`` residual保留的Top-k表示。
    """
    if ids.ndim != 2 or ids.shape[0] != hidden.shape[0]:
        raise ValueError("ids必须是[B,k]")
    selected_hidden = torch.gather(
        hidden, 2, ids[:, None, :].expand(hidden.shape[0], hidden.shape[1], ids.shape[1])
    )
    selected_decoder = decoder[ids]
    return base + torch.einsum("bpk,bkd->bpd", selected_hidden, selected_decoder)


def aggregate_patients(images: pd.DataFrame) -> pd.DataFrame:
    """将各level逐图概率按患者内图像等权平均。

    Args:
        images (pd.DataFrame): 每图每level一行的概率表。

    Returns:
        pd.DataFrame: 每患者每level一行的原始/保留概率和变化。
    """
    patients = images.groupby(["level", "patient_id"], as_index=False, sort=False).agg(
        label=("label", "first"), images=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        retained_probability=("retained_probability", "mean"),
    )
    patients["delta_probability"] = patients.retained_probability - patients.original_probability
    patients["abs_delta_probability"] = patients.delta_probability.abs()
    return patients


def summarize_curve(images: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """汇总Top-k相对k=0的概率误差恢复、决策一致和AUC。

    Args:
        images (pd.DataFrame): 逐图各level结果。
        patients (pd.DataFrame): 逐患者各level结果。

    Returns:
        pd.DataFrame: 按``LEVELS``排列的恢复曲线表。
    """
    rows = []
    residual_patient = patients[patients.level.eq("k0")]
    residual_image = images[images.level.eq("k0")]
    patient_mae0 = float(residual_patient.abs_delta_probability.mean())
    image_mae0 = float(residual_image.abs_delta_probability.mean())
    original_patient_prediction = residual_patient.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
    residual_patient_prediction = residual_patient.retained_probability.ge(FROZEN_PATIENT_THRESHOLD)
    residual_wrong = pd.Series(
        (original_patient_prediction != residual_patient_prediction).to_numpy(),
        index=residual_patient.patient_id.astype(str),
    )
    flips0 = int(residual_wrong.sum())
    for level in LEVELS:
        image = images[images.level.eq(level)]
        patient = patients[patients.level.eq(level)]
        image_mae = float(image.abs_delta_probability.mean())
        patient_mae = float(patient.abs_delta_probability.mean())
        image_prediction = image.retained_probability.ge(FROZEN_IMAGE_THRESHOLD)
        original_image_prediction = image.original_probability.ge(FROZEN_IMAGE_THRESHOLD)
        patient_prediction = patient.retained_probability.ge(FROZEN_PATIENT_THRESHOLD)
        original_patient_prediction = patient.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
        current_wrong = pd.Series(
            (patient_prediction != original_patient_prediction).to_numpy(),
            index=patient.patient_id.astype(str),
        ).reindex(residual_wrong.index)
        if current_wrong.isna().any():
            raise RuntimeError(f"{level}患者集合与k0不一致")
        patient_flips = int(current_wrong.sum())
        patient_abs = patient.abs_delta_probability.to_numpy()
        image_abs = image.abs_delta_probability.to_numpy()
        row = {
            "level": level,
            "retained_feature_count": 2560 if level == "all" else int(level.removeprefix("k")),
            "image_probability_mae": image_mae,
            "image_probability_abs_error_median": float(np.median(image_abs)),
            "image_probability_abs_error_p95": float(np.quantile(image_abs, 0.95)),
            "image_probability_abs_error_max": float(image_abs.max()),
            "patient_probability_mae": patient_mae,
            "patient_probability_abs_error_median": float(np.median(patient_abs)),
            "patient_probability_abs_error_p95": float(np.quantile(patient_abs, 0.95)),
            "patient_probability_abs_error_max": float(patient_abs.max()),
            "image_error_recovery_vs_k0": (
                None if image_mae0 == 0 else (image_mae0 - image_mae) / image_mae0
            ),
            "patient_error_recovery_vs_k0": (
                None if patient_mae0 == 0 else (patient_mae0 - patient_mae) / patient_mae0
            ),
            "image_prediction_agreement": float((image_prediction == original_image_prediction).mean()),
            "patient_prediction_agreement": float((patient_prediction == original_patient_prediction).mean()),
            "image_flip_count": int((image_prediction != original_image_prediction).sum()),
            "patient_flip_count": patient_flips,
            "patients_recovered_to_original_vs_k0": int((residual_wrong & ~current_wrong).sum()),
            "patients_newly_inconsistent_vs_k0": int((~residual_wrong & current_wrong).sum()),
            "patient_flip_reduction_vs_k0": flips0 - patient_flips,
            "image_auc": float(roc_auc_score(image.label, image.retained_probability)),
            "patient_auc": float(roc_auc_score(patient.label, patient.retained_probability)),
        }
        for label, name in ((1, "cancer"), (0, "noncancer")):
            selected = patient[patient.label.eq(label)]
            row[f"{name}_patient_mean_signed_delta_probability"] = float(selected.delta_probability.mean())
            row[f"{name}_patient_mean_abs_delta_probability"] = float(selected.abs_delta_probability.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_image_labels(images: pd.DataFrame) -> pd.DataFrame:
    """按level和癌/非癌标签汇总逐图概率变化。

    Args:
        images (pd.DataFrame): 逐图各level结果。

    Returns:
        pd.DataFrame: 带符号均值以及绝对误差均值/中位数/P95/最大值。
    """
    rows = []
    for level in LEVELS:
        for label, name in ((1, "cancer"), (0, "noncancer")):
            selected = images[images.level.eq(level) & images.label.eq(label)]
            absolute = selected.abs_delta_probability.to_numpy()
            rows.append({
                "level": level, "label": label, "label_name": name, "images": len(selected),
                "mean_signed_delta_probability": float(selected.delta_probability.mean()),
                "mean_abs_delta_probability": float(absolute.mean()),
                "median_abs_delta_probability": float(np.median(absolute)),
                "p95_abs_delta_probability": float(np.quantile(absolute, 0.95)),
                "max_abs_delta_probability": float(absolute.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    """在完整val497上运行k=0/1/3/6/10/20/all residual保留实验。

    Args:
        None: 设备、批量和输出目录由CLI提供。

    Returns:
        None: 写入排名、逐图/患者、恢复曲线和验证产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    definition = {
        "model": "original fixed RA-SAE duration100, not expanded-pool candidate",
        "cohort": "full val 497 images / 260 patients",
        "encoder_k": ENCODER_K,
        "retain_counts_per_image": list(RETAIN_COUNTS),
        "ranking": "abs(sum_p h_pj * (grad_Fp cancer-minus-noncancer margin dot decoder_j)); feature ID ascending ties",
        "ranking_point": "original spatial representation; computed once per image and reused for every retain count",
        "construction": "residual + decoder_bias + selected components; k0 retains residual+bias; all retains all encoded components",
        "patient_probability": "mean image cancer probability per patient",
        "primary": "probability error and prediction agreement relative to k0",
        "recovery_denominator_zero": "not applicable; no epsilon added",
        "limits": [
            "residual may retain information overlapping SAE components",
            "only this fixed ranking, retain construction and k set are evaluated",
            "limited Top-k recovery does not imply the full dictionary lacks explanatory value",
            "medical semantic stability is not evaluated",
        ],
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = pd.read_csv(CACHE / "val_metadata.csv").reset_index().rename(columns={"index": "source_row"})
    if len(frame) != 497 or frame.patient_id.nunique() != 260:
        raise RuntimeError("val队列不是497图/260人")
    device = torch.device(args.device)
    sae, head, classifier_weight, classifier_bias = load_models(device)
    decoder = sae.decoder_weight.detach()
    attention_weight = head.weight.detach().flatten()
    margin_weight = (classifier_weight[1] - classifier_weight[0]).detach()
    image_rows, ranking_rows = [], []
    verification = {
        "max_analytic_vs_autograd_gradient_error": 0.0,
        "max_k0_formula_error": 0.0,
        "max_all_spatial_error": 0.0,
        "max_all_attention_error": 0.0,
        "max_all_margin_error": 0.0,
        "max_all_probability_error": 0.0,
        "max_explicit_retain_error": 0.0,
        "original_probability_cache_error": 0.0,
        "ranking_ties": "feature_id_ascending",
        "all_selected_ids_unique": True,
        "test_read": False, "external_read": False,
    }
    gradient_checked = explicit_checked = False
    for start in range(0, len(frame), args.batch_size):
        stop = min(start + args.batch_size, len(frame))
        batch_frame = frame.iloc[start:stop]
        data = read_subset(batch_frame, "val", device)
        spatial = data["spatial"]
        with torch.no_grad():
            hidden = sae.encode(spatial, ENCODER_K)
            original_attention = attention_from_features(spatial, head)
            original_pool = pooled_from_features(spatial, original_attention)
            original_logits = original_pool @ classifier_weight.T + classifier_bias
            original_margin = original_logits[:, 1] - original_logits[:, 0]
            original_probability = original_logits.softmax(1)[:, 1]
            gradient = analytic_margin_gradient(
                spatial, original_attention, attention_weight, margin_weight
            )
            signed_scores = (hidden * (gradient @ decoder.T)).sum(1)
            top20_np = deterministic_top_ids(signed_scores.cpu().numpy(), max(RETAIN_COUNTS))
            top20 = torch.as_tensor(top20_np, dtype=torch.long, device=device)
            full_component = hidden @ decoder
            decoded = full_component + sae.decoder_bias
            residual = spatial - decoded
            base = residual + sae.decoder_bias
            verification["max_k0_formula_error"] = max(
                verification["max_k0_formula_error"], float((base - (spatial - full_component)).abs().max())
            )
            representations = {"k0": base}
            for count in RETAIN_COUNTS:
                representations[f"k{count}"] = retained_representation(
                    base, hidden, decoder, top20[:, :count]
                )
            representations["all"] = base + full_component
            all_attention = attention_from_features(representations["all"], head)
            all_pool = pooled_from_features(representations["all"], all_attention)
            all_logits = all_pool @ classifier_weight.T + classifier_bias
            all_margin = all_logits[:, 1] - all_logits[:, 0]
            all_probability = all_logits.softmax(1)[:, 1]
            verification["max_all_spatial_error"] = max(
                verification["max_all_spatial_error"], float((representations["all"] - spatial).abs().max())
            )
            verification["max_all_attention_error"] = max(
                verification["max_all_attention_error"], float((all_attention - original_attention).abs().max())
            )
            verification["max_all_margin_error"] = max(
                verification["max_all_margin_error"], float((all_margin - original_margin).abs().max())
            )
            verification["max_all_probability_error"] = max(
                verification["max_all_probability_error"], float((all_probability - original_probability).abs().max())
            )
            expected_probability = torch.as_tensor(
                batch_frame.cancer_probability.to_numpy(dtype=np.float32, copy=True), device=device
            )
            verification["original_probability_cache_error"] = max(
                verification["original_probability_cache_error"],
                float((original_probability - expected_probability).abs().max()),
            )
        if not gradient_checked:
            sample = spatial[:2].detach().clone().requires_grad_(True)
            sample_attention = attention_from_features(sample, head)
            sample_pool = pooled_from_features(sample, sample_attention)
            sample_logits = sample_pool @ classifier_weight.T + classifier_bias
            sample_margin = sample_logits[:, 1] - sample_logits[:, 0]
            autograd_value = torch.autograd.grad(sample_margin.sum(), sample)[0]
            analytic_value = analytic_margin_gradient(
                sample.detach(), sample_attention.detach(), attention_weight, margin_weight
            )
            verification["max_analytic_vs_autograd_gradient_error"] = float(
                (autograd_value - analytic_value).abs().max()
            )
            gradient_checked = True
        with torch.no_grad():
            if not explicit_checked:
                ids = top20[:2, :6]
                direct = base[:2].clone()
                for row in range(len(ids)):
                    for feature in ids[row]:
                        direct[row] += hidden[row, :, feature, None] * decoder[feature]
                vectorized = retained_representation(base[:2], hidden[:2], decoder, ids)
                verification["max_explicit_retain_error"] = float((direct - vectorized).abs().max())
                explicit_checked = True
            outputs = {}
            for level, representation in representations.items():
                attention = attention_from_features(representation, head)
                pooled = pooled_from_features(representation, attention)
                logits = pooled @ classifier_weight.T + classifier_bias
                outputs[level] = {
                    "probability": logits.softmax(1)[:, 1],
                    "margin": logits[:, 1] - logits[:, 0],
                }
            for local, record in enumerate(batch_frame.itertuples()):
                ranked_ids = top20_np[local]
                if len(np.unique(ranked_ids)) != len(ranked_ids):
                    verification["all_selected_ids_unique"] = False
                for rank, feature in enumerate(ranked_ids, 1):
                    ranking_rows.append({
                        "source_row": int(record.source_row), "patient_id": str(record.patient_id),
                        "rank": rank, "feature_id": int(feature),
                        "signed_gradient_activation": float(signed_scores[local, feature]),
                        "abs_gradient_activation": float(abs(signed_scores[local, feature])),
                        "feature_active": bool(hidden[local, :, feature].gt(1e-8).any()),
                    })
                for level in LEVELS:
                    probability = outputs[level]["probability"][local]
                    margin = outputs[level]["margin"][local]
                    image_rows.append({
                        "source_row": int(record.source_row), "patient_id": str(record.patient_id),
                        "label": int(record.label), "level": level,
                        "original_probability": float(original_probability[local]),
                        "retained_probability": float(probability),
                        "delta_probability": float(probability - original_probability[local]),
                        "abs_delta_probability": float(abs(probability - original_probability[local])),
                        "original_margin": float(original_margin[local]),
                        "retained_margin": float(margin),
                        "delta_margin": float(margin - original_margin[local]),
                    })
        print(f"val: {stop}/{len(frame)}", flush=True)
    if not verification["all_selected_ids_unique"]:
        raise RuntimeError("Top-20排名出现重复Feature ID")
    for key in (
        "max_analytic_vs_autograd_gradient_error", "max_k0_formula_error",
        "max_all_spatial_error", "max_all_attention_error", "max_all_margin_error",
        "max_all_probability_error", "max_explicit_retain_error", "original_probability_cache_error",
    ):
        if verification[key] > 1e-5:
            raise RuntimeError(f"{key}超过数值容差: {verification[key]}")
    images = pd.DataFrame(image_rows)
    patients = aggregate_patients(images)
    curve = summarize_curve(images, patients)
    image_label_summary = summarize_image_labels(images)
    all_patients = patients[patients.level.eq("all")]
    all_flips = (
        all_patients.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
        != all_patients.retained_probability.ge(FROZEN_PATIENT_THRESHOLD)
    )
    numerical_boundary = (
        all_patients.original_probability.sub(FROZEN_PATIENT_THRESHOLD).abs()
        <= verification["max_all_probability_error"] + 1e-12
    )
    verification["all_patient_threshold_flip_count_raw"] = int(all_flips.sum())
    verification["all_patient_threshold_flips_within_noop_probability_error"] = int(
        (all_flips & numerical_boundary).sum()
    )
    verification["all_noop_passed_within_1e_5"] = bool(
        max(
            verification["max_all_spatial_error"],
            verification["max_all_attention_error"],
            verification["max_all_margin_error"],
            verification["max_all_probability_error"],
        ) <= 1e-5
        and (all_flips & ~numerical_boundary).sum() == 0
    )
    ranking_frame = pd.DataFrame(ranking_rows)
    verification["top20_inactive_count"] = int((~ranking_frame.feature_active).sum())
    ranking_frame.to_csv(args.output / "gradient_top20_by_image.csv", index=False)
    images.to_csv(args.output / "image_results.csv", index=False)
    patients.to_csv(args.output / "patient_results.csv", index=False)
    curve.to_csv(args.output / "recovery_curve.csv", index=False)
    image_label_summary.to_csv(args.output / "image_label_summary.csv", index=False)
    (args.output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "cohort": {"images": 497, "patients": 260},
        "curve": curve.to_dict(orient="records"),
        "image_label_summary": image_label_summary.to_dict(orient="records"),
        "interpretation_limits": definition["limits"],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
