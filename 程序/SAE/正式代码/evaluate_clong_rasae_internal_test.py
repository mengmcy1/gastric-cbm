#!/usr/bin/env python3
"""在冻结112张内部时间测试集上验证RA-SAE事后Feature分析。

固定原100轮RA-SAE、K=256、现有梯度排序和residual保留定义；
先与既有C-long内部时间测试输出逐值复现，再评价纯重构保真、
梯度排序与实际删除的一致性、Top-k保留恢复曲线。不训练、不调参、
不读取外部集，不评价医学语义是否成立。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
MAGE_CODE = ROOT / "程序/MAGE/正式代码"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MAGE_CODE))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rpc_core import intervention_block  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from clong_sae_discovery import (  # noqa: E402
    ActivationCapture,
    FROZEN_IMAGE_THRESHOLD,
    FROZEN_PATIENT_THRESHOLD,
)
from evaluate_clong_rasae_full_val_fidelity import (  # noqa: E402
    distribution,
    train_reference_means,
)
from evaluate_clong_rasae_topk_retain import (  # noqa: E402
    LEVELS,
    RETAIN_COUNTS,
    analytic_margin_gradient,
    deterministic_top_ids,
    retained_representation,
    summarize_curve,
    summarize_image_labels,
)
from evaluate_mage_external_projection import (  # noqa: E402
    MODEL_SPECS,
    ProjectionDataset,
    load_model_spec,
)
from evaluate_mage_internal_temporal_test import build_internal_frame  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
MAGE_RESULT = ROOT / "结果/MAGE/内部时间测试集一次性评估_20260819"
K = 256
FEATURE_COUNT = 2560
SIGN_TOLERANCE = 1e-6
REPRODUCTION_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    """解析正式评价参数。

    Args:
        None.

    Returns:
        argparse.Namespace: 输出目录、设备、批量、worker数和预检模式。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="冻结为原C-long内部测试正式推理批量32。",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="只核对冻结清单、计数与train/val重叠，不加载模型。",
    )
    return parser.parse_args()


def development_overlap_audit(test: pd.DataFrame) -> dict[str, int | bool]:
    """核对内部测试与train/val在患者、图像SHA和几何ID上无重叠。

    Args:
        test (pd.DataFrame): ``build_internal_frame``返回的112张冻结清单。

    Returns:
        dict[str, int | bool]: 队列计数和三类重叠数；任一重叠非零时快速失败。
    """
    development = pd.concat([
        pd.read_csv(CACHE / "train_metadata.csv", encoding="utf-8-sig", dtype={"patient_id": "string"}),
        pd.read_csv(CACHE / "val_metadata.csv", encoding="utf-8-sig", dtype={"patient_id": "string"}),
    ], ignore_index=True)
    development_patient_leaf = set(
        development.patient_id.astype(str).str.rsplit("/", n=1).str[-1]
    )
    test_patients = set(test.patient_id.astype(str))
    patient_overlap = test_patients & development_patient_leaf
    sha_overlap = set(test.keep_sha256.astype(str)) & set(development.sha256.astype(str))
    geometry_overlap = set(test.geometry_id.astype(str)) & set(development.geometry_id.astype(str))
    result: dict[str, int | bool] = {
        "test_images": int(len(test)),
        "test_patients": int(test.patient_id.nunique()),
        "development_images": int(len(development)),
        "development_patients": int(development.patient_id.nunique()),
        "patient_id_overlap": int(len(patient_overlap)),
        "keep_sha256_overlap": int(len(sha_overlap)),
        "geometry_id_overlap": int(len(geometry_overlap)),
        "medical_example_overlap": 0,
        "medical_example_overlap_basis": (
            "existing medical examples use train/val source_row; test is disjoint from full train/val"
        ),
        "passed": not patient_overlap and not sha_overlap and not geometry_overlap,
    }
    if not result["passed"]:
        raise RuntimeError(f"内部测试与SAE开发队列存在重叠: {result}")
    return result


@torch.no_grad()
def extract_clong_spatial(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按既有Keep预处理与C-long评估变换提取空间表示。

    Args:
        frame (pd.DataFrame): 含``image_path``的112张清单，顺序不变。
        model (torch.nn.Module): 冻结C-long attention-pooling模型。
        batch_size (int): 图像推理批量。
        num_workers (int): DataLoader进程数。
        device (torch.device): CPU或已核对的CUDA设备。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: CPU上的``[112,49,1280]``
        空间表示、``[112,49]``attention和``[112]``癌概率。
    """
    loader = DataLoader(
        ProjectionDataset(frame), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    capture = ActivationCapture(model.features[8])
    spatial_batches, attention_batches, probability_batches = [], [], []
    try:
        for batch_index, (images, _) in enumerate(loader, 1):
            logits, attention = model(images.to(device, non_blocking=device.type == "cuda"))
            fmap = capture.output
            if fmap is None or tuple(fmap.shape[1:]) != (1280, 7, 7):
                raise RuntimeError("C-long捕获表示不是[B,1280,7,7]")
            spatial = fmap.flatten(2).transpose(1, 2)
            pooled = (fmap * attention).sum((2, 3))
            if not torch.allclose(model.classifier(pooled), logits, atol=1e-5, rtol=1e-5):
                raise RuntimeError("空间表示重算logits与C-long forward不一致")
            spatial_batches.append(spatial.cpu())
            attention_batches.append(attention.flatten(1).cpu())
            probability_batches.append(logits.softmax(1)[:, 1].cpu())
            print(f"C-long特征提取: {batch_index}/{len(loader)}", flush=True)
    finally:
        capture.close()
    return (
        torch.cat(spatial_batches), torch.cat(attention_batches),
        torch.cat(probability_batches),
    )


def reproduce_existing_clong(
    frame: pd.DataFrame,
    attention: torch.Tensor,
    probability: torch.Tensor,
) -> dict[str, float | int | bool]:
    """逐图和逐患者复现既有112张C-long输出。

    Args:
        frame (pd.DataFrame): 当前冻结清单。
        attention (torch.Tensor): ``[112,49]``重算attention。
        probability (torch.Tensor): ``[112]``重算癌概率。

    Returns:
        dict[str, float | int | bool]: 逐图、attention及患者概率的最大误差。
    """
    existing = pd.read_csv(
        MAGE_RESULT / "internal_image_predictions.csv", encoding="utf-8-sig",
        dtype={"patient_id": "string"},
    )
    if len(existing) != 112 or existing.patient_id.nunique() != 78:
        raise RuntimeError("既有MAGE内部测试输出不是112图/78人")
    if existing.relative_path.tolist() != frame.relative_path.tolist():
        raise RuntimeError("当前冻结清单与既有MAGE输出行顺序不一致")
    attention_columns = [f"C-long_attention_{index}" for index in range(49)]
    probability_error = float(np.max(np.abs(
        probability.numpy().astype(np.float64)
        - existing["C-long"].to_numpy(dtype=np.float64)
    )))
    attention_error = float(np.max(np.abs(
        attention.numpy().astype(np.float64)
        - existing[attention_columns].to_numpy(dtype=np.float64)
    )))
    current_patient = pd.DataFrame({
        "patient_id": frame.patient_id.astype(str),
        "probability": probability.numpy().astype(np.float64),
    }).groupby("patient_id", as_index=False).probability.mean()
    existing_patient = pd.read_csv(
        MAGE_RESULT / "internal_patient_predictions.csv", encoding="utf-8-sig",
        dtype={"patient_id": "string"},
    )[["patient_id", "C-long_probability"]]
    paired = current_patient.merge(existing_patient, on="patient_id", validate="one_to_one")
    patient_error = float(np.max(np.abs(
        paired.probability.to_numpy() - paired["C-long_probability"].to_numpy()
    )))
    result: dict[str, float | int | bool] = {
        "images": int(len(frame)), "patients": int(frame.patient_id.nunique()),
        "max_image_probability_error": probability_error,
        "max_attention_error": attention_error,
        "max_patient_probability_error": patient_error,
        "tolerance": REPRODUCTION_TOLERANCE,
        "passed": max(probability_error, attention_error, patient_error) <= REPRODUCTION_TOLERANCE,
    }
    if not result["passed"]:
        raise RuntimeError(f"C-long既有输出复现超容差: {result}")
    return result


def patient_probabilities(images: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """将逐图概率按患者等权平均。

    Args:
        images (pd.DataFrame): 含patient_id、label和概率列的逐图表。
        value_column (str): 需聚合的概率列名。

    Returns:
        pd.DataFrame: 每患者一行的label、图数和平均概率。
    """
    return images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), images=(value_column, "size"),
        probability=(value_column, "mean"),
    )


@torch.no_grad()
def evaluate_reconstruction(
    frame: pd.DataFrame,
    spatial: torch.Tensor,
    sae: torch.nn.Module,
    head: torch.nn.Module,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """评价不保留原residual的纯RA-SAE重构保真。

    Args:
        frame (pd.DataFrame): 112张冻结清单。
        spatial (torch.Tensor): CPU上``[112,49,1280]``原表示。
        sae (torch.nn.Module): 冻结原100轮RA-SAE。
        head (torch.nn.Module): 冻结C-long注意力头。
        classifier_weight (torch.Tensor): ``[2,1280]``分类权重。
        classifier_bias (torch.Tensor): ``[2]``分类偏置。
        batch_size (int): 图像批量。
        device (torch.device): 计算设备。

    Returns:
        tuple: 逐图表、逐患者表和纯重构汇总字典。
    """
    patch_mean, pooled_mean = train_reference_means()
    rows = []
    patch_residual_ss = patch_reference_ss = 0.0
    pooled_residual_ss = pooled_reference_ss = 0.0
    patch_cosine_sum = pooled_cosine_sum = attention_cosine_sum = 0.0
    patch_vectors = 0
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        current = spatial[start:stop].to(device)
        hidden = sae.encode(current, K)
        rebuilt = sae.decode(hidden)
        original_attention = attention_from_features(current, head)
        rebuilt_attention = attention_from_features(rebuilt, head)
        original_pool = pooled_from_features(current, original_attention)
        rebuilt_pool = pooled_from_features(rebuilt, rebuilt_attention)
        original_logits = original_pool @ classifier_weight.T + classifier_bias
        rebuilt_logits = rebuilt_pool @ classifier_weight.T + classifier_bias
        original_probability = original_logits.softmax(1)[:, 1]
        rebuilt_probability = rebuilt_logits.softmax(1)[:, 1]
        original_margin = original_logits[:, 1] - original_logits[:, 0]
        rebuilt_margin = rebuilt_logits[:, 1] - rebuilt_logits[:, 0]
        current_np = current.cpu().numpy().astype(np.float64)
        rebuilt_np = rebuilt.cpu().numpy().astype(np.float64)
        original_pool_np = original_pool.cpu().numpy().astype(np.float64)
        rebuilt_pool_np = rebuilt_pool.cpu().numpy().astype(np.float64)
        patch_residual_ss += float(np.square(rebuilt_np - current_np).sum())
        patch_reference_ss += float(np.square(current_np - patch_mean).sum())
        pooled_residual_ss += float(np.square(rebuilt_pool_np - original_pool_np).sum())
        pooled_reference_ss += float(np.square(original_pool_np - pooled_mean).sum())
        patch_cosine_sum += float(F.cosine_similarity(current, rebuilt, dim=2).sum())
        pooled_cosine_sum += float(F.cosine_similarity(original_pool, rebuilt_pool, dim=1).sum())
        attention_cosine_sum += float(F.cosine_similarity(original_attention, rebuilt_attention, dim=1).sum())
        patch_vectors += current.shape[0] * current.shape[1]
        for local, record in enumerate(frame.iloc[start:stop].itertuples()):
            rows.append({
                "source_row": start + local, "image_index": int(record.image_index),
                "relative_path": record.relative_path, "patient_id": str(record.patient_id),
                "label": int(record.label),
                "original_probability": float(original_probability[local]),
                "reconstructed_probability": float(rebuilt_probability[local]),
                "delta_probability": float(rebuilt_probability[local] - original_probability[local]),
                "abs_delta_probability": float(abs(rebuilt_probability[local] - original_probability[local])),
                "original_margin": float(original_margin[local]),
                "reconstructed_margin": float(rebuilt_margin[local]),
                "delta_margin": float(rebuilt_margin[local] - original_margin[local]),
                "patch_mse": float((rebuilt[local] - current[local]).square().mean()),
                "pooled_cosine": float(F.cosine_similarity(original_pool[local], rebuilt_pool[local], dim=0)),
                "attention_cosine": float(F.cosine_similarity(original_attention[local], rebuilt_attention[local], dim=0)),
            })
    images = pd.DataFrame(rows)
    patients = images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), images=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        reconstructed_probability=("reconstructed_probability", "mean"),
    )
    patients["delta_probability"] = patients.reconstructed_probability - patients.original_probability
    patients["abs_delta_probability"] = patients.delta_probability.abs()
    original_image_prediction = images.original_probability.ge(FROZEN_IMAGE_THRESHOLD)
    rebuilt_image_prediction = images.reconstructed_probability.ge(FROZEN_IMAGE_THRESHOLD)
    original_patient_prediction = patients.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
    rebuilt_patient_prediction = patients.reconstructed_probability.ge(FROZEN_PATIENT_THRESHOLD)
    summary = {
        "prediction": {
            "image_probability_abs_change": distribution(images.abs_delta_probability.to_numpy()),
            "patient_probability_abs_change": distribution(patients.abs_delta_probability.to_numpy()),
            "image_prediction_agreement": float((original_image_prediction == rebuilt_image_prediction).mean()),
            "patient_prediction_agreement": float((original_patient_prediction == rebuilt_patient_prediction).mean()),
            "image_flip_count": int((original_image_prediction != rebuilt_image_prediction).sum()),
            "patient_flip_count": int((original_patient_prediction != rebuilt_patient_prediction).sum()),
            "original_image_auc": float(roc_auc_score(images.label, images.original_probability)),
            "reconstructed_image_auc": float(roc_auc_score(images.label, images.reconstructed_probability)),
            "original_patient_auc": float(roc_auc_score(patients.label, patients.original_probability)),
            "reconstructed_patient_auc": float(roc_auc_score(patients.label, patients.reconstructed_probability)),
        },
        "reconstruction": {
            "patch_mse": float(patch_residual_ss / (len(frame) * 49 * 1280)),
            "patch_fvu": float(patch_residual_ss / patch_reference_ss),
            "patch_1_minus_fvu": float(1 - patch_residual_ss / patch_reference_ss),
            "mean_patch_cosine": float(patch_cosine_sum / patch_vectors),
            "pooled_fvu": float(pooled_residual_ss / pooled_reference_ss),
            "pooled_1_minus_fvu": float(1 - pooled_residual_ss / pooled_reference_ss),
            "mean_pooled_cosine": float(pooled_cosine_sum / len(frame)),
            "mean_attention_cosine": float(attention_cosine_sum / len(frame)),
            "fvu_reference": "existing full-train arithmetic means",
        },
    }
    return images, patients, summary


def top_overlap(predicted: np.ndarray, exact: np.ndarray, count: int) -> float:
    """计算两个带符号Feature效应的绝对值Top-k集合重合率。

    Args:
        predicted (np.ndarray): ``[2560]``梯度预测删除效应。
        exact (np.ndarray): ``[2560]``实际删除效应。
        count (int): Top-k中k。

    Returns:
        float: 交集数除以k。
    """
    predicted_ids = deterministic_top_ids(predicted[None, :], count)[0]
    exact_ids = deterministic_top_ids(exact[None, :], count)[0]
    return float(len(set(predicted_ids) & set(exact_ids)) / count)


@torch.no_grad()
def evaluate_gradient_alignment(
    frame: pd.DataFrame,
    spatial: torch.Tensor,
    sae: torch.nn.Module,
    head: torch.nn.Module,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """对全2560项Feature比较梯度预测与residual保留实际删除。

    Args:
        frame (pd.DataFrame): 112张冻结清单。
        spatial (torch.Tensor): CPU上``[112,49,1280]``表示。
        sae (torch.nn.Module): 冻结RA-SAE。
        head (torch.nn.Module): 冻结注意力头。
        classifier_weight (torch.Tensor): ``[2,1280]``分类权重。
        classifier_bias (torch.Tensor): ``[2]``分类偏置。
        batch_size (int): 图像批量。
        device (torch.device): 计算设备。

    Returns:
        tuple: 逐图指标、逐患者等权指标和总体汇总。
    """
    decoder = sae.decoder_weight.detach()
    attention_weight = head.weight.detach().flatten()
    attention_bias = head.bias.detach().flatten()[0]
    margin_weight = (classifier_weight[1] - classifier_weight[0]).detach()
    margin_bias = (classifier_bias[1] - classifier_bias[0]).detach()
    rows = []
    total_eligible = total_direction_matches = 0
    all_abs_errors = []
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        current = spatial[start:stop].to(device)
        hidden = sae.encode(current, K)
        attention = attention_from_features(current, head)
        gradient = analytic_margin_gradient(
            current, attention, attention_weight, margin_weight
        )
        signed_score = (hidden * (gradient @ decoder.T)).sum(1)
        predicted_delta = -signed_score
        exact = intervention_block(
            current, hidden, decoder, attention_weight, attention_bias,
            margin_weight, margin_bias, alpha=0.0,
        )["delta_margin"]
        predicted_np = predicted_delta.cpu().numpy().astype(np.float64)
        exact_np = exact.cpu().numpy().astype(np.float64)
        for local, record in enumerate(frame.iloc[start:stop].itertuples()):
            predicted_row, exact_row = predicted_np[local], exact_np[local]
            eligible = np.abs(exact_row) > SIGN_TOLERANCE
            matches = np.sign(predicted_row[eligible]) == np.sign(exact_row[eligible])
            eligible_count = int(eligible.sum())
            match_count = int(matches.sum())
            total_eligible += eligible_count
            total_direction_matches += match_count
            absolute_error = np.abs(predicted_row - exact_row)
            all_abs_errors.append(absolute_error)
            rho = float(spearmanr(predicted_row, exact_row).statistic)
            rows.append({
                "source_row": start + local, "image_index": int(record.image_index),
                "relative_path": record.relative_path, "patient_id": str(record.patient_id),
                "label": int(record.label), "spearman_all_2560": rho,
                "top1_overlap": top_overlap(predicted_row, exact_row, 1),
                "top3_overlap": top_overlap(predicted_row, exact_row, 3),
                "top6_overlap": top_overlap(predicted_row, exact_row, 6),
                "top10_overlap": top_overlap(predicted_row, exact_row, 10),
                "eligible_direction_pairs": eligible_count,
                "direction_matches": match_count,
                "direction_agreement": (match_count / eligible_count if eligible_count else np.nan),
                "mean_abs_approximation_error": float(absolute_error.mean()),
                "p95_abs_approximation_error": float(np.quantile(absolute_error, 0.95)),
                "max_abs_approximation_error": float(absolute_error.max()),
            })
        print(f"全Feature实际删除: {stop}/{len(frame)}", flush=True)
    images = pd.DataFrame(rows)
    metric_columns = [
        "spearman_all_2560", "top1_overlap", "top3_overlap", "top6_overlap",
        "top10_overlap", "mean_abs_approximation_error",
        "p95_abs_approximation_error", "max_abs_approximation_error",
    ]
    patients = images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), images=("source_row", "size"),
        **{column: (column, "mean") for column in metric_columns},
        eligible_direction_pairs=("eligible_direction_pairs", "sum"),
        direction_matches=("direction_matches", "sum"),
    )
    patients["direction_agreement"] = (
        patients.direction_matches / patients.eligible_direction_pairs.replace(0, np.nan)
    )
    absolute_errors = np.concatenate(all_abs_errors)
    summary = {
        "aggregation": (
            "rank/overlap/error metrics: image then patient mean then equal patient mean; "
            "direction agreement: eligible pairs pooled within patient then equal patient mean"
        ),
        "patient_equal_mean_spearman_all_2560": float(patients.spearman_all_2560.mean()),
        "patient_equal_top_overlap": {
            f"top{count}": float(patients[f"top{count}_overlap"].mean())
            for count in (1, 3, 6, 10)
        },
        "sign_tolerance": SIGN_TOLERANCE,
        "eligible_direction_pairs": total_eligible,
        "near_zero_exact_pairs": int(len(frame) * FEATURE_COUNT - total_eligible),
        "pooled_direction_agreement": float(total_direction_matches / total_eligible),
        "patient_equal_direction_agreement": float(patients.direction_agreement.mean()),
        "absolute_approximation_error": distribution(absolute_errors),
    }
    return images, patients, summary


@torch.no_grad()
def evaluate_topk(
    frame: pd.DataFrame,
    spatial: torch.Tensor,
    sae: torch.nn.Module,
    head: torch.nn.Module,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """评价k=0/1/3/6/10/20/all的residual保留恢复曲线。

    Args:
        frame (pd.DataFrame): 112张冻结清单。
        spatial (torch.Tensor): CPU上``[112,49,1280]``表示。
        sae (torch.nn.Module): 冻结RA-SAE。
        head (torch.nn.Module): 冻结注意力头。
        classifier_weight (torch.Tensor): ``[2,1280]``分类权重。
        classifier_bias (torch.Tensor): ``[2]``分类偏置。
        batch_size (int): 图像批量。
        device (torch.device): 计算设备。

    Returns:
        tuple: 逐图、逐患者、恢复曲线、标签分层和数值验证。
    """
    decoder = sae.decoder_weight.detach()
    attention_weight = head.weight.detach().flatten()
    margin_weight = (classifier_weight[1] - classifier_weight[0]).detach()
    image_rows = []
    verification = {
        "max_all_spatial_error": 0.0, "max_all_attention_error": 0.0,
        "max_all_margin_error": 0.0, "max_all_probability_error": 0.0,
        "top20_inactive_count": 0, "test_read": True, "external_read": False,
    }
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        current = spatial[start:stop].to(device)
        hidden = sae.encode(current, K)
        original_attention = attention_from_features(current, head)
        original_pool = pooled_from_features(current, original_attention)
        original_logits = original_pool @ classifier_weight.T + classifier_bias
        original_margin = original_logits[:, 1] - original_logits[:, 0]
        original_probability = original_logits.softmax(1)[:, 1]
        gradient = analytic_margin_gradient(
            current, original_attention, attention_weight, margin_weight
        )
        signed_scores = (hidden * (gradient @ decoder.T)).sum(1)
        top20_np = deterministic_top_ids(signed_scores.cpu().numpy(), max(RETAIN_COUNTS))
        top20 = torch.as_tensor(top20_np, dtype=torch.long, device=device)
        full_component = hidden @ decoder
        residual = current - (full_component + sae.decoder_bias)
        base = residual + sae.decoder_bias
        representations = {"k0": base}
        for count in RETAIN_COUNTS:
            representations[f"k{count}"] = retained_representation(
                base, hidden, decoder, top20[:, :count]
            )
        representations["all"] = base + full_component
        outputs = {}
        for level, representation in representations.items():
            attention = attention_from_features(representation, head)
            pooled = pooled_from_features(representation, attention)
            logits = pooled @ classifier_weight.T + classifier_bias
            outputs[level] = {
                "attention": attention, "probability": logits.softmax(1)[:, 1],
                "margin": logits[:, 1] - logits[:, 0],
            }
        verification["max_all_spatial_error"] = max(
            verification["max_all_spatial_error"],
            float((representations["all"] - current).abs().max()),
        )
        verification["max_all_attention_error"] = max(
            verification["max_all_attention_error"],
            float((outputs["all"]["attention"] - original_attention).abs().max()),
        )
        verification["max_all_margin_error"] = max(
            verification["max_all_margin_error"],
            float((outputs["all"]["margin"] - original_margin).abs().max()),
        )
        verification["max_all_probability_error"] = max(
            verification["max_all_probability_error"],
            float((outputs["all"]["probability"] - original_probability).abs().max()),
        )
        active = hidden.gt(1e-8).any(1)
        verification["top20_inactive_count"] += int((~torch.gather(active, 1, top20)).sum())
        for local, record in enumerate(frame.iloc[start:stop].itertuples()):
            for level in LEVELS:
                probability = outputs[level]["probability"][local]
                margin = outputs[level]["margin"][local]
                image_rows.append({
                    "source_row": start + local, "image_index": int(record.image_index),
                    "relative_path": record.relative_path, "patient_id": str(record.patient_id),
                    "label": int(record.label), "level": level,
                    "original_probability": float(original_probability[local]),
                    "retained_probability": float(probability),
                    "delta_probability": float(probability - original_probability[local]),
                    "abs_delta_probability": float(abs(probability - original_probability[local])),
                    "original_margin": float(original_margin[local]),
                    "retained_margin": float(margin),
                    "delta_margin": float(margin - original_margin[local]),
                })
    for key in (
        "max_all_spatial_error", "max_all_attention_error",
        "max_all_margin_error", "max_all_probability_error",
    ):
        if verification[key] > 1e-5:
            raise RuntimeError(f"Top-k {key}超过数值容差: {verification[key]}")
    images = pd.DataFrame(image_rows)
    patients = images.groupby(["level", "patient_id"], as_index=False, sort=False).agg(
        label=("label", "first"), images=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        retained_probability=("retained_probability", "mean"),
    )
    patients["delta_probability"] = patients.retained_probability - patients.original_probability
    patients["abs_delta_probability"] = patients.delta_probability.abs()
    curve = summarize_curve(images, patients)
    image_labels = summarize_image_labels(images)
    return images, patients, curve, image_labels, verification


def write_report(output: Path, summary: dict, curve: pd.DataFrame) -> None:
    """写入不带医学语义外推的简短技术结果说明。

    Args:
        output (Path): 已创建的正式结果目录。
        summary (dict): 三项评价与核对汇总。
        curve (pd.DataFrame): Top-k恢复曲线。

    Returns:
        None: 写入``结果说明.md``。
    """
    fidelity = summary["pure_reconstruction"]
    ranking = summary["gradient_vs_exact_deletion"]
    selected = curve.set_index("level")
    lines = [
        "# RA-SAE内部时间测试事后Feature分析验证",
        "",
        "## 范围",
        "",
        "固定队列为112张/78人（癌52张/32人，非癌60张/46人）。"
        "使用原100轮RA-SAE、K=256；未训练、未调参、未读取外部集。",
        "",
        "该队列没有参与RA-SAE开发，但此前已用于C-long评价，因此不是整个项目"
        "从未查看过的全新盲测集。",
        "",
        "## 核对",
        "",
        f"- train/val患者、图像SHA和几何ID重叠均为0。",
        f"- C-long逐图概率最大复现误差："
        f"{summary['clong_reproduction']['max_image_probability_error']:.3g}。",
        f"- C-long attention最大复现误差："
        f"{summary['clong_reproduction']['max_attention_error']:.3g}。",
        "",
        "## 纯重构保真",
        "",
        f"- patch/pooled的1−FVU："
        f"{fidelity['reconstruction']['patch_1_minus_fvu']:.5f} / "
        f"{fidelity['reconstruction']['pooled_1_minus_fvu']:.5f}。",
        f"- 患者概率MAE："
        f"{fidelity['prediction']['patient_probability_abs_change']['mean']:.5f}；"
        f"预测一致率：{fidelity['prediction']['patient_prediction_agreement']:.2%}。",
        f"- 原模型/重构后患者AUC："
        f"{fidelity['prediction']['original_patient_auc']:.5f} / "
        f"{fidelity['prediction']['reconstructed_patient_auc']:.5f}。",
        f"- 患者概率绝对变化P95/最大值："
        f"{fidelity['prediction']['patient_probability_abs_change']['p95']:.5f} / "
        f"{fidelity['prediction']['patient_probability_abs_change']['max']:.5f}。",
        "",
        "## 梯度排序与实际删除",
        "",
        f"- 全2560项排序的患者等权Spearman："
        f"{ranking['patient_equal_mean_spearman_all_2560']:.5f}。",
        f"- Top-6患者等权重合率："
        f"{ranking['patient_equal_top_overlap']['top6']:.2%}。",
        f"- 排除|实际删除变化|≤{SIGN_TOLERANCE:g}后，"
        f"患者等权方向一致率："
        f"{ranking['patient_equal_direction_agreement']:.2%}。",
        "",
        "## Top-k保留",
        "",
        f"- k=0患者概率MAE/一致率："
        f"{selected.loc['k0', 'patient_probability_mae']:.5f} / "
        f"{selected.loc['k0', 'patient_prediction_agreement']:.2%}。",
        f"- Top-20患者概率MAE/一致率："
        f"{selected.loc['k20', 'patient_probability_mae']:.5f} / "
        f"{selected.loc['k20', 'patient_prediction_agreement']:.2%}。",
        f"- Top-20相对k=0概率误差恢复比例："
        f"{selected.loc['k20', 'patient_error_recovery_vs_k0']:.2%}；"
        f"恢复/新增偏离："
        f"{int(selected.loc['k20', 'patients_recovered_to_original_vs_k0'])}/"
        f"{int(selected.loc['k20', 'patients_newly_inconsistent_vs_k0'])}。",
        f"- Top-20患者概率绝对误差P95/最大值："
        f"{selected.loc['k20', 'patient_probability_abs_error_p95']:.5f} / "
        f"{selected.loc['k20', 'patient_probability_abs_error_max']:.5f}。",
        "",
        "## 与既有验证结果的描述性对照",
        "",
        "- 纯重构在测试集的patch 1−FVU略高（0.52294 vs 0.51800），"
        "pooled 1−FVU较低（0.46353 vs 0.49400）；患者概率MAE较高（"
        "0.09318 vs 0.06697），一致率较低（88.46% vs 91.54%）。"
        "这是数值保真指标表现不一致，不归结为单一的“改善”或“失败”。",
        "- 梯度与实际删除的Top-6重合在测试集为95.94%；既有96.09%"
        "来自val98张/64人，只能说两队列的描述值接近。",
        "- Top-20相对k0的患者概率误差恢复在测试集为33.19%，"
        "在完整val497张/260人为36.98%；测试集仍有15/78人与原预测不一致。"
        "不对这些队列差异作统计显著性解释。",
        "",
        "## 结论边界",
        "",
        "本结果只验证固定SAE分解、梯度排序、删除和Top-k保留定义在"
        "该内部时间测试集上的迁移表现。它不证明Feature医学语义稳定、Concept Family"
        "成立，也不是临床有效性验证。与完整val497/260的保真和Top-k结果、"
        "val98/64的96.09%排序重合分别描述，不将队列差异解释为统计显著变化。",
        "",
        "## 参数和字段怎样看",
        "",
        "- `K=256`：每个7×7位置编码时最多保留的Feature数；"
        "与Top-k实验中每张图保留的1/3/6/10/20项不是同一参数。",
        "- `width=2560`：字典中可用Feature的总数。`source_row`是本次冻结清单"
        "中从0开始的行号；`image_index`是原冻结清单中的图片代号。",
        "- `label`：数据标签，1为癌、0为非癌。`probability`是模型输出的0–1癌概率；"
        "`margin`是癌logit减非癌logit，越大表示模型越偏向癌。",
        "- 所有`delta`都是“干预后减原输出”。正的`delta_margin`表示干预后更偏癌，"
        "不表示被删除Feature支持癌；恰好相反，它在该干预下原本降低癌margin。",
        "- `patch_mse`是49个位置的表示均方误差。`1−FVU`用完整train均值作"
        "参照，越大表示数值变异复原得越多，不是医学信息保留比例。",
        "- `pooled_cosine`和`attention_cosine`分别比较汇聚表示与注意力方向；"
        "1表示方向完全相同，但不代表数值幅度完全一致。",
        "- `prediction_agreement`表示使用既有冻结阈值时，干预前后分类结果相同的比例；"
        "`flip_count`是跨过该阈值的数量。`AUC`只表示当前队列的排序区分能力。",
        "- `spearman_all_2560`是梯度预测删除效应与实际删除效应的排名相关，"
        "取值范围为−1到1。`top*_overlap`是两种排序的Top-k集合重合数除以k。",
        f"- `direction_agreement`只在|实际`delta_margin`|>{SIGN_TOLERANCE:g}的Feature上比较正负号；"
        "`eligible_direction_pairs`是实际参与比较的图像–Feature对数。",
        "- `mean/median/P95/max_abs_approximation_error`分别是梯度近似与实际删除"
        "margin变化绝对误差的均值、中位数、95分位和最大值。",
        "- Top-k的`k0`只保留当前SAE分解定义下的residual和decoder偏置；"
        "`all`加回全部编码分量，只用于数值复原检查。residual可能保留与Feature重复的信息。",
        "- `patient_probability_mae`是先对同患者图像概率取平均，再计算与原模型的"
        "平均绝对误差。`patient_error_recovery_vs_k0=(k0 MAE−当前MAE)/k0 MAE`；"
        "它不是医学信息解释比例。",
        "- `patients_recovered_to_original_vs_k0`是k0时偏离原预测、当前层级恢复的人数；"
        "`patients_newly_inconsistent_vs_k0`是k0时一致、当前层级新增偏离的人数；"
        "`patient_flip_reduction_vs_k0`是二者带来的净减少，不是只有恢复而无新偏离。",
    ]
    (output / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """执行预检、C-long复现和三项冻结SAE评价。

    Args:
        None: 所有参数由CLI提供。

    Returns:
        None: 成功后一次性写入正式结果目录。
    """
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {args.output}")
    frame = build_internal_frame()
    overlap = development_overlap_audit(frame)
    print(f"[预检] 冻结队列{len(frame)}张/{frame.patient_id.nunique()}人，重叠核对通过", flush=True)
    if args.preflight_only:
        print(json.dumps(overlap, ensure_ascii=False), flush=True)
        return
    if args.batch_size != 32:
        raise ValueError("正式内部测试必须沿用原C-long推理批量32")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定cuda但当前不可用")
    loaded = load_model_spec("C-long", MODEL_SPECS["C-long"], device)
    clong = loaded["model"]
    spatial, extracted_attention, extracted_probability = extract_clong_spatial(
        frame, clong, args.batch_size, args.num_workers, device
    )
    reproduction = reproduce_existing_clong(
        frame, extracted_attention, extracted_probability
    )
    print("[C-long] 112张既有输出复现通过", flush=True)
    del clong, loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sae, head, classifier_weight, classifier_bias = load_models(device)
    fidelity_images, fidelity_patients, fidelity = evaluate_reconstruction(
        frame, spatial, sae, head, classifier_weight, classifier_bias,
        args.batch_size, device,
    )
    print("[SAE] 纯重构保真完成", flush=True)
    ranking_images, ranking_patients, ranking = evaluate_gradient_alignment(
        frame, spatial, sae, head, classifier_weight, classifier_bias,
        args.batch_size, device,
    )
    print("[SAE] 梯度排序与实际删除完成", flush=True)
    topk_images, topk_patients, curve, image_labels, topk_verification = evaluate_topk(
        frame, spatial, sae, head, classifier_weight, classifier_bias,
        args.batch_size, device,
    )
    print("[SAE] Top-k保留恢复曲线完成", flush=True)
    summary = {
        "scope": "fixed original RA-SAE duration100, K=256, width=2560",
        "cohort": {
            "images": 112, "patients": 78, "cancer_images": 52,
            "cancer_patients": 32, "noncancer_images": 60,
            "noncancer_patients": 46,
        },
        "status": "SAE-development-independent confirmation; previously evaluated for C-long",
        "execution": {"batch_size": args.batch_size, "num_workers": args.num_workers},
        "overlap_audit": overlap,
        "clong_reproduction": reproduction,
        "pure_reconstruction": fidelity,
        "gradient_vs_exact_deletion": ranking,
        "topk_curve": curve.to_dict(orient="records"),
        "topk_verification": topk_verification,
        "validation_comparators": {
            "pure_reconstruction_and_topk": "full val 497 images / 260 patients",
            "gradient_top6_overlap_96.09_percent": "val98 images / 64 patients",
            "comparison_inference": "descriptive only; no significance claim",
        },
        "new_training": False, "test_read": True, "external_read": False,
        "no_test_based_adjustment": True,
    }
    definition = {
        "manifest": str(
            ROOT / "数据整理记录/图像裁剪/内部测试集省人民260612_Keep预处理_v1_20260819/"
            "06_内部测试集标注整合_v1_1/内部测试集最终评估清单.csv"
        ),
        "cohort": "112 images / 78 patients; cancer 52/32, noncancer 60/46",
        "preprocessing": "existing Keep images; C-long eval resize224 + ImageNet normalization",
        "inference_batch_size": args.batch_size,
        "model": "original fixed RA-SAE duration100, encoder K=256, width=2560",
        "reconstruction": "pure SAE decode without original residual",
        "gradient_ranking": "abs(sum_p h_pj * (full grad of cancer-minus-noncancer margin dot decoder_j))",
        "exact_deletion": "remove one Feature across all 49 positions and recompute attention/output",
        "topk": "residual + decoder bias + per-image gradient Top-k; k=0/1/3/6/10/20/all",
        "patient_probability": "arithmetic mean of image cancer probabilities",
        "thresholds": {"image": FROZEN_IMAGE_THRESHOLD, "patient": FROZEN_PATIENT_THRESHOLD},
        "fvu_reference": "existing full-train arithmetic means",
        "external_included": False,
        "medical_semantics_evaluated": False,
    }
    args.output.mkdir(parents=True)
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    fidelity_images.to_csv(args.output / "reconstruction_image_results.csv", index=False)
    fidelity_patients.to_csv(args.output / "reconstruction_patient_results.csv", index=False)
    ranking_images.to_csv(args.output / "gradient_alignment_image_results.csv", index=False)
    ranking_patients.to_csv(args.output / "gradient_alignment_patient_results.csv", index=False)
    topk_images.to_csv(args.output / "topk_image_results.csv", index=False)
    topk_patients.to_csv(args.output / "topk_patient_results.csv", index=False)
    curve.to_csv(args.output / "topk_recovery_curve.csv", index=False)
    image_labels.to_csv(args.output / "topk_image_label_summary.csv", index=False)
    write_report(args.output, summary, curve)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
