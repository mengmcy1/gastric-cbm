#!/usr/bin/env python3
"""在完整验证集评价固定RA-SAE 100轮、K=256的纯重构保真度。

原始C-long``[B,49,1280]``表示经RA-SAE编码和解码后，使用重构表示重新计算
冻结注意力、汇聚和分类头。本评价不保留原始residual，因此回答SAE纯重构
本身保留了多少原模型预测与表示。
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
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_clong_rasae_medical_feedback import load_models  # noqa: E402
from clong_rasae_core import ArchetypalMatryoshkaSAE  # noqa: E402
from clong_s2b_core import attention_from_features, pooled_from_features  # noqa: E402
from clong_sae_discovery import FROZEN_IMAGE_THRESHOLD, FROZEN_PATIENT_THRESHOLD  # noqa: E402
from run_clong_rasae_pilot import read_subset  # noqa: E402


BASE = ROOT / "结果/SAE/RA_SAE_Pilot_20260908"
CACHE = ROOT / "结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache"
K = 256


def train_reference_means() -> tuple[np.ndarray, np.ndarray]:
    """计算FVU使用的完整train算术平均参考中心。

    Args:
        None.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``[1280]`` patch均值和pooled均值，均只由train得到。
    """
    spatial = np.load(CACHE / "train_spatial_features.npy", mmap_mode="r")
    pooled = np.load(CACHE / "train_pooled_features.npy", mmap_mode="r")
    patch_mean = np.asarray(spatial).sum(axis=(0, 1), dtype=np.float64) / (spatial.shape[0] * spatial.shape[1])
    pooled_mean = np.asarray(pooled).sum(axis=0, dtype=np.float64) / pooled.shape[0]
    return patch_mean, pooled_mean


def aggregate_patients(images: pd.DataFrame) -> pd.DataFrame:
    """沿用现有口径，对同一患者的图像癌概率等权平均。

    Args:
        images (pd.DataFrame): 逐图原始与重构概率表。

    Returns:
        pd.DataFrame: 每患者一行的原始/重构概率和差值。
    """
    patients = images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), images=("source_row", "size"),
        original_probability=("original_probability", "mean"),
        reconstructed_probability=("reconstructed_probability", "mean"),
    )
    patients["delta_probability"] = (
        patients.reconstructed_probability - patients.original_probability
    )
    patients["abs_delta_probability"] = patients.delta_probability.abs()
    return patients


def distribution(values: np.ndarray) -> dict[str, float]:
    """返回一维绝对变化的均值、中位数、P95和最大值。

    Args:
        values (np.ndarray): 有限一维数组。

    Returns:
        dict[str, float]: 四个描述统计。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("分布统计要求有限一维数组")
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def main() -> None:
    """重构完整val497并写入逐图、逐患者和总体保真结果。

    Args:
        None: 设备、批量和输出目录由CLI提供。

    Returns:
        None: 输出目录已存在时快速失败。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sae-checkpoint", type=Path,
                        help="可选的RA-SAE checkpoint；省略时使用当前duration100权重")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    definition = {
        "model": "fixed_RA_duration100_K256_width2560",
        "sae_checkpoint": str(args.sae_checkpoint) if args.sae_checkpoint else "current_duration100_ra_final",
        "cohort": "full available val 497 images / 260 patients",
        "reconstruction": "pure SAE decode(encode_K256(original_spatial)); original residual not retained",
        "downstream": "recompute frozen attention, pooling and original classifier",
        "patient_probability": "mean image cancer probability per patient",
        "thresholds": {"image": FROZEN_IMAGE_THRESHOLD, "patient": FROZEN_PATIENT_THRESHOLD},
        "fvu_reference": "full-train arithmetic mean for patch and pooled representations",
        "new_training": False, "test_read": False, "external_read": False,
    }
    (args.output / "analysis_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = pd.read_csv(CACHE / "val_metadata.csv").reset_index().rename(columns={"index": "source_row"})
    if len(frame) != 497 or frame.patient_id.nunique() != 260:
        raise RuntimeError("val队列不是497图/260人")
    patch_mean, pooled_mean = train_reference_means()
    device = torch.device(args.device)
    sae, head, classifier_weight, classifier_bias = load_models(device)
    if args.sae_checkpoint is not None:
        candidate_checkpoint = torch.load(
            args.sae_checkpoint, map_location=device, weights_only=False
        )
        candidate_state = candidate_checkpoint["state_dict"]
        candidate_config = candidate_checkpoint["config"]
        if candidate_config["hidden_dim"] != 2560 or candidate_config["k_list"] != [64, 128, 256]:
            raise RuntimeError("候选checkpoint不是固定2560宽度/K列表")
        sae = ArchetypalMatryoshkaSAE(
            candidate_state["points"], candidate_state["decoder_bias"],
            candidate_config["hidden_dim"], tuple(candidate_config["k_list"]),
            candidate_config["delta"], True, candidate_config["seed"],
            candidate_config["initialization"],
        ).to(device)
        sae.load_state_dict(candidate_state)
        sae.requires_grad_(False).eval()
    image_rows = []
    patch_residual_ss = patch_reference_ss = 0.0
    pooled_residual_ss = pooled_reference_ss = 0.0
    patch_cosine_sum = pooled_cosine_sum = attention_cosine_sum = 0.0
    patch_vectors = 0
    max_attention_cache_error = max_pool_cache_error = max_probability_cache_error = 0.0
    with torch.no_grad():
        for start in range(0, len(frame), args.batch_size):
            stop = min(start + args.batch_size, len(frame))
            batch_frame = frame.iloc[start:stop]
            data = read_subset(batch_frame, "val", device)
            spatial = data["spatial"]
            original_attention = attention_from_features(spatial, head)
            original_pool = pooled_from_features(spatial, original_attention)
            hidden = sae.encode(spatial, K)
            reconstructed = sae.decode(hidden)
            rebuilt_attention = attention_from_features(reconstructed, head)
            rebuilt_pool = pooled_from_features(reconstructed, rebuilt_attention)
            original_logits = original_pool @ classifier_weight.T + classifier_bias
            rebuilt_logits = rebuilt_pool @ classifier_weight.T + classifier_bias
            original_probability = original_logits.softmax(1)[:, 1]
            rebuilt_probability = rebuilt_logits.softmax(1)[:, 1]
            original_margin = original_logits[:, 1] - original_logits[:, 0]
            rebuilt_margin = rebuilt_logits[:, 1] - rebuilt_logits[:, 0]
            max_attention_cache_error = max(
                max_attention_cache_error, float((original_attention - data["attention"]).abs().max())
            )
            max_pool_cache_error = max(
                max_pool_cache_error, float((original_pool - data["pooled"]).abs().max())
            )
            expected_probability = torch.as_tensor(
                batch_frame.cancer_probability.to_numpy(dtype=np.float32, copy=True), device=device
            )
            max_probability_cache_error = max(
                max_probability_cache_error, float((original_probability - expected_probability).abs().max())
            )
            original_np = spatial.cpu().numpy().astype(np.float64)
            reconstructed_np = reconstructed.cpu().numpy().astype(np.float64)
            original_pool_np = original_pool.cpu().numpy().astype(np.float64)
            rebuilt_pool_np = rebuilt_pool.cpu().numpy().astype(np.float64)
            patch_residual_ss += float(np.square(reconstructed_np - original_np).sum())
            patch_reference_ss += float(np.square(original_np - patch_mean).sum())
            pooled_residual_ss += float(np.square(rebuilt_pool_np - original_pool_np).sum())
            pooled_reference_ss += float(np.square(original_pool_np - pooled_mean).sum())
            patch_cosine_sum += float(F.cosine_similarity(spatial, reconstructed, dim=2).sum())
            pooled_cosine_sum += float(F.cosine_similarity(original_pool, rebuilt_pool, dim=1).sum())
            attention_cosine_sum += float(F.cosine_similarity(original_attention, rebuilt_attention, dim=1).sum())
            patch_vectors += spatial.shape[0] * spatial.shape[1]
            patch_mse = (reconstructed - spatial).square().mean((1, 2)).cpu().numpy()
            pool_cosine = F.cosine_similarity(original_pool, rebuilt_pool, dim=1).cpu().numpy()
            attention_cosine = F.cosine_similarity(original_attention, rebuilt_attention, dim=1).cpu().numpy()
            for local, record in enumerate(batch_frame.itertuples()):
                image_rows.append({
                    "source_row": int(record.source_row), "patient_id": str(record.patient_id),
                    "label": int(record.label),
                    "original_probability": float(original_probability[local]),
                    "reconstructed_probability": float(rebuilt_probability[local]),
                    "delta_probability": float(rebuilt_probability[local] - original_probability[local]),
                    "abs_delta_probability": float(abs(rebuilt_probability[local] - original_probability[local])),
                    "original_margin": float(original_margin[local]),
                    "reconstructed_margin": float(rebuilt_margin[local]),
                    "delta_margin": float(rebuilt_margin[local] - original_margin[local]),
                    "patch_mse": float(patch_mse[local]),
                    "pooled_cosine": float(pool_cosine[local]),
                    "attention_cosine": float(attention_cosine[local]),
                })
            print(f"val: {stop}/{len(frame)}", flush=True)
    if max(max_attention_cache_error, max_pool_cache_error, max_probability_cache_error) > 1e-4:
        raise RuntimeError("原始下游复算与冻结缓存不一致")
    images = pd.DataFrame(image_rows)
    patients = aggregate_patients(images)
    original_image_prediction = images.original_probability.ge(FROZEN_IMAGE_THRESHOLD)
    rebuilt_image_prediction = images.reconstructed_probability.ge(FROZEN_IMAGE_THRESHOLD)
    original_patient_prediction = patients.original_probability.ge(FROZEN_PATIENT_THRESHOLD)
    rebuilt_patient_prediction = patients.reconstructed_probability.ge(FROZEN_PATIENT_THRESHOLD)
    patch_elements = len(frame) * 49 * 1280
    summary = {
        "cohort": {"images": len(images), "patients": len(patients),
                   "cancer_patients": int(patients.label.sum()),
                   "noncancer_patients": int((patients.label == 0).sum())},
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
            "image_margin_mae": float(images.delta_margin.abs().mean()),
            "patient_probability_pearson": float(np.corrcoef(
                patients.original_probability, patients.reconstructed_probability
            )[0, 1]),
        },
        "reconstruction": {
            "patch_mse": float(patch_residual_ss / patch_elements),
            "patch_rmse": float(np.sqrt(patch_residual_ss / patch_elements)),
            "patch_fvu": float(patch_residual_ss / patch_reference_ss),
            "patch_explained_variance": float(1 - patch_residual_ss / patch_reference_ss),
            "mean_patch_cosine": float(patch_cosine_sum / patch_vectors),
            "pooled_fvu": float(pooled_residual_ss / pooled_reference_ss),
            "pooled_explained_variance": float(1 - pooled_residual_ss / pooled_reference_ss),
            "mean_pooled_cosine": float(pooled_cosine_sum / len(frame)),
            "mean_attention_cosine": float(attention_cosine_sum / len(frame)),
        },
        "verification": {
            "max_attention_cache_error": max_attention_cache_error,
            "max_pool_cache_error": max_pool_cache_error,
            "max_probability_cache_error": max_probability_cache_error,
            "train_reference_only_for_fvu": True,
            "test_read": False, "external_read": False,
        },
    }
    images.to_csv(args.output / "image_fidelity.csv", index=False)
    patients.to_csv(args.output / "patient_fidelity.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
