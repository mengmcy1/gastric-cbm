#!/usr/bin/env python3
"""分解Top-K SAE重构误差，并检验冻结分类margin方向对患者预测的影响。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from efficientnet_sae_discovery import (
    PROJECT_ROOT,
    load_model,
    lock_sensitivity_threshold,
    patient_mean,
    product_paths,
)


A1B_ROOT = PROJECT_ROOT / "结果/SAE/EfficientNet全局局部_0804/SAE-A1b_TopK比较"
RUNS = {
    "global": A1B_ROOT / "a1b_global_efficientnet_b0_seed42_h10240_topk1024",
    "local": A1B_ROOT / "a1b_local_efficientnet_b0_seed42_h10240_topk1024",
}
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """稳定计算二分类癌类别概率。"""
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def patient_metrics(
    metadata: pd.DataFrame,
    original_probability: np.ndarray,
    candidate_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """按患者均值聚合，返回AUC、阈值预测一致率和概率误差。"""
    frame = metadata[["patient_id", "label"]].copy()
    frame["original"] = original_probability
    frame["candidate"] = candidate_probability
    patients = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), original=("original", "mean"), candidate=("candidate", "mean"),
    )
    return {
        "patient_auc": float(roc_auc_score(patients.label, patients.candidate)),
        "patient_agreement": float(np.mean(
            patients.original.ge(threshold) == patients.candidate.ge(threshold)
        )),
        "patient_probability_mae": float(np.mean(np.abs(patients.original - patients.candidate))),
    }


def analyze_branch(branch: str, run_dir: Path) -> tuple[dict, list[dict]]:
    """读取一个分支的冻结val特征，沿分类权重方向逐级校正SAE误差。"""
    cache = run_dir / "特征缓存"
    original = np.load(cache / "val_gap_features.npy").astype(np.float64)
    reconstructed = np.load(cache / "val_sae_projection.npz")["reconstructed"].astype(np.float64)
    metadata = pd.read_csv(
        cache / "val_metadata.csv", encoding="utf-8-sig", dtype={"patient_id": str},
    )
    model = load_model(branch, product_paths(branch, 42), torch.device("cpu"))
    classifier = model.classifier[1]
    weight = (classifier.weight[1] - classifier.weight[0]).detach().numpy().astype(np.float64)
    bias = float((classifier.bias[1] - classifier.bias[0]).detach())

    original_margin = original @ weight + bias
    reconstructed_margin = reconstructed @ weight + bias
    original_probability = sigmoid(original_margin)
    recorded_probability = metadata.cancer_probability.to_numpy(float)
    probability_error = float(np.max(np.abs(original_probability - recorded_probability)))
    if probability_error > 1e-5:
        raise RuntimeError(f"{branch}冻结头概率复算不一致: {probability_error:.3e}")

    # 阈值必须沿用正式特征提取时落盘的概率；边界样本会放大1e-7量级的复算差异。
    patient_original = patient_mean(metadata, "cancer_probability")
    threshold = lock_sensitivity_threshold(
        patient_original.label.to_numpy(), patient_original.cancer_probability.to_numpy(),
    )

    error = original - reconstructed
    weight_norm_sq = float(weight @ weight)
    margin_error = original_margin - reconstructed_margin
    projection = (margin_error / weight_norm_sq)[:, None] * weight[None, :]
    total_error_energy = float(np.square(error).sum())
    projection_energy = float(np.square(projection).sum())
    summary = {
        "branch": branch,
        "n_images": int(len(metadata)),
        "n_patients": int(metadata.patient_id.nunique()),
        "locked_patient_threshold": threshold,
        "margin_mae": float(np.mean(np.abs(margin_error))),
        "margin_rmse": float(np.sqrt(np.mean(np.square(margin_error)))),
        "margin_correlation": float(np.corrcoef(original_margin, reconstructed_margin)[0, 1]),
        "classification_direction_error_energy_fraction": projection_energy / total_error_energy,
        "classification_direction_correction_relative_to_feature_norm": float(
            np.linalg.norm(projection) / np.linalg.norm(original)
        ),
        "frozen_probability_recompute_max_abs_error": probability_error,
    }

    rows: list[dict] = []
    for alpha in ALPHAS:
        corrected = reconstructed + alpha * projection
        probability = sigmoid(corrected @ weight + bias)
        row = {"branch": branch, "correction_fraction": alpha}
        row.update(patient_metrics(metadata, original_probability, probability, threshold))
        row.update({
            "image_auc": float(roc_auc_score(metadata.label, probability)),
            "image_probability_mae": float(np.mean(np.abs(original_probability - probability))),
            "feature_cosine": float(np.mean(
                np.sum(original * corrected, axis=1)
                / np.maximum(np.linalg.norm(original, axis=1) * np.linalg.norm(corrected, axis=1), 1e-12)
            )),
        })
        rows.append(row)
    return summary, rows


def main() -> None:
    """分析两个正式分支并写出机器可读CSV/JSON，不读取test或external。"""
    for run_dir in RUNS.values():
        if not (run_dir / "metrics.json").is_file():
            raise FileNotFoundError(f"缺少正式A1b结果: {run_dir}")
    summary_path = A1B_ROOT / "margin_error_attribution.json"
    curve_path = A1B_ROOT / "margin_correction_curve.csv"
    if summary_path.exists() or curve_path.exists():
        raise FileExistsError("margin误差归因结果已存在，拒绝覆盖")

    summaries, rows = [], []
    for branch, run_dir in RUNS.items():
        summary, current_rows = analyze_branch(branch, run_dir)
        summaries.append(summary)
        rows.extend(current_rows)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(curve_path, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\n分类方向渐进校正:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n输出: {summary_path}\n      {curve_path}")


if __name__ == "__main__":
    main()
