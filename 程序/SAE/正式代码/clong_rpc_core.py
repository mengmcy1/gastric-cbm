#!/usr/bin/env python3
"""RP-C1 residual-preserving干预、患者聚合和分轨的纯函数。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def remove_feature_group(
    spatial: torch.Tensor,
    hidden: torch.Tensor,
    decoder: torch.Tensor,
    members: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """在原始空间表示中同时删除一组SAE Feature分量。

    Args:
        spatial (torch.Tensor): 原始表示，shape为``[B,49,D]``。
        hidden (torch.Tensor): 同一原始表示的SAE激活，shape为``[B,49,H]``。
        decoder (torch.Tensor): SAE decoder方向，shape为``[H,D]``。
        members (list[int] | tuple[int, ...]): 需同时删除的Feature列号。

    Returns:
        torch.Tensor: ``spatial - sum_j(h_j*d_j)``，shape与``spatial``相同。
    """
    if spatial.ndim != 3 or hidden.ndim != 3 or spatial.shape[:2] != hidden.shape[:2]:
        raise ValueError("spatial/hidden必须是[B,49,D]和[B,49,H]")
    if decoder.ndim != 2 or hidden.shape[2] != decoder.shape[0] or spatial.shape[2] != decoder.shape[1]:
        raise ValueError("hidden/decoder/spatial维度不一致")
    ids = torch.as_tensor(members, dtype=torch.long, device=hidden.device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("members必须是非空一维Feature列表")
    if torch.unique(ids).numel() != ids.numel() or ids.min() < 0 or ids.max() >= hidden.shape[2]:
        raise ValueError("members含重复或越界Feature")
    component = hidden.index_select(2, ids) @ decoder.index_select(0, ids)
    return spatial - component


@torch.no_grad()
def intervention_block(
    spatial: torch.Tensor,
    hidden: torch.Tensor,
    decoder: torch.Tensor,
    attention_weight: torch.Tensor,
    attention_bias: torch.Tensor,
    margin_weight: torch.Tensor,
    margin_bias: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """对一批图像和Anchor精确计算residual-preserving干预结果。"""
    if spatial.ndim != 3 or hidden.ndim != 3 or spatial.shape[:2] != hidden.shape[:2]:
        raise ValueError("spatial/hidden必须是[B,49,D]和[B,49,A]")
    scale = float(alpha) - 1.0
    attention_logits = spatial @ attention_weight + attention_bias
    original_attention = torch.softmax(attention_logits, dim=1)
    patch_margin = spatial @ margin_weight
    original_margin = (original_attention * patch_margin).sum(1) + margin_bias
    attention_direction = decoder @ attention_weight
    margin_direction = decoder @ margin_weight
    changed_logits = attention_logits.unsqueeze(2) + scale * hidden * attention_direction
    changed_attention = torch.softmax(changed_logits, dim=1)
    changed_patch_margin = patch_margin.unsqueeze(2) + scale * hidden * margin_direction
    changed_margin = (changed_attention * changed_patch_margin).sum(1) + margin_bias
    original_probability = torch.sigmoid(original_margin)
    changed_probability = torch.sigmoid(changed_margin)
    reference = original_attention.unsqueeze(2)
    cosine = (changed_attention * reference).sum(1) / (
        changed_attention.square().sum(1).sqrt()
        * reference.square().sum(1).sqrt().clamp_min(1e-12)
    )
    return {
        "original_margin": original_margin,
        "original_probability": original_probability,
        "delta_margin": changed_margin - original_margin.unsqueeze(1),
        "delta_probability": changed_probability - original_probability.unsqueeze(1),
        "attention_cosine": cosine,
        "attention_l1": (changed_attention - reference).abs().sum(1),
        "peak_changed": changed_attention.argmax(1) != original_attention.argmax(1).unsqueeze(1),
        "active_image": hidden.gt(1e-8).any(1),
    }


def aggregate_image_matrix_by_patient(
    values: np.ndarray, image_patient_ids: np.ndarray, patient_ids: np.ndarray,
) -> np.ndarray:
    """先在每位患者内对图像等权聚合。"""
    if values.shape[0] != len(image_patient_ids):
        raise ValueError("图像矩阵与patient_id数量不一致")
    lookup = {str(patient): index for index, patient in enumerate(patient_ids)}
    output = np.zeros((len(patient_ids), values.shape[1]), dtype=np.float32)
    counts = np.zeros(len(patient_ids), dtype=np.int32)
    for row, patient in enumerate(image_patient_ids):
        index = lookup[str(patient)]
        output[index] += values[row]
        counts[index] += 1
    if np.any(counts == 0):
        raise RuntimeError("存在无图像患者")
    return output / counts[:, None]


def aggregate_active_by_patient(
    active: np.ndarray, image_patient_ids: np.ndarray, patient_ids: np.ndarray,
) -> np.ndarray:
    """患者任一图像任一位置激活即为active。"""
    lookup = {str(patient): index for index, patient in enumerate(patient_ids)}
    output = np.zeros((len(patient_ids), active.shape[1]), dtype=bool)
    for row, patient in enumerate(image_patient_ids):
        output[lookup[str(patient)]] |= active[row]
    return output


def label_seed_metrics(
    patient_delta_margin: np.ndarray,
    patient_delta_probability: np.ndarray,
    patient_attention_cosine: np.ndarray,
    patient_attention_l1: np.ndarray,
    patient_peak_changed: np.ndarray,
    active_patient: np.ndarray,
    labels: np.ndarray,
    original_patient_probability: np.ndarray,
    patient_threshold: float,
) -> list[dict]:
    """返回每个Anchor的癌/非癌、全体/active患者效应。"""
    rows = []
    ablated_probability = original_patient_probability[:, None] + patient_delta_probability
    original_prediction = original_patient_probability >= float(patient_threshold)
    for feature in range(patient_delta_margin.shape[1]):
        record: dict[str, float | int] = {"anchor_column": feature}
        for label, name in ((1, "cancer"), (0, "noncancer")):
            selected = labels == label
            active_selected = selected & active_patient[:, feature]
            values = patient_delta_margin[selected, feature]
            record[f"{name}_patient_count"] = int(selected.sum())
            record[f"{name}_active_patient_count"] = int(active_selected.sum())
            record[f"{name}_mean_delta_margin"] = float(values.mean())
            record[f"{name}_median_delta_margin"] = float(np.median(values))
            record[f"{name}_mean_abs_delta_margin"] = float(np.abs(values).mean())
            record[f"{name}_mean_delta_probability"] = float(
                patient_delta_probability[selected, feature].mean()
            )
            record[f"{name}_mean_attention_cosine"] = float(
                patient_attention_cosine[selected, feature].mean()
            )
            record[f"{name}_mean_attention_l1"] = float(
                patient_attention_l1[selected, feature].mean()
            )
            record[f"{name}_peak_changed_rate"] = float(
                patient_peak_changed[selected, feature].mean()
            )
            record[f"{name}_mean_label_support"] = float(
                (1 if label == 0 else -1) * values.mean()
            )
            if active_selected.any():
                active_values = patient_delta_margin[active_selected, feature]
                record[f"{name}_active_mean_delta_margin"] = float(active_values.mean())
                record[f"{name}_active_median_delta_margin"] = float(np.median(active_values))
                record[f"{name}_active_mean_abs_delta_margin"] = float(np.abs(active_values).mean())
            else:
                record[f"{name}_active_mean_delta_margin"] = 0.0
                record[f"{name}_active_median_delta_margin"] = 0.0
                record[f"{name}_active_mean_abs_delta_margin"] = 0.0
        delta = patient_delta_margin[:, feature]
        changed_prediction = ablated_probability[:, feature] >= float(patient_threshold)
        record["overall_mean_abs_delta_margin"] = float(np.abs(delta).mean())
        record["class_separation_delta_D"] = float(
            record["noncancer_mean_delta_margin"] - record["cancer_mean_delta_margin"]
        )
        record["patient_flip_rate"] = float((changed_prediction != original_prediction).mean())
        record["cancer_to_noncancer_flip_count"] = int(
            (labels.astype(bool) & original_prediction & ~changed_prediction).sum()
        )
        record["noncancer_to_cancer_flip_count"] = int(
            ((labels == 0) & ~original_prediction & changed_prediction).sum()
        )
        rows.append(record)
    return rows


def exact_top_ids(frame: pd.DataFrame, metric: str, count: int, ascending: bool) -> set[str]:
    """按指标与anchor_id确定性选取固定数量。"""
    ordered = frame.sort_values([metric, "anchor_id"], ascending=[ascending, True], kind="stable")
    return set(ordered.head(int(count)).anchor_id.astype(str))


def assign_rpc2_tracks(master: pd.DataFrame) -> pd.DataFrame:
    """执行结果前冻结的RP-C2五轨入选和20个低效对照。"""
    output = master.copy()
    n_top = int(math.ceil(len(output) * 0.05))
    overall = exact_top_ids(output, "overall_abs_effect", n_top, ascending=False)
    separation = exact_top_ids(output, "class_separation_effect", n_top, ascending=False)
    source_frame = output[output.source_risk.astype(bool)]
    source_count = int(math.ceil(len(source_frame) * 0.25))
    source = exact_top_ids(source_frame, "overall_abs_effect", source_count, ascending=False)
    sensitivity = set(output.loc[output.sharedness_source_sensitivity_changed, "anchor_id"].astype(str))
    cancer = set(output.loc[output.sharedness_v1.eq("cancer_enriched"), "anchor_id"].astype(str))
    low = exact_top_ids(output, "overall_abs_effect", 20, ascending=True)
    tracks = {
        "rpc2_track_overall": overall,
        "rpc2_track_separation": separation,
        "rpc2_track_source": source,
        "rpc2_track_sensitivity": sensitivity,
        "rpc2_track_cancer_enriched": cancer,
        "low_effect_control": low,
    }
    for column, ids in tracks.items():
        output[column] = output.anchor_id.isin(ids)
    output["rpc2_selected"] = output[
        [column for column in tracks if column.startswith("rpc2_track_")]
    ].any(axis=1)
    return output
