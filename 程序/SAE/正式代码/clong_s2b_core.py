#!/usr/bin/env python3
"""C-long S2b 的 SAE 核心模块。

本文件只放可复用的数学与模型逻辑：Top-K/BatchTopK、patch 完整替换、
BatchTopK 的 train-only 全局阈值求解，以及 AiB/nAiB/PGA 评价。文件读写和
S0 血缘校验由正式入口负责。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

ACTIVE_EPS = 1e-8


class StructuredSparseAutoencoder(nn.Module):
    """支持逐向量 Top-K 和批级 BatchTopK 的共享字典 SAE。

    输入是 ``[..., input_dim]``，输出与输入同形；最后一维之外的所有维度
    都可以视为基本向量。patch 模式因此自然对 49 个位置共享同一字典。
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, feature_center: torch.Tensor,
        activation_mode: str, target_k: int,
    ) -> None:
        super().__init__()
        if activation_mode not in {"topk", "batch_topk"}:
            raise ValueError(f"未知稀疏模式: {activation_mode}")
        if not 0 < target_k <= hidden_dim:
            raise ValueError("target_k必须在(0, hidden_dim]内")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_mode = activation_mode
        self.target_k = int(target_k)
        self.decoder_bias = nn.Parameter(feature_center.clone())
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.decoder_weight, a=0, nonlinearity="relu")
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder_weight)
            self.encoder.bias.zero_()

    def preactivation(self, features: torch.Tensor) -> torch.Tensor:
        """返回 ReLU 后、稀疏化前的非负激活。"""
        return torch.relu(self.encoder(features - self.decoder_bias))

    def sparsify(self, dense: torch.Tensor) -> torch.Tensor:
        """按冻结模式在最后一维或整个 batch 上保留最大激活。"""
        flat = dense.reshape(-1, dense.shape[-1])
        if self.activation_mode == "topk":
            values, indices = flat.topk(self.target_k, dim=1, sorted=False)
            sparse = torch.zeros_like(flat).scatter_(1, indices, values)
        else:
            keep = min(flat.numel(), flat.shape[0] * self.target_k)
            values, indices = flat.reshape(-1).topk(keep, sorted=False)
            sparse_flat = torch.zeros_like(flat.reshape(-1)).scatter_(0, indices, values)
            sparse = sparse_flat.reshape_as(flat)
        return sparse.reshape_as(dense)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.sparsify(self.preactivation(features))

    def encode_threshold(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        """BatchTopK 冻结后的单样本推理：仅保留 ``activation >= theta``。"""
        dense = self.preactivation(features)
        return dense * dense.ge(float(threshold))

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.decoder_weight + self.decoder_bias

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(features)
        return self.decode(hidden), hidden

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_weight.div_(
            self.decoder_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        )


def project_decoder_gradient(model: StructuredSparseAutoencoder) -> None:
    """移除 decoder 梯度在当前 decoder 方向上的分量。"""
    weight = model.decoder_weight
    gradient = weight.grad
    if gradient is None:
        raise RuntimeError("decoder_weight没有梯度")
    projection = (gradient * weight).sum(1, keepdim=True)
    gradient.sub_(
        projection / weight.square().sum(1, keepdim=True).clamp_min(1e-12) * weight
    )


def attention_from_features(
    spatial_features: torch.Tensor, attention_head: nn.Module,
) -> torch.Tensor:
    """从 ``[B,49,1280]`` 重算归一化的 ``[B,49]`` 注意力。"""
    if spatial_features.ndim != 3 or spatial_features.shape[1] != 49:
        raise ValueError("spatial_features必须是[B,49,C]")
    fmap = spatial_features.transpose(1, 2).reshape(
        spatial_features.shape[0], spatial_features.shape[2], 7, 7
    )
    logits = attention_head(fmap).flatten(1)
    return torch.softmax(logits, dim=1)


def pooled_from_features(
    spatial_features: torch.Tensor, attention: torch.Tensor,
) -> torch.Tensor:
    """按 ``[B,49]`` 注意力对 ``[B,49,C]`` 加权汇聚。"""
    return (spatial_features * attention.unsqueeze(-1)).sum(dim=1)


def patch_position_weights(attention: torch.Tensor) -> torch.Tensor:
    """返回均值严格为1的 ``0.5 + 0.5*49*a_p`` 位置权重。"""
    weights = 0.5 + 0.5 * attention.shape[1] * attention
    if not torch.allclose(weights.mean(1), torch.ones_like(weights[:, 0]), atol=1e-6):
        raise RuntimeError("位置权重均值不是1")
    return weights


def normalized_margin_mse(
    original: torch.Tensor, reconstructed: torch.Tensor,
    margin_vector: torch.Tensor, margin_std: float,
) -> torch.Tensor:
    """计算原始与重构表示的标准化分类 margin 差异。"""
    delta = (reconstructed - original) @ margin_vector
    return (delta / float(margin_std)).square().mean()


@dataclass(frozen=True)
class BatchTopKThreshold:
    threshold: float
    target_k: int
    vector_count: int
    target_total: int
    actual_total: int
    actual_mean_l0: float
    relative_deviation: float
    positive_count: int
    tie_count: int

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "target_k": self.target_k,
            "vector_count": self.vector_count,
            "target_total": self.target_total,
            "actual_total": self.actual_total,
            "actual_mean_l0": self.actual_mean_l0,
            "relative_deviation": self.relative_deviation,
            "positive_count": self.positive_count,
            "tie_count": self.tie_count,
        }


def solve_threshold_from_chunks(
    chunk_factory: Callable[[], Iterable[np.ndarray]], vector_count: int,
    target_k: int, max_relative_deviation: float = 0.01,
) -> BatchTopKThreshold:
    """分块 Top-K 归并求全train BatchTopK阈值。

    ``chunk_factory`` 每次调用必须按相同顺序产生一维非负 numpy 数组。
    第一遍每次只保留当前所有已见激活的全局最大 ``N*K`` 个候选；
    第二遍在冻结阈值上精确计数，记录 ``>=theta`` 造成的并列偏差。
    """
    target_total = int(vector_count) * int(target_k)
    positive_count = 0
    retained = np.empty(0, dtype=np.float32)
    for chunk in chunk_factory():
        values = np.asarray(chunk, dtype=np.float32).reshape(-1)
        positive = values[values > 0]
        positive_count += int(positive.size)
        if not positive.size:
            continue
        retained = np.concatenate((retained, positive))
        if retained.size > target_total:
            boundary = retained.size - target_total
            retained = np.partition(retained, boundary)[boundary:].copy()
    if positive_count < target_total:
        raise RuntimeError(
            f"正激活总数{positive_count}小于目标总激活{target_total}"
        )
    if retained.size != target_total:
        raise RuntimeError("分块Top-K归并后候选数不等于目标总数")
    threshold = float(retained.min())
    if threshold <= 0:
        raise RuntimeError("BatchTopK阈值theta<=0")

    actual_total = 0
    tie_count = 0
    for chunk in chunk_factory():
        values = np.asarray(chunk, dtype=np.float32).reshape(-1)
        actual_total += int(np.count_nonzero(values >= threshold))
        tie_count += int(np.count_nonzero(values == threshold))
    actual_mean = actual_total / vector_count
    deviation = abs(actual_mean - target_k) / target_k
    if deviation > max_relative_deviation:
        raise RuntimeError(
            f"BatchTopK并列导致mean L0={actual_mean:.4f}，"
            f"相对目标K偏差{deviation:.2%}>1%"
        )
    return BatchTopKThreshold(
        threshold=threshold, target_k=target_k, vector_count=vector_count,
        target_total=target_total, actual_total=actual_total,
        actual_mean_l0=float(actual_mean), relative_deviation=float(deviation),
        positive_count=positive_count, tie_count=tie_count,
    )


def array_sequence_sha(values: np.ndarray) -> str:
    """返回固定顺序数组的SHA-256。"""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cell_overlap_map(bbox: np.ndarray, grid_size: int = 7) -> np.ndarray:
    """返回归一化bbox对每个网格单元的面积覆盖比。"""
    overlap = np.zeros((grid_size, grid_size), dtype=np.float32)
    x1, y1, x2, y2 = map(float, bbox)
    cell = 1.0 / grid_size
    for row in range(grid_size):
        for column in range(grid_size):
            cx1, cx2 = column * cell, (column + 1) * cell
            cy1, cy2 = row * cell, (row + 1) * cell
            width = max(0.0, min(x2, cx2) - max(x1, cx1))
            height = max(0.0, min(y2, cy2) - max(y1, cy1))
            overlap[row, column] = width * height / (cell * cell)
    return overlap


def spatial_alignment_metrics(
    attention: np.ndarray, labels: np.ndarray, bboxes: np.ndarray,
) -> dict:
    """按 C-long 冻结口径计算癌图 AiB、nAiB 和 PGA。"""
    rows = []
    for index in np.flatnonzero(labels == 1):
        box = np.asarray(bboxes[index], dtype=float)
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        mass = np.asarray(attention[index], dtype=float).reshape(7, 7)
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        aib = float((mass * cell_overlap_map(box)).sum())
        peak_y, peak_x = divmod(int(mass.argmax()), 7)
        center_x, center_y = (peak_x + 0.5) / 7, (peak_y + 0.5) / 7
        rows.append((
            aib,
            (aib - area) / (1 - area) if area < 0.99 else np.nan,
            float(box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]),
        ))
    if not rows:
        return {"cancer_images": 0, "mean_aib": None,
                "mean_normalized_aib": None, "pga": None}
    values = np.asarray(rows, dtype=float)
    return {
        "cancer_images": int(len(values)),
        "mean_aib": float(np.mean(values[:, 0])),
        "mean_normalized_aib": float(np.nanmean(values[:, 1])),
        "pga": float(np.mean(values[:, 2])),
    }


def attention_drift_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """计算重构前后注意力的KL和cosine。"""
    p = np.clip(np.asarray(original, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(reconstructed, dtype=float), 1e-12, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    cosine = np.sum(p * q, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12
    )
    return {"mean_kl_original_to_reconstructed": float(kl.mean()),
            "mean_cosine": float(cosine.mean())}
