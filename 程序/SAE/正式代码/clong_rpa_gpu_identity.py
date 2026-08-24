#!/usr/bin/env python3
"""RP-A residual-preserving no-op identity的纯函数核心。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from clong_s2b_core import attention_from_features, pooled_from_features


LEVELS = ("patch", "attention", "pooled", "logits", "probability")


@dataclass
class ErrorAccumulator:
    """累计单级绝对/相对误差及reference尺度。"""

    count: int = 0
    absolute_sum: float = 0.0
    max_absolute: float = 0.0
    max_relative: float = 0.0
    max_reference: float = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        """加入一个batch，要求两路shape一致且数值有限。"""
        if reference.shape != candidate.shape:
            raise ValueError("identity两路shape不一致")
        difference = (candidate - reference).abs().to(torch.float64)
        reference64 = reference.abs().to(torch.float64)
        if not torch.isfinite(difference).all() or not torch.isfinite(reference64).all():
            raise RuntimeError("identity输入或误差出现NaN/Inf")
        self.count += difference.numel()
        self.absolute_sum += float(difference.sum())
        self.max_absolute = max(self.max_absolute, float(difference.max()))
        relative = difference / reference64.clamp_min(1e-12)
        self.max_relative = max(self.max_relative, float(relative.max()))
        self.max_reference = max(self.max_reference, float(reference64.max()))

    def summary(self) -> dict[str, float]:
        """返回容差校准需要的冻结统计。"""
        if self.count == 0:
            raise RuntimeError("identity没有可汇总元素")
        return {
            "max_abs_error": self.max_absolute,
            "mean_abs_error": self.absolute_sum / self.count,
            "max_relative_error": self.max_relative,
            "max_abs_reference": self.max_reference,
        }


@torch.no_grad()
def residual_preserving_noop(
    features: torch.Tensor, sae: torch.nn.Module, k: int = 1024,
) -> torch.Tensor:
    """执行 ``D(h)+(F-D(h))``，其中干预后的 ``h'=h``。"""
    hidden = sae.encode(features, k=k)
    decoded = sae.decode(hidden)
    residual = features - decoded
    return sae.decode(hidden) + residual


@torch.no_grad()
def downstream_levels(features: torch.Tensor, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """从patch表示依次计算attention、pooled、logits和癌概率。"""
    attention = attention_from_features(features, model.attention_head)
    pooled = pooled_from_features(features, attention)
    logits = model.classifier(pooled)
    probability = torch.softmax(logits, dim=1)[:, 1]
    return {
        "patch": features,
        "attention": attention,
        "pooled": pooled,
        "logits": logits,
        "probability": probability,
    }


def derive_tolerances(summaries: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """按预注册公式从固定audit误差生成每级atol/rtol。"""
    epsilon = float(np.finfo(np.float32).eps)
    rtol = 32.0 * epsilon
    tolerances = {}
    for level in LEVELS:
        summary = summaries[level]
        atol = max(
            4.0 * float(summary["max_abs_error"]),
            rtol * max(1.0, float(summary["max_abs_reference"])),
        )
        tolerances[level] = {"atol": float(atol), "rtol": float(rtol)}
    return tolerances


def identity_passes(
    reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float,
) -> bool:
    """执行正式逐元素identity闸门，不使用额外默认容差。"""
    return bool(torch.all((candidate - reference).abs() <= atol + rtol * reference.abs()))
