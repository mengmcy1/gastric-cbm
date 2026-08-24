#!/usr/bin/env python3
"""RP-A spatial指标的候选数值核心。

输入是同一批图像上的两组 ``[image, 49, feature]`` 非负激活。
模块只实现图像、患者和pair三级语义，不执行Feature matching。
"""

from __future__ import annotations

import numpy as np
import torch

from clong_s2b_core import ACTIVE_EPS


def image_active(activations: torch.Tensor) -> torch.Tensor:
    """按冻结presence语义返回 ``[image, feature]`` active标记。"""
    if activations.ndim != 3 or activations.shape[1] != 49:
        raise ValueError("activations必须为[image,49,feature]")
    return activations.amax(dim=1) > ACTIVE_EPS


@torch.no_grad()
def spatial_pair_matrix(
    source: torch.Tensor,
    target: torch.Tensor,
    image_patient_index: torch.Tensor,
    patient_count: int,
    accumulation_dtype: torch.dtype,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """计算全部source-target pair的患者平衡spatial。

    49维cosine始终按输入float32计算；``accumulation_dtype``只控制
    图像到患者、患者到pair的求和精度。返回pair矩阵和诊断计数。
    """
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source/target必须为三维")
    if source.shape[:2] != target.shape[:2] or source.shape[1] != 49:
        raise ValueError("source/target图像和位置维必须一致")
    if source.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("图像级cosine输入必须为float32")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("累积dtype只能是float32或float64")
    if image_patient_index.shape != (source.shape[0],):
        raise ValueError("image_patient_index shape不一致")
    if patient_count <= 0:
        raise ValueError("patient_count必须为正")

    source_active = image_active(source)
    target_active = image_active(target)
    source_maps = source.permute(0, 2, 1)
    dot = torch.bmm(source_maps, target)
    source_norm = torch.linalg.vector_norm(source_maps, dim=2)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    denominator = source_norm.unsqueeze(2) * target_norm.unsqueeze(1)
    both = source_active.unsqueeze(2) & target_active.unsqueeze(1)
    union = source_active.unsqueeze(2) | target_active.unsqueeze(1)
    cosine = dot / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    cosine.masked_fill_(~both, 0.0)
    if not torch.isfinite(cosine[union]).all():
        raise RuntimeError("有效图像spatial出现NaN/Inf")
    if ((cosine[union] < 0.0) | (cosine[union] > 1.0)).any():
        raise RuntimeError("有效图像spatial超出[0,1]，禁止裁剪")

    shape = (patient_count, source.shape[2], target.shape[2])
    patient_sum = torch.zeros(shape, dtype=accumulation_dtype, device=source.device)
    patient_images = torch.zeros(shape, dtype=accumulation_dtype, device=source.device)
    patient_one_side = torch.zeros(shape, dtype=accumulation_dtype, device=source.device)
    patient_sum.index_add_(0, image_patient_index, cosine.to(accumulation_dtype))
    patient_images.index_add_(0, image_patient_index, union.to(accumulation_dtype))
    patient_one_side.index_add_(
        0, image_patient_index, (union & ~both).to(accumulation_dtype)
    )
    patient_valid = patient_images > 0
    patient_value = patient_sum / patient_images.clamp_min(1)
    valid_count = patient_valid.sum(dim=0)
    pair_sum = (patient_value * patient_valid).sum(dim=0, dtype=accumulation_dtype)
    pair = pair_sum / valid_count.clamp_min(1).to(accumulation_dtype)
    pair[valid_count == 0] = torch.nan

    union_images = patient_images.sum(dim=0, dtype=torch.float64)
    one_side_images = patient_one_side.sum(dim=0, dtype=torch.float64)
    both_images = union_images - one_side_images
    fraction_one_side = one_side_images / union_images.clamp_min(1)
    fraction_both = both_images / union_images.clamp_min(1)
    diagnostics = {
        "n_evaluable_patient_instances": valid_count.cpu().numpy().astype(np.int64),
        "n_union_active_image_instances": union_images.cpu().numpy(),
        "fraction_one_side_zero": fraction_one_side.cpu().numpy(),
        "fraction_both_active": fraction_both.cpu().numpy(),
    }
    return pair.cpu().numpy().astype(np.float64), diagnostics


def apply_minimum_support(
    spatial: np.ndarray, valid_count: np.ndarray, minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按split冻结下限标记可评价pair；下限不足返回NA而不是0。"""
    values = np.asarray(spatial, dtype=np.float64)
    counts = np.asarray(valid_count, dtype=np.int64)
    if values.shape != counts.shape or values.ndim != 2:
        raise ValueError("spatial和valid_count必须是同shape二维矩阵")
    if minimum <= 0:
        raise ValueError("minimum必须为正")
    evaluable = counts >= minimum
    if not np.isfinite(values[evaluable]).all():
        raise ValueError("可评价spatial不得包含NaN/Inf")
    output = values.copy()
    output[~evaluable] = np.nan
    return output, evaluable
