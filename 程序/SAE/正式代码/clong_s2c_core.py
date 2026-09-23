#!/usr/bin/env python3
"""C-long S2c Matryoshka patch SAE 核心结构。

协议冻结于 ``SAE实验进度与结果讨论.md`` S2c（2026-08-20冻结）：
同一个 ``1280 -> 10240 -> 1280`` 共享字典，在同一次正预激活降序排序上
构造 ``K={64,128,256,512,1024}`` 五层嵌套Top-K激活，保证小K激活集是
大K的真子集；激活为0的位置即使入选也不产生非零值，不做0填充。

本模块只放结构与纯函数，训练/校准/评价流程在
``clong_s2c_matryoshka.py``。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

K_LIST = (64, 128, 256, 512, 1024)


class MatryoshkaSparseAutoencoder(nn.Module):
    """单排序嵌套Top-K的共享字典SAE。

    参数:
        input_dim (int): 输入维数，本项目为1280。
        hidden_dim (int): 字典宽度，本项目为10240。
        feature_center (torch.Tensor): train患者/类别平衡特征中心，
            shape [input_dim]，作为decoder偏置。
        k_list (tuple[int, ...]): 递增嵌套粒度，必须都在(0, hidden_dim]内。

    状态:
        encoder/decoder_weight/decoder_bias：与S2b相同的参数化，
        decoder行向量始终单位归一化。

    forward输入为 ``[..., input_dim]``，返回 ``dict[K, (reconstructed, hidden)]``，
    其中hidden由同一次排序的递增前缀mask产生，满足嵌套关系。
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, feature_center: torch.Tensor,
        k_list: tuple[int, ...] = K_LIST,
    ) -> None:
        super().__init__()
        if tuple(k_list) != tuple(sorted(k_list)) or len(set(k_list)) != len(k_list):
            raise ValueError("k_list必须严格递增且无重复")
        if not all(0 < k <= hidden_dim for k in k_list):
            raise ValueError("每个K必须在(0, hidden_dim]内")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_list = tuple(int(k) for k in k_list)
        self.decoder_bias = nn.Parameter(feature_center.clone())
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.decoder_weight, a=0, nonlinearity="relu")
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder_weight)
            self.encoder.bias.zero_()

    def preactivation(self, features: torch.Tensor) -> torch.Tensor:
        """返回ReLU后的非负预激活，shape [..., hidden_dim]。"""
        return torch.relu(self.encoder(features - self.decoder_bias))

    def nested_hidden(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        """对单次降序排序构造全部嵌套Top-K激活。

        参数:
            features (torch.Tensor): 输入特征，shape [..., input_dim]。

        返回:
            dict[int, torch.Tensor]: K到稀疏激活的映射，shape与预激活相同。
            由同一排序的递增前缀产生，保证 ``TopK(i) subset TopK(j)``（i<j）；
            正预激活不足K的位置保留0值，真实L0可低于K。
        """
        dense = self.preactivation(features)
        order = dense.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            -1, order,
            torch.arange(dense.shape[-1], device=dense.device).expand_as(order),
        )
        return {k: dense * (ranks < k) for k in self.k_list}

    def encode(self, features: torch.Tensor, k: int | None = None):
        """返回嵌套激活；指定k时只返回该层，否则返回全部层的dict。"""
        hidden = self.nested_hidden(features)
        if k is None:
            return hidden
        if k not in self.k_list:
            raise ValueError(f"k={k}不在冻结K-list {self.k_list}内")
        return hidden[k]

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """将任一层的稀疏激活重构回输入空间，shape [..., input_dim]。"""
        return hidden @ self.decoder_weight + self.decoder_bias

    def forward(
        self, features: torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """返回 ``dict[K, (reconstructed, hidden)]``，用于五层联合损失。"""
        return {k: (self.decode(h), h) for k, h in self.nested_hidden(features).items()}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """把decoder每个Feature方向归一化为单位范数。"""
        self.decoder_weight.div_(
            self.decoder_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        )


class SingleKView:
    """把Matryoshka SAE的某一K层包装成S2b ``project_patch``兼容接口。

    参数:
        sae (MatryoshkaSparseAutoencoder): 已加载权重的嵌套SAE。
        k (int): 冻结K-list中的某一层。

    只暴露 ``hidden_dim``、``encode(x)`` 和 ``decode(h)``；
    ``encode_threshold`` 不可用，S2c没有BatchTopK阈值推理路径。
    """

    def __init__(self, sae: MatryoshkaSparseAutoencoder, k: int) -> None:
        if k not in sae.k_list:
            raise ValueError(f"k={k}不在冻结K-list {sae.k_list}内")
        self.sae = sae
        self.k = int(k)

    @property
    def hidden_dim(self) -> int:
        """返回字典宽度，供覆盖统计分配计数数组。"""
        return self.sae.hidden_dim

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """返回该K层的嵌套稀疏激活，shape [..., hidden_dim]。"""
        return self.sae.encode(features, self.k)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """重构该K层激活，shape [..., input_dim]。"""
        return self.sae.decode(hidden)


def union_nondead_mask(per_k_position_counts: dict[int, np.ndarray]) -> np.ndarray:
    """按S2c口径返回五层train激活并集的非死亡掩码。

    参数:
        per_k_position_counts (dict[int, np.ndarray]): 每个K在train上的
            逐Feature激活位置计数，shape [hidden_dim]。

    返回:
        np.ndarray: bool掩码，Feature在任一K层激活过即为True。
        嵌套结构下并集必须等于最大K层，不一致说明嵌套实现被破坏，直接报错。
    """
    layers = {int(k): np.asarray(v) > 0 for k, v in per_k_position_counts.items()}
    if not layers:
        raise ValueError("per_k_position_counts为空")
    union = np.logical_or.reduce(list(layers.values()))
    widest = layers[max(layers)]
    if not np.array_equal(union, widest):
        raise RuntimeError("五层激活并集不等于最大K层，嵌套关系被破坏")
    return union


def gamma_from_medians(median_patch: float, median_pool: float) -> float:
    """冻结 ``gamma_pool=0.25*median(joint_patch)/median(joint_pool)``。

    参数:
        median_patch (float): 74批joint_patch的中位数。
        median_pool (float): 74批joint_pool的中位数，小于1e-8时快速失败。

    返回:
        float: gamma_pool，使pool项初始量级约为patch项的25%。
    """
    if not np.isfinite(median_patch) or not np.isfinite(median_pool):
        raise RuntimeError("gamma_pool校准中位数含非有限值")
    if median_pool < 1e-8:
        raise RuntimeError("gamma_pool校准的median joint_pool<1e-8")
    return float(0.25 * median_patch / median_pool)


def fvu_from_sums(residual_sum_squares: float, reference_sum_squares: float) -> dict:
    """由平方和计算FVU与解释方差。

    参数:
        residual_sum_squares (float): 原始表示与重构表示之差的平方和。
        reference_sum_squares (float): 原始表示相对冻结train均值的平方和。

    返回:
        dict: ``fvu``与``explained_variance=1-fvu``。解释方差不裁剪，
        允许负值如实表示重构比train均值基线更差。
    """
    if not np.isfinite(residual_sum_squares) or not np.isfinite(reference_sum_squares):
        raise RuntimeError("FVU平方和包含非有限值")
    if residual_sum_squares < 0:
        raise RuntimeError("FVU残差平方和不能为负")
    if reference_sum_squares < 1e-12:
        raise RuntimeError("FVU参考平方和小于1e-12")
    fvu = float(residual_sum_squares / reference_sum_squares)
    return {"fvu": fvu, "explained_variance": float(1.0 - fvu)}


def select_min_passing_k(gates_by_k: dict[int, dict[str, bool]]) -> int | None:
    """返回同时通过八项门槛的最小K；全部失败时返回None。

    参数:
        gates_by_k (dict[int, dict[str, bool]]): 每个K的八项门槛判定。

    返回:
        int | None: 正式产品K；不做二次挑选。
    """
    passing = [k for k in sorted(gates_by_k) if all(gates_by_k[k].values())]
    return passing[0] if passing else None
