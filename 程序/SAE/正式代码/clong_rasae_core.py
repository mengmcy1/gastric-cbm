#!/usr/bin/env python3
"""C-long局部RA-SAE双臂原型；尚无正式训练协议或默认实验参数。

复用S2c嵌套Top-K，仅比较自由字典与D=softmax(L)@C+R。
C必须由train在同一个固定中心化坐标系中构建；本模块不读取数据。
两臂从相同字典及encoder起步，都不施加字典单位范数化或可学习整体倍数。
RA臂使用softmax确保凸组合，区别于Overcomplete的ReLU后行归一化；
R按行投影到半径delta的L2球。此为数学适配原型，不是官方逐参复现。
依据：https://github.com/KempnerInstitute/overcomplete/blob/main/
overcomplete/sae/archetypal_dictionary.py
"""

import math

import torch
from torch import nn

from clong_s2c_core import MatryoshkaSparseAutoencoder
from clong_s2b_core import patch_position_weights


class ArchetypalMatryoshkaSAE(MatryoshkaSparseAutoencoder):
    """共享初始化的局部SAE，供自由/受约束字典配对比较。

    参数：points Tensor[M,D]，train代表点减去固定feature_center后的坐标；
    feature_center Tensor[D]，两臂共享且冻结的train中心；hidden_dim int，字典宽度；
    k_list tuple[int,...]，递增稀疏预算；delta float，中心化特征单位的松弛半径；
    constrained bool，True为RA字典，False为同初始值自由字典；seed int，仅控制初始化；
    initialization str，legacy复现首轮，unique要求代表点不少于字典行数并无放回初始化。
    状态：encoder可训练，中心固定；RA臂保存代表点、混合logits与松弛向量。
    forward接收[...,D]原始特征，返回每个K的(reconstructed[...,D], hidden[...,H])。
    """

    def __init__(self, points: torch.Tensor, feature_center: torch.Tensor,
                 hidden_dim: int, k_list: tuple[int, ...], delta: float,
                 constrained: bool, seed: int, initialization: str = 'legacy'):
        nn.Module.__init__(self)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("points必须为非空[M,D]代表点")
        if feature_center.shape != (points.shape[1],):
            raise ValueError("feature_center必须与代表点通道数一致")
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta必须为非负有限数")
        if not k_list or tuple(sorted(set(k_list))) != k_list:
            raise ValueError("k_list必须非空且严格递增")
        if not all(0 < k <= hidden_dim for k in k_list):
            raise ValueError("K必须位于(0,hidden_dim]")
        if initialization not in ('legacy', 'unique'):
            raise ValueError('initialization必须为legacy或unique')
        if initialization == 'unique' and len(points) < hidden_dim:
            raise ValueError('unique初始化不允许以较少代表点重复扩宽字典')
        self.input_dim = points.shape[1]
        self.hidden_dim = hidden_dim
        self.k_list = k_list
        self.delta = float(delta)
        self.constrained = constrained
        self.register_buffer("decoder_bias", feature_center.detach().clone())
        generator = torch.Generator(device=points.device).manual_seed(seed)
        # 每行独立选择代表点并混入少量其他点；H>M也不会出现零行。
        logits = torch.full((hidden_dim, len(points)), -math.log(len(points)) - 4,
                            device=points.device, dtype=points.dtype)
        # 当H>M时，单一代表点重复抽取会制造完全相同的字典/编码器行。
        # 对背景组合加入固定幅度的初始化扰动；两臂共用相同generator。
        logits += torch.rand(logits.shape, generator=generator, device=points.device,
                             dtype=points.dtype) - 0.5
        if initialization == 'unique':
            indices = torch.randperm(len(points), generator=generator,
                                     device=points.device)[:hidden_dim, None]
        else:
            indices = torch.randint(len(points), (hidden_dim, 1), generator=generator,
                                    device=points.device)
        logits.scatter_(1, indices, 0)
        initial = logits.softmax(-1) @ points
        if constrained:
            self.register_buffer("points", points.detach().clone())
            self.mixture_logits = nn.Parameter(logits)
            self.relaxation = nn.Parameter(torch.zeros_like(initial))
        else:
            self.free_dictionary = nn.Parameter(initial.clone())
        self.encoder = nn.Linear(self.input_dim, hidden_dim, device=points.device,
                                 dtype=points.dtype)
        with torch.no_grad():
            # 非单位字典以投影系数初始化编码器，避免把长度n重复放大为n²。
            # 两臂完全一致；只调整初始化，不归一化decoder、不改变其凸包。
            self.encoder.weight.copy_(initial / initial.square().sum(1, keepdim=True).clamp_min(1e-12))
            self.encoder.bias.zero_()

    @property
    def decoder_weight(self) -> torch.Tensor:
        """参数：无。返回Tensor[H,D]实际解码方向，不作单位范数化。"""
        if self.constrained:
            return self.mixture_logits.softmax(-1) @ self.points + self.relaxation
        return self.free_dictionary

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """参数：无；返回None。沿用S2c调用接口，仅投影RA松弛项，自由臂不操作。

        必须在optimizer.step之后调用；不对完整字典归一化，以免破坏凸组合约束。
        """
        if self.constrained:
            norm = self.relaxation.norm(dim=1, keepdim=True)
            scale = (self.delta / norm.clamp_min(torch.finfo(norm.dtype).tiny)).clamp(max=1)
            self.relaxation.mul_(scale)

    def forward(self, features: torch.Tensor) -> dict:
        """参数features Tensor[...,D]。返回dict[K,(重构,激活)]；共享一次字典计算。"""
        dictionary = self.decoder_weight
        return {k: (h @ dictionary + self.decoder_bias, h)
                for k, h in self.nested_hidden(features).items()}


@torch.no_grad()
def calibrate_initial_encoder(model: ArchetypalMatryoshkaSAE, spatial: torch.Tensor,
                              attention: torch.Tensor, image_weights: torch.Tensor,
                              batch_size: int) -> dict:
    """仅以train解析校准所有K共享的初始encoder幅度，禁止在已训练模型上调用。

    参数model为新初始化SAE（encoder bias为0）；spatial [N,49,D]原始特征；
    attention [N,49]冻结注意力；image_weights [N]患者/类别平衡权重；batch_size int。
    对t=F-center、r_K=h_K D，最小化sum_K E[w_pos ||s*r_K-t||²]。
    s=sum_K E[w_pos <r_K,t>]/sum_K E[w_pos ||r_K||²]，无验证标签或网格搜索。
    返回dict含scale及逐K校准前/后/均值参照MSE；只乘encoder权重，不修改decoder。
    正标量保持ReLU和TopK支持集，训练后不再次校准。
    """
    if bool(model.encoder.bias.ne(0).any()):
        raise ValueError('初始化幅度校准要求encoder bias为0')
    weights = image_weights / image_weights.sum()
    dictionary = model.decoder_weight
    cross = torch.zeros(len(model.k_list), dtype=torch.float64, device=spatial.device)
    square = torch.zeros_like(cross)
    target_square = torch.zeros((), dtype=torch.float64, device=spatial.device)
    for start in range(0, len(spatial), batch_size):
        x = spatial[start:start + batch_size]
        t = x - model.decoder_bias
        position_weights = patch_position_weights(attention[start:start + batch_size])
        iw = weights[start:start + batch_size]
        target_square += ((t.square().mean(2) * position_weights).mean(1).double() * iw).sum()
        for index, hidden in enumerate(model.nested_hidden(x).values()):
            rebuilt = hidden @ dictionary
            cross[index] += (((rebuilt * t).mean(2) * position_weights).mean(1).double() * iw).sum()
            square[index] += ((rebuilt.square().mean(2) * position_weights).mean(1).double() * iw).sum()
    scale = cross.sum() / square.sum()
    if not torch.isfinite(scale) or scale <= 0:
        raise RuntimeError('无法得到正且有限的初始化幅度；停止而不静默回退')
    model.encoder.weight.mul_(scale.to(model.encoder.weight.dtype))
    return {'scale': float(scale), 'per_k': {
        str(k): {'before': float(square[i] - 2 * cross[i] + target_square),
                 'after': float(scale.square() * square[i] - 2 * scale * cross[i] + target_square),
                 'center_only': float(target_square)} for i, k in enumerate(model.k_list)}}
