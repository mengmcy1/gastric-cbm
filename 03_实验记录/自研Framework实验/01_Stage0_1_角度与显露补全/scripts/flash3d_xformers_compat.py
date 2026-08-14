"""Explicit xFormers compatibility used by the Flash3D smoke test.

The Nystrom formula below is a minimal PyTorch port of the BSD-licensed
xFormers v0.0.25.post1 implementation at commit
7fffd3d65c0a30053d26d23871b985ac7665c36d.  That implementation accepts
3-D [batch*heads, sequence, head_dim] tensors.  UniDepth v1 passes 4-D
[batch, sequence, heads, head_dim] tensors, so this adapter performs the
standard head-fold/unfold explicitly before applying the same formula.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


XFORMERS_SOURCE_COMMIT = "7fffd3d65c0a30053d26d23871b985ac7665c36d"


class _AvgPool(nn.Module):
    def __init__(self, landmarks: int):
        super().__init__()
        self.landmarks = landmarks

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sequence, head_dim = value.shape[1:]
        segment = sequence // self.landmarks
        if segment <= 0:
            raise ValueError("num_landmarks must not exceed sequence length here")
        if sequence % self.landmarks == 0:
            return value.reshape(
                -1, self.landmarks, segment, head_dim
            ).mean(dim=-2)
        rounded = self.landmarks - sequence % self.landmarks
        even = value[:, : rounded * segment].reshape(
            -1, rounded, segment, head_dim
        ).mean(dim=-2)
        uneven = value[:, rounded * segment :].reshape(
            -1, self.landmarks - rounded, segment + 1, head_dim
        ).mean(dim=-2)
        return torch.cat((even, uneven), dim=-2)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(
        (q / math.sqrt(k.size(-1))) @ k.transpose(-2, -1), dim=-1
    )
    return probabilities @ v


def _kernel(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.softmax(
        (q / math.sqrt(k.size(-1))) @ k.transpose(-2, -1), dim=-1
    )


def _iterative_pinv(matrix: torch.Tensor, iterations: int) -> torch.Tensor:
    identity = torch.eye(
        matrix.size(-1), device=matrix.device, dtype=matrix.dtype
    )
    inverse = (
        1
        / torch.max(torch.sum(matrix, dim=-2), dim=-1).values[:, None, None]
        * matrix.transpose(-1, -2)
    )
    for _ in range(iterations):
        product = matrix @ inverse
        inverse = 0.25 * inverse @ (
            13 * identity
            - product @ (15 * identity - product @ (7 * identity - product))
        )
    return inverse


class CompatNystromAttention(nn.Module):
    """xFormers 0.0.25.post1 Nystrom math with explicit head folding."""

    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_landmarks: int = 64,
        inv_iterations: int = 6,
        **kwargs,
    ) -> None:
        super().__init__()
        unsupported = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False)
        }
        if unsupported:
            raise ValueError(f"unsupported Nystrom compatibility options: {unsupported}")
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.inv_iterations = inv_iterations
        self.dropout = nn.Dropout(dropout)
        self.pool = _AvgPool(num_landmarks)

    def _forward_3d(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        sequence = k.size(-2)
        if self.num_landmarks >= sequence:
            return self.dropout(_attention(q, k, v))
        q_landmarks = self.pool(q)
        k_landmarks = self.pool(k)
        kernel_1 = _kernel(q, k_landmarks)
        kernel_2 = _kernel(q_landmarks, k_landmarks)
        kernel_3 = _attention(q_landmarks, k, v)
        output = kernel_1 @ _iterative_pinv(
            kernel_2, self.inv_iterations
        ) @ kernel_3
        return self.dropout(output)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            raise ValueError("Flash3D smoke does not support a Nystrom padding mask")
        if kwargs.get("att_mask") is not None:
            raise ValueError("Flash3D smoke does not support a Nystrom attention mask")
        if q.ndim == 3:
            return self._forward_3d(q, k, v)
        if q.ndim != 4:
            raise ValueError(f"expected 3-D or 4-D q/k/v, got {q.ndim}-D")
        batch, sequence, heads, head_dim = q.shape
        if heads != self.num_heads:
            raise ValueError(f"expected {self.num_heads} heads, got {heads}")

        def fold(value: torch.Tensor) -> torch.Tensor:
            return value.permute(0, 2, 1, 3).reshape(
                batch * heads, sequence, head_dim
            )

        output = self._forward_3d(fold(q), fold(k), fold(v))
        return output.reshape(batch, heads, sequence, head_dim).permute(
            0, 2, 1, 3
        )


def install_unidepth_compatibility() -> None:
    """Expose the removed class before UniDepth imports it."""
    import xformers.components.attention as attention_components

    attention_components.NystromAttention = CompatNystromAttention


def force_dino_reference_attention() -> None:
    """Use UniDepth's own PyTorch reference attention on unsupported SM 12.0."""
    from unidepth.models.backbones.metadinov2 import attention

    attention.XFORMERS_AVAILABLE = False
