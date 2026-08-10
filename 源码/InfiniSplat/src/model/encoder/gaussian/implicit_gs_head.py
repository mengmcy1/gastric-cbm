from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ImplicitMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_list: Sequence[int],
    ) -> None:
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_list:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.ReLU()))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ImplicitGSHead(nn.Module):
    """Decode Gaussian parameters from DINOv3 and BasicEncoder features."""

    def __init__(
        self,
        hidden_dim: int,
        basic_dim: int,
        hidden_list: Sequence[int],
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(basic_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_layer = ImplicitMLP(
            in_dim=hidden_dim,
            out_dim=14,
            hidden_list=hidden_list,
        )

    def encode_feat(self, features, patch_h: int, patch_w: int) -> torch.Tensor:
        """Reshape the last DINOv3 layer into a spatial feature map."""
        tokens = features[-1][0]
        return tokens.permute(0, 2, 1).reshape(
            tokens.shape[0],
            tokens.shape[-1],
            patch_h,
            patch_w,
        )

    def decode_dpt(
        self,
        feat: torch.Tensor,
        basic_feat: torch.Tensor,
        coord: torch.Tensor,
    ) -> torch.Tensor:
        coord = coord.clone()
        coord.clamp_(-1 + 1e-6, 1 - 1e-6)

        query_dino = F.grid_sample(
            feat,
            coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        query_basic = F.grid_sample(
            basic_feat,
            coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        query_basic = self.gate_proj(query_basic)
        gate_weights = self.gate(torch.cat((query_dino, query_basic), dim=-1))
        fused_features = gate_weights * query_dino + (1 - gate_weights) * query_basic
        return self.out_layer(fused_features)
