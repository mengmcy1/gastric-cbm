import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        # Final embedding: [sin, cos] x (2 axes x F frequencies per axis) = 4F.
        # Therefore hidden_dim must be divisible by four.
        assert hidden_dim % 4 == 0

        self.embedding_dim = hidden_dim
        F = self.embedding_dim // 4  # Number of frequencies per axis.

        # Frequencies: pi * {1, 2, 4, 8, ...}.
        e = torch.pow(2, torch.arange(F)).float() * np.pi  # (F,)

        # Build bases for the two axes (u, v).
        # Shape: (2, 2F), with the first F columns for u and the rest for v.
        basis = torch.stack([
            torch.cat([e, torch.zeros(F)]),  # Frequencies on u, zeros on v.
            torch.cat([torch.zeros(F), e]),  # Frequencies on v, zeros on u.
        ])  # (2, 2F)
        self.register_buffer('basis', basis)

        # Append the original (u, v), producing hidden_dim + 2 channels.
        self.mlp = nn.Linear(self.embedding_dim + 2, dim)

    @staticmethod
    def embed(input, basis):
        # input: (B, N, 2), basis: (2, 2F) -> projections: (B, N, 2F)
        projections = torch.einsum('bnd,de->bne', input, basis)
        # Concatenate sine and cosine: (B, N, 4F) = hidden_dim.
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        """
        input: (B, N, 2)  # (u, v)
        return: (B, N, dim)
        """
        enc = self.embed(input, self.basis)             # (B, N, hidden_dim)
        out = self.mlp(torch.cat([enc, input], dim=2))  # (B, N, hidden_dim+2) -> (B, N, dim)
        return out


class ImplicitMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, output_act='elu'):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers += [nn.Linear(lastv, hidden), nn.ReLU()]
            lastv = hidden

        if out_dim is not None:
            layers.append(nn.Linear(lastv, out_dim))
            act = {
                "sigmoid": nn.Sigmoid(),
                "relu": nn.ReLU(),
                "elu": nn.ELU(),
            }.get(output_act, nn.Identity())
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LowLevelImplicitHead(nn.Module):
    """
    Implicit head that fuses DINOv3 semantic features and BasicEncoder low-level features
    via grid_sample + gated fusion + MLP.

    Used as a drop-in replacement for CNN-based Gaussian heads in implicit mode.

    Args:
        hidden_dim: DINOv3 feature dimension (e.g. 1024)
        basic_dim: BasicEncoder output dim (e.g. 128)
        fusion_type: "concat", "gated", or "dino_only"
        out_dim: Output dimension (num_gaussian_parameters for GS, 1 for depth)
        hidden_list: MLP hidden layer dimensions
        output_act: Activation applied to MLP output ('elu', 'sigmoid', 'relu', None/'identity')
    """
    def __init__(
            self,
            hidden_dim,
            basic_dim=128,
            fusion_type="gated",
            out_dim=1,
            hidden_list=[1024, 256, 32],
            output_act=None,
            ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.basic_dim = basic_dim
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            in_channels = hidden_dim + basic_dim
        elif fusion_type == "gated":
            self.gate_proj = nn.Linear(basic_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            in_channels = hidden_dim
        elif fusion_type == "dino_only":
            in_channels = hidden_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.out_layer = ImplicitMLP(
            in_dim=in_channels,
            out_dim=out_dim,
            hidden_list=hidden_list,
            output_act=output_act if output_act is not None else 'identity',
        )

    def encode_feat(self, features, patch_h, patch_w):
        """Reshape last DINOv3 layer into spatial feature map."""
        x = features[-1][0]
        return x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def decode_dpt(self, feat, basic_feat, coord, cell=None):
        """
        Query features at given coordinates and decode.

        Args:
            feat: DINOv3 feature map [B, hidden_dim, H_dino, W_dino]
            basic_feat: BasicEncoder feature map `[B, basic_dim, H / 4, W / 4]`,
                or `None` when `fusion_type="dino_only"`.
            coord: Query coordinates [B, N, 2] in range [-1, 1], (y, x) order
            cell: unused, kept for API compatibility

        Returns:
            pred: [B, N, out_dim]
        """
        coord_ = coord.clone()
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

        # Sample DINOv3 features: coord is (y,x), grid_sample expects (x,y) → flip(-1)
        q_feat_dino = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)  # [B, N, hidden_dim]

        if self.fusion_type == "dino_only":
            q_feat_fused = q_feat_dino
        elif basic_feat is not None:
            q_feat_basic = F.grid_sample(
                basic_feat, coord_.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)  # [B, N, basic_dim]
            q_feat_fused = self._fuse_features(q_feat_dino, q_feat_basic)
        else:
            raise ValueError(
                f"fusion_type='{self.fusion_type}' requires BasicEncoder features."
            )

        return self.out_layer(q_feat_fused)

    def _fuse_features(self, feat_dino, feat_basic):
        if self.fusion_type == "concat":
            return torch.cat([feat_dino, feat_basic], dim=-1)
        elif self.fusion_type == "gated":
            feat_basic_proj = self.gate_proj(feat_basic)  # [B, N, hidden_dim]
            gate_weights = self.gate(torch.cat([feat_dino, feat_basic_proj], dim=-1))
            return gate_weights * feat_dino + (1 - gate_weights) * feat_basic_proj
        elif self.fusion_type == "dino_only":
            return feat_dino

    def forward(self, features, basic_feat, patch_h, patch_w, coords, cell=None):
        feat = self.encode_feat(features, patch_h, patch_w)
        return self.decode_dpt(feat, basic_feat, coords, cell)


class BasicImplicitGSHead(nn.Module):
    def __init__(
            self,
            basic_dim=128,
            coord_dim=None,
            out_dim=1,
            hidden_list=[1024, 256, 32],
            output_act=None,
            ):
        super().__init__()
        self.basic_dim = basic_dim
        self.coord_dim = coord_dim or basic_dim
        self.coord_embed = CoordEmbed(dim=self.coord_dim)
        self.out_layer = ImplicitMLP(
            in_dim=basic_dim + self.coord_dim,
            out_dim=out_dim,
            hidden_list=hidden_list,
            output_act=output_act if output_act is not None else 'identity',
        )

    def decode_dpt(self, basic_feat, coord, cell=None):
        coord_ = coord.clone()
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

        q_feat_basic = F.grid_sample(
            basic_feat, coord_.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        coord_feat = self.coord_embed(coord_.flip(-1))
        return self.out_layer(torch.cat([q_feat_basic, coord_feat], dim=-1))

    def forward(self, basic_feat, coords, cell=None):
        return self.decode_dpt(basic_feat, coords, cell)
