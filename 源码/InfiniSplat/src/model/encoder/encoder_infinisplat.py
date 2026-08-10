from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.model.types import BatchedViews
from src.model.encoder.gaussian.gaussian_decoder import (
    GaussianDecoder,
    GaussianDecoderCfg,
)
from src.model.encoder.depth.depthpro.depthpro_wrapper import DepthPro
from src.model.encoder.depth.infinidepth.sampling_utils import (
    SparseSamplingOutput,
    make_sparse_surface_samples,
)
from src.model.encoder.gaussian.basic_encoder import BasicEncoder
from src.model.encoder.gaussian.implicit_gs_head import ImplicitGSHead
from src.utils.gaussians import Gaussians3D, unproject_gaussians

from src.model.encoder.encoder import Encoder


class DinoBasicImageFeatureBranch(nn.Module):
    """DINOv3 image branch with BasicEncoder low-level features.

    Args:
        backbone_type: DINOv3 backbone size identifier.
        basic_dim: Output channels for the BasicEncoder branch.
    """

    def __init__(
        self,
        backbone_type: str,
        basic_dim: int,
    ) -> None:
        super().__init__()

        dinov3_layer_indices = {
            "vitl16": [4, 11, 17, 23],
            "vith16plus": [7, 15, 23, 31],
        }
        dinov3_repo_dir = (Path(__file__).resolve().parent / "blocks" / "torchhub" / "dinov3")

        if backbone_type not in dinov3_layer_indices:
            raise ValueError(f"Unsupported DINOv3 encoder: {backbone_type}")

        self.backbone = torch.hub.load(
            str(dinov3_repo_dir),
            f"dinov3_{backbone_type}",
            source="local",
            pretrained=False,
        )
        self.layer_indices = dinov3_layer_indices[backbone_type]
        self.patch_size = 16
        self.hidden_dim = self.backbone.blocks[0].attn.qkv.in_features
        self.basic_encoder = BasicEncoder(input_dim=3, output_dim=basic_dim, stride=4)

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor):
        """Extract DINOv3 tokens and BasicEncoder features.

        Args:
            image: RGB tensor with shape `[B, 3, H, W]` in `[0, 1]`.

        Returns:
            A tuple `(features, basic_feat, patch_h, patch_w)`, where `features`
            are DINOv3 intermediate outputs, `basic_feat` has shape
            `[B, C_basic, H / 4, W / 4]`, and `patch_h`, `patch_w` describe the
            DINO patch grid.
        """
        h, w = image.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        # DINO ViT-L is the heaviest forward pass; run in bf16 for speed.
        # Downstream fp32 ops auto-upcast the bf16 features.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features = self.backbone.get_intermediate_layers(
                (image - self._mean) / self._std,
                n=self.layer_indices,
                return_class_token=True,
            )
        basic_feat = self.basic_encoder(2.0 * image - 1.0)
        return features, basic_feat, patch_h, patch_w


@dataclass
class EncoderInfiniSplatCfg:
    name: Literal["infinisplat"]
    sample_point_num: int
    image_basic_dim: int
    image_backbone_type: str
    implicit_gs_query_batch_size: int
    implicit_gs_hidden_list: list[int]
    gaussian_decoder: GaussianDecoderCfg


class EncoderInfiniSplat(Encoder[EncoderInfiniSplatCfg]):
    def __init__(self, cfg: EncoderInfiniSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = DepthPro()
        self.depth_predictor.eval()

        self.image_feature_branch = DinoBasicImageFeatureBranch(
            backbone_type=cfg.image_backbone_type,
            basic_dim=cfg.image_basic_dim,
        )
        self.implicit_gs_head = ImplicitGSHead(
            hidden_dim=self.image_feature_branch.hidden_dim,
            basic_dim=cfg.image_basic_dim,
            hidden_list=list(cfg.implicit_gs_hidden_list),
        )

        self.gaussian_decoder = GaussianDecoder(cfg=cfg.gaussian_decoder)

    def _sample_map(
        self,
        feature_map: torch.Tensor,
        coords_yx: torch.Tensor,
    ) -> torch.Tensor:
        sampled = F.grid_sample(
            feature_map,
            coords_yx.flip(-1).unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        )
        return sampled[:, :, 0, :].transpose(1, 2)

    def _sample_sparse_coords(
        self,
        dense_depthmap_flat: torch.Tensor,
        intrinsics_flat: torch.Tensor,
        image_flat: torch.Tensor,
    ) -> SparseSamplingOutput:
        sample_coords_yx_ndc = []
        sample_kind = []
        sample_responsibility_area_metric = []

        for sample_index, (depth_hw, intrinsic, image_chw) in enumerate(zip(
            dense_depthmap_flat[:, 0],
            intrinsics_flat,
            image_flat,
        )):
            try:
                sampling_output = make_sparse_surface_samples(
                    depth_hw=depth_hw,
                    image_chw=image_chw.detach(),
                    fx=float(intrinsic[0, 0].item()),
                    fy=float(intrinsic[1, 1].item()),
                    cx=float(intrinsic[0, 2].item()),
                    cy=float(intrinsic[1, 2].item()),
                    sample_point_num=int(self.cfg.sample_point_num),
                )
            except (RuntimeError, ValueError) as exc:
                valid = depth_hw[torch.isfinite(depth_hw) & (depth_hw > 0.0)]
                if valid.numel() == 0:
                    depth_stats = "no positive finite depth"
                else:
                    depth_stats = (
                        f"valid={int(valid.numel())}/{int(depth_hw.numel())}, "
                        f"min={float(valid.min().item()):.6g}, "
                        f"median={float(valid.median().item()):.6g}, "
                        f"max={float(valid.max().item()):.6g}"
                    )
                raise RuntimeError(
                    "Surface sampling failed for context sample "
                    f"flat_index={sample_index}; "
                    f"{depth_stats}. Original error: {exc}"
                ) from exc
            sample_coords_yx_ndc.append(sampling_output.coords_yx_ndc)
            sample_kind.append(sampling_output.sample_kind)
            sample_responsibility_area_metric.append(
                sampling_output.sample_responsibility_area_metric
            )
        return SparseSamplingOutput(
            coords_yx_ndc=torch.stack(sample_coords_yx_ndc, dim=0),
            sample_responsibility_area_metric=torch.stack(sample_responsibility_area_metric, dim=0),
            sample_kind=torch.stack(sample_kind, dim=0),
        )

    def _decode_dino_gaussian_delta(
        self,
        features,
        basic_feat: torch.Tensor,
        patch_h: int,
        patch_w: int,
        coords_yx: torch.Tensor,
    ) -> torch.Tensor:
        feat_map = self.implicit_gs_head.encode_feat(features, patch_h, patch_w)
        query_batch_size = int(self.cfg.implicit_gs_query_batch_size)
        num_queries = coords_yx.shape[1]
        chunks = []
        for start in range(0, num_queries, query_batch_size):
            end = min(start + query_batch_size, num_queries)
            chunks.append(
                self.implicit_gs_head.decode_dpt(
                    feat_map,
                    basic_feat,
                    coords_yx[:, start:end],
                )
            )
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        context: BatchedViews,
    ):
        b, v, _, h, w = context["image"].shape

        image_flat = rearrange(context["image"], "b v c h w -> (b v) c h w")
        intrinsics_norm_flat = rearrange(context["intrinsics"], "b v i j -> (b v) i j")
        intrinsics = context["intrinsics"].clone()
        intrinsics[:, :, 0] = intrinsics[:, :, 0] * w
        intrinsics[:, :, 1] = intrinsics[:, :, 1] * h
        intrinsics_flat = rearrange(intrinsics, "b v i j -> (b v) i j")

        with torch.no_grad():
            self.depth_predictor.eval()
            dense_depthmap_flat = self.depth_predictor(
                {
                    "image": image_flat,
                    "intrinsics": intrinsics_norm_flat,
                }
            )

        if dense_depthmap_flat.ndim != 4 or dense_depthmap_flat.shape[1] != 1:
            raise AssertionError(
                "InfiniSplat expects the selected depth model to return a single dense depth layer."
            )

        sampling_output_flat = self._sample_sparse_coords(
            dense_depthmap_flat=dense_depthmap_flat.detach(),
            intrinsics_flat=intrinsics_flat,
            image_flat=image_flat,
        )
        sample_coords_yx_ndc_flat = sampling_output_flat.coords_yx_ndc
        sample_kind_flat = sampling_output_flat.sample_kind
        sample_responsibility_area_metric_flat = (
            sampling_output_flat.sample_responsibility_area_metric
        )
        sample_depths_flat = self._sample_map(dense_depthmap_flat, sample_coords_yx_ndc_flat)
        sampled_rgb_flat = self._sample_map(image_flat, sample_coords_yx_ndc_flat)

        features, basic_feat, patch_h, patch_w = self.image_feature_branch(image_flat)
        gaussian_delta_flat = self._decode_dino_gaussian_delta(
            features=features,
            basic_feat=basic_feat,
            patch_h=patch_h,
            patch_w=patch_w,
            coords_yx=sample_coords_yx_ndc_flat,
        )

        sample_depths = rearrange(sample_depths_flat, "(b v) n c -> b v n c", b=b, v=v)
        sample_coords_yx_ndc = rearrange(
            sample_coords_yx_ndc_flat,
            "(b v) n c -> b v n c",
            b=b,
            v=v,
        )
        sample_responsibility_area_metric = rearrange(
            sample_responsibility_area_metric_flat,
            "(b v) n -> b v n",
            b=b,
            v=v,
        )
        sample_kind = rearrange(sample_kind_flat, "(b v) n -> b v n", b=b, v=v)
        sampled_rgb = rearrange(sampled_rgb_flat, "(b v) n c -> b v n c", b=b, v=v)
        gaussian_delta = rearrange(gaussian_delta_flat, "(b v) n c -> b v n c", b=b, v=v)
        gaussians_ndc: Gaussians3D = self.gaussian_decoder(
            delta=gaussian_delta,
            coords_yx_ndc=sample_coords_yx_ndc,
            depths=sample_depths,
            rgb=sampled_rgb,
            intrinsics=intrinsics,
            sample_kind=sample_kind,
            sample_responsibility_area_metric=sample_responsibility_area_metric,
            image_shape=(h, w),
        )
        gaussians: Gaussians3D = unproject_gaussians(
            gaussians_ndc,
            context["extrinsics"],
            intrinsics,
            (w, h),
        )

        return {"gaussians": gaussians}
