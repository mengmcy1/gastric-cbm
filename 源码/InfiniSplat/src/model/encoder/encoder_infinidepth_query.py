from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn.functional as F
from einops import rearrange

from src.model.types import BatchedViews
from src.model.encoder.depth.infinidepth.infinidepth_wrapper import InfiniDepth
from src.model.encoder.depth.infinidepth.sampling_utils import (
    SparseSamplingOutput,
    make_sparse_surface_samples,
)
from src.model.encoder.encoder import Encoder
from src.model.encoder.encoder_infinisplat import DinoBasicImageFeatureBranch
from src.model.encoder.gaussian.gaussian_decoder import (
    GaussianDecoder,
    GaussianDecoderCfg,
)
from src.model.encoder.gaussian.implicit_gs_head import ImplicitGSHead
from src.utils.gaussians import Gaussians3D, unproject_gaussians

@dataclass
class EncoderInfiniDepthQueryCfg:
    name: Literal["infinisplat_infinidepth"]
    sample_point_num: int
    image_basic_dim: int
    image_backbone_type: str
    implicit_gs_query_batch_size: int
    implicit_gs_hidden_list: list[int]
    gaussian_decoder: GaussianDecoderCfg


class EncoderInfiniDepthQuery(Encoder[EncoderInfiniDepthQueryCfg]):
    def __init__(self, cfg: EncoderInfiniDepthQueryCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = InfiniDepth()
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
        """Sample sparse support coordinates from the selected dense depth surface.

        Args:
            dense_depthmap_flat: Dense depth with shape `[B*V, 1, H, W]`.
            intrinsics_flat: Pixel-space intrinsics with shape `[B*V, 3, 3]`.
            image_flat: Context RGB images with shape `[B*V, 3, H, W]`.

        Returns:
            Sparse sampling output whose tensors have leading shape `[B*V, N]`.
        """
        sample_coords_yx_ndc = []
        sample_kind = []
        sample_responsibility_area_metric = []

        for sample_index, (depth_hw, intrinsic, image_chw) in enumerate(
            zip(dense_depthmap_flat[:, 0], intrinsics_flat, image_flat, strict=True)
        ):
            sampling_output = make_sparse_surface_samples(
                depth_hw=depth_hw,
                image_chw=image_chw.detach(),
                fx=float(intrinsic[0, 0].item()),
                fy=float(intrinsic[1, 1].item()),
                cx=float(intrinsic[0, 2].item()),
                cy=float(intrinsic[1, 2].item()),
                sample_point_num=int(self.cfg.sample_point_num),
                coord_norm="minus_one_to_one",
            )

            sample_coords_yx_ndc.append(sampling_output.coords_yx_ndc)
            sample_kind.append(sampling_output.sample_kind)
            sample_responsibility_area_metric.append(
                sampling_output.sample_responsibility_area_metric
            )

        return SparseSamplingOutput(
            coords_yx_ndc=torch.stack(sample_coords_yx_ndc, dim=0),
            sample_responsibility_area_metric=torch.stack(
                sample_responsibility_area_metric,
                dim=0,
            ),
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
        """Encode prompt-conditioned context views with InfiniDepth.

        Args:
            context: Batched context views.
        Returns:
            Encoder output dictionary following the InfiniSplat contract.
        """
        required_keys = ("prompt_disparity", "prompt_mask")
        missing_keys = [key for key in required_keys if key not in context]
        if missing_keys:
            raise AssertionError(
                "EncoderInfiniDepthQuery requires prompt-conditioned context inputs. "
                f"Missing keys: {missing_keys}"
            )

        b, v, _, h, w = context["image"].shape

        image_flat = rearrange(context["image"], "b v c h w -> (b v) c h w")
        intrinsics = context["intrinsics"].clone()
        intrinsics[:, :, 0] = intrinsics[:, :, 0] * w
        intrinsics[:, :, 1] = intrinsics[:, :, 1] * h
        intrinsics_flat = rearrange(intrinsics, "b v i j -> (b v) i j")

        with torch.no_grad():
            self.depth_predictor.eval()
            dense_depthmap_flat = self.depth_predictor(
                {
                    "image": image_flat,
                    "prompt_disparity": rearrange(
                        context["prompt_disparity"],
                        "b v c h w -> (b v) c h w",
                    ),
                    "prompt_mask": rearrange(context["prompt_mask"], "b v c h w -> (b v) c h w"),
                }
            )
        sampling_output_flat = self._sample_sparse_coords(
            dense_depthmap_flat=dense_depthmap_flat.detach(),
            intrinsics_flat=intrinsics_flat,
            image_flat=image_flat,
        )
        sample_depths_flat = self._sample_map(
            dense_depthmap_flat,
            sampling_output_flat.coords_yx_ndc,
        )

        sample_coords_yx_ndc_flat = sampling_output_flat.coords_yx_ndc
        sample_kind_flat = sampling_output_flat.sample_kind
        sample_responsibility_area_metric_flat = (
            sampling_output_flat.sample_responsibility_area_metric
        )
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
