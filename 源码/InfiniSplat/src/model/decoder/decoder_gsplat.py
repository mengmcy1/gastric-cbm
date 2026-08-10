from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.model.decoder.decoder import Decoder
from src.utils.color_space import linearRGB2sRGB
from src.utils.gaussians import Gaussians3D

try:
    from gsplat import rasterization
except ImportError:
    rasterization = None


def is_gsplat_available() -> bool:
    """Return whether optional novel-view rendering is available."""
    return rasterization is not None


@dataclass
class DecoderGsplatCfg:
    name: Literal["gsplat"]
    background_color: list[float]


class DecoderGsplat(Decoder[DecoderGsplatCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderGsplatCfg,
    ) -> None:
        super().__init__(cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, "batch view 3 height width"]:
        if rasterization is None:
            raise RuntimeError(
                "Novel-view rendering requires the optional `gsplat` package."
            )

        h, w = image_shape

        xyzs = gaussians.mean_vectors.float()
        opacitys = gaussians.opacities.float()
        rotations = gaussians.quaternions.float()
        scales = gaussians.singular_values.float()
        colors = gaussians.colors.float()
        covariances = gaussians.covariances.float() if gaussians.covariances is not None else None

        # The public batch interface stores extrinsics as OpenCV world-to-camera.
        test_w2c = extrinsics.float()
        test_intr_normalized = intrinsics.float()
        test_intr = test_intr_normalized.clone()
        test_intr[:, :, 0] = test_intr_normalized[:, :, 0] * w
        test_intr[:, :, 1] = test_intr_normalized[:, :, 1] * h

        rendering, alpha, _ = rasterization(
            xyzs,
            rotations,
            scales,
            opacitys,
            colors,
            test_w2c,
            test_intr,
            w,
            h,
            sh_degree=None,
            render_mode="RGB",
            packed=True,
            covars=covariances,
            eps2d=1e-8,
        )

        rendered_rgb = rearrange(rendering, "b v h w c -> b v c h w")
        alpha_rgb = rearrange(alpha, "b v h w 1 -> b v 1 h w")

        rendered_rgb = linearRGB2sRGB(rendered_rgb)

        backgrounds = rearrange(self.background_color.to(rendered_rgb), "c -> 1 1 c 1 1")
        rendered_rgb = rendered_rgb + backgrounds * (1.0 - alpha_rgb)
        rendered_rgb = rendered_rgb.clamp(0.0, 1.0)

        return rendered_rgb
