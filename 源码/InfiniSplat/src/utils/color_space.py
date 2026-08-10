from typing import Literal

import torch
from torch import Tensor


ColorSpace = Literal["sRGB", "linearRGB"]

_SRGB_THRESHOLD = 0.04045
_LINEAR_THRESHOLD = 0.0031308


def encode_color_space(color_space: ColorSpace) -> int:
    return 0 if color_space == "sRGB" else 1


def sRGB2linearRGB(srgb: Tensor) -> Tensor:
    return torch.where(
        srgb <= _SRGB_THRESHOLD,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055).pow(2.4),
    )


def linearRGB2sRGB(linear_rgb: Tensor) -> Tensor:
    return torch.where(
        linear_rgb <= _LINEAR_THRESHOLD,
        linear_rgb * 12.92,
        1.055 * linear_rgb.clamp(min=0.0).pow(1.0 / 2.4) - 0.055,
    )
