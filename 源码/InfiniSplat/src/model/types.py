from typing import TypedDict

from jaxtyping import Float
from torch import Tensor


class BatchedViews(TypedDict, total=False):
    """Single- or multi-view tensors consumed by the inference encoders."""

    extrinsics: Float[Tensor, "batch view 4 4"]
    intrinsics: Float[Tensor, "batch view 3 3"]
    image: Float[Tensor, "batch view channel height width"]
    prompt_disparity: Float[Tensor, "batch view channel height width"]
    prompt_mask: Float[Tensor, "batch view channel height width"]
