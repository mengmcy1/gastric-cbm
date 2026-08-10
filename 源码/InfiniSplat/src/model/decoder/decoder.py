from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from src.utils.gaussians import Gaussians3D


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, "batch view 3 height width"]:
        raise NotImplementedError
