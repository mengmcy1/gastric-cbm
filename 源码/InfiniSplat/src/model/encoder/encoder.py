from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from torch import nn

from src.model.types import BatchedViews

EncoderOutput = dict[str, Any]

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
    ) -> EncoderOutput:
        """Encode context views into the Gaussian scene representation."""
        raise NotImplementedError
