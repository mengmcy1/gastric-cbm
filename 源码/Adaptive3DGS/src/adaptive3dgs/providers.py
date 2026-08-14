"""Replaceable provider contracts; implementations must return validated CLB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .clb import CanonicalLayerPatch


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Single-image inference input with no implicit camera convention."""

    sample_id: str
    rgb_uint8: npt.NDArray[np.uint8]
    intrinsics_3x3_float64: npt.NDArray[np.float64]
    world_to_camera_4x4_float64: npt.NDArray[np.float64]
    depth_z_float32: npt.NDArray[np.float32] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Provider output before fusion or Gaussian spawning."""

    provider_id: str
    patches: Sequence[CanonicalLayerPatch]
    unsupported_regions: Sequence[str] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class OcclusionHiddenProvider(ABC):
    """Predict surfaces hidden behind content inside the source frustum."""

    provider_id: str

    @abstractmethod
    def predict(self, context: ProviderContext) -> ProviderResult:
        raise NotImplementedError

class OutsideFOVProvider(ABC):
    """Predict world content beyond the source image frustum."""

    provider_id: str

    @abstractmethod
    def predict(self, context: ProviderContext) -> ProviderResult:
        raise NotImplementedError
