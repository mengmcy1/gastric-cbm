"""Canonical Layer Bundle (CLB) value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


class SupportType(str, Enum):
    OBSERVED_SURFACE = "observed_surface"
    OCCLUSION_HIDDEN = "occlusion_hidden"
    OUTSIDE_SOURCE_FOV = "outside_source_fov"


class Provenance(IntEnum):
    INVALID = 0
    OBSERVED_CENTER = 1
    REPROJECTED_OBSERVATION = 2
    PREDICTED_HIDDEN_GEOMETRY = 3
    PREDICTED_HIDDEN_APPEARANCE = 4
    EXTERNAL_MULTIVIEW = 5
    SYNTHETIC_ORACLE = 6


@dataclass(frozen=True, slots=True)
class Camera:
    camera_id: str
    intrinsics_3x3_float64: npt.NDArray[np.float64]
    world_to_camera_4x4_float64: npt.NDArray[np.float64]
    angle_deg: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalLayerPatch:
    """One CLB surface patch anchored to an explicit source camera."""

    patch_id: str
    support_type: SupportType
    rgb_uint8: npt.NDArray[np.uint8]
    depth_z_float32: npt.NDArray[np.float32]
    alpha_float32: npt.NDArray[np.float32]
    valid_mask_uint8: npt.NDArray[np.uint8]
    provenance_uint8: npt.NDArray[np.uint8]
    geometry_confidence_float32: npt.NDArray[np.float32]
    appearance_confidence_float32: npt.NDArray[np.float32]
    intrinsics_3x3_float64: npt.NDArray[np.float64]
    world_to_camera_4x4_float64: npt.NDArray[np.float64]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.depth_z_float32.shape)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CanonicalLayerBundle:
    bundle_id: str
    deployment: bool
    coordinate_convention: Mapping[str, Any]
    cameras: Mapping[str, Camera]
    patches: Sequence[CanonicalLayerPatch]
    metadata: Mapping[str, Any] = field(default_factory=dict)
