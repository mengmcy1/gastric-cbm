"""Read-only Flash3D reference adapter.

This module deliberately has no dependency on the Flash3D repository or
PyTorch.  A repository-specific backend performs inference and returns NumPy
arrays; this adapter only maps the frozen official semantics into the common
Provider/CLB/evaluator contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
import numpy.typing as npt

from ..clb import CanonicalLayerPatch, Provenance, SupportType
from ..evaluation import GeometryPrediction
from ..providers import OcclusionHiddenProvider, ProviderContext, ProviderResult


FLASH3D_SH_C0 = 0.28209479177387814


@dataclass(frozen=True, slots=True)
class Flash3DSecondLayerOutput:
    """Source-anchored arrays extracted from an unchanged Flash3D model."""

    layer1_depth_z_float32: npt.NDArray[np.float32]
    layer2_depth_z_float32: npt.NDArray[np.float32]
    layer2_alpha_float32: npt.NDArray[np.float32]
    layer2_features_dc_hwc_float32: npt.NDArray[np.float32]
    intrinsics_3x3_float64: npt.NDArray[np.float64]
    world_to_camera_4x4_float64: npt.NDArray[np.float64]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Flash3DInferenceBackend(Protocol):
    """External backend boundary; implementations may depend on Flash3D/Torch."""

    backend_id: str

    def predict_second_layer(self, context: ProviderContext) -> Flash3DSecondLayerOutput:
        ...


def _float32_image(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be HxW")
    return result.copy()


class Flash3DReadOnlyAdapter(OcclusionHiddenProvider):
    """Expose Flash3D layer 2 as an uncalibrated occlusion-hidden Provider."""

    provider_id = "flash3d-read-only-reference"

    def __init__(self, backend: Flash3DInferenceBackend) -> None:
        self._backend = backend

    def predict(self, context: ProviderContext) -> ProviderResult:
        output = self._backend.predict_second_layer(context)
        layer1 = _float32_image(output.layer1_depth_z_float32, "layer1 depth")
        layer2 = _float32_image(output.layer2_depth_z_float32, "layer2 depth")
        alpha = _float32_image(output.layer2_alpha_float32, "layer2 alpha")
        if layer2.shape != layer1.shape or alpha.shape != layer1.shape:
            raise ValueError("Flash3D layer arrays must have the same HxW shape")
        if np.any(np.isfinite(alpha) & ((alpha < 0.0) | (alpha > 1.0))):
            raise ValueError("Flash3D layer2 alpha outside [0,1]")

        features_dc = np.asarray(output.layer2_features_dc_hwc_float32, dtype=np.float32)
        if features_dc.shape != (*layer1.shape, 3):
            raise ValueError("Flash3D layer2 features_dc must be HxWx3")
        intrinsics = np.asarray(output.intrinsics_3x3_float64, dtype=np.float64)
        world_to_camera = np.asarray(output.world_to_camera_4x4_float64, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError("Flash3D output intrinsics must be 3x3")
        if world_to_camera.shape != (4, 4):
            raise ValueError("Flash3D output world_to_camera must be 4x4")

        valid = (
            np.isfinite(layer1)
            & np.isfinite(layer2)
            & np.isfinite(alpha)
            & np.isfinite(features_dc).all(axis=2)
            & (layer1 > 0.0)
            & (layer2 > layer1)
        )
        safe_dc = np.where(np.isfinite(features_dc), features_dc, 0.0)
        rgb_float = np.clip(0.5 + FLASH3D_SH_C0 * safe_dc, 0.0, 1.0)
        rgb_uint8 = np.round(rgb_float * 255.0).astype(np.uint8)
        depth = layer2.copy()
        depth[~valid] = np.nan
        clean_alpha = np.where(np.isfinite(alpha), alpha, 0.0).astype(np.float32)
        provenance = np.zeros(layer1.shape, dtype=np.uint8)
        provenance[valid] = Provenance.PREDICTED_HIDDEN_GEOMETRY
        zero_confidence = np.zeros(layer1.shape, dtype=np.float32)

        patch = CanonicalLayerPatch(
            patch_id=f"{context.sample_id}-flash3d-layer2",
            support_type=SupportType.OCCLUSION_HIDDEN,
            rgb_uint8=rgb_uint8,
            depth_z_float32=depth,
            alpha_float32=clean_alpha,
            valid_mask_uint8=valid.astype(np.uint8),
            provenance_uint8=provenance,
            geometry_confidence_float32=zero_confidence.copy(),
            appearance_confidence_float32=zero_confidence.copy(),
            intrinsics_3x3_float64=intrinsics.copy(),
            world_to_camera_4x4_float64=world_to_camera.copy(),
            metadata={
                **dict(output.metadata),
                "anchor_camera": str(output.metadata.get("anchor_camera", "center")),
                "adapter": self.provider_id,
                "backend_id": self._backend.backend_id,
                "appearance_encoding": "sRGB_uint8_from_clamp(0.5+SH_C0*features_dc)",
                "alpha_semantics": "Flash3D Gaussian opacity; support only, not confidence",
                "confidence_status": "uncalibrated_zero_not_accepted_for_fusion",
                "outside_source_fov_supported": False,
            },
        )
        return ProviderResult(
            provider_id=self.provider_id,
            patches=(patch,),
            unsupported_regions=(SupportType.OUTSIDE_SOURCE_FOV.value,),
            diagnostics={
                "backend_id": self._backend.backend_id,
                "valid_hidden_pixels": int(valid.sum()),
                "confidence_calibrated": False,
                "read_only_reference": True,
            },
        )


def flash3d_render_to_geometry_prediction(
    weighted_depth_z: npt.ArrayLike,
    accumulated_alpha: npt.ArrayLike,
    *,
    model_depth_units_per_metric_unit: float = 1.0,
) -> GeometryPrediction:
    """Normalize a native Flash3D layer-2 target render for the evaluator.

    Flash3D's renderer returns opacity-weighted Z.  Dividing by an independently
    rendered accumulated alpha reproduces the frozen Stage 1.4 normalization.
    Alpha is support probability only; this adapter intentionally emits zero
    geometry confidence because Flash3D has no calibrated confidence head.
    """

    weighted = _float32_image(weighted_depth_z, "weighted depth")
    alpha = _float32_image(accumulated_alpha, "accumulated alpha")
    if alpha.shape != weighted.shape:
        raise ValueError("weighted depth and accumulated alpha shapes differ")
    if not np.isfinite(model_depth_units_per_metric_unit) or model_depth_units_per_metric_unit <= 0:
        raise ValueError("model_depth_units_per_metric_unit must be finite and positive")
    if not np.isfinite(alpha).all() or np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("accumulated alpha must be finite and in [0,1]")

    supported = alpha > 0.0
    depth = np.full(weighted.shape, np.nan, dtype=np.float32)
    valid = supported & np.isfinite(weighted)
    depth[valid] = (
        weighted[valid]
        / np.maximum(alpha[valid], np.float32(1e-8))
        / np.float32(model_depth_units_per_metric_unit)
    )
    depth[~(np.isfinite(depth) & (depth > 0.0))] = np.nan
    return GeometryPrediction(
        depth_z_float32=depth,
        support_probability_float32=alpha,
        geometry_confidence_float32=np.zeros(weighted.shape, dtype=np.float32),
    )
