"""Hard validation gates applied before fusion or Gaussian spawning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clb import CanonicalLayerPatch, Provenance, SupportType
from .providers import ProviderResult


class ValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationReport:
    patch_count: int
    valid_pixel_count: int
    support_types: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_patch(patch: CanonicalLayerPatch, *, deployment: bool = True) -> int:
    height, width = patch.image_shape
    _require(height > 0 and width > 0, f"{patch.patch_id}: empty image")
    _require(patch.rgb_uint8.shape == (height, width, 3), f"{patch.patch_id}: rgb shape mismatch")
    _require(patch.rgb_uint8.dtype == np.uint8, f"{patch.patch_id}: rgb must be uint8")
    for name in (
        "depth_z_float32",
        "alpha_float32",
        "geometry_confidence_float32",
        "appearance_confidence_float32",
    ):
        value = getattr(patch, name)
        _require(value.shape == (height, width), f"{patch.patch_id}: {name} shape mismatch")
        _require(value.dtype == np.float32, f"{patch.patch_id}: {name} must be float32")
    for name in ("valid_mask_uint8", "provenance_uint8"):
        value = getattr(patch, name)
        _require(value.shape == (height, width), f"{patch.patch_id}: {name} shape mismatch")
        _require(value.dtype == np.uint8, f"{patch.patch_id}: {name} must be uint8")
    _require(patch.intrinsics_3x3_float64.shape == (3, 3), f"{patch.patch_id}: intrinsics must be 3x3")
    _require(patch.intrinsics_3x3_float64.dtype == np.float64, f"{patch.patch_id}: intrinsics must be float64")
    _require(patch.world_to_camera_4x4_float64.shape == (4, 4), f"{patch.patch_id}: extrinsics must be 4x4")
    _require(patch.world_to_camera_4x4_float64.dtype == np.float64, f"{patch.patch_id}: extrinsics must be float64")
    _require(np.isfinite(patch.intrinsics_3x3_float64).all(), f"{patch.patch_id}: non-finite intrinsics")
    _require(np.isfinite(patch.world_to_camera_4x4_float64).all(), f"{patch.patch_id}: non-finite extrinsics")
    _require(np.isin(patch.valid_mask_uint8, [0, 1]).all(), f"{patch.patch_id}: valid mask must be binary")
    valid = patch.valid_mask_uint8.astype(bool)
    _require(np.isfinite(patch.depth_z_float32[valid]).all(), f"{patch.patch_id}: non-finite valid depth")
    _require((patch.depth_z_float32[valid] > 0).all(), f"{patch.patch_id}: valid depth must be positive Z")
    for name in ("alpha_float32", "geometry_confidence_float32", "appearance_confidence_float32"):
        value = getattr(patch, name)
        _require(np.isfinite(value).all(), f"{patch.patch_id}: non-finite {name}")
        _require(((value >= 0) & (value <= 1)).all(), f"{patch.patch_id}: {name} outside [0,1]")
    _require((patch.provenance_uint8[valid] != Provenance.INVALID).all(), f"{patch.patch_id}: valid pixel has invalid provenance")
    _require(
        np.isin(patch.provenance_uint8, [value.value for value in Provenance]).all(),
        f"{patch.patch_id}: unknown provenance value",
    )
    if patch.support_type is SupportType.OBSERVED_SURFACE:
        _require(
            np.isin(
                patch.provenance_uint8[valid],
                [Provenance.OBSERVED_CENTER, Provenance.REPROJECTED_OBSERVATION],
            ).all(),
            f"{patch.patch_id}: observed surface contains predicted provenance",
        )
    if deployment:
        _require(
            not np.any(patch.provenance_uint8[valid] == Provenance.SYNTHETIC_ORACLE),
            f"{patch.patch_id}: synthetic oracle is forbidden in deployment",
        )
    return int(valid.sum())


def validate_result(
    result: ProviderResult,
    *,
    expected_support: SupportType,
    deployment: bool = True,
) -> ValidationReport:
    _require(bool(result.provider_id), "provider_id is empty")
    _require(bool(result.patches), f"{result.provider_id}: provider returned no patches")
    patch_ids = [patch.patch_id for patch in result.patches]
    _require(len(patch_ids) == len(set(patch_ids)), f"{result.provider_id}: duplicate patch_id")
    _require(
        all(patch.support_type is expected_support for patch in result.patches),
        f"{result.provider_id}: provider returned a patch outside {expected_support.value}",
    )
    valid_pixels = sum(validate_patch(patch, deployment=deployment) for patch in result.patches)
    return ValidationReport(
        patch_count=len(result.patches),
        valid_pixel_count=valid_pixels,
        support_types=tuple(sorted({patch.support_type.value for patch in result.patches})),
    )
