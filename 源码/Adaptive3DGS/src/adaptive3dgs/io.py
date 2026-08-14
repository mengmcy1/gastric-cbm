"""Safe, portable CLB v1 manifest and NPZ I/O."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .clb import Camera, CanonicalLayerBundle, CanonicalLayerPatch, SupportType
from .validation import ValidationError, validate_patch


SCHEMA_VERSION = "1.0-canonical-layer-bundle"
ARRAY_NAMES = (
    "rgb_uint8",
    "depth_z_float32",
    "alpha_float32",
    "valid_mask_uint8",
    "provenance_uint8",
    "geometry_confidence_float32",
    "appearance_confidence_float32",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValidationError(f"{name} must be a finite {shape} matrix")
    return matrix


def _camera_from_manifest(camera_id: str, value: dict[str, Any]) -> Camera:
    known = {"angle_deg", "intrinsics_3x3", "extrinsics_world_to_camera_4x4"}
    return Camera(
        camera_id=camera_id,
        intrinsics_3x3_float64=_matrix(value.get("intrinsics_3x3"), (3, 3), f"{camera_id} intrinsics"),
        world_to_camera_4x4_float64=_matrix(
            value.get("extrinsics_world_to_camera_4x4"), (4, 4), f"{camera_id} extrinsics"
        ),
        angle_deg=float(value["angle_deg"]) if value.get("angle_deg") is not None else None,
        metadata={key: item for key, item in value.items() if key not in known},
    )


def load_bundle(manifest_path: str | Path, *, validate: bool = True) -> CanonicalLayerBundle:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"unsupported CLB schema: {manifest.get('schema_version')!r}")
    camera_values = manifest.get("cameras")
    if not isinstance(camera_values, dict) or not camera_values:
        raise ValidationError("manifest cameras must be a non-empty object")
    cameras = {
        str(camera_id): _camera_from_manifest(str(camera_id), value)
        for camera_id, value in camera_values.items()
    }
    layer_values = manifest.get("layers")
    if not isinstance(layer_values, list) or not layer_values:
        raise ValidationError("manifest layers must be a non-empty list")
    patches: list[CanonicalLayerPatch] = []
    deployment = bool(manifest.get("deployment", False))
    for layer in layer_values:
        patch_id = str(layer.get("layer_id", ""))
        if not SAFE_ID.fullmatch(patch_id):
            raise ValidationError(f"unsafe or empty layer_id: {patch_id!r}")
        try:
            support_type = SupportType(layer["support_type"])
            anchor_camera = str(layer["anchor_camera"])
            camera = cameras[anchor_camera]
            relative_asset = Path(layer["arrays_npz"])
        except (KeyError, ValueError) as error:
            raise ValidationError(f"{patch_id}: invalid support or anchor camera") from error
        if relative_asset.is_absolute() or ".." in relative_asset.parts:
            raise ValidationError(f"{patch_id}: arrays_npz must stay inside the bundle directory")
        arrays_path = (manifest_path.parent / relative_asset).resolve()
        if arrays_path.parent != manifest_path.parent:
            raise ValidationError(f"{patch_id}: nested or escaped NPZ paths are not allowed")
        if not arrays_path.is_file():
            raise ValidationError(f"{patch_id}: NPZ not found: {arrays_path}")
        expected_hash = layer.get("arrays_sha256")
        if expected_hash is not None and _sha256(arrays_path) != str(expected_hash).lower():
            raise ValidationError(f"{patch_id}: NPZ SHA256 mismatch")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            missing = sorted(set(ARRAY_NAMES).difference(arrays.files))
            if missing:
                raise ValidationError(f"{patch_id}: missing arrays {missing}")
            values = {name: arrays[name] for name in ARRAY_NAMES}
        known = {"layer_id", "support_type", "anchor_camera", "arrays_npz", "arrays_sha256"}
        patch = CanonicalLayerPatch(
            patch_id=patch_id,
            support_type=support_type,
            intrinsics_3x3_float64=camera.intrinsics_3x3_float64.copy(),
            world_to_camera_4x4_float64=camera.world_to_camera_4x4_float64.copy(),
            metadata={
                "anchor_camera": anchor_camera,
                **{key: item for key, item in layer.items() if key not in known},
            },
            **values,
        )
        if validate:
            validate_patch(patch, deployment=deployment)
        patches.append(patch)
    known_top = {"schema_version", "bundle_id", "experiment_id", "deployment", "coordinate_convention", "cameras", "layers"}
    bundle_id = str(manifest.get("bundle_id") or manifest.get("experiment_id") or "")
    if not bundle_id:
        raise ValidationError("manifest requires bundle_id or legacy experiment_id")
    return CanonicalLayerBundle(
        bundle_id=bundle_id,
        deployment=deployment,
        coordinate_convention=dict(manifest.get("coordinate_convention", {})),
        cameras=cameras,
        patches=patches,
        metadata={key: item for key, item in manifest.items() if key not in known_top},
    )


def save_bundle(bundle: CanonicalLayerBundle, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    if not SAFE_ID.fullmatch(bundle.bundle_id):
        raise ValidationError(f"unsafe or empty bundle_id: {bundle.bundle_id!r}")
    if not bundle.cameras or not bundle.patches:
        raise ValidationError("bundle requires cameras and patches")
    output_dir.mkdir(parents=True)
    camera_manifest: dict[str, Any] = {}
    for camera_id, camera in bundle.cameras.items():
        if not SAFE_ID.fullmatch(camera_id) or camera.camera_id != camera_id:
            raise ValidationError(f"unsafe or inconsistent camera_id: {camera_id!r}")
        _matrix(camera.intrinsics_3x3_float64, (3, 3), f"{camera_id} intrinsics")
        _matrix(camera.world_to_camera_4x4_float64, (4, 4), f"{camera_id} extrinsics")
        camera_manifest[camera_id] = {
            **dict(camera.metadata),
            "angle_deg": camera.angle_deg,
            "intrinsics_3x3": camera.intrinsics_3x3_float64.tolist(),
            "extrinsics_world_to_camera_4x4": camera.world_to_camera_4x4_float64.tolist(),
        }
    layer_manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for patch in bundle.patches:
        if not SAFE_ID.fullmatch(patch.patch_id) or patch.patch_id in seen:
            raise ValidationError(f"unsafe or duplicate patch_id: {patch.patch_id!r}")
        seen.add(patch.patch_id)
        anchor_camera = str(patch.metadata.get("anchor_camera", ""))
        if anchor_camera not in bundle.cameras:
            raise ValidationError(f"{patch.patch_id}: unknown anchor_camera {anchor_camera!r}")
        camera = bundle.cameras[anchor_camera]
        if not np.array_equal(patch.intrinsics_3x3_float64, camera.intrinsics_3x3_float64):
            raise ValidationError(f"{patch.patch_id}: patch intrinsics differ from anchor camera")
        if not np.array_equal(patch.world_to_camera_4x4_float64, camera.world_to_camera_4x4_float64):
            raise ValidationError(f"{patch.patch_id}: patch extrinsics differ from anchor camera")
        validate_patch(patch, deployment=bundle.deployment)
        arrays_name = f"{patch.patch_id}.npz"
        arrays_path = output_dir / arrays_name
        np.savez_compressed(arrays_path, **{name: getattr(patch, name) for name in ARRAY_NAMES})
        reserved = {"anchor_camera", "arrays_npz", "arrays_sha256", "layer_id", "support_type"}
        layer_manifest.append(
            {
                **{key: item for key, item in patch.metadata.items() if key not in reserved},
                "layer_id": patch.patch_id,
                "support_type": patch.support_type.value,
                "anchor_camera": anchor_camera,
                "arrays_npz": arrays_name,
                "arrays_sha256": _sha256(arrays_path),
            }
        )
    reserved_top = {"schema_version", "bundle_id", "experiment_id", "deployment", "coordinate_convention", "cameras", "layers"}
    manifest = {
        **{key: item for key, item in bundle.metadata.items() if key not in reserved_top},
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "deployment": bundle.deployment,
        "coordinate_convention": dict(bundle.coordinate_convention),
        "cameras": camera_manifest,
        "layers": layer_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path
