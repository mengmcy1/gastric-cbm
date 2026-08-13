from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "rgb_uint8",
    "depth_z_float32",
    "alpha_float32",
    "valid_mask_uint8",
    "provenance_uint8",
    "geometry_confidence_float32",
    "appearance_confidence_float32",
}
SUPPORT_TYPES = {"observed_surface", "occlusion_hidden", "outside_source_fov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Canonical Layer Bundle manifest and arrays.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_asset(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def validate_layer(manifest_path: Path, layer: dict, deployment: bool) -> dict:
    layer_id = str(layer.get("layer_id", "<missing>"))
    require(layer.get("support_type") in SUPPORT_TYPES, f"{layer_id}: invalid support_type")
    require(layer.get("anchor_camera") in {"center", "left", "right"}, f"{layer_id}: invalid anchor")
    require("arrays_npz" in layer, f"{layer_id}: arrays_npz missing")
    arrays_path = resolve_asset(manifest_path, layer["arrays_npz"])
    require(arrays_path.is_file(), f"{layer_id}: NPZ not found: {arrays_path}")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        require(REQUIRED_ARRAYS.issubset(arrays.files), f"{layer_id}: missing arrays {sorted(REQUIRED_ARRAYS - set(arrays.files))}")
        rgb = arrays["rgb_uint8"]
        depth = arrays["depth_z_float32"]
        alpha = arrays["alpha_float32"]
        valid = arrays["valid_mask_uint8"]
        provenance = arrays["provenance_uint8"]
        geometry_confidence = arrays["geometry_confidence_float32"]
        appearance_confidence = arrays["appearance_confidence_float32"]

        require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, f"{layer_id}: rgb must be uint8 HxWx3")
        spatial_shape = rgb.shape[:2]
        for name, value in {
            "depth": depth,
            "alpha": alpha,
            "valid": valid,
            "provenance": provenance,
            "geometry_confidence": geometry_confidence,
            "appearance_confidence": appearance_confidence,
        }.items():
            require(value.shape == spatial_shape, f"{layer_id}: {name} shape mismatch")
        require(depth.dtype == np.float32, f"{layer_id}: depth must be float32")
        require(alpha.dtype == np.float32, f"{layer_id}: alpha must be float32")
        require(valid.dtype == np.uint8, f"{layer_id}: valid must be uint8")
        require(provenance.dtype == np.uint8, f"{layer_id}: provenance must be uint8")
        require(geometry_confidence.dtype == np.float32, f"{layer_id}: geometry confidence must be float32")
        require(appearance_confidence.dtype == np.float32, f"{layer_id}: appearance confidence must be float32")

        valid_bool = valid != 0
        require(np.all((alpha >= 0.0) & (alpha <= 1.0)), f"{layer_id}: alpha outside [0,1]")
        for name, value in {
            "geometry_confidence": geometry_confidence,
            "appearance_confidence": appearance_confidence,
        }.items():
            require(np.all(np.isfinite(value)), f"{layer_id}: {name} contains non-finite values")
            require(np.all((value >= 0.0) & (value <= 1.0)), f"{layer_id}: {name} outside [0,1]")
        require(np.all(np.isfinite(depth[valid_bool]) & (depth[valid_bool] > 0.0)), f"{layer_id}: invalid valid-depth values")
        require(np.all(provenance[valid_bool] != 0), f"{layer_id}: valid pixels with invalid provenance")
        require(np.all(np.isin(provenance, np.arange(0, 7, dtype=np.uint8))), f"{layer_id}: unknown provenance")
        if layer["support_type"] == "observed_surface":
            require(np.all(np.isin(provenance[valid_bool], [1, 2])), f"{layer_id}: observed layer contains predicted/oracle provenance")
        if deployment:
            require(not np.any(provenance[valid_bool] == 6), f"{layer_id}: synthetic oracle forbidden in deployment")

    return {
        "layer_id": layer_id,
        "support_type": layer["support_type"],
        "anchor_camera": layer["anchor_camera"],
        "resolution_hw": list(spatial_shape),
        "valid_pixels": int(valid_bool.sum()),
    }


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    protocol = load_json(args.protocol)
    require(protocol.get("schema_version") == "1.0-canonical-layer-bundle-protocol", "unsupported protocol")
    require(manifest.get("schema_version") == "1.0-canonical-layer-bundle", "unsupported manifest")
    require(manifest.get("coordinate_convention") == protocol["coordinate_convention"], "coordinate convention mismatch")
    require(isinstance(manifest.get("cameras"), dict), "cameras missing")
    for camera_name in {"center", "left", "right"}:
        camera = manifest["cameras"].get(camera_name)
        require(isinstance(camera, dict), f"camera missing: {camera_name}")
        require(len(camera.get("intrinsics_3x3", [])) == 3, f"{camera_name}: invalid intrinsics")
        require(len(camera.get("extrinsics_world_to_camera_4x4", [])) == 4, f"{camera_name}: invalid extrinsics")
    layers = manifest.get("layers")
    require(isinstance(layers, list) and layers, "layers must be a non-empty list")
    deployment = bool(manifest.get("deployment", False))
    results = [validate_layer(args.manifest, layer, deployment) for layer in layers]
    print(json.dumps({"status": "pass", "deployment": deployment, "layers": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
