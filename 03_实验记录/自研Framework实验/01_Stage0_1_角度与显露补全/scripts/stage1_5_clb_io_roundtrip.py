#!/usr/bin/env python3
"""Round-trip an existing CLB through Adaptive3DGS and cross-check the legacy validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ARRAY_FIELDS = (
    "rgb_uint8",
    "depth_z_float32",
    "alpha_float32",
    "valid_mask_uint8",
    "provenance_uint8",
    "geometry_confidence_float32",
    "appearance_confidence_float32",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive3dgs-src", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-validator", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("adaptive3dgs_src", "source_manifest", "output_dir", "legacy_validator", "protocol", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    sys.path.insert(0, str(args.adaptive3dgs_src))
    from adaptive3dgs import load_bundle, save_bundle

    source = load_bundle(args.source_manifest)
    output_manifest = save_bundle(source, args.output_dir)
    restored = load_bundle(output_manifest)
    gates: dict[str, bool] = {
        "bundle_id_exact": source.bundle_id == restored.bundle_id,
        "deployment_exact": source.deployment == restored.deployment,
        "coordinate_convention_exact": source.coordinate_convention == restored.coordinate_convention,
        "camera_ids_exact": set(source.cameras) == set(restored.cameras),
        "patch_ids_exact": [value.patch_id for value in source.patches]
        == [value.patch_id for value in restored.patches],
    }
    for camera_id, camera in source.cameras.items():
        restored_camera = restored.cameras[camera_id]
        gates[f"camera_{camera_id}_intrinsics_exact"] = np.array_equal(
            camera.intrinsics_3x3_float64, restored_camera.intrinsics_3x3_float64
        )
        gates[f"camera_{camera_id}_extrinsics_exact"] = np.array_equal(
            camera.world_to_camera_4x4_float64, restored_camera.world_to_camera_4x4_float64
        )
        gates[f"camera_{camera_id}_angle_exact"] = camera.angle_deg == restored_camera.angle_deg
    restored_by_id = {value.patch_id: value for value in restored.patches}
    for patch in source.patches:
        roundtrip_patch = restored_by_id[patch.patch_id]
        gates[f"patch_{patch.patch_id}_support_exact"] = patch.support_type == roundtrip_patch.support_type
        gates[f"patch_{patch.patch_id}_metadata_exact"] = patch.metadata == roundtrip_patch.metadata
        for field in ARRAY_FIELDS:
            gates[f"patch_{patch.patch_id}_{field}_exact"] = np.array_equal(
                getattr(patch, field), getattr(roundtrip_patch, field), equal_nan=True
            )
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    hash_gates = []
    for layer in manifest["layers"]:
        arrays_path = output_manifest.parent / layer["arrays_npz"]
        hash_gates.append(sha256(arrays_path) == layer.get("arrays_sha256"))
    gates["all_written_npz_hashes_match"] = all(hash_gates)
    legacy = subprocess.run(
        [
            sys.executable,
            str(args.legacy_validator),
            str(output_manifest),
            "--protocol",
            str(args.protocol),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gates["legacy_validator_pass"] = legacy.returncode == 0
    status = "pass" if all(gates.values()) else "fail"
    record = {
        "schema_version": "stage1.5-clb-io-roundtrip-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "source_manifest": portable(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "output_manifest": portable(output_manifest),
        "output_manifest_sha256": sha256(output_manifest),
        "summary": {
            "cameras": len(source.cameras),
            "patches": len(source.patches),
            "valid_pixels": sum(int(value.valid_mask_uint8.sum()) for value in source.patches),
            "exact_roundtrip_gate_count": len(gates),
            "all_exact_and_cross_compatible": all(gates.values()),
        },
        "gates": gates,
        "legacy_validator": {
            "returncode": legacy.returncode,
            "stdout": legacy.stdout,
            "stderr": legacy.stderr,
        },
        "safety": {
            "output_reuse_refused": True,
            "record_overwrite_refused": True,
            "files_deleted": False,
        },
        "quality_boundary": "CLB serialization and validator compatibility only; no provider quality claim",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"]}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
