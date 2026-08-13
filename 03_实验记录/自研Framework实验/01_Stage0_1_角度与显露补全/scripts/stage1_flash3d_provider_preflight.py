#!/usr/bin/env python3
"""CPU-only preflight for the official Flash3D Stage 1.4 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


EXPECTED_CHECKPOINT_SHA256 = "4d717b5543941069fc120733ab4001e6d11fe73ea094d6c6054c5e8b48922420"
EXPECTED_REPO_COMMIT = "a71c9b92b07a76cf944f2ed894f384f0891a1960"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--angle-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_hash = sha256(args.checkpoint)
    repo_commit = git_commit(args.repo)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
    keys = tuple(state_dict)
    depth_keys = [key for key in keys if ".depth." in key or "models.depth" in key]
    gaussian_decoder_keys = [key for key in keys if "gauss_decoder_" in key]
    layer_0_keys = [key for key in gaussian_decoder_keys if "gauss_decoder_0" in key]
    layer_1_keys = [key for key in gaussian_decoder_keys if "gauss_decoder_1" in key]
    angle_record = json.loads(args.angle_record.read_text(encoding="utf-8"))

    checks = {
        "checkpoint_sha256_matches": checkpoint_hash == EXPECTED_CHECKPOINT_SHA256,
        "repo_commit_matches": repo_commit == EXPECTED_REPO_COMMIT,
        "checkpoint_has_model_state": bool(state_dict),
        "checkpoint_has_depth_increment_decoder": bool(depth_keys),
        "checkpoint_has_gaussian_decoder_0": bool(layer_0_keys),
        "checkpoint_has_gaussian_decoder_1": bool(layer_1_keys),
        "angle_record_has_30deg_triplets": angle_record["summary"]["saved_triplet_count"] > 0,
        "official_temporal_offsets_not_used_as_angles": not angle_record["angle_semantics"][
            "official_temporal_offset_labels_used_as_angles"
        ],
    }
    result = {
        "schema_version": "stage1.4-flash3d-provider-preflight-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "candidate_scope": "occlusion_hidden_inside_source_frustum_only",
        "outside_source_fov_supported": False,
        "independent_confidence_native": False,
        "opacity_must_not_be_used_as_confidence": True,
        "inputs": {
            "repo": str(args.repo),
            "repo_commit": repo_commit,
            "checkpoint": str(args.checkpoint),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "angle_record": str(args.angle_record),
            "angle_record_sha256": sha256(args.angle_record),
        },
        "checkpoint": {
            "top_level_keys": list(checkpoint) if isinstance(checkpoint, dict) else [],
            "model_tensor_count": len(state_dict),
            "depth_decoder_key_count": len(depth_keys),
            "gaussian_decoder_0_key_count": len(layer_0_keys),
            "gaussian_decoder_1_key_count": len(layer_1_keys),
        },
        "angle_data": angle_record["summary"],
        "checks": checks,
        "next_required_before_inference": [
            "create isolated Blackwell-compatible Flash3D environment",
            "download selected source/target RGB frames for HLP_APP_01",
            "implement Flash3D output to CLB adapter",
            "define confidence calibration independent of opacity",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
