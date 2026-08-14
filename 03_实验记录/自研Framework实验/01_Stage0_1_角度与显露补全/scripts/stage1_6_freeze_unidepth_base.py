#!/usr/bin/env python3
"""Freeze RGB-only UniDepth V1 base-depth inputs for HLP-TRAIN-01.

The evidence archives also contain label-only source depth.  This producer
deliberately reads only ``source_rgb_uint8`` and never supplies intrinsics to
UniDepth, making the no-depth-truth input boundary auditable in source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from flash3d_xformers_compat import (
    force_dino_reference_attention,
    install_unidepth_compatibility,
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


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unidepth-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--evidence-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--machine-id", default="linux5080")
    parser.add_argument("--expected-samples", type=int, default=10)
    parser.add_argument("--record-schema", default="stage1.6-hlp-train-01-frozen-base-depth-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("unidepth_repo", "weights", "evidence_record", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to reuse output directory or overwrite record")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen UniDepth producer")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.split(",")[0].strip() != str(args.physical_gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must begin with --physical-gpu-index so the GPU mapping is auditable"
        )

    evidence = json.loads(args.evidence_record.read_text(encoding="utf-8"))
    if evidence.get("status") != "pass" or len(evidence.get("results", [])) != args.expected_samples:
        raise RuntimeError("evidence record status/count does not match --expected-samples")
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)

    install_unidepth_compatibility()
    sys.path.insert(0, str(args.unidepth_repo))
    from unidepth.models import UniDepthV1

    model_config_path = args.unidepth_repo / "configs" / "config_v1_vitl14.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model = UniDepthV1(model_config)
    loaded = model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=False)
    allowed_missing = {
        "pixel_encoder.register_tokens",
        "pixel_encoder.norm.weight",
        "pixel_encoder.norm.bias",
    }
    if set(loaded.missing_keys) - allowed_missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"unexpected UniDepth weight mismatch: missing={loaded.missing_keys}, "
            f"unexpected={loaded.unexpected_keys}"
        )
    force_dino_reference_attention()
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    all_finite = True
    all_positive = True
    for item in evidence["results"]:
        evidence_path = Path(item["arrays"])
        if not evidence_path.is_absolute():
            evidence_path = Path.cwd() / evidence_path
        if sha256(evidence_path) != item["arrays_sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {evidence_path}")
        # Leakage boundary: do not access any other array in this archive.
        with np.load(evidence_path, allow_pickle=False) as archive:
            rgb = np.asarray(archive["source_rgb_uint8"], dtype=np.uint8)
        rgb_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device)
        sample_started = time.perf_counter()
        prediction = model.infer(tensor, intrinsics=None)
        depth = prediction["depth"].squeeze(0).squeeze(0).float().cpu().numpy()
        intrinsics = prediction["intrinsics"].squeeze(0).float().cpu().numpy()
        finite = bool(np.isfinite(depth).all() and np.isfinite(intrinsics).all())
        positive = bool((depth > 0).all())
        all_finite &= finite
        all_positive &= positive

        scene_id = f"{item['scene']}-{item['camera']}-{int(item['source_frame']):04d}"
        arrays_path = args.output_dir / f"{scene_id}-unidepth-v1-base.npz"
        preview_path = args.output_dir / f"{scene_id}-unidepth-v1-depth.png"
        np.savez_compressed(
            arrays_path,
            base_depth_z_float32=depth.astype(np.float32),
            predicted_intrinsics_3x3_float32=intrinsics.astype(np.float32),
            source_rgb_sha256_utf8=np.asarray(rgb_digest),
        )
        lo, hi = np.quantile(depth[np.isfinite(depth)], [0.02, 0.98])
        normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        Image.fromarray(np.uint8(np.rint((1 - normalized) * 255))).save(preview_path)
        results.append(
            {
                "split": item["split"],
                "scene": item["scene"],
                "camera": item["camera"],
                "source_frame": item["source_frame"],
                "source_rgb_sha256": rgb_digest,
                "shape_hw": list(depth.shape),
                "depth_m": {
                    "minimum": float(depth.min()),
                    "median": float(np.median(depth)),
                    "p99": float(np.quantile(depth, 0.99)),
                    "maximum": float(depth.max()),
                },
                "all_finite": finite,
                "all_positive": positive,
                "arrays": portable(arrays_path),
                "arrays_sha256": sha256(arrays_path),
                "preview": portable(preview_path),
                "preview_sha256": sha256(preview_path),
                "runtime_seconds": time.perf_counter() - sample_started,
            }
        )

    passed = len(results) == args.expected_samples and all_finite and all_positive
    record = {
        "schema_version": args.record_schema,
        "status": "pass" if passed else "failed",
        "producer_machine_id": args.machine_id,
        "physical_gpu_index": args.physical_gpu_index,
        "visible_cuda_devices": visible,
        "inputs": {
            "evidence_record": portable(args.evidence_record),
            "evidence_record_sha256": sha256(args.evidence_record),
            "allowed_array_keys": ["source_rgb_uint8"],
            "source_truth_depth_model_input": False,
            "source_truth_intrinsics_model_input": False,
        },
        "model": {
            "provider_id": "unidepth-v1-vitl14-rgb-only-frozen-v1",
            "repo": portable(args.unidepth_repo),
            "repo_commit": git_commit(args.unidepth_repo),
            "config": portable(model_config_path),
            "config_sha256": sha256(model_config_path),
            "weights": portable(args.weights),
            "weights_sha256": sha256(args.weights),
            "weights_bytes": args.weights.stat().st_size,
            "missing_weight_keys": loaded.missing_keys,
            "unexpected_weight_keys": loaded.unexpected_keys,
            "eval_mode": True,
            "gradient_enabled": False,
        },
        "results": results,
        "summary": {
            "samples": len(results),
            "expected_samples": args.expected_samples,
            "train_samples": sum(value["split"] == "train" for value in results),
            "validation_samples": sum(value["split"] == "val" for value in results),
            "all_finite": all_finite,
            "all_positive": all_positive,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "runtime_seconds": time.perf_counter() - started,
        },
        "quality_boundary": (
            "frozen RGB-only metric-depth input artifact; no source truth depth or truth intrinsics "
            "were passed to UniDepth; this record does not assess hidden-layer quality"
        ),
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
