#!/usr/bin/env python3
"""Prepare RGB-only evidence for new endpoint sources in the Stage 1.9 pair pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-config", type=Path, required=True)
    parser.add_argument("--existing-base-record", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def tonemap(color_linear: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    color = np.asarray(color_linear, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(color).all(axis=2)
    brightness = 0.3 * color[:, :, 0] + 0.59 * color[:, :, 1] + 0.11 * color[:, :, 2]
    percentile = float(np.percentile(brightness[valid], 90)) if np.any(valid) else 0.0
    if 0 < percentile < 0.0001:
        scale = 0.0
    elif percentile == 0:
        scale = 1.0
    else:
        scale = 0.8 ** 2.2 / percentile
    mapped = np.power(np.maximum(scale * color, 0), 1.0 / 2.2)
    return np.rint(np.clip(mapped, 0, 1) * 255).astype(np.uint8), scale


def main() -> int:
    args = parse_args()
    for name in ("pair_config", "existing_base_record", "dataset_root", "output_dir", "record"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite endpoint RGB evidence")
    pairs = json.loads(args.pair_config.read_text(encoding="utf-8"))
    existing = json.loads(args.existing_base_record.read_text(encoding="utf-8"))
    if pairs.get("status") != "pass" or existing.get("status") != "pass":
        raise ValueError("pair config and existing BaseDepth record must pass")
    existing_keys = {(x["scene"], x["camera"], int(x["source_frame"])) for x in existing["results"]}
    source_items = {}
    for pair in pairs["pairs"]:
        key = (pair["scene"], pair["camera"], int(pair["source_frame"]))
        source_items.setdefault(key, pair["split"])
    missing_keys = sorted(set(source_items).difference(existing_keys))
    if len(missing_keys) != 95:
        raise RuntimeError(f"expected 95 new endpoint sources, got {len(missing_keys)}")

    args.output_dir.mkdir(parents=True)
    results = []
    for scene, camera, frame in missing_keys:
        color_path = args.dataset_root / scene / "images" / f"scene_{camera}_final_hdf5" / f"frame.{frame:04d}.color.hdf5"
        depth_path = args.dataset_root / scene / "images" / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.depth_meters.hdf5"
        color = load_hdf5(color_path).astype(np.float32)
        depth = load_hdf5(depth_path).astype(np.float32)
        rgb, scale = tonemap(color, np.isfinite(depth) & (depth > 0))
        arrays = args.output_dir / f"{scene}-{camera}-{frame:04d}-source-rgb.npz"
        np.savez_compressed(arrays, source_rgb_uint8=rgb)
        results.append({
            "split": source_items[(scene, camera, frame)],
            "scene": scene,
            "camera": camera,
            "source_frame": frame,
            "source_role": "endpoint",
            "source_rgb_uint8_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "tonemap_scale": float(scale),
            "arrays": portable(arrays),
            "arrays_sha256": sha256(arrays),
            "source_truth_depth_model_input": False,
        })
    record = {
        "schema_version": "stage1.9-endpoint-source-rgb-evidence-v1",
        "status": "pass" if len(results) == 95 else "failed",
        "producer_machine_id": args.machine_id,
        "inputs": {
            "pair_config": portable(args.pair_config), "pair_config_sha256": sha256(args.pair_config),
            "existing_base_record": portable(args.existing_base_record),
            "existing_base_record_sha256": sha256(args.existing_base_record),
            "dataset_root": portable(args.dataset_root),
        },
        "results": results,
        "summary": {
            "new_endpoint_sources": len(results),
            "train_sources": sum(x["split"] == "train" for x in results),
            "validation_sources": sum(x["split"] == "val" for x in results),
            "existing_center_sources_reused": len(existing_keys),
            "all_pair_sources_accounted_for": set(source_items) == existing_keys.union(missing_keys),
        },
        "leakage_boundary": "Depth is read only to reproduce the frozen Hypersim tonemap validity mask. Each output archive contains source_rgb_uint8 only; no truth depth, target RGB, or target camera is exposed to UniDepth.",
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "summary": record["summary"]}, indent=2))
    return 0 if record["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
