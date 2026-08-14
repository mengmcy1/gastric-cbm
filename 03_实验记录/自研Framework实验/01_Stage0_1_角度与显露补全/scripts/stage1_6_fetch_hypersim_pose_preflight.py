#!/usr/bin/env python3
"""Fetch only Hypersim pose metadata for HLP-TRAIN-01 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import requests

from stage1_fetch_hypersim_triplets import URL_TEMPLATE, VerifiedRemoteFile


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "dataset" not in handle:
            raise KeyError(f"dataset key missing: {path}")
        value = handle["dataset"]
        return {"shape": list(value.shape), "dtype": str(value.dtype)}


def required_entries(scene: str, camera: str) -> tuple[str, ...]:
    return (
        f"{scene}/_detail/{camera}/camera_keyframe_frame_indices.hdf5",
        f"{scene}/_detail/{camera}/camera_keyframe_orientations.hdf5",
        f"{scene}/_detail/{camera}/camera_keyframe_positions.hdf5",
        f"{scene}/_detail/metadata_scene.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--record-schema", default="stage1.6-hlp-train-01-pose-fetch-v1")
    parser.add_argument("--user-agent", default="2Dto3D-HLP-TRAIN-01/1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.record = args.record.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = [
        ("train", value) for value in config["train_candidates"]
    ] + [("val", value) for value in config["validation_candidates"]]
    scenes = [str(value["scene"]) for _, value in candidates]
    held_out = set(config["held_out_test_scenes"])
    if len(scenes) != len(set(scenes)) or held_out.intersection(scenes):
        raise ValueError("candidate scenes must be unique and disjoint from held-out test scenes")

    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    completed: list[dict[str, object]] = []
    archives: list[dict[str, object]] = []
    status = "pass"
    error_message = None
    try:
        for split, candidate in candidates:
            scene, camera = str(candidate["scene"]), str(candidate["camera"])
            remote = VerifiedRemoteFile(URL_TEMPLATE.format(scene=scene), session, timeout=args.timeout)
            entries = required_entries(scene, camera)
            with zipfile.ZipFile(remote) as archive:
                missing = sorted(set(entries).difference(archive.namelist()))
                if missing:
                    raise FileNotFoundError(f"{scene}: missing pose members: {missing}")
                for entry in entries:
                    destination = args.output_dir / entry
                    if destination.exists():
                        raise FileExistsError(f"refusing to overwrite: {destination}")
                    data = archive.read(entry)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    item: dict[str, object] = {
                        "split": split,
                        "scene": scene,
                        "camera": camera,
                        "entry": entry,
                        "path": str(destination.relative_to(Path.cwd())),
                        "bytes": len(data),
                        "sha256": sha256_bytes(data),
                    }
                    if destination.suffix == ".hdf5":
                        item["hdf5"] = validate_hdf5(destination)
                    completed.append(item)
            archives.append(
                {
                    "split": split,
                    "scene": scene,
                    "url": remote.url,
                    "content_length": remote.size,
                    "range_request_count": len(remote.requests),
                    "range_bytes_received": sum(int(value["bytes"]) for value in remote.requests),
                }
            )
    except Exception as error:
        status = "partial_failure" if completed else "failed"
        error_message = f"{type(error).__name__}: {error}"

    record = {
        "schema_version": args.record_schema,
        "status": status,
        "producer_machine_id": "linux5080",
        "config": str(args.config.relative_to(Path.cwd())),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "held_out_test_scenes": sorted(held_out),
        "archives": archives,
        "completed_files": completed,
        "summary": {
            "candidate_scenes": len(candidates),
            "train_scenes": sum(split == "train" for split, _ in candidates),
            "validation_scenes": sum(split == "val" for split, _ in candidates),
            "files_expected": 4 * len(candidates),
            "files_completed": len(completed),
            "bytes_completed": sum(int(value["bytes"]) for value in completed),
            "all_hdf5_readable": all(
                "hdf5" in value for value in completed if str(value["path"]).endswith(".hdf5")
            ),
            "held_out_overlap_count": len(held_out.intersection(scenes)),
        },
        "error": error_message,
        "safety": {"full_scene_zips_downloaded": False, "overwrites_refused": True, "files_deleted": False},
        "quality_boundary": "pose/file integrity only; 30-degree triplets and hidden evidence are not selected yet",
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"], "error": error_message}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
