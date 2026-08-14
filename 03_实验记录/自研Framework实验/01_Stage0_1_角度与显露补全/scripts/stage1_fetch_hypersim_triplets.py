#!/usr/bin/env python3
"""Fetch the exact HLP-GEO-01 files from official remote Hypersim ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import BinaryIO
import zipfile

import h5py
import requests


URL_TEMPLATE = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"


class VerifiedRemoteFile:
    def __init__(self, url: str, session: requests.Session, timeout: int = 180) -> None:
        response = session.head(url, timeout=timeout)
        response.raise_for_status()
        self.url = url
        self.session = session
        self.timeout = timeout
        self.size = int(response.headers["content-length"])
        self.offset = 0
        self.requests: list[dict[str, object]] = []

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self.offset = min(max(self.offset, 0), self.size)
        return self.offset

    def read(self, count: int | None = None) -> bytes:
        available = self.size - self.offset
        count = available if count is None or count < 0 else min(count, available)
        if count == 0:
            return b""
        start = self.offset
        end = start + count - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        expected_range = f"bytes {start}-{end}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise RuntimeError(
                f"invalid range response: HTTP {response.status_code}, {response.headers.get('Content-Range')!r}"
            )
        data = response.content
        if len(data) != count:
            raise RuntimeError(f"range length {len(data)} != {count}")
        self.offset += len(data)
        self.requests.append({"start": start, "end": end, "bytes": len(data)})
        return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def required_entries(triplet: dict[str, object]) -> list[str]:
    scene = str(triplet["scene"])
    camera = str(triplet["camera"])
    prefix = f"{scene}"
    entries = [
        f"{prefix}/_detail/{camera}/camera_keyframe_frame_indices.hdf5",
        f"{prefix}/_detail/{camera}/camera_keyframe_orientations.hdf5",
        f"{prefix}/_detail/{camera}/camera_keyframe_positions.hdf5",
        f"{prefix}/_detail/metadata_scene.csv",
    ]
    for frame in (triplet["source_frame"], triplet["left_frame"], triplet["right_frame"]):
        stem = f"frame.{int(frame):04d}"
        entries.extend(
            [
                f"{prefix}/images/scene_{camera}_final_hdf5/{stem}.color.hdf5",
                f"{prefix}/images/scene_{camera}_geometry_hdf5/{stem}.depth_meters.hdf5",
                f"{prefix}/images/scene_{camera}_geometry_hdf5/{stem}.position.hdf5",
            ]
        )
    return entries


def validate_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if "dataset" not in handle:
            raise KeyError(f"dataset key missing: {path}")
        dataset = handle["dataset"]
        return {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--record-schema", default="stage1.4-hlp-geo-01-fetch-v1")
    parser.add_argument("--user-agent", default="2Dto3D-HLP-GEO-01/1")
    parser.add_argument(
        "--quality-boundary",
        default="file integrity only; visibility masks and geometry metrics are not yet computed",
    )
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
    required_by_scene: dict[str, list[str]] = {}
    for triplet in config["triplets"]:
        scene = str(triplet["scene"])
        existing = required_by_scene.setdefault(scene, [])
        seen = set(existing)
        for entry in required_entries(triplet):
            if entry not in seen:
                existing.append(entry)
                seen.add(entry)
    args.output_dir.mkdir(parents=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    completed: list[dict[str, object]] = []
    archives: list[dict[str, object]] = []
    status = "pass"
    error_message: str | None = None
    try:
        for scene, required in required_by_scene.items():
            url = URL_TEMPLATE.format(scene=scene)
            remote = VerifiedRemoteFile(url, session, timeout=args.timeout)
            with zipfile.ZipFile(remote) as archive:
                names = set(archive.namelist())
                missing = sorted(set(required).difference(names))
                if missing:
                    raise FileNotFoundError(f"{scene} ZIP missing entries: {missing}")
                for entry in required:
                    destination = args.output_dir / entry
                    if destination.exists():
                        raise FileExistsError(f"refusing to overwrite: {destination}")
                    data = archive.read(entry)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    item: dict[str, object] = {
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
                    "scene": scene,
                    "url": url,
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
        "archives": archives,
        "completed_files": completed,
        "summary": {
            "triplets_requested": len(config["triplets"]),
            "scenes_requested": len(required_by_scene),
            "files_expected": sum(len(entries) for entries in required_by_scene.values()),
            "files_completed": len(completed),
            "bytes_completed": sum(int(value["bytes"]) for value in completed),
            "all_hdf5_readable": all("hdf5" in value for value in completed if str(value["path"]).endswith(".hdf5")),
        },
        "error": error_message,
        "safety": {"full_scene_zips_downloaded": False, "overwrites_refused": True, "files_deleted": False},
        "quality_boundary": args.quality_boundary,
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": record["summary"], "archives": archives, "error": error_message}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
