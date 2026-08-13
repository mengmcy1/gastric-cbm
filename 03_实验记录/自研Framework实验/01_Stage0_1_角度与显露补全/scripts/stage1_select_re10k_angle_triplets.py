#!/usr/bin/env python3
"""Select RealEstate10K source/left/right triplets by camera angle.

The official Flash3D split names targets by temporal offsets (5/10/random).
Those labels are deliberately ignored here: Stage 1.4 requires a measured
30-degree total range, i.e. approximately -15 and +15 degrees around a source.

Only camera metadata is read from the official RealEstate10K tarball.  Images
are not downloaded and the archive is not extracted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: str
    intrinsics: np.ndarray
    w2c: np.ndarray
    center_world: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(payload: bytes) -> tuple[str, list[Frame]]:
    lines = io.StringIO(payload.decode("utf-8")).read().splitlines()
    if not lines:
        raise ValueError("empty RealEstate10K metadata file")
    video_url = lines[0].strip()
    frames: list[Frame] = []
    for index, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) != 19:
            raise ValueError(f"frame {index}: expected 19 columns, got {len(parts)}")
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
        intrinsics = values[:6]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = values[6:].reshape(3, 4)
        c2w = np.linalg.inv(w2c)
        frames.append(
            Frame(
                index=index,
                timestamp=parts[0],
                intrinsics=intrinsics,
                w2c=w2c,
                center_world=c2w[:3, 3],
            )
        )
    return video_url, frames


def relative_horizontal_angle_deg(source: Frame, target: Frame) -> float:
    """Signed target optical-axis angle expressed in the source camera frame."""
    source_r_w2c = source.w2c[:3, :3]
    target_forward_world = target.w2c[:3, :3].T[:, 2]
    target_forward_source = source_r_w2c @ target_forward_world
    return math.degrees(math.atan2(target_forward_source[0], target_forward_source[2]))


def intrinsics_relative_change(source: Frame, target: Frame) -> float:
    denom = np.maximum(np.abs(source.intrinsics[:4]), 1e-12)
    return float(np.max(np.abs(target.intrinsics[:4] - source.intrinsics[:4]) / denom))


def find_triplets(
    sequence: str,
    video_url: str,
    frames: list[Frame],
    target_deg: float,
    tolerance_deg: float,
    max_intrinsics_change: float,
) -> tuple[list[dict], dict]:
    candidates: list[dict] = []
    max_negative = 0.0
    max_positive = 0.0
    for source in frames:
        eligible = []
        for target in frames:
            if target.index == source.index:
                continue
            intrinsics_change = intrinsics_relative_change(source, target)
            if intrinsics_change > max_intrinsics_change:
                continue
            angle = relative_horizontal_angle_deg(source, target)
            translation = float(np.linalg.norm(target.center_world - source.center_world))
            eligible.append((target, angle, translation, intrinsics_change))
            max_negative = min(max_negative, angle)
            max_positive = max(max_positive, angle)

        negative = [item for item in eligible if item[1] < 0]
        positive = [item for item in eligible if item[1] > 0]
        if not negative or not positive:
            continue
        left = min(negative, key=lambda item: abs(item[1] + target_deg))
        right = min(positive, key=lambda item: abs(item[1] - target_deg))
        left_error = abs(left[1] + target_deg)
        right_error = abs(right[1] - target_deg)
        if left_error > tolerance_deg or right_error > tolerance_deg:
            continue
        candidates.append(
            {
                "sequence": sequence,
                "video_url": video_url,
                "source_index": source.index,
                "source_timestamp": source.timestamp,
                "left_index": left[0].index,
                "left_timestamp": left[0].timestamp,
                "left_angle_deg": left[1],
                "left_translation": left[2],
                "left_intrinsics_relative_change": left[3],
                "right_index": right[0].index,
                "right_timestamp": right[0].timestamp,
                "right_angle_deg": right[1],
                "right_translation": right[2],
                "right_intrinsics_relative_change": right[3],
                "total_angle_deg": right[1] - left[1],
                "angle_error_sum_deg": left_error + right_error,
            }
        )
    candidates.sort(key=lambda item: (item["angle_error_sum_deg"], -min(item["left_translation"], item["right_translation"])))
    return candidates, {
        "sequence": sequence,
        "frame_count": len(frames),
        "max_negative_angle_deg": max_negative,
        "max_positive_angle_deg": max_positive,
        "qualifying_source_count": len(candidates),
    }


def read_sequence_ids(split_file: Path) -> list[str]:
    sequence_ids = []
    seen = set()
    for line in split_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sequence = line.split()[0]
        if sequence not in seen:
            seen.add(sequence)
            sequence_ids.append(sequence)
    return sequence_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-tar", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--archive-split", choices=("train", "test"), default="test")
    parser.add_argument("--target-deg", type=float, default=15.0)
    parser.add_argument("--tolerance-deg", type=float, default=2.5)
    parser.add_argument("--max-intrinsics-change", type=float, default=0.02)
    parser.add_argument("--max-triplets-per-sequence", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    wanted = set(read_sequence_ids(args.split_file))
    payloads: dict[str, bytes] = {}
    prefix = f"RealEstate10K/{args.archive_split}/"
    with tarfile.open(args.metadata_tar, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix) or not member.name.endswith(".txt"):
                continue
            sequence = Path(member.name).stem
            if sequence not in wanted:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member.name}")
            payloads[sequence] = extracted.read()

    all_candidates = []
    sequence_stats = []
    missing = sorted(wanted - payloads.keys())
    for sequence in sorted(payloads):
        video_url, frames = parse_metadata(payloads[sequence])
        candidates, stats = find_triplets(
            sequence,
            video_url,
            frames,
            args.target_deg,
            args.tolerance_deg,
            args.max_intrinsics_change,
        )
        sequence_stats.append(stats)
        all_candidates.extend(candidates[: args.max_triplets_per_sequence])
    all_candidates.sort(key=lambda item: (item["angle_error_sum_deg"], item["sequence"], item["source_index"]))

    result = {
        "schema_version": "stage1.4-re10k-angle-triplets-v1",
        "angle_semantics": {
            "total_range_deg": 2.0 * args.target_deg,
            "endpoint_target_deg": [-args.target_deg, args.target_deg],
            "tolerance_deg": args.tolerance_deg,
            "measurement": "signed target optical-axis angle in source camera coordinates",
            "official_temporal_offset_labels_used_as_angles": False,
        },
        "inputs": {
            "metadata_tar": str(args.metadata_tar),
            "metadata_tar_sha256": sha256(args.metadata_tar),
            "split_file": str(args.split_file),
            "split_file_sha256": sha256(args.split_file),
            "archive_split": args.archive_split,
            "max_intrinsics_relative_change": args.max_intrinsics_change,
        },
        "summary": {
            "requested_sequence_count": len(wanted),
            "found_sequence_count": len(payloads),
            "missing_sequence_count": len(missing),
            "sequences_with_triplets": sum(stat["qualifying_source_count"] > 0 for stat in sequence_stats),
            "saved_triplet_count": len(all_candidates),
        },
        "missing_sequences": missing,
        "sequence_stats": sequence_stats,
        "triplets": all_candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "sequence", "source_index", "source_timestamp", "left_index", "left_timestamp",
        "left_angle_deg", "left_translation", "right_index", "right_timestamp",
        "right_angle_deg", "right_translation", "total_angle_deg", "angle_error_sum_deg",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_candidates)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
