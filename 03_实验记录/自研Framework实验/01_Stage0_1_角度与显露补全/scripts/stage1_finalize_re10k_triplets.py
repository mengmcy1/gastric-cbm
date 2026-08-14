#!/usr/bin/env python3
"""Validate and merge targeted RealEstate10K triplet download receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_record_path(repo: Path, record: dict) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else repo / path


def validate_file(repo: Path, record: dict) -> Path:
    path = resolve_record_path(repo, record)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"byte-size mismatch: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-successes", type=int, default=5)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    successes: list[dict] = []
    failures: list[dict] = []
    receipt_records: list[dict] = []
    seen: set[str] = set()
    for receipt_arg in args.receipt:
        receipt_path = receipt_arg.resolve()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_records.append(
            {
                "path": str(receipt_path.relative_to(repo)).replace("\\", "/"),
                "sha256": sha256(receipt_path),
                "status": receipt["status"],
            }
        )
        failures.extend(receipt["failures"])
        for sequence in receipt["sequences"]:
            sequence_id = sequence["sequence"]
            if sequence_id in seen:
                raise ValueError(f"duplicate successful sequence: {sequence_id}")
            seen.add(sequence_id)
            metadata_path = validate_file(repo, sequence["metadata"])
            video_path = validate_file(repo, sequence["video"])
            metadata_timestamps = {
                line.split()[0]
                for line in metadata_path.read_text(encoding="utf-8").splitlines()[1:]
                if line.strip()
            }
            dimensions: dict[str, list[int]] = {}
            for role, frame in sequence["frames"].items():
                frame_path = validate_file(repo, frame)
                if frame["timestamp_microseconds"] not in metadata_timestamps:
                    raise ValueError(
                        f"timestamp missing from metadata: {sequence_id}/{role}"
                    )
                with Image.open(frame_path) as image:
                    image.verify()
                with Image.open(frame_path) as image:
                    dimensions[role] = list(image.size)
            if len({tuple(value) for value in dimensions.values()}) != 1:
                raise ValueError(f"frame dimensions differ within {sequence_id}")
            successes.append(
                {
                    "sequence": sequence_id,
                    "video_url": sequence["video_url"],
                    "angle_error_sum_deg": sequence["triplet"]["angle_error_sum_deg"],
                    "total_angle_deg": sequence["triplet"]["total_angle_deg"],
                    "frame_dimensions_wh": dimensions["source"],
                    "video": sequence["video"],
                    "metadata": sequence["metadata"],
                    "frames": sequence["frames"],
                    "triplet": sequence["triplet"],
                }
            )
            if video_path.stat().st_size == 0:
                raise ValueError(f"empty video: {video_path}")

    successes.sort(key=lambda item: item["angle_error_sum_deg"])
    status = "pass" if len(successes) == args.expected_successes else "fail"
    result = {
        "schema_version": "stage1.4-hlp-app-01-final-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "dataset_id": "HLP-APP-01",
        "scope": "RealEstate10K RGB/pose appearance truth; no dense depth truth",
        "selection_rule": "lowest-angle-error distinct candidates, skipping only recorded download failures",
        "summary": {
            "expected_successful_sequences": args.expected_successes,
            "successful_sequences": len(successes),
            "failed_attempts": len(failures),
            "rgb_frames": 3 * len(successes),
            "all_files_hash_verified": True,
            "all_frame_timestamps_in_metadata": True,
            "all_triplets_have_consistent_dimensions": True,
        },
        "input_receipts": receipt_records,
        "sequences": successes,
        "failed_attempts": failures,
        "quality_boundary": "Dataset readiness only; no Flash3D novel-view or CLB quality conclusion.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"] | {"status": status}, ensure_ascii=False, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
