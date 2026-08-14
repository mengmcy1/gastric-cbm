#!/usr/bin/env python3
"""Download only selected RealEstate10K videos and extract angle triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from pytubefix import YouTube


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def select_unique(
    triplets: list[dict], count: int, excluded: set[str]
) -> list[dict]:
    selected: list[dict] = []
    sequences: set[str] = set()
    for triplet in sorted(triplets, key=lambda item: item["angle_error_sum_deg"]):
        if triplet["sequence"] in sequences or triplet["sequence"] in excluded:
            continue
        selected.append(triplet)
        sequences.add(triplet["sequence"])
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)} unique sequences are available")


def extract_metadata(archive: tarfile.TarFile, sequence: str, output: Path) -> None:
    member_name = f"RealEstate10K/test/{sequence}.txt"
    member = archive.getmember(member_name)
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"could not read {member_name}")
    output.write_bytes(stream.read())


def extract_frame(video: Path, timestamp: str, output: Path) -> None:
    seconds = int(timestamp) / 1_000_000.0
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
            "-ss", f"{seconds:.6f}", "-i", str(video),
            "-frames:v", "1", "-f", "image2", str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create {output}")


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": portable(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--metadata-tar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--sequence-count", type=int, default=5)
    parser.add_argument("--exclude-sequence", action="append", default=[])
    parser.add_argument("--producer-machine-id", default="linux5080")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    triplets_path = args.triplets.resolve()
    metadata_tar = args.metadata_tar.resolve()
    output_root = args.output_root.resolve()
    record_path = args.record.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite record: {record_path}")
    if args.sequence_count <= 0:
        raise ValueError("sequence-count must be positive")

    source = json.loads(triplets_path.read_text(encoding="utf-8"))
    excluded = set(args.exclude_sequence)
    selected = select_unique(source["triplets"], args.sequence_count, excluded)
    output_root.mkdir(parents=True)
    results: list[dict] = []
    failures: list[dict] = []

    with tarfile.open(metadata_tar, "r:gz") as archive:
        for triplet in selected:
            sequence = triplet["sequence"]
            sequence_dir = output_root / sequence
            sequence_dir.mkdir()
            metadata_path = sequence_dir / "metadata.txt"
            video_path = sequence_dir / "source_video.mp4"
            try:
                extract_metadata(archive, sequence, metadata_path)
                youtube = YouTube(triplet["video_url"])
                stream = youtube.streams.filter(
                    progressive=True, file_extension="mp4", res="360p"
                ).first()
                if stream is None:
                    raise RuntimeError("official 360p progressive MP4 stream unavailable")
                downloaded = Path(
                    stream.download(
                        output_path=str(sequence_dir), filename=video_path.name
                    )
                ).resolve()
                if downloaded != video_path or not video_path.is_file():
                    raise RuntimeError(f"unexpected download path: {downloaded}")

                frames: dict[str, dict] = {}
                for role in ("source", "left", "right"):
                    frame_path = sequence_dir / f"{role}.jpg"
                    timestamp = triplet[f"{role}_timestamp"]
                    extract_frame(video_path, timestamp, frame_path)
                    frames[role] = {
                        **file_record(frame_path, repo_root),
                        "timestamp_microseconds": timestamp,
                        "frame_index": triplet[f"{role}_index"],
                        "angle_deg": 0.0 if role == "source" else triplet[f"{role}_angle_deg"],
                    }
                results.append(
                    {
                        "sequence": sequence,
                        "video_url": triplet["video_url"],
                        "stream": {
                            "itag": stream.itag,
                            "resolution": stream.resolution,
                            "fps": stream.fps,
                            "mime_type": stream.mime_type,
                        },
                        "video": file_record(video_path, repo_root),
                        "metadata": file_record(metadata_path, repo_root),
                        "triplet": triplet,
                        "frames": frames,
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "sequence": sequence,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    receipt = {
        "schema_version": "stage1.4-hlp-app-01-v1",
        "status": "pass" if not failures else "partial_failure",
        "producer_machine_id": args.producer_machine_id,
        "scope": "five distinct RealEstate10K source/left/right RGB triplets; no depth truth",
        "selection": {
            "rule": "lowest angle_error_sum_deg, at most one triplet per sequence",
            "requested_sequence_count": args.sequence_count,
            "excluded_sequences": sorted(excluded),
            "selected_sequences": [item["sequence"] for item in selected],
            "successful_sequence_count": len(results),
            "failed_sequence_count": len(failures),
        },
        "inputs": {
            "triplets": file_record(triplets_path, repo_root),
            "metadata_tar": file_record(metadata_tar, repo_root),
        },
        "source_protocol": {
            "downloader_basis": "Flash3D official download_realestate10k.py",
            "stream": "360p progressive MP4",
            "timestamp_unit": "microseconds",
            "full_dataset_downloaded": False,
            "files_deleted": False,
        },
        "sequences": results,
        "failures": failures,
        "quality_boundary": "RGB/pose appearance set only; no dense depth and no provider-quality conclusion",
    }
    receipt_text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    (output_root / "run_receipt.json").write_text(receipt_text, encoding="utf-8")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(receipt_text, encoding="utf-8")
    print(json.dumps(receipt["selection"] | {"status": receipt["status"]}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
