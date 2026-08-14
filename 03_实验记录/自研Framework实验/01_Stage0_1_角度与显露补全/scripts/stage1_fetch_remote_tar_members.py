#!/usr/bin/env python3
"""Fetch selected regular-file members from a remote, uncompressed tar.

The tool uses verified HTTP byte ranges and tar headers.  It is intended for
the 10 GB Flash3D RealEstate10K sparse-COLMAP archive, where downloading the
whole archive to obtain a few sequence members is unnecessary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BLOCK = 512
DEFAULT_URL = "https://thor.robots.ox.ac.uk/flash3d/pcl.test.tar"
DEFAULT_ARCHIVE_BYTES = 10_312_816_640


@dataclass(frozen=True)
class TarHeader:
    offset: int
    name: str
    size: int
    typeflag: bytes

    @property
    def data_offset(self) -> int:
        return self.offset + BLOCK

    @property
    def next_offset(self) -> int:
        return self.data_offset + ((self.size + BLOCK - 1) // BLOCK) * BLOCK


class RangeReader:
    def __init__(self, url: str, archive_bytes: int, timeout: int, retries: int) -> None:
        self.url = url
        self.archive_bytes = archive_bytes
        self.timeout = timeout
        self.retries = retries
        self.requests: list[dict[str, object]] = []
        self.cache: dict[tuple[int, int], bytes] = {}

    def read(self, start: int, end: int, *, cache: bool = True) -> bytes:
        start = max(0, start)
        end = min(self.archive_bytes - 1, end)
        if end < start:
            return b""
        key = (start, end)
        if cache and key in self.cache:
            return self.cache[key]

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            began = time.monotonic()
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "2Dto3D-stage1-remote-tar/1",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", response.getcode())
                    content_range = response.headers.get("Content-Range")
                    payload = response.read()
                expected = end - start + 1
                expected_range = f"bytes {start}-{end}/{self.archive_bytes}"
                if status != 206:
                    raise RuntimeError(f"range request returned HTTP {status}, expected 206")
                if content_range != expected_range:
                    raise RuntimeError(
                        f"Content-Range {content_range!r}, expected {expected_range!r}"
                    )
                if len(payload) != expected:
                    raise RuntimeError(f"received {len(payload)} bytes, expected {expected}")
                self.requests.append(
                    {
                        "start": start,
                        "end": end,
                        "bytes": len(payload),
                        "attempt": attempt,
                        "seconds": time.monotonic() - began,
                    }
                )
                if cache:
                    self.cache[key] = payload
                return payload
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"range {start}-{end} failed after {self.retries} attempts") from last_error


def _parse_octal(raw: bytes) -> int:
    stripped = raw.rstrip(b"\0 ").lstrip(b" ")
    return int(stripped or b"0", 8)


def parse_header(block: bytes, offset: int) -> TarHeader | None:
    if len(block) != BLOCK or block == bytes(BLOCK):
        return None
    try:
        stored_checksum = _parse_octal(block[148:156])
    except ValueError:
        return None
    computed_checksum = sum(block[:148]) + 8 * ord(" ") + sum(block[156:])
    if stored_checksum != computed_checksum:
        return None
    try:
        size = _parse_octal(block[124:136])
    except ValueError:
        return None
    name = block[:100].split(b"\0", 1)[0]
    prefix = block[345:500].split(b"\0", 1)[0]
    if prefix:
        name = prefix + b"/" + name
    try:
        decoded_name = name.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not decoded_name:
        return None
    return TarHeader(offset=offset, name=decoded_name, size=size, typeflag=block[156:157])


def headers_in_range(reader: RangeReader, start: int, end: int) -> list[TarHeader]:
    aligned_start = max(0, (start // BLOCK) * BLOCK)
    aligned_end = min(reader.archive_bytes - 1, ((end + BLOCK) // BLOCK) * BLOCK - 1)
    payload = reader.read(aligned_start, aligned_end)
    headers: list[TarHeader] = []
    for relative in range(0, len(payload) - BLOCK + 1, BLOCK):
        header = parse_header(payload[relative : relative + BLOCK], aligned_start + relative)
        if header is not None:
            headers.append(header)
    return headers


def find_header(
    reader: RangeReader,
    target: str,
    *,
    initial_window_bytes: int,
    max_window_bytes: int,
) -> TarHeader:
    low = 0
    high = reader.archive_bytes
    seen_bounds: set[tuple[int, int]] = set()
    for _ in range(80):
        if low >= high:
            break
        midpoint = ((low + high) // 2 // BLOCK) * BLOCK
        window = initial_window_bytes
        headers: list[TarHeader] = []
        while window <= max_window_bytes:
            start = max(low, midpoint - window // 2)
            end = min(high - 1, midpoint + window // 2 - 1)
            bounds = (start, end)
            if bounds not in seen_bounds:
                seen_bounds.add(bounds)
                headers = headers_in_range(reader, start, end)
            if headers:
                break
            window *= 2
        if not headers:
            raise RuntimeError(
                f"no verified tar header near byte {midpoint}; increase --max-window-mib"
            )

        regular = sorted(
            (header for header in headers if header.typeflag in (b"0", b"\0")),
            key=lambda header: header.offset,
        )
        if not regular:
            raise RuntimeError(f"no regular-file tar header near byte {midpoint}")
        exact = next((header for header in regular if header.name == target), None)
        if exact is not None:
            return exact

        names = [header.name for header in regular]
        if target < names[0]:
            new_high = regular[0].offset
            if new_high >= high:
                raise RuntimeError(f"binary search did not shrink above {target}")
            high = new_high
            continue
        if target > names[-1]:
            new_low = regular[-1].next_offset
            if new_low <= low:
                raise RuntimeError(f"binary search did not shrink below {target}")
            low = new_low
            continue

        # All consecutive headers within the downloaded window were inspected.
        # A lexically sorted archive cannot contain target between two such names.
        for left, right in zip(regular, regular[1:]):
            if left.name < target < right.name:
                raise FileNotFoundError(f"{target!r} is absent between {left.name!r} and {right.name!r}")
        raise FileNotFoundError(f"could not locate {target!r} in a sorted header window")
    raise FileNotFoundError(f"tar member not found: {target}")


def validate_sparse_pickle(compressed: bytes) -> dict[str, object]:
    decoded = gzip.decompress(compressed)
    value = pickle.loads(decoded)
    if not isinstance(value, dict):
        raise TypeError(f"sparse point cloud is {type(value).__name__}, expected dict")
    required = {"xys", "p3D_ids", "xyz"}
    missing = sorted(required.difference(value))
    if missing:
        raise KeyError(f"sparse point cloud missing keys: {missing}")
    xyz = value["xyz"]
    xys = value["xys"]
    point_ids = value["p3D_ids"]
    return {
        "pickle_uncompressed_bytes": len(decoded),
        "keys": sorted(str(key) for key in value),
        "xyz_shape": list(getattr(xyz, "shape", ())),
        "frame_count_xys": len(xys),
        "frame_count_p3D_ids": len(point_ids),
    }


def resolve_members(sequences: Iterable[str]) -> list[str]:
    members = []
    for sequence in sequences:
        sequence = sequence.strip()
        if not sequence or "/" in sequence or sequence in {".", ".."}:
            raise ValueError(f"invalid sequence id: {sequence!r}")
        members.append(f"pcl.test/{sequence}.pickle.gz")
    if len(members) != len(set(members)):
        raise ValueError("sequence ids must be unique")
    return members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive-bytes", type=int, default=DEFAULT_ARCHIVE_BYTES)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--initial-window-mib", type=int, default=8)
    parser.add_argument("--max-window-mib", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    members = resolve_members(args.sequence)
    if args.record.exists():
        raise FileExistsError(f"refusing to overwrite record: {args.record}")
    for member in members:
        destination = args.output_dir / Path(member).name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite output: {destination}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    reader = RangeReader(args.url, args.archive_bytes, args.timeout, args.retries)
    outcomes: list[dict[str, object]] = []
    status = "pass"
    error_message: str | None = None
    try:
        for member in sorted(members):
            header = find_header(
                reader,
                member,
                initial_window_bytes=args.initial_window_mib * 1024 * 1024,
                max_window_bytes=args.max_window_mib * 1024 * 1024,
            )
            compressed = reader.read(
                header.data_offset,
                header.data_offset + header.size - 1,
                cache=False,
            )
            validation = validate_sparse_pickle(compressed)
            destination = args.output_dir / Path(member).name
            # The entire small member is validated in memory before this sole write.
            destination.write_bytes(compressed)
            outcomes.append(
                {
                    "member": member,
                    "header_offset": header.offset,
                    "member_bytes": header.size,
                    "output": str(destination),
                    "sha256": hashlib.sha256(compressed).hexdigest(),
                    "validation": validation,
                }
            )
    except Exception as error:  # Preserve a machine-readable failure receipt.
        status = "partial_failure" if outcomes else "failed"
        error_message = f"{type(error).__name__}: {error}"

    record = {
        "schema_version": "stage1.4-remote-tar-members-v1",
        "status": status,
        "producer_machine_id": "linux5080",
        "source": {
            "url": args.url,
            "archive_bytes": args.archive_bytes,
            "official_archive_sha512": "1f36131b23b04ed0ab293d5d509cc191e6f24c30214411aad27ae42737ee0a27dd70bf33443b0f1bdc726e9b58504e6761c99754e43b7bef759535eeb759d065",
            "archive_sha512_scope": "official full-archive checksum; not recomputed from selected ranges",
        },
        "requested_members": sorted(members),
        "completed_members": outcomes,
        "error": error_message,
        "range_request_count": len(reader.requests),
        "range_bytes_received": sum(int(request["bytes"]) for request in reader.requests),
        "range_requests": reader.requests,
        "safety": {
            "full_archive_downloaded": False,
            "overwrites_refused": True,
            "files_deleted": False,
        },
    }
    args.record.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
