#!/usr/bin/env python3
"""Compare the latest linux5080 and win5060 artifact-tree snapshots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "AGENTS.md").is_file():
            return candidate
    raise RuntimeError("无法定位 2Dto3D 仓库根目录。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两台机器的最新目录结构快照。")
    parser.add_argument(
        "--snapshot-root",
        default="03_实验记录/目录结构快照",
        help="快照目录（仓库相对路径）。",
    )
    parser.add_argument(
        "--output",
        default="03_实验记录/目录结构快照/comparisons/latest_comparison.json",
        help="比较结果（仓库相对路径）。",
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    return parser.parse_args()


def resolve_inside_repo(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve(strict=False)
    candidate.relative_to(repo_root)
    return candidate


def relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_snapshot(path: Path, expected_machine: str) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries: dict[str, dict[str, object]] = {}
    captured_values: set[str] = set()
    scan_roots_values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("producer_machine_id") != expected_machine:
                raise ValueError(f"{path}:{line_number} 机器 ID 不是 {expected_machine}")
            key = item.get("relative_path")
            if not isinstance(key, str) or key.startswith("/") or "\\" in key:
                raise ValueError(f"{path}:{line_number} 不是合法仓库相对路径")
            if key in entries:
                raise ValueError(f"{path}:{line_number} 重复路径：{key}")
            entries[key] = item
            captured_values.add(str(item.get("captured_at")))
            scan_roots_values.add(json.dumps(item.get("scan_roots"), ensure_ascii=False, sort_keys=True))
    if len(captured_values) > 1:
        raise ValueError(f"{path} 包含多个 captured_at：{sorted(captured_values)}")
    if len(scan_roots_values) > 1:
        raise ValueError(f"{path} 包含多组 scan_roots")
    scan_roots = json.loads(next(iter(scan_roots_values), "[]"))
    summary = {
        "producer_machine_id": expected_machine,
        "captured_at": next(iter(captured_values), None),
        "scan_roots": scan_roots,
        "entries": len(entries),
        "directories": sum(item.get("entry_type") == "directory" for item in entries.values()),
        "files": sum(item.get("entry_type") == "file" for item in entries.values()),
        "file_bytes": sum(int(item.get("bytes") or 0) for item in entries.values() if item.get("entry_type") == "file"),
    }
    return entries, summary


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__))
    snapshot_root = resolve_inside_repo(repo_root, args.snapshot_root)
    linux_path = snapshot_root / "linux5080_latest.jsonl"
    windows_path = snapshot_root / "win5060_latest.jsonl"
    linux, linux_summary = load_snapshot(linux_path, "linux5080")
    windows, windows_summary = load_snapshot(windows_path, "win5060")

    linux_paths = set(linux)
    windows_paths = set(windows)
    common = sorted(linux_paths & windows_paths)
    type_mismatch = [path for path in common if linux[path].get("entry_type") != windows[path].get("entry_type")]
    size_mismatch = [
        path
        for path in common
        if linux[path].get("entry_type") == windows[path].get("entry_type") == "file"
        and linux[path].get("bytes") != windows[path].get("bytes")
    ]
    hash_mismatch = [
        path
        for path in common
        if linux[path].get("sha256") is not None
        and windows[path].get("sha256") is not None
        and linux[path].get("sha256") != windows[path].get("sha256")
    ]
    result = {
        "schema_version": "1.0-artifact-tree-comparison",
        "generated_at": datetime.now(ZoneInfo(args.timezone)).isoformat(timespec="seconds"),
        "snapshots": {
            "linux5080": {"path": relative(repo_root, linux_path), **linux_summary},
            "win5060": {"path": relative(repo_root, windows_path), **windows_summary},
        },
        "counts": {
            "common": len(common),
            "only_on_linux5080": len(linux_paths - windows_paths),
            "only_on_win5060": len(windows_paths - linux_paths),
            "type_mismatch": len(type_mismatch),
            "size_mismatch": len(size_mismatch),
            "hash_mismatch": len(hash_mismatch),
        },
        "only_on_linux5080": sorted(linux_paths - windows_paths),
        "only_on_win5060": sorted(windows_paths - linux_paths),
        "type_mismatch": type_mismatch,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "policy": "Differences are reported only and never trigger copy, deletion, or forced alignment.",
    }
    output = resolve_inside_repo(repo_root, args.output)
    atomic_write_json(output, result)
    print(json.dumps({"status": "success", "output": relative(repo_root, output), **result["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
