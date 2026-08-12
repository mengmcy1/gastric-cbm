#!/usr/bin/env python3
"""Compare the latest linux5080 and win5060 artifact-tree snapshots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo


VALID_ENTRY_TYPES = {"directory", "file", "symlink", "other"}
LEGACY_ENTRY_SCHEMA = "1.0-artifact-tree-entry"
CURRENT_ENTRY_SCHEMA = "1.1-artifact-tree-entry"
CURRENT_SNAPSHOT_SCHEMA = "1.1-artifact-tree-snapshot"


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


def validate_relative_path(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or (len(value) >= 2 and value[1] == ":")
    ):
        raise ValueError(f"{source} 不是合法仓库相对路径")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{source} 不是合法仓库相对路径")
    return value


def validate_scan_roots(value: object, source: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} scan_roots 必须是非空数组")
    roots = [validate_relative_path(item, f"{source} scan_roots") for item in value]
    if len(set(roots)) != len(roots):
        raise ValueError(f"{source} scan_roots 包含重复路径")
    return roots


def validate_captured_at(value: object, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} captured_at 必须是带时区字符串")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{source} captured_at 不是合法 ISO 8601 时间") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{source} captured_at 必须包含时区")
    return value


def validate_entry(item: object, source: str, expected_schema: str) -> tuple[str, dict[str, object]]:
    if not isinstance(item, dict):
        raise ValueError(f"{source} 必须是 JSON 对象")
    if item.get("schema_version") != expected_schema:
        raise ValueError(
            f"{source} 不支持的 schema_version：{item.get('schema_version')!r}；"
            f"期望 {expected_schema}"
        )
    key = validate_relative_path(item.get("relative_path"), source)
    kind = item.get("entry_type")
    if kind not in VALID_ENTRY_TYPES:
        raise ValueError(f"{source} entry_type 不受支持：{kind!r}")
    size = item.get("bytes")
    if kind in {"file", "symlink"}:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{source} {kind} 的 bytes 必须是非负整数")
    elif size is not None:
        raise ValueError(f"{source} {kind} 的 bytes 必须是 null")
    return key, item


def load_metadata(path: Path, expected_machine: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} 必须是 JSON 对象")
    if metadata.get("schema_version") != CURRENT_SNAPSHOT_SCHEMA:
        raise ValueError(
            f"{path} 不支持的 schema_version：{metadata.get('schema_version')!r}"
        )
    if metadata.get("producer_machine_id") != expected_machine:
        raise ValueError(f"{path} 机器 ID 不是 {expected_machine}")
    metadata["captured_at"] = validate_captured_at(metadata.get("captured_at"), str(path))
    metadata["scan_roots"] = validate_scan_roots(metadata.get("scan_roots"), str(path))
    return metadata


def load_snapshot(path: Path, expected_machine: str) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata_path = path.with_name(f"{path.stem}.meta.json")
    current_metadata = load_metadata(metadata_path, expected_machine) if metadata_path.is_file() else None
    entries: dict[str, dict[str, object]] = {}
    captured_values: set[str] = set()
    scan_roots_values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            source = f"{path}:{line_number}"
            expected_schema = CURRENT_ENTRY_SCHEMA if current_metadata is not None else LEGACY_ENTRY_SCHEMA
            key, item = validate_entry(item, source, expected_schema)
            if key in entries:
                raise ValueError(f"{source} 重复路径：{key}")
            entries[key] = item
            if current_metadata is None:
                if item.get("producer_machine_id") != expected_machine:
                    raise ValueError(f"{source} 机器 ID 不是 {expected_machine}")
                captured_values.add(validate_captured_at(item.get("captured_at"), source))
                roots = validate_scan_roots(item.get("scan_roots"), source)
                scan_roots_values.add(json.dumps(roots, ensure_ascii=False, sort_keys=True))

    if current_metadata is not None:
        captured_at = current_metadata["captured_at"]
        scan_roots = current_metadata["scan_roots"]
        schema_version = CURRENT_SNAPSHOT_SCHEMA
    else:
        if not entries:
            raise ValueError(f"{path} 为空且缺少 schema 1.1 元数据文件 {metadata_path.name}")
        if len(captured_values) != 1:
            raise ValueError(f"{path} 包含多组 captured_at：{sorted(captured_values)}")
        if len(scan_roots_values) != 1:
            raise ValueError(f"{path} 包含多组 scan_roots")
        captured_at = next(iter(captured_values))
        scan_roots = json.loads(next(iter(scan_roots_values)))
        schema_version = "1.0-artifact-tree-snapshot"

    summary = {
        "schema_version": schema_version,
        "producer_machine_id": expected_machine,
        "captured_at": captured_at,
        "scan_roots": scan_roots,
        "entries": len(entries),
        "directories": sum(item.get("entry_type") == "directory" for item in entries.values()),
        "files": sum(item.get("entry_type") == "file" for item in entries.values()),
        "file_bytes": sum(int(item.get("bytes") or 0) for item in entries.values() if item.get("entry_type") == "file"),
    }
    if current_metadata is not None:
        for field in ("entries", "file_bytes"):
            if current_metadata.get(field) != summary[field]:
                raise ValueError(
                    f"{metadata_path} 的 {field}={current_metadata.get(field)!r} "
                    f"与 JSONL 实际值 {summary[field]!r} 不一致"
                )
        expected_counts = {
            kind: sum(item.get("entry_type") == kind for item in entries.values())
            for kind in sorted(VALID_ENTRY_TYPES)
        }
        if current_metadata.get("counts") != expected_counts:
            raise ValueError(
                f"{metadata_path} 的 counts={current_metadata.get('counts')!r} "
                f"与 JSONL 实际值 {expected_counts!r} 不一致"
            )
    return entries, summary


def ensure_matching_scan_roots(
    linux_summary: dict[str, object], windows_summary: dict[str, object]
) -> None:
    if linux_summary["scan_roots"] != windows_summary["scan_roots"]:
        raise ValueError(
            "两台机器的 scan_roots 不一致，不能进行正式结构对账："
            f"linux5080={linux_summary['scan_roots']!r}, "
            f"win5060={windows_summary['scan_roots']!r}"
        )


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__))
    snapshot_root = resolve_inside_repo(repo_root, args.snapshot_root)
    linux_path = snapshot_root / "linux5080_latest.jsonl"
    windows_path = snapshot_root / "win5060_latest.jsonl"
    linux, linux_summary = load_snapshot(linux_path, "linux5080")
    windows, windows_summary = load_snapshot(windows_path, "win5060")

    ensure_matching_scan_roots(linux_summary, windows_summary)

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
    result = {
        "schema_version": "1.1-artifact-tree-comparison",
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
        },
        "only_on_linux5080": sorted(linux_paths - windows_paths),
        "only_on_win5060": sorted(windows_paths - linux_paths),
        "type_mismatch": type_mismatch,
        "size_mismatch": size_mismatch,
        "policy": "Differences are reported only and never trigger copy, deletion, or forced alignment.",
    }
    output = resolve_inside_repo(repo_root, args.output)
    atomic_write_json(output, result)
    print(json.dumps({"status": "success", "output": relative(repo_root, output), **result["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
