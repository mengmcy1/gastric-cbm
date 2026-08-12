#!/usr/bin/env python3
"""Create a Git-friendly inventory of local experiment artifact paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_SCAN_ROOTS = (
    "03_实验记录/SHARP浅3D实验",
    "03_实验记录/自研Framework实验",
    "03_实验记录/InfiniSplat复现实验",
)
DEFAULT_EXCLUDED_DIR_NAMES = {"__pycache__"}


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "AGENTS.md").is_file():
            return candidate
    raise RuntimeError("无法定位 2Dto3D 仓库根目录。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成当前机器的实验目录结构快照。")
    parser.add_argument("--machine-id", required=True, choices=("linux5080", "win5060"))
    parser.add_argument(
        "--scan-root",
        action="append",
        default=None,
        help="仓库相对扫描根；可重复传入。默认扫描三条正式实验线。",
    )
    parser.add_argument(
        "--output-root",
        default="03_实验记录/目录结构快照",
        help="快照输出目录（仓库相对路径）。",
    )
    parser.add_argument(
        "--hash-path",
        action="append",
        default=[],
        help="需要额外计算 SHA256 的仓库相对文件；可重复传入。",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="captured_at 的 IANA 时区，默认 Asia/Shanghai。",
    )
    return parser.parse_args()


def resolve_inside_repo(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve(strict=False)
    candidate.relative_to(repo_root)
    return candidate


def repo_relative(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def entry_type(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def collect_entries(
    repo_root: Path,
    scan_roots: list[Path],
    scan_root_strings: list[str],
    machine_id: str,
    captured_at: str,
    hash_paths: set[str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        relative = repo_relative(repo_root, path)
        if relative in seen:
            return
        seen.add(relative)
        metadata = path.lstat()
        kind = entry_type(metadata.st_mode)
        digest = sha256(path) if kind == "file" and relative in hash_paths else None
        entries.append(
            {
                "schema_version": "1.0-artifact-tree-entry",
                "producer_machine_id": machine_id,
                "captured_at": captured_at,
                "scan_roots": scan_root_strings,
                "relative_path": relative,
                "entry_type": kind,
                "bytes": metadata.st_size if kind in {"file", "symlink"} else None,
                "mtime_ns": metadata.st_mtime_ns,
                "sha256": digest,
            }
        )

    for root in scan_roots:
        if not root.exists():
            continue
        add(root)
        for current_text, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(current_text)
            directory_names[:] = sorted(
                name for name in directory_names if name not in DEFAULT_EXCLUDED_DIR_NAMES
            )
            for name in directory_names:
                add(current / name)
            for name in sorted(file_names):
                add(current / name)
    entries.sort(key=lambda item: str(item["relative_path"]))
    return entries


def render_tree(entries: list[dict[str, object]], scan_root_strings: list[str]) -> str:
    lines = [
        f"# producer_machine_id: {entries[0]['producer_machine_id'] if entries else 'unknown'}",
        f"# captured_at: {entries[0]['captured_at'] if entries else 'unknown'}",
        f"# scan_roots: {json.dumps(scan_root_strings, ensure_ascii=False)}",
        "# D=directory F=file L=symlink O=other; paths are repository-relative",
    ]
    for item in entries:
        relative = str(item["relative_path"])
        depth = relative.count("/")
        marker = {"directory": "D", "file": "F", "symlink": "L"}.get(
            str(item["entry_type"]), "O"
        )
        suffix = "/" if item["entry_type"] == "directory" else ""
        size = "" if item["bytes"] is None else f" ({item['bytes']} bytes)"
        lines.append(f"{'  ' * depth}[{marker}] {Path(relative).name}{suffix}{size}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__))
    scan_root_strings = args.scan_root or list(DEFAULT_SCAN_ROOTS)
    scan_roots = [resolve_inside_repo(repo_root, value) for value in scan_root_strings]
    missing = [value for value, path in zip(scan_root_strings, scan_roots) if not path.exists()]
    if missing:
        print(f"警告：以下扫描根不存在，已记录但跳过：{missing}")

    hash_paths = {repo_relative(repo_root, resolve_inside_repo(repo_root, value)) for value in args.hash_path}
    for relative in sorted(hash_paths):
        path = resolve_inside_repo(repo_root, relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"--hash-path 必须是仓库内普通文件：{relative}")

    captured_at = datetime.now(ZoneInfo(args.timezone)).isoformat(timespec="seconds")
    entries = collect_entries(
        repo_root,
        scan_roots,
        scan_root_strings,
        args.machine_id,
        captured_at,
        hash_paths,
    )
    output_root = resolve_inside_repo(repo_root, args.output_root)
    jsonl_path = output_root / f"{args.machine_id}_latest.jsonl"
    tree_path = output_root / f"{args.machine_id}_latest.txt"
    jsonl = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in entries)
    atomic_write_text(jsonl_path, jsonl)
    atomic_write_text(tree_path, render_tree(entries, scan_root_strings))

    counts = {kind: sum(item["entry_type"] == kind for item in entries) for kind in ("directory", "file", "symlink", "other")}
    total_bytes = sum(int(item["bytes"] or 0) for item in entries if item["entry_type"] == "file")
    print(
        json.dumps(
            {
                "status": "success",
                "producer_machine_id": args.machine_id,
                "captured_at": captured_at,
                "scan_roots": scan_root_strings,
                "counts": counts,
                "file_bytes": total_bytes,
                "jsonl": repo_relative(repo_root, jsonl_path),
                "tree": repo_relative(repo_root, tree_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
