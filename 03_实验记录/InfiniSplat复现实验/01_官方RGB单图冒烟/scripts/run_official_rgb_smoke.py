#!/usr/bin/env python3
"""Cross-platform InfiniSplat RGB smoke runner for Windows and Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen InfiniSplat RGB smoke config without overwriting outputs."
    )
    parser.add_argument("--config", required=True, help="Config JSON path.")
    parser.add_argument(
        "--machine-id",
        required=True,
        choices=("win5060", "linux5080"),
        help="Stable producer machine identifier written to run_receipt.json.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify environment, source metadata, input and checkpoint without running inference.",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Append a portable suffix to the configured output directory and log filename.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    resolved = (repo_root / relative_path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {relative_path}") from exc
    return resolved


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def verify_source(
    repo_root: Path, experiment_root: Path, source_path: Path, config: dict[str, Any]
) -> None:
    if not source_path.is_dir():
        raise FileNotFoundError(f"InfiniSplat source directory is missing: {source_path}")

    expected_commit = config["source_commit"]
    nested_git = source_path / ".git"
    if nested_git.exists():
        actual_commit = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_commit != expected_commit:
            raise RuntimeError(
                f"Source commit mismatch: expected {expected_commit}, actual {actual_commit}"
            )
        return

    manifest_path = experiment_root / "inputs" / "infinisplat_source_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["source_path"] != str(source_path.relative_to(repo_root)).replace("\\", "/"):
        raise RuntimeError("Vendored source path does not match source manifest")
    if manifest["upstream_commit"] != expected_commit:
        raise RuntimeError(
            "Vendored source manifest commit does not match the frozen config commit"
        )


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    experiment_root = script_dir.parent
    repo_root = experiment_root.parents[2]
    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)

    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_suffix):
        raise ValueError("--run-suffix may contain only ASCII letters, digits, dot, underscore and hyphen")
    if args.machine_id == "linux5080" and not args.run_suffix and not args.validate_only:
        raise ValueError("linux5080 runs must provide a unique --run-suffix, for example linux5080_20260811")

    expected_env = config["conda_env"]
    actual_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    if actual_env != expected_env:
        raise RuntimeError(
            f"Activate the dedicated '{expected_env}' environment first; current environment is '{actual_env}'."
        )

    source_path = resolve_repo_path(repo_root, config["source_path"])
    input_manifest_path = resolve_repo_path(repo_root, config["input_manifest"])
    checkpoint_path = resolve_repo_path(repo_root, config["checkpoint_path"])
    output_dir = resolve_repo_path(repo_root, config["output_dir"])
    log_path = resolve_repo_path(repo_root, config["log_path"])
    if args.run_suffix:
        output_dir = output_dir.with_name(f"{output_dir.name}_{args.run_suffix}")
        log_path = log_path.with_name(f"{log_path.stem}_{args.run_suffix}{log_path.suffix}")
    input_manifest = load_json(input_manifest_path)
    input_path = resolve_repo_path(repo_root, input_manifest["path"])

    verify_source(repo_root, experiment_root, source_path, config)
    require_file(input_path, "Input image")
    require_file(checkpoint_path, "Checkpoint")

    actual_input_hash = sha256(input_path)
    if actual_input_hash != input_manifest["sha256"]:
        raise RuntimeError(
            f"Input SHA256 mismatch: expected {input_manifest['sha256']}, actual {actual_input_hash}"
        )

    checkpoint_bytes = checkpoint_path.stat().st_size
    if checkpoint_bytes != int(config["checkpoint_expected_bytes"]):
        raise RuntimeError(
            f"Checkpoint size mismatch: expected {config['checkpoint_expected_bytes']}, actual {checkpoint_bytes}"
        )
    actual_checkpoint_hash = sha256(checkpoint_path)
    if actual_checkpoint_hash != config["checkpoint_expected_sha256"]:
        raise RuntimeError(
            "Checkpoint SHA256 mismatch: "
            f"expected {config['checkpoint_expected_sha256']}, actual {actual_checkpoint_hash}"
        )

    if args.validate_only:
        print("Validation passed.")
        print(f"Machine: {args.machine_id}")
        print(f"Config: {config_path}")
        print(f"Input: {input_path}")
        print(f"Checkpoint: {checkpoint_path}")
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    if log_path.exists():
        raise FileExistsError(f"Log file already exists; refusing to overwrite: {log_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "src.demo.infer_batch_images",
        "--mode",
        config["mode"],
        "--checkpoint",
        str(checkpoint_path),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--device",
        config["device"],
    ]
    video_enabled = bool(config.get("video", config.get("video_frames", 0) > 0))
    if not video_enabled:
        command.append("--no-video")
    if not config.get("html", False):
        command.append("--no-export-html")
    if not config.get("floater_filter", True):
        command.append("--disable-floater-filter")
    if config.get("sample_point_num_override") is not None:
        command.extend(
            ["--sample-point-num", str(int(config["sample_point_num_override"]))]
        )
    camera_overrides = [
        config.get("intrinsics_file"),
        config.get("focal_px"),
        config.get("focal_mm"),
    ]
    if sum(value is not None for value in camera_overrides) > 1:
        raise ValueError(
            "Only one of intrinsics_file, focal_px, and focal_mm may be configured"
        )
    if config.get("intrinsics_file") is not None:
        intrinsics_path = resolve_repo_path(repo_root, config["intrinsics_file"])
        require_file(intrinsics_path, "Camera intrinsics override")
        command.extend(["--intrinsics-file", str(intrinsics_path)])
    elif config.get("focal_px") is not None:
        command.extend(["--focal-px", str(float(config["focal_px"]))])
    elif config.get("focal_mm") is not None:
        command.extend(["--focal-mm", str(float(config["focal_mm"]))])

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started_at = time.time()
    with log_path.open("x", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=source_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"InfiniSplat smoke failed with exit code {return_code}. Log: {log_path}"
        )

    case_dir = output_dir / input_path.stem
    ply_path = case_dir / f"{input_path.stem}.ply"
    video_path = case_dir / f"{input_path.stem}.mp4"
    require_file(ply_path, "Expected PLY")
    if video_enabled:
        require_file(video_path, "Expected video")

    receipt = {
        "schema_version": "1.0-cross-platform-run-receipt",
        "status": "success",
        "run_id": output_dir.name,
        "producer_machine_id": args.machine_id,
        "project_git_commit": git_commit(repo_root),
        "upstream_source_commit": config["source_commit"],
        "config_path": str(config_path.relative_to(repo_root)).replace("\\", "/"),
        "input_path": input_manifest["path"],
        "input_sha256": actual_input_hash,
        "checkpoint_sha256": actual_checkpoint_hash,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "duration_seconds": round(time.time() - started_at, 3),
        "sample_point_num_override": config.get("sample_point_num_override"),
        "artifacts": [
            {
                "path": str(ply_path.relative_to(repo_root)).replace("\\", "/"),
                "bytes": ply_path.stat().st_size,
                "sha256": sha256(ply_path),
            }
        ],
    }
    if video_enabled:
        receipt["artifacts"].append(
            {
                "path": str(video_path.relative_to(repo_root)).replace("\\", "/"),
                "bytes": video_path.stat().st_size,
                "sha256": sha256(video_path),
            }
        )
    receipt_path = output_dir / "run_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"InfiniSplat smoke completed: {output_dir}")
    print(f"Run receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
