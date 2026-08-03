"""Run one hardware-specific SHARP performance baseline without mixing profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = EXPERIMENT_ROOT / "07_性能测试" / "performance_profiles.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs" / "07_性能测试"
WORKER = Path(__file__).resolve().parent / "sharp_performance_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("rtx5080_server", "rtx5060_laptop"), required=True
    )
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--run-id", required=True, help="例如 baseline_v1_20260803")
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="覆盖配置中的权重文件路径；仍强制校验配置记录的 SHA256",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有运行目录续跑；只复用已存在且可解析的 metrics.json",
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_csv(line: str) -> dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 7:
        raise ValueError(f"无法解析 nvidia-smi GPU 输出：{line!r}")
    return {
        "physical_index": int(parts[0]),
        "uuid": parts[1],
        "name": parts[2],
        "driver_version": parts[3],
        "memory_total_mb": int(parts[4]),
        "memory_used_mb": int(parts[5]),
        "utilization_gpu_percent": int(parts[6]),
    }


def gpu_preflight(physical_index: int, profile: dict[str, Any]) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"期望一张 GPU，实际 nvidia-smi 返回 {len(lines)} 行")
    gpu = parse_gpu_csv(lines[0])
    expected = profile["expected_gpu_name"]
    if expected not in gpu["name"]:
        raise RuntimeError(f"GPU 不匹配：期望 {expected!r}，实际 {gpu['name']!r}")
    used_fraction = gpu["memory_used_mb"] / gpu["memory_total_mb"]
    gpu["memory_used_fraction"] = round(used_fraction, 6)
    if gpu["utilization_gpu_percent"] > profile["max_preflight_gpu_util_percent"]:
        raise RuntimeError(
            f"GPU 利用率 {gpu['utilization_gpu_percent']}% 超过规范测试上限 "
            f"{profile['max_preflight_gpu_util_percent']}%"
        )
    if used_fraction > profile["max_preflight_memory_used_fraction"]:
        raise RuntimeError(
            f"GPU 已用显存比例 {used_fraction:.1%} 超过规范测试上限 "
            f"{profile['max_preflight_memory_used_fraction']:.1%}"
        )

    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in process_query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[0] == gpu["uuid"]:
            used_memory_mb = None
            if parts[3] not in {"[N/A]", "N/A", ""}:
                used_memory_mb = int(parts[3])
            processes.append(
                {
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": used_memory_mb,
                }
            )
    blocking_processes = [
        process for process in processes if process["used_memory_mb"] is not None
    ]
    if blocking_processes:
        raise RuntimeError(
            f"选定 GPU 存在其他可计量计算进程，拒绝运行：{blocking_processes}"
        )
    # Windows WDDM may report ordinary desktop C+G processes here with [N/A]
    # memory. Keep them in the manifest for audit, while the utilization and
    # total-memory thresholds above remain the enforceable idle checks.
    gpu["compute_processes"] = processes
    return gpu


def new_output_dir(path: Path, resume: bool) -> Path:
    path = path.resolve()
    if path.exists():
        if not resume:
            raise FileExistsError(f"运行目录已存在，拒绝覆盖：{path}")
        return path
    path.mkdir(parents=True)
    return path


def worker_command(
    args: argparse.Namespace,
    common: dict[str, Any],
    mode: str,
    run_kind: str,
    repeat_index: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--max-disparity",
        str(common["max_disparity"]),
        "--crop-single-side-percent",
        str(common["crop_single_side_percent"]),
        "--num-steps",
        str(common["num_steps"]),
        "--fps",
        str(common["fps"]),
        "--run-kind",
        run_kind,
        "--repeat-index",
        str(repeat_index),
        "--profile",
        args.profile,
        "--physical-gpu-index",
        str(args.physical_gpu_index),
    ]
    if mode == "generation":
        command.extend(
            [
                "--input-image",
                str(REPO_ROOT / common["input_image"]),
                "--checkpoint",
                str(REPO_ROOT / common["checkpoint"]),
            ]
        )
    else:
        command.extend(["--ply", str(REPO_ROOT / common["input_ply"])])
    return command


def run_worker(
    args: argparse.Namespace,
    common: dict[str, Any],
    mode: str,
    run_kind: str,
    repeat_index: int,
    run_root: Path,
) -> dict[str, Any]:
    label = f"{mode}_{run_kind}_r{repeat_index:02d}"
    output_dir = run_root / label
    metrics_path = output_dir / "metrics.json"
    if output_dir.exists():
        if not metrics_path.is_file():
            raise FileExistsError(
                f"{label} 目录已存在但没有完整 metrics.json，拒绝覆盖：{output_dir}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("orchestrator_wall_s", None)
        metrics["resumed_existing_result"] = True
        return metrics
    command = worker_command(
        args, common, mode, run_kind, repeat_index, output_dir
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index)
    python_bin = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    wall_s = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        failure = {
            "label": label,
            "returncode": completed.returncode,
            "wall_s": round(wall_s, 6),
            "command": command,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{label} 失败，请查看 {output_dir / 'stderr.log'}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{label} 未生成 metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["orchestrator_wall_s"] = round(wall_s, 6)
    metrics["resumed_existing_result"] = False
    return metrics


def write_partial_results(run_root: Path, results: list[dict[str, Any]]) -> None:
    path = run_root / "partial_results.json"
    path.write_text(
        json.dumps(
            {
                "completed_workers": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    timings = result.get("timings_s", {})
    frame = result.get("pure_render_frame_ms", {})
    environment = result["environment"]
    return {
        "profile": environment["profile"],
        "mode": result["mode"],
        "run_kind": environment["run_kind"],
        "repeat_index": environment["repeat_index"],
        "gpu_name": environment["gpu_name"],
        "orchestrator_wall_s": result.get("orchestrator_wall_s"),
        "worker_wall_s": result.get("worker_wall_s"),
        "measured_pipeline_total_s": timings.get("measured_pipeline_total_s"),
        "checkpoint_load_s": timings.get("checkpoint_load_s"),
        "model_construct_s": timings.get("model_construct_s"),
        "state_dict_load_s": timings.get("state_dict_load_s"),
        "model_upload_s": timings.get("model_upload_s"),
        "input_load_s": timings.get("input_load_s"),
        "preprocess_s": timings.get("preprocess_s"),
        "network_forward_s": timings.get("network_forward_s"),
        "unproject_postprocess_s": timings.get("unproject_postprocess_s"),
        "ply_write_s": timings.get("ply_write_s"),
        "ply_load_s": timings.get("ply_load_s"),
        "gpu_upload_s": timings.get("gpu_upload_s"),
        "renderer_warmup_s": timings.get("renderer_warmup_s"),
        "pure_render_60_s": result.get("pure_render_60_s"),
        "pure_render_fps": result.get("pure_render_fps"),
        "pure_render_frame_median_ms": frame.get("median"),
        "pure_render_frame_p95_ms": frame.get("p95"),
        "render_transfer_crop_60_s": timings.get("render_transfer_crop_60_s"),
        "video_encode_60_s": timings.get("video_encode_60_s"),
        "peak_rss_mb": result.get("peak_rss_mb"),
        "network_peak_cuda_allocated_mb": result.get(
            "network_peak_cuda_allocated_mb"
        ),
        "network_peak_cuda_reserved_mb": result.get(
            "network_peak_cuda_reserved_mb"
        ),
        "pure_render_peak_cuda_allocated_mb": result.get(
            "pure_render_peak_cuda_allocated_mb"
        ),
        "pure_render_peak_cuda_reserved_mb": result.get(
            "pure_render_peak_cuda_reserved_mb"
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("generation", "render"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        fields: dict[str, Any] = {}
        for field in rows[0]:
            values = [
                float(row[field])
                for row in mode_rows
                if isinstance(row.get(field), (int, float))
            ]
            if values:
                fields[field] = {
                    "count": len(values),
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                }
        result[mode] = fields
    return result


def main() -> None:
    args = parse_args()
    profiles_path = args.profiles_file.resolve()
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    common = dict(profiles["common"])
    if args.checkpoint is not None:
        common["checkpoint"] = str(args.checkpoint.resolve())
    profile = profiles["profiles"][args.profile]
    for field in ("input_image", "input_ply", "checkpoint"):
        path = REPO_ROOT / common[field]
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_path = REPO_ROOT / common["checkpoint"]
    actual_hash = sha256(checkpoint_path)
    if actual_hash.lower() != common["checkpoint_sha256"].lower():
        raise RuntimeError(
            f"权重 SHA256 不匹配：{actual_hash} != {common['checkpoint_sha256']}"
        )

    preflight = gpu_preflight(args.physical_gpu_index, profile)
    run_root = new_output_dir(
        args.output_root / args.profile / args.run_id, resume=args.resume
    )
    manifest_path = run_root / "run_manifest.json"
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("profile") != args.profile:
            raise ValueError(
                f"断点 profile 不匹配：{manifest.get('profile')} != {args.profile}"
            )
        manifest["status"] = "running"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["gpu_preflight_resume"] = preflight
    else:
        manifest = {
            "schema_version": "1.0",
            "status": "running",
            "profile": args.profile,
            "profile_definition": profile,
            "common": common,
            "gpu_preflight": preflight,
            "checkpoint_sha256_verified": actual_hash,
            "python_executable": sys.executable,
            "worker": str(WORKER.relative_to(REPO_ROOT)),
            "resume_count": 0,
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    all_results: list[dict[str, Any]] = []
    try:
        for mode in ("generation", "render"):
            for repeat_index in range(common["warmup_runs"]):
                all_results.append(
                    run_worker(
                        args,
                        common,
                        mode,
                        "warmup",
                        repeat_index,
                        run_root,
                    )
                )
                write_partial_results(run_root, all_results)
            for repeat_index in range(1, common["formal_runs"] + 1):
                all_results.append(
                    run_worker(
                        args,
                        common,
                        mode,
                        "formal",
                        repeat_index,
                        run_root,
                    )
                )
                write_partial_results(run_root, all_results)
    except Exception:
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    formal_results = [
        result
        for result in all_results
        if result["environment"]["run_kind"] == "formal"
    ]
    rows = [flatten_result(result) for result in formal_results]
    csv_path = run_root / "performance_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "schema_version": "1.0",
        "profile": args.profile,
        "comparison_policy": profiles["comparison_policy"],
        "gpu_preflight": preflight,
        "warmup_results": [
            result
            for result in all_results
            if result["environment"]["run_kind"] == "warmup"
        ],
        "formal_results": formal_results,
        "formal_aggregate": aggregate(rows),
    }
    json_path = run_root / "performance_results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["status"] = "completed"
    manifest["results_csv"] = csv_path.name
    manifest["results_json"] = json_path.name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_root": str(run_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
