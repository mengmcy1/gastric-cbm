"""Run the frozen SHARP configuration on the complete independent set.

Samples are assigned round-robin to explicitly selected idle physical GPUs. Each
GPU processes its assigned samples sequentially, while different GPUs run in
parallel. Existing complete sample phases may only be reused with ``--resume``;
incomplete directories are never overwritten automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from run_performance_benchmark import gpu_preflight, sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "09_独立验证" / "validation_manifest.csv"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "09_独立验证" / "validation_config.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs" / "09_独立验证"
GENERATION_WORKER = Path(__file__).resolve().parent / "sharp_performance_worker.py"
RENDER_WORKER = Path(__file__).resolve().parent / "sharp_prune_render.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--physical-gpu-indices",
        required=True,
        help="逗号分隔的空闲物理 GPU，例如 0,2,3",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_gpu_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"GPU 编号为空或重复：{value!r}")
    return indices


def load_and_validate_samples(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        raise ValueError("独立验证清单为空")
    ids: set[str] = set()
    hashes: set[str] = set()
    samples: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["validation_id"]
        if sample_id in ids:
            raise ValueError(f"重复 validation_id：{sample_id}")
        input_path = REPO_ROOT / row["input_path"]
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        actual_hash = sha256(input_path)
        expected_hash = row["input_sha256"].lower()
        if actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"{sample_id} 输入 SHA256 不匹配：{actual_hash} != {expected_hash}"
            )
        if actual_hash in hashes:
            raise ValueError(f"清单包含重复图片：{sample_id}")
        ids.add(sample_id)
        hashes.add(actual_hash)
        samples.append({**row, "input_path_resolved": input_path.resolve()})
    return samples


def run_command(
    command: list[str],
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> float:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"命令失败，returncode={completed.returncode}，查看 {stderr_path}"
        )
    return elapsed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_complete(sample_dir: Path) -> bool:
    required = [
        sample_dir / "generation" / "metrics.json",
        sample_dir / "generation" / "scene_full.ply",
        sample_dir / "render" / "config.json",
        sample_dir / "render" / "static_metrics.json",
        sample_dir / "render" / "video_manual.json",
        sample_dir / "render" / "color.mp4",
        sample_dir / "render" / "depth.mp4",
    ]
    return all(path.is_file() for path in required)


def collect_result(
    sample: dict[str, Any], sample_dir: Path, physical_gpu: int, resumed: bool
) -> dict[str, Any]:
    generation = json.loads(
        (sample_dir / "generation" / "metrics.json").read_text(encoding="utf-8")
    )
    render_config = json.loads(
        (sample_dir / "render" / "config.json").read_text(encoding="utf-8")
    )
    static = json.loads(
        (sample_dir / "render" / "static_metrics.json").read_text(encoding="utf-8")
    )
    ply_path = sample_dir / "generation" / "scene_full.ply"
    return {
        "validation_id": sample["validation_id"],
        "input_path": sample["input_path"],
        "input_sha256": sample["input_sha256"],
        "physical_gpu_index": physical_gpu,
        "gpu_name": generation["environment"]["gpu_name"],
        "resumed_existing_result": resumed,
        "generation_worker_wall_s": generation.get("worker_wall_s"),
        "network_forward_s": generation.get("timings_s", {}).get(
            "network_forward_s"
        ),
        "unproject_postprocess_s": generation.get("timings_s", {}).get(
            "unproject_postprocess_s"
        ),
        "gaussian_count": generation.get("gaussian_count"),
        "ply_bytes": ply_path.stat().st_size,
        "ply_sha256": file_sha256(ply_path),
        "render_total_s": render_config.get("total_seconds"),
        "render_resolution": render_config.get("render_resolution"),
        "color_video": render_config.get("encoded_color_video"),
        "depth_video": render_config.get("encoded_depth_video"),
        "psnr_db_vs_input": static.get("psnr_db"),
        "ssim_vs_input": static.get("ssim"),
        "lpips_alex_vs_input": static.get("lpips_alex"),
        "manual_review_status": "pending",
    }


def main() -> None:
    args = parse_args()
    gpu_indices = parse_gpu_indices(args.physical_gpu_indices)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    samples = load_and_validate_samples(args.manifest.resolve())
    if len(samples) != int(config["sample_count"]):
        raise ValueError(
            f"配置样本数 {config['sample_count']} 与清单 {len(samples)} 不一致"
        )
    checkpoint = REPO_ROOT / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash.lower() != config["checkpoint_sha256"].lower():
        raise RuntimeError("冻结权重 SHA256 不匹配")

    profile = {
        "expected_gpu_name": "NVIDIA GeForce RTX 5080",
        "max_preflight_gpu_util_percent": 5,
        "max_preflight_memory_used_fraction": 0.1,
    }
    preflight = {
        str(index): gpu_preflight(index, profile) for index in gpu_indices
    }

    run_root = (args.output_root / args.run_id).resolve()
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"运行目录已存在，拒绝覆盖：{run_root}")
    run_root.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = run_root / "run_manifest.json"
    if args.resume and manifest_path.is_file():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_manifest["status"] = "running"
        run_manifest["resume_count"] = int(run_manifest.get("resume_count", 0)) + 1
        run_manifest["gpu_preflight_resume"] = preflight
    else:
        run_manifest = {
            "schema_version": "1.0-independent-validation",
            "status": "running",
            "run_id": args.run_id,
            "sample_count": len(samples),
            "manifest": str(args.manifest.resolve()),
            "config": config,
            "checkpoint_sha256_verified": checkpoint_hash,
            "physical_gpu_indices": gpu_indices,
            "gpu_preflight": preflight,
            "python_executable": sys.executable,
            "resume_count": 0,
        }
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    assignments: dict[int, list[dict[str, Any]]] = {
        index: [] for index in gpu_indices
    }
    for position, sample in enumerate(samples):
        assignments[gpu_indices[position % len(gpu_indices)]].append(sample)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    progress_path = run_root / "progress.json"
    fixed = config["fixed_variables"]

    def write_progress() -> None:
        progress_path.write_text(
            json.dumps(
                {
                    "completed": len(results),
                    "failed": len(failures),
                    "total": len(samples),
                    "results": sorted(results, key=lambda item: item["validation_id"]),
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def process_gpu(physical_gpu: int) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        python_bin = str(Path(sys.executable).resolve().parent)
        environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
        for sample in assignments[physical_gpu]:
            sample_id = sample["validation_id"]
            sample_dir = run_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                if sample_complete(sample_dir):
                    if not args.resume:
                        raise FileExistsError(f"{sample_id} 已存在完整结果")
                    item = collect_result(sample, sample_dir, physical_gpu, True)
                else:
                    generation_dir = sample_dir / "generation"
                    generation_metrics = generation_dir / "metrics.json"
                    ply_path = generation_dir / "scene_full.ply"
                    if generation_dir.exists() and not (
                        generation_metrics.is_file() and ply_path.is_file()
                    ):
                        raise FileExistsError(
                            f"{sample_id} generation 目录不完整，拒绝覆盖"
                        )
                    if not generation_metrics.is_file():
                        generation_command = [
                            sys.executable,
                            str(GENERATION_WORKER),
                            "--mode",
                            "generation",
                            "--output-dir",
                            str(generation_dir),
                            "--input-image",
                            str(sample["input_path_resolved"]),
                            "--checkpoint",
                            str(checkpoint),
                            "--max-disparity",
                            str(fixed["max_disparity"]),
                            "--crop-single-side-percent",
                            str(fixed["crop_single_side_percent"]),
                            "--num-steps",
                            str(fixed["num_steps"]),
                            "--fps",
                            str(fixed["fps"]),
                            "--run-kind",
                            "formal",
                            "--repeat-index",
                            "1",
                            "--profile",
                            "independent_validation",
                            "--physical-gpu-index",
                            str(physical_gpu),
                        ]
                        run_command(
                            generation_command,
                            environment,
                            sample_dir / "generation_stdout.log",
                            sample_dir / "generation_stderr.log",
                        )

                    render_dir = sample_dir / "render"
                    render_config = render_dir / "config.json"
                    if render_dir.exists() and not render_config.is_file():
                        raise FileExistsError(
                            f"{sample_id} render 目录不完整，拒绝覆盖"
                        )
                    if not render_config.is_file():
                        render_command = [
                            sys.executable,
                            str(RENDER_WORKER),
                            "--ply",
                            str(ply_path),
                            "--input-image",
                            str(sample["input_path_resolved"]),
                            "--output-dir",
                            str(render_dir),
                            "--keep-percent",
                            "100",
                            "--crop-single-side-percent",
                            str(fixed["crop_single_side_percent"]),
                            "--max-disparity",
                            str(fixed["max_disparity"]),
                            "--num-steps",
                            str(fixed["num_steps"]),
                            "--fps",
                            str(fixed["fps"]),
                            "--device",
                            "cuda",
                            "--manual-schema",
                            "validation",
                        ]
                        run_command(
                            render_command,
                            environment,
                            sample_dir / "render_stdout.log",
                            sample_dir / "render_stderr.log",
                        )
                    item = collect_result(sample, sample_dir, physical_gpu, False)
                with lock:
                    results.append(item)
                    write_progress()
            except Exception as error:
                failure = {
                    "validation_id": sample_id,
                    "physical_gpu_index": physical_gpu,
                    "error": str(error),
                }
                (sample_dir / "failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                with lock:
                    failures.append(failure)
                    write_progress()

    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        futures = [executor.submit(process_gpu, index) for index in gpu_indices]
        for future in futures:
            future.result()

    ordered = sorted(results, key=lambda item: item["validation_id"])
    summary_path = run_root / "automatic_results.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0-independent-validation-auto",
                "completed": len(ordered),
                "failed": len(failures),
                "total": len(samples),
                "results": ordered,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if ordered:
        csv_path = run_root / "automatic_results.csv"
        flat_rows = []
        for item in ordered:
            flat_rows.append(
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {"render_resolution", "color_video", "depth_video"}
                    },
                    "render_resolution": "x".join(
                        str(value) for value in item["render_resolution"]
                    ),
                    "color_video_frames": item["color_video"]["frame_count"],
                    "color_video_resolution": "x".join(
                        str(value) for value in item["color_video"]["resolution"]
                    ),
                    "depth_video_frames": item["depth_video"]["frame_count"],
                }
            )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    run_manifest["status"] = "completed" if not failures else "completed_with_failures"
    run_manifest["completed_samples"] = len(ordered)
    run_manifest["failed_samples"] = len(failures)
    run_manifest["automatic_results"] = summary_path.name
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "completed": len(ordered),
                "failed": len(failures),
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise RuntimeError(f"独立验证存在 {len(failures)} 个失败样本")


if __name__ == "__main__":
    main()
