from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0+1 正式矩阵调度：每个样本/角度固定同一基础 PLY，生成左右补全和 A/B。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_command(name: str, command: list[str], logs_dir: Path, records: list[dict]) -> None:
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - start
    (logs_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    record = {"name": name, "command": command, "returncode": result.returncode, "elapsed_seconds": elapsed}
    records.append(record)
    if result.returncode != 0:
        raise RuntimeError(f"矩阵步骤失败：{name}，详见 {logs_dir}")


def build_commands(config: dict, output_root: Path) -> list[tuple[str, list[str]]]:
    py = sys.executable
    checkpoint = repo_path(config["sharp_checkpoint"])
    lama_model = repo_path(config["lama_model"])
    commands: list[tuple[str, list[str]]] = []
    for sample in config["samples"]:
        sample_id = sample["sample_id"]
        base_ply = repo_path(sample["base_ply"])
        focal_px_render = sample["focal_px_render"]
        for angle in config["angles"]:
            angle_total = float(angle["angle_total_deg"])
            angle_tag = f"ang{int(angle_total)}"
            run_dir = output_root / sample_id / angle_tag
            render_a = run_dir / "render_a"
            commands.append(
                (
                    f"{sample_id}_{angle_tag}_01_render_a",
                    [
                        py, str(SCRIPT_DIR / "stage01_angle_render.py"),
                        "--ply", str(base_ply), "--output-dir", str(render_a),
                        "--angle-total", str(angle_total), "--num-steps", str(angle["num_steps"]),
                        "--trajectory-mode", config["trajectory_mode"],
                    ],
                )
            )
            supplement_paths = []
            for side in ("left", "right"):
                prefix = f"{sample_id}_{angle_tag}_{side}"
                mask_dir = run_dir / f"{side}_mask"
                lama_dir = run_dir / f"{side}_lama"
                depth_dir = run_dir / f"{side}_depth"
                supplement_dir = run_dir / f"{side}_supplement"
                supplement_paths.append(supplement_dir / "supplement.ply")
                commands.extend(
                    [
                        (
                            f"{prefix}_02_mask",
                            [
                                py, str(SCRIPT_DIR / "stage1_detect_disocclusion.py"),
                                "--endpoint-alpha", str(render_a / f"alpha_{side}_u16.png"),
                                "--center-alpha", str(render_a / "alpha_center_u16.png"),
                                "--endpoint-depth", str(render_a / f"depth_{side}_float32.npy"),
                                "--endpoint-rgb", str(render_a / f"frame_{side}.png"),
                                "--output-dir", str(mask_dir),
                                "--alpha-threshold", str(config["mask"]["alpha_threshold"]),
                                "--hard-hole-threshold", str(config["mask"]["hard_hole_threshold"]),
                                "--dilate-px", str(config["mask"]["dilate_px"]),
                                "--min-component-px", str(config["mask"]["min_component_px"]),
                                "--boundary-band-px", str(config["mask"]["boundary_band_px"]),
                            ],
                        ),
                        (
                            f"{prefix}_03_lama",
                            [
                                py, str(SCRIPT_DIR / "stage1_lama_inpaint.py"),
                                "--image", str(render_a / f"frame_{side}.png"),
                                "--mask", str(mask_dir / "mask_accepted.png"),
                                "--model", str(lama_model), "--output-dir", str(lama_dir),
                                "--device", config["lama"]["device"],
                                "--max-side", str(config["lama"]["max_side"]),
                            ],
                        ),
                        (
                            f"{prefix}_04_depth",
                            [
                                py, str(SCRIPT_DIR / "stage1_estimate_align_depth.py"),
                                "--image", str(lama_dir / "inpaint_composited.png"),
                                "--mask", str(mask_dir / "mask_accepted.png"),
                                "--reference-depth", str(render_a / f"depth_{side}_float32.npy"),
                                "--checkpoint", str(checkpoint), "--focal-px", str(focal_px_render),
                                "--output-dir", str(depth_dir),
                                "--depth-layer", str(config["depth"]["layer"]),
                                "--ring-px", str(config["depth"]["ring_px"]),
                                "--alignment-model", config["depth"].get("alignment_model", "depth_affine"),
                                "--max-normalized-rmse", str(config["depth"]["max_normalized_rmse"]),
                            ] + (["--allow-poor-alignment"] if config["depth"].get("allow_poor_alignment", False) else []),
                        ),
                        (
                            f"{prefix}_05_supplement",
                            [
                                py, str(SCRIPT_DIR / "stage1_build_supplement_gaussians.py"), "build",
                                "--base-ply", str(base_ply),
                                "--rgb", str(lama_dir / "inpaint_composited.png"),
                                "--depth", str(depth_dir / "depth_filled_float32.npy"),
                                "--mask", str(depth_dir / "mask_depth_accepted.png"),
                                "--angle-total", str(angle_total), "--side", side,
                                "--trajectory-mode", config["trajectory_mode"],
                                "--output-dir", str(supplement_dir),
                                "--stride", str(config["gaussians"]["stride"]),
                            ],
                        ),
                    ]
                )
            merge_dir = run_dir / "merged"
            merge_command = [
                py, str(SCRIPT_DIR / "stage1_build_supplement_gaussians.py"), "merge",
                "--base-ply", str(base_ply),
            ]
            for supplement in supplement_paths:
                merge_command.extend(["--supplement-ply", str(supplement)])
            merge_command.extend(["--output-dir", str(merge_dir)])
            commands.append((f"{sample_id}_{angle_tag}_06_merge", merge_command))
            render_b = run_dir / "render_b"
            commands.append(
                (
                    f"{sample_id}_{angle_tag}_07_render_b",
                    [
                        py, str(SCRIPT_DIR / "stage01_angle_render.py"),
                        "--ply", str(base_ply),
                        "--supplement-left-ply", str(supplement_paths[0]),
                        "--supplement-right-ply", str(supplement_paths[1]),
                        "--visibility-mode", config["visibility"]["mode"],
                        "--output-dir", str(render_b), "--angle-total", str(angle_total),
                        "--num-steps", str(angle["num_steps"]),
                        "--trajectory-mode", config["trajectory_mode"],
                    ],
                )
            )
            commands.append(
                (
                    f"{sample_id}_{angle_tag}_08_summary",
                    [
                        py, str(SCRIPT_DIR / "stage1_summarize_ab.py"),
                        "--baseline-dir", str(render_a), "--completed-dir", str(render_b),
                        "--evaluation-mask-left", str(run_dir / "left_depth" / "mask_depth_accepted.png"),
                        "--evaluation-mask-right", str(run_dir / "right_depth" / "mask_depth_accepted.png"),
                        "--output-dir", str(run_dir / "summary"),
                    ],
                )
            )
            commands.append(
                (
                    f"{sample_id}_{angle_tag}_09_validate",
                    [
                        py, str(SCRIPT_DIR / "stage01_validate_outputs.py"),
                        "--run-root", str(run_dir),
                        "--expected-frames", str(angle["num_steps"]),
                        "--output", str(run_dir / "integrity.json"),
                        "--require-right",
                    ],
                )
            )
    return commands


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_root = prepare_output_dir(args.output_root)
    for path, expected in (
        (repo_path(config["sharp_checkpoint"]), config["hashes"]["sharp_checkpoint_sha256"]),
        (repo_path(config["lama_model"]), config["hashes"]["lama_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_hash(path) != expected.upper():
            raise ValueError(f"哈希不匹配：{path}")
    for sample in config["samples"]:
        path = repo_path(sample["base_ply"])
        if not path.is_file() or file_hash(path) != sample["base_ply_sha256"].upper():
            raise ValueError(f"样本 PLY 缺失或哈希不匹配：{sample['sample_id']}")

    commands = build_commands(config, output_root)
    write_json(
        output_root / "execution_plan.json",
        {
            "schema_version": "1.0-stage01-matrix-plan",
            "config": str(args.config.resolve()),
            "dry_run": args.dry_run,
            "command_count": len(commands),
            "commands": [{"name": name, "command": command} for name, command in commands],
            "note": "简单左右 PLY 追加属于 Stage 1；跨端点冲突消解与一致性优化属于 Stage 2。",
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "commands": len(commands)}, ensure_ascii=False))
        return

    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    records: list[dict] = []
    try:
        for name, command in commands:
            run_command(name, command, logs_dir, records)
    finally:
        write_json(
            output_root / "run_manifest.json",
            {
                "schema_version": "1.0-stage01-matrix-run",
                "status": "completed" if len(records) == len(commands) and all(r["returncode"] == 0 for r in records) else "failed",
                "steps": records,
            },
        )


if __name__ == "__main__":
    main()
