from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按冻结方案 B 运行并验收 P01 真圆弧 A 基线。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)["streams"][0]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_numbers(row: dict) -> dict:
    integer_keys = {"frame", "valid_gaussian_count"}
    return {
        key: int(value) if key in integer_keys else float(value)
        for key, value in row.items()
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if config.get("status") != "frozen" or config.get("trajectory_mode") != "true_arc":
        raise ValueError("Stage 0 正式入口只接受 status=frozen 的 true_arc 配置。")
    if float(config.get("fixed_angle_step_deg", -1)) != 0.5:
        raise ValueError("冻结协议要求固定 0.5° 步长。")

    base_ply = repo_path(config["base_ply"])
    if not base_ply.is_file():
        raise FileNotFoundError(base_ply)
    observed_hash = file_hash(base_ply)
    if observed_hash != config["base_ply_sha256"].upper():
        raise ValueError(f"P01 PLY SHA256 不匹配：{observed_hash}")

    output_root = prepare_output_dir(args.output_root)
    commands = []
    for item in config["angles"]:
        total = float(item["angle_total_deg"])
        steps = int(item["num_steps"])
        expected_steps = round(total / 0.5) + 1
        if steps != expected_steps:
            raise ValueError(f"{total}° 帧数应为 {expected_steps}，配置为 {steps}。")
        commands.append(
            {
                "angle_total_deg": total,
                "num_steps": steps,
                "output_dir": output_root / f"ang{int(total)}" / "render_a",
                "command": [
                    sys.executable,
                    str(SCRIPT_DIR / "stage01_angle_render.py"),
                    "--ply",
                    str(base_ply),
                    "--output-dir",
                    str(output_root / f"ang{int(total)}" / "render_a"),
                    "--angle-total",
                    str(total),
                    "--num-steps",
                    str(steps),
                    "--trajectory-mode",
                    "true_arc",
                    "--fps",
                    str(config["fps"]),
                ],
            }
        )

    write_json(
        output_root / "execution_plan.json",
        {
            "schema_version": "1.0-stage0-a-plan",
            "config": str(args.config.resolve()),
            "config_snapshot": config,
            "dry_run": args.dry_run,
            "commands": [
                {**{k: v for k, v in item.items() if k != "output_dir"}}
                for item in commands
            ],
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "commands": len(commands)}))
        return

    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    run_records = []
    try:
        for item in commands:
            started = time.perf_counter()
            result = subprocess.run(
                item["command"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.perf_counter() - started
            tag = f"ang{int(item['angle_total_deg'])}"
            (logs_dir / f"{tag}.stdout.txt").write_text(result.stdout, encoding="utf-8")
            (logs_dir / f"{tag}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            run_records.append(
                {
                    "angle_total_deg": item["angle_total_deg"],
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                    "command": item["command"],
                }
            )
            if result.returncode != 0:
                raise RuntimeError(f"{tag} A 基线失败，详见 {logs_dir}。")
    finally:
        write_json(
            output_root / "run_manifest.json",
            {
                "schema_version": "1.0-stage0-a-run",
                "status": "completed"
                if len(run_records) == len(commands)
                and all(item["returncode"] == 0 for item in run_records)
                else "failed",
                "steps": run_records,
            },
        )

    summaries = []
    center_rgbs = []
    center_alphas = []
    failures = []
    for item in commands:
        total = item["angle_total_deg"]
        render_dir = item["output_dir"]
        render_config = read_json(render_dir / "config.json")
        rows = [row_numbers(row) for row in load_rows(render_dir / "per_frame_metrics.csv")]
        left, center, right = rows[0], rows[len(rows) // 2], rows[-1]
        video = probe_video(render_dir / "color.mp4")
        center_rgbs.append(np.asarray(Image.open(render_dir / "frame_center.png").convert("RGB")))
        center_alphas.append(np.asarray(Image.open(render_dir / "alpha_center_u16.png")))

        symmetry = {
            "eye_x_sum_abs": abs(left["eye_x"] + right["eye_x"]),
            "eye_y_diff_abs": abs(left["eye_y"] - right["eye_y"]),
            "eye_z_diff_abs": abs(left["eye_z"] - right["eye_z"]),
            "view_direction_x_sum_abs": abs(
                left["view_direction_x"] + right["view_direction_x"]
            ),
            "view_direction_y_diff_abs": abs(
                left["view_direction_y"] - right["view_direction_y"]
            ),
            "view_direction_z_diff_abs": abs(
                left["view_direction_z"] - right["view_direction_z"]
            ),
        }
        checks = {
            "frame_count": int(video["nb_read_frames"]) == int(item["num_steps"]),
            "codec_h264": video["codec_name"] == "h264",
            "pix_fmt_yuv420p": video["pix_fmt"] == "yuv420p",
            "resolution_matches": [int(video["width"]), int(video["height"])]
            == render_config["render_resolution_wh"],
            "angle_step_0p5": abs(float(render_config["angle_step_deg"]) - 0.5) < 1e-9,
            "center_eye_origin": max(abs(value) for value in render_config["center_eye_xyz"])
            < 1e-8,
            "center_camera_matches_original": max(
                render_config["center_camera_consistency"].values()
            )
            < 1e-8,
            "intrinsics_fixed": render_config["fixed_intrinsics_max_abs_diff_across_frames"]
            < 1e-8,
            "endpoint_geometry_symmetric": max(symmetry.values()) < 1e-5,
        }
        if not all(checks.values()):
            failures.append({"angle_total_deg": total, "checks": checks, "symmetry": symmetry})
        summaries.append(
            {
                "angle_total_deg": total,
                "num_steps": item["num_steps"],
                "render_resolution_wh": render_config["render_resolution_wh"],
                "reference_depth_m": render_config["reference_depth_m"],
                "arc_radius_m": render_config["arc_radius_m"],
                "look_at_point_xyz": render_config["look_at_point_xyz"],
                "left": left,
                "center": center,
                "right": right,
                "endpoint_symmetry": symmetry,
                "video": video,
                "checks": checks,
            }
        )

    center_comparisons = []
    for index in range(1, len(center_rgbs)):
        rgb_abs = np.abs(center_rgbs[index].astype(np.int16) - center_rgbs[0].astype(np.int16))
        alpha_abs = np.abs(
            center_alphas[index].astype(np.int32) - center_alphas[0].astype(np.int32)
        )
        comparison = {
            "reference_angle_total_deg": summaries[0]["angle_total_deg"],
            "compared_angle_total_deg": summaries[index]["angle_total_deg"],
            "rgb_max_abs_0_255": int(rgb_abs.max()),
            "rgb_mean_abs_0_255": float(rgb_abs.mean()),
            "alpha_max_abs_u16": int(alpha_abs.max()),
            "alpha_mean_abs_u16": float(alpha_abs.mean()),
        }
        center_comparisons.append(comparison)
        if comparison["rgb_max_abs_0_255"] != 0 or comparison["alpha_max_abs_u16"] != 0:
            failures.append({"center_frame_mismatch": comparison})

    write_json(
        output_root / "stage0_summary.json",
        {
            "schema_version": "1.0-stage0-protocol-b-summary",
            "status": "passed" if not failures else "failed",
            "config": str(args.config.resolve()),
            "base_ply_sha256": observed_hash,
            "angles": summaries,
            "center_frame_cross_angle_comparisons": center_comparisons,
            "failures": failures,
            "conclusion_boundary": (
                "本汇总只验收 Stage 0 相机协议与 A 基线完整性，不评价显露补全质量。"
            ),
        },
    )
    if failures:
        raise RuntimeError(f"Stage 0 协议验收失败：{failures}")
    print(json.dumps({"status": "passed", "angles": len(summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
