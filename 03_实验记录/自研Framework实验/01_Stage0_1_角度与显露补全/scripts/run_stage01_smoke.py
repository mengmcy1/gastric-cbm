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
    parser = argparse.ArgumentParser(description="运行 P01 左端点 Stage 0+1 最小端到端冒烟测试。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run_step(name: str, command: list[str], logs_dir: Path, records: list[dict]) -> None:
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - start
    (logs_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    records.append(
        {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
        }
    )
    if result.returncode != 0:
        raise RuntimeError(f"步骤 {name} 失败（exit={result.returncode}），详见 {logs_dir}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_root = prepare_output_dir(args.output_root)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    records: list[dict] = []
    py = sys.executable

    base_ply = resolve_repo_path(config["base_ply"])
    angle_source = resolve_repo_path(config["angle_source_dir"])
    checkpoint = resolve_repo_path(config["sharp_checkpoint"])
    lama_model = resolve_repo_path(config["lama_model"])
    for path in (base_ply, angle_source, checkpoint, lama_model):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = config["hashes"]
    for label, path in (("base_ply_sha256", base_ply), ("sharp_checkpoint_sha256", checkpoint)):
        observed = file_hash(path)
        if observed != expected[label].upper():
            raise ValueError(f"{label} 不匹配：{observed}")

    side = config.get("smoke_side", "left")
    common_angle = str(config["angle_total_deg"])
    trajectory_mode = config["trajectory_mode"]

    run_step(
        "01_detect_mask",
        [
            py, str(SCRIPT_DIR / "stage1_detect_disocclusion.py"),
            "--endpoint-alpha", str(angle_source / f"alpha_{side}_u16.png"),
            "--center-alpha", str(angle_source / "alpha_center_u16.png"),
            "--endpoint-depth", str(angle_source / f"depth_{side}_float32.npy"),
            "--endpoint-rgb", str(angle_source / f"frame_{side}.png"),
            "--output-dir", str(output_root / f"{side}_mask"),
            "--alpha-threshold", str(config["mask"]["alpha_threshold"]),
            "--hard-hole-threshold", str(config["mask"]["hard_hole_threshold"]),
            "--dilate-px", str(config["mask"]["dilate_px"]),
        ],
        logs_dir,
        records,
    )
    run_step(
        "02_lama",
        [
            py, str(SCRIPT_DIR / "stage1_lama_inpaint.py"),
            "--image", str(angle_source / f"frame_{side}.png"),
            "--mask", str(output_root / f"{side}_mask" / "mask_final.png"),
            "--model", str(lama_model),
            "--output-dir", str(output_root / f"{side}_lama"),
            "--device", config["lama"]["device"],
            "--max-side", str(config["lama"]["max_side"]),
        ],
        logs_dir,
        records,
    )
    run_step(
        "03_depth",
        [
            py, str(SCRIPT_DIR / "stage1_estimate_align_depth.py"),
            "--image", str(output_root / f"{side}_lama" / "inpaint_composited.png"),
            "--mask", str(output_root / f"{side}_mask" / "mask_final.png"),
            "--reference-depth", str(angle_source / f"depth_{side}_float32.npy"),
            "--checkpoint", str(checkpoint),
            "--focal-px", str(config["focal_px_render"]),
            "--output-dir", str(output_root / f"{side}_depth"),
            "--depth-layer", str(config["depth"]["layer"]),
            "--ring-px", str(config["depth"]["ring_px"]),
        ],
        logs_dir,
        records,
    )
    run_step(
        "04_build_supplement",
        [
            py, str(SCRIPT_DIR / "stage1_build_supplement_gaussians.py"), "build",
            "--base-ply", str(base_ply),
            "--rgb", str(output_root / f"{side}_lama" / "inpaint_composited.png"),
            "--depth", str(output_root / f"{side}_depth" / "depth_filled_float32.npy"),
            "--mask", str(output_root / f"{side}_mask" / "mask_final.png"),
            "--angle-total", common_angle,
            "--side", side,
            "--trajectory-mode", trajectory_mode,
            "--output-dir", str(output_root / f"{side}_supplement"),
            "--stride", str(config["gaussians"]["stride"]),
        ],
        logs_dir,
        records,
    )
    run_step(
        "05_merge",
        [
            py, str(SCRIPT_DIR / "stage1_build_supplement_gaussians.py"), "merge",
            "--base-ply", str(base_ply),
            "--supplement-ply", str(output_root / f"{side}_supplement" / "supplement.ply"),
            "--output-dir", str(output_root / "merged"),
        ],
        logs_dir,
        records,
    )
    for label, ply in (
        ("a", base_ply),
        ("b", output_root / "merged" / "scene_base_plus_supplement.ply"),
    ):
        run_step(
            f"06_render_{label}",
            [
                py, str(SCRIPT_DIR / "stage01_angle_render.py"),
                "--ply", str(ply),
                "--output-dir", str(output_root / f"render_{label}"),
                "--angle-total", common_angle,
                "--num-steps", str(config["smoke_render_frames"]),
                "--trajectory-mode", trajectory_mode,
            ],
            logs_dir,
            records,
        )
    run_step(
        "07_summarize",
        [
            py, str(SCRIPT_DIR / "stage1_summarize_ab.py"),
            "--baseline-dir", str(output_root / "render_a"),
            "--completed-dir", str(output_root / "render_b"),
            "--evaluation-mask", str(output_root / f"{side}_mask" / "mask_final.png"),
            "--output-dir", str(output_root / "summary"),
        ],
        logs_dir,
        records,
    )
    run_step(
        "08_validate",
        [
            py, str(SCRIPT_DIR / "stage01_validate_outputs.py"),
            "--run-root", str(output_root),
            "--expected-frames", str(config["smoke_render_frames"]),
            "--output", str(output_root / "integrity.json"),
        ],
        logs_dir,
        records,
    )
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "1.0-stage01-smoke-run",
            "status": "completed",
            "config": str(args.config.resolve()),
            "config_snapshot": config,
            "steps": records,
            "scope": (
                "P01 单端点、三帧端到端冒烟；仅证明接口可运行，不构成补全质量结论。"
            ),
        },
    )
    print(json.dumps({"status": "completed", "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
