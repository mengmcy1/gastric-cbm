"""批量运行高斯保留率实验：5 样本 × 4 档位 = 20 组。"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent / "sharp_prune_render.py"
PYTHON = sys.executable

SAMPLES = {
    "P01": ("P01_building",       "input.jpg"),
    "P02": ("P02_occlusion_flower","input.jpg"),
    "P03": ("P03_ferris_wheel",    "input.jpg"),
    "P04": ("P04_person_wall_shadow","input.jpg"),
    "P05": ("P05_night_reflection","input.jpg"),
}
KEEP_PERCENTS = [100, 75, 50, 25]
MAX_DISPARITY = 0.04
NUM_STEPS = 60

PILOT_BASE = REPO_ROOT / "03_实验记录" / "SHARP浅3D实验" / "01_先导样本"
OUTPUT_BASE = REPO_ROOT / "03_实验记录" / "SHARP浅3D实验" / "outputs" / "06_剪枝实验_crop03正式"


def main() -> None:
    total = len(SAMPLES) * len(KEEP_PERCENTS)
    done = 0
    errors = []

    for sample_id, (folder, img_name) in SAMPLES.items():
        ply = PILOT_BASE / folder / "scene_full.ply"
        img = PILOT_BASE / folder / img_name

        for kp in KEEP_PERCENTS:
            out = OUTPUT_BASE / sample_id / f"keep{kp:03d}"
            done += 1
            print(f"\n[{done}/{total}] {sample_id} keep={kp}% crop=3%  → {out.relative_to(REPO_ROOT)}")

            cmd = [
                PYTHON, str(SCRIPT),
                "--ply", str(ply),
                "--input-image", str(img),
                "--output-dir", str(out),
                "--keep-percent", str(kp),
                "--crop-single-side-percent", "3",
                "--max-disparity", str(MAX_DISPARITY),
                "--num-steps", str(NUM_STEPS),
            ]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                errors.append(f"{sample_id}/keep{kp:03d}")

    print(f"\n{'='*50}")
    print(f"完成 {total - len(errors)}/{total}，失败 {len(errors)} 组")
    if errors:
        print("失败列表：", errors)


if __name__ == "__main__":
    main()