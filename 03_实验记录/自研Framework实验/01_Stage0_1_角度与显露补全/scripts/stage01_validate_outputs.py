from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from stage01_common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Stage 0+1 冒烟输出的存在性、图片尺寸和视频帧数。")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-right", action="store_true")
    return parser.parse_args()


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries",
            "stream=codec_name,pix_fmt,width,height,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    args = parse_args()
    required = [
        "left_mask/mask_final.png",
        "left_lama/inpaint_composited.png",
        "left_depth/depth_filled_float32.npy",
        "left_supplement/supplement.ply",
        "merged/scene_base_plus_supplement.ply",
        "render_a/color.mp4",
        "render_b/color.mp4",
        "summary/ab_summary.json",
    ]
    if args.require_right:
        required.extend(
            [
                "right_mask/mask_final.png",
                "right_lama/inpaint_composited.png",
                "right_depth/depth_filled_float32.npy",
                "right_supplement/supplement.ply",
            ]
        )
    checks = []
    for relative in required:
        path = args.run_root / relative
        checks.append({"path": relative, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None})
    failures = [item for item in checks if not item["exists"]]
    videos = {}
    for label in ("render_a", "render_b"):
        path = args.run_root / label / "color.mp4"
        if path.is_file():
            videos[label] = probe_video(path)
            if int(videos[label].get("nb_read_frames", -1)) != args.expected_frames:
                failures.append({"path": str(path), "reason": "frame_count_mismatch"})
    image_sizes = {}
    for label in ("render_a", "render_b"):
        path = args.run_root / label / "frame_center.png"
        if path.is_file():
            image_sizes[label] = list(Image.open(path).size)
    if len(image_sizes) == 2 and image_sizes["render_a"] != image_sizes["render_b"]:
        failures.append({"reason": "ab_resolution_mismatch", "sizes": image_sizes})
    report = {
        "schema_version": "1.0-stage01-integrity",
        "run_root": str(args.run_root.resolve()),
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "videos": videos,
        "image_sizes_wh": image_sizes,
        "failures": failures,
    }
    write_json(args.output, report)
    if failures:
        raise RuntimeError(f"完整性检查失败：{failures}")


if __name__ == "__main__":
    main()
