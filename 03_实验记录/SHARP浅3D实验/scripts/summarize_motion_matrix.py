"""汇总 SHARP 五样本四档 60 帧运动矩阵的自动指标。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    motion_root = args.motion_root.resolve()
    csv_path = motion_root / "motion_results.csv"
    json_path = motion_root / "motion_results.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(f"拒绝覆盖已有汇总：{', '.join(path.name for path in existing)}")

    summaries: list[dict] = []
    for sample_dir in sorted(path for path in motion_root.iterdir() if path.is_dir()):
        for config_dir in sorted(path for path in sample_dir.iterdir() if path.is_dir()):
            config = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
            rows = read_rows(config_dir / "per_frame_metrics.csv")
            if len(rows) != 60:
                raise ValueError(f"{config_dir}: 逐帧记录不是 60 行，而是 {len(rows)}")
            if config["encoded_color_video"]["frame_count"] != 60:
                raise ValueError(f"{config_dir}: 彩色视频不是 60 帧")
            if config["encoded_depth_video"]["frame_count"] != 60:
                raise ValueError(f"{config_dir}: 深度视频不是 60 帧")
            if config["encoded_color_video"]["resolution"] != config["render_resolution"]:
                raise ValueError(f"{config_dir}: 彩色视频编码尺寸与渲染尺寸不一致")
            if config["encoded_depth_video"]["resolution"] != config["render_resolution"]:
                raise ValueError(f"{config_dir}: 深度视频编码尺寸与渲染尺寸不一致")

            left, right = rows[0], rows[-1]
            adjacent = [(int(row["frame"]), as_float(row, "adjacent_ssim")) for row in rows if row["adjacent_ssim"]]
            min_adjacent_frame, min_adjacent_ssim = min(adjacent, key=lambda item: item[1])
            warm_render_ms = [as_float(row, "render_ms") for row in rows[1:]]
            max_depth_rejected = max(int(row["rejected_nonpositive_depth_count"]) for row in rows)

            summary = {
                "sample_id": sample_dir.name,
                "config_name": config_dir.name,
                "max_disparity": float(config["max_disparity"]),
                "frame_count": len(rows),
                "render_width": int(config["render_resolution"][0]),
                "render_height": int(config["render_resolution"][1]),
                "gaussian_count": int(config["gaussian_count"]),
                "left_p95_px": round(as_float(left, "proxy_p95_px"), 6),
                "left_p95_width_percent": round(as_float(left, "proxy_p95_width_percent"), 6),
                "left_p99_px": round(as_float(left, "proxy_p99_px"), 6),
                "left_max_px": round(as_float(left, "proxy_max_px"), 6),
                "right_p95_px": round(as_float(right, "proxy_p95_px"), 6),
                "right_p95_width_percent": round(as_float(right, "proxy_p95_width_percent"), 6),
                "right_p99_px": round(as_float(right, "proxy_p99_px"), 6),
                "right_max_px": round(as_float(right, "proxy_max_px"), 6),
                "min_alpha_ge_099": round(min(as_float(row, "alpha_ge_099") for row in rows), 6),
                "min_alpha_ge_095": round(min(as_float(row, "alpha_ge_095") for row in rows), 6),
                "min_adjacent_ssim": round(min_adjacent_ssim, 6),
                "min_adjacent_ssim_frame": min_adjacent_frame,
                "median_warm_render_ms": round(statistics.median(warm_render_ms), 3),
                "total_seconds": float(config["total_seconds"]),
                "max_rejected_nonpositive_depth_count": max_depth_rejected,
                "relative_output_dir": str(config_dir.relative_to(motion_root.parent.parent)).replace("\\", "/"),
            }
            summaries.append(summary)

    if len(summaries) != 20:
        raise ValueError(f"期望 20 个配置，实际找到 {len(summaries)} 个")

    fieldnames = list(summaries[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config_count": len(summaries), "csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
