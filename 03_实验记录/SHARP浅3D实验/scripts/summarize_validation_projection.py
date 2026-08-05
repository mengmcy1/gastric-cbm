"""Compute left/right endpoint projection-proxy statistics for validation PLYs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from sharp_experiment_render import _compute_proxy_stats, project_points


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

# sharp_experiment_render has already inserted the local SHARP source path.
from sharp.utils import camera  # noqa: E402
from sharp.utils.gaussians import load_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-disparity", type=float, default=0.04)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--output-stem", default="projection_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    automatic_path = run_root / "automatic_results.json"
    if not automatic_path.is_file():
        raise FileNotFoundError(automatic_path)
    automatic = json.loads(automatic_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for item in automatic["results"]:
        sample_id = item["validation_id"]
        ply_path = run_root / sample_id / "generation" / "scene_full.ply"
        gaussians, metadata = load_ply(ply_path)
        width, height = (int(value) for value in metadata.resolution_px)
        focal = float(metadata.focal_length_px)
        intrinsics = torch.tensor(
            [
                [focal, 0, (width - 1) / 2, 0],
                [0, focal, (height - 1) / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        camera_model = camera.create_camera_model(
            gaussians, intrinsics, resolution_px=metadata.resolution_px
        )
        trajectory = camera.create_eye_trajectory(
            gaussians,
            camera.TrajectoryParams(
                type="swipe",
                max_disparity=args.max_disparity,
                num_steps=args.num_steps,
            ),
            metadata.resolution_px,
            focal,
        )
        center_info = camera_model.compute(torch.zeros(3, dtype=torch.float32))
        render_width = int(center_info.width)
        render_height = int(center_info.height)
        means = gaussians.mean_vectors[0]
        center_uv, center_depth = project_points(
            means, center_info.extrinsics, center_info.intrinsics
        )
        endpoints = {}
        for label, eye in (("left", trajectory[0]), ("right", trajectory[-1])):
            camera_info = camera_model.compute(eye)
            endpoints[label] = _compute_proxy_stats(
                means,
                camera_info,
                center_uv,
                center_depth,
                render_width,
                render_height,
            )
        rows.append(
            {
                "validation_id": sample_id,
                "render_width": render_width,
                "render_height": render_height,
                "left_p95_px": endpoints["left"]["proxy_p95_px"],
                "left_p95_width_percent": endpoints["left"][
                    "proxy_p95_width_percent"
                ],
                "left_p99_px": endpoints["left"]["proxy_p99_px"],
                "left_max_px": endpoints["left"]["proxy_max_px"],
                "right_p95_px": endpoints["right"]["proxy_p95_px"],
                "right_p95_width_percent": endpoints["right"][
                    "proxy_p95_width_percent"
                ],
                "right_p99_px": endpoints["right"]["proxy_p99_px"],
                "right_max_px": endpoints["right"]["proxy_max_px"],
                "max_rejected_nonpositive_depth_count": max(
                    endpoints["left"]["rejected_nonpositive_depth_count"],
                    endpoints["right"]["rejected_nonpositive_depth_count"],
                ),
            }
        )

    csv_path = run_root / f"{args.output_stem}.csv"
    json_path = run_root / f"{args.output_stem}.json"
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有投影汇总：{path}")
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    endpoint_p95_px = [
        float(row[key])
        for row in rows
        for key in ("left_p95_px", "right_p95_px")
    ]
    endpoint_p95_percent = [
        float(row[key])
        for row in rows
        for key in ("left_p95_width_percent", "right_p95_width_percent")
    ]
    result = {
        "schema_version": "1.0-validation-projection",
        "max_disparity": args.max_disparity,
        "trajectory": "swipe",
        "num_steps": args.num_steps,
        "proxy_definition": (
            "Gaussian-center projection displacement relative to the strict-zero "
            "camera; valid centers must be in both canvases with positive depth."
        ),
        "sample_count": len(rows),
        "endpoint_p95_px_range": [min(endpoint_p95_px), max(endpoint_p95_px)],
        "endpoint_p95_width_percent_range": [
            min(endpoint_p95_percent),
            max(endpoint_p95_percent),
        ],
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
