#!/usr/bin/env python3
"""Summarize near-black background area in cross-scene scouting videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["scenes"]:
        reader = imageio.get_reader(item["video"])
        fractions = []
        for frame in reader:
            rgb = np.asarray(frame)[..., :3]
            fractions.append(float((rgb.max(axis=-1) <= 5).mean()))
        reader.close()
        sample_indices = [0, 15, 30, 45, len(fractions) - 1]
        rows.append(
            {
                "scene_id": item["scene_id"],
                "scene_class": item["scene_class"],
                "mode": item["mode"],
                "frame_count": len(fractions),
                "near_black_fraction_mean": float(np.mean(fractions)),
                "near_black_fraction_max": float(np.max(fractions)),
                "near_black_start": fractions[sample_indices[0]],
                "near_black_quarter": fractions[sample_indices[1]],
                "near_black_middle": fractions[sample_indices[2]],
                "near_black_three_quarter": fractions[sample_indices[3]],
                "near_black_end": fractions[sample_indices[4]],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_scene_video_black_area_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0-cross-scene-near-black-area-summary",
                "warning": "RGB near-black proxy (max channel <= 5/255), not renderer Alpha; dark scene content can be counted as background.",
                "scenes": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cross_scene_video_black_area_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
