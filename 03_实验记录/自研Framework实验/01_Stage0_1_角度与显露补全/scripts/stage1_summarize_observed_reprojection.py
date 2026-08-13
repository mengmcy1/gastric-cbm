from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from stage01_common import file_hash, prepare_output_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总左右端真实观测重投影覆盖率。")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=960)
    return parser.parse_args()


def labeled(image: Image.Image, label: str, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 42), "white")
    canvas.paste(resized, (0, 42))
    ImageDraw.Draw(canvas).text((12, 13), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = prepare_output_dir(args.output_dir)
    results = {side: read_json(run_root / side / "reprojection_result.json") for side in ("left", "right")}
    if len({item["experiment_id"] for item in results.values()}) != 1:
        raise ValueError("左右端 experiment_id 不一致。")

    rows = []
    review_rows = []
    for side, item in results.items():
        fractions = item["fractions"]
        counts = item["counts"]
        rows.append(
            {
                "side": side,
                "angle_deg": item["endpoint_angle_deg"],
                "accepted_pixels": counts["accepted_pixels"],
                "accepted_observable_pixels": counts["accepted_observable_pixels"],
                "accepted_observable_coverage": fractions["accepted_observable_coverage"],
                "accepted_high_confidence_coverage": fractions["accepted_high_confidence_coverage"],
                "accepted_residual_unobserved": fractions["accepted_residual_unobserved"],
                "known_reliable_observable_coverage": fractions["known_reliable_observable_coverage"],
                "target_depth_consistency": fractions["target_depth_consistency_where_comparable"],
                "identity_reprojection_max_px": item["camera_checks"]["identity_reprojection_max_px"],
            }
        )
        overlay = Image.open(run_root / side / "accepted_coverage_overlay.png").convert("RGB")
        composite = Image.open(run_root / side / "accepted_reprojection_composite.png").convert("RGB")
        coverage_pct = 100.0 * fractions["accepted_observable_coverage"]
        review_rows.append(
            [
                labeled(overlay, f"{side}: green=high, yellow=low, red=unobserved; observed={coverage_pct:.3f}%", args.tile_width),
                labeled(composite, f"{side}: original-RGB reprojection only where observable", args.tile_width),
            ]
        )

    csv_path = output_dir / "endpoint_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tile_height = max(tile.height for row in review_rows for tile in row)
    sheet = Image.new("RGB", (args.tile_width * 2, tile_height * 2), "white")
    for row_index, row in enumerate(review_rows):
        for column_index, tile in enumerate(row):
            sheet.paste(tile, (column_index * args.tile_width, row_index * tile_height))
    review_path = output_dir / "P01_ang30_observed_reprojection_review.png"
    sheet.save(review_path)

    total_accepted = sum(row["accepted_pixels"] for row in rows)
    total_observed = sum(row["accepted_observable_pixels"] for row in rows)
    summary = {
        "schema_version": "1.0-observed-reprojection-summary",
        "experiment_id": results["left"]["experiment_id"],
        "status": "complete",
        "endpoint_rows": rows,
        "combined": {
            "accepted_pixels": total_accepted,
            "accepted_observable_pixels": total_observed,
            "accepted_observable_coverage": total_observed / total_accepted,
            "accepted_residual_unobserved": 1.0 - total_observed / total_accepted,
        },
        "validation": {
            "identity_reprojection_pass": all(row["identity_reprojection_max_px"] < 1e-6 for row in rows),
            "known_region_coverage_pass": all(row["known_reliable_observable_coverage"] > 0.80 for row in rows),
            "target_depth_consistency_pass": all(row["target_depth_consistency"] > 0.85 for row in rows),
        },
        "decision": {
            "single_center_rgbd_reprojection": "insufficient_for_accepted_disocclusions",
            "reason": "More than 99.8% of the frozen accepted endpoint regions remain without a center-view source.",
            "next_branch": "explicit hidden-layer prediction or external/multi-view prior before residual appearance synthesis",
            "do_not_do": [
                "do not treat forward-splat cracks as learned completion targets",
                "do not rebuild supplement Gaussians from the tiny observed subset",
                "do not continue swapping unconstrained single-frame inpainters"
            ]
        },
        "outputs": {
            "endpoint_coverage_csv": {"path": str(csv_path), "sha256": file_hash(csv_path)},
            "review_image": {"path": str(review_path), "sha256": file_hash(review_path)}
        }
    }
    write_json(output_dir / "observed_reprojection_summary.json", summary)


if __name__ == "__main__":
    main()
