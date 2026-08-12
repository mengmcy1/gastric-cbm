from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成固定端点二维补全静态人工复核图。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=768)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def labeled_tile(image: Image.Image, label: str, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 36), "white")
    canvas.paste(resized, (0, 36))
    ImageDraw.Draw(canvas).text((12, 10), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    candidate_root = args.candidate_root.resolve()
    output_dir = prepare_output_dir(args.output_dir)
    rows: list[list[Image.Image]] = []
    checks: dict[str, dict] = {}

    for side in ("left", "right"):
        endpoint = manifest["endpoints"][side]
        source = np.asarray(Image.open(repo_path(endpoint["image"]["path"])).convert("RGB"))
        mask = np.asarray(Image.open(repo_path(endpoint["mask"]["path"])).convert("L")) > 0
        lama = np.asarray(Image.open(repo_path(endpoint["lama_baseline"]["path"])).convert("RGB"))
        candidate_path = candidate_root / side / "inpaint_composited.png"
        candidate = np.asarray(Image.open(candidate_path).convert("RGB"))
        if source.shape != lama.shape or source.shape != candidate.shape or source.shape[:2] != mask.shape:
            raise ValueError(f"{side} 的源图、掩码和输出分辨率不一致。")
        known_exact = bool(np.array_equal(candidate[~mask], source[~mask]))
        if not known_exact:
            raise AssertionError(f"{side} 候选输出改写了掩码外像素。")

        overlay = source.copy()
        overlay[mask] = np.round(overlay[mask] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
        rows.append([
            labeled_tile(Image.fromarray(overlay), f"{side}: fixed mask (red)", args.tile_width),
            labeled_tile(Image.fromarray(lama), f"{side}: Big-LaMa baseline", args.tile_width),
            labeled_tile(Image.fromarray(candidate), f"{side}: MAT Places-512", args.tile_width),
        ])
        checks[side] = {
            "known_pixels_exact": known_exact,
            "candidate_sha256": file_hash(candidate_path),
            "hole_fraction": float(mask.mean()),
        }

    tile_height = max(tile.height for row in rows for tile in row)
    sheet = Image.new("RGB", (args.tile_width * 3, tile_height * 2), "white")
    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            sheet.paste(tile, (column_index * args.tile_width, row_index * tile_height))
    review_path = output_dir / "P01_ang30_LaMa_vs_MAT_static_review.png"
    sheet.save(review_path)
    write_json(output_dir / "static_review.json", {
        "schema_version": "1.0-static-inpaint-review",
        "experiment_id": manifest["experiment_id"],
        "layout": "rows=left/right endpoints; columns=fixed mask, Big-LaMa, MAT Places-512",
        "candidate_root": str(candidate_root),
        "checks": checks,
        "review_image": str(review_path),
        "review_image_sha256": file_hash(review_path),
        "manual_review": {
            "left_smear_severity_0_to_3": None,
            "right_smear_severity_0_to_3": None,
            "mat_vs_lama": None,
            "static_gate_pass": None,
            "notes": None
        }
    })


if __name__ == "__main__":
    main()
