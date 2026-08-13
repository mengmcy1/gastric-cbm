from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stage01_common import REPO_ROOT, file_hash, prepare_output_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 P01 30°左右端点 A/B/C/D 静态复核图。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mat-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=640)
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def tile(array: np.ndarray, label: str, width: int) -> Image.Image:
    image = Image.fromarray(array)
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 40), "white")
    canvas.paste(image, (0, 40))
    ImageDraw.Draw(canvas).text((12, 12), label, fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("experiment_id") != "P01_ang30_static_inpaint_ab_v1":
        raise ValueError("拒绝使用未冻结的静态 A/B 清单。")
    output_dir = prepare_output_dir(args.output_dir)
    rows: list[list[Image.Image]] = []
    checks: dict[str, dict] = {}

    for side in ("left", "right"):
        endpoint = manifest["endpoints"][side]
        source_path = repo_path(endpoint["image"]["path"])
        mask_path = repo_path(endpoint["mask"]["path"])
        lama_path = repo_path(endpoint["lama_baseline"]["path"])
        mat_path = args.mat_root.resolve() / side / "inpaint_composited.png"
        candidate_path = args.candidate_root.resolve() / side / "inpaint_composited.png"
        paths = {
            "source": source_path,
            "mask": mask_path,
            "lama": lama_path,
            "mat": mat_path,
            "candidate": candidate_path,
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{side} {label} 不存在：{path}")

        source = np.asarray(Image.open(source_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        lama = np.asarray(Image.open(lama_path).convert("RGB"))
        mat = np.asarray(Image.open(mat_path).convert("RGB"))
        candidate = np.asarray(Image.open(candidate_path).convert("RGB"))
        if source.shape != lama.shape or source.shape != mat.shape or source.shape != candidate.shape:
            raise ValueError(f"{side} A/B/C/D 分辨率不一致。")
        if source.shape[:2] != mask.shape:
            raise ValueError(f"{side} RGB 与掩码分辨率不一致。")
        known_exact = bool(np.array_equal(candidate[~mask], source[~mask]))
        if not known_exact:
            raise AssertionError(f"{side} D 改写了 accepted mask 外像素。")

        rows.append(
            [
                tile(source, f"{side} A: uncompleted endpoint", args.tile_width),
                tile(lama, f"{side} B: Big-LaMa", args.tile_width),
                tile(mat, f"{side} C: MAT Places-512", args.tile_width),
                tile(candidate, f"{side} D: 3D Photo adapter v1", args.tile_width),
            ]
        )
        checks[side] = {
            "known_pixels_exact": known_exact,
            "hole_fraction": float(mask.mean()),
            "source_sha256": file_hash(source_path),
            "lama_sha256": file_hash(lama_path),
            "mat_sha256": file_hash(mat_path),
            "candidate_sha256": file_hash(candidate_path),
        }

    tile_height = max(item.height for row in rows for item in row)
    sheet = Image.new("RGB", (args.tile_width * 4, tile_height * 2), "white")
    for row_index, row in enumerate(rows):
        for column_index, item in enumerate(row):
            sheet.paste(item, (column_index * args.tile_width, row_index * tile_height))
    review_path = output_dir / "P01_ang30_ABCD_3dphoto_adapter_v1.png"
    sheet.save(review_path)

    write_json(
        output_dir / "abcd_review.json",
        {
            "schema_version": "1.0-stage1-abcd-review",
            "experiment_id": manifest["experiment_id"],
            "layout": "rows=left/right; columns=A uncompleted, B Big-LaMa, C MAT Places-512, D 3D Photo adapter v1",
            "checks": checks,
            "review_image": str(review_path),
            "review_image_sha256": file_hash(review_path),
            "assistant_precheck": {
                "status": "visual_fail_expected_pending_user",
                "observation": "D has large dark green/gray smeared regions at both outer sides; structure-aware network execution did not yield credible content for the frozen alpha-derived holes.",
                "decision": "Do not build supplement Gaussians or rerun videos before user review."
            },
            "manual_review": {
                "left_smear_severity_0_to_3": None,
                "right_smear_severity_0_to_3": None,
                "false_structure_0_or_1": None,
                "d_vs_b_and_c": None,
                "static_gate_pass": None,
                "notes": None
            }
        },
    )


if __name__ == "__main__":
    main()
