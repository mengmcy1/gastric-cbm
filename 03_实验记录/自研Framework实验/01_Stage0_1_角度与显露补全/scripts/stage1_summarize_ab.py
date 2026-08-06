from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from stage01_common import file_hash, prepare_output_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总同协议 A/B 端点 Alpha 与 RGB 差异。")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--completed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-mask", type=Path, help="兼容旧入口：等同于左端点掩码")
    parser.add_argument("--evaluation-mask-left", type=Path)
    parser.add_argument("--evaluation-mask-right", type=Path)
    parser.add_argument("--alpha-threshold", type=float, default=0.95)
    return parser.parse_args()


def alpha(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array.astype(np.float32) / float(np.iinfo(array.dtype).max)


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir)
    config_a = read_json(args.baseline_dir / "config.json")
    config_b = read_json(args.completed_dir / "config.json")
    protocol_keys = ("angle_total_deg", "num_steps", "trajectory_mode", "render_resolution_wh")
    mismatches = {
        key: [config_a.get(key), config_b.get(key)]
        for key in protocol_keys
        if config_a.get(key) != config_b.get(key)
    }
    if mismatches:
        raise ValueError(f"A/B 协议不一致：{mismatches}")

    mask_paths = {
        "left": args.evaluation_mask_left or args.evaluation_mask,
        "right": args.evaluation_mask_right,
    }
    evaluation_masks = {
        side: (np.asarray(Image.open(path).convert("L")) > 0) if path else None
        for side, path in mask_paths.items()
    }
    rows = []
    for side in ("left", "center", "right"):
        alpha_a = alpha(args.baseline_dir / f"alpha_{side}_u16.png")
        alpha_b = alpha(args.completed_dir / f"alpha_{side}_u16.png")
        if alpha_a.shape != alpha_b.shape:
            raise ValueError(f"{side} A/B Alpha 分辨率不一致。")
        region = np.ones_like(alpha_a, dtype=bool)
        evaluation_mask = evaluation_masks.get(side)
        if evaluation_mask is not None:
            if evaluation_mask.shape != alpha_a.shape:
                raise ValueError("evaluation_mask 与端点分辨率不一致。")
            region = evaluation_mask
        holes_a = alpha_a < args.alpha_threshold
        holes_b = alpha_b < args.alpha_threshold
        denominator = max(int(region.sum()), 1)
        rate_a = float((holes_a & region).sum() / denominator)
        rate_b = float((holes_b & region).sum() / denominator)
        rows.append(
            {
                "side": side,
                "region": "evaluation_mask" if evaluation_mask is not None else "full_frame",
                "region_pixels": int(region.sum()),
                "alpha_threshold": args.alpha_threshold,
                "hole_rate_a": rate_a,
                "hole_rate_b": rate_b,
                "relative_hole_reduction": ((rate_a - rate_b) / rate_a) if rate_a > 0 else None,
                "alpha_mean_a": float(alpha_a[region].mean()),
                "alpha_mean_b": float(alpha_b[region].mean()),
            }
        )

    with (output_dir / "ab_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "ab_summary.json",
        {
            "schema_version": "1.0-stage01-ab",
            "baseline_dir": str(args.baseline_dir.resolve()),
            "completed_dir": str(args.completed_dir.resolve()),
            "protocol": {key: config_a.get(key) for key in protocol_keys},
            "evaluation_masks": {
                side: {
                    "path": str(path.resolve()),
                    "sha256": file_hash(path),
                }
                if path
                else None
                for side, path in mask_paths.items()
            },
            "rows": rows,
            "decision_boundary": (
                "自动 Alpha 仅说明几何覆盖代理变化；不能证明补全颜色、深度、遮挡关系或时序质量正确。"
            ),
        },
    )


if __name__ == "__main__":
    main()
