#!/usr/bin/env python3
"""冻结EfficientNet局部SAE使用的Y3-F预测ROI清单。

脚本只读取Y0F train/val。train由同seed冻结Y3-F重新生成Top-1预测，val直接复用Y6训练时
已经冻结的预测文件，避免浮点边界漂移。输出保留全部图片及``has_roi``，正式局部SAE只
使用``has_roi=True``的行，因为这些ROI才会在部署数据流中进入Y6局部分类器。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
TRAIN_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAIN_CODE))

from efficientnet_y6_full_roi_classifier import (  # noqa: E402
    DEFAULT_MAPPING,
    DEFAULT_OUTPUT as Y6_ROOT,
    MARGIN,
    load_mapping,
    product_paths,
    square_box,
)
from evaluate_y2_yolo26 import predict_val  # noqa: E402
from run_y1_yolo26_smoke import assert_locked_library_behavior, file_sha256  # noqa: E402
from train_utils import git_snapshot, json_ready  # noqa: E402


SEEDS = (42, 202, 503)
DEFAULT_OUTPUT = PROJECT_ROOT / "数据整理记录/SAE_EfficientNet_0804/冻结局部ROI清单"


def parse_args() -> argparse.Namespace:
    """解析单seed清单构建参数；GPU编号由外层CUDA_VISIBLE_DEVICES隔离。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def attach_roi_coordinates(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """按冻结阈值和Y6扩边协议生成确定性正方形ROI坐标。"""
    result = frame.copy()
    result["has_roi"] = result["top1_confidence"].ge(threshold)
    boxes = []
    for row in result.itertuples(index=False):
        if bool(row.has_roi):
            box = square_box(
                np.array([row.top1_x1, row.top1_y1, row.top1_x2, row.top1_y2]),
                int(row.width), int(row.height), MARGIN,
            )
        else:
            box = np.array([0.0, 0.0, 1.0, 1.0])
        boxes.append(box)
    result[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]] = np.vstack(boxes)
    return result


def validate_reused_val(generated: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """确认当前血缘与Y6冻结val的路径、置信度、触发状态和ROI完全一致。"""
    columns = [
        "image_relpath", "top1_confidence", "has_roi",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
    ]
    left = generated[columns].sort_values("image_relpath").reset_index(drop=True)
    right = frozen[columns].sort_values("image_relpath").reset_index(drop=True)
    if not left.image_relpath.equals(right.image_relpath):
        raise RuntimeError("重新生成的val路径与Y6冻结val不一致")
    if not np.array_equal(left.has_roi.to_numpy(), right.has_roi.to_numpy()):
        raise RuntimeError("重新生成的val门控状态与Y6冻结val不一致")
    numeric = ["top1_confidence", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
    if not np.allclose(left[numeric], right[numeric], rtol=0.0, atol=1e-12):
        raise RuntimeError("重新生成的val数值与Y6冻结val不一致")


def run_self_test() -> None:
    """验证触发阈值使用大于等于，以及未触发图严格回退全图坐标。"""
    frame = pd.DataFrame({
        "top1_confidence": [0.2, 0.19], "width": [100, 100], "height": [100, 100],
        "top1_x1": [0.2, np.nan], "top1_y1": [0.2, np.nan],
        "top1_x2": [0.4, np.nan], "top1_y2": [0.4, np.nan],
    })
    result = attach_roi_coordinates(frame, 0.2)
    assert result.has_roi.tolist() == [True, False]
    assert np.allclose(result.loc[1, ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]], [0, 0, 1, 1])
    print("local SAE manifest self-test passed")


def main() -> None:
    """生成train预测ROI、绑定Y6冻结val并保存带完整血缘的单seed清单。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.device != "0":
        raise ValueError("请用CUDA_VISIBLE_DEVICES选择物理GPU，脚本内--device保持0")

    products = product_paths(args.seed)
    mapping = load_mapping(args.mapping, False, 0, args.seed)
    train = mapping.loc[mapping.split.eq("train")].reset_index(drop=True)
    yolo = YOLO(str(products["y3_checkpoint"]))
    assert_locked_library_behavior(yolo)
    source = args.mapping.parent / Path(train.yolo_image_relpath.iloc[0]).parent
    predicted_train = predict_val(
        yolo, train, int(products["imgsz"]), args.device, source_dir=source,
    )
    predicted_train = attach_roi_coordinates(predicted_train, float(products["threshold"]))

    y6_run = Y6_ROOT / f"y6_full_efficientnet_b0_seed{args.seed}"
    y6_config_path = y6_run / "config.json"
    y6_config = json.loads(y6_config_path.read_text(encoding="utf-8"))
    if int(y6_config["seed"]) != args.seed:
        raise ValueError("Y6配置seed不一致")
    frozen_val_path = y6_run / "frozen_val_predicted_roi.csv"
    frozen_val = pd.read_csv(
        frozen_val_path, encoding="utf-8-sig", dtype={"patient_id": str},
        float_precision="round_trip",
    )
    # 重新计算只用于血缘核验；正式输出沿用Y6落盘值。
    raw_val = mapping.loc[mapping.split.eq("val")].reset_index(drop=True)
    generated_val = attach_roi_coordinates(
        frozen_val.drop(columns=["has_roi", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]),
        float(products["threshold"]),
    )
    if set(raw_val.image_relpath) != set(generated_val.image_relpath):
        raise RuntimeError("Y6冻结val不属于当前Y0F mapping")
    validate_reused_val(generated_val, frozen_val)

    output_csv = args.output_root / f"efficientnet_local_roi_seed{args.seed}.csv"
    output_config = args.output_root / f"efficientnet_local_roi_seed{args.seed}.json"
    if output_csv.exists() or output_config.exists():
        raise FileExistsError(f"局部SAE清单已存在，拒绝覆盖: seed{args.seed}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([predicted_train, frozen_val], ignore_index=True, sort=False)
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "stage": "efficientnet_sae_local_roi_manifest",
        "seed": args.seed,
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": file_sha256(args.mapping),
        "y3_checkpoint": str(Path(products["y3_checkpoint"]).resolve()),
        "y3_checkpoint_sha256": file_sha256(Path(products["y3_checkpoint"])),
        "y3_deployment_threshold": float(products["threshold"]),
        "y3_imgsz": int(products["imgsz"]),
        "y6_config": str(y6_config_path.resolve()),
        "y6_config_sha256": file_sha256(y6_config_path),
        "frozen_val_manifest": str(frozen_val_path.resolve()),
        "frozen_val_manifest_sha256": file_sha256(frozen_val_path),
        "roi_margin": MARGIN,
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": file_sha256(output_csv),
        "counts": {
            split: {
                "images": int(len(part)),
                "patients": int(part.patient_id.nunique()),
                "triggered_images": int(part.has_roi.sum()),
                "triggered_patients": int(part.loc[part.has_roi, "patient_id"].nunique()),
                "trigger_rate_cancer": float(part.loc[part.label.eq(1), "has_roi"].mean()),
                "trigger_rate_noncancer": float(part.loc[part.label.eq(0), "has_roi"].mean()),
            }
            for split, part in combined.groupby("split", sort=False)
        },
        "test_evaluated": False,
        "external_evaluated": False,
        **git_snapshot(),
    }
    output_config.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"局部SAE冻结清单: {output_csv}")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
