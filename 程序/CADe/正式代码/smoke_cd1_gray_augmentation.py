#!/usr/bin/env python3
"""用锁定Ultralytics训练管线验证CD1 Gray增强后三通道仍相等。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAINING_CODE))

from train_y2_yolo26 import FROZEN_TRAIN_ARGS  # noqa: E402

from prepare_cd1_gray_data import DEFAULT_OUTPUT, file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析Gray数据目录、抽样数量和固定随机种子。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gray-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """构建真实训练dataset、抽样增强tensor并保存通道审计。"""
    args = parse_args()
    root = args.gray_root.resolve()
    output = root / "augmentation_smoke.json"
    if output.exists():
        raise FileExistsError(f"增强smoke记录已存在，拒绝覆盖: {output}")
    config_path = root / "gray_data_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["gray_manifest"])
    if file_sha256(manifest_path) != config["gray_manifest_sha256"]:
        raise ValueError("Gray manifest SHA与数据配置不一致")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    gray_train_args = {
        **FROZEN_TRAIN_ARGS,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.4,
        "imgsz": 640,
        "task": "detect",
        "mode": "train",
        "fraction": 1.0,
        "single_cls": False,
        "classes": None,
    }
    dataset = build_yolo_dataset(
        get_cfg(overrides=gray_train_args),
        str(root / "images/train"),
        batch=16,
        data=data,
        mode="train",
        rect=False,
        stride=32,
    )
    if args.samples > len(dataset):
        raise ValueError(f"抽样数{args.samples}超过train规模{len(dataset)}")
    maximum_difference = 0
    for index in range(args.samples):
        image = dataset[index]["img"].to(torch.int16)
        maximum_difference = max(
            maximum_difference,
            int((image[0] - image[1]).abs().max()),
            int((image[1] - image[2]).abs().max()),
        )
    if maximum_difference != 0:
        raise RuntimeError(f"Gray真实增强产生通道差异: {maximum_difference}")

    audit = {
        "stage": "CD1",
        "role": "gray_ultralytics_augmentation_smoke",
        "gray_data_config": str(config_path),
        "gray_data_config_sha256": file_sha256(config_path),
        "gray_manifest_sha256": config["gray_manifest_sha256"],
        "seed": args.seed,
        "samples": args.samples,
        "dataset_size": len(dataset),
        "tensor_shape": list(dataset[0]["img"].shape),
        "max_channel_difference": maximum_difference,
        "training_args": gray_train_args,
        "passed": True,
    }
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CD1 Gray增强smoke通过: {args.samples}张, 最大通道差={maximum_difference}")
    print(f"审计记录: {output}")


if __name__ == "__main__":
    main()
