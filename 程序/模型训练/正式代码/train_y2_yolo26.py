#!/usr/bin/env python3
"""Train one pre-registered Y2-B YOLO26s resolution candidate.

The formal path always uses the frozen balanced Y0 train/val view and explicit
optimizer/augmentation settings. ``--debug`` runs one epoch on the complete
train split in a separately named directory so both positive boxes and empty
negative labels are exercised; it can never be mistaken for a formal result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from run_y1_yolo26_smoke import (
    LOCKED_ULTRALYTICS_VERSION,
    PROJECT_ROOT,
    Y0_ROOT,
    assert_locked_library_behavior,
    assert_y0_dataset,
    file_sha256,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y2平衡分辨率预筛"
PRETRAINED_PATH = PROJECT_ROOT / "yolo26s.pt"
FORMAL_EPOCHS = 100
FORMAL_PATIENCE = 20

# Values are explicit so Balanced and Full cannot silently choose different
# optimizers when their iteration counts straddle Ultralytics' auto threshold.
FROZEN_TRAIN_ARGS = {
    "batch": 16,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 10,
    "rect": False,
    "cos_lr": False,
    "multi_scale": 0.0,
    "amp": True,
    "cache": False,
    "pretrained": True,
    "val": True,
    "plots": True,
    "deterministic": True,
}


def parse_args() -> argparse.Namespace:
    """Parse the resolution, selected CUDA device and isolated debug switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, choices=(640, 960), required=True)
    parser.add_argument("--device", required=True, help="CUDA index selected after nvidia-smi.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def expected_run_name(imgsz: int, seed: int, debug: bool) -> str:
    """Return a name that makes debug and formal products unambiguous."""
    suffix = "_debug" if debug else ""
    return f"y2b_yolo26s_{imgsz}_seed{seed}{suffix}"


def verify_args_yaml(path: Path, imgsz: int, seed: int, debug: bool) -> dict:
    """Check the persisted Ultralytics arguments against the frozen protocol."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        **FROZEN_TRAIN_ARGS,
        "imgsz": imgsz,
        "seed": seed,
        "epochs": 1 if debug else FORMAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "fraction": 1.0,
        "nms": False,
        "max_det": 300,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": values.get(key)}
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Y2训练参数与冻结协议不一致: {mismatches}")
    return values


def verify_training_products(run_dir: Path, debug: bool) -> dict:
    """Require the checkpoints, history and exact persisted training arguments."""
    required = [
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Y2训练产物不完整: {missing}")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if debug and len(history) != 1:
        raise ValueError(f"Y2 debug应完成1轮，实际{len(history)}轮")
    if not debug and not 1 <= len(history) <= FORMAL_EPOCHS:
        raise ValueError(f"Y2正式训练轮数异常: {len(history)}")
    return {
        "epochs_completed": int(len(history)),
        "best_pt_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_pt_sha256": file_sha256(run_dir / "weights/last.pt"),
        "best_map50_95": float(history["metrics/mAP50-95(B)"].max()),
    }


def main() -> None:
    """Run one isolated Y2-B candidate and write its acceptance record."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝启动Y2")
    if args.seed != 42:
        raise ValueError("Y2-B预筛只允许seed42；其他seed属于Y3-B")
    y0_audit = assert_y0_dataset()
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(f"缺少冻结预训练权重: {PRETRAINED_PATH}")

    output_root = args.output_root.resolve()
    run_name = expected_run_name(args.imgsz, args.seed, args.debug)
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Y2输出已存在，拒绝覆盖: {run_dir}")

    model = YOLO(str(PRETRAINED_PATH))
    environment = assert_locked_library_behavior(model)
    model.train(
        data=str(Y0_ROOT / "data.yaml"),
        model=str(PRETRAINED_PATH),
        epochs=1 if args.debug else FORMAL_EPOCHS,
        patience=FORMAL_PATIENCE,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        fraction=1.0,
        project=str(output_root),
        name=run_name,
        exist_ok=False,
        verbose=True,
        **FROZEN_TRAIN_ARGS,
    )

    persisted_args = verify_args_yaml(run_dir / "args.yaml", args.imgsz, args.seed, args.debug)
    products = verify_training_products(run_dir, args.debug)
    best_model = YOLO(str(run_dir / "weights/best.pt"))
    best_behavior = assert_locked_library_behavior(best_model)
    config = {
        "stage": "Y2-B",
        "role": "balanced_primary_resolution_screen",
        "debug": args.debug,
        "accepted_engineering_run": True,
        "performance_selection_allowed": not args.debug,
        "internal_test_read": False,
        "external_read": False,
        "locked_ultralytics": LOCKED_ULTRALYTICS_VERSION,
        "environment": environment,
        "best_checkpoint_behavior": best_behavior,
        "data": y0_audit,
        "pretrained_checkpoint": str(PRETRAINED_PATH),
        "pretrained_sha256": file_sha256(PRETRAINED_PATH),
        "training_protocol": {
            "model": "yolo26s.pt",
            "imgsz": args.imgsz,
            "seed": args.seed,
            "epochs": 1 if args.debug else FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE,
            **FROZEN_TRAIN_ARGS,
        },
        "persisted_args": persisted_args,
        "products": products,
        "geometry_evaluated": False,
    }
    (run_dir / "y2_train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y2-B训练完成: {run_dir}")
    print("下一步必须运行冻结Top-1几何评估；不得只按mAP选择分辨率。")


if __name__ == "__main__":
    main()
