#!/usr/bin/env python3
"""Train one fixed-resolution Y3-B or Y3-F YOLO26s replicate.

Balanced runs use the frozen Y0-B train/val view; full diagnostic runs use the
Y0-F train/val view. Both start independently from the same official
``yolo26s.pt`` and reuse every Y2-B training parameter at 640 pixels. Seed 42
for Balanced is the existing Y2-B run and is deliberately not retrained here.
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
from train_y2_yolo26 import (
    DEFAULT_OUTPUT_ROOT as Y2_ROOT,
    FORMAL_EPOCHS,
    FORMAL_PATIENCE,
    FROZEN_TRAIN_ARGS,
    PRETRAINED_PATH,
)


Y0_FULL_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "11_Y0F_YOLO26完整诊断数据_20260813"
)
Y2_SELECTION = Y2_ROOT / "自动汇总/y2_selection.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"
ALLOWED_SEEDS = (42, 202, 503)
IMAGE_SIZE = 640


def parse_args() -> argparse.Namespace:
    """Parse role, seed, inspected CUDA device and isolated debug flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("balanced", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def expected_run_name(role: str, seed: int, debug: bool) -> str:
    """Return a role-explicit run name that cannot collide with Y2-B."""
    suffix = "_debug" if debug else ""
    return f"y3_{role}_yolo26s_640_seed{seed}{suffix}"


def load_data_audit(role: str) -> tuple[Path, dict]:
    """Return the role-specific data root after checking its frozen audit.

    Args:
        role: ``balanced`` for Y3-B or ``full`` for diagnostic Y3-F.

    Returns:
        The YOLO dataset root and its immutable audit configuration.
    """
    if role == "balanced":
        return Y0_ROOT, assert_y0_dataset()
    config_path = Y0_FULL_ROOT / "y0f_config.json"
    data_path = Y0_FULL_ROOT / "data.yaml"
    if not config_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Y0-F is not complete: {Y0_FULL_ROOT}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["stage"] != "Y0-F" or config["role"] != "full_data_diagnostic_only":
        raise ValueError("Y0-F config role mismatch")
    if config["exact_duplicate_rows"] != 0:
        raise ValueError("Y0-F exact-duplicate audit failed")
    if config["full_original_test_exported_to_yolo"] or config["full_original_test_evaluated"]:
        raise ValueError("Y0-F crossed the full-data original-test boundary")
    secondary_path = Y0_FULL_ROOT / "audit/train_val_near_duplicate_secondary_audit.json"
    if not secondary_path.is_file():
        raise FileNotFoundError("Y0-F train/val secondary duplicate audit is missing")
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    if secondary["pairs_at_or_above_review_threshold"] != 0:
        raise ValueError("Y0-F secondary duplicate audit still has review candidates")
    config["train_val_secondary_duplicate_audit"] = secondary
    return Y0_FULL_ROOT, config


def load_y2_selection() -> dict:
    """Require the immutable Y2 decision that selected 640 pixels."""
    if not Y2_SELECTION.is_file():
        raise FileNotFoundError(f"Missing Y2 selection: {Y2_SELECTION}")
    selection = json.loads(Y2_SELECTION.read_text(encoding="utf-8"))
    if selection["selected_imgsz"] != IMAGE_SIZE or selection["decision"] != "selected_by_preregistered_hierarchy":
        raise ValueError("Y2 selection does not freeze 640 pixels")
    if selection["internal_test_read"] or selection["external_read"]:
        raise ValueError("Y2 selection crossed a locked evaluation boundary")
    return selection


def verify_args_yaml(path: Path, role: str, seed: int, debug: bool, data_root: Path) -> dict:
    """Verify that Ultralytics persisted the complete fixed Y2-B protocol.

    Args:
        path: Generated ``args.yaml`` path.
        role: Balanced or full diagnostic dataset role.
        seed: Independent replicate seed.
        debug: Whether exactly one debug epoch was requested.
        data_root: Expected role-specific YOLO dataset root.

    Returns:
        The parsed Ultralytics settings after all equality checks pass.
    """
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        **FROZEN_TRAIN_ARGS,
        "imgsz": IMAGE_SIZE,
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
    if Path(str(values.get("data", ""))).resolve() != (data_root / "data.yaml").resolve():
        mismatches["data"] = {"expected_role": role, "actual": values.get("data")}
    if mismatches:
        raise RuntimeError(f"Y3 persisted argument mismatch: {mismatches}")
    return values


def verify_products(run_dir: Path, debug: bool) -> dict:
    """Validate one run's checkpoints/history and return product metadata."""
    required = [
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Y3 products are incomplete: {missing}")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if debug and len(history) != 1:
        raise ValueError(f"Y3 debug must complete one epoch, got {len(history)}")
    if not debug and not 1 <= len(history) <= FORMAL_EPOCHS:
        raise ValueError(f"Y3 formal epoch count is invalid: {len(history)}")
    return {
        "epochs_completed": int(len(history)),
        "best_pt_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_pt_sha256": file_sha256(run_dir / "weights/last.pt"),
        "best_map50_95": float(history["metrics/mAP50-95(B)"].max()),
    }


def main() -> None:
    """Train one independent replicate without reading test or external data."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start Y3")
    if args.role == "balanced" and args.seed == 42 and not args.debug:
        raise ValueError("Balanced seed42 must reuse the frozen Y2-B 640 product")
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(f"Missing official pretrained checkpoint: {PRETRAINED_PATH}")
    data_root, data_audit = load_data_audit(args.role)
    selection = load_y2_selection()

    output_root = args.output_root.resolve()
    run_name = expected_run_name(args.role, args.seed, args.debug)
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Y3 output already exists: {run_dir}")

    model = YOLO(str(PRETRAINED_PATH))
    environment = assert_locked_library_behavior(model)
    model.train(
        data=str(data_root / "data.yaml"),
        model=str(PRETRAINED_PATH),
        epochs=1 if args.debug else FORMAL_EPOCHS,
        patience=FORMAL_PATIENCE,
        imgsz=IMAGE_SIZE,
        device=args.device,
        seed=args.seed,
        fraction=1.0,
        project=str(output_root),
        name=run_name,
        exist_ok=False,
        verbose=True,
        **FROZEN_TRAIN_ARGS,
    )

    persisted_args = verify_args_yaml(
        run_dir / "args.yaml", args.role, args.seed, args.debug, data_root
    )
    products = verify_products(run_dir, args.debug)
    best_model = YOLO(str(run_dir / "weights/best.pt"))
    config = {
        "stage": "Y3-B" if args.role == "balanced" else "Y3-F",
        "role": args.role,
        "seed": args.seed,
        "debug": args.debug,
        "internal_test_read": False,
        "external_read": False,
        "full_original_test_read": False,
        "locked_ultralytics": LOCKED_ULTRALYTICS_VERSION,
        "environment": environment,
        "best_checkpoint_behavior": assert_locked_library_behavior(best_model),
        "data_root": str(data_root),
        "data_audit": data_audit,
        "y2_selection": str(Y2_SELECTION),
        "y2_selection_sha256": file_sha256(Y2_SELECTION),
        "pretrained_checkpoint": str(PRETRAINED_PATH),
        "pretrained_sha256": file_sha256(PRETRAINED_PATH),
        "training_protocol": {
            "model": "yolo26s.pt",
            "imgsz": IMAGE_SIZE,
            "seed": args.seed,
            "epochs": 1 if args.debug else FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE,
            **FROZEN_TRAIN_ARGS,
        },
        "persisted_args": persisted_args,
        "products": products,
        "geometry_evaluated": False,
    }
    (run_dir / "y3_train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y3 training complete: {run_dir}")
    print("Run same-seed M1-paired Top-1 geometry evaluation next.")


if __name__ == "__main__":
    main()
