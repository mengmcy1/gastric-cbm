#!/usr/bin/env python3
"""Fine-tune both frozen Y2 resolutions under one no-Mosaic protocol.

This supplementary convergence experiment starts from each formal Y2-B
``best.pt`` and gives 640 and 960 the same 20 low-learning-rate epochs. It uses
only the frozen balanced train/val view and writes to an isolated output root;
it cannot replace or overwrite the primary Y2-B products.
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
from train_y2_yolo26 import DEFAULT_OUTPUT_ROOT as Y2_ROOT
from train_y2_yolo26 import FROZEN_TRAIN_ARGS, expected_run_name as y2_run_name


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y2补充敏感性_无Mosaic微调"
FORMAL_EPOCHS = 20
FORMAL_PATIENCE = 20

# Only the convergence-related settings differ from Y2-B. Keeping the other
# augmentations unchanged isolates extra low-LR, no-Mosaic optimization.
SENSITIVITY_TRAIN_ARGS = {
    **FROZEN_TRAIN_ARGS,
    "lr0": 0.0001,
    "lrf": 0.1,
    "warmup_epochs": 1.0,
    "mosaic": 0.0,
    "close_mosaic": 0,
}


def parse_args() -> argparse.Namespace:
    """Parse one resolution, CUDA device and isolated debug role."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, choices=(640, 960), required=True)
    parser.add_argument("--device", required=True, help="CUDA index selected after nvidia-smi.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def expected_run_name(imgsz: int, seed: int, debug: bool) -> str:
    """Return an output name distinct from every primary Y2 product."""
    suffix = "_debug" if debug else ""
    return f"y2s_yolo26s_{imgsz}_seed{seed}_nomosaic{suffix}"


def load_parent(imgsz: int, seed: int) -> tuple[Path, dict]:
    """Load and validate the immutable formal Y2 parent checkpoint."""
    parent_dir = Y2_ROOT / y2_run_name(imgsz, seed, False)
    config_path = parent_dir / "y2_train_config.json"
    checkpoint = parent_dir / "weights/best.pt"
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Y2 parent is incomplete: {parent_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["training_protocol"]
    if config["debug"] or int(protocol["imgsz"]) != imgsz or int(protocol["seed"]) != seed:
        raise ValueError(f"Y2 parent role mismatch: {parent_dir}")
    if config["internal_test_read"] or config["external_read"]:
        raise ValueError("Y2 parent crossed the locked evaluation boundary")
    if file_sha256(checkpoint) != config["products"]["best_pt_sha256"]:
        raise ValueError(f"Y2 parent checkpoint SHA mismatch: {checkpoint}")
    return checkpoint, config


def verify_args_yaml(path: Path, parent: Path, imgsz: int, seed: int, debug: bool) -> dict:
    """Verify persisted Ultralytics arguments against the sensitivity protocol."""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        **SENSITIVITY_TRAIN_ARGS,
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
    persisted_model = Path(str(values.get("model", ""))).resolve()
    if persisted_model != parent.resolve():
        mismatches["model"] = {"expected": str(parent.resolve()), "actual": str(persisted_model)}
    if mismatches:
        raise RuntimeError(f"Y2-S persisted argument mismatch: {mismatches}")
    return values


def verify_products(run_dir: Path, debug: bool) -> dict:
    """Require complete checkpoints and the exact requested training length."""
    required = [
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Y2-S products are incomplete: {missing}")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    expected_epochs = 1 if debug else FORMAL_EPOCHS
    if len(history) != expected_epochs:
        raise ValueError(f"Y2-S must complete {expected_epochs} epochs, got {len(history)}")
    return {
        "epochs_completed": int(len(history)),
        "best_pt_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_pt_sha256": file_sha256(run_dir / "weights/last.pt"),
        "best_map50_95": float(history["metrics/mAP50-95(B)"].max()),
    }


def main() -> None:
    """Run one isolated Y2-S candidate and save its provenance record."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start Y2-S")
    if args.seed != 42:
        raise ValueError("Y2-S convergence sensitivity is frozen to seed 42")
    data_audit = assert_y0_dataset()
    parent, parent_config = load_parent(args.imgsz, args.seed)

    output_root = args.output_root.resolve()
    run_name = expected_run_name(args.imgsz, args.seed, args.debug)
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Y2-S output already exists: {run_dir}")

    model = YOLO(str(parent))
    environment = assert_locked_library_behavior(model)
    model.train(
        data=str(Y0_ROOT / "data.yaml"),
        model=str(parent),
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
        **SENSITIVITY_TRAIN_ARGS,
    )

    persisted_args = verify_args_yaml(run_dir / "args.yaml", parent, args.imgsz, args.seed, args.debug)
    products = verify_products(run_dir, args.debug)
    best_model = YOLO(str(run_dir / "weights/best.pt"))
    config = {
        "stage": "Y2-S",
        "role": "supplementary_no_mosaic_convergence_sensitivity",
        "debug": args.debug,
        "primary_y2_replacement_allowed": False,
        "internal_test_read": False,
        "external_read": False,
        "locked_ultralytics": LOCKED_ULTRALYTICS_VERSION,
        "environment": environment,
        "best_checkpoint_behavior": assert_locked_library_behavior(best_model),
        "data": data_audit,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": file_sha256(parent),
        "parent_training_config": str(Y2_ROOT / y2_run_name(args.imgsz, args.seed, False) / "y2_train_config.json"),
        "parent_best_map50_95": parent_config["products"]["best_map50_95"],
        "training_protocol": {
            "model": "yolo26s.pt",
            "imgsz": args.imgsz,
            "seed": args.seed,
            "epochs": 1 if args.debug else FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE,
            **SENSITIVITY_TRAIN_ARGS,
        },
        "persisted_args": persisted_args,
        "products": products,
        "geometry_evaluated": False,
    }
    (run_dir / "y2s_train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y2-S training complete: {run_dir}")
    print("Run the frozen Top-1 geometry evaluation before interpreting convergence.")


if __name__ == "__main__":
    main()
