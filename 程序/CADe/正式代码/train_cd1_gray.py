#!/usr/bin/env python3
"""训练一个CD1 Gray YOLO26s-640 seed并保存可追溯产品血缘。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_CODE = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(TRAINING_CODE))

from run_y1_yolo26_smoke import (  # noqa: E402
    LOCKED_ULTRALYTICS_VERSION,
    assert_locked_library_behavior,
    file_sha256,
)
from train_y2_yolo26 import (  # noqa: E402
    FORMAL_EPOCHS,
    FORMAL_PATIENCE,
    FROZEN_TRAIN_ARGS,
    PRETRAINED_PATH,
)
from train_utils import git_snapshot  # noqa: E402


GRAY_DATA_ROOT = PROJECT_ROOT / "数据整理记录/CADe_CD1_Gray_20260825"
Y3F_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y3固定640三种子"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "结果/CADe/CD1_Gray_Qualification_20260825/训练"
ALLOWED_SEEDS = (42, 202, 503)
IMAGE_SIZE = 640
CD1_GRAY_TRAIN_ARGS = {
    **FROZEN_TRAIN_ARGS,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.4,
}


def parse_args() -> argparse.Namespace:
    """解析单seed训练、设备、debug和中断恢复选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def expected_run_name(seed: int, debug: bool) -> str:
    """返回不会与正式RGB或Gray产品混淆的run名。"""
    suffix = "_debug" if debug else ""
    return f"cd1_gray_yolo26s_640_seed{seed}{suffix}"


def load_gray_data_audit() -> dict:
    """验证Gray manifest、数据配置和真实增强smoke的绑定关系。"""
    config_path = GRAY_DATA_ROOT / "gray_data_config.json"
    smoke_path = GRAY_DATA_ROOT / "augmentation_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["gray_manifest"])
    if config["stage"] != "CD1" or config["role"] != "gray_train_val_view":
        raise ValueError("Gray数据配置角色异常")
    if config["images"] != {"train": 2350, "val": 497}:
        raise ValueError("Gray图像规模异常")
    if config["patients"] != {"train": 1212, "val": 260}:
        raise ValueError("Gray患者规模异常")
    if config["test_exported"] or config["external_exported"]:
        raise ValueError("Gray数据越过CD1 train/val边界")
    if file_sha256(manifest_path) != config["gray_manifest_sha256"]:
        raise ValueError("Gray manifest SHA不一致")
    if not smoke["passed"] or smoke["max_channel_difference"] != 0:
        raise ValueError("Gray真实增强smoke未通过")
    if smoke["gray_data_config_sha256"] != file_sha256(config_path):
        raise ValueError("Gray增强smoke未绑定当前数据配置")
    if smoke["gray_manifest_sha256"] != config["gray_manifest_sha256"]:
        raise ValueError("Gray增强smoke未绑定当前manifest")
    expected_smoke_args = {
        **CD1_GRAY_TRAIN_ARGS,
        "imgsz": IMAGE_SIZE,
        "task": "detect",
        "mode": "train",
        "fraction": 1.0,
        "single_cls": False,
        "classes": None,
    }
    if smoke["training_args"] != expected_smoke_args:
        raise ValueError("Gray增强smoke使用的训练参数与CD1冻结配置不一致")
    data_yaml = yaml.safe_load((GRAY_DATA_ROOT / "data.yaml").read_text(encoding="utf-8"))
    expected_yaml = {
        "path": str(GRAY_DATA_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "early_cancer_or_HGD"},
    }
    if data_yaml != expected_yaml:
        raise ValueError("CD1 Gray data.yaml不是冻结train/val视图")
    return {
        "data_root": str(GRAY_DATA_ROOT),
        "data_yaml": str(GRAY_DATA_ROOT / "data.yaml"),
        "data_config": str(config_path),
        "data_config_sha256": file_sha256(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": config["gray_manifest_sha256"],
        "augmentation_smoke": str(smoke_path),
        "augmentation_smoke_sha256": file_sha256(smoke_path),
        "images": config["images"],
        "patients": config["patients"],
    }


def load_rgb_reference(seed: int) -> dict:
    """读取同seed冻结Y3-F配置并确认除颜色外的训练参照未漂移。"""
    run_dir = Y3F_ROOT / f"y3_full_yolo26s_640_seed{seed}"
    config_path = run_dir / "y3_train_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["stage"] != "Y3-F" or config["role"] != "full" or config["seed"] != seed:
        raise ValueError(f"Y3-F RGB参考角色异常: seed{seed}")
    expected = {
        "model": "yolo26s.pt",
        "imgsz": IMAGE_SIZE,
        "seed": seed,
        "epochs": FORMAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        **FROZEN_TRAIN_ARGS,
    }
    if config["training_protocol"] != expected:
        raise ValueError(f"Y3-F RGB参考训练参数已漂移: seed{seed}")
    checkpoint = run_dir / "weights/best.pt"
    if file_sha256(checkpoint) != config["products"]["best_pt_sha256"]:
        raise ValueError(f"Y3-F RGB参考checkpoint SHA不一致: seed{seed}")
    return {
        "run": str(run_dir),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": config["products"]["best_pt_sha256"],
    }


def verify_args_yaml(path: Path, seed: int, debug: bool) -> dict:
    """逐项核对Ultralytics实际持久化的CD1 Gray训练参数。"""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        **CD1_GRAY_TRAIN_ARGS,
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
    expected_data = (GRAY_DATA_ROOT / "data.yaml").resolve()
    if Path(str(values.get("data", ""))).resolve() != expected_data:
        mismatches["data"] = {"expected": str(expected_data), "actual": values.get("data")}
    if mismatches:
        raise RuntimeError(f"CD1 Gray持久化训练参数不一致: {mismatches}")
    return values


def verify_products(run_dir: Path, debug: bool) -> dict:
    """检查训练历史和checkpoint并返回产品摘要。"""
    required = [
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CD1 Gray训练产物不完整: {missing}")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if debug and len(history) != 1:
        raise ValueError(f"CD1 Gray debug应恰好完成1轮，实际{len(history)}轮")
    if not debug and not 1 <= len(history) <= FORMAL_EPOCHS:
        raise ValueError(f"CD1 Gray正式训练轮数异常: {len(history)}")
    return {
        "epochs_completed": int(len(history)),
        "best_checkpoint": str(run_dir / "weights/best.pt"),
        "best_pt_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_checkpoint": str(run_dir / "weights/last.pt"),
        "last_pt_sha256": file_sha256(run_dir / "weights/last.pt"),
        "results_csv": str(run_dir / "results.csv"),
        "results_csv_sha256": file_sha256(run_dir / "results.csv"),
        "best_validation_map50_95": float(history["metrics/mAP50-95(B)"].max()),
        "checkpoint_selection": "ultralytics_validation_fitness_map50_95",
    }


def main() -> None:
    """训练一个独立Gray seed，不计算CD1资格指标或部署阈值。"""
    args = parse_args()
    if args.debug and args.resume:
        raise ValueError("debug运行不允许resume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝启动CD1 Gray训练")
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(f"缺少冻结预训练权重: {PRETRAINED_PATH}")
    data_audit = load_gray_data_audit()
    rgb_reference = load_rgb_reference(args.seed)
    code_version = git_snapshot()

    output_root = args.output_root.resolve()
    run_name = expected_run_name(args.seed, args.debug)
    run_dir = output_root / run_name
    resume_metadata = None
    if args.resume:
        last_checkpoint = run_dir / "weights/last.pt"
        history_path = run_dir / "results.csv"
        if not last_checkpoint.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"CD1 Gray恢复产物不完整: {run_dir}")
        if (run_dir / "cd1_gray_train_config.json").exists():
            raise FileExistsError("CD1 Gray配置已存在，拒绝恢复已完成run")
        verify_args_yaml(run_dir / "args.yaml", args.seed, False)
        history = pd.read_csv(history_path)
        if not 1 <= len(history) < FORMAL_EPOCHS:
            raise ValueError(f"CD1 Gray恢复前历史轮数异常: {len(history)}")
        resume_metadata = {
            "checkpoint": str(last_checkpoint),
            "checkpoint_sha256": file_sha256(last_checkpoint),
            "completed_epochs_before_resume": int(len(history)),
        }
        model = YOLO(str(last_checkpoint))
        environment = assert_locked_library_behavior(model)
        model.train(resume=True, device=args.device)
    else:
        if run_dir.exists():
            raise FileExistsError(f"CD1 Gray输出已存在，拒绝覆盖: {run_dir}")
        model = YOLO(str(PRETRAINED_PATH))
        environment = assert_locked_library_behavior(model)
        model.train(
            data=str(GRAY_DATA_ROOT / "data.yaml"),
            model=str(PRETRAINED_PATH),
            epochs=1 if args.debug else FORMAL_EPOCHS,
            patience=FORMAL_PATIENCE,
            imgsz=IMAGE_SIZE,
            device=args.device,
            seed=args.seed,
            fraction=1.0,
            nms=False,
            max_det=300,
            project=str(output_root),
            name=run_name,
            exist_ok=False,
            verbose=True,
            **CD1_GRAY_TRAIN_ARGS,
        )

    persisted_args = verify_args_yaml(run_dir / "args.yaml", args.seed, args.debug)
    products = verify_products(run_dir, args.debug)
    best_model = YOLO(products["best_checkpoint"])
    args_path = run_dir / "args.yaml"
    config = {
        "stage": "CD1",
        "role": "gray_detector",
        "seed": args.seed,
        "debug": args.debug,
        "internal_temporal_test_read": False,
        "external_read": False,
        "qualification_evaluated": False,
        "deployment_threshold_frozen": False,
        "code_git_commit": code_version["git_commit"],
        "git_dirty": code_version["git_dirty"],
        "locked_ultralytics": LOCKED_ULTRALYTICS_VERSION,
        "environment": environment,
        "best_checkpoint_behavior": assert_locked_library_behavior(best_model),
        "gray_data": data_audit,
        "rgb_reference": rgb_reference,
        "pretrained_checkpoint": str(PRETRAINED_PATH),
        "pretrained_sha256": file_sha256(PRETRAINED_PATH),
        "training_protocol": {
            "model": "yolo26s.pt",
            "imgsz": IMAGE_SIZE,
            "seed": args.seed,
            "epochs": 1 if args.debug else FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE,
            **CD1_GRAY_TRAIN_ARGS,
        },
        "args_yaml": str(args_path),
        "args_yaml_sha256": file_sha256(args_path),
        "persisted_args": persisted_args,
        "resume": resume_metadata,
        "products": products,
    }
    config_path = run_dir / "cd1_gray_train_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CD1 Gray seed{args.seed} {'debug' if args.debug else 'formal'}训练完成: {run_dir}")
    print("本入口不作Gray资格判断；正式三seed完成后必须由evaluate_cd1_val.py统一评价。")


if __name__ == "__main__":
    main()
