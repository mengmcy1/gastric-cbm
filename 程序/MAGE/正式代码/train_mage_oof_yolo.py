#!/usr/bin/env python3
"""训练一个MG0b交叉拟合YOLO，并预测该模型未见过的留出折。

每个检测器只接触当前折的拟合和监控患者；训练结束后仅对留出患者预测一次，
保存低阈值Top-1候选供MAGE教师使用，不读取项目验证、测试或外部数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MODEL_CODE_DIR = PROJECT_ROOT / "程序/模型训练/正式代码"
sys.path.insert(0, str(MODEL_CODE_DIR))

from evaluate_y2_yolo26 import predict_val  # noqa: E402
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


DATA_ROOT = PROJECT_ROOT / "数据整理记录/MAGE/MG0b_OOF_YOLO数据_20260817"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/MAGE/MG0b_OOF_YOLO_20260817"
IMAGE_SIZE = 640
N_FOLDS = 5
BASE_SEED = 42
# 在当前锁定的Pillow版本中，Ultralytics异步批次绘图可能收到坐标方向颠倒的框。
# 绘图只属于诊断副作用，因此MG0b将其关闭；这不会改变优化、验证指标或权重选择。
MAGE_TRAIN_ARGS = {**FROZEN_TRAIN_ARGS, "plots": False}


def parse_args() -> argparse.Namespace:
    """解析外层折、CUDA设备、输出目录、调试和续训模式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), required=True)
    parser.add_argument("--device", required=True, help="nvidia-smi检查后选定的可见CUDA编号。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_name(fold: int, debug: bool) -> str:
    """生成包含折号的运行名称，并隔离调试产物。"""
    return f"mg0b_oof_yolo26s_fold{fold}{'_debug' if debug else ''}"


def load_fold_mapping(fold: int) -> tuple[Path, pd.DataFrame]:
    """加载一个MG0b折，并校验拟合/监控/留出患者互斥。"""
    fold_root = DATA_ROOT / f"fold_{fold}"
    mapping_path = fold_root / "fold_mapping.csv"
    config_path = DATA_ROOT / "mg0b_data_views_config.json"
    if not mapping_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"MG0b fold数据视图不完整: {fold_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["stage"] != "MG0b_data_views" or config["original_project_val_used_for_training_or_early_stop"]:
        raise ValueError("MG0b数据视图角色或val边界异常")
    frame = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype={"patient_id": str})
    if len(frame) != 2350 or set(frame.oof_role) != {"fit", "monitor", "holdout"}:
        raise ValueError(f"fold{fold} mapping规模或角色异常")
    patient_sets = {
        role: set(frame.loc[frame.oof_role.eq(role), "patient_id"])
        for role in ("fit", "monitor", "holdout")
    }
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in (("fit", "monitor"), ("fit", "holdout"), ("monitor", "holdout"))
    ):
        raise ValueError(f"fold{fold}存在患者跨fit/monitor/holdout")
    return fold_root, frame


def verify_args_yaml(path: Path, fold: int, debug: bool, data_yaml: Path) -> dict:
    """将Ultralytics保存的设置与冻结Y3-F协议逐项核对。"""
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        **MAGE_TRAIN_ARGS,
        "imgsz": IMAGE_SIZE,
        "seed": BASE_SEED + fold,
        "epochs": 1 if debug else FORMAL_EPOCHS,
        "patience": FORMAL_PATIENCE,
        "fraction": 1.0,
        "nms": False,
        "max_det": 300,
    }
    mismatches = {
        key: {"expected": value, "actual": values.get(key)}
        for key, value in expected.items()
        if values.get(key) != value
    }
    if Path(str(values.get("data", ""))).resolve() != data_yaml.resolve():
        mismatches["data"] = {"expected": str(data_yaml), "actual": values.get("data")}
    if mismatches:
        raise RuntimeError(f"MG0b训练参数不一致: {mismatches}")
    return values


def verify_training_products(run_dir: Path, debug: bool) -> dict:
    """确认权重文件存在，且训练历史包含有效的1至100轮记录。"""
    required = [
        run_dir / "args.yaml", run_dir / "results.csv",
        run_dir / "weights/best.pt", run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MG0b训练产物不完整: {missing}")
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if debug and len(history) != 1:
        raise ValueError("MG0b debug应只完成1 epoch")
    if not debug and not 1 <= len(history) <= FORMAL_EPOCHS:
        raise ValueError("MG0b正式训练epoch数异常")
    return {
        "epochs_completed": int(len(history)),
        "best_checkpoint_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_checkpoint_sha256": file_sha256(run_dir / "weights/last.pt"),
        "best_map50_95": float(history["metrics/mAP50-95(B)"].max()),
    }


def predict_holdout(
    run_dir: Path, fold_root: Path, mapping: pd.DataFrame, device: str
) -> tuple[Path, dict]:
    """保存当前检测器未见患者的Top-1预测。"""
    holdout = mapping.loc[mapping.oof_role.eq("holdout")].copy().reset_index(drop=True)
    model = YOLO(str(run_dir / "weights/best.pt"))
    behavior = assert_locked_library_behavior(model)
    predictions = predict_val(
        model,
        holdout,
        IMAGE_SIZE,
        device,
        source_dir=fold_root / "images/holdout",
    )
    if len(predictions) != len(holdout) or predictions.patient_id.nunique() != holdout.patient_id.nunique():
        raise RuntimeError("MG0b holdout预测未完整覆盖冻结队列")
    predictions["prediction_provenance"] = f"oof_fold_{int(holdout.oof_fold.iloc[0])}"
    output_path = run_dir / "holdout_top1_predictions.csv"
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    cancer = predictions.label.eq(1)
    return output_path, {
        "images": int(len(predictions)),
        "patients": int(predictions.patient_id.nunique()),
        "cancer_images": int(cancer.sum()),
        "noncancer_images": int((~cancer).sum()),
        "images_with_candidate_at_0p001": int(predictions.top1_confidence.ge(0.001).sum()),
        "noncancer_with_candidate_at_0p001": int(
            predictions.loc[~cancer, "top1_confidence"].ge(0.001).sum()
        ),
        "checkpoint_behavior": behavior,
    }


def run_self_test() -> None:
    """自测运行命名，并确认五个折的数据视图均已准备。"""
    assert run_name(2, False) == "mg0b_oof_yolo26s_fold2"
    assert run_name(2, True).endswith("_debug")
    for fold in range(N_FOLDS):
        load_fold_mapping(fold)
    print("MG0b训练入口self-test通过: 5折数据视图与患者隔离有效")


def main() -> None:
    """训练或续训一个检测器，预测留出折并冻结产物血缘。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.debug and args.resume:
        raise ValueError("debug不允许resume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，拒绝启动MG0b OOF YOLO")
    fold_root, mapping = load_fold_mapping(args.fold)
    data_yaml = fold_root / "data.yaml"
    output_root = args.output_root.resolve()
    name = run_name(args.fold, args.debug)
    run_dir = output_root / name
    if args.resume:
        last = run_dir / "weights/last.pt"
        if not last.is_file() or (run_dir / "mg0b_config.json").exists():
            raise FileNotFoundError(f"MG0b无可安全恢复的last.pt: {run_dir}")
        verify_args_yaml(run_dir / "args.yaml", args.fold, False, data_yaml)
        model = YOLO(str(last))
        environment = assert_locked_library_behavior(model)
        model.train(resume=True, device=args.device)
    else:
        if run_dir.exists():
            raise FileExistsError(f"MG0b输出已存在: {run_dir}")
        model = YOLO(str(PRETRAINED_PATH))
        environment = assert_locked_library_behavior(model)
        model.train(
            data=str(data_yaml),
            model=str(PRETRAINED_PATH),
            epochs=1 if args.debug else FORMAL_EPOCHS,
            patience=FORMAL_PATIENCE,
            imgsz=IMAGE_SIZE,
            device=args.device,
            seed=BASE_SEED + args.fold,
            fraction=1.0,
            project=str(output_root),
            name=name,
            exist_ok=False,
            verbose=True,
            **MAGE_TRAIN_ARGS,
        )

    persisted = verify_args_yaml(run_dir / "args.yaml", args.fold, args.debug, data_yaml)
    products = verify_training_products(run_dir, args.debug)
    predictions_path, prediction_summary = predict_holdout(
        run_dir, fold_root, mapping, args.device
    )
    config = {
        "stage": "MG0b_oof_yolo",
        "fold": args.fold,
        "debug": args.debug,
        "train_seed": BASE_SEED + args.fold,
        "data_yaml": str(data_yaml),
        "fold_mapping": str(fold_root / "fold_mapping.csv"),
        "fold_mapping_sha256": file_sha256(fold_root / "fold_mapping.csv"),
        "pretrained": str(PRETRAINED_PATH),
        "pretrained_sha256": file_sha256(PRETRAINED_PATH),
        "locked_ultralytics": LOCKED_ULTRALYTICS_VERSION,
        "environment": environment,
        "training_protocol": {
            "model": "yolo26s.pt", "imgsz": IMAGE_SIZE,
            "epochs": 1 if args.debug else FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE, **MAGE_TRAIN_ARGS,
        },
        "persisted_args": persisted,
        "products": products,
        "holdout_predictions": str(predictions_path),
        "holdout_predictions_sha256": file_sha256(predictions_path),
        "holdout_summary": prediction_summary,
        "project_val_read": False,
        "project_test_read": False,
        "internal_test_read": False,
        "external_read": False,
    }
    (run_dir / "mg0b_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MG0b fold{args.fold}完成: {run_dir}")
    print(prediction_summary)


if __name__ == "__main__":
    main()
