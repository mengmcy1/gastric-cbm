#!/usr/bin/env python3
"""Run the five-epoch Y1 YOLO26n pipeline smoke test.

Y1 validates the exported YOLO dataset, the locked Ultralytics environment,
end-to-end model behavior, training/validation, and expected artifacts. Its
metrics must not be used to choose a model, resolution, threshold, or epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import Metric


PROJECT_ROOT = Path(__file__).resolve().parents[3]
Y0_ROOT = PROJECT_ROOT / (
    "数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/"
    "10_Y0_YOLO26检测数据_20260813"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y1_smoke"
LOCKED_ULTRALYTICS_VERSION = "8.4.118"


def parse_args() -> argparse.Namespace:
    """Return the small set of settings allowed for the Y1 engineering run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="CUDA device index selected after nvidia-smi.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="y1_yolo26n_640_seed42")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one configuration or checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_y0_dataset() -> dict:
    """Validate the frozen Y0 view without reading or exporting internal-test labels."""
    config_path = Y0_ROOT / "y0_config.json"
    data_yaml_path = Y0_ROOT / "data.yaml"
    mapping_path = Y0_ROOT / "y0_mapping.csv"
    for path in (config_path, data_yaml_path, mapping_path):
        if not path.is_file():
            raise FileNotFoundError(f"Y1缺少Y0产物: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["test_exported_to_yolo"] or config["test_evaluated"]:
        raise ValueError("Y0 test隔离标志异常，拒绝运行Y1")
    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if "test" in data_yaml:
        raise ValueError("data.yaml不得包含internal test")
    if data_yaml.get("names") != {0: "early_cancer_or_HGD"}:
        raise ValueError("YOLO类别定义与冻结协议不一致")

    mapping = pd.read_csv(mapping_path)
    development = mapping[mapping["split"].isin(["train", "val"])]
    test = mapping[mapping["split"].eq("test")]
    if development["yolo_image_relpath"].isna().any() or development["yolo_label_relpath"].isna().any():
        raise ValueError("train/val存在未导出的YOLO路径")
    if test["yolo_image_relpath"].notna().any() or test["yolo_label_relpath"].notna().any():
        raise ValueError("internal test被意外导出到YOLO视图")

    label_counts = {"positive": 0, "empty_negative": 0}
    for row in development.itertuples(index=False):
        image_path = Y0_ROOT / row.yolo_image_relpath
        label_path = Y0_ROOT / row.yolo_label_relpath
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"YOLO视图文件缺失: {image_path} / {label_path}")
        content = label_path.read_text(encoding="ascii").strip()
        if int(row.label) == 1:
            fields = content.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"癌图YOLO标签格式异常: {label_path}")
            label_counts["positive"] += 1
        elif content:
            raise ValueError(f"非癌标签必须为空: {label_path}")
        else:
            label_counts["empty_negative"] += 1
    return {
        "y0_config_sha256": file_sha256(config_path),
        "data_yaml_sha256": file_sha256(data_yaml_path),
        "development_images": int(len(development)),
        "locked_test_images": int(len(test)),
        "label_counts": label_counts,
    }


def assert_locked_library_behavior(model: YOLO) -> dict:
    """Verify the installed version, NMS-free head and mAP50-95 checkpoint fitness."""
    if ultralytics.__version__ != LOCKED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Ultralytics版本漂移: {ultralytics.__version__} != {LOCKED_ULTRALYTICS_VERSION}"
        )
    head = model.model.model[-1]
    head_end2end = bool(getattr(head, "end2end", False))
    model_end2end = bool(getattr(model.model, "end2end", False))
    if not head_end2end or not model_end2end:
        raise RuntimeError("锁定YOLO26模型不是end2end=True，拒绝运行正式Y1")

    fitness_source = inspect.getsource(Metric.fitness)
    expected_weights = "[0.0, 0.0, 0.0, 1.0]"
    if expected_weights not in fitness_source:
        raise RuntimeError("Detection fitness源码已变化，需先重新冻结checkpoint规则")
    return {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "head_type": type(head).__name__,
        "head_end2end": head_end2end,
        "model_end2end": model_end2end,
        "nms_used": False,
        "checkpoint_fitness": "val_mAP50-95",
        "fitness_weights": [0.0, 0.0, 0.0, 1.0],
    }


def verify_training_artifacts(run_dir: Path) -> dict:
    """Require the standard training products needed before Y2 can be designed."""
    expected = [
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        run_dir / "weights/last.pt",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Y1训练产物不完整: {missing}")
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = results.columns.str.strip()
    if len(results) != 5:
        raise ValueError(f"Y1应完成5个epoch，实际为{len(results)}")
    required_metrics = {"metrics/mAP50(B)", "metrics/mAP50-95(B)"}
    if not required_metrics.issubset(results.columns):
        raise ValueError(f"results.csv缺少验证指标: {sorted(required_metrics - set(results.columns))}")
    return {
        "epochs_completed": int(len(results)),
        "best_pt_sha256": file_sha256(run_dir / "weights/best.pt"),
        "last_pt_sha256": file_sha256(run_dir / "weights/last.pt"),
        "final_map50": float(results.iloc[-1]["metrics/mAP50(B)"]),
        "final_map50_95": float(results.iloc[-1]["metrics/mAP50-95(B)"]),
    }


def main() -> None:
    """Execute Y1 and save a machine-readable smoke-test acceptance record."""
    args = parse_args()
    output_root = args.output_root.resolve()
    run_dir = output_root / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Y1输出已存在，拒绝覆盖: {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用CUDA，拒绝启动Y1 GPU smoke")

    y0_audit = assert_y0_dataset()
    pretrained_path = PROJECT_ROOT / "yolo26n.pt"
    model = YOLO(str(pretrained_path if pretrained_path.is_file() else "yolo26n.pt"))
    environment = assert_locked_library_behavior(model)

    model.train(
        data=str(Y0_ROOT / "data.yaml"),
        epochs=5,
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(output_root),
        name=args.run_name,
        exist_ok=False,
        pretrained=True,
        val=True,
        plots=True,
        verbose=True,
    )

    artifacts = verify_training_artifacts(run_dir)
    best_model = YOLO(str(run_dir / "weights/best.pt"))
    best_behavior = assert_locked_library_behavior(best_model)
    config = {
        "stage": "Y1",
        "purpose": "pipeline_smoke_only",
        "performance_selection_allowed": False,
        "internal_test_read": False,
        "external_read": False,
        "data": y0_audit,
        "environment": environment,
        "best_checkpoint_behavior": best_behavior,
        "training": {
            "model": "yolo26n.pt",
            "imgsz": 640,
            "epochs": 5,
            "batch": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "device": args.device,
        },
        "artifacts": artifacts,
        "accepted": True,
    }
    (run_dir / "y1_smoke_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Y1 smoke通过: {run_dir}")
    print("注意: Y1指标只证明流程可运行，不得用于Y2模型或分辨率选择。")


if __name__ == "__main__":
    main()
