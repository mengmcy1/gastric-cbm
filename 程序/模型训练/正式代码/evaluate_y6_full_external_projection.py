#!/usr/bin/env python3
"""将冻结Y6局部融合投影到既有外部Keep集，禁止外部调参。

本脚本复用Y5的YOLO框/触发状态和M0-F外部逐图概率，只对触发ROI运行Y6局部
EfficientNet。输出强制候选融合及Y6 val产品保护两种预先冻结口径。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_train_debiased import build_model  # noqa: E402
from efficientnet_y6_full_roi_classifier import (  # noqa: E402
    MARGIN,
    build_transforms,
    patient_mean,
    square_box,
)
from run_y1_yolo26_smoke import file_sha256  # noqa: E402


SEEDS = (42, 202, 503)
Y5_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y5外部诊断_Full_20260814"
M0_EXTERNAL_ROOT = PROJECT_ROOT / "结果/M0全量诊断_0804/外部多中心完整测试_Keep_v1"
M0_RUN_ROOT = PROJECT_ROOT / "结果/M0全量诊断_0804/正式验证集筛选"
Y6_ROOT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y6预测ROI局部分类_Full_20260814"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/YOLO26定位_0804/Y6外部探索投影_Full_20260814"


def parse_args() -> argparse.Namespace:
    """解析GPU、batch size、bootstrap次数和唯一输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class TriggeredROIDataset(Dataset):
    """读取Y5触发图片并裁剪扩边ROI，返回[3,224,224]和原表索引。"""
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index().rename(columns={"index": "source_index"})
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        box = square_box(
            np.array([row.top1_x1, row.top1_y1, row.top1_x2, row.top1_y2]),
            image.width, image.height, MARGIN,
        )
        x1, y1, x2, y2 = box
        crop = image.crop((
            int(x1 * image.width), int(y1 * image.height),
            max(int(x2 * image.width), int(x1 * image.width) + 1),
            max(int(y2 * image.height), int(y1 * image.height) + 1),
        )).resize((224, 224), Image.Resampling.BILINEAR)
        return self.transform(crop), int(row.source_index)


def load_seed_inputs(seed: int) -> tuple[pd.DataFrame, dict, dict]:
    """逐SHA/路径绑定Y5框、M0-F概率和Y6产品，返回外部图片表。"""
    y5_path = Y5_ROOT / f"y5_full_seed{seed}_external_image_predictions.csv"
    m0_path = (
        M0_EXTERNAL_ROOT / f"m0_full_keep_efficientnet_b0_seed{seed}"
        / "image_predictions.csv"
    )
    y6_dir = Y6_ROOT / f"y6_full_efficientnet_b0_seed{seed}"
    y6_config_path = y6_dir / "config.json"
    y6_config = json.loads(y6_config_path.read_text(encoding="utf-8"))
    m0_config_path = M0_RUN_ROOT / f"m0_full_keep_efficientnet_b0_seed{seed}" / "config.json"
    m0_config = json.loads(m0_config_path.read_text(encoding="utf-8"))
    y5 = pd.read_csv(y5_path, encoding="utf-8-sig", dtype={"patient_id": str})
    m0 = pd.read_csv(
        m0_path, encoding="utf-8-sig", dtype={"patient_id": str},
        usecols=["patient_id", "label", "original_sha256", "cancer_probability"],
    ).rename(columns={"cancer_probability": "p_global"})
    if y5.original_sha256.duplicated().any() or m0.original_sha256.duplicated().any():
        raise ValueError(f"seed{seed}外部逐图输入存在重复SHA")
    frame = y5.merge(
        m0, on=["patient_id", "label", "original_sha256"],
        how="left", validate="one_to_one",
    )
    if len(frame) != 1941 or frame.p_global.isna().any():
        raise ValueError(f"seed{seed} Y5与M0-F外部概率连接不完整")
    for path in frame.image_path:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    checkpoint = y6_dir / "y6_best_local.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["seed"]) != seed:
        raise ValueError(f"Y6局部checkpoint seed不一致: {seed}")
    if payload["m0_checkpoint_sha256"] != y6_config["m0_checkpoint_sha256"]:
        raise ValueError(f"Y6局部checkpoint M0血缘不一致: seed{seed}")
    frozen = {
        "y5_predictions": str(y5_path), "y5_predictions_sha256": file_sha256(y5_path),
        "m0_external_predictions": str(m0_path),
        "m0_external_predictions_sha256": file_sha256(m0_path),
        "y6_config": str(y6_config_path), "y6_config_sha256": file_sha256(y6_config_path),
        "y6_checkpoint": str(checkpoint), "y6_checkpoint_sha256": file_sha256(checkpoint),
        "candidate_threshold": float(
            y6_config["diagnostics"]["patient_candidate_threshold_metrics"]["threshold"]
        ),
        "global_threshold": float(m0_config["threshold_selection"]["threshold"]),
        "fell_back_on_val": bool(y6_config["selection"]["fell_back_to_global"]),
    }
    return frame, payload, frozen


def infer_local(
    frame: pd.DataFrame, payload: dict, device: torch.device,
    batch_size: int, num_workers: int,
) -> pd.DataFrame:
    """只对Y5冻结阈值已触发的ROI推理局部癌概率。"""
    output = frame.copy()
    output["p_local"] = np.nan
    triggered = output.loc[output.triggered.astype(bool)].copy()
    _, eval_transform = build_transforms()
    dataset = TriggeredROIDataset(triggered, eval_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(pretrained=False)
    model.load_state_dict(payload["local_model_state_dict"], strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        for images, source_indices in loader:
            probability = torch.softmax(
                model(images.to(device, non_blocking=True)), dim=1,
            )[:, 1].cpu().numpy()
            output.loc[source_indices.numpy(), "p_local"] = probability
    if output.loc[output.triggered.astype(bool), "p_local"].isna().any():
        raise RuntimeError("触发ROI局部概率未完整生成")
    output["p_candidate"] = output.p_global
    mask = output.triggered.astype(bool)
    output.loc[mask, "p_candidate"] = (
        output.loc[mask, "p_global"] + output.loc[mask, "p_local"]
    ) / 2
    return output


def metric_row(frame: pd.DataFrame, column: str, threshold: float) -> dict:
    """计算AUC及冻结阈值下Accuracy/Sensitivity/Specificity。"""
    prediction = frame[column].ge(threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(frame.label, prediction, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(frame.label, frame[column])),
        "Accuracy": float((prediction == frame.label).mean()),
        "Sensitivity": float(tp / (tp + fn)),
        "Specificity": float(tn / (tn + fp)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "threshold": float(threshold),
    }


def paired_bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> list[float]:
    """患者分层有放回抽样候选融合减全局患者AUC差。"""
    groups = {
        label: frame.loc[frame.label.eq(label), "patient_id"].to_numpy()
        for label in (0, 1)
    }
    rows = frame.set_index("patient_id")
    rng = np.random.default_rng(20260814 + seed)
    differences = []
    for _ in range(iterations):
        sampled = np.concatenate([
            rng.choice(groups[label], len(groups[label]), replace=True) for label in (0, 1)
        ])
        boot = rows.loc[sampled].reset_index()
        differences.append(float(
            roc_auc_score(boot.label, boot.p_candidate)
            - roc_auc_score(boot.label, boot.p_global)
        ))
    return differences


def run_self_test() -> None:
    """验证无框回退与有框固定均值公式。"""
    frame = pd.DataFrame({"p_global": [0.2, 0.4], "p_local": [np.nan, 0.8],
                          "triggered": [False, True]})
    frame["p_candidate"] = frame.p_global
    mask = frame.triggered
    frame.loc[mask, "p_candidate"] = (frame.loc[mask, "p_global"] + frame.loc[mask, "p_local"]) / 2
    assert np.allclose(frame.p_candidate, [0.2, 0.6])
    print("Y6 external projection self-test passed")


def main() -> None:
    """运行三seed外部投影并保存逐图、逐患者、指标和冻结血缘。"""
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.output.exists():
        raise FileExistsError(f"外部投影输出已存在，拒绝覆盖: {args.output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.device != "0":
        raise ValueError("请用CUDA_VISIBLE_DEVICES隔离物理GPU，脚本--device传0")
    args.output.mkdir(parents=True)
    summaries = []
    provenance = {}
    for seed in SEEDS:
        frame, payload, frozen = load_seed_inputs(seed)
        images = infer_local(frame, payload, device, args.batch_size, args.num_workers)
        patients_global = patient_mean(images, "p_global")
        patients_candidate = patient_mean(images, "p_candidate")
        images["p_product"] = (
            images.p_global if frozen["fell_back_on_val"] else images.p_candidate
        )
        patients_product = patient_mean(images, "p_product")
        ci = np.quantile(
            paired_bootstrap(patients_candidate.merge(
                patients_global, on=["patient_id", "label"], validate="one_to_one"
            ), args.bootstrap, seed), [0.025, 0.975],
        ).tolist()
        metrics = {
            "seed": seed,
            "image_global": metric_row(images, "p_global", frozen["global_threshold"]),
            "image_candidate": metric_row(images, "p_candidate", frozen["candidate_threshold"]),
            "patient_global": metric_row(patients_global, "p_global", frozen["global_threshold"]),
            "patient_candidate": metric_row(
                patients_candidate, "p_candidate", frozen["candidate_threshold"]
            ),
            "patient_product": metric_row(
                patients_product, "p_product",
                frozen["global_threshold"] if frozen["fell_back_on_val"]
                else frozen["candidate_threshold"],
            ),
            "patient_candidate_minus_global_auc": float(
                roc_auc_score(patients_candidate.label, patients_candidate.p_candidate)
                - roc_auc_score(patients_global.label, patients_global.p_global)
            ),
            "patient_candidate_minus_global_auc_ci95": ci,
            "fell_back_on_val": frozen["fell_back_on_val"],
        }
        summaries.append(metrics)
        provenance[str(seed)] = frozen
        images.to_csv(
            args.output / f"seed{seed}_external_image_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        patients_global.merge(
            patients_candidate, on=["patient_id", "label"], validate="one_to_one"
        ).merge(
            patients_product, on=["patient_id", "label"], validate="one_to_one"
        ).to_csv(
            args.output / f"seed{seed}_external_patient_predictions.csv",
            index=False, encoding="utf-8-sig",
        )
        (args.output / f"seed{seed}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"seed{seed}: image AUC {metrics['image_global']['AUC']:.4f}->"
            f"{metrics['image_candidate']['AUC']:.4f}; patient AUC "
            f"{metrics['patient_global']['AUC']:.4f}->"
            f"{metrics['patient_candidate']['AUC']:.4f} "
            f"Δ={metrics['patient_candidate_minus_global_auc']:+.4f}; "
            f"val回退={frozen['fell_back_on_val']}"
        )
    aggregate = {
        "stage": "Y6_failed_posthoc_external_projection",
        "external_images": 1941, "external_patients": 1329,
        "no_external_tuning": True, "external_bbox_available": False,
        "summaries": summaries, "provenance": provenance,
        "mean_image_global_auc": float(np.mean([x["image_global"]["AUC"] for x in summaries])),
        "mean_image_candidate_auc": float(np.mean([x["image_candidate"]["AUC"] for x in summaries])),
        "mean_patient_global_auc": float(np.mean([x["patient_global"]["AUC"] for x in summaries])),
        "mean_patient_candidate_auc": float(np.mean([x["patient_candidate"]["AUC"] for x in summaries])),
        "mean_patient_delta_auc": float(np.mean([
            x["patient_candidate_minus_global_auc"] for x in summaries
        ])),
    }
    (args.output / "external_projection_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
