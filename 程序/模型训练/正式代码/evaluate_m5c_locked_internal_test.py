#!/usr/bin/env python3
"""按冻结协议一次性评价M5c-A内部test，不允许从test重新选择参数。

正式患者概率：全局/局部图片概率分别top2_mean，再固定0.5/0.5平均。图像级固定均值仅
作为辅助诊断。患者阈值、图像辅助阈值、ROI协议和checkpoint SHA均来自冻结协议。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m1_localization import EfficientNetM1  # noqa: E402
from efficientnet_m5b_fusion import (  # noqa: E402
    M5bDataset, M5bModel, build_transforms, patient_top2_mean,
)
from train_utils import file_sha256, git_snapshot, json_ready, seed_worker  # noqa: E402

DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "结果/M5c概率融合_0804/冻结内部test协议"
    / "m5c_internal_test_protocol.json"
)
ROI_ROOT = PROJECT_ROOT / "结果/M5c概率融合_0804/冻结内部test_ROI清单"
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M5c概率融合_0804/锁定内部test评估"
SEEDS = (42, 202, 503)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def classification_metrics(labels, probabilities, threshold):
    """返回固定阈值下Accuracy/Sensitivity/Specificity/Precision/F1和混淆矩阵。"""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = probabilities >= threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * sensitivity / max(precision + sensitivity, 1e-12),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def paired_bootstrap(frame, iterations, seed):
    """在患者行上有放回抽样，估计两路AUC及其配对差异的95%区间。"""
    labels = frame.label.to_numpy(dtype=int)
    global_scores = frame.p_global.to_numpy(dtype=float)
    fused_scores = frame.p_fixed_mean.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    global_values = []
    fused_values = []
    differences = []
    for _ in range(iterations):
        index = rng.choice(len(frame), size=len(frame), replace=True)
        sampled = labels[index]
        if np.unique(sampled).size < 2:
            continue
        global_auc = roc_auc_score(sampled, global_scores[index])
        fused_auc = roc_auc_score(sampled, fused_scores[index])
        global_values.append(global_auc)
        fused_values.append(fused_auc)
        differences.append(fused_auc - global_auc)
    if not differences:
        raise RuntimeError("内部test bootstrap没有有效双类别抽样")
    return {
        "global_auc_ci95": np.quantile(global_values, [0.025, 0.975]).tolist(),
        "m5c_auc_ci95": np.quantile(fused_values, [0.025, 0.975]).tolist(),
        "delta_mean": float(np.mean(differences)),
        "delta_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "iterations_used": len(differences),
    }


def load_test_roi(path, image_root, debug):
    """校验test ROI字段与图像存在性；正式模式要求198张/153人。"""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"patient_id": str})
    required = {
        "split", "image_relpath", "patient_id", "label", "roi_source",
        "roi_crop_bbox_x1", "roi_crop_bbox_y1", "roi_crop_bbox_x2",
        "roi_crop_bbox_y2", "width", "height", "roi_label", "gate_pass",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"test ROI清单缺少字段: {sorted(missing)}")
    if set(frame.split.unique()) != {"test"}:
        raise ValueError("锁定评价只接受split=test")
    if not debug and (len(frame) != 198 or frame.patient_id.nunique() != 153):
        raise ValueError("正式test ROI规模不是198张/153人")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("test患者跨标签")
    for row in frame.itertuples(index=False):
        crop = (row.roi_crop_bbox_x1, row.roi_crop_bbox_y1,
                row.roi_crop_bbox_x2, row.roi_crop_bbox_y2)
        if not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError(f"test ROI越界: {row.image_relpath}")
        if not (image_root / row.image_relpath).is_file():
            raise FileNotFoundError(row.image_relpath)
    return frame.reset_index(drop=True)


def extract_predictions(model, dataset, device, batch_size, num_workers):
    """冻结前向得到每张test图的全局、局部及辅助固定均值概率。"""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            rois = batch["roi"].to(device, non_blocking=True)
            gate = batch["gate_pass"].to(device)
            outputs = model(images, rois, gate)
            for local, row_index in enumerate(batch["row_index"].tolist()):
                meta = dataset.df.iloc[row_index]
                p_global = float(torch.sigmoid(outputs["m_global"][local]).cpu())
                p_local = float(outputs["p_local"][local].cpu())
                rows.append({
                    "row_index": row_index, "image_relpath": meta.image_relpath,
                    "patient_id": meta.patient_id, "label": int(meta.label),
                    "p_global": p_global, "p_local": p_local,
                    "p_fixed_mean_image_auxiliary": 0.5 * (p_global + p_local),
                    "roi_source": meta.roi_source,
                    **{
                        column: meta[column]
                        for column in ("source", "center", "size_group")
                        if column in dataset.df.columns
                    },
                })
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def aggregate_patients(images):
    """两分支分别top2_mean后固定等权平均，严格对应M5c-A正式公式。"""
    global_patients = patient_top2_mean(images, "p_global").rename(
        columns={"patient_probability": "p_global"}
    )
    local_patients = patient_top2_mean(images, "p_local").rename(
        columns={"patient_probability": "p_local"}
    )
    patients = global_patients.merge(
        local_patients[["patient_id", "label", "p_local"]],
        on=["patient_id", "label"], validate="one_to_one",
    )
    patients["p_fixed_mean"] = 0.5 * (patients.p_global + patients.p_local)
    return patients.sort_values("patient_id").reset_index(drop=True)


def stratified_report(images):
    """按冻结元数据分层报告样本量和患者AUC，不参与模型或阈值选择。"""
    report = {}
    for column in ("source", "center", "size_group"):
        if column not in images.columns:
            continue
        groups = {}
        for value, group in images.groupby(column, dropna=False):
            patients = aggregate_patients(group)
            record = {
                "n_images": int(len(group)),
                "n_patients": int(len(patients)),
                "patient_labels": {
                    str(k): int(v)
                    for k, v in patients.label.value_counts().sort_index().items()
                },
            }
            if patients.label.nunique() == 2:
                global_auc = float(roc_auc_score(patients.label, patients.p_global))
                m5c_auc = float(roc_auc_score(patients.label, patients.p_fixed_mean))
                record.update({
                    "global_patient_auc": global_auc,
                    "m5c_patient_auc": m5c_auc,
                    "delta_patient_auc": m5c_auc - global_auc,
                })
            groups[str(value)] = record
        report[column] = groups
    return report


def self_test():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    if result["confusion_matrix"] != {"tn": 1, "fp": 1, "fn": 1, "tp": 1}:
        raise AssertionError("固定阈值指标计算错误")


def main():
    """校验冻结链，执行一次test前向并保存正式患者/图像辅助结果。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M5c锁定test评价自测通过: 固定阈值指标")
        return
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.batch_size != int(protocol["inference_batch_size"]):
        raise ValueError("评价batch size与冻结协议不一致")
    seed_record = protocol["seed_records"][str(args.seed)]
    checkpoint = Path(seed_record["m5b_checkpoint"])
    if file_sha256(checkpoint) != seed_record["m5b_checkpoint_sha256"]:
        raise ValueError("冻结M5b-B checkpoint SHA不一致")
    roi_manifest = args.roi_manifest or (
        ROI_ROOT / f"m5c_internal_test_roi_seed{args.seed}.csv"
    )
    roi_config_path = Path(str(roi_manifest).replace(".csv", ".json"))
    if not roi_manifest.is_file() or not roi_config_path.is_file():
        raise FileNotFoundError(f"缺少test ROI清单/config: {roi_manifest}")
    roi_config = json.loads(roi_config_path.read_text(encoding="utf-8"))
    if roi_config.get("debug", True) and not args.debug:
        raise ValueError("正式test评价拒绝debug ROI清单")
    if roi_config["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("test ROI与当前冻结协议不同源")
    if roi_config["roi_csv_sha256"] != file_sha256(roi_manifest):
        raise ValueError("test ROI CSV SHA与config不一致")

    run_name = args.run_name or f"m5c_locked_internal_test_seed{args.seed}"
    output = args.output_root / (run_name + ("_debug" if args.debug else ""))
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"test评价目录已存在，拒绝覆盖: {output}")
    frame = load_test_roi(roi_manifest, args.image_root, args.debug)
    eval_transform, _, roi_eval_transform = build_transforms()
    dataset = M5bDataset(frame, args.image_root, eval_transform, roi_eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = EfficientNetM1()
    model = M5bModel(base.backbone, "gate_local").to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    images = extract_predictions(
        model, dataset, device, args.batch_size, args.num_workers,
    )
    patients = aggregate_patients(images)

    global_auc = float(roc_auc_score(patients.label, patients.p_global))
    fused_auc = float(roc_auc_score(patients.label, patients.p_fixed_mean))
    image_auc = float(roc_auc_score(
        images.label, images.p_fixed_mean_image_auxiliary
    ))
    results = {
        "global_patient_auc": global_auc,
        "m5c_patient_auc": fused_auc,
        "delta_patient_auc": fused_auc - global_auc,
        "patient_paired_bootstrap": paired_bootstrap(
            patients, args.bootstrap, args.seed,
        ),
        "m5c_patient_at_frozen_sens90_threshold": classification_metrics(
            patients.label, patients.p_fixed_mean,
            seed_record["patient_threshold_sens90"],
        ),
        "m5c_patient_at_0_5": classification_metrics(
            patients.label, patients.p_fixed_mean, 0.5,
        ),
        "global_patient_at_frozen_sens90_threshold": classification_metrics(
            patients.label, patients.p_global,
            seed_record["global_patient_threshold_sens90"],
        ),
        "image_auxiliary_auc": image_auc,
        "image_auxiliary_at_frozen_sens90_threshold": classification_metrics(
            images.label, images.p_fixed_mean_image_auxiliary,
            seed_record["image_threshold_sens90_auxiliary"],
        ),
        "image_auxiliary_at_0_5": classification_metrics(
            images.label, images.p_fixed_mean_image_auxiliary, 0.5,
        ),
        "stratified_diagnostics": stratified_report(images),
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": file_sha256(args.protocol),
        "roi_manifest": str(roi_manifest.resolve()),
        "roi_manifest_sha256": file_sha256(roi_manifest),
        "m5b_checkpoint": str(checkpoint.resolve()),
        "m5b_checkpoint_sha256": file_sha256(checkpoint),
        "n_test_images": int(len(images)),
        "n_test_patients": int(len(patients)),
        "primary_formula": "0.5*p_global_patient_top2_mean + 0.5*p_local_patient_top2_mean",
        "thresholds_from_val_only": True,
        "results": results,
        "test_evaluated": True,
        "external_evaluated": False,
        **git_snapshot(),
    }
    output.mkdir(parents=True, exist_ok=args.overwrite)
    images.to_csv(output / "test_image_predictions.csv", index=False, encoding="utf-8-sig")
    patients.to_csv(
        output / "test_patient_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output / "config.json").write_text(
        json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    threshold_metrics = results["m5c_patient_at_frozen_sens90_threshold"]
    print(
        f"seed={args.seed}: global AUC={global_auc:.4f} -> M5c AUC={fused_auc:.4f} "
        f"Δ={fused_auc-global_auc:+.4f} CI={results['patient_paired_bootstrap']['delta_ci95']}"
    )
    print(
        f"冻结阈值={threshold_metrics['threshold']:.6f} | "
        f"Acc={threshold_metrics['accuracy']:.4f} "
        f"Sens={threshold_metrics['sensitivity']:.4f} "
        f"Spec={threshold_metrics['specificity']:.4f} "
        f"CM={threshold_metrics['confusion_matrix']}"
    )
    print(f"输出目录: {output}")


if __name__ == "__main__":
    main()
