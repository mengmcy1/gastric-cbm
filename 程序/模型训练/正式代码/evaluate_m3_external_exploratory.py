#!/usr/bin/env python3
"""探索性比较M1与M3 seed42在完整外部多中心集上的定位置信度。

该脚本不重新选择阈值，不参与模型选择。它固定使用M3 config中由内部val锁定的M1
定位阈值，在同一批已裁剪外部图像上比较M1/M3的癌召回和非癌FP。外部集没有bbox，
因此不报告IoU或center-hit；manifest也没有医院字段，只按分辨率、画幅和裁剪状态等
风格代理分层。结果必须标注为“仅val成功seed42的事后探索性外部诊断”。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_m1_localization import BBoxAwareTransform, EfficientNetM1  # noqa: E402
from train_utils import file_sha256, json_ready, seed_worker  # noqa: E402


EXTERNAL_MANIFEST = (
    PROJECT_ROOT / "结果/去偏重训练_v1/外部多中心完整测试_v1/preprocess_manifest.csv"
)
M1_CHECKPOINT = (
    PROJECT_ROOT / "结果/M1辅助定位_0804/正式验证集筛选"
    / "m1_balanced_keep_efficientnet_b0_seed42_warmup_product"
    / "m1_best_warmup_localization.pth"
)
M3_RUN_DIR = (
    PROJECT_ROOT / "结果/M3定位抑制_0804/正式验证集筛选"
    / "m3_balanced_keep_efficientnet_b0_seed42"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "结果/M3定位抑制_0804/探索性外部诊断_seed42"
)


class ExternalDataset(Dataset):
    """读取已裁剪外部图，并应用与M1验证阶段一致的确定性224变换。"""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.transform = BBoxAwareTransform(training=False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.frame.iloc[index].processed_path) as source:
            image = source.convert("RGB")
        image, _ = self.transform(image, None)
        return image, torch.tensor(index, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=EXTERNAL_MANIFEST)
    parser.add_argument("--m1-checkpoint", type=Path, default=M1_CHECKPOINT)
    parser.add_argument("--m3-run-dir", type=Path, default=M3_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-patients-per-class", type=int, default=0,
        help="每类抽取指定患者数做链路调试；0表示完整外部集。",
    )
    return parser.parse_args()


def debug_subset(frame, patients_per_class, seed):
    """按标签抽取患者，确保调试集仍同时含癌与非癌。"""
    if patients_per_class <= 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in frame[["patient_id", "label"]].drop_duplicates().groupby("label"):
        patients = group.patient_id.to_numpy()
        selected.extend(rng.choice(
            patients, size=min(patients_per_class, len(patients)), replace=False
        ))
    return frame.loc[frame.patient_id.isin(selected)].copy()


def load_model(checkpoint, device):
    """加载M1架构checkpoint并返回eval模型。"""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = EfficientNetM1()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def infer(model, loader, device, model_name):
    """提取分类概率、定位最大响应和单候选预测框，保持原始清单顺序。"""
    count = len(loader.dataset)
    cancer_probability = np.zeros(count, dtype=np.float32)
    localization_confidence = np.zeros(count, dtype=np.float32)
    boxes = np.zeros((count, 4), dtype=np.float32)
    processed = 0
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            outputs = model(images.to(device, non_blocking=True))
            heatmap = torch.sigmoid(outputs["heatmap_logits"])
            confidence, peak = heatmap.flatten(1).max(dim=1)
            cell_y = torch.div(peak, 14, rounding_mode="floor")
            cell_x = peak % 14
            flat_size = outputs["size"].flatten(2).transpose(1, 2)
            flat_offset = outputs["offset"].flatten(2).transpose(1, 2)
            gather = peak[:, None, None].expand(-1, 1, 2)
            size = flat_size.gather(1, gather).squeeze(1)
            offset = flat_offset.gather(1, gather).squeeze(1)
            center_x = (cell_x.float() + offset[:, 0]) / 14
            center_y = (cell_y.float() + offset[:, 1]) / 14
            predicted = torch.stack((
                center_x - size[:, 0] / 2,
                center_y - size[:, 1] / 2,
                center_x + size[:, 0] / 2,
                center_y + size[:, 1] / 2,
            ), dim=1).clamp(0, 1)
            index_array = indices.numpy()
            cancer_probability[index_array] = (
                torch.softmax(outputs["logits"], dim=1)[:, 1].cpu().numpy()
            )
            localization_confidence[index_array] = confidence.cpu().numpy()
            boxes[index_array] = predicted.cpu().numpy()
            processed += len(images)
            if batch_index % 10 == 0 or processed == count:
                print(
                    f"{model_name} 推理进度: {processed}/{count}张 "
                    f"({processed / count:.1%})",
                    flush=True,
                )
    return cancer_probability, localization_confidence, boxes


def detection_metrics(labels, scores, threshold):
    """计算固定阈值灵敏度、非癌FP、特异度和置信度AUC。"""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    cancer = labels == 1
    noncancer = ~cancer
    return {
        "count": int(len(labels)),
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_recall": float(predicted[cancer].mean()),
        "noncancer_fp_rate": float(predicted[noncancer].mean()),
        "specificity": float((~predicted[noncancer]).mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "cancer_detected": int(predicted[cancer].sum()),
        "noncancer_false_positive": int(predicted[noncancer].sum()),
    }


def add_style_proxies(frame):
    """根据外部manifest现有字段构造可复现的设备/画面风格代理。"""
    result = frame.copy()
    long_side = result[["original_width", "original_height"]].max(axis=1)
    result["resolution_group"] = pd.cut(
        long_side,
        bins=[-np.inf, 800, 1280, 1920, np.inf],
        labels=["small_le800", "medium_801_1280", "large_1281_1920", "xlarge_gt1920"],
    ).astype(str)
    ratio = result.original_width / result.original_height.clip(lower=1)
    result["aspect_proxy"] = np.select(
        [ratio < 0.9, ratio <= 1.1, ratio < 1.5, ratio <= 1.9],
        ["portrait", "square", "landscape_4_3", "landscape_16_9"],
        default="wide",
    )
    return result


def style_diagnostics(frame, threshold):
    """按风格代理分别输出M1/M3固定阈值指标；小组仅描述，不作模型选择。"""
    rows = []
    columns = [
        "resolution_group", "aspect_proxy", "crop_method", "crop_status",
        "edge_refined", "progress_bar_detected", "legacy_crop_reused",
    ]
    for column in columns:
        if column not in frame:
            continue
        for value, group in frame.groupby(column, dropna=False):
            for model_name in ["m1", "m3"]:
                metrics = detection_metrics(
                    group.label, group[f"{model_name}_localization_confidence"], threshold
                ) if group.label.nunique() == 2 else {
                    "count": len(group),
                    "cancer_count": int(group.label.eq(1).sum()),
                    "noncancer_count": int(group.label.eq(0).sum()),
                    "cancer_recall": float(
                        (group.loc[group.label.eq(1), f"{model_name}_localization_confidence"] >= threshold).mean()
                    ) if group.label.eq(1).any() else np.nan,
                    "noncancer_fp_rate": float(
                        (group.loc[group.label.eq(0), f"{model_name}_localization_confidence"] >= threshold).mean()
                    ) if group.label.eq(0).any() else np.nan,
                    "specificity": np.nan,
                    "auc": np.nan,
                    "cancer_detected": 0,
                    "noncancer_false_positive": 0,
                }
                rows.append({"grouping": column, "group": str(value), "model": model_name, **metrics})
    return pd.DataFrame(rows)


def patient_metrics(frame, score_column, threshold):
    """每患者取最大定位响应，报告任一图像触发时的患者级检出与误报。"""
    patient = frame.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), score=(score_column, "max")
    )
    return detection_metrics(patient.label, patient.score, threshold)


def paired_patient_bootstrap(frame, threshold, iterations, seed):
    """按标签分层后对患者有放回抽样，估计M3-M1变化的95%CI。

    癌与非癌患者分别保持原患者数进行重采样，既保留患者内多图相关性与重复抽样权重，
    也避免小样本bootstrap偶然只抽到一个类别而产生空均值。
    """
    rng = np.random.default_rng(seed)
    patient_rows = {
        patient: group.index.to_numpy()
        for patient, group in frame.groupby("patient_id", sort=False)
    }
    patient_table = frame[["patient_id", "label"]].drop_duplicates("patient_id")
    class_patients = {
        label: patient_table.loc[patient_table.label.eq(label), "patient_id"].to_numpy()
        for label in [0, 1]
    }
    if any(len(patients) == 0 for patients in class_patients.values()):
        raise ValueError("患者bootstrap要求癌与非癌患者均非空")
    recall_delta, fp_delta = [], []
    labels = frame.label.to_numpy()
    m1 = frame.m1_localization_confidence.to_numpy()
    m3 = frame.m3_localization_confidence.to_numpy()
    for _ in range(iterations):
        sampled = np.concatenate([
            rng.choice(patients, size=len(patients), replace=True)
            for patients in class_patients.values()
        ])
        indices = np.concatenate([patient_rows[patient] for patient in sampled])
        cancer = labels[indices] == 1
        noncancer = ~cancer
        recall_delta.append(float(
            (m3[indices][cancer] >= threshold).mean()
            - (m1[indices][cancer] >= threshold).mean()
        ))
        fp_delta.append(float(
            (m3[indices][noncancer] >= threshold).mean()
            - (m1[indices][noncancer] >= threshold).mean()
        ))
    return {
        "m3_minus_m1_cancer_recall_delta_ci": np.quantile(recall_delta, [0.025, 0.975]).tolist(),
        "m3_minus_m1_noncancer_fp_delta_ci": np.quantile(fp_delta, [0.025, 0.975]).tolist(),
    }


def main():
    args = parse_args()
    m3_checkpoint = args.m3_run_dir / "m3_best_localization.pth"
    m3_config_path = args.m3_run_dir / "config.json"
    for path in [args.manifest, args.m1_checkpoint, m3_checkpoint, m3_config_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir
    if args.debug_patients_per_class > 0:
        output = output.with_name(output.name + "_debug")
    if output.exists():
        raise FileExistsError(f"输出已存在，请更换输出目录: {output}")
    output.mkdir(parents=True)

    frame = pd.read_csv(args.manifest, encoding="utf-8-sig")
    required = {"patient_id", "label", "processed_path", "original_width", "original_height"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"外部manifest缺少列: {sorted(missing)}")
    frame = debug_subset(frame, args.debug_patients_per_class, args.seed)
    frame = add_style_proxies(frame).reset_index(drop=True)
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("评估集必须同时包含癌与非癌")
    if not frame.processed_path.map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("外部manifest存在缺失的processed_path")

    m3_config = json.loads(m3_config_path.read_text(encoding="utf-8"))
    if not m3_config["m3_selection"]["passed_seed_success"]:
        raise ValueError("指定M3 run不是val成功seed，拒绝作为本探索性对照")
    threshold = float(m3_config["fixed_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExternalDataset(frame)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
    )
    print(
        f"设备: {device}; 外部集: {frame.patient_id.nunique()}人/{len(frame)}张; "
        f"固定内部阈值={threshold:.4f}"
    )
    m1_model = load_model(args.m1_checkpoint, device)
    m3_model = load_model(m3_checkpoint, device)
    for name, model in [("m1", m1_model), ("m3", m3_model)]:
        probability, confidence, boxes = infer(model, loader, device, name.upper())
        frame[f"{name}_cancer_probability"] = probability
        frame[f"{name}_localization_confidence"] = confidence
        for index, coordinate in enumerate(["x1", "y1", "x2", "y2"]):
            frame[f"{name}_pred_{coordinate}"] = boxes[:, index]

    classification_diff = float(np.max(np.abs(
        frame.m1_cancer_probability - frame.m3_cancer_probability
    )))
    if classification_diff > 1e-6:
        raise RuntimeError(f"冻结分类概率不一致: max_abs_diff={classification_diff:.2e}")

    summary = {
        "protocol": "仅val成功seed42的事后探索性外部诊断；阈值固定，不参与模型选择。",
        "external_bbox_available": False,
        "center_metadata_available": False,
        "threshold_source": "M3 seed42 config中的内部val固定M1阈值",
        "fixed_threshold": threshold,
        "image": {
            "m1": detection_metrics(frame.label, frame.m1_localization_confidence, threshold),
            "m3": detection_metrics(frame.label, frame.m3_localization_confidence, threshold),
        },
        "patient_max": {
            "m1": patient_metrics(frame, "m1_localization_confidence", threshold),
            "m3": patient_metrics(frame, "m3_localization_confidence", threshold),
        },
        "classification_max_abs_diff": classification_diff,
        "paired_patient_bootstrap": paired_patient_bootstrap(
            frame, threshold, args.bootstrap, args.seed
        ),
        "counts": {
            "images": int(len(frame)),
            "patients": int(frame.patient_id.nunique()),
            "cancer_images": int(frame.label.eq(1).sum()),
            "noncancer_images": int(frame.label.eq(0).sum()),
        },
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "m1_checkpoint": str(args.m1_checkpoint.resolve()),
            "m1_checkpoint_sha256": file_sha256(args.m1_checkpoint),
            "m3_checkpoint": str(m3_checkpoint.resolve()),
            "m3_checkpoint_sha256": file_sha256(m3_checkpoint),
        },
        "bootstrap_iterations": args.bootstrap,
        "seed": args.seed,
    }

    m1_positive = frame.m1_localization_confidence >= threshold
    m3_positive = frame.m3_localization_confidence >= threshold
    frame["transition"] = np.select(
        [m1_positive & m3_positive, m1_positive & ~m3_positive,
         ~m1_positive & m3_positive],
        ["positive_to_positive", "positive_to_negative", "negative_to_positive"],
        default="negative_to_negative",
    )
    style = style_diagnostics(frame, threshold)
    frame.to_csv(output / "external_image_predictions_m1_vs_m3.csv", index=False, encoding="utf-8-sig")
    style.to_csv(output / "style_proxy_diagnostics.csv", index=False, encoding="utf-8-sig")
    frame.loc[frame.label.eq(0) & m1_positive & ~m3_positive].to_csv(
        output / "noncancer_suppressed_by_m3.csv", index=False, encoding="utf-8-sig"
    )
    frame.loc[frame.label.eq(1) & m1_positive & ~m3_positive].to_csv(
        output / "cancer_detection_lost_by_m3.csv", index=False, encoding="utf-8-sig"
    )
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), output / "source_entry.py")
    print(
        "图像级 M1: "
        f"癌召回={summary['image']['m1']['cancer_recall']:.3f}, "
        f"非癌FP={summary['image']['m1']['noncancer_fp_rate']:.3f}; "
        "M3: "
        f"癌召回={summary['image']['m3']['cancer_recall']:.3f}, "
        f"非癌FP={summary['image']['m3']['noncancer_fp_rate']:.3f}"
    )
    print(f"探索性结果已保存至: {output}")


if __name__ == "__main__":
    main()
