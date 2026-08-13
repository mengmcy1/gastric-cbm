#!/usr/bin/env python3
"""将冻结M1+M3c-B三seed投影到完整外部多中心集作诊断性比较。

外部集没有bbox，因此只报告全局分类概率与q_region的图像/患者AUC，以及冻结val门槛下
的癌召回和非癌FP；不报告IoU、中心命中，也不从外部数据重新选择阈值或模型。
"""

import argparse
import json
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
from efficientnet_m3c_region_gate import FrozenM1RegionGate  # noqa: E402
from train_utils import file_sha256, git_snapshot, json_ready, seed_worker  # noqa: E402

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "结果/M0平衡_0804/外部多中心完整测试_Keep_v1"
    / "external_keep_manifest.csv"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "结果/M3c区域门控_0804/冻结内部test协议"
    / "m3c_internal_test_protocol.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "结果/M3c区域门控_0804/诊断性外部多中心投影"
SEEDS = (42, 202, 503)


class ExternalDataset(Dataset):
    """读取冻结外部图像并应用与M1验证一致的确定性224变换。"""

    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.transform = BBoxAwareTransform(training=False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.frame.iloc[index].image_path) as source:
            image = source.convert("RGB")
        image, _ = self.transform(image, None)
        return image, torch.tensor(index, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def threshold_metrics(labels, scores, threshold):
    """返回冻结门槛下癌召回、非癌FP、特异度和计数。"""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    decisions = scores >= threshold
    cancer = labels == 1
    noncancer = labels == 0
    return {
        "threshold": float(threshold),
        "cancer_count": int(cancer.sum()),
        "noncancer_count": int(noncancer.sum()),
        "cancer_detected": int(decisions[cancer].sum()),
        "noncancer_false_positive": int(decisions[noncancer].sum()),
        "cancer_recall": float(decisions[cancer].mean()),
        "noncancer_fp": float(decisions[noncancer].mean()),
        "specificity": float((~decisions[noncancer]).mean()),
    }


def patient_frame(images, score_column, aggregation):
    """按患者均值或最大值聚合图像分数。"""
    if aggregation not in {"mean", "max"}:
        raise ValueError(aggregation)
    grouped = images.groupby("patient_id", as_index=False).agg(
        label=("label", "first"), score=(score_column, aggregation),
        image_count=("image_path", "size"),
    )
    return grouped


def stratified_patient_auc_ci(patient, iterations, seed):
    """按患者标签分层有放回估计患者AUC的95%区间。"""
    groups = {
        label: patient.loc[patient.label.eq(label)].reset_index(drop=True)
        for label in (0, 1)
    }
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(iterations):
        sampled = pd.concat([
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in groups.values()
        ], ignore_index=True)
        aucs.append(roc_auc_score(sampled.label, sampled.score))
    return np.quantile(aucs, [0.025, 0.975]).tolist()


def patient_auc_record(images, score_column, aggregation, iterations, seed):
    patient = patient_frame(images, score_column, aggregation)
    return patient, {
        "aggregation": aggregation,
        "auc": float(roc_auc_score(patient.label, patient.score)),
        "auc_ci95": stratified_patient_auc_ci(patient, iterations, seed),
        "n_patients": int(len(patient)),
    }


def self_test():
    metrics = threshold_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
    if metrics["cancer_recall"] != 0.5 or metrics["noncancer_fp"] != 0.5:
        raise AssertionError("外部门控固定阈值指标错误")


def main():
    """校验冻结链，前向完整外部集并保存单seed诊断结果。"""
    args = parse_args()
    if args.self_test:
        self_test()
        print("M3c-B外部诊断自测通过")
        return
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.batch_size != int(protocol["inference_batch_size"]):
        raise ValueError("外部推理batch size必须与冻结M3c-B协议一致")
    record = protocol["seed_records"][str(args.seed)]
    m1_checkpoint = Path(record["m1_checkpoint"])
    m3c_checkpoint = Path(record["m3c_checkpoint"])
    if file_sha256(m1_checkpoint) != record["m1_checkpoint_sha256"]:
        raise ValueError("M1 checkpoint SHA与冻结协议不一致")
    if file_sha256(m3c_checkpoint) != record["m3c_checkpoint_sha256"]:
        raise ValueError("M3c-B checkpoint SHA与冻结协议不一致")

    output = args.output_root / f"m3c_external_diagnostic_seed{args.seed}"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"外部诊断结果已存在，拒绝覆盖: {output}")
    frame = pd.read_csv(
        args.manifest, encoding="utf-8-sig", dtype={"patient_id": str}
    )
    required = {"patient_id", "label", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"外部清单缺少字段: {sorted(missing)}")
    if len(frame) != 1941 or frame.patient_id.nunique() != 1329:
        raise ValueError(f"外部队列规模异常: {len(frame)}张/{frame.patient_id.nunique()}人")
    if frame.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("外部患者跨标签")
    missing_paths = [value for value in frame.image_path if not Path(value).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"外部缺失{len(missing_paths)}张图: {missing_paths[0]}")

    m1_payload = torch.load(m1_checkpoint, map_location="cpu", weights_only=False)
    m3c_payload = torch.load(m3c_checkpoint, map_location="cpu", weights_only=False)
    m1 = EfficientNetM1()
    m1.load_state_dict(m1_payload["model_state_dict"], strict=True)
    model = FrozenM1RegionGate(m1)
    model.gate.load_state_dict(m3c_payload["gate_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = ExternalDataset(frame)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    q_region = np.zeros(len(frame), dtype=np.float32)
    p_cls = np.zeros(len(frame), dtype=np.float32)
    localization = np.zeros(len(frame), dtype=np.float32)
    boxes = np.zeros((len(frame), 4), dtype=np.float32)
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            outputs = model(images.to(device, non_blocking=True))
            index = indices.numpy()
            q_region[index] = torch.sigmoid(outputs["gate_logits"]).cpu().numpy()
            p_cls[index] = outputs["cancer_probability"].cpu().numpy()
            localization[index] = outputs["localization_confidence"].cpu().numpy()
            boxes[index] = outputs["boxes"].cpu().numpy()
            if batch_index % 10 == 0 or batch_index == len(loader):
                print(f"seed{args.seed}: {min(batch_index * args.batch_size, len(frame))}/{len(frame)}张", flush=True)
    images = frame.copy()
    images["q_region"] = q_region
    images["cancer_probability"] = p_cls
    images["localization_confidence"] = localization
    for index, name in enumerate(("x1", "y1", "x2", "y2")):
        images[f"m1_pred_bbox_{name}"] = boxes[:, index]

    gate_threshold = float(record["m3c_gate_threshold"])
    m1_threshold = float(record["m1_localization_threshold"])
    q_mean, q_mean_record = patient_auc_record(
        images, "q_region", "mean", args.bootstrap, args.seed
    )
    q_max, q_max_record = patient_auc_record(
        images, "q_region", "max", args.bootstrap, args.seed + 1000
    )
    cls_mean, cls_record = patient_auc_record(
        images, "cancer_probability", "mean", args.bootstrap, args.seed + 2000
    )
    summary = {
        "analysis_role": "diagnostic_external_projection_after_internal_test_nonconfirmation",
        "seed": args.seed,
        "n_images": int(len(images)),
        "n_patients": int(images.patient_id.nunique()),
        "image_labels": {str(k): int(v) for k, v in images.label.value_counts().items()},
        "patient_labels": {
            str(k): int(v) for k, v in
            images[["patient_id", "label"]].drop_duplicates().label.value_counts().items()
        },
        "image_auc": {
            "m1_classification_probability": float(roc_auc_score(images.label, images.cancer_probability)),
            "m3c_q_region": float(roc_auc_score(images.label, images.q_region)),
            "m1_localization_confidence": float(roc_auc_score(images.label, images.localization_confidence)),
        },
        "patient_auc": {
            "m1_classification_probability_mean": cls_record,
            "m3c_q_region_mean": q_mean_record,
            "m3c_q_region_max": q_max_record,
        },
        "frozen_threshold_metrics": {
            "m1_localization_image": threshold_metrics(images.label, images.localization_confidence, m1_threshold),
            "m3c_q_region_image": threshold_metrics(images.label, images.q_region, gate_threshold),
            "m1_localization_patient_max": threshold_metrics(q_max.label, patient_frame(images, "localization_confidence", "max").score, m1_threshold),
            "m3c_q_region_patient_max": threshold_metrics(q_max.label, q_max.score, gate_threshold),
        },
        "reference_auc": {
            "early_resnet50_external_patient_auc": 0.69336,
            "current_balanced_efficientnet_external_patient_auc_mean": 0.6807121485778921,
            "current_full_efficientnet_external_patient_auc_mean": 0.739917993906555,
            "note": "参考模型/训练集不同；只作并列表述，不作同一模型配对显著性检验。",
        },
        "m1_checkpoint": str(m1_checkpoint.resolve()),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint),
        "m3c_checkpoint": str(m3c_checkpoint.resolve()),
        "m3c_checkpoint_sha256": file_sha256(m3c_checkpoint),
        "m1_localization_threshold": m1_threshold,
        "m3c_gate_threshold": gate_threshold,
        "inference_batch_size": args.batch_size,
        "external_bbox_available": False,
        "threshold_reselected_on_external": False,
        "internal_test_conclusion_changed": False,
        **git_snapshot(),
    }
    output.mkdir(parents=True, exist_ok=True)
    images.to_csv(output / "external_image_predictions.csv", index=False, encoding="utf-8-sig")
    q_mean.rename(columns={"score": "q_region_mean"}).to_csv(
        output / "external_patient_q_region_mean.csv", index=False, encoding="utf-8-sig"
    )
    q_max.rename(columns={"score": "q_region_max"}).to_csv(
        output / "external_patient_q_region_max.csv", index=False, encoding="utf-8-sig"
    )
    cls_mean.rename(columns={"score": "cancer_probability_mean"}).to_csv(
        output / "external_patient_classification.csv", index=False, encoding="utf-8-sig"
    )
    (output / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"seed{args.seed}: p_cls患者AUC={cls_record['auc']:.4f}; "
        f"q_region患者AUC(mean/max)={q_mean_record['auc']:.4f}/{q_max_record['auc']:.4f}; "
        f"q固定阈值癌召回={summary['frozen_threshold_metrics']['m3c_q_region_image']['cancer_recall']:.4f}, "
        f"非癌FP={summary['frozen_threshold_metrics']['m3c_q_region_image']['noncancer_fp']:.4f}"
    )
    print(f"输出: {output}")


if __name__ == "__main__":
    main()
