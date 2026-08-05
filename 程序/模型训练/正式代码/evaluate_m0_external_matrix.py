#!/usr/bin/env python3
"""在当前Keep预处理的独立多中心集上评估M0锁定模型矩阵。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from efficientnet_train_debiased import build_model as build_efficientnet  # noqa: E402
from resnet_train_debiased import build_model as build_resnet  # noqa: E402
from train_utils import build_transforms  # noqa: E402


DEFAULT_PREPROCESS_ROOT = (
    PROJECT_DIR / "数据整理记录" / "图像裁剪"
    / "胃镜多中心测试集_M0Keep预处理_v1_20260805"
)
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_DIR / "结果" / "去偏重训练_v1"
    / "外部多中心完整测试_v1" / "preprocess_manifest.csv"
)
DEFAULT_RUN_ROOT = (
    PROJECT_DIR / "结果" / "M0全量诊断_0804" / "正式验证集筛选"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "结果" / "M0全量诊断_0804"
    / "外部多中心完整测试_Keep_v1"
)
MODEL_CONFIG = {
    "resnet50": (build_resnet, "resnet50_debiased_best.pth"),
    "efficientnet_b0": (
        build_efficientnet,
        "efficientnet_b0_debiased_best.pth",
    ),
}


class ExternalDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.frame.iloc[index]["image_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), index


def parse_args():
    parser = argparse.ArgumentParser(
        description="M0 Keep锁定模型的独立多中心外部评估"
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--stage1-mapping",
        type=Path,
        default=DEFAULT_PREPROCESS_ROOT / "01_亮度四边裁剪" / "mapping.csv",
    )
    parser.add_argument(
        "--fov-mapping",
        type=Path,
        default=DEFAULT_PREPROCESS_ROOT / "02_FOV遮罩" / "mapping.csv",
    )
    parser.add_argument(
        "--dark-decisions",
        type=Path,
        default=(
            DEFAULT_PREPROCESS_ROOT / "01b_暗部裁剪安全复核"
            / "人工复核决定_20260805.csv"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--run-prefix",
        default="m0_full_keep",
        help="运行目录名前缀，例如m0_full_keep或m0_balanced_keep。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=("all", "resnet50", "efficientnet_b0"),
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 202, 503])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_external_manifest(args):
    source = pd.read_csv(
        args.source_manifest,
        encoding="utf-8-sig",
        dtype={"patient_id": str},
    )
    stage1 = pd.read_csv(args.stage1_mapping, encoding="utf-8-sig")
    fov = pd.read_csv(args.fov_mapping, encoding="utf-8-sig")
    if stage1.review_status.eq("error").any() or fov.review_status.eq("error").any():
        raise ValueError("外部Keep预处理存在error行，拒绝评估")
    if source.original_sha256.duplicated().any():
        raise ValueError("外部测试集存在重复SHA256")

    stage1 = stage1[["source", "crop", "relative_path"]].rename(
        columns={"source": "original_path", "crop": "stage1_path"}
    )
    fov = fov[["source", "masked", "review_status", "review_reasons"]].rename(
        columns={
            "source": "stage1_path",
            "masked": "image_path",
            "review_status": "fov_review_status",
            "review_reasons": "fov_review_reasons",
        }
    )
    for frame, column in ((source, "original_path"), (stage1, "original_path"),
                          (stage1, "stage1_path"), (fov, "stage1_path")):
        frame[column] = frame[column].map(lambda value: str(Path(value).resolve()))
    merged = source.merge(stage1, on="original_path", how="inner", validate="one_to_one")
    merged = merged.merge(fov, on="stage1_path", how="inner", validate="one_to_one")
    if len(merged) != len(source):
        raise ValueError(f"外部预处理连接后数量不一致: {len(merged)}/{len(source)}")

    decisions = pd.read_csv(args.dark_decisions, encoding="utf-8-sig")
    decisions = decisions.loc[
        decisions.decision.eq("use_conservative"),
        ["source", "conservative_fov_masked_candidate", "decision"],
    ].rename(columns={"source": "original_path"})
    decisions["original_path"] = decisions.original_path.map(
        lambda value: str(Path(value).resolve())
    )
    decisions["conservative_fov_masked_candidate"] = (
        decisions.conservative_fov_masked_candidate.map(
            lambda value: str(Path(value).resolve())
        )
    )
    merged = merged.merge(
        decisions,
        on="original_path",
        how="left",
        validate="one_to_one",
    )
    merged["input_variant"] = np.where(
        merged.decision.eq("use_conservative"),
        "conservative_fov_masked",
        "standard_fov_masked",
    )
    use_conservative = merged.decision.eq("use_conservative")
    merged.loc[use_conservative, "image_path"] = merged.loc[
        use_conservative, "conservative_fov_masked_candidate"
    ]
    if int(use_conservative.sum()) != len(decisions):
        raise ValueError("暗部人工回退决定未全部连接到外部清单")
    missing = [path for path in merged.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少{len(missing)}张FOV处理图")
    if merged.groupby("patient_id").label.nunique().gt(1).any():
        raise ValueError("外部患者跨标签")
    return merged.reset_index(drop=True)


def aggregate_patients(frame):
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        labels = group.label.unique()
        if len(labels) != 1:
            raise ValueError(f"患者标签不唯一: {patient_id}")
        rows.append({
            "patient_id": patient_id,
            "label": int(labels[0]),
            "image_count": int(len(group)),
            "cancer_probability": float(group.cancer_probability.mean()),
        })
    return pd.DataFrame(rows)


def metric_row(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "Accuracy": float((predictions == labels).mean()),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1": float(
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity else 0.0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def clustered_auc_ci(image_frame, probability_column, iterations, seed):
    rng = np.random.default_rng(seed)
    patient_table = image_frame[["patient_id", "label"]].drop_duplicates()
    groups = {
        label: patient_table.loc[patient_table.label.eq(label), "patient_id"].to_numpy()
        for label in (0, 1)
    }
    by_patient = {
        patient_id: group[["label", probability_column]].to_numpy(dtype=float)
        for patient_id, group in image_frame.groupby("patient_id", sort=False)
    }
    aucs = []
    for _ in range(iterations):
        blocks = []
        for label in (0, 1):
            sampled = rng.choice(groups[label], size=len(groups[label]), replace=True)
            blocks.extend(by_patient[patient_id] for patient_id in sampled)
        values = np.concatenate(blocks, axis=0)
        aucs.append(roc_auc_score(values[:, 0], values[:, 1]))
    return np.quantile(aucs, [0.025, 0.975]).tolist()


def patient_auc_ci(patient_frame, iterations, seed):
    return clustered_auc_ci(
        patient_frame.rename(columns={"cancer_probability": "probability"}),
        "probability",
        iterations,
        seed,
    )


def load_run(model_name, seed, args, device):
    run_name = f"{args.run_prefix}_{model_name}_seed{seed}"
    run_dir = args.run_root / run_name
    with (run_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("defer_test"):
        raise ValueError(f"{run_name}不是defer-test锁定运行")
    threshold = float(config["threshold_selection"]["threshold"])
    builder, weight_name = MODEL_CONFIG[model_name]
    checkpoint = torch.load(
        run_dir / weight_name,
        map_location=device,
        weights_only=False,
    )
    model = builder(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return run_name, run_dir, model.to(device).eval(), threshold


def collect_probabilities(model, loader, device):
    probabilities = np.zeros(len(loader.dataset), dtype=np.float32)
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            probabilities[indices.numpy()] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  batch {batch_index}/{len(loader)}", flush=True)
    return probabilities


def build_cross_seed_summary(summary):
    metrics = [
        "patient_AUC", "patient_Accuracy", "patient_Sensitivity",
        "patient_Specificity", "image_AUC", "image_Accuracy",
        "image_Sensitivity", "image_Specificity",
    ]
    rows = []
    for model, group in summary.groupby("model", sort=True):
        row = {"model": model, "seed_count": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_worst"] = float(group[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary, cross_seed, output_path, run_prefix):
    columns = [
        "model", "seed", "threshold", "patient_AUC", "patient_AUC_CI95",
        "patient_Accuracy", "patient_Sensitivity", "patient_Specificity",
        "image_AUC", "image_AUC_CI95", "image_Accuracy",
        "image_Sensitivity", "image_Specificity",
    ]
    experiment_label = "M0来源内平衡模型" if "balanced" in run_prefix else "M0全量模型"
    lines = [
        f"# {experiment_label}独立多中心外部评估",
        "",
        "> 模型与阈值均在val阶段锁定；本结果不得反向用于选择模型或调参。",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## 跨种子汇总",
        "",
        "| model | patient_AUC mean±SD | patient_Accuracy | patient_Sensitivity | patient_Specificity | image_AUC mean±SD | image_Accuracy | image_Sensitivity | image_Specificity |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for row in cross_seed.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['patient_AUC_mean']:.4f}±{row['patient_AUC_std']:.4f} "
            f"| {row['patient_Accuracy_mean']:.4f} | {row['patient_Sensitivity_mean']:.4f} "
            f"| {row['patient_Specificity_mean']:.4f} | "
            f"{row['image_AUC_mean']:.4f}±{row['image_AUC_std']:.4f} "
            f"| {row['image_Accuracy_mean']:.4f} | {row['image_Sensitivity_mean']:.4f} "
            f"| {row['image_Specificity_mean']:.4f} |"
        )
    lines.extend([
        "",
        "患者概率采用每位患者全部图片癌概率均值。图片级阈值与患者级阈值均沿用",
        "对应模型在val患者上按Sensitivity>=0.90锁定的同一阈值；图片级没有重新调阈值。",
        "图片AUC置信区间按患者整簇bootstrap计算。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空，拒绝覆盖: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)

    external = build_external_manifest(args)
    external.to_csv(
        args.output / "external_keep_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    patient_counts = external[["patient_id", "label"]].drop_duplicates().label.value_counts().sort_index().to_dict()
    print(
        f"外部数据: {external.patient_id.nunique()}人/{len(external)}张; "
        f"患者={patient_counts}; 图片={external.label.value_counts().sort_index().to_dict()}",
        flush=True,
    )

    _, eval_transform = build_transforms()
    dataset = ExternalDataset(external, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model_names = list(MODEL_CONFIG) if args.model == "all" else [args.model]
    summary_rows = []
    for model_index, model_name in enumerate(model_names):
        for seed in args.seeds:
            run_name, run_dir, model, threshold = load_run(
                model_name, seed, args, device
            )
            print(f"[{run_name}] threshold={threshold:.6f}", flush=True)
            probabilities = collect_probabilities(model, loader, device)
            image_predictions = external.copy()
            image_predictions["cancer_probability"] = probabilities
            image_predictions["prediction"] = (
                image_predictions.cancer_probability >= threshold
            ).astype(int)
            patients = aggregate_patients(image_predictions)
            patients["prediction"] = (
                patients.cancer_probability >= threshold
            ).astype(int)
            image_metrics = metric_row(
                image_predictions.label, probabilities, threshold
            )
            patient_metrics = metric_row(
                patients.label, patients.cancer_probability, threshold
            )
            image_ci = clustered_auc_ci(
                image_predictions,
                "cancer_probability",
                args.bootstrap,
                args.bootstrap_seed + model_index * 1000 + seed,
            )
            patient_ci = patient_auc_ci(
                patients,
                args.bootstrap,
                args.bootstrap_seed + model_index * 1000 + seed,
            )
            run_output = args.output / run_name
            run_output.mkdir(parents=True, exist_ok=True)
            image_predictions.to_csv(
                run_output / "image_predictions.csv", index=False, encoding="utf-8-sig"
            )
            patients.to_csv(
                run_output / "patient_predictions.csv", index=False, encoding="utf-8-sig"
            )
            detail = {
                "run_name": run_name,
                "run_dir": str(run_dir.resolve()),
                "threshold_source": "val_patient_locked",
                "threshold": threshold,
                "patient_aggregation": "mean",
                "image": {**image_metrics, "AUC_CI95": image_ci},
                "patient": {**patient_metrics, "AUC_CI95": patient_ci},
            }
            (run_output / "metrics.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary_rows.append({
                "run_name": run_name,
                "model": model_name,
                "seed": seed,
                "threshold": threshold,
                "patient_AUC": patient_metrics["AUC"],
                "patient_AUC_CI95": f"[{patient_ci[0]:.4f}, {patient_ci[1]:.4f}]",
                "patient_Accuracy": patient_metrics["Accuracy"],
                "patient_Sensitivity": patient_metrics["Sensitivity"],
                "patient_Specificity": patient_metrics["Specificity"],
                "image_AUC": image_metrics["AUC"],
                "image_AUC_CI95": f"[{image_ci[0]:.4f}, {image_ci[1]:.4f}]",
                "image_Accuracy": image_metrics["Accuracy"],
                "image_Sensitivity": image_metrics["Sensitivity"],
                "image_Specificity": image_metrics["Specificity"],
            })
            print(
                f"  patient AUC={patient_metrics['AUC']:.4f}, "
                f"Acc={patient_metrics['Accuracy']:.4f}, "
                f"Sens={patient_metrics['Sensitivity']:.4f}, "
                f"Spec={patient_metrics['Specificity']:.4f}; "
                f"image AUC={image_metrics['AUC']:.4f}, "
                f"Acc={image_metrics['Accuracy']:.4f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "external_metrics_summary.csv", index=False, encoding="utf-8-sig")
    cross_seed = build_cross_seed_summary(summary)
    cross_seed.to_csv(
        args.output / "external_cross_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown(
        summary,
        cross_seed,
        args.output / "外部评估汇总.md",
        args.run_prefix,
    )
    print(f"结果已保存至: {args.output}", flush=True)


if __name__ == "__main__":
    main()
