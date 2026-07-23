"""
多中心独立数据验证脚本。

当前可只评估癌数据的图片级、患者级敏感度；后续提供非癌目录后，
同一脚本会自动补充 AUC、特异度、准确率、Precision 和 F1。

默认直接运行：
  python evaluate_multicenter.py

指定单个模型：
  python evaluate_multicenter.py --model resnet50

后续增加非癌数据：
  python evaluate_multicenter.py --noncancer-dir /path/to/非癌
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from inference import DEVICE, MODEL_REGISTRY, PROJECT_DIR, load_model, transform


# 默认数据与输出位置；不传参数时可直接点击运行。
DEFAULT_CANCER_DIR = os.path.join(PROJECT_DIR, '数据', '胃镜多中心早癌图片挑选')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果', '多中心外部验证')

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
BATCH_SIZE = 32


class ExternalDataset(Dataset):
    """根据验证清单读取图片，返回模型输入和清单行号。"""

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image = Image.open(self.dataframe.iloc[index]['image_path']).convert('RGB')
        return transform(image), index


def scan_class_directory(class_dir, label):
    """扫描“患者/图片”目录，生成一类数据的验证清单。"""
    class_dir = Path(class_dir)
    records = []

    for patient_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
        for image_path in sorted(patient_dir.rglob('*')):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    'patient_id': patient_dir.name,
                    'label': label,
                    'class_name': '癌' if label == 1 else '非癌',
                    'image_name': image_path.name,
                    'image_path': str(image_path.resolve()),
                })

    return records


def build_manifest(cancer_dir, noncancer_dir=None):
    """合并癌和非癌目录；当前允许只提供癌目录。"""
    records = scan_class_directory(cancer_dir, label=1)
    if noncancer_dir:
        records.extend(scan_class_directory(noncancer_dir, label=0))

    manifest = pd.DataFrame(records)
    label_counts = manifest.groupby('patient_id')['label'].nunique()
    conflicting_patients = label_counts[label_counts > 1]
    if len(conflicting_patients):
        raise ValueError(f'发现跨标签患者: {conflicting_patients.index.tolist()}')

    return manifest


@torch.no_grad()
def collect_probabilities(model, manifest, batch_size, num_workers):
    """批量计算每张图片属于癌类的概率。"""
    dataset = ExternalDataset(manifest)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == 'cuda',
    )
    probabilities = np.zeros(len(dataset), dtype=np.float32)

    for images, indices in loader:
        logits = model(images.to(DEVICE, non_blocking=True))
        probabilities[indices.numpy()] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return probabilities


def wilson_interval(successes, total, z=1.96):
    """计算二项比例的 Wilson 95% 置信区间。"""
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)
    ) / denominator
    return center - margin, center + margin


def calculate_metrics(y_true, y_probability, threshold):
    """计算固定阈值指标；单类别数据中不可计算的指标返回空值。"""
    y_true = np.asarray(y_true, dtype=int)
    y_probability = np.asarray(y_probability, dtype=float)
    y_pred = (y_probability >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    accuracy = (tp + tn) / len(y_true)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) \
        if precision + sensitivity > 0 else np.nan
    auc = roc_auc_score(y_true, y_probability) if len(np.unique(y_true)) == 2 else np.nan
    sens_low, sens_high = wilson_interval(tp, tp + fn)
    spec_low, spec_high = wilson_interval(tn, tn + fp)

    return {
        'threshold': threshold,
        'sample_count': len(y_true),
        'positive_count': int((y_true == 1).sum()),
        'negative_count': int((y_true == 0).sum()),
        'auc': auc,
        'sensitivity': sensitivity,
        'sensitivity_ci_low': sens_low,
        'sensitivity_ci_high': sens_high,
        'specificity': specificity,
        'specificity_ci_low': spec_low,
        'specificity_ci_high': spec_high,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def aggregate_patients(image_results, locked_threshold):
    """汇总每位患者的概率；默认以最高概率作为患者级判定分数。"""
    patient_results = image_results.groupby(['patient_id', 'label'], as_index=False).agg(
        image_count=('cancer_probability', 'size'),
        max_probability=('cancer_probability', 'max'),
        mean_probability=('cancer_probability', 'mean'),
        median_probability=('cancer_probability', 'median'),
    )
    positive_counts = image_results.assign(
        positive_locked=image_results['cancer_probability'] >= locked_threshold,
        positive_0_50=image_results['cancer_probability'] >= 0.50,
    ).groupby('patient_id', as_index=False).agg(
        positive_images_locked=('positive_locked', 'sum'),
        positive_images_0_50=('positive_0_50', 'sum'),
    )
    return patient_results.merge(positive_counts, on='patient_id', how='left')


def print_metrics(title, metrics):
    """打印一组图片级或患者级固定阈值结果。"""
    print(f'\n{title}（阈值 {metrics["threshold"]:.2f}）')
    if not np.isnan(metrics['auc']):
        print(f'  AUC:         {metrics["auc"]:.4f}')
    if not np.isnan(metrics['sensitivity']):
        print(f'  Sensitivity: {metrics["sensitivity"]:.3f} '
              f'(95% CI {metrics["sensitivity_ci_low"]:.3f}–{metrics["sensitivity_ci_high"]:.3f})')
    if not np.isnan(metrics['specificity']):
        print(f'  Specificity: {metrics["specificity"]:.3f} '
              f'(95% CI {metrics["specificity_ci_low"]:.3f}–{metrics["specificity_ci_high"]:.3f})')
        print(f'  Accuracy:    {metrics["accuracy"]:.3f}')
        print(f'  Precision:   {metrics["precision"]:.3f}')
        print(f'  F1:          {metrics["f1"]:.3f}')
    print(f'  CM: TN={metrics["tn"]}, FP={metrics["fp"]}, '
          f'FN={metrics["fn"]}, TP={metrics["tp"]}')


def evaluate_model(model_name, manifest, args, run_dir):
    """评估一个模型并保存图片级、患者级和指标汇总结果。"""
    weight_file, locked_threshold = MODEL_REGISTRY[model_name]
    model_path = os.path.join(PROJECT_DIR, '结果', '模型权重', weight_file)
    model = load_model(model_name, model_path)

    image_results = manifest.copy()
    image_results['cancer_probability'] = collect_probabilities(
        model, manifest, args.batch_size, args.num_workers
    )
    image_results['prediction_locked'] = (
        image_results['cancer_probability'] >= locked_threshold
    ).astype(int)
    image_results['prediction_0_50'] = (
        image_results['cancer_probability'] >= 0.50
    ).astype(int)

    patient_results = aggregate_patients(image_results, locked_threshold)
    metric_rows = []
    for threshold_name, threshold in [('locked', locked_threshold), ('natural', 0.50)]:
        image_metrics = calculate_metrics(
            image_results['label'], image_results['cancer_probability'], threshold
        )
        patient_metrics = calculate_metrics(
            patient_results['label'], patient_results['max_probability'], threshold
        )
        metric_rows.append({
            'model': model_name,
            'level': 'image',
            'patient_aggregation': '',
            'threshold_name': threshold_name,
            **image_metrics,
        })
        metric_rows.append({
            'model': model_name,
            'level': 'patient',
            'patient_aggregation': 'max_probability',
            'threshold_name': threshold_name,
            **patient_metrics,
        })
        print_metrics(f'{model_name} 图片级', image_metrics)
        print_metrics(f'{model_name} 患者级（最高概率）', patient_metrics)

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    image_results.to_csv(model_dir / 'image_predictions.csv', index=False, encoding='utf-8-sig')
    patient_results.to_csv(model_dir / 'patient_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metric_rows).to_csv(
        model_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig'
    )

    missed = patient_results[
        (patient_results['label'] == 1)
        & (patient_results['max_probability'] < locked_threshold)
    ].sort_values('max_probability')
    missed.to_csv(model_dir / 'missed_cancer_patients.csv', index=False, encoding='utf-8-sig')

    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='多中心独立数据固定阈值验证')
    parser.add_argument('--model', default='all',
                        choices=['resnet50', 'efficientnet_b0', 'all'],
                        help='评估模型，默认依次评估两个模型')
    parser.add_argument('--cancer-dir', default=DEFAULT_CANCER_DIR,
                        help='癌数据目录，结构为患者/图片')
    parser.add_argument('--noncancer-dir', default=None,
                        help='非癌数据目录；未提供时只评估癌类敏感度')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='结果根目录')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='推理批大小，默认32')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据读取进程数，默认4')
    args = parser.parse_args()

    manifest = build_manifest(args.cancer_dir, args.noncancer_dir)
    run_name = '癌非癌完整验证' if args.noncancer_dir else '癌数据初步验证'
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(run_dir / 'evaluation_manifest.csv', index=False, encoding='utf-8-sig')

    patient_count = manifest['patient_id'].nunique()
    print(f'设备: {DEVICE}')
    print(f'验证图片: {len(manifest)} 张')
    print(f'验证患者: {patient_count} 人')
    print(f'标签分布: {manifest["label"].value_counts().sort_index().to_dict()}')
    if args.noncancer_dir is None:
        print('当前只有癌数据：AUC、Specificity 等指标暂不计算。')

    models = ['resnet50', 'efficientnet_b0'] if args.model == 'all' else [args.model]
    for model_name in models:
        print(f'\n{"=" * 60}\n开始评估 {model_name}\n{"=" * 60}')
        evaluate_model(model_name, manifest, args, run_dir)

    print(f'\n结果已保存至: {run_dir}')


if __name__ == '__main__':
    main()
