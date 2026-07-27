#!/usr/bin/env python3
"""使用冻结模型和阈值评估独立多中心二分类测试集。"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SCRIPT_DIR = PROJECT_DIR / '数据整理脚本'
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_SCRIPT_DIR))

from efficientnet_train_debiased import build_model as build_efficientnet  # noqa: E402
from preprocess_second_batch_v1_1 import (  # noqa: E402
    OUTPUT_SIZE,
    detect_roi,
    draw_preview,
    read_image,
    refine_roi_by_edge_scan,
    resize_square,
    save_contact_sheets,
    write_jpeg,
)
from resnet_train_debiased import build_model as build_resnet  # noqa: E402
from train_utils import build_transforms, file_sha256  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_DIR / '数据' / '胃镜多中心测试集'
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / '结果' / '去偏重训练_v1' / '外部多中心完整测试_v1'
)
DEFAULT_TRAIN_MANIFEST = (
    PROJECT_DIR / '数据整理记录' / '第二批' / '预处理_v1_1'
    / 'full' / 'processed_manifest.csv'
)
DEFAULT_ENSEMBLE_SUMMARY = (
    PROJECT_DIR / '结果' / '去偏重训练_v1'
    / 'expA_ensemble_tta_v1' / 'summary.json'
)
DEFAULT_LEGACY_CANCER_MANIFEST = (
    PROJECT_DIR / '结果' / '去偏重训练_v1'
    / '外部多中心验证_v1' / 'preprocess_manifest.csv'
)
DEFAULT_RUNS = {
    'resnet50': PROJECT_DIR / '结果' / '去偏重训练_v1'
    / 'expA_full_resnet50_seed42',
    'efficientnet_b0': PROJECT_DIR / '结果' / '去偏重训练_v1'
    / 'expA_full_efficientnet_b0_seed42',
}
CLASS_DIRECTORIES = {
    0: Path('非癌') / '胃镜多中心非癌图片挑选（按患者）',
    1: Path('早癌'),
}
CLASS_NAMES = {0: '非癌', 1: '早癌'}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExternalProcessedDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        with Image.open(self.dataframe.iloc[index]['processed_path']) as source:
            image = source.convert('RGB')
        return self.transform(image), index


def parse_args():
    parser = argparse.ArgumentParser(
        description='冻结协议下的独立多中心癌/非癌测试'
    )
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model', choices=('all', 'resnet50', 'efficientnet_b0'),
                        default='all')
    parser.add_argument('--resnet-run-dir', type=Path,
                        default=DEFAULT_RUNS['resnet50'])
    parser.add_argument('--efficientnet-run-dir', type=Path,
                        default=DEFAULT_RUNS['efficientnet_b0'])
    parser.add_argument('--ensemble-summary', type=Path,
                        default=DEFAULT_ENSEMBLE_SUMMARY)
    parser.add_argument('--training-manifest', type=Path,
                        default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument('--legacy-cancer-manifest', type=Path,
                        default=DEFAULT_LEGACY_CANCER_MANIFEST)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug-patients-per-class', type=int, default=3)
    parser.add_argument('--preprocess-only', action='store_true',
                        help='仅生成裁剪结果和审计记录，不加载模型')
    parser.add_argument('--overwrite-preprocessed', action='store_true')
    return parser.parse_args()


def scan_external_directory(root):
    root = Path(root)
    rows = []
    for label, relative_root in CLASS_DIRECTORIES.items():
        class_root = root / relative_root
        if not class_root.is_dir():
            raise FileNotFoundError(f'缺少类别目录: {class_root}')
        patient_dirs = sorted(path for path in class_root.iterdir() if path.is_dir())
        for patient_dir in patient_dirs:
            image_paths = sorted(
                path for path in patient_dir.rglob('*')
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            )
            for image_index, image_path in enumerate(image_paths, start=1):
                rows.append({
                    'patient_id': patient_dir.name,
                    'label': label,
                    'class_name': CLASS_NAMES[label],
                    'image_name': image_path.name,
                    'image_index': image_index,
                    'original_path': str(image_path.resolve()),
                    'original_relpath': str(image_path.relative_to(root)),
                })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError('外部测试集未扫描到图片')
    if manifest.groupby('patient_id')['label'].nunique().gt(1).any():
        raise ValueError('检测到同名患者跨癌/非癌目录')
    if set(manifest['label']) != {0, 1}:
        raise ValueError('外部测试集必须同时包含癌和非癌')
    return manifest


def debug_subset(manifest, patients_per_class, seed):
    patient_table = manifest[['patient_id', 'label']].drop_duplicates()
    selected = pd.concat([
        group.sample(
            n=min(patients_per_class, len(group)),
            random_state=seed,
        )
        for _, group in patient_table.groupby('label', sort=True)
    ])
    result = manifest[
        manifest['patient_id'].isin(selected['patient_id'])
    ].reset_index(drop=True)
    print(
        f'[DEBUG] {result.patient_id.nunique()}人/{len(result)}张，'
        f'患者标签={selected.label.value_counts().sort_index().to_dict()}'
    )
    return result


def validate_against_training(manifest, training_manifest_path):
    external_hashes = manifest['original_sha256']
    if external_hashes.duplicated().any():
        duplicates = int(external_hashes.duplicated(keep=False).sum())
        raise ValueError(f'外部测试集存在{duplicates}张完全重复图片')
    training = pd.read_csv(
        training_manifest_path,
        encoding='utf-8-sig',
        usecols=['sha256'],
    )
    overlap = set(external_hashes) & set(training['sha256'].astype(str))
    if overlap:
        raise ValueError(f'外部测试集与训练原图存在{len(overlap)}个SHA重叠')
    return {
        'external_unique_sha256': int(external_hashes.nunique()),
        'training_unique_sha256': int(training['sha256'].nunique()),
        'exact_sha256_overlap': 0,
    }


def load_legacy_cancer_crops(manifest_path):
    if not manifest_path.is_file():
        return {}
    legacy = pd.read_csv(
        manifest_path,
        encoding='utf-8-sig',
        dtype={'patient_id': str},
    )
    required = {
        'patient_id', 'image_name', 'processed_path',
        'original_width', 'original_height',
    }
    if not required.issubset(legacy.columns):
        return {}
    return {
        (str(row.patient_id), str(row.image_name)): row
        for row in legacy.itertuples(index=False)
        if Path(row.processed_path).is_file()
    }


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def preprocess_images(manifest, output_dir, overwrite, legacy_cancer_manifest):
    processed_dir = output_dir / '裁剪后图像_v1_1'
    manifest_path = output_dir / 'preprocess_manifest.csv'
    if manifest_path.exists() and not overwrite:
        cached = pd.read_csv(
            manifest_path,
            encoding='utf-8-sig',
            dtype={'patient_id': str},
        )
        same_sources = (
            len(cached) == len(manifest)
            and cached['original_path'].tolist() == manifest['original_path'].tolist()
        )
        outputs_exist = cached['processed_path'].map(
            lambda path: Path(path).exists()
        ).all()
        if same_sources and outputs_exist:
            print(f'复用已裁剪外部数据: {len(cached)}张')
            return cached

    legacy_crops = load_legacy_cancer_crops(legacy_cancer_manifest)
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        original = read_image(row.original_path)
        original_h, original_w = original.shape[:2]
        destination = (
            processed_dir / row.class_name / row.patient_id
            / f'{row.image_index:04d}_{Path(row.image_name).stem}.jpg'
        )
        legacy = legacy_crops.get((str(row.patient_id), str(row.image_name)))
        if (
            row.label == 1
            and legacy is not None
            and int(legacy.original_width) == original_w
            and int(legacy.original_height) == original_h
        ):
            link_or_copy(Path(legacy.processed_path), destination)
            legacy_values = legacy._asdict()
            rows.append({
                **row._asdict(),
                'original_sha256': file_sha256(row.original_path),
                'processed_path': str(destination.resolve()),
                'original_width': original_w,
                'original_height': original_h,
                **{
                    column: legacy_values.get(column)
                    for column in [
                        'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
                        'crop_status', 'crop_method', 'failure_reason',
                        'edge_refined', 'progress_bar_detected',
                        'edge_trim_left', 'edge_trim_top',
                        'edge_trim_right', 'edge_trim_bottom',
                        'progress_jump', 'progress_tail_std',
                    ]
                },
                'legacy_crop_reused': True,
            })
            if number % 100 == 0 or number == len(manifest):
                print(f'外部数据裁剪/复用: {number}/{len(manifest)}')
            continue

        roi = detect_roi(original)
        bbox, refine = refine_roi_by_edge_scan(original, roi['bbox'])
        x1, y1, x2, y2 = bbox
        processed = resize_square(original[y1:y2, x1:x2], OUTPUT_SIZE)
        write_jpeg(str(destination), processed)
        rows.append({
            **row._asdict(),
            'original_sha256': file_sha256(row.original_path),
            'processed_path': str(destination.resolve()),
            'original_width': original_w,
            'original_height': original_h,
            'crop_x1': x1,
            'crop_y1': y1,
            'crop_x2': x2,
            'crop_y2': y2,
            'crop_status': 'cropped' if refine['edge_refined'] else roi['crop_status'],
            'crop_method': (
                roi['crop_method'] + '+edge_scan_v1_1'
                if refine['edge_refined'] else roi['crop_method']
            ),
            'failure_reason': roi['failure_reason'],
            **refine,
            'legacy_crop_reused': False,
        })
        if number % 100 == 0 or number == len(manifest):
            print(f'外部数据裁剪/复用: {number}/{len(manifest)}')

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    save_crop_review(result, output_dir / '裁剪复核')
    return result


def save_crop_review(manifest, output_dir):
    dimensions = np.minimum(manifest['original_width'], manifest['original_height'])
    trim_ratio = (
        manifest['edge_trim_left'] + manifest['edge_trim_right']
        + manifest['edge_trim_top'] + manifest['edge_trim_bottom']
    ) / dimensions
    risk_indices = trim_ratio.nlargest(min(32, len(manifest))).index.tolist()
    sample_indices = []
    for _, group in manifest.groupby('label', sort=True):
        sample_indices.extend(
            group.sample(n=min(16, len(group)), random_state=42).index.tolist()
        )
    selected = manifest.loc[list(dict.fromkeys(risk_indices + sample_indices))]
    previews = []
    for row in selected.itertuples(index=False):
        original = read_image(row.original_path)
        processed = read_image(row.processed_path)
        bbox = (row.crop_x1, row.crop_y1, row.crop_x2, row.crop_y2)
        title = (
            f'{row.class_name} {row.patient_id} '
            f'edge={int(row.edge_refined)} bar={int(row.progress_bar_detected)}'
        )
        previews.append(draw_preview(original, processed, bbox, title))
    save_contact_sheets(previews, str(output_dir))


def collect_probabilities(model, manifest, batch_size, num_workers):
    _, eval_transform = build_transforms()
    dataset = ExternalProcessedDataset(manifest, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=num_workers > 0,
    )
    probabilities = np.zeros(len(dataset), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for images, indices in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            probabilities[indices.numpy()] = (
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
    return probabilities


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)
    ) / denominator
    return center - margin, center + margin


def bootstrap_auc_ci(labels, probabilities, iterations, seed):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    aucs = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        aucs[index] = roc_auc_score(labels[sampled], probabilities[sampled])
    return np.quantile(aucs, [0.025, 0.975]).tolist()


def binary_metrics(labels, probabilities, threshold, bootstrap, seed):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    sens_ci = wilson_interval(tp, tp + fn)
    spec_ci = wilson_interval(tn, tn + fp)
    result = {
        'threshold': float(threshold),
        'sample_count': int(len(labels)),
        'positive_count': int(labels.sum()),
        'negative_count': int((labels == 0).sum()),
        'auc': float(roc_auc_score(labels, probabilities)),
        'accuracy': float((predictions == labels).mean()),
        'balanced_accuracy': float((sensitivity + specificity) / 2),
        'sensitivity': float(sensitivity),
        'sensitivity_ci_low': float(sens_ci[0]),
        'sensitivity_ci_high': float(sens_ci[1]),
        'specificity': float(specificity),
        'specificity_ci_low': float(spec_ci[0]),
        'specificity_ci_high': float(spec_ci[1]),
        'precision': float(precision),
        'npv': float(npv),
        'f1': float(
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity else 0
        ),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    if bootstrap:
        result['auc_ci95'] = bootstrap_auc_ci(
            labels, probabilities, bootstrap, seed
        )
    return result


def aggregate_patients(image_results, probability_column, method):
    rows = []
    for patient_id, group in image_results.groupby('patient_id', sort=True):
        labels = group['label'].unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        values = group[probability_column].to_numpy(dtype=float)
        if method == 'mean':
            probability = values.mean()
        elif method == 'top2_mean':
            probability = np.sort(values)[-min(2, len(values)):].mean()
        else:
            raise ValueError(method)
        rows.append({
            'patient_id': patient_id,
            'label': int(labels[0]),
            'class_name': CLASS_NAMES[int(labels[0])],
            'image_count': int(len(group)),
            'cancer_probability': float(probability),
        })
    return pd.DataFrame(rows)


def load_thresholds(run_dir):
    with open(run_dir / 'locked_threshold_summary.json', encoding='utf-8') as file:
        summary = json.load(file)
    return {
        level: float(summary['levels'][level]['threshold'])
        for level in ('image', 'patient')
    }


def load_trained_model(model_name, run_dir):
    builders = {
        'resnet50': (build_resnet, 'resnet50_debiased_best.pth'),
        'efficientnet_b0': (build_efficientnet, 'efficientnet_b0_debiased_best.pth'),
    }
    builder, weight_name = builders[model_name]
    checkpoint = torch.load(
        run_dir / weight_name,
        map_location=DEVICE,
        weights_only=False,
    )
    model = builder(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(DEVICE)


def evaluate_model(model_name, run_dir, image_results, args, output_dir):
    thresholds = load_thresholds(run_dir)
    model = load_trained_model(model_name, run_dir)
    probability_column = f'{model_name}_probability'
    image_results[probability_column] = collect_probabilities(
        model, image_results, args.batch_size, args.num_workers
    )
    patient_results = aggregate_patients(
        image_results, probability_column, method='mean'
    )
    image_metrics = binary_metrics(
        image_results['label'],
        image_results[probability_column],
        thresholds['image'],
        bootstrap=0,
        seed=args.seed,
    )
    patient_metrics = binary_metrics(
        patient_results['label'],
        patient_results['cancer_probability'],
        thresholds['patient'],
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    patient_results.to_csv(
        model_dir / 'patient_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    patient_results[
        (patient_results['label'] == 1)
        & (patient_results['cancer_probability'] < thresholds['patient'])
    ].sort_values('cancer_probability').to_csv(
        model_dir / 'false_negative_patients.csv',
        index=False,
        encoding='utf-8-sig',
    )
    patient_results[
        (patient_results['label'] == 0)
        & (patient_results['cancer_probability'] >= thresholds['patient'])
    ].sort_values('cancer_probability', ascending=False).to_csv(
        model_dir / 'false_positive_patients.csv',
        index=False,
        encoding='utf-8-sig',
    )
    print(
        f'{model_name}: patient AUC={patient_metrics["auc"]:.3f}, '
        f'Sens={patient_metrics["sensitivity"]:.3f}, '
        f'Spec={patient_metrics["specificity"]:.3f}'
    )
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'model': model_name,
        'patient_aggregation': 'mean',
        'image': image_metrics,
        'patient': patient_metrics,
    }


def evaluate_locked_ensemble(image_results, summary_path, args, output_dir):
    with open(summary_path, encoding='utf-8') as file:
        locked = json.load(file)
    if locked['selected_view'] != 'base':
        raise ValueError('当前脚本只执行已锁定的无TTA集成')
    resnet_weight = float(locked['selected_resnet_weight'])
    efficientnet_weight = float(locked['selected_efficientnet_weight'])
    aggregation = locked['selected_aggregation']
    threshold = float(locked['selected_view_results']['threshold'])
    image_results['ensemble_probability'] = (
        resnet_weight * image_results['resnet50_probability']
        + efficientnet_weight * image_results['efficientnet_b0_probability']
    )
    patients = aggregate_patients(
        image_results, 'ensemble_probability', method=aggregation
    )
    metrics = binary_metrics(
        patients['label'],
        patients['cancer_probability'],
        threshold,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    ensemble_dir = output_dir / 'ensemble'
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(
        ensemble_dir / 'patient_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    patients[
        (patients['label'] == 1)
        & (patients['cancer_probability'] < threshold)
    ].sort_values('cancer_probability').to_csv(
        ensemble_dir / 'false_negative_patients.csv',
        index=False,
        encoding='utf-8-sig',
    )
    patients[
        (patients['label'] == 0)
        & (patients['cancer_probability'] >= threshold)
    ].sort_values('cancer_probability', ascending=False).to_csv(
        ensemble_dir / 'false_positive_patients.csv',
        index=False,
        encoding='utf-8-sig',
    )
    print(
        f'ensemble: patient AUC={metrics["auc"]:.3f}, '
        f'Sens={metrics["sensitivity"]:.3f}, '
        f'Spec={metrics["specificity"]:.3f}'
    )
    return {
        'model': 'ensemble',
        'resnet_weight': resnet_weight,
        'efficientnet_weight': efficientnet_weight,
        'patient_aggregation': aggregation,
        'tta': False,
        'patient': metrics,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir
    if args.debug:
        output_dir = output_dir.with_name(output_dir.name + '_debug')

    manifest = scan_external_directory(args.data_dir)
    if args.debug:
        manifest = debug_subset(
            manifest, args.debug_patients_per_class, args.seed
        )
    patient_counts = (
        manifest[['patient_id', 'label']].drop_duplicates()['label']
        .value_counts().sort_index().to_dict()
    )
    image_counts = manifest['label'].value_counts().sort_index().to_dict()
    print(
        f'设备: {DEVICE}; 外部数据: {manifest.patient_id.nunique()}人/'
        f'{len(manifest)}张; 患者={patient_counts}; 图片={image_counts}'
    )

    processed = preprocess_images(
        manifest,
        output_dir,
        args.overwrite_preprocessed,
        args.legacy_cancer_manifest,
    )
    leakage_audit = validate_against_training(
        processed, args.training_manifest
    )
    preprocess_audit = {
        'debug': args.debug,
        'data_dir': str(args.data_dir.resolve()),
        'output_dir': str(output_dir.resolve()),
        'preprocess': 'preprocess_second_batch_v1_1',
        'patient_counts': {CLASS_NAMES[k]: int(v) for k, v in patient_counts.items()},
        'image_counts': {CLASS_NAMES[k]: int(v) for k, v in image_counts.items()},
        'crop_status_counts': {
            str(key): int(value)
            for key, value in processed['crop_status'].value_counts().items()
        },
        'edge_refined_count': int(processed['edge_refined'].sum()),
        'progress_bar_detected_count': int(
            processed['progress_bar_detected'].sum()
        ),
        'legacy_crop_reused_count': int(
            processed.get(
                'legacy_crop_reused',
                pd.Series(False, index=processed.index),
            ).sum()
        ),
        'leakage_audit': leakage_audit,
    }
    (output_dir / 'preprocess_audit.json').write_text(
        json.dumps(preprocess_audit, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    if args.preprocess_only:
        print(f'预处理与审计已保存至: {output_dir.resolve()}')
        return

    run_dirs = {
        'resnet50': args.resnet_run_dir,
        'efficientnet_b0': args.efficientnet_run_dir,
    }
    model_names = list(run_dirs) if args.model == 'all' else [args.model]
    model_results = []
    image_results = processed.copy()
    for model_name in model_names:
        model_results.append(evaluate_model(
            model_name,
            run_dirs[model_name],
            image_results,
            args,
            output_dir,
        ))

    ensemble_result = None
    if args.model == 'all':
        ensemble_result = evaluate_locked_ensemble(
            image_results, args.ensemble_summary, args, output_dir
        )
    image_results.to_csv(
        output_dir / 'image_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )

    summary = {
        'protocol': (
            '独立外部测试集不参与训练、模型选择、患者聚合、集成权重、'
            'TTA或阈值选择；所有参数复用实验A验证集锁定结果。'
        ),
        'debug': args.debug,
        'device': str(DEVICE),
        'data_dir': str(args.data_dir.resolve()),
        'output_dir': str(output_dir.resolve()),
        'preprocess': 'preprocess_second_batch_v1_1',
        'patient_counts': {CLASS_NAMES[k]: int(v) for k, v in patient_counts.items()},
        'image_counts': {CLASS_NAMES[k]: int(v) for k, v in image_counts.items()},
        'leakage_audit': leakage_audit,
        'model_run_dirs': {
            name: str(run_dirs[name].resolve()) for name in model_names
        },
        'model_results': model_results,
        'ensemble_result': ensemble_result,
        'bootstrap_iterations': args.bootstrap,
        'seed': args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'evaluation_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'结果已保存至: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
