#!/usr/bin/env python3
"""Evaluate legacy weights on the original held-out patients after preprocessing."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset

from inference import DEVICE, load_model, transform


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_DIR / '结果' / '去偏重训练_v1' / '旧模型新预处理验证_v1'
MODEL_CONFIG = {
    'resnet50': ('resnet50_transfer_best.pth', 0.30),
    'efficientnet_b0': ('efficientnet_b0_best.pth', 0.22),
}


def parse_args():
    parser = argparse.ArgumentParser(description='旧模型在v1.1裁剪数据上的无重训练评估')
    parser.add_argument('--old-split', type=Path, default=PROJECT_DIR / '结果/数据划分/image_split_seed42.csv')
    parser.add_argument('--processed-manifest', type=Path, default=PROJECT_DIR / '数据整理记录/第二批/预处理_v1_1/full/processed_manifest.csv')
    parser.add_argument('--selected-manifest', type=Path, default=PROJECT_DIR / '数据整理记录/第二批/筛图_v1_1/full/patient_limited_manifest.csv')
    parser.add_argument('--strict-manifest', type=Path, default=PROJECT_DIR / '数据整理记录/第二批/匹配_v1/matched_weak_strict_common_support_v1_seed42.csv')
    parser.add_argument('--original-root', type=Path, default=PROJECT_DIR / '数据/第二批整理后')
    parser.add_argument('--processed-root', type=Path, default=PROJECT_DIR / '数据/第二批裁剪后_v1_1')
    parser.add_argument('--weight-root', type=Path, default=PROJECT_DIR / '结果/模型权重')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image = Image.open(self.frame.iloc[index]['absolute_path']).convert('RGB')
        return transform(image), index


@torch.no_grad()
def predict(model, frame, batch_size, num_workers):
    dataset = ImageDataset(frame)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=num_workers > 0,
    )
    probabilities = np.empty(len(dataset), dtype=np.float32)
    for images, indices in loader:
        logits = model(images.to(DEVICE, non_blocking=True))
        probabilities[indices.numpy()] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probabilities


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    auc = roc_auc_score(labels, probabilities) if np.unique(labels).size == 2 else np.nan
    return {
        'auc': float(auc),
        'accuracy': float(accuracy_score(labels, predictions)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else np.nan,
        'specificity': float(tn / (tn + fp)) if tn + fp else np.nan,
        'precision': float(tp / (tp + fp)) if tp + fp else np.nan,
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def auc_ci(frame, iterations, seed):
    labels = frame['label'].to_numpy(dtype=int)
    if np.unique(labels).size < 2:
        return np.nan, np.nan
    probabilities = frame['cancer_probability'].to_numpy(dtype=float)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        values.append(roc_auc_score(labels[indices], probabilities[indices]))
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def paired_auc_comparison(predictions, model_name, baseline, candidate, iterations, seed):
    columns = ['patient_id', 'label', 'cancer_probability']
    left = predictions[
        predictions['model'].eq(model_name) & predictions['scenario'].eq(baseline)
    ][columns].rename(columns={'label': 'label_baseline', 'cancer_probability': 'probability_baseline'})
    right = predictions[
        predictions['model'].eq(model_name) & predictions['scenario'].eq(candidate)
    ][columns].rename(columns={'label': 'label_candidate', 'cancer_probability': 'probability_candidate'})
    paired = left.merge(right, on='patient_id', validate='one_to_one')
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f'{model_name}: 配对场景患者集合不一致')
    if not paired['label_baseline'].equals(paired['label_candidate']):
        raise ValueError(f'{model_name}: 配对场景患者标签不一致')

    labels = paired['label_baseline'].to_numpy(dtype=int)
    baseline_probability = paired['probability_baseline'].to_numpy(dtype=float)
    candidate_probability = paired['probability_candidate'].to_numpy(dtype=float)
    baseline_auc = roc_auc_score(labels, baseline_probability)
    candidate_auc = roc_auc_score(labels, candidate_probability)
    observed = candidate_auc - baseline_auc
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indices = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        differences.append(
            roc_auc_score(labels[indices], candidate_probability[indices])
            - roc_auc_score(labels[indices], baseline_probability[indices])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    p_value = min(
        1.0,
        2 * min(np.mean(np.asarray(differences) <= 0), np.mean(np.asarray(differences) >= 0)),
    )
    return {
        'model': model_name,
        'baseline': baseline,
        'candidate': candidate,
        'patients': int(len(paired)),
        'baseline_auc': float(baseline_auc),
        'candidate_auc': float(candidate_auc),
        'auc_difference_candidate_minus_baseline': float(observed),
        'difference_ci95_low': float(low),
        'difference_ci95_high': float(high),
        'paired_bootstrap_p_value_two_sided': float(p_value),
    }


def patient_predictions(images):
    metadata = ['center', 'year', 'style_group', 'size_group', 'frame_profile', 'old_split']
    rows = []
    for patient_id, group in images.groupby('patient_id', sort=True):
        labels = group['label'].unique()
        if len(labels) != 1:
            raise ValueError(f'患者标签不唯一: {patient_id}')
        row = {
            'patient_id': patient_id,
            'label': int(labels[0]),
            'cancer_probability': float(group['cancer_probability'].mean()),
            'image_count': int(len(group)),
        }
        for column in metadata:
            if column in group:
                values = group[column].dropna().astype(str)
                row[column] = sorted(values.mode())[0] if len(values) else ''
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_frames(args):
    old = pd.read_csv(args.old_split, encoding='utf-8-sig').rename(
        columns={'瘤变标签': 'label', '图片名字': 'image_path'}
    )
    processed = pd.read_csv(args.processed_manifest, encoding='utf-8-sig')
    selected = pd.read_csv(args.selected_manifest, encoding='utf-8-sig')
    strict = pd.read_csv(args.strict_manifest, encoding='utf-8-sig')

    if len(old) != 5229 or old['sha256'].duplicated().any():
        raise ValueError('旧划分不是预期的5229张唯一图片')
    if set(old['sha256']) != set(processed['sha256']):
        raise ValueError('旧划分与v1.1裁剪清单图片集合不一致')
    old_exposure = old[['patient_id', 'split']].drop_duplicates()
    if old_exposure['patient_id'].duplicated().any():
        raise ValueError('旧划分中患者跨split')

    joined = old[['patient_id', 'label', 'sha256', 'split', 'image_path']].merge(
        processed,
        on=['patient_id', 'label', 'sha256'],
        how='left',
        validate='one_to_one',
        suffixes=('_old', ''),
    )
    if joined['processed_path'].isna().any():
        raise ValueError('旧划分存在无法连接的裁剪图片')
    joined = joined.rename(columns={'split': 'old_split'})

    original_test = joined[joined['old_split'].eq('test')].copy()
    original_test['absolute_path'] = original_test['image_path_old'].map(
        lambda value: str((args.original_root / value).resolve())
    )
    cropped_test = joined[joined['old_split'].eq('test')].copy()
    cropped_test['absolute_path'] = cropped_test['processed_path'].map(
        lambda value: str((args.processed_root / value).resolve())
    )

    selected = selected.merge(old[['sha256', 'split']], on='sha256', how='left', validate='one_to_one')
    selected = selected.rename(columns={'split': 'old_split'})
    selected_test = selected[selected['old_split'].eq('test')].copy()
    selected_test['absolute_path'] = selected_test['processed_path'].map(
        lambda value: str((args.processed_root / value).resolve())
    )

    strict = strict.merge(old[['sha256', 'split']], on='sha256', how='left', validate='one_to_one')
    strict = strict.rename(columns={'split': 'old_split'})
    strict['absolute_path'] = strict['processed_path'].map(
        lambda value: str((args.processed_root / value).resolve())
    )

    scenarios = {
        'old_test_original_full': (original_test, 'heldout_original'),
        'old_test_cropped_full': (cropped_test, 'heldout_preprocessed'),
        'old_test_cropped_patient_limited': (selected_test, 'heldout_preprocessed_selected'),
        'strict_all_mixed_exposure': (strict, 'contaminated_mixed'),
    }
    for split in ['train', 'val', 'test']:
        scenarios[f'strict_old_{split}'] = (
            strict[strict['old_split'].eq(split)].copy(),
            {'train': 'seen_training', 'val': 'seen_model_selection', 'test': 'heldout'}[split],
        )

    for name, (frame, _) in scenarios.items():
        if frame.empty:
            raise ValueError(f'{name} 为空')
        missing = [path for path in frame['absolute_path'] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f'{name} 缺少图片: {missing[0]}')
    return scenarios


def main():
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f'输出已存在: {args.output_dir}；确认后使用 --overwrite')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = prepare_frames(args)
    audit_rows = []
    for name, (frame, exposure) in scenarios.items():
        audit_rows.append({
            'scenario': name,
            'exposure': exposure,
            'patients': int(frame['patient_id'].nunique()),
            'images': int(len(frame)),
            'cancer_patients': int(frame.loc[frame['label'].eq(1), 'patient_id'].nunique()),
            'noncancer_patients': int(frame.loc[frame['label'].eq(0), 'patient_id'].nunique()),
            'cancer_images': int(frame['label'].eq(1).sum()),
            'noncancer_images': int(frame['label'].eq(0).sum()),
        })
    pd.DataFrame(audit_rows).to_csv(
        args.output_dir / 'exposure_audit.csv', index=False, encoding='utf-8-sig'
    )

    metrics_rows = []
    all_image_predictions = []
    all_patient_predictions = []
    for model_name, (weight_file, locked_threshold) in MODEL_CONFIG.items():
        print(f'\n加载旧模型: {model_name}')
        model = load_model(model_name, args.weight_root / weight_file)
        for scenario, (source, exposure) in scenarios.items():
            images = source.copy().reset_index(drop=True)
            images['cancer_probability'] = predict(
                model, images, args.batch_size, args.num_workers
            )
            patients = patient_predictions(images)
            for level, frame in [('image', images), ('patient', patients)]:
                ci_low, ci_high = (
                    auc_ci(frame, args.bootstrap, args.seed) if level == 'patient' else (np.nan, np.nan)
                )
                for threshold_name, threshold in [('fixed_0.5', 0.5), ('legacy_locked', locked_threshold)]:
                    values = compute_metrics(frame['label'], frame['cancer_probability'], threshold)
                    metrics_rows.append({
                        'model': model_name,
                        'scenario': scenario,
                        'exposure': exposure,
                        'level': level,
                        'threshold_name': threshold_name,
                        'threshold': threshold,
                        'patients': int(frame['patient_id'].nunique()),
                        'samples': int(len(frame)),
                        'auc_ci95_low': ci_low,
                        'auc_ci95_high': ci_high,
                        **values,
                    })
            images['model'] = model_name
            images['scenario'] = scenario
            images['exposure'] = exposure
            patients['model'] = model_name
            patients['scenario'] = scenario
            patients['exposure'] = exposure
            all_image_predictions.append(images[
                ['model', 'scenario', 'exposure', 'patient_id', 'label', 'sha256',
                 'absolute_path', 'cancer_probability', 'old_split']
            ])
            all_patient_predictions.append(patients)
            patient_auc = metrics_rows[-2]['auc']
            print(
                f'{scenario}: {len(patients)}人/{len(images)}张, '
                f'patient AUC={patient_auc:.4f}'
            )
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(all_image_predictions, ignore_index=True).to_csv(
        args.output_dir / 'image_predictions.csv', index=False, encoding='utf-8-sig'
    )
    patient_output = pd.concat(all_patient_predictions, ignore_index=True)
    patient_output.to_csv(
        args.output_dir / 'patient_predictions.csv', index=False, encoding='utf-8-sig'
    )
    comparisons = []
    for model_name in MODEL_CONFIG:
        comparisons.append(paired_auc_comparison(
            patient_output,
            model_name,
            'old_test_original_full',
            'old_test_cropped_full',
            args.bootstrap,
            args.seed,
        ))
        comparisons.append(paired_auc_comparison(
            patient_output,
            model_name,
            'old_test_cropped_full',
            'old_test_cropped_patient_limited',
            args.bootstrap,
            args.seed,
        ))
    pd.DataFrame(comparisons).to_csv(
        args.output_dir / 'paired_auc_comparisons.csv', index=False, encoding='utf-8-sig'
    )
    config = {
        'device': str(DEVICE),
        'old_split': str(args.old_split.resolve()),
        'old_split_sha256': sha256(args.old_split),
        'processed_manifest_sha256': sha256(args.processed_manifest),
        'selected_manifest_sha256': sha256(args.selected_manifest),
        'strict_manifest_sha256': sha256(args.strict_manifest),
        'legacy_thresholds': {name: value[1] for name, value in MODEL_CONFIG.items()},
        'note': 'Only scenarios marked heldout are valid unseen-patient evaluations.',
    }
    (args.output_dir / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\n结果目录: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
