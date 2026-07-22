#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='汇总患者级交叉验证 OOF 结果')
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--run-template', required=True, help='含 {fold} 的运行目录名')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--target-sensitivity', type=float, default=0.90)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size == 2 else np.nan
    return {
        'auc': auc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def choose_threshold(y_true, y_prob, target_sensitivity):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], y_prob, [1.0])))
    eligible = []
    for threshold in candidates:
        values = metrics(y_true, y_prob, float(threshold))
        if values['sensitivity'] + 1e-12 >= target_sensitivity:
            eligible.append((values['specificity'], float(threshold), values))
    if not eligible:
        raise RuntimeError('验证集没有满足目标敏感度的阈值')
    return max(eligible, key=lambda item: (item[0], item[1]))[1]


def bootstrap_auc_ci(frame, iterations, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    labels = frame['label'].to_numpy(dtype=int)
    probabilities = frame['cancer_probability'].to_numpy(dtype=float)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    for _ in range(iterations):
        indices = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        aucs.append(roc_auc_score(labels[indices], probabilities[indices]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def main():
    args = parse_args()
    if '{fold}' not in args.run_template:
        raise ValueError('--run-template 必须包含 {fold}')

    fold_rows = []
    patient_frames = []
    image_frames = []
    for fold in range(args.folds):
        run_dir = args.input_root / args.run_template.format(fold=fold)
        config_path = run_dir / 'config.json'
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if int(config['fold']) != fold:
            raise ValueError(f'{run_dir}: config fold 不一致')

        val = pd.read_csv(run_dir / 'val_patient_predictions.csv')
        test = pd.read_csv(run_dir / 'test_patient_predictions.csv')
        test_images = pd.read_csv(run_dir / 'test_image_predictions.csv')
        threshold = choose_threshold(
            val['label'], val['cancer_probability'], args.target_sensitivity
        )
        val_metrics = metrics(val['label'], val['cancer_probability'], threshold)
        test_metrics = metrics(test['label'], test['cancer_probability'], threshold)
        fixed_metrics = metrics(test['label'], test['cancer_probability'], 0.5)

        row = {
            'fold': fold,
            'selected_stage': config['selected_stage'],
            'threshold': threshold,
            'val_auc': val_metrics['auc'],
            'val_sensitivity': val_metrics['sensitivity'],
            'val_specificity': val_metrics['specificity'],
        }
        row.update({f'test_{key}': value for key, value in test_metrics.items()})
        row.update({f'test_fixed05_{key}': value for key, value in fixed_metrics.items()})
        fold_rows.append(row)

        test = test.copy()
        test['fold'] = fold
        test['locked_threshold'] = threshold
        test['locked_prediction'] = (test['cancer_probability'] >= threshold).astype(int)
        patient_frames.append(test)

        test_images = test_images.copy()
        test_images['fold'] = fold
        test_images['locked_threshold'] = threshold
        test_images['locked_prediction'] = (
            test_images['cancer_probability'] >= threshold
        ).astype(int)
        image_frames.append(test_images)

    fold_summary = pd.DataFrame(fold_rows)
    patient_oof = pd.concat(patient_frames, ignore_index=True)
    image_oof = pd.concat(image_frames, ignore_index=True)
    if patient_oof['patient_id'].duplicated().any():
        raise ValueError('OOF 患者重复，交叉验证测试折存在泄漏')

    patient_auc = roc_auc_score(patient_oof['label'], patient_oof['cancer_probability'])
    patient_ci = bootstrap_auc_ci(patient_oof, args.bootstrap, args.seed)
    locked = metrics(
        patient_oof['label'],
        patient_oof['locked_prediction'],
        threshold=0.5,
    )
    locked['auc'] = float(patient_auc)
    fixed = metrics(patient_oof['label'], patient_oof['cancer_probability'], 0.5)
    image_auc = roc_auc_score(image_oof['label'], image_oof['cancer_probability'])

    summary = {
        'folds': args.folds,
        'patient_count': int(len(patient_oof)),
        'image_count': int(len(image_oof)),
        'target_validation_sensitivity': args.target_sensitivity,
        'patient_oof_auc': float(patient_auc),
        'patient_oof_auc_ci95': [patient_ci[0], patient_ci[1]],
        'patient_locked_threshold_metrics': locked,
        'patient_fixed_0_5_metrics': fixed,
        'image_oof_auc': float(image_auc),
        'fold_test_auc_mean': float(fold_summary['test_auc'].mean()),
        'fold_test_auc_std': float(fold_summary['test_auc'].std(ddof=1)),
        'note': '每折阈值仅由该折验证集选择；OOF AUC不依赖分类阈值。',
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fold_summary.to_csv(args.output_dir / 'fold_summary.csv', index=False, encoding='utf-8-sig')
    patient_oof.to_csv(args.output_dir / 'patient_oof_predictions.csv', index=False, encoding='utf-8-sig')
    image_oof.to_csv(args.output_dir / 'image_oof_predictions.csv', index=False, encoding='utf-8-sig')
    (args.output_dir / 'cv_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print(f"患者 OOF AUC: {patient_auc:.4f} (95% CI {patient_ci[0]:.4f}-{patient_ci[1]:.4f})")
    print(
        '验证集锁定阈值汇总: '
        f"Sens={locked['sensitivity']:.4f}, Spec={locked['specificity']:.4f}, "
        f"Acc={locked['accuracy']:.4f}, F1={locked['f1']:.4f}"
    )
    print(f'图片 OOF AUC: {image_auc:.4f}')
    print(f'结果目录: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
