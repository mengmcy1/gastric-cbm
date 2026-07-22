#!/usr/bin/env python3
"""Select thresholds on validation predictions and lock them on a holdout test set."""

import argparse
import json
from pathlib import Path

import pandas as pd

from summarize_cv_results import bootstrap_auc_ci, choose_threshold, metrics


def parse_args():
    parser = argparse.ArgumentParser(description='固定划分验证集阈值与测试集锁定汇总')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--target-sensitivity', type=float, default=0.90)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv = args.run_dir / 'locked_threshold_summary.csv'
    output_json = args.run_dir / 'locked_threshold_summary.json'
    if (output_csv.exists() or output_json.exists()) and not args.overwrite:
        raise FileExistsError(f'锁定阈值结果已存在: {args.run_dir}')

    rows = []
    summary = {
        'run_dir': str(args.run_dir.resolve()),
        'target_validation_sensitivity': args.target_sensitivity,
        'levels': {},
    }
    for level in ['image', 'patient']:
        val = pd.read_csv(args.run_dir / f'val_{level}_predictions.csv')
        test = pd.read_csv(args.run_dir / f'test_{level}_predictions.csv')
        threshold = choose_threshold(
            val['label'], val['cancer_probability'], args.target_sensitivity
        )
        val_locked = metrics(val['label'], val['cancer_probability'], threshold)
        test_locked = metrics(test['label'], test['cancer_probability'], threshold)
        test_fixed = metrics(test['label'], test['cancer_probability'], 0.5)
        ci = (
            bootstrap_auc_ci(test, args.bootstrap, args.seed)
            if level == 'patient' else (None, None)
        )
        summary['levels'][level] = {
            'threshold': threshold,
            'test_auc_ci95': ci,
            'validation_locked': val_locked,
            'test_locked': test_locked,
            'test_fixed_0_5': test_fixed,
        }
        for split, threshold_name, used_threshold, values in [
            ('val', 'validation_selected', threshold, val_locked),
            ('test', 'validation_locked', threshold, test_locked),
            ('test', 'fixed_0.5', 0.5, test_fixed),
        ]:
            rows.append({
                'level': level,
                'split': split,
                'threshold_name': threshold_name,
                'threshold': used_threshold,
                'samples': len(val) if split == 'val' else len(test),
                'auc_ci95_low': ci[0] if split == 'test' and level == 'patient' else None,
                'auc_ci95_high': ci[1] if split == 'test' and level == 'patient' else None,
                **values,
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding='utf-8-sig')
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'结果: {output_csv.resolve()}')
    patient = summary['levels']['patient']
    locked = patient['test_locked']
    print(
        f"Patient threshold={patient['threshold']:.4f} | "
        f"AUC={locked['auc']:.4f} | Sens={locked['sensitivity']:.4f} | "
        f"Spec={locked['specificity']:.4f} | Acc={locked['accuracy']:.4f}"
    )


if __name__ == '__main__':
    main()
