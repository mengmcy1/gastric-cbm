#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='配对比较两个模型的患者级 OOF AUC')
    parser.add_argument('--model-a', type=Path, required=True)
    parser.add_argument('--model-b', type=Path, required=True)
    parser.add_argument('--name-a', required=True)
    parser.add_argument('--name-b', required=True)
    parser.add_argument('--filter-a', action='append', default=[], help='COLUMN=VALUE，可重复')
    parser.add_argument('--filter-b', action='append', default=[], help='COLUMN=VALUE，可重复')
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def read_filtered(path, filters):
    frame = pd.read_csv(path)
    for specification in filters:
        if '=' not in specification:
            raise ValueError(f'筛选条件必须为 COLUMN=VALUE: {specification}')
        column, value = specification.split('=', 1)
        if column not in frame:
            raise ValueError(f'{path} 缺少筛选字段: {column}')
        frame = frame[frame[column].astype(str).eq(value)]
    return frame


def main():
    args = parse_args()
    columns = ['patient_id', 'label', 'cancer_probability']
    model_a = read_filtered(args.model_a, args.filter_a)[columns].rename(
        columns={'label': 'label_a', 'cancer_probability': 'probability_a'}
    )
    model_b = read_filtered(args.model_b, args.filter_b)[columns].rename(
        columns={'label': 'label_b', 'cancer_probability': 'probability_b'}
    )
    if model_a.empty or model_b.empty:
        raise ValueError('筛选后没有患者预测')
    paired = model_a.merge(model_b, on='patient_id', how='inner', validate='one_to_one')
    if len(paired) != len(model_a) or len(paired) != len(model_b):
        raise ValueError('两个模型的 OOF 患者集合不一致')
    if not paired['label_a'].equals(paired['label_b']):
        raise ValueError('两个模型的患者标签不一致')

    labels = paired['label_a'].to_numpy(dtype=int)
    probability_a = paired['probability_a'].to_numpy(dtype=float)
    probability_b = paired['probability_b'].to_numpy(dtype=float)
    auc_a = roc_auc_score(labels, probability_a)
    auc_b = roc_auc_score(labels, probability_b)
    observed = auc_a - auc_b

    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.bootstrap):
        indices = np.concatenate((
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ))
        differences.append(
            roc_auc_score(labels[indices], probability_a[indices])
            - roc_auc_score(labels[indices], probability_b[indices])
        )
    differences = np.asarray(differences)
    low, high = np.percentile(differences, [2.5, 97.5])
    p_value = min(
        1.0,
        2 * min(np.mean(differences <= 0), np.mean(differences >= 0)),
    )

    result = {
        'patient_count': int(len(paired)),
        'model_a': args.name_a,
        'model_b': args.name_b,
        'auc_a': float(auc_a),
        'auc_b': float(auc_b),
        'auc_difference_a_minus_b': float(observed),
        'paired_bootstrap_ci95': [float(low), float(high)],
        'paired_bootstrap_p_value_two_sided': float(p_value),
        'bootstrap_iterations': args.bootstrap,
        'seed': args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f'{args.name_a} - {args.name_b}: {observed:.4f} '
        f'(95% CI {low:.4f}-{high:.4f}, paired bootstrap p={p_value:.4f})'
    )
    print(f'结果: {args.output.resolve()}')


if __name__ == '__main__':
    main()
