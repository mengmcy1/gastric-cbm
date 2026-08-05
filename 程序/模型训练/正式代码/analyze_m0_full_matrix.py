"""Aggregate the frozen M0 validation matrix without touching test outputs."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


RUN_PATTERN = re.compile(
    r'^m0_full_(keep|notch)_(resnet50|efficientnet_b0)_seed(42|202|503)$'
)
PREPROCESSING = ('keep', 'notch')
MODELS = ('resnet50', 'efficientnet_b0')
SEEDS = (42, 202, 503)
METRICS = ('AUC', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'F1')


def parse_args():
    project_root = Path(__file__).resolve().parents[3]
    default_root = project_root / '结果' / 'M0全量诊断_0804' / '正式验证集筛选'
    parser = argparse.ArgumentParser(description='M0 full validation matrix analysis')
    parser.add_argument('--result-root', type=Path, default=default_root)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260805)
    return parser.parse_args()


def run_key(preprocessing, model, seed):
    return f'm0_full_{preprocessing}_{model}_seed{seed}'


def load_runs(result_root):
    runs = {}
    missing = []
    for preprocessing in PREPROCESSING:
        for model in MODELS:
            for seed in SEEDS:
                name = run_key(preprocessing, model, seed)
                run_dir = result_root / name
                required = [
                    'config.json',
                    'training_history.csv',
                    'val_image_predictions.csv',
                    'val_patient_predictions.csv',
                    'val_patient_threshold_scan.csv',
                    'val_subgroup_metrics.csv',
                    'source_train_utils.py',
                    'source_entry.py',
                ]
                absent = [item for item in required if not (run_dir / item).is_file()]
                if absent:
                    missing.append({'run': name, 'missing': absent})
                    continue
                if list(run_dir.glob('test_*')):
                    raise RuntimeError(f'{name} contains forbidden test outputs')
                config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
                runs[(preprocessing, model, seed)] = {
                    'name': name,
                    'dir': run_dir,
                    'config': config,
                    'history': pd.read_csv(run_dir / 'training_history.csv'),
                    'image': pd.read_csv(run_dir / 'val_image_predictions.csv'),
                    'patient': pd.read_csv(run_dir / 'val_patient_predictions.csv'),
                    'subgroup': pd.read_csv(run_dir / 'val_subgroup_metrics.csv'),
                }
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))
    return runs


def run_metrics_table(runs):
    rows = []
    for (preprocessing, model, seed), run in sorted(runs.items()):
        config = run['config']
        history = run['history']
        best = history.loc[history['val_patient_AUC'].idxmax()]
        row = {
            'preprocessing': preprocessing,
            'model': model,
            'seed': seed,
            'run_name': run['name'],
            'selected_stage': config['selected_stage'],
            'best_epoch': int(best['epoch']),
            'threshold': config['threshold_selection']['threshold'],
            'history_rows': len(history),
        }
        for level in ('patient', 'image'):
            values = config['metrics'][f'val_{level}']
            for metric in METRICS:
                row[f'{level}_{metric}'] = values[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def cross_seed_summary(run_metrics):
    rows = []
    value_columns = [
        f'{level}_{metric}'
        for level in ('patient', 'image')
        for metric in METRICS
    ]
    for (preprocessing, model), group in run_metrics.groupby(
        ['preprocessing', 'model'], sort=True
    ):
        row = {'preprocessing': preprocessing, 'model': model, 'seeds': len(group)}
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            row[f'{column}_mean'] = float(values.mean())
            row[f'{column}_std'] = float(values.std(ddof=1))
            row[f'{column}_worst'] = float(values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(run_metrics):
    rows = []
    indexed = run_metrics.set_index(['preprocessing', 'model', 'seed'])
    for model in MODELS:
        for seed in SEEDS:
            keep = indexed.loc[('keep', model, seed)]
            notch = indexed.loc[('notch', model, seed)]
            row = {'comparison': 'notch_minus_keep', 'model': model, 'seed': seed}
            for level in ('patient', 'image'):
                for metric in METRICS:
                    column = f'{level}_{metric}'
                    row[f'delta_{column}'] = float(notch[column] - keep[column])
            row['passes_patient_auc'] = row['delta_patient_AUC'] >= -0.005
            row['passes_patient_sensitivity'] = (
                row['delta_patient_Sensitivity'] >= -0.02
            )
            row['passes_patient_specificity'] = (
                row['delta_patient_Specificity'] >= -0.02
            )
            row['passes_all_patient_gates'] = all([
                row['passes_patient_auc'],
                row['passes_patient_sensitivity'],
                row['passes_patient_specificity'],
            ])
            rows.append(row)
    return pd.DataFrame(rows)


def align_predictions(left, right, level):
    keys = ['patient_id'] if level == 'patient' else ['patient_id', 'image_relpath']
    columns = keys + ['label', 'cancer_probability']
    left = left[columns].rename(columns={'cancer_probability': 'probability_left'})
    right = right[columns].rename(columns={'cancer_probability': 'probability_right'})
    merged = left.merge(right, on=keys, suffixes=('_left', '_right'), validate='one_to_one')
    if not merged['label_left'].equals(merged['label_right']):
        raise ValueError(f'{level} labels differ between paired runs')
    return merged.rename(columns={'label_left': 'label'}).drop(columns='label_right')


def clustered_auc_delta_bootstrap(paired, level, samples, rng):
    patient_groups = [
        group.index.to_numpy()
        for _, group in paired.groupby('patient_id', sort=False)
    ]
    group_count = len(patient_groups)
    deltas = []
    attempts = 0
    max_attempts = samples * 10
    while len(deltas) < samples and attempts < max_attempts:
        attempts += 1
        sampled = rng.integers(0, group_count, size=group_count)
        indices = np.concatenate([patient_groups[index] for index in sampled])
        boot = paired.loc[indices]
        if boot['label'].nunique() < 2:
            continue
        left_auc = fast_auc(boot['label'], boot['probability_left'])
        right_auc = fast_auc(boot['label'], boot['probability_right'])
        deltas.append(right_auc - left_auc)
    if len(deltas) != samples:
        raise RuntimeError(f'{level} bootstrap only produced {len(deltas)}/{samples}')
    values = np.asarray(deltas)
    return {
        'bootstrap_samples': samples,
        'delta_auc_mean': float(values.mean()),
        'delta_auc_ci_low': float(np.quantile(values, 0.025)),
        'delta_auc_ci_high': float(np.quantile(values, 0.975)),
        'probability_delta_ge_zero': float(np.mean(values >= 0)),
        'probability_delta_ge_minus_0_005': float(np.mean(values >= -0.005)),
    }


def fast_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = labels.eq(1) if isinstance(labels, pd.Series) else labels == 1
    positive_count = int(np.sum(positives))
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError('AUC requires both classes')
    ranks = rankdata(probabilities, method='average')
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def bootstrap_comparisons(runs, samples, seed):
    rows = []
    rng = np.random.default_rng(seed)
    comparisons = []
    for model in MODELS:
        for run_seed in SEEDS:
            comparisons.append((
                'notch_minus_keep',
                model,
                run_seed,
                runs[('keep', model, run_seed)],
                runs[('notch', model, run_seed)],
            ))
    for preprocessing in PREPROCESSING:
        for run_seed in SEEDS:
            comparisons.append((
                'efficientnet_b0_minus_resnet50',
                preprocessing,
                run_seed,
                runs[(preprocessing, 'resnet50', run_seed)],
                runs[(preprocessing, 'efficientnet_b0', run_seed)],
            ))
    for comparison, stratum, run_seed, left, right in comparisons:
        for level in ('patient', 'image'):
            paired = align_predictions(left[level], right[level], level)
            values = clustered_auc_delta_bootstrap(paired, level, samples, rng)
            rows.append({
                'comparison': comparison,
                'stratum': stratum,
                'seed': run_seed,
                'level': level,
                'left_run': left['name'],
                'right_run': right['name'],
                **values,
            })
    return pd.DataFrame(rows)


def subgroup_table(runs):
    frames = []
    for (preprocessing, model, seed), run in sorted(runs.items()):
        frame = run['subgroup'].copy()
        frame.insert(0, 'seed', seed)
        frame.insert(0, 'model', model)
        frame.insert(0, 'preprocessing', preprocessing)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def dataframe_to_markdown(frame):
    columns = [str(column) for column in frame.columns]
    rows = [columns]
    rows.extend([
        [str(value).replace('|', '\\|') for value in row]
        for row in frame.itertuples(index=False, name=None)
    ])
    widths = [
        max(len(row[index]) for row in rows)
        for index in range(len(columns))
    ]
    lines = [
        '| ' + ' | '.join(
            value.ljust(widths[index]) for index, value in enumerate(rows[0])
        ) + ' |',
        '| ' + ' | '.join('-' * width for width in widths) + ' |',
    ]
    lines.extend([
        '| ' + ' | '.join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + ' |'
        for row in rows[1:]
    ])
    return '\n'.join(lines)


def write_markdown(output_path, run_metrics, summary, deltas, bootstrap):
    display = run_metrics[[
        'preprocessing', 'model', 'seed', 'selected_stage', 'best_epoch',
        'patient_AUC', 'patient_Accuracy', 'patient_Sensitivity',
        'patient_Specificity', 'image_AUC', 'image_Accuracy',
        'image_Sensitivity', 'image_Specificity', 'threshold',
    ]].copy()
    for column in display.select_dtypes(include='number'):
        if column not in {'seed', 'best_epoch'}:
            display[column] = display[column].map(lambda value: f'{value:.4f}')
    summary_display = summary[[
        'preprocessing', 'model',
        'patient_AUC_mean', 'patient_AUC_std', 'patient_AUC_worst',
        'image_AUC_mean', 'image_AUC_std', 'image_AUC_worst',
        'patient_Accuracy_mean', 'image_Accuracy_mean',
        'patient_Sensitivity_mean', 'patient_Specificity_mean',
    ]].copy()
    for column in summary_display.select_dtypes(include='number'):
        summary_display[column] = summary_display[column].map(lambda value: f'{value:.4f}')
    gate_display = deltas[[
        'model', 'seed', 'delta_patient_AUC', 'delta_patient_Sensitivity',
        'delta_patient_Specificity', 'delta_image_AUC', 'passes_all_patient_gates',
    ]].copy()
    for column in gate_display.select_dtypes(include='number'):
        if column != 'seed':
            gate_display[column] = gate_display[column].map(lambda value: f'{value:.4f}')
    patient_bootstrap = bootstrap.loc[bootstrap['level'].eq('patient'), [
        'comparison', 'stratum', 'seed', 'delta_auc_mean',
        'delta_auc_ci_low', 'delta_auc_ci_high',
        'probability_delta_ge_minus_0_005',
    ]].copy()
    for column in patient_bootstrap.select_dtypes(include='number'):
        if column != 'seed':
            patient_bootstrap[column] = patient_bootstrap[column].map(
                lambda value: f'{value:.4f}'
            )
    text = '\n'.join([
        '# M0全量诊断验证矩阵自动汇总',
        '',
        '> 本报告只使用val。internal test和外部test未参与选择。',
        '',
        '## 逐运行指标',
        '',
        dataframe_to_markdown(display),
        '',
        '## 跨种子汇总',
        '',
        dataframe_to_markdown(summary_display),
        '',
        '## Notch相对Keep的配对差值',
        '',
        dataframe_to_markdown(gate_display),
        '',
        '## 患者整簇Bootstrap AUC差值',
        '',
        dataframe_to_markdown(patient_bootstrap),
        '',
        '图片级AUC为重要次要指标；图片级bootstrap也按患者整簇重抽样，',
        '以避免把同一患者多张图误当为独立样本。',
        '',
    ])
    output_path.write_text(text, encoding='utf-8')


def main():
    args = parse_args()
    result_root = args.result_root.resolve()
    output_dir = result_root / '自动汇总'
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(result_root)
    run_metrics = run_metrics_table(runs)
    summary = cross_seed_summary(run_metrics)
    deltas = paired_deltas(run_metrics)
    bootstrap = bootstrap_comparisons(runs, args.bootstrap, args.seed)
    subgroups = subgroup_table(runs)

    run_metrics.to_csv(output_dir / 'm0_run_metrics.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output_dir / 'm0_cross_seed_summary.csv', index=False, encoding='utf-8-sig')
    deltas.to_csv(output_dir / 'm0_paired_deltas.csv', index=False, encoding='utf-8-sig')
    bootstrap.to_csv(output_dir / 'm0_paired_bootstrap.csv', index=False, encoding='utf-8-sig')
    subgroups.to_csv(output_dir / 'm0_all_subgroup_metrics.csv', index=False, encoding='utf-8-sig')
    write_markdown(
        output_dir / 'M0全量诊断验证矩阵汇总.md',
        run_metrics,
        summary,
        deltas,
        bootstrap,
    )
    print(f'Runs: {len(run_metrics)}')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
