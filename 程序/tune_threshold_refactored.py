"""在验证集或测试集上扫描分类阈值，优先满足早癌检出率。"""

import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from inference_refactored import DEVICE, MODEL_REGISTRY, load_model
from train_utils import (
    EVAL_TRANSFORM,
    OUTPUT_DIR,
    GastricDataset,
    compute_metrics,
    load_matched_dataframe,
    split_dataframe,
)


BATCH_SIZE = 32
RANDOM_SEED = 42


@torch.no_grad()
def collect_predictions(model, loader):
    labels_all = []
    probs_all = []
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(DEVICE)), dim=1)[:, 1]
        labels_all.extend(labels.numpy())
        probs_all.extend(probabilities.cpu().numpy())
    return np.asarray(labels_all), np.asarray(probs_all)


def scan_thresholds(y_true, y_prob, step):
    results = []
    for threshold in np.arange(0.10, 0.90 + step / 2, step):
        metrics = compute_metrics(y_true, y_prob, threshold)
        metrics['Threshold'] = round(float(threshold), 4)
        metrics['Youden'] = metrics['Sensitivity'] + metrics['Specificity'] - 1
        results.append(metrics)
    return results


def find_best_threshold(results, sensitivity_floor):
    candidates = [
        result for result in results
        if result['Sensitivity'] >= sensitivity_floor
    ]
    if candidates:
        return max(candidates, key=lambda result: result['Specificity']), (
            f'Sensitivity ≥ {sensitivity_floor:.2f}，选 Specificity 最高'
        )

    best = max(results, key=lambda result: result['Sensitivity'])
    return best, f'无阈值满足底线 {sensitivity_floor:.2f}，改选 Sensitivity 最高'


def print_threshold_table(results, dataset_name):
    header = f"{'Th':>6}  {'Sens':>6}  {'Spec':>6}  {'Acc':>6}  {'Prec':>6}  {'F1':>6}  {'Youden':>6}  {'CM'}"
    print(f'\n{dataset_name} 阈值扫描结果')
    print(header)
    print('-' * len(header))
    for result in results:
        print(
            f"{result['Threshold']:>6.2f}  {result['Sensitivity']:>6.3f}  "
            f"{result['Specificity']:>6.3f}  {result['Accuracy']:>6.3f}  "
            f"{result['Precision']:>6.3f}  {result['F1']:>6.3f}  "
            f"{result['Youden']:>6.3f}  {str(result['CM']):>15}"
        )


def print_metrics(title, dataset_name, dataset_size, threshold, metrics):
    tn, fp, fn, tp = metrics['CM']
    youden = metrics['Sensitivity'] + metrics['Specificity'] - 1
    print(f'\n===== {title} =====')
    print(f'数据集: {dataset_name}（{dataset_size} 张）')
    print(f'阈值: {threshold:.4f}')
    print(f'  Sensitivity: {metrics["Sensitivity"]:.3f}  (漏诊率: {1 - metrics["Sensitivity"]:.1%})')
    print(f'  Specificity: {metrics["Specificity"]:.3f}  (误诊率: {1 - metrics["Specificity"]:.1%})')
    print(f'  Accuracy:    {metrics["Accuracy"]:.3f}')
    print(f'  Precision:   {metrics["Precision"]:.3f}')
    print(f'  F1:          {metrics["F1"]:.3f}')
    print(f'  Youden:      {youden:.3f}')
    print(f'  混淆矩阵 — TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print(f'  每 100 例真癌漏诊 {fn / (tp + fn) * 100:.0f} 例')
    print(f'  每 100 例非癌误报 {fp / (tn + fp) * 100:.0f} 例')
    return youden


def main():
    parser = argparse.ArgumentParser(description='分类阈值调优')
    parser.add_argument('--model', choices=MODEL_REGISTRY, default='resnet50')
    parser.add_argument('--sens', type=float, default=0.90, help='Sensitivity 目标底线')
    parser.add_argument('--step', type=float, default=0.02, help='阈值扫描步长')
    parser.add_argument('--data', choices=['val', 'test'], default='val')
    parser.add_argument('--fixed-threshold', type=float, help='仅评估指定阈值')
    args = parser.parse_args()

    _, validation, test = split_dataframe(load_matched_dataframe(), RANDOM_SEED)
    dataframe = {'val': validation, 'test': test}[args.data]
    dataset = GastricDataset(dataframe, EVAL_TRANSFORM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'当前设备: {DEVICE}')
    print(f'模型: {args.model}')
    print(f'调优数据集: {args.data}（{len(dataset)} 张）')
    y_true, y_prob = collect_predictions(load_model(args.model), loader)

    output_dir = os.path.join(OUTPUT_DIR, '阈值扫描')
    os.makedirs(output_dir, exist_ok=True)

    if args.fixed_threshold is not None:
        metrics = compute_metrics(y_true, y_prob, args.fixed_threshold)
        youden = print_metrics(
            '固定阈值评估', args.data, len(dataset), args.fixed_threshold, metrics,
        )
        output_path = os.path.join(
            output_dir,
            f'fixed_threshold_{args.model}_{args.data}_th{args.fixed_threshold:.2f}.csv',
        )
        pd.DataFrame([{**metrics, 'Threshold': args.fixed_threshold, 'Youden': youden}]).to_csv(
            output_path, index=False, encoding='utf-8-sig',
        )
        print(f'\n结果已保存至: {output_path}')
        return

    results = scan_thresholds(y_true, y_prob, args.step)
    print_threshold_table(results, args.data)
    best, reason = find_best_threshold(results, args.sens)
    print(f'\n选择策略: {reason}')
    print_metrics(
        '推荐阈值', args.data, len(dataset), best['Threshold'], best,
    )

    sensitivity_percent = round(args.sens * 100)
    output_path = os.path.join(
        output_dir,
        f'threshold_scan_{args.model}_{args.data}_sens{sensitivity_percent}.csv',
    )
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'\n扫描结果已保存至: {output_path}')


if __name__ == '__main__':
    main()
