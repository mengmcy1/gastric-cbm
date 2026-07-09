"""
阈值调优脚本
加载已训练好的模型，在验证集上扫描阈值，根据临床需求选择最佳阈值。

策略（胃早癌筛查场景）：
  - Sens 优先：漏诊 (FN) 的代价远大于误诊 (FP)
  - 在满足 Sens ≥ 目标底线的前提下，选 Spec 最高的阈值

用法：
  python tune_threshold.py                                          # 默认 ResNet50 + Sens 0.90
  python tune_threshold.py --model efficientnet_b0                   # EfficientNet-B0 对照
  python tune_threshold.py --sens 0.85 --model resnet50             # 指定 Sens 底线
  python tune_threshold.py --model resnet50 --data test --fixed-threshold 0.20      # 固定阈值评估
  python tune_threshold.py --model efficientnet_b0 --data test --fixed-threshold 0.22
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score,
)

# 复用训练脚本中的组件
from resnet_train_final import GastricDataset, compute_metrics, format_metrics

# ---- 路径配置 ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', '数据', '胃图文带特征标签数据集 3600+ 1933瘤变')
CSV_PATH   = os.path.join(BASE_DIR, '..', '数据', '胃图文标签表格-添加瘤变标签.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), '结果')

# 模型名 → (权重文件名, 构建函数)
MODEL_REGISTRY = {
    'resnet50':        ('resnet50_transfer_best.pth', 'resnet'),
    'efficientnet_b0': ('efficientnet_b0_best.pth',   'efficientnet'),
}

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model(model_name, weight_path):
    """根据模型名构建对应架构并恢复权重。"""
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f'不支持的模型: {model_name}')

    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def collect_predictions(model, loader):
    """遍历数据集，收集所有标签和预测概率。"""
    all_labels = []
    all_probs  = []

    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def scan_thresholds(y_true, y_prob, step=0.02):
    """
    从 0.10 到 0.90 扫描阈值，返回每档的完整指标。

    Returns:
        list[dict]: 每个阈值一行，包含 threshold / Sens / Spec / Acc / Prec / F1 / Youden / CM
    """
    results = []
    thresholds = np.arange(0.10, 0.92, step)

    for th in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=th)
        m['Threshold'] = round(th, 2)
        m['Youden']    = m['Sensitivity'] + m['Specificity'] - 1
        results.append(m)

    return results


def find_best_threshold(results, sens_floor=0.90):
    """
    在所有满足 Sens ≥ sens_floor 的阈值中，选 AUC/Youden 最高的。

    如果没有阈值能满足 Sens 底线，降级策略：
      1. 选 Sens 最高的阈值（即使不达标）
      2. 选 Youden 最大的阈值（统计最优，不管 Sens 底线）
    """
    # 策略 A：Sens 底线
    candidates = [r for r in results if r['Sensitivity'] >= sens_floor]
    if candidates:
        best = max(candidates, key=lambda r: r['Specificity'])
        return 'sens_floor', best, f'Sens ≥ {sens_floor}，选 Spec 最高'

    # 降级 B：Sens 最高
    best_by_sens = max(results, key=lambda r: r['Sensitivity'])
    # 降级 C：Youden 最大
    best_by_youden = max(results, key=lambda r: r['Youden'])

    return (
        'fallback',
        best_by_sens,
        f'Sens 底线 {sens_floor} 无法满足 → 降级为 Sens 最高 ({best_by_sens["Sensitivity"]:.3f})，'
        f'Youden 最优阈值 {best_by_youden["Threshold"]}（Youden={best_by_youden["Youden"]:.3f}）'
    )


def print_threshold_table(results):
    """打印完整的阈值扫描表。"""
    header = f"{'Th':>6}  {'Sens':>6}  {'Spec':>6}  {'Acc':>6}  {'Prec':>6}  {'F1':>6}  {'Youden':>6}  {'CM'}"
    print('\n' + '=' * len(header))
    print('阈值扫描结果（验证集）')
    print('=' * len(header))
    print(header)
    print('-' * len(header))

    for r in results:
        print(f"{r['Threshold']:>6.2f}  "
              f"{r['Sensitivity']:>6.3f}  "
              f"{r['Specificity']:>6.3f}  "
              f"{r['Accuracy']:>6.3f}  "
              f"{r['Precision']:>6.3f}  "
              f"{r['F1']:>6.3f}  "
              f"{r['Youden']:>6.3f}  "
              f"{str(r['CM']):>15}")

    print('-' * len(header))


def main():
    parser = argparse.ArgumentParser(description='阈值调优')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0'],
                        help='模型选择，默认 resnet50')
    parser.add_argument('--sens', type=float, default=0.90,
                        help='Sensitivity 目标底线，默认 0.90')
    parser.add_argument('--step', type=float, default=0.02,
                        help='扫描步长，默认 0.02')
    parser.add_argument('--data', type=str, default='val',
                        choices=['val', 'test'],
                        help='在哪个数据集上调优：val（验证集）或 test（测试集）')
    parser.add_argument('--fixed-threshold', type=float, default=None,
                        help='固定阈值评估模式：不扫描、不推荐，仅输出该阈值下的完整指标')
    args = parser.parse_args()

    # ---- 加载数据 ----
    weight_file, _ = MODEL_REGISTRY[args.model]
    model_path = os.path.join(OUTPUT_DIR, '模型权重', weight_file)

    print(f'当前设备: {DEVICE}')
    print(f'模型: {args.model}')
    print(f'权重: {model_path}')

    # 读 CSV 并重复训练时的数据划分逻辑
    from resnet_train_final import load_matched_dataframe, split_dataframe

    df_valid = load_matched_dataframe(CSV_PATH, DATA_DIR)
    train_df, val_df, test_df = split_dataframe(df_valid)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    df = val_df if args.data == 'val' else test_df
    dataset = GastricDataset(df, DATA_DIR, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f'调优数据集: {args.data}（{len(dataset)} 张）')

    # ---- 加载模型 & 收集预测 ----
    model = build_model(args.model, model_path)
    y_true, y_prob = collect_predictions(model, loader)

    # ---- 固定阈值模式：不扫描、不推荐，仅评估 ----
    if args.fixed_threshold is not None:
        m = compute_metrics(y_true, y_prob, threshold=args.fixed_threshold)
        youden = m['Sensitivity'] + m['Specificity'] - 1

        print(f'\n===== 固定阈值评估结果 =====')
        print(f'数据集:  {args.data}（{len(dataset)} 张）')
        print(f'阈值:    {args.fixed_threshold}')
        print(f'  Sensitivity: {m["Sensitivity"]:.3f}  (漏诊率: {1-m["Sensitivity"]:.1%})')
        print(f'  Specificity: {m["Specificity"]:.3f}  (误诊率: {1-m["Specificity"]:.1%})')
        print(f'  Accuracy:    {m["Accuracy"]:.3f}')
        print(f'  Precision:   {m["Precision"]:.3f}')
        print(f'  F1:          {m["F1"]:.3f}')
        print(f'  Youden:      {youden:.3f}')
        tn, fp, fn, tp = m['CM']
        print(f'  混淆矩阵 — TN={tn}, FP={fp}, FN={fn}, TP={tp}')
        print(f'  每 100 例真癌漏诊 {fn/(tp+fn)*100:.0f} 例')
        print(f'  每 100 例非癌误报 {fp/(tn+fp)*100:.0f} 例')

        output_csv = os.path.join(
            OUTPUT_DIR, '阈值扫描',
            f'fixed_threshold_{args.model}_{args.data}_th{args.fixed_threshold:.2f}.csv',
        )
        pd.DataFrame([{**m, 'Threshold': args.fixed_threshold, 'Youden': youden}]
                     ).to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f'\n结果已保存至: {output_csv}')
        return

    # ---- 扫描模式 ----
    results = scan_thresholds(y_true, y_prob, step=args.step)

    # ---- 打印 ----
    print_threshold_table(results)

    # ---- 找最佳阈值 ----
    strategy, best, reason = find_best_threshold(results, sens_floor=args.sens)

    print(f'\n===== 推荐阈值 =====')
    print(f'策略: {reason}')
    print(f'最佳阈值: {best["Threshold"]:.2f}')
    print(f'  Sensitivity: {best["Sensitivity"]:.3f}  (漏诊率: {1-best["Sensitivity"]:.1%})')
    print(f'  Specificity: {best["Specificity"]:.3f}  (误诊率: {1-best["Specificity"]:.1%})')
    print(f'  Accuracy:    {best["Accuracy"]:.3f}')
    print(f'  Precision:   {best["Precision"]:.3f}')
    print(f'  F1:          {best["F1"]:.3f}')
    print(f'  Youden:      {best["Youden"]:.3f}')
    tn, fp, fn, tp = best['CM']
    print(f'  混淆矩阵 — TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    print(f'  每 100 例真癌漏诊 {fn/(tp+fn)*100:.0f} 例')
    print(f'  每 100 例非癌误报 {fp/(tn+fp)*100:.0f} 例')

    # ---- 保存 ----
    output_csv = os.path.join(OUTPUT_DIR, '阈值扫描', f'threshold_scan_{args.model}_{args.data}_sens{args.sens:.0f}.csv')
    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f'\n扫描结果已保存至: {output_csv}')


if __name__ == '__main__':
    main()
