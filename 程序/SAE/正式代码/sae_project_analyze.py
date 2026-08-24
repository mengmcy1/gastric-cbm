"""加载锁定的 SAE，仅投影验证/测试集并生成患者级错误分析。"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
import torch

from sae_discovery import (
    ACTIVE_EPS, EVAL_TRANSFORM, ActivationCapture, SparseAutoencoder, derive_source,
    extract_split_features, fit_panel, heatmap_overlay, load_font,
    load_resnet50, project_features, reconstruct_masked_features,
    reconstruction_metrics, seed_everything, softmax_numpy,
)


GROUPS = ['TP', 'TN', 'FP', 'FN']
TRANSITIONS = ['TP_to_FN', 'TN_to_FP', 'FP_to_TN', 'FN_to_TP']


def parse_args():
    """读取冻结投影、概览数量和输出目录参数。"""
    parser = argparse.ArgumentParser(description='锁定 SAE 的独立数据投影与错误分析')
    parser.add_argument('--sae-run', required=True, help='已锁定的 SAE 实验目录')
    parser.add_argument('--split', choices=['val', 'test', 'external'], default='val')
    parser.add_argument(
        '--external-manifest', default=None,
        help='外部预处理manifest；提供后自动使用external split，不参与训练或调参',
    )
    parser.add_argument('--experiment', required=True, help='新建的独立输出目录名')
    parser.add_argument('--image-batch-size', type=int, default=32)
    parser.add_argument('--sae-batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--overview-features', type=int, default=10,
                        help='每个分析类别展示的 feature 数')
    parser.add_argument('--top-images', type=int, default=6,
                        help='每个 feature 展示的不同患者数')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def confusion_types(labels, probabilities, threshold):
    """按固定临床阈值返回 TP/TN/FP/FN。"""
    predicted = (probabilities >= threshold).astype(int)
    return np.select(
        [(labels == 1) & (predicted == 1), (labels == 0) & (predicted == 0),
         (labels == 0) & (predicted == 1), (labels == 1) & (predicted == 0)],
        GROUPS, default='',
    )


def classification_metrics(labels, probabilities, threshold):
    """计算固定阈值下的常用分类指标。"""
    predicted = (probabilities >= threshold).astype(int)
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tp = int(((labels == 1) & (predicted == 1)).sum())
    return {
        'auc': float(roc_auc_score(labels, probabilities)),
        'sensitivity': float(tp / max(tp + fn, 1)),
        'specificity': float(tn / max(tn + fp, 1)),
        'accuracy': float((tp + tn) / len(labels)),
        'precision': float(tp / max(tp + fp, 1)),
        'f1': float(2 * tp / max(2 * tp + fp + fn, 1)),
        'threshold': threshold,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def load_locked_sae(run_dir, device):
    """恢复 SAE checkpoint 和训练期确定的 feature 保留掩码。"""
    checkpoint = torch.load(
        os.path.join(run_dir, 'SAE模型', 'sae_best.pth'),
        map_location=device, weights_only=False,
    )
    center = checkpoint['sae_state_dict']['decoder_bias'].detach().clone().to(device)
    sae = SparseAutoencoder(
        checkpoint['input_dim'], checkpoint['hidden_dim'], center,
    ).to(device)
    sae.load_state_dict(checkpoint['sae_state_dict'])
    sae.eval()
    for parameter in sae.parameters():
        parameter.requires_grad = False
    decisions = pd.read_csv(
        os.path.join(run_dir, 'feature筛选', 'feature_pruning_decisions.csv'),
        encoding='utf-8-sig',
    ).sort_values('feature_id')
    kept_mask = decisions['kept_after_pruning'].astype(bool).to_numpy(copy=True)
    return sae, checkpoint, kept_mask


def load_or_extract_features(args, output_cache, model, capture, device):
    """验证集复用锁定run缓存；无缓存时从原图提取指定split。"""
    source_cache = os.path.join(args.sae_run, '特征缓存')
    feature_path = os.path.join(source_cache, f'{args.split}_gap_features.npy')
    metadata_path = os.path.join(source_cache, f'{args.split}_metadata.csv')
    if (args.external_manifest is None
            and os.path.exists(feature_path) and os.path.exists(metadata_path)):
        features = np.load(feature_path).astype(np.float32)
        metadata = pd.read_csv(metadata_path, encoding='utf-8-sig')
        print(f'复用 {args.split} 特征缓存: {features.shape}')
        return features, metadata
    manifest_path = args.external_manifest or args.split_csv
    dataframe = pd.read_csv(manifest_path, encoding='utf-8-sig').rename(
        columns={'瘤变标签': 'label'},
    )
    if 'image_relpath' not in dataframe:
        for column in ['processed_path', '图片名字', 'image_path']:
            if column in dataframe:
                dataframe['image_relpath'] = dataframe[column]
                break
    if 'image_relpath' not in dataframe:
        raise ValueError('划分CSV缺少 image_relpath/processed_path/图片名字/image_path')
    required = {'patient_id', 'label'}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f'清单缺少必要字段: {sorted(missing)}')
    if args.external_manifest is not None:
        dataframe['split'] = 'external'
        dataframe['domain'] = '外院'
        dataframe['hospital'] = (
            dataframe['center'].fillna('外部多中心').astype(str)
            if 'center' in dataframe else '外部多中心'
        )
    else:
        if 'center' not in dataframe:
            raise ValueError('划分CSV缺少center字段')
        source = dataframe['center'].apply(derive_source)
        dataframe['domain'] = source.str[0]
        dataframe['hospital'] = source.str[1]
    extract_args = SimpleNamespace(
        image_batch_size=args.image_batch_size, num_workers=args.num_workers,
        data_dir=args.data_dir, image_threshold=args.image_threshold,
    )
    return extract_split_features(
        model, capture, dataframe, args.split, extract_args, device, output_cache,
    )


def make_prediction_tables(
    metadata, features, reconstructed, pruned_reconstructed, fc_weight, fc_bias,
    image_threshold, patient_threshold,
):
    """生成图像级、患者级概率、混淆类型及SAE错误转换。"""
    labels = metadata['label'].to_numpy(dtype=int)
    probabilities = {
        'original_probability': softmax_numpy(features @ fc_weight.T + fc_bias)[:, 1],
        'sae_full_probability': softmax_numpy(
            reconstructed @ fc_weight.T + fc_bias,
        )[:, 1],
        'sae_pruned_probability': softmax_numpy(
            pruned_reconstructed @ fc_weight.T + fc_bias,
        )[:, 1],
    }
    images = metadata.copy()
    for name, values in probabilities.items():
        images[name] = values
    images['original_confusion'] = confusion_types(
        labels, probabilities['original_probability'], image_threshold,
    )
    images['sae_pruned_confusion'] = confusion_types(
        labels, probabilities['sae_pruned_probability'], image_threshold,
    )
    images['prediction_transition'] = (
        images['original_confusion'] + '_to_' + images['sae_pruned_confusion']
    )
    images['absolute_probability_change'] = np.abs(
        probabilities['sae_pruned_probability'] - probabilities['original_probability'],
    )
    patients = images.groupby('patient_id', sort=True).agg(
        label=('label', 'first'), domain=('domain', 'first'),
        hospital=('hospital', 'first'), image_count=('label', 'size'),
        original_probability=('original_probability', 'mean'),
        sae_full_probability=('sae_full_probability', 'mean'),
        sae_pruned_probability=('sae_pruned_probability', 'mean'),
    ).reset_index()
    patient_labels = patients['label'].to_numpy(dtype=int)
    patients['original_confusion'] = confusion_types(
        patient_labels, patients['original_probability'].to_numpy(), patient_threshold,
    )
    patients['sae_pruned_confusion'] = confusion_types(
        patient_labels, patients['sae_pruned_probability'].to_numpy(), patient_threshold,
    )
    patients['prediction_transition'] = (
        patients['original_confusion'] + '_to_' + patients['sae_pruned_confusion']
    )
    patients['absolute_probability_change'] = np.abs(
        patients['sae_pruned_probability'] - patients['original_probability'],
    )
    return images, patients


def patient_activation_tables(activations, metadata, patient_predictions):
    """将多张图片激活汇总为患者最大值和患者总激活。"""
    patient_codes, patients = pd.factorize(metadata['patient_id'], sort=True)
    patient_max = np.zeros((len(patients), activations.shape[1]), dtype=np.float32)
    patient_sum = np.zeros_like(patient_max)
    np.maximum.at(patient_max, patient_codes, activations)
    np.add.at(patient_sum, patient_codes, activations)
    patient_meta = patient_predictions.set_index('patient_id').loc[patients].reset_index()
    return patient_max, patient_sum, patient_meta, patient_codes


def group_mean(values, groups, name):
    """返回指定患者组的 feature 均值；空组以 NaN 表示。"""
    mask = groups == name
    if not mask.any():
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    return values[mask].mean(axis=0)


def group_active_count(values, groups, name):
    """返回指定患者组中激活每个 feature 的患者数。"""
    mask = groups == name
    if not mask.any():
        return np.zeros(values.shape[1], dtype=int)
    return (values[mask] > ACTIVE_EPS).sum(axis=0)


def build_feature_summary(
    activations, metadata, patients, sae, fc_weight, fc_bias,
    kept_mask, pruned_reconstructed,
):
    """生成标签、来源、错误分组、患者贡献和消融统计。"""
    patient_max, patient_sum, patient_meta, patient_codes = patient_activation_tables(
        activations, metadata, patients,
    )
    margin_direction = (
        sae.decoder_weight.detach().cpu().numpy() @ (fc_weight[1] - fc_weight[0])
    )
    total_activation = patient_sum.sum(axis=0)
    summary = pd.DataFrame({
        'feature_id': np.arange(activations.shape[1]),
        'kept_after_pruning': kept_mask,
        'active_image_count_test': (activations > ACTIVE_EPS).sum(axis=0),
        'active_patient_count_test': (patient_max > ACTIVE_EPS).sum(axis=0),
        'activation_density_test': (activations > ACTIVE_EPS).mean(axis=0),
        'mean_patient_max_test': patient_max.mean(axis=0),
        'max_patient_contribution_ratio_test': (
            patient_sum.max(axis=0) / np.maximum(total_activation, 1e-12)
        ),
        'cancer_margin_direction': margin_direction,
    })
    labels = patient_meta['label'].to_numpy(dtype=int)
    domains = patient_meta['domain'].to_numpy()
    original_groups = patient_meta['original_confusion'].to_numpy()
    sae_groups = patient_meta['sae_pruned_confusion'].to_numpy()
    transitions = patient_meta['prediction_transition'].to_numpy()
    for label in [0, 1]:
        summary[f'mean_patient_max_label_{label}'] = patient_max[labels == label].mean(0)
    for domain, suffix in [('省人民', 'provincial'), ('外院', 'external')]:
        summary[f'mean_patient_max_{suffix}'] = patient_max[domains == domain].mean(0)
        summary[f'active_patient_count_{suffix}'] = (
            patient_max[domains == domain] > ACTIVE_EPS
        ).sum(0)
    for group in GROUPS:
        summary[f'mean_patient_max_original_{group}'] = group_mean(
            patient_max, original_groups, group,
        )
        summary[f'active_patient_count_original_{group}'] = group_active_count(
            patient_max, original_groups, group,
        )
        summary[f'mean_patient_max_sae_{group}'] = group_mean(
            patient_max, sae_groups, group,
        )
    for transition in TRANSITIONS:
        summary[f'mean_patient_max_{transition}'] = group_mean(
            patient_max, transitions, transition,
        )
        summary[f'patient_count_{transition}'] = int((transitions == transition).sum())
    summary['cancer_activation_difference'] = (
        summary['mean_patient_max_label_1'] - summary['mean_patient_max_label_0']
    )
    summary['source_activation_difference'] = (
        summary['mean_patient_max_external'] - summary['mean_patient_max_provincial']
    )
    summary['fp_vs_tn_activation_difference'] = (
        summary['mean_patient_max_original_FP'] - summary['mean_patient_max_original_TN']
    )
    summary['fn_vs_tp_activation_difference'] = (
        summary['mean_patient_max_original_FN'] - summary['mean_patient_max_original_TP']
    )

    base_logits = pruned_reconstructed @ fc_weight.T + fc_bias
    base_margin = base_logits[:, 1] - base_logits[:, 0]
    feature_margin = activations * margin_direction[None, :] * kept_mask[None, :]
    ablated_probability = 1 / (1 + np.exp(-(base_margin[:, None] - feature_margin)))
    patient_ablated_sum = np.zeros_like(patient_max)
    np.add.at(patient_ablated_sum, patient_codes, ablated_probability)
    patient_image_counts = np.bincount(patient_codes)[:, None]
    patient_ablated_mean = patient_ablated_sum / patient_image_counts
    ablation_delta = (
        patient_meta['sae_pruned_probability'].to_numpy()[:, None] - patient_ablated_mean
    )
    summary['mean_patient_ablation_delta'] = ablation_delta.mean(0)
    for group in GROUPS:
        summary[f'mean_ablation_delta_original_{group}'] = group_mean(
            ablation_delta, original_groups, group,
        )
    return summary, patient_meta


def build_confusion_stats(summary, patients):
    """输出适合统计软件读取的长表错误分组feature统计。"""
    records = []
    for group in GROUPS:
        count = int((patients['original_confusion'] == group).sum())
        for row in summary.itertuples(index=False):
            records.append({
                'feature_id': int(row.feature_id), 'group': group,
                'patient_count': count,
                'active_patient_count': int(getattr(
                    row, f'active_patient_count_original_{group}',
                )),
                'mean_patient_max': getattr(row, f'mean_patient_max_original_{group}'),
                'mean_ablation_delta': getattr(
                    row, f'mean_ablation_delta_original_{group}',
                ),
            })
    return pd.DataFrame(records)


def build_rankings(summary, patient_meta, top_n):
    """按医学审阅目标生成feature候选排行。"""
    kept = summary[summary['kept_after_pruning']].copy()
    specs = [
        ('cancer_related', 'cancer_activation_difference', False, False),
        ('noncancer_related', 'cancer_activation_difference', True, False),
        ('source_related', 'source_activation_difference', False, True),
        ('FP_related', 'fp_vs_tn_activation_difference', False, False),
        ('FN_related', 'fn_vs_tp_activation_difference', False, True),
    ]
    records = []
    for category, column, ascending, absolute in specs:
        candidates = kept.dropna(subset=[column]).copy()
        candidates['ranking_score'] = candidates[column].abs() if absolute else candidates[column]
        candidates = candidates.sort_values('ranking_score', ascending=ascending).head(top_n)
        for rank, row in enumerate(candidates.itertuples(index=False), 1):
            records.append({
                'category': category, 'rank': rank,
                'feature_id': int(row.feature_id),
                'ranking_score': float(row.ranking_score),
                'signed_difference': float(getattr(row, column)),
            })
    transitions = patient_meta['prediction_transition'].to_numpy()
    for transition in TRANSITIONS:
        if not (transitions == transition).any():
            continue
        column = f'mean_patient_max_{transition}'
        for rank, row in enumerate(kept.dropna(subset=[column]).nlargest(
            top_n, column,
        ).itertuples(index=False), 1):
            records.append({
                'category': transition, 'rank': rank,
                'feature_id': int(row.feature_id),
                'ranking_score': float(getattr(row, column)),
                'signed_difference': np.nan,
            })
    return pd.DataFrame(records)


def category_mask(category, images):
    """限制概览候选图片到对应标签、错误组或转换组。"""
    if category == 'cancer_related':
        return images['label'].to_numpy() == 1
    if category == 'noncancer_related':
        return images['label'].to_numpy() == 0
    if category == 'FP_related':
        return images['original_confusion'].to_numpy() == 'FP'
    if category == 'FN_related':
        return images['original_confusion'].to_numpy() == 'FN'
    if category in TRANSITIONS:
        return images['prediction_transition'].to_numpy() == category
    return np.ones(len(images), dtype=bool)


@torch.no_grad()
def make_overviews(
    model, capture, sae, activations, images, rankings, output_dir, top_images,
    data_dir, device,
):
    """为每类候选feature保存不同患者的原图与encoder激活热图。"""
    title_font, text_font = load_font(21), load_font(14)
    records = []
    for category, rows in rankings.groupby('category', sort=False):
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir)
        allowed = category_mask(category, images)
        for rank_row in rows.itertuples(index=False):
            feature_id = int(rank_row.feature_id)
            selected, used = [], set()
            for index in np.argsort(activations[:, feature_id])[::-1]:
                patient_id = images.iloc[index]['patient_id']
                if (not allowed[index] or activations[index, feature_id] <= ACTIVE_EPS
                        or patient_id in used):
                    continue
                selected.append(index)
                used.add(patient_id)
                if len(selected) == top_images:
                    break
            canvas = Image.new('RGB', (640, 70 + 290 * len(selected)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 8), f'{category} | Feature {feature_id} | '
                f'score={rank_row.ranking_score:.4f}',
                fill=(20, 20, 20), font=title_font,
            )
            direction = sae.encoder.weight[feature_id].detach()
            for row_index, index in enumerate(selected):
                item = images.iloc[index]
                image_path = item['image_relpath']
                if not os.path.isabs(image_path):
                    image_path = os.path.join(data_dir, image_path)
                original = Image.open(image_path).convert('RGB')
                model(EVAL_TRANSFORM(original).unsqueeze(0).to(device))
                concept_map = torch.relu(
                    (capture.output[0] * direction[:, None, None]).sum(0)
                ).cpu().numpy()
                y = 70 + row_index * 290
                canvas.paste(fit_panel(original), (0, y))
                canvas.paste(fit_panel(heatmap_overlay(original, concept_map)), (320, y))
                draw.text(
                    (8, y + 245),
                    f'{item.patient_id}  label={item.label}  {item.domain}  '
                    f'{item.original_confusion}→{item.sae_pruned_confusion}  '
                    f'激活={activations[index, feature_id]:.4f}',
                    fill=(20, 20, 20), font=text_font,
                )
                records.append({
                    'category': category, 'feature_id': feature_id,
                    'rank': row_index + 1, 'image_name': item['image_relpath'],
                    'patient_id': item['patient_id'], 'label': int(item['label']),
                    'domain': item['domain'], 'hospital': item['hospital'],
                    'original_confusion': item['original_confusion'],
                    'sae_pruned_confusion': item['sae_pruned_confusion'],
                    'prediction_transition': item['prediction_transition'],
                    'activation': float(activations[index, feature_id]),
                })
            canvas.save(os.path.join(category_dir, f'feature_{feature_id:04d}.png'))
            print(f'保存 {category} feature {feature_id}，展示 {len(selected)} 位患者')
    pd.DataFrame(records).to_csv(
        os.path.join(output_dir, 'feature_top_examples.csv'),
        index=False, encoding='utf-8-sig',
    )


def save_error_cases(images, patients, output_dir):
    """分别保存原模型FP/FN以及SAE前后错误转换病例。"""
    for group in ['FP', 'FN']:
        images[images['original_confusion'] == group].to_csv(
            os.path.join(output_dir, f'original_{group}_images.csv'),
            index=False, encoding='utf-8-sig',
        )
        patients[patients['original_confusion'] == group].to_csv(
            os.path.join(output_dir, f'original_{group}_patients.csv'),
            index=False, encoding='utf-8-sig',
        )
    for transition in TRANSITIONS:
        images[images['prediction_transition'] == transition].to_csv(
            os.path.join(output_dir, f'{transition}_images.csv'),
            index=False, encoding='utf-8-sig',
        )
        patients[patients['prediction_transition'] == transition].to_csv(
            os.path.join(output_dir, f'{transition}_patients.csv'),
            index=False, encoding='utf-8-sig',
        )


def main():
    """执行冻结模型投影、患者级统计、错误分组和概览保存。"""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.sae_run = os.path.abspath(args.sae_run)
    if args.external_manifest is not None:
        args.external_manifest = os.path.abspath(args.external_manifest)
        args.split = 'external'
    elif args.split == 'external':
        raise ValueError('--split external 必须同时提供 --external-manifest')
    with open(os.path.join(args.sae_run, 'config.json'), encoding='utf-8') as file:
        source_config = json.load(file)
    for name in [
        'data_dir', 'split_csv', 'weight_path', 'output_root',
        'image_threshold', 'patient_threshold',
    ]:
        if name not in source_config:
            raise ValueError(f'SAE训练配置缺少必要字段: {name}')
        setattr(args, name, source_config[name])
    output_dir = os.path.join(args.output_root, args.experiment)
    os.makedirs(output_dir, exist_ok=False)
    cache_dir = os.path.join(output_dir, '特征缓存')
    error_dir = os.path.join(output_dir, '错误病例')
    overview_dir = os.path.join(output_dir, 'feature概览')
    for directory in [cache_dir, error_dir, overview_dir]:
        os.makedirs(directory)

    model = load_resnet50(device, args.weight_path)
    capture = ActivationCapture(model.layer4[-1])
    sae, checkpoint, kept_mask = load_locked_sae(args.sae_run, device)
    features, metadata = load_or_extract_features(args, cache_dir, model, capture, device)
    reconstructed, activations = project_features(
        sae, features, args.sae_batch_size, device,
    )
    pruned = reconstruct_masked_features(
        sae, activations, kept_mask, args.sae_batch_size, device,
    )
    np.save(os.path.join(cache_dir, f'{args.split}_gap_features.npy'), features)
    np.savez_compressed(
        os.path.join(cache_dir, f'{args.split}_sae_projection.npz'),
        activations=activations, reconstructed=reconstructed, pruned_reconstructed=pruned,
    )
    metadata.to_csv(
        os.path.join(cache_dir, f'{args.split}_metadata.csv'),
        index=False, encoding='utf-8-sig',
    )

    fc_weight = model.fc.weight.detach().cpu().numpy()
    fc_bias = model.fc.bias.detach().cpu().numpy()
    images, patients = make_prediction_tables(
        metadata, features, reconstructed, pruned, fc_weight, fc_bias,
        args.image_threshold, args.patient_threshold,
    )
    summary, patient_meta = build_feature_summary(
        activations, metadata, patients, sae, fc_weight, fc_bias,
        kept_mask, pruned,
    )
    development_path = os.path.join(args.sae_run, 'feature_summary.csv')
    if os.path.exists(development_path):
        development = pd.read_csv(development_path, encoding='utf-8-sig')
        columns = [c for c in development if (
            c == 'feature_id' or c.endswith('_train') or c.endswith('_val')
        )]
        summary = summary.merge(development[columns], on='feature_id', how='left')
    confusion_stats = build_confusion_stats(summary, patients)
    rankings = build_rankings(summary, patient_meta, args.overview_features)

    images.to_csv(os.path.join(output_dir, 'image_predictions.csv'), index=False,
                  encoding='utf-8-sig')
    patients.to_csv(os.path.join(output_dir, 'patient_predictions.csv'), index=False,
                    encoding='utf-8-sig')
    summary.to_csv(os.path.join(output_dir, 'feature_test_summary.csv'), index=False,
                   encoding='utf-8-sig')
    confusion_stats.to_csv(
        os.path.join(output_dir, 'confusion_group_feature_stats.csv'),
        index=False, encoding='utf-8-sig',
    )
    rankings.to_csv(os.path.join(output_dir, 'feature_rankings.csv'), index=False,
                    encoding='utf-8-sig')
    patients[[
        'patient_id', 'label', 'domain', 'hospital', 'original_confusion',
        'sae_pruned_confusion', 'prediction_transition', 'original_probability',
        'sae_pruned_probability', 'absolute_probability_change',
    ]].to_csv(os.path.join(output_dir, 'prediction_transitions.csv'), index=False,
              encoding='utf-8-sig')
    save_error_cases(images, patients, error_dir)

    image_labels = images['label'].to_numpy(dtype=int)
    patient_labels = patients['label'].to_numpy(dtype=int)
    metrics = {
        'source_sae_run': args.sae_run, 'split': args.split,
        'external_manifest': args.external_manifest,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'lambda_l1': float(checkpoint['lambda_l1']),
        'kept_feature_count': int(kept_mask.sum()),
        'total_feature_count': int(len(kept_mask)),
        'reconstruction_full': reconstruction_metrics(
            features, reconstructed, metadata, fc_weight, fc_bias,
            args.patient_threshold,
        ),
        'reconstruction_pruned': reconstruction_metrics(
            features, pruned, metadata, fc_weight, fc_bias,
            args.patient_threshold,
        ),
        'image_level': {}, 'patient_level': {},
        'image_transition_counts': images['prediction_transition'].value_counts().to_dict(),
        'patient_transition_counts': patients['prediction_transition'].value_counts().to_dict(),
    }
    for name in ['original', 'sae_full', 'sae_pruned']:
        metrics['image_level'][name] = classification_metrics(
            image_labels, images[f'{name}_probability'].to_numpy(),
            args.image_threshold,
        )
        metrics['patient_level'][name] = classification_metrics(
            patient_labels, patients[f'{name}_probability'].to_numpy(),
            args.patient_threshold,
        )
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
    make_overviews(
        model, capture, sae, activations, images, rankings,
        overview_dir, args.top_images, args.data_dir, device,
    )
    capture.close()
    print(f'投影完成: {args.split}，未训练 SAE')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()
