"""使用医生确认的概念标签训练 M-CBM 概念层和稀疏癌/非癌分类器。"""

import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, '结果', 'M-CBM', '第二批', 'resnet50')
INPUT_DIM = 2048


def parse_args():
    """读取特征缓存、概念标注和两阶段训练参数。"""
    parser = argparse.ArgumentParser(description='M-CBM 概念瓶颈训练')
    parser.add_argument('--sae-run', required=True,
                        help='已完成 SAE 实验目录，读取其中的 GAP 特征缓存')
    parser.add_argument('--concept-catalog', required=True,
                        help='概念目录 CSV，使用 use_for_cbl=1 的概念')
    parser.add_argument('--annotations', required=True,
                        help='医生共识概念标注 CSV')
    parser.add_argument('--include-test', action='store_true',
                        help='锁定 CBL 和分类器后评估测试集')
    parser.add_argument('--experiment', default=None, help='输出实验名；默认时间戳')
    parser.add_argument('--epochs', type=int, default=500, help='CBL 最大 epoch')
    parser.add_argument('--patience', type=int, default=30, help='CBL 验证损失早停')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='CBL 学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='CBL batch size')
    parser.add_argument('--classifier-c-grid', type=float, nargs='+',
                        default=[0.01, 0.1, 1.0, 10.0],
                        help='稀疏逻辑回归 C 候选值')
    parser.add_argument('--target-ncc', type=float, default=5.0,
                        help='优先选择 NCC95 不超过该值的分类器')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()


def seed_everything(seed):
    """固定 CBL 和稀疏分类器随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_catalog(path):
    """读取最终概念顺序，仅保留 use_for_cbl=1 的命名概念。"""
    catalog = pd.read_csv(path, encoding='utf-8-sig', dtype={'concept_id': str})
    catalog = catalog[catalog['use_for_cbl'].astype(int) == 1].copy()
    catalog['concept_id'] = catalog['concept_id'].astype(str)
    catalog = catalog.drop_duplicates('concept_id').reset_index(drop=True)
    return catalog


def load_feature_split(sae_run, split):
    """读取 SAE 阶段缓存的 GAP 特征和同序图片元数据。"""
    feature_dir = os.path.join(sae_run, '特征缓存')
    features = np.load(os.path.join(feature_dir, f'{split}_gap_features.npy'))
    metadata = pd.read_csv(
        os.path.join(feature_dir, f'{split}_metadata.csv'), encoding='utf-8-sig',
    )
    metadata['image_path'] = metadata['图片名字']
    return features.astype(np.float32), metadata


def load_concept_targets(annotation_path, metadata, concept_ids):
    """把长表概念标注转换为与图片同序的 [N,K] 目标矩阵。"""
    annotations = pd.read_csv(
        annotation_path, encoding='utf-8-sig', dtype={'concept_id': str},
    )
    annotations['concept_id'] = annotations['concept_id'].astype(str)
    annotations = annotations[annotations['concept_id'].isin(concept_ids)].copy()
    duplicated = annotations.duplicated(['image_path', 'concept_id'], keep=False)
    if duplicated.any():
        raise ValueError('共识标注中同一 image_path + concept_id 只能保留一行')
    table = annotations.pivot(
        index='image_path', columns='concept_id', values='concept_label',
    ).reindex(index=metadata['image_path'], columns=concept_ids)
    return table.fillna(-1).to_numpy(dtype=np.float32)


class ConceptBottleneckLayer(nn.Module):
    """把 ResNet50 GAP 特征 [B,2048] 映射为 K 个命名概念 logits。"""

    def __init__(self, input_dim, num_concepts):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_concepts)

    def forward(self, features):
        return self.linear(features)


def concept_pos_weight(targets):
    """按训练标注中每个概念的阴阳性数量计算 BCE 正类权重。"""
    known = targets >= 0
    positive = ((targets == 1) & known).sum(axis=0)
    negative = ((targets == 0) & known).sum(axis=0)
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError('每个纳入 CBL 的概念在训练集中都必须有阳性和阴性标注')
    return torch.from_numpy((negative / positive).astype(np.float32))


def annotated_patient_weights(metadata, targets):
    """只保留至少一个已知概念的图片，并令每位患者总采样权重相等。"""
    annotated = (targets >= 0).any(axis=1)
    patients = metadata.loc[annotated, 'patient_id']
    patient_sizes = patients.value_counts()
    weights = patients.map(lambda value: 1.0 / patient_sizes[value]).to_numpy(np.float32)
    return annotated, weights


def masked_bce(logits, targets, pos_weight):
    """只在 concept_label 为0或1的位置计算多标签 BCE。"""
    known = targets >= 0
    safe_targets = targets.clamp_min(0)
    losses = F.binary_cross_entropy_with_logits(
        logits, safe_targets, pos_weight=pos_weight, reduction='none',
    )
    return losses, known


def make_train_loader(features, metadata, targets, batch_size, seed):
    """建立患者平衡的概念标注训练 DataLoader。"""
    annotated, weights = annotated_patient_weights(metadata, targets)
    dataset = TensorDataset(
        torch.from_numpy(features[annotated]),
        torch.from_numpy(targets[annotated]),
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


@torch.no_grad()
def validation_loss(cbl, features, metadata, targets, pos_weight, batch_size, device):
    """按患者平衡计算已标注验证图片的 masked BCE。"""
    annotated, weights = annotated_patient_weights(metadata, targets)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features[annotated]),
            torch.from_numpy(targets[annotated]),
            torch.from_numpy(weights),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    numerator = 0.0
    denominator = 0.0
    cbl.eval()
    for batch_features, batch_targets, sample_weights in loader:
        logits = cbl(batch_features.to(device))
        losses, known = masked_bce(
            logits, batch_targets.to(device), pos_weight,
        )
        entry_weights = sample_weights.to(device)[:, None] * known
        numerator += (losses * entry_weights).sum().item()
        denominator += entry_weights.sum().item()
    return numerator / denominator


def train_cbl(split_data, catalog, output_dir, args, device):
    """用训练概念标签拟合 CBL，并按患者平衡验证损失早停。"""
    train_features, train_metadata, train_targets = split_data['train']
    val_features, val_metadata, val_targets = split_data['val']
    train_loader = make_train_loader(
        train_features, train_metadata, train_targets, args.batch_size, args.seed,
    )
    pos_weight = concept_pos_weight(train_targets).to(device)
    cbl = ConceptBottleneckLayer(INPUT_DIM, len(catalog)).to(device)
    optimizer = torch.optim.Adam(cbl.parameters(), lr=args.learning_rate)
    history = []
    best_loss = np.inf
    stale_epochs = 0
    best_path = os.path.join(output_dir, 'cbl_best.pth')

    for epoch in range(1, args.epochs + 1):
        cbl.train()
        train_loss = 0.0
        known_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses, known = masked_bce(cbl(features), targets, pos_weight)
            loss = losses[known].mean()
            loss.backward()
            optimizer.step()
            train_loss += losses[known].sum().item()
            known_count += known.sum().item()

        train_loss /= known_count
        val_loss = validation_loss(
            cbl, val_features, val_metadata, val_targets,
            pos_weight, args.batch_size, device,
        )
        history.append({'epoch': epoch, 'train_bce': train_loss, 'val_bce': val_loss})
        print(f'CBL Epoch {epoch:03d}  train BCE={train_loss:.4f}  val BCE={val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save({
                'cbl_state_dict': cbl.state_dict(),
                'input_dim': INPUT_DIM,
                'num_concepts': len(catalog),
                'concept_ids': catalog['concept_id'].tolist(),
                'concept_names': catalog['concept_name'].tolist(),
                'epoch': epoch,
                'val_bce': val_loss,
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'CBL 验证损失连续 {args.patience} 轮未改善，提前停止')
                break

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, 'cbl_training_history.csv'),
        index=False, encoding='utf-8-sig',
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    cbl.load_state_dict(checkpoint['cbl_state_dict'])
    return cbl, checkpoint


@torch.no_grad()
def predict_concepts(cbl, features, batch_size, device):
    """返回所有图片的概念 logits 和 sigmoid 概率。"""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False,
    )
    logits = []
    cbl.eval()
    for (batch,) in loader:
        logits.append(cbl(batch.to(device)).cpu().numpy())
    logits = np.concatenate(logits).astype(np.float32)
    probabilities = 1 / (1 + np.exp(-logits))
    return logits, probabilities


def concept_metrics(probabilities, targets, catalog, split):
    """按概念计算有人工标签图片上的 AUC和0.5阈值指标。"""
    records = []
    for index, concept in catalog.iterrows():
        known = targets[:, index] >= 0
        labels = targets[known, index].astype(int)
        predictions = probabilities[known, index]
        predicted_labels = predictions >= 0.5
        positive_count = int((labels == 1).sum())
        negative_count = int((labels == 0).sum())
        records.append({
            'split': split,
            'concept_id': concept['concept_id'],
            'concept_name': concept['concept_name'],
            'annotated_images': int(known.sum()),
            'positive_images': positive_count,
            'negative_images': negative_count,
            'auc': (
                float(roc_auc_score(labels, predictions))
                if positive_count and negative_count else np.nan
            ),
            'accuracy_at_0.5': (
                float(accuracy_score(labels, predicted_labels))
                if len(labels) else np.nan
            ),
            'sensitivity_at_0.5': (
                float(recall_score(labels, predicted_labels))
                if positive_count else np.nan
            ),
            'specificity_at_0.5': (
                ((predicted_labels == 0) & (labels == 0)).sum() / (labels == 0).sum()
                if negative_count else np.nan
            ),
            'precision_at_0.5': (
                float(precision_score(labels, predicted_labels, zero_division=0))
                if len(labels) else np.nan
            ),
        })
    return records


def ncc95(standardized_logits, coefficients):
    """计算覆盖95%绝对分类贡献所需的平均概念数。"""
    contributions = np.abs(standardized_logits * coefficients[None, :])
    ordered = np.sort(contributions, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    target = 0.95 * cumulative[:, -1:]
    counts = (cumulative < target).sum(axis=1) + 1
    counts[cumulative[:, -1] <= 1e-12] = 0
    return float(counts.mean())


def select_threshold(labels, probabilities):
    """在验证集 Sensitivity>=0.90 下选择 Specificity 最高阈值。"""
    best = None
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        predictions = probabilities >= threshold
        sensitivity = ((predictions == 1) & (labels == 1)).sum() / (labels == 1).sum()
        specificity = ((predictions == 0) & (labels == 0)).sum() / (labels == 0).sum()
        candidate = (specificity, threshold, sensitivity)
        if sensitivity >= 0.90 and (best is None or candidate > best):
            best = candidate
    return float(best[1])


def classification_metrics(metadata, probabilities, threshold):
    """返回图像级和患者级癌/非癌指标。"""
    labels = metadata['label'].to_numpy(dtype=int)
    predictions = probabilities >= threshold
    results = {
        'auc': float(roc_auc_score(labels, probabilities)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'sensitivity': float(recall_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions)),
    }
    results['specificity'] = float(
        ((predictions == 0) & (labels == 0)).sum() / (labels == 0).sum()
    )
    patient = pd.DataFrame({
        'patient_id': metadata['patient_id'],
        'label': labels,
        'probability': probabilities,
    }).groupby('patient_id').agg(
        label=('label', 'first'), probability=('probability', 'max'),
    )
    patient_labels = patient['label'].to_numpy(dtype=int)
    patient_probabilities = patient['probability'].to_numpy()
    patient_predictions = patient_probabilities >= threshold
    results['patient_auc'] = float(roc_auc_score(patient_labels, patient_probabilities))
    results['patient_accuracy'] = float(
        accuracy_score(patient_labels, patient_predictions)
    )
    results['patient_sensitivity'] = float(
        recall_score(patient_labels, patient_predictions)
    )
    results['patient_specificity'] = float(
        ((patient_predictions == 0) & (patient_labels == 0)).sum()
        / (patient_labels == 0).sum()
    )
    return results


def train_sparse_classifier(concept_logits, split_data, catalog, args):
    """用全部训练图片拟合 elastic-net 分类器，并在验证集选择稀疏度。"""
    train_logits = concept_logits['train']
    mean = train_logits.mean(axis=0)
    std = train_logits.std(axis=0).clip(min=1e-6)
    standardized = {
        split: (values - mean) / std for split, values in concept_logits.items()
    }
    train_labels = split_data['train'][1]['label'].to_numpy(dtype=int)
    val_labels = split_data['val'][1]['label'].to_numpy(dtype=int)
    candidates = []

    for c_value in args.classifier_c_grid:
        classifier = LogisticRegression(
            solver='saga', l1_ratio=0.99,
            C=c_value, max_iter=10000, random_state=args.seed,
        )
        classifier.fit(standardized['train'], train_labels)
        val_probability = classifier.predict_proba(standardized['val'])[:, 1]
        candidates.append({
            'C': c_value,
            'val_auc': roc_auc_score(val_labels, val_probability),
            'ncc95': ncc95(standardized['val'], classifier.coef_[0]),
            'nonzero_weights': int((np.abs(classifier.coef_[0]) > 1e-8).sum()),
            'classifier': classifier,
        })

    eligible = [item for item in candidates if item['ncc95'] <= args.target_ncc]
    if eligible:
        selected = max(eligible, key=lambda item: (item['val_auc'], -item['ncc95']))
    else:
        selected = min(candidates, key=lambda item: (item['ncc95'], -item['val_auc']))
    classifier = selected['classifier']
    best = {key: value for key, value in selected.items() if key != 'classifier'}
    best['target_ncc'] = args.target_ncc
    best['target_met'] = bool(eligible)
    search = [
        {key: value for key, value in item.items() if key != 'classifier'}
        for item in candidates
    ]
    val_probability = classifier.predict_proba(standardized['val'])[:, 1]
    threshold = select_threshold(val_labels, val_probability)
    return classifier, mean, std, standardized, threshold, search, best


def save_predictions(
    split, metadata, probabilities, standardized_logits, classifier,
    threshold, catalog, output_dir,
):
    """保存概念概率、癌概率和每张图的主要概念贡献。"""
    output = metadata.copy()
    for index, concept in catalog.iterrows():
        output[f'concept_{concept.concept_id}_{concept.concept_name}'] = probabilities[:, index]
    cancer_probability = classifier.predict_proba(standardized_logits)[:, 1]
    output['mcbm_cancer_probability'] = cancer_probability
    output['mcbm_pred_label'] = (cancer_probability >= threshold).astype(int)
    contributions = standardized_logits * classifier.coef_[0][None, :]
    names = catalog['concept_name'].tolist()
    top_names = []
    top_values = []
    for row in contributions:
        order = np.argsort(np.abs(row))[::-1][:5]
        top_names.append(';'.join(names[index] for index in order))
        top_values.append(';'.join(f'{row[index]:.6f}' for index in order))
    output['top_contributing_concepts'] = top_names
    output['top_concept_contributions'] = top_values
    output.to_csv(
        os.path.join(output_dir, f'{split}_predictions.csv'),
        index=False, encoding='utf-8-sig',
    )
    return cancer_probability


def main():
    """完成概念监督、稀疏分类器训练和锁定 split 评估。"""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.experiment is None:
        args.experiment = f'cbl_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir = os.path.join(OUTPUT_ROOT, args.experiment)
    os.makedirs(output_dir, exist_ok=False)
    catalog = load_catalog(args.concept_catalog)
    catalog.to_csv(
        os.path.join(output_dir, 'concept_catalog_snapshot.csv'),
        index=False, encoding='utf-8-sig',
    )
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    splits = ['train', 'val'] + (['test'] if args.include_test else [])
    split_data = {}
    concept_ids = catalog['concept_id'].tolist()
    for split in splits:
        features, metadata = load_feature_split(args.sae_run, split)
        targets = load_concept_targets(args.annotations, metadata, concept_ids)
        split_data[split] = (features, metadata, targets)

    cbl, checkpoint = train_cbl(split_data, catalog, output_dir, args, device)
    all_logits = {}
    all_probabilities = {}
    concept_metric_records = []
    for split, (features, _, targets) in split_data.items():
        logits, probabilities = predict_concepts(cbl, features, args.batch_size, device)
        all_logits[split] = logits
        all_probabilities[split] = probabilities
        concept_metric_records.extend(concept_metrics(
            probabilities, targets, catalog, split,
        ))
    pd.DataFrame(concept_metric_records).to_csv(
        os.path.join(output_dir, 'concept_metrics.csv'),
        index=False, encoding='utf-8-sig',
    )

    classifier, mean, std, standardized, threshold, search, best = (
        train_sparse_classifier(all_logits, split_data, catalog, args)
    )
    joblib.dump({
        'classifier': classifier,
        'concept_mean': mean,
        'concept_std': std,
        'threshold': threshold,
        'concept_ids': concept_ids,
        'concept_names': catalog['concept_name'].tolist(),
    }, os.path.join(output_dir, 'sparse_classifier.joblib'))
    pd.DataFrame(search).to_csv(
        os.path.join(output_dir, 'classifier_search.csv'),
        index=False, encoding='utf-8-sig',
    )
    for item in search:
        print(
            f'分类器候选 C={item["C"]:g}  AUC={item["val_auc"]:.4f}  '
            f'NCC95={item["ncc95"]:.2f}  非零权重={item["nonzero_weights"]}'
        )
    if not best['target_met']:
        print(f'没有候选达到目标 NCC95<={args.target_ncc:g}，已选择 NCC95 最低者')

    metrics = {
        'cbl_best_epoch': int(checkpoint['epoch']),
        'classifier_selection': best,
        'clinical_threshold_from_val': threshold,
        'splits': {},
    }
    for split, (_, metadata, _) in split_data.items():
        cancer_probability = save_predictions(
            split, metadata, all_probabilities[split], standardized[split],
            classifier, threshold, catalog, output_dir,
        )
        metrics['splits'][split] = classification_metrics(
            metadata, cancer_probability, threshold,
        )
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f'CBL 最佳 epoch: {checkpoint["epoch"]}')
    print(f'概念数: {len(catalog)}  稀疏分类器非零权重: {best["nonzero_weights"]}')
    print(f'验证集确定阈值: {threshold:.4f}')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()
