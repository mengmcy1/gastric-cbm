"""从冻结的 ResNet50 整图特征中训练 SAE，并生成候选概念概览。"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet50


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
DEFAULT_RUN_DIR = os.path.join(
    PROJECT_DIR, '结果', '去偏重训练_v1', 'expA_full_resnet50_seed42',
)
DEFAULT_SPLIT_CSV = os.path.join(DEFAULT_RUN_DIR, 'frozen_split_snapshot.csv')
DEFAULT_WEIGHT_PATH = os.path.join(DEFAULT_RUN_DIR, 'resnet50_debiased_best.pth')
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROJECT_DIR, '结果', 'SAE', '去偏重训练_v1', 'resnet50',
)

INPUT_DIM = 2048
DEFAULT_IMAGE_THRESHOLD = 0.15484211
DEFAULT_PATIENT_THRESHOLD = 0.3849397003650665
ACTIVE_EPS = 1e-8
MAX_PRUNING_CURVE_POINTS = 100
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def parse_args(argv=None):
    """读取正式训练和小规模演示参数。"""
    parser = argparse.ArgumentParser(description='ResNet50 SAE 概念发现')
    parser.add_argument('--demo', action='store_true', help='少量患者快速验证完整流程')
    parser.add_argument('--include-test', action='store_true', help='锁定方案后投影测试集')
    parser.add_argument('--experiment', type=str, default=None, help='实验目录名；默认使用时间戳')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='裁剪后图片根目录')
    parser.add_argument('--split-csv', default=DEFAULT_SPLIT_CSV, help='冻结患者划分CSV')
    parser.add_argument('--weight-path', default=DEFAULT_WEIGHT_PATH, help='冻结ResNet50权重')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help='SAE结果根目录')
    parser.add_argument('--image-threshold', type=float, default=DEFAULT_IMAGE_THRESHOLD)
    parser.add_argument('--patient-threshold', type=float, default=DEFAULT_PATIENT_THRESHOLD)
    parser.add_argument('--hidden-dim', type=int, default=None, help='SAE 字典大小')
    parser.add_argument('--lambda-l1', type=float, default=5e-4,
                        help='隐藏激活 L1 权重；正式方案锁定为5e-4')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='SAE 学习率')
    parser.add_argument('--epochs', type=int, default=None, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=None, help='验证损失早停轮数')
    parser.add_argument('--image-batch-size', type=int, default=32, help='ResNet50 特征提取 batch')
    parser.add_argument('--sae-batch-size', type=int, default=32, help='SAE 特征 batch')
    parser.add_argument('--num-workers', type=int, default=None, help='图片读取进程数')
    parser.add_argument('--demo-patients-per-class', type=int, default=4,
                        help='demo 每个 split、每类抽取患者数')
    parser.add_argument('--overview-features', type=int, default=None,
                        help='生成概览的 feature 数')
    parser.add_argument('--top-images', type=int, default=6,
                        help='每个 feature 展示的不同患者图片数')
    parser.add_argument('--pruning-min-active-patients', type=int, default=5,
                        help='候选 feature 至少激活的训练患者数')
    parser.add_argument('--pruning-ce-tolerance', type=float, default=0.01,
                        help='剪枝后 recovered CE 允许的最大下降')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args(argv)


def finalize_args(args):
    """根据 demo 或正式模式补齐默认参数。"""
    if args.hidden_dim is None:
        args.hidden_dim = 128 if args.demo else 512
    if args.epochs is None:
        args.epochs = 3 if args.demo else 1000
    if args.patience is None:
        args.patience = 2 if args.demo else 50
    if args.num_workers is None:
        args.num_workers = 0 if args.demo else 4
    if args.overview_features is None:
        args.overview_features = 6 if args.demo else 20
    prefix = 'demo' if args.demo else 'formal'
    if args.experiment is None:
        args.experiment = f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    for name in ['data_dir', 'split_csv', 'weight_path', 'output_root']:
        setattr(args, name, os.path.abspath(getattr(args, name)))
    for name in ['image_threshold', 'patient_threshold']:
        if not 0 < getattr(args, name) < 1:
            raise ValueError(f'{name}必须位于(0, 1)')
    return args


def seed_everything(seed):
    """固定随机数，保证患者抽样和 SAE 训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_source(center_value):
    """从清单的结构化中心字段派生省人民/外院和医院名称。"""
    center = str(center_value).strip()
    if not center or center.lower() == 'nan':
        raise ValueError('center字段存在空值')
    if center == '武大省人民':
        return '省人民', center
    return '外院', center


def load_split_dataframe(args):
    """读取固定患者划分，并在 demo 模式按患者抽样。"""
    dataframe = pd.read_csv(args.split_csv, encoding='utf-8-sig')
    dataframe = dataframe.rename(columns={'瘤变标签': 'label'})
    if 'image_relpath' not in dataframe:
        for column in ['processed_path', '图片名字', 'image_path']:
            if column in dataframe:
                dataframe['image_relpath'] = dataframe[column]
                break
    if 'image_relpath' not in dataframe:
        raise ValueError('划分CSV缺少 image_relpath/processed_path/图片名字/image_path')
    if 'center' not in dataframe:
        raise ValueError('划分CSV缺少center字段')
    source = dataframe['center'].apply(derive_source)
    dataframe['domain'] = source.str[0]
    dataframe['hospital'] = source.str[1]

    if args.demo:
        selected = []
        for split in ['train', 'val']:
            split_data = dataframe[dataframe['split'] == split]
            patients = split_data[['patient_id', 'label']].drop_duplicates()
            for _, group in patients.groupby('label'):
                selected.extend(
                    group.sample(
                        n=min(args.demo_patients_per_class, len(group)),
                        random_state=args.seed,
                    )['patient_id'].tolist()
                )
        dataframe = dataframe[dataframe['patient_id'].isin(selected)].copy()

    requested_splits = ['train', 'val']
    if args.include_test and not args.demo:
        requested_splits.append('test')
    dataframe = dataframe[dataframe['split'].isin(requested_splits)].reset_index(drop=True)
    missing_splits = set(requested_splits) - set(dataframe['split'])
    if missing_splits:
        raise ValueError(f'冻结划分缺少split: {sorted(missing_splits)}')
    if dataframe.groupby('patient_id')['split'].nunique().max() > 1:
        raise ValueError('同一患者出现在多个split中')
    if dataframe.groupby('patient_id')['label'].nunique().max() > 1:
        raise ValueError('同一患者存在多个分类标签')
    for split in requested_splits:
        if set(dataframe.loc[dataframe['split'] == split, 'label']) != {0, 1}:
            raise ValueError(f'{split}未同时包含两个标签')
    return dataframe


class ImageFeatureDataset(Dataset):
    """读取胃镜图片；每个返回值保留对应 DataFrame 行号。"""

    def __init__(self, dataframe, data_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index]['image_relpath']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_dir, image_path)
        image = Image.open(image_path).convert('RGB')
        return EVAL_TRANSFORM(image), index


class ActivationCapture:
    """保存 ResNet50 layer4 最后残差块的输出 [B,2048,7,7]。"""

    def __init__(self, layer):
        self.output = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


def load_resnet50(device, weight_path=DEFAULT_WEIGHT_PATH):
    """恢复正式 ResNet50 权重并冻结所有参数。"""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def extract_split_features(model, capture, dataframe, split, args, device, feature_dir):
    """提取一个 split 的 GAP 特征、原模型预测和图片元数据。"""
    split_data = dataframe[dataframe['split'] == split].reset_index(drop=True)
    loader = DataLoader(
        ImageFeatureDataset(split_data, args.data_dir),
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    features = []
    probabilities = []

    for batch_index, (images, _) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        gap_features = capture.output.mean(dim=(2, 3))
        features.append(gap_features.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        print(f'提取 {split}: [{batch_index}/{len(loader)}]')

    feature_array = np.concatenate(features).astype(np.float32)
    probability_array = np.concatenate(probabilities).astype(np.float32)
    split_data['cancer_probability'] = probability_array
    split_data['pred_label'] = (probability_array >= args.image_threshold).astype(int)
    split_data['confusion_type'] = np.select(
        [
            (split_data['label'] == 1) & (split_data['pred_label'] == 1),
            (split_data['label'] == 0) & (split_data['pred_label'] == 0),
            (split_data['label'] == 0) & (split_data['pred_label'] == 1),
            (split_data['label'] == 1) & (split_data['pred_label'] == 0),
        ],
        ['TP', 'TN', 'FP', 'FN'],
        default='',
    )

    np.save(os.path.join(feature_dir, f'{split}_gap_features.npy'), feature_array)
    split_data.to_csv(
        os.path.join(feature_dir, f'{split}_metadata.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    print(f'{split}: {len(split_data)}张，特征形状={feature_array.shape}')
    return feature_array, split_data


class SparseAutoencoder(nn.Module):
    """M-CBM 风格的 Linear-ReLU encoder 和线性 decoder。"""

    def __init__(self, input_dim, hidden_dim, feature_center):
        super().__init__()
        self.decoder_bias = nn.Parameter(feature_center.clone())
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.decoder_weight, a=0, nonlinearity='relu')
        self.normalize_decoder()
        with torch.no_grad():
            # encoder 与 decoder 仅初始化同向，训练时是相互独立的参数。
            self.encoder.weight.copy_(self.decoder_weight)
            self.encoder.bias.zero_()

    def encode(self, features):
        """输入 [B,2048]，返回非负稀疏激活 [B,d_hidden]。"""
        return torch.relu(self.encoder(features - self.decoder_bias))

    def decode(self, hidden):
        """把隐藏激活重构回 [B,2048]。"""
        return hidden @ self.decoder_weight + self.decoder_bias

    def forward(self, features):
        hidden = self.encode(features)
        return self.decode(hidden), hidden

    @torch.no_grad()
    def normalize_decoder(self):
        """固定每个 decoder 方向的 L2 范数，避免缩放绕过 L1。"""
        self.decoder_weight.div_(
            self.decoder_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        )


def patient_class_weights(metadata):
    """返回类别平衡且每位患者总权重相等的图片权重。"""
    patient_sizes = metadata['patient_id'].value_counts()
    patient_table = metadata[['patient_id', 'label']].drop_duplicates()
    class_patient_counts = patient_table['label'].value_counts()
    return np.asarray([
        1.0 / (patient_sizes[row.patient_id] * class_patient_counts[row.label])
        for row in metadata.itertuples(index=False)
    ], dtype=np.float32)


def make_patient_balanced_loader(features, metadata, args):
    """令两类总采样概率相等，且同类中每位患者权重相等。"""
    weights = patient_class_weights(metadata)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(features),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=args.sae_batch_size,
        sampler=sampler,
    )


def make_feature_loader(features, batch_size):
    """按固定顺序读取缓存特征。"""
    return DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )


def sae_loss(reconstructed, features, hidden, lambda_l1):
    """逐元素 MSE 与隐藏激活 L1（对齐 M-CBM 官方损失尺度）。"""
    reconstruction = (reconstructed - features).pow(2).mean()
    sparsity = hidden.abs().sum(dim=1).mean()
    return reconstruction + lambda_l1 * sparsity, reconstruction, sparsity


@torch.no_grad()
def evaluate_sae_loss(sae, features, metadata, lambda_l1, batch_size, device):
    """按患者和类别平衡返回验证损失、L1及非零激活数。"""
    weights = patient_class_weights(metadata)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(weights)),
        batch_size=batch_size,
        shuffle=False,
    )
    totals = np.zeros(4, dtype=np.float64)
    weight_total = 0.0
    sae.eval()
    for features, sample_weights in loader:
        features = features.to(device)
        reconstructed, hidden = sae(features)
        reconstruction = (reconstructed - features).pow(2).mean(dim=1)
        sparsity = hidden.abs().sum(dim=1)
        l0 = (hidden > ACTIVE_EPS).sum(dim=1).float()
        sample_weights = sample_weights.numpy()
        values = torch.stack([
            reconstruction + lambda_l1 * sparsity,
            reconstruction,
            sparsity,
            l0,
        ], dim=1).cpu().numpy()
        totals += (values * sample_weights[:, None]).sum(axis=0)
        weight_total += sample_weights.sum()
    return totals / weight_total


def train_sae(
    train_features, train_metadata, val_features, val_metadata,
    model_dir, args, device,
):
    """患者平衡训练 SAE，并按验证总损失保存最佳权重。"""
    train_loader = make_patient_balanced_loader(train_features, train_metadata, args)
    center_weights = patient_class_weights(train_metadata)
    center = torch.from_numpy(np.average(
        train_features, axis=0, weights=center_weights,
    ).astype(np.float32)).to(device)
    sae = SparseAutoencoder(INPUT_DIM, args.hidden_dim, center).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    history = []
    best_loss = np.inf
    stale_epochs = 0
    best_path = os.path.join(model_dir, 'sae_best.pth')

    for epoch in range(1, args.epochs + 1):
        sae.train()
        train_totals = np.zeros(4, dtype=np.float64)
        sample_count = 0
        for (features,) in train_loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, hidden = sae(features)
            total, reconstruction, sparsity = sae_loss(
                reconstructed, features, hidden, args.lambda_l1,
            )
            total.backward()
            with torch.no_grad():
                dw = sae.decoder_weight
                g = dw.grad
                dot = (g * dw).sum(dim=1, keepdim=True)
                norm_sq = (dw * dw).sum(dim=1, keepdim=True).clamp_min(1e-12)
                g -= dot / norm_sq * dw
            optimizer.step()
            sae.normalize_decoder()
            count = len(features)
            train_totals += np.asarray([
                total.item(), reconstruction.item(), sparsity.item(),
                (hidden > ACTIVE_EPS).sum(dim=1).float().mean().item(),
            ]) * count
            sample_count += count

        train_metrics = train_totals / sample_count
        val_metrics = evaluate_sae_loss(
            sae, val_features, val_metadata, args.lambda_l1,
            args.sae_batch_size, device,
        )
        history.append({
            'epoch': epoch,
            'train_total_loss': train_metrics[0],
            'train_l2': train_metrics[1],
            'train_l1': train_metrics[2],
            'train_weighted_l1': args.lambda_l1 * train_metrics[2],
            'train_l0': train_metrics[3],
            'val_total_loss': val_metrics[0],
            'val_l2': val_metrics[1],
            'val_l1': val_metrics[2],
            'val_weighted_l1': args.lambda_l1 * val_metrics[2],
            'val_l0': val_metrics[3],
        })
        print(
            f'Epoch {epoch:03d}  '
            f'train MSE={train_metrics[1]:.4f} '
            f'λL1={args.lambda_l1 * train_metrics[2]:.4f} '
            f'L0={train_metrics[3]:.2f}  '
            f'val MSE={val_metrics[1]:.4f} '
            f'λL1={args.lambda_l1 * val_metrics[2]:.4f} '
            f'L0={val_metrics[3]:.2f}'
        )

        if val_metrics[0] < best_loss:
            best_loss = val_metrics[0]
            stale_epochs = 0
            torch.save({
                'sae_state_dict': sae.state_dict(),
                'input_dim': INPUT_DIM,
                'hidden_dim': args.hidden_dim,
                'lambda_l1': args.lambda_l1,
                'epoch': epoch,
                'val_total_loss': best_loss,
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f'验证损失连续 {args.patience} 轮未改善，提前停止')
                break

    pd.DataFrame(history).to_csv(
        os.path.join(model_dir, 'training_history.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    sae.load_state_dict(checkpoint['sae_state_dict'])
    return sae, checkpoint


@torch.no_grad()
def project_features(sae, features, batch_size, device):
    """使用冻结 SAE 返回重构特征和隐藏激活。"""
    reconstructed_all = []
    activations_all = []
    sae.eval()
    for (batch,) in make_feature_loader(features, batch_size):
        reconstructed, hidden = sae(batch.to(device))
        reconstructed_all.append(reconstructed.cpu().numpy())
        activations_all.append(hidden.cpu().numpy())
    return (
        np.concatenate(reconstructed_all).astype(np.float32),
        np.concatenate(activations_all).astype(np.float32),
    )


@torch.no_grad()
def reconstruct_masked_features(sae, activations, kept_mask, batch_size, device):
    """仅用保留的 SAE feature 重构 [N,2048] 特征。"""
    mask = torch.as_tensor(kept_mask, dtype=torch.float32, device=device)
    reconstructed_all = []
    for (batch,) in make_feature_loader(activations, batch_size):
        reconstructed_all.append(sae.decode(batch.to(device) * mask).cpu().numpy())
    return np.concatenate(reconstructed_all).astype(np.float32)


def softmax_numpy(logits):
    """计算二分类 softmax 概率。"""
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def cross_entropy_numpy(logits, labels):
    """返回平均交叉熵。"""
    probabilities = softmax_numpy(logits)
    return float(-np.log(probabilities[np.arange(len(labels)), labels] + 1e-12).mean())


def weighted_head_metrics(features, reconstructed, metadata, fc_weight, fc_bias):
    """按患者和类别平衡评估重构特征的分类恢复。"""
    labels = metadata['label'].to_numpy(dtype=int)
    weights = patient_class_weights(metadata).astype(np.float64)
    weights /= weights.sum()
    original_logits = features @ fc_weight.T + fc_bias
    reconstructed_logits = reconstructed @ fc_weight.T + fc_bias
    zero_logits = np.zeros_like(features) @ fc_weight.T + fc_bias
    original_probabilities = softmax_numpy(original_logits)
    reconstructed_probabilities = softmax_numpy(reconstructed_logits)
    zero_probabilities = softmax_numpy(zero_logits)
    row_indices = np.arange(len(labels))
    original_ce = float(np.sum(
        -np.log(original_probabilities[row_indices, labels] + 1e-12) * weights
    ))
    reconstructed_ce = float(np.sum(
        -np.log(reconstructed_probabilities[row_indices, labels] + 1e-12) * weights
    ))
    zero_ce = float(np.sum(
        -np.log(zero_probabilities[row_indices, labels] + 1e-12) * weights
    ))
    recovered_ce = 1 - (reconstructed_ce - original_ce) / (
        zero_ce - original_ce + 1e-12
    )
    reconstructed_labels = reconstructed_logits.argmax(axis=1)
    original_labels = original_logits.argmax(axis=1)
    return {
        'recovered_cross_entropy': float(recovered_ce),
        'reconstructed_cross_entropy': reconstructed_ce,
        'reconstructed_accuracy': float(np.sum(
            (reconstructed_labels == labels) * weights
        )),
        'prediction_agreement': float(np.sum(
            (reconstructed_labels == original_labels) * weights
        )),
        'cancer_probability_mae': float(np.sum(
            np.abs(reconstructed_probabilities[:, 1] - original_probabilities[:, 1])
            * weights
        )),
    }


def patient_active_counts(activations, metadata):
    """统计每个 feature 激活的不同患者数。"""
    patient_codes, patients = pd.factorize(metadata['patient_id'], sort=True)
    patient_max = np.zeros((len(patients), activations.shape[1]), dtype=np.float32)
    np.maximum.at(patient_max, patient_codes, activations)
    return (patient_max > ACTIVE_EPS).sum(axis=0).astype(np.int64)


def pruning_thresholds(active_patient_counts, min_active_patients):
    """返回按训练患者激活数剪枝的候选阈值。"""
    minimum_threshold = min_active_patients - 1
    maximum_count = int(active_patient_counts.max())
    thresholds = {minimum_threshold}
    thresholds.update(
        int(count) for count in np.unique(active_patient_counts)
        if minimum_threshold <= int(count) < maximum_count
    )
    return sorted(thresholds)


def run_feature_pruning(
    sae, train_activations, train_metadata, val_features, val_activations,
    val_metadata, fc_weight, fc_bias, args, device,
):
    """以训练患者覆盖率剪枝，用验证 recovered CE 选严格阈值。"""
    active_counts = patient_active_counts(train_activations, train_metadata)
    all_mask = np.ones(args.hidden_dim, dtype=bool)
    baseline_reconstructed = reconstruct_masked_features(
        sae, val_activations, all_mask, args.sae_batch_size, device,
    )
    baseline_metrics = weighted_head_metrics(
        val_features, baseline_reconstructed, val_metadata, fc_weight, fc_bias,
    )
    thresholds = pruning_thresholds(
        active_counts, args.pruning_min_active_patients,
    )
    curve = []
    selected = None
    first_failed = False
    post_failure_indices = None

    for threshold_index, threshold in enumerate(thresholds):
        if post_failure_indices is not None and threshold_index not in post_failure_indices:
            continue
        kept_mask = active_counts > threshold
        kept_count = int(kept_mask.sum())
        if kept_count == 0:
            continue
        reconstructed = reconstruct_masked_features(
            sae, val_activations, kept_mask, args.sae_batch_size, device,
        )
        current = weighted_head_metrics(
            val_features, reconstructed, val_metadata, fc_weight, fc_bias,
        )
        recovered_drop = (
            baseline_metrics['recovered_cross_entropy']
            - current['recovered_cross_entropy']
        )
        if not first_failed and recovered_drop <= args.pruning_ce_tolerance:
            selected = {
                'threshold': int(threshold),
                'kept_mask': kept_mask.copy(),
                'metrics': current,
                'recovered_ce_drop': float(recovered_drop),
            }
        elif not first_failed:
            first_failed = True
            remaining = list(range(threshold_index + 1, len(thresholds)))
            if len(remaining) > MAX_PRUNING_CURVE_POINTS:
                positions = np.linspace(
                    0, len(remaining) - 1,
                    MAX_PRUNING_CURVE_POINTS, dtype=int,
                )
                remaining = [remaining[position] for position in positions]
            post_failure_indices = set(remaining)
        curve.append({
            'active_patient_threshold': int(threshold),
            'kept_feature_count': kept_count,
            'pruned_feature_count': int(args.hidden_dim - kept_count),
            'recovered_cross_entropy': current['recovered_cross_entropy'],
            'recovered_ce_drop': float(recovered_drop),
            'reconstructed_accuracy': current['reconstructed_accuracy'],
            'prediction_agreement': current['prediction_agreement'],
            'cancer_probability_mae': current['cancer_probability_mae'],
        })

    if selected is None:
        fallback_threshold = args.pruning_min_active_patients - 1
        fallback_mask = active_counts > fallback_threshold
        if not fallback_mask.any():
            fallback_threshold = -1
            fallback_mask = all_mask
        fallback_reconstructed = reconstruct_masked_features(
            sae, val_activations, fallback_mask,
            args.sae_batch_size, device,
        )
        fallback_metrics = weighted_head_metrics(
            val_features, fallback_reconstructed,
            val_metadata, fc_weight, fc_bias,
        )
        selected = {
            'threshold': int(fallback_threshold),
            'kept_mask': fallback_mask,
            'metrics': fallback_metrics,
            'recovered_ce_drop': float(
                baseline_metrics['recovered_cross_entropy']
                - fallback_metrics['recovered_cross_entropy']
            ),
        }
    kept_indices = np.flatnonzero(selected['kept_mask'])
    summary = {
        'total_feature_count': int(args.hidden_dim),
        'kept_feature_count': int(len(kept_indices)),
        'pruned_feature_count': int(args.hidden_dim - len(kept_indices)),
        'selected_active_patient_threshold': int(selected['threshold']),
        'minimum_active_patients': int(args.pruning_min_active_patients),
        'recovered_ce_tolerance': float(args.pruning_ce_tolerance),
        'baseline_recovered_cross_entropy': baseline_metrics['recovered_cross_entropy'],
        'selected_recovered_cross_entropy': selected['metrics']['recovered_cross_entropy'],
        'recovered_ce_drop': selected['recovered_ce_drop'],
        'selection_within_tolerance': bool(
            selected['recovered_ce_drop'] <= args.pruning_ce_tolerance
        ),
        'selected_reconstructed_accuracy': selected['metrics']['reconstructed_accuracy'],
        'selected_prediction_agreement': selected['metrics']['prediction_agreement'],
        'selected_cancer_probability_mae': selected['metrics']['cancer_probability_mae'],
    }
    return kept_indices, active_counts, curve, summary


def reconstruction_metrics(
    features, reconstructed, metadata, fc_weight, fc_bias,
    patient_threshold=DEFAULT_PATIENT_THRESHOLD,
):
    """评估 SAE 重构质量和原分类头性能恢复情况。"""
    labels = metadata['label'].to_numpy(dtype=int)
    original_logits = features @ fc_weight.T + fc_bias
    reconstructed_logits = reconstructed @ fc_weight.T + fc_bias
    zero_logits = np.zeros_like(features) @ fc_weight.T + fc_bias
    original_prob = softmax_numpy(original_logits)[:, 1]
    reconstructed_prob = softmax_numpy(reconstructed_logits)[:, 1]
    original_ce = cross_entropy_numpy(original_logits, labels)
    reconstructed_ce = cross_entropy_numpy(reconstructed_logits, labels)
    zero_ce = cross_entropy_numpy(zero_logits, labels)
    recovered_ce = 1 - (reconstructed_ce - original_ce) / (zero_ce - original_ce)
    patient_results = pd.DataFrame({
        'patient_id': metadata['patient_id'].to_numpy(),
        'label': labels,
        'original_probability': original_prob,
        'reconstructed_probability': reconstructed_prob,
    }).groupby('patient_id').agg(
        label=('label', 'first'),
        original_probability=('original_probability', 'mean'),
        reconstructed_probability=('reconstructed_probability', 'mean'),
    )
    patient_labels = patient_results['label'].to_numpy(dtype=int)
    patient_original = patient_results['original_probability'].to_numpy()
    patient_reconstructed = patient_results['reconstructed_probability'].to_numpy()
    return {
        'feature_mse': float(np.mean((reconstructed - features) ** 2)),
        'feature_cosine': float(np.mean(
            np.sum(features * reconstructed, axis=1)
            / (np.linalg.norm(features, axis=1)
               * np.linalg.norm(reconstructed, axis=1) + 1e-12)
        )),
        'original_accuracy': float(accuracy_score(labels, original_logits.argmax(axis=1))),
        'reconstructed_accuracy': float(
            accuracy_score(labels, reconstructed_logits.argmax(axis=1))
        ),
        'original_auc': float(roc_auc_score(labels, original_prob)),
        'reconstructed_auc': float(roc_auc_score(labels, reconstructed_prob)),
        'prediction_agreement': float(np.mean(
            original_logits.argmax(axis=1) == reconstructed_logits.argmax(axis=1)
        )),
        'cancer_probability_mae': float(np.mean(
            np.abs(original_prob - reconstructed_prob)
        )),
        'original_cross_entropy': original_ce,
        'reconstructed_cross_entropy': reconstructed_ce,
        'recovered_cross_entropy': float(recovered_ce),
        'patient_original_auc': float(roc_auc_score(patient_labels, patient_original)),
        'patient_reconstructed_auc': float(
            roc_auc_score(patient_labels, patient_reconstructed)
        ),
        'patient_prediction_agreement_at_locked_threshold': float(np.mean(
            (patient_original >= patient_threshold)
            == (patient_reconstructed >= patient_threshold)
        )),
        'patient_cancer_probability_mae': float(np.mean(
            np.abs(patient_original - patient_reconstructed)
        )),
    }


def build_feature_summary(activations, metadata, sae, fc_weight):
    """按患者汇总每个 SAE feature 的覆盖、来源差异和分类方向。"""
    patient_codes, patients = pd.factorize(metadata['patient_id'], sort=True)
    patient_count = len(patients)
    hidden_dim = activations.shape[1]
    patient_max = np.zeros((patient_count, hidden_dim), dtype=np.float32)
    patient_sum = np.zeros((patient_count, hidden_dim), dtype=np.float32)
    np.maximum.at(patient_max, patient_codes, activations)
    np.add.at(patient_sum, patient_codes, activations)

    patient_meta = metadata[['patient_id', 'label', 'domain']].drop_duplicates(
        'patient_id'
    ).set_index('patient_id').loc[patients]
    total_activation = patient_sum.sum(axis=0)
    max_patient_ratio = patient_sum.max(axis=0) / np.maximum(total_activation, 1e-12)
    decoder = sae.decoder_weight.detach().cpu().numpy()
    cancer_margin = decoder @ (fc_weight[1] - fc_weight[0])
    mean_activation = activations.mean(axis=0)

    summary = pd.DataFrame({
        'feature_id': np.arange(hidden_dim),
        'active_image_count': (activations > ACTIVE_EPS).sum(axis=0),
        'active_patient_count': (patient_max > ACTIVE_EPS).sum(axis=0),
        'activation_density': (activations > ACTIVE_EPS).mean(axis=0),
        'mean_activation': mean_activation,
        'max_activation': activations.max(axis=0),
        'max_patient_contribution_ratio': max_patient_ratio,
        'mean_patient_max_label_1': patient_max[
            patient_meta['label'].to_numpy() == 1
        ].mean(axis=0),
        'mean_patient_max_label_0': patient_max[
            patient_meta['label'].to_numpy() == 0
        ].mean(axis=0),
        'mean_patient_max_provincial': patient_max[
            patient_meta['domain'].to_numpy() == '省人民'
        ].mean(axis=0),
        'mean_patient_max_external': patient_max[
            patient_meta['domain'].to_numpy() == '外院'
        ].mean(axis=0),
        'cancer_margin_direction': cancer_margin,
        'mean_abs_margin_contribution': mean_activation * np.abs(cancer_margin),
    })
    summary['dead_feature'] = summary['active_image_count'] == 0
    for confusion_type in ['TP', 'TN', 'FP', 'FN']:
        type_mask = metadata['confusion_type'].to_numpy() == confusion_type
        if not type_mask.any():
            summary[f'active_patient_count_{confusion_type}'] = 0
            summary[f'mean_patient_max_{confusion_type}'] = np.nan
            continue
        type_codes, type_patients = pd.factorize(
            metadata.loc[type_mask, 'patient_id'], sort=True,
        )
        type_patient_max = np.zeros(
            (len(type_patients), hidden_dim), dtype=np.float32,
        )
        np.maximum.at(type_patient_max, type_codes, activations[type_mask])
        summary[f'active_patient_count_{confusion_type}'] = (
            type_patient_max > ACTIVE_EPS
        ).sum(axis=0)
        summary[f'mean_patient_max_{confusion_type}'] = type_patient_max.mean(axis=0)
    return summary


def load_font(size):
    """使用项目环境已有中文字体。"""
    return ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=size,
    )


def fit_panel(image, size=(320, 240)):
    """保持比例放入固定黑色面板。"""
    image = ImageOps.contain(image.convert('RGB'), size, Image.Resampling.LANCZOS)
    panel = Image.new('RGB', size)
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def heatmap_overlay(original, concept_map):
    """将 [7,7] 概念响应上采样后以红色覆盖原图。"""
    concept_map = np.maximum(concept_map, 0)
    concept_map = concept_map / (concept_map.max() + 1e-12)
    mask = Image.fromarray((concept_map * 255).astype(np.uint8)).resize(
        original.size, Image.Resampling.BICUBIC,
    )
    mask_array = np.asarray(mask, dtype=np.float32) / 255
    image_array = np.asarray(original, dtype=np.float32)
    red = np.zeros_like(image_array)
    red[..., 0] = 255
    alpha = 0.55 * mask_array[..., None]
    overlay = image_array * (1 - alpha) + red * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


@torch.no_grad()
def make_feature_overviews(
    model, capture, sae, activations, metadata, feature_summary,
    overview_dir, args, device,
):
    """为分类贡献较高的 feature 保存不同患者 Top 图片和 encoder 热图。"""
    candidates = feature_summary[
        (feature_summary['active_patient_count'] >= 2)
        & (~feature_summary['dead_feature'])
        & (feature_summary['kept_after_pruning'])
    ].nlargest(args.overview_features, 'mean_abs_margin_contribution')
    title_font = load_font(22)
    text_font = load_font(15)
    records = []

    for feature_row in candidates.itertuples(index=False):
        feature_id = int(feature_row.feature_id)
        order = np.argsort(activations[:, feature_id])[::-1]
        selected = []
        used_patients = set()
        for index in order:
            patient_id = metadata.iloc[index]['patient_id']
            if activations[index, feature_id] <= ACTIVE_EPS:
                break
            if patient_id not in used_patients:
                selected.append(index)
                used_patients.add(patient_id)
            if len(selected) == args.top_images:
                break

        row_height = 300
        panel_size = (320, 240)
        canvas = Image.new('RGB', (640, 70 + row_height * len(selected)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 8),
            f'Feature {feature_id}  患者={feature_row.active_patient_count}  '
            f'癌方向={feature_row.cancer_margin_direction:.4f}',
            fill=(20, 20, 20),
            font=title_font,
        )
        encoder_direction = sae.encoder.weight[feature_id].detach()

        for row_index, index in enumerate(selected):
            item = metadata.iloc[index]
            image_path = item['image_relpath']
            if not os.path.isabs(image_path):
                image_path = os.path.join(args.data_dir, image_path)
            original = Image.open(image_path).convert('RGB')
            model(EVAL_TRANSFORM(original).unsqueeze(0).to(device))
            feature_map = capture.output[0]
            concept_map = torch.relu(
                (feature_map * encoder_direction[:, None, None]).sum(dim=0)
            ).cpu().numpy()
            overlay = heatmap_overlay(original, concept_map)
            y = 70 + row_index * row_height
            canvas.paste(fit_panel(original, panel_size), (0, y))
            canvas.paste(fit_panel(overlay, panel_size), (320, y))
            description = (
                f'{item.patient_id}  label={item.label}  {item.domain}  '
                f'{item.confusion_type}  激活={activations[index, feature_id]:.4f}'
            )
            draw.text((8, y + 245), description, fill=(20, 20, 20), font=text_font)
            records.append({
                'feature_id': feature_id,
                'rank': row_index + 1,
                'image_name': item['image_relpath'],
                'patient_id': item['patient_id'],
                'split': item['split'],
                'label': int(item['label']),
                'domain': item['domain'],
                'hospital': item['hospital'],
                'confusion_type': item['confusion_type'],
                'activation': float(activations[index, feature_id]),
            })

        canvas.save(os.path.join(overview_dir, f'feature_{feature_id:04d}.png'))
        print(f'保存 feature {feature_id} 概览，展示 {len(selected)} 位患者')

    pd.DataFrame(records).to_csv(
        os.path.join(overview_dir, 'top_examples.csv'),
        index=False,
        encoding='utf-8-sig',
    )


def main(argv=None):
    """完成固定划分特征提取、SAE训练、投影、统计和概念概览。"""
    args = finalize_args(parse_args(argv))
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_root, args.experiment)
    os.makedirs(output_dir, exist_ok=False)
    feature_dir = os.path.join(output_dir, '特征缓存')
    model_dir = os.path.join(output_dir, 'SAE模型')
    overview_dir = os.path.join(output_dir, 'feature概览')
    pruning_dir = os.path.join(output_dir, 'feature筛选')
    os.makedirs(feature_dir)
    os.makedirs(model_dir)
    os.makedirs(overview_dir)
    os.makedirs(pruning_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    dataframe = load_split_dataframe(args)
    print(f'设备: {device}')
    print(dataframe.groupby('split').agg(
        patients=('patient_id', 'nunique'), images=('patient_id', 'size')
    ))
    model = load_resnet50(device, args.weight_path)
    capture = ActivationCapture(model.layer4[-1])
    split_features = {}
    split_metadata = {}

    for split in dataframe['split'].drop_duplicates():
        split_features[split], split_metadata[split] = extract_split_features(
            model, capture, dataframe, split, args, device, feature_dir,
        )

    sae, best_checkpoint = train_sae(
        split_features['train'], split_metadata['train'],
        split_features['val'], split_metadata['val'],
        model_dir, args, device,
    )

    fc_weight = model.fc.weight.detach().cpu().numpy()
    fc_bias = model.fc.bias.detach().cpu().numpy()
    all_activations = []
    all_metadata = []
    split_activations = {}
    metrics = {'best_epoch': int(best_checkpoint['epoch']), 'splits': {}}

    for split, features in split_features.items():
        reconstructed, activations = project_features(
            sae, features, args.sae_batch_size, device,
        )
        np.savez_compressed(
            os.path.join(feature_dir, f'{split}_sae_projection.npz'),
            activations=activations,
            reconstructed=reconstructed,
        )
        split_metrics = reconstruction_metrics(
            features,
            reconstructed,
            split_metadata[split],
            fc_weight,
            fc_bias,
            args.patient_threshold,
        )
        split_metrics['mean_l0'] = float(
            (activations > ACTIVE_EPS).sum(axis=1).mean()
        )
        split_metrics['dead_feature_count'] = int(
            ((activations > ACTIVE_EPS).sum(axis=0) == 0).sum()
        )
        metrics['splits'][split] = split_metrics
        split_activations[split] = activations
        all_activations.append(activations)
        all_metadata.append(split_metadata[split])

    kept_indices, train_active_counts, pruning_curve, pruning_summary = (
        run_feature_pruning(
            sae,
            split_activations['train'], split_metadata['train'],
            split_features['val'], split_activations['val'], split_metadata['val'],
            fc_weight, fc_bias, args, device,
        )
    )
    kept_mask = np.zeros(args.hidden_dim, dtype=bool)
    kept_mask[kept_indices] = True
    pruning_split_metrics = {}
    for split, split_activation_array in split_activations.items():
        pruned_reconstructed = reconstruct_masked_features(
            sae, split_activation_array, kept_mask,
            args.sae_batch_size, device,
        )
        pruning_split_metrics[split] = reconstruction_metrics(
            split_features[split], pruned_reconstructed,
            split_metadata[split], fc_weight, fc_bias, args.patient_threshold,
        )
    pruning_summary['splits'] = pruning_split_metrics
    metrics['pruning'] = pruning_summary
    pd.DataFrame({
        'feature_id': np.arange(args.hidden_dim),
        'active_patient_count_train': train_active_counts,
        'kept_after_pruning': kept_mask,
    }).to_csv(
        os.path.join(pruning_dir, 'feature_pruning_decisions.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(pruning_curve).to_csv(
        os.path.join(pruning_dir, 'pruning_curve.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    with open(
        os.path.join(pruning_dir, 'pruning_summary.json'), 'w', encoding='utf-8',
    ) as file:
        json.dump(pruning_summary, file, ensure_ascii=False, indent=2)
    print(
        f'Feature 筛选: 保留 {len(kept_indices)}/{args.hidden_dim}，'
        f'训练患者阈值>{pruning_summary["selected_active_patient_threshold"]}，'
        f'recovered CE 下降={pruning_summary["recovered_ce_drop"]:.4f}'
    )
    if not pruning_summary['selection_within_tolerance']:
        print(
            '警告: 最低激活患者门槛已超过 recovered CE 容差，'
            '该次筛选需要人工复核。'
        )

    activations = np.concatenate(all_activations)
    metadata = pd.concat(all_metadata, ignore_index=True)
    development_mask = metadata['split'].to_numpy() != 'test'
    feature_summary = build_feature_summary(
        activations[development_mask], metadata.loc[development_mask].reset_index(drop=True),
        sae, fc_weight,
    )
    feature_summary['kept_after_pruning'] = kept_mask
    feature_summary['pruning_active_patient_count_train'] = train_active_counts
    feature_summary['pruning_status'] = np.where(kept_mask, 'kept', 'pruned')
    for split in metadata['split'].unique():
        split_mask = metadata['split'].to_numpy() == split
        current_activations = activations[split_mask]
        split_metadata_frame = metadata.loc[split_mask].reset_index(drop=True)
        split_codes, split_patients = pd.factorize(
            split_metadata_frame['patient_id'], sort=True,
        )
        split_patient_max = np.zeros(
            (len(split_patients), activations.shape[1]), dtype=np.float32,
        )
        np.maximum.at(split_patient_max, split_codes, current_activations)
        feature_summary[f'active_patient_count_{split}'] = (
            split_patient_max > ACTIVE_EPS
        ).sum(axis=0)
        feature_summary[f'mean_patient_max_{split}'] = split_patient_max.mean(axis=0)
    feature_summary.to_csv(
        os.path.join(output_dir, 'feature_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    make_feature_overviews(
        model, capture, sae,
        activations[development_mask], metadata.loc[development_mask].reset_index(drop=True),
        feature_summary,
        overview_dir, args, device,
    )
    capture.close()
    print(f'最佳 epoch: {best_checkpoint["epoch"]}')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()
