#!/usr/bin/env python3
"""用授权train/val清单贯通教师训练和学生蒸馏。

复用MG1b/MG2的数据变换、模型、损失、分阶段冻结和采样权重。此入口用于
接收方的新训练，不冒充历史冻结实验；原训练入口的资格和SHA检查保持不变。
固定末轮保存；只用val生成本次模型阈值，不读取test或external。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from build_mage_teacher_roi_manifest import file_sha256
from train_mage_mg1_teacher import seed_everything, patient_class_balanced_weights, sensitivity_threshold_metrics
from train_mage_mg1b_attention_teacher import (
    MG1bDataset, AttentionPoolingTeacher, set_stage, train_epoch as train_teacher,
)
from train_mage_mg2_student import MG2StudentDataset, train_epoch as train_student
from build_mage_mg2_teacher_cache import build_entries

ROOT = Path(__file__).resolve().parents[3]


def read_manifest(path, image_root):
    """读取完整图坐标的ROI清单；新数据必须明确提供ROI，不猜测非癌框。"""
    frame = pd.read_csv(path, dtype={'patient_id': str})
    roi = ['base_crop_x1', 'base_crop_y1', 'base_crop_x2', 'base_crop_y2']
    bbox = ['bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm']
    required = ['image_relpath', 'patient_id', 'label', 'split'] + roi + bbox
    if not set(required) <= set(frame) or frame.empty:
        raise ValueError(f'清单必须非空且包含: {required}')
    if set(frame.split) != {'train', 'val'} or not set(frame.label) <= {0, 1}:
        raise ValueError('训练入口只接受train/val，标签只接受0/1')
    if frame.groupby('patient_id').split.nunique().max() > 1:
        raise ValueError('同一患者不能跨train/val')
    if frame.groupby('patient_id').label.nunique().max() > 1:
        raise ValueError('同一患者出现跨类别标签，需先明确数据口径')
    if any(set(g.label) != {0, 1} for _, g in frame.groupby('split')):
        raise ValueError('train和val均需包含癌与非癌')
    for columns, selected in [(roi, frame), (bbox, frame[frame.label == 1])]:
        values = selected[columns].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or (
            values[:, 0] >= values[:, 2]).any() or (values[:, 1] >= values[:, 3]).any():
            raise ValueError('ROI/癌框必须是完整预处理图上的有效归一化矩形')
    cancer = frame[frame.label == 1]
    if ((cancer.bbox_x1_norm < cancer.base_crop_x1) | (cancer.bbox_y1_norm < cancer.base_crop_y1)
        | (cancer.bbox_x2_norm > cancer.base_crop_x2) | (cancer.bbox_y2_norm > cancer.base_crop_y2)).any():
        raise ValueError('癌图教师ROI必须完整包含癌框')
    frame['image_relpath'] = frame.image_relpath.map(lambda p: str((image_root / str(p)).resolve()))
    if 'relative_path' not in frame:
        frame['relative_path'] = frame.image_relpath
    # 使用现有逐图SHA机制确保缓存、翻转和实际输入对应。
    frame['sha256'] = frame.image_relpath.map(lambda p: file_sha256(Path(p)))
    if frame.sha256.duplicated().any():
        raise ValueError('清单含重复图片内容，应先处理重复或跨集合泄漏')
    return frame


@torch.inference_mode()
def evaluate(model, frame, role, device, batch_size):
    dataset = MG1bDataset(frame, False, 42) if role == 'teacher' else MG2StudentDataset(frame, False, 42, None)
    model.eval()
    probabilities = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits, _ = model(batch['image'].to(device))
        probabilities.extend(logits.softmax(1)[:, 1].cpu().tolist())
    images = frame[['patient_id', 'label', 'image_relpath']].copy()
    images['cancer_probability'] = probabilities
    patients = images.groupby('patient_id', as_index=False).agg(label=('label', 'first'), probability=('cancer_probability', 'mean'))
    metrics = {
        'val_image_auc': float(roc_auc_score(images.label, images.cancer_probability)),
        'val_patient_auc': float(roc_auc_score(patients.label, patients.probability)),
        'image_threshold_metrics': sensitivity_threshold_metrics(images, 'cancer_probability'),
        'patient_threshold_metrics': sensitivity_threshold_metrics(patients, 'probability'),
    }
    return metrics, images, patients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['teacher', 'student'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--teacher', type=Path, help='学生训练使用本参数指定的已训练教师')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stage-a-epochs', type=int, default=5)
    parser.add_argument('--stage-b-epochs', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.19362648121926898,
                        help='新训练的固定attention蒸馏系数；不声称对新数据完成旧协议校准')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--random-init', action='store_true', help='仅供小样本链路验证；正式新训练默认ImageNet初始化')
    args = parser.parse_args()
    if args.batch_size < 2 or min(args.stage_a_epochs, args.stage_b_epochs) < 0 or args.stage_a_epochs + args.stage_b_epochs < 1:
        parser.error('batch-size至少2，训练轮数必须有效')
    if args.role == 'student' and args.teacher is None:
        parser.error('学生训练必须显式指定--teacher')
    frame = read_manifest(args.manifest, args.image_root)
    train = frame[frame.split == 'train'].reset_index(drop=True)
    val = frame[frame.split == 'val'].reset_index(drop=True)
    if len(train) % args.batch_size == 1:
        parser.error('末批仅1图不能用于训练BN，请换batch-size')
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device(args.device)
    cache = None
    if args.role == 'student':
        checkpoint = torch.load(args.teacher, map_location='cpu', weights_only=False)
        if checkpoint.get('architecture') != 'efficientnet_b0_attention_pooling_7x7' or 'arm' in checkpoint:
            raise ValueError('必须提供灰度ROI教师checkpoint，不能使用学生权重')
        teacher = AttentionPoolingTeacher(pretrained=False).to(device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher.eval()
        cache = {'entries': build_entries(train, teacher, device, args.batch_size)}
        torch.save(cache, args.output / 'teacher_cache.pt')
        del teacher
    model = AttentionPoolingTeacher(pretrained=not args.random_init).to(device)
    dataset = MG1bDataset(train, True, args.seed) if args.role == 'teacher' else MG2StudentDataset(train, True, args.seed, cache)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(patient_class_balanced_weights(train), dtype=torch.double),
                                   num_samples=len(train), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    history = []
    epoch = 0
    optimizer = None
    for stage, count, lr in [('A', args.stage_a_epochs, 1e-3), ('B', args.stage_b_epochs, 1e-4)]:
        if count == 0:
            continue
        set_stage(model, stage)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
        for _ in range(count):
            epoch += 1
            dataset.set_epoch(epoch)
            losses = train_teacher(model, loader, optimizer, device, stage) if args.role == 'teacher' else train_student(
                'C', model, loader, optimizer, device, stage, args.beta)
            history.append({'stage': stage, 'epoch': epoch, **losses})
            pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
            print(json.dumps(history[-1]), flush=True)
    metrics, images, patients = evaluate(model, val, args.role, device, args.batch_size)
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'architecture': 'efficientnet_b0_attention_pooling_7x7', 'metrics': metrics,
               'scope': 'new_training_handoff_not_historical_frozen_product', 'epoch': epoch}
    if args.role == 'student':
        payload['arm'] = 'C'
    torch.save(payload, args.output / 'model.pth')
    images.to_csv(args.output / 'val_image_predictions.csv', index=False)
    patients.to_csv(args.output / 'val_patient_predictions.csv', index=False)
    (args.output / 'config.json').write_text(json.dumps({**vars(args), 'metrics': metrics,
        'selection': 'fixed final epoch; threshold selected on val only',
        'historical_protocol_reproduction': False}, default=str, ensure_ascii=False, indent=2))
    print('完成教师训练' if args.role == 'teacher' else '完成学生蒸馏训练', flush=True)


if __name__ == '__main__':
    main()
