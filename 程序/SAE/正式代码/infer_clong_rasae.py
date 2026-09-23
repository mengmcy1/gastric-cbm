#!/usr/bin/env python3
"""原C-long分类＋RA-SAE Feature筛选、精确删除与可选响应输出。

只对用户指定的已预处理图片运行。分类结果来自原C-long，SAE重构与删除为研究输出。
发布权重保持原Feature编号；不能用于受约束RA字典继续训练或自动医学命名。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '程序/MAGE/正式代码'))
from infer_mage import image_tensor, AttentionPoolingTeacher
from clong_s2c_core import MatryoshkaSparseAutoencoder


def spatial_forward(model, spatial):
    """model为冻结C-long，spatial为[B,49,1280]；返回logits[B,2]、attention[B,49]。"""
    feature = spatial.transpose(1, 2).reshape(-1, 1280, 7, 7)
    attention = model.attention_head(feature).flatten(1).softmax(1)
    pooled = (spatial * attention.unsqueeze(-1)).sum(1)
    return model.classifier(pooled), attention


def feature_scores(model, spatial, hidden, decoder):
    """给定冻结模型、[B,49,D]表示、[B,49,H]激活、[H,D]字典，返回[B,H]带符号激活×梯度。"""
    logits, attention = spatial_forward(model, spatial)
    wm = model.classifier[1].weight[1] - model.classifier[1].weight[0]
    wa = model.attention_head.weight.reshape(-1)
    local = spatial @ wm
    mean = (attention * local).sum(1, keepdim=True)
    gradient = attention.unsqueeze(-1) * (wm + (local - mean).unsqueeze(-1) * wa)
    return (hidden * (gradient @ decoder.T)).sum(1)


def main():
    """CLI读取单图或CSV，保存原预测及每图Top-k Feature删除结果；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--manifest', type=Path, help='CSV必含image_path,patient_id；相对路径基于CSV目录')
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--sae', type=Path, default=ROOT / '模型资产/sae/ra100_inference.pth')
    parser.add_argument('--clong', type=Path, help='必须为原C-long，不接受微调后模型与旧SAE混用')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--top-k', type=int, default=20, help='每图展示数量，不是编码器K；编码器固定256')
    parser.add_argument('--save-maps', action='store_true', help='保存选中Feature原激活与原注意力7×7数组')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 2560:
        parser.error('top-k必须在1至2560之间')
    catalog = json.loads((ROOT / '模型资产/mage_handoff.json').read_text())
    clong_path = args.clong or ROOT / catalog['models']['student']['path']
    exported = torch.load(args.sae, map_location='cpu', weights_only=True)
    if exported['format'] != 'clong_rasae_effective_dictionary_v1':
        raise ValueError('请使用交接版等价推理权重')
    with clong_path.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != exported['clong_sha256']:
            raise ValueError('该SAE绑定原C-long；换分类器后必须另行训练和评价SAE')
    device = torch.device(args.device)
    checkpoint = torch.load(clong_path, map_location='cpu', weights_only=False)
    model = AttentionPoolingTeacher(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    s = exported['state_dict']
    sae = MatryoshkaSparseAutoencoder(1280, 2560, s['decoder_bias'], tuple(exported['k_list'])).to(device)
    sae.load_state_dict(s, strict=True)
    sae.eval().requires_grad_(False)
    if args.image:
        frame = pd.DataFrame({'image_path': [str(args.image.resolve())], 'patient_id': ['single_image']})
        image_root = Path('.')
    else:
        frame = pd.read_csv(args.manifest, dtype={'patient_id': str})
        if not {'image_path', 'patient_id'} <= set(frame) or frame.empty or frame[['image_path', 'patient_id']].isna().any().any():
            raise ValueError('CSV必须有非空image_path和patient_id')
        image_root = args.image_root or args.manifest.resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    if args.save_maps:
        (args.output / 'maps').mkdir()
    image_rows, feature_rows = [], []
    image_threshold = checkpoint['metrics']['image_threshold_metrics']['threshold']
    patient_threshold = checkpoint['metrics']['patient_threshold_metrics']['threshold']
    q99 = exported['train_q99'].numpy()
    with torch.inference_mode():
        for number, row in enumerate(frame.itertuples(index=False)):
            inputs = image_tensor(image_root / row.image_path, 'student').unsqueeze(0).to(device)
            spatial = model.features(inputs).flatten(2).transpose(1, 2)
            logits, attention = spatial_forward(model, spatial)
            original_margin = logits[0, 1] - logits[0, 0]
            probability = float(logits.softmax(1)[0, 1])
            hidden = sae.encode(spatial, 256)
            score = feature_scores(model, spatial, hidden, sae.decoder_weight)[0]
            active = hidden[0].gt(0).any(0)
            order = score.abs().masked_fill(~active, -1).argsort(descending=True, stable=True)
            selected = order[:min(args.top_k, int(active.sum()))]
            rebuilt_logits, _ = spatial_forward(model, sae.decode(hidden))
            image_rows.append({'image_index': number, 'image_path': row.image_path,
                'patient_id': row.patient_id, 'cancer_probability': probability,
                'image_prediction': int(probability >= image_threshold),
                'pure_sae_reconstruction_probability': float(rebuilt_logits.softmax(1)[0, 1])})
            for rank, feature in enumerate(selected.tolist(), 1):
                changed = spatial - hidden[:, :, feature:feature+1] * sae.decoder_weight[feature]
                changed_logits, _ = spatial_forward(model, changed)
                delta_margin = float(changed_logits[0, 1] - changed_logits[0, 0] - original_margin)
                feature_rows.append({'image_index': number, 'feature_id': feature,
                    'feature_name': f'RA-F{feature:04d}', 'gradient_rank': rank,
                    'activation_gradient': float(score[feature]), 'delta_margin': delta_margin,
                    'delta_probability': float(changed_logits.softmax(1)[0, 1]) - probability,
                    'peak_over_train_q99': float(hidden[0, :, feature].max()) / float(q99[feature])})
            if args.save_maps:
                np.savez_compressed(args.output / 'maps' / f'{number:05d}.npz',
                    feature_ids=selected.cpu().numpy(),
                    raw_feature_activation=hidden[0, :, selected].T.reshape(-1, 7, 7).cpu().numpy(),
                    train_q99=q99[selected.cpu().numpy()], original_attention=attention[0].reshape(7, 7).cpu().numpy())
            print(f'{number + 1}/{len(frame)} 完成', flush=True)
    images = pd.DataFrame(image_rows)
    images.to_csv(args.output / 'image_predictions.csv', index=False)
    pd.DataFrame(feature_rows).to_csv(args.output / 'feature_effects.csv', index=False)
    patients = images.groupby('patient_id', sort=False).cancer_probability.mean().reset_index()
    patients['prediction'] = patients.cancer_probability.ge(patient_threshold).astype(int)
    patients.to_csv(args.output / 'patient_predictions.csv', index=False)
    (args.output / 'definition.json').write_text(json.dumps({
        'encoder_k': 256, 'requested_features_per_image': args.top_k,
        'ranking': 'absolute activation times full attention-chain gradient at original representation',
        'delta': 'after whole-image feature deletion minus original; residual preserved',
        'positive_delta_margin': 'deleted component originally lowers cancer margin',
        'diagnosis': 'original C-long; SAE pure reconstruction and deletion are research outputs',
        'map_coordinates': '7x7 full preprocessed image; not segmentation',
        'image_threshold': image_threshold, 'patient_threshold': patient_threshold,
        'medical_concepts_confirmed': False,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
