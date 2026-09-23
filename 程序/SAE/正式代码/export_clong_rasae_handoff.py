#!/usr/bin/env python3
"""将已训练RA-SAE导出为等价推理权重，不发布训练代表点或原checkpoint元数据。"""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    """读取本地原权重及train Q99，输出仅含推理参数的文件；参数来自CLI，无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--q99', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    s, c = original['state_dict'], original['config']
    if original['arm'] != 'ra' or c['hidden_dim'] != 2560 or c['epochs'] != 100:
        raise ValueError('仅导出原100轮、2560项的RA臂')
    dictionary = s['mixture_logits'].softmax(-1) @ s['points'] + s['relaxation']
    q99 = torch.from_numpy(np.load(args.q99)).float()
    if q99.shape != (2560,) or not torch.isfinite(q99).all() or not q99.gt(0).all():
        raise ValueError('train Q99应为2560项有限正值')
    payload = {
        'format': 'clong_rasae_effective_dictionary_v1',
        'state_dict': {key: s[key] for key in ('encoder.weight', 'encoder.bias', 'decoder_bias')},
        'train_q99': q99,
        'input_dim': 1280, 'hidden_dim': 2560, 'k_list': [64, 128, 256], 'evaluation_k': 256,
        'clong_sha256': c['checkpoint_sha256'],
        'scope': 'posthoc_features_not_confirmed_medical_concepts',
        'training_resume': False,
    }
    payload['state_dict']['decoder_weight'] = dictionary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f'已导出等价推理权重：{args.output}；原训练参数化保留在受控研究包中。')


if __name__ == '__main__':
    main()
