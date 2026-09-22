#!/usr/bin/env python3
"""可移植的MG1b教师/C-long学生推理及患者级后处理。

输入必须是已经完成正式裁边/FOV处理的图片。学生使用完整彩色图；教师还需
提供在该图片上的归一化ROI，裁图后转灰度。不会自动猜测患者或教师ROI。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from train_mage_mg1b_attention_teacher import AttentionPoolingTeacher

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / '模型资产/mage_handoff.json'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_tensor(path, role, roi=None):
    """按冻结评价方式处理一张已裁边/FOV遮罩图；ROI坐标相对当前输入图。"""
    with Image.open(path) as source:
        image = source.convert('RGB')
    if role == 'teacher':
        box = np.asarray(roi, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not (
            0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
        ):
            raise ValueError('教师必须提供有效的归一化ROI：x1 y1 x2 y2')
        w, h = image.size
        pixels = np.round(box * [w, h, w, h]).astype(int)
        if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
            raise ValueError('教师ROI取整后为空')
        image = image.crop(tuple(pixels))
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    if role == 'teacher':
        image = image.convert('L').convert('RGB')
    return TF.normalize(TF.pil_to_tensor(image).float() / 255, MEAN, STD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['student', 'teacher'], default='student')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--manifest', type=Path, help='CSV：image_path,patient_id；教师另需roi_x1,roi_y1,roi_x2,roi_y2')
    parser.add_argument('--image-root', type=Path, help='CSV相对图片路径的根目录；默认CSV所在目录')
    parser.add_argument('--roi', type=float, nargs=4, help='单张教师输入的归一化ROI')
    parser.add_argument('--weights', type=Path, help='默认加载mage_handoff.json中的正式权重')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-attention', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size必须大于0')
    if args.role == 'student' and args.roi is not None:
        parser.error('学生输入是完整预处理图，不接受ROI裁图')
    if args.manifest is not None and args.roi is not None:
        parser.error('CSV模式请为每行提供ROI列，不使用--roi')
    catalog = json.loads(ASSETS.read_text(encoding='utf-8'))
    path = args.weights or ROOT / catalog['models'][args.role]['path']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['architecture'] != 'efficientnet_b0_attention_pooling_7x7':
        raise ValueError('不是MG1b/C-long attention-pooling架构')
    if args.role == 'student' and checkpoint.get('arm') != 'C':
        raise ValueError('学生入口要求C组checkpoint')
    if args.role == 'teacher' and 'arm' in checkpoint:
        raise ValueError('不能将学生权重用于教师灰度ROI入口')
    image_threshold = float(checkpoint['metrics']['image_threshold_metrics']['threshold'])
    patient_threshold = float(checkpoint['metrics']['patient_threshold_metrics']['threshold'])
    device = torch.device(args.device)
    model = AttentionPoolingTeacher(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    roi_columns = ['roi_x1', 'roi_y1', 'roi_x2', 'roi_y2']
    if args.image is not None:
        frame = pd.DataFrame({'image_path': [str(args.image.resolve())], 'patient_id': ['single_image']})
        image_root = Path('.')
        if args.role == 'teacher':
            if args.roi is None:
                parser.error('单张教师推理必须提供--roi')
            for key, value in zip(roi_columns, args.roi):
                frame[key] = value
    else:
        frame = pd.read_csv(args.manifest, dtype={'patient_id': str})
        required = ['image_path', 'patient_id'] + (roi_columns if args.role == 'teacher' else [])
        if not set(required) <= set(frame) or frame[required].isna().any().any() or frame.empty:
            raise ValueError('CSV缺少必需列、存在空值或没有图片')
        image_root = args.image_root or args.manifest.resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    rows, maps = [], []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            batch = frame.iloc[start:start + args.batch_size]
            tensors = [image_tensor(image_root / str(row.image_path), args.role,
                        [getattr(row, key) for key in roi_columns] if args.role == 'teacher' else None)
                       for row in batch.itertuples()]
            logits, attention = model(torch.stack(tensors).to(device))
            probabilities = logits.softmax(1)[:, 1].cpu().numpy()
            for row, probability in zip(batch.itertuples(), probabilities):
                rows.append({'image_path': row.image_path, 'patient_id': row.patient_id,
                             'cancer_probability': float(probability),
                             'image_prediction': int(probability >= image_threshold)})
            if args.save_attention:
                maps.append(attention[:, 0].cpu().numpy())
    images = pd.DataFrame(rows)
    patients = images.groupby('patient_id', sort=False).agg(
        image_count=('cancer_probability', 'size'), cancer_probability=('cancer_probability', 'mean')).reset_index()
    patients['patient_prediction'] = (patients.cancer_probability >= patient_threshold).astype(int)
    images.to_csv(args.output / 'image_predictions.csv', index=False)
    patients.to_csv(args.output / 'patient_predictions.csv', index=False)
    if maps:
        np.save(args.output / 'attention_7x7.npy', np.concatenate(maps))
    (args.output / 'inference_config.json').write_text(json.dumps({
        'role': args.role, 'checkpoint': str(path), 'image_threshold': image_threshold,
        'patient_threshold': patient_threshold, 'patient_aggregation': 'mean image probability',
        'input': 'already preprocessed image; no automatic raw crop or ROI proposal',
        'attention_coordinates': 'teacher ROI' if args.role == 'teacher' else 'full preprocessed image',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'完成：{len(images)}张图，{len(patients)}位患者；输出 {args.output}')


if __name__ == '__main__':
    main()
