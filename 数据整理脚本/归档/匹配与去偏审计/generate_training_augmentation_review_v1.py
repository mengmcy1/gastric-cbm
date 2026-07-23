"""生成 P6 原图与破坏性训练增强对照，供正式训练前肉眼确认。"""

import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CODE_DIR = os.path.join(PROJECT_DIR, '程序', '模型训练', '正式代码')
sys.path.insert(0, TRAIN_CODE_DIR)

from train_utils import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
)


RISK_MANIFEST = os.path.join(
    PROJECT_DIR,
    '数据整理记录',
    '第二批',
    '匹配_v1',
    'P5视觉复核_v1',
    '风险分层复核清单.csv',
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, '数据整理记录', '第二批', '训练增强复核_v1'
)
RANDOM_SEED = 42
SAMPLES_PER_CLASS = 12
VARIANTS = 3
PANEL_SIZE = 256
TITLE_HEIGHT = 48
ROWS_PER_SHEET = 4


def parse_args():
    parser = argparse.ArgumentParser(description='生成 P6 训练增强复核图 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_pil(relative_path):
    path = os.path.join(DEFAULT_IMAGE_ROOT, relative_path)
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def tensor_to_bgr(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = ((tensor.cpu() * std + mean).clamp(0, 1) * 255).byte()
    rgb = rgb.permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def pil_to_bgr(image):
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def titled_panel(image, title, color):
    image = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (1, 1), (PANEL_SIZE - 2, PANEL_SIZE - 2), color, 3)
    canvas = np.full((TITLE_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 255, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = image
    cv2.putText(canvas, title, (5, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1)
    return canvas


def augmented_variants(image, sample_number, transform):
    variants = []
    for variant in range(VARIANTS):
        seed = RANDOM_SEED + sample_number * 100 + variant
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        variants.append(tensor_to_bgr(transform(image.copy())))
    return variants


def write_sheet(rows, page, output_dir):
    panel_height = TITLE_HEIGHT + PANEL_SIZE
    columns = 1 + VARIANTS
    sheet = np.full(
        (ROWS_PER_SHEET * panel_height, columns * PANEL_SIZE, 3),
        245,
        dtype=np.uint8,
    )
    for row_number, row in enumerate(rows):
        color = (20, 40, 210) if int(row['label']) == 1 else (40, 160, 40)
        images = [row['original']] + row['variants']
        titles = [
            f'#{row["review_number"]:02d} y={int(row["label"])} ORIGINAL',
            *[f'AUGMENTED v{number}' for number in range(1, VARIANTS + 1)],
        ]
        for column, (image, title) in enumerate(zip(images, titles)):
            item = titled_panel(image, title, color)
            y1 = row_number * panel_height
            x1 = column * PANEL_SIZE
            sheet[y1:y1 + panel_height, x1:x1 + PANEL_SIZE] = item
    path = os.path.join(output_dir, f'训练增强对照_{page:03d}.jpg')
    success, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not success:
        raise ValueError(f'无法编码增强复核图: {path}')
    encoded.tofile(path)


def main():
    args = parse_args()
    if os.path.exists(OUTPUT_DIR):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    risk = pd.read_csv(RISK_MANIFEST, encoding='utf-8-sig')
    selected = pd.concat([
        group.sample(
            n=min(SAMPLES_PER_CLASS, len(group)),
            random_state=RANDOM_SEED + int(label),
        )
        for label, group in risk.groupby('label', sort=True)
    ]).sort_values(['label', 'match_group_id', 'processed_path'], kind='stable')
    selected = selected.reset_index(drop=True)
    selected.insert(0, 'augmentation_review_number', np.arange(1, len(selected) + 1))
    selected.to_csv(
        os.path.join(OUTPUT_DIR, '训练增强复核清单.csv'), index=False, encoding='utf-8-sig'
    )

    train_transform, _ = build_transforms()
    rendered = []
    for row in selected.itertuples(index=False):
        original_pil = read_pil(row.processed_path)
        rendered.append({
            'review_number': int(row.augmentation_review_number),
            'label': int(row.label),
            'original': pil_to_bgr(original_pil),
            'variants': augmented_variants(
                original_pil,
                int(row.augmentation_review_number),
                train_transform,
            ),
        })
    for page, start in enumerate(range(0, len(rendered), ROWS_PER_SHEET), start=1):
        write_sheet(rendered[start:start + ROWS_PER_SHEET], page, OUTPUT_DIR)
    summary = (
        f'复核样本: {len(selected)}\n'
        f'癌图: {int(selected.label.eq(1).sum())}\n'
        f'非癌图: {int(selected.label.eq(0).sum())}\n'
        f'每图增强版本: {VARIANTS}\n'
        f'联系图页数: {(len(selected) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET}\n'
    )
    with open(os.path.join(OUTPUT_DIR, '训练增强复核汇总.txt'), 'w', encoding='utf-8') as file:
        file.write(summary)
    print(summary)
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
