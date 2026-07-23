"""在医学生整理后的严格平衡概念集上运行 MOCE。"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[3]
MOCE_DIR = PROJECT_DIR / '程序' / 'MOCE' / '正式代码'
TRAIN_DIR = PROJECT_DIR / '程序' / '模型训练' / '正式代码'
sys.path.insert(0, str(MOCE_DIR))
sys.path.insert(0, str(TRAIN_DIR))

import moce_cluster as moce  # noqa: E402
from efficientnet_train_debiased import build_model as build_efficientnet  # noqa: E402
from resnet_train_debiased import build_model as build_resnet  # noqa: E402


MANIFEST_PATH = (
    PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1' / '核心清单_v1'
    / 'strict_pair_image_balanced_manifest.csv'
)
DATA_DIR = PROJECT_DIR / '数据' / '胃早癌概念提取训练集_裁剪后_v1_1'
OUTPUT_ROOT = PROJECT_DIR / '结果' / 'MOCE聚类' / '概念严格平衡_v1'

MODEL_SPECS = {
    'resnet50': {
        'builder': build_resnet,
        'weight': (
            PROJECT_DIR / '结果' / '去偏重训练_v1'
            / 'expA_full_resnet50_seed42' / 'resnet50_debiased_best.pth'
        ),
    },
    'efficientnet_b0': {
        'builder': build_efficientnet,
        'weight': (
            PROJECT_DIR / '结果' / '去偏重训练_v1'
            / 'expA_full_efficientnet_b0_seed42'
            / 'efficientnet_b0_debiased_best.pth'
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='在严格1:1图片平衡概念集上运行MOCE',
    )
    parser.add_argument(
        '--model', choices=sorted(MODEL_SPECS), default='resnet50',
    )
    parser.add_argument(
        '--mode', choices=['debug', 'full'], default='debug',
        help='debug按匹配对各取一张图；full使用严格清单全部502张',
    )
    parser.add_argument(
        '--debug-pairs', type=int, default=20,
        help='debug使用的匹配对数量，默认得到20张非癌和20张癌图',
    )
    parser.add_argument('--clusters', type=int, default=25)
    parser.add_argument(
        '--overwrite', action='store_true',
        help='删除本模型同模式的已有输出后重跑',
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(frame):
    required = {
        'image_id', 'label', 'patient_key', 'processed_relpath',
        'processed_path', 'original_sha256', 'source', 'match_id',
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'严格清单缺少字段: {missing}')
    if len(frame) != 502 or frame['patient_key'].nunique() != 314:
        raise ValueError(
            f'严格清单规模异常: {len(frame)}张/{frame["patient_key"].nunique()}人'
        )
    if frame['image_id'].duplicated().any():
        raise ValueError('严格清单存在重复image_id')
    if frame['original_sha256'].duplicated().any():
        raise ValueError('严格清单存在重复原图SHA')
    if frame.groupby('patient_key')['label'].nunique().max() != 1:
        raise ValueError('严格清单存在跨标签患者')
    label_counts = frame['label'].value_counts().to_dict()
    if label_counts != {0: 251, 1: 251}:
        raise ValueError(f'严格清单图片标签不再1:1: {label_counts}')
    missing_paths = [path for path in frame['processed_path'] if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(f'裁剪图片缺失{len(missing_paths)}张: {missing_paths[0]}')


def select_samples(frame, mode, debug_pairs):
    if mode == 'full':
        return frame.sort_values(['label', 'match_id', 'patient_key', 'image_id'])
    if debug_pairs <= 0:
        raise ValueError('--debug-pairs必须为正整数')
    match_ids = sorted(frame['match_id'].unique())[:debug_pairs]
    selected = (
        frame[frame['match_id'].isin(match_ids)]
        .sort_values(['match_id', 'label', 'patient_key', 'image_id'])
        .groupby(['match_id', 'label'], as_index=False, sort=True)
        .head(1)
    )
    expected = min(debug_pairs, frame['match_id'].nunique())
    counts = selected['label'].value_counts().to_dict()
    if counts != {0: expected, 1: expected}:
        raise ValueError(f'debug匹配对抽样不完整: {counts}')
    return selected.sort_values(['label', 'match_id']).reset_index(drop=True)


def load_debiased_model(model_name):
    spec = MODEL_SPECS[model_name]
    checkpoint = torch.load(
        spec['weight'], map_location=moce.DEVICE, weights_only=False,
    )
    model = spec['builder'](pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(moce.DEVICE).eval()


def enrich_records(records, metadata):
    by_path = metadata.set_index('processed_relpath').to_dict('index')
    for record in records:
        row = by_path[record['image_name']]
        record.update({
            'image_id': row['image_id'],
            'source_name': row['source'],
            'match_id': row['match_id'],
            'original_sha256': row['original_sha256'],
        })


def prepare_output(args, selected):
    output_dir = OUTPUT_ROOT / args.mode / args.model
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f'输出目录已存在: {output_dir}；确认重跑时添加--overwrite'
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        output_dir / 'frozen_manifest_snapshot.csv',
        index=False,
        encoding='utf-8-sig',
    )
    config = {
        'model': args.model,
        'mode': args.mode,
        'debug_pairs': args.debug_pairs if args.mode == 'debug' else None,
        'clusters': args.clusters,
        'random_seed': moce.RANDOM_SEED,
        'manifest_path': str(MANIFEST_PATH),
        'manifest_sha256': file_sha256(MANIFEST_PATH),
        'weight_path': str(MODEL_SPECS[args.model]['weight']),
        'weight_sha256': file_sha256(MODEL_SPECS[args.model]['weight']),
        'image_count': len(selected),
        'patient_count': selected['patient_key'].nunique(),
        'label_image_counts': {
            str(key): int(value)
            for key, value in selected['label'].value_counts().sort_index().items()
        },
    }
    with open(output_dir / 'run_config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return output_dir


def run(args):
    manifest = pd.read_csv(MANIFEST_PATH, encoding='utf-8-sig', low_memory=False)
    validate_manifest(manifest)
    selected = select_samples(manifest, args.mode, args.debug_pairs)
    output_dir = prepare_output(args, selected)

    moce.DATA_DIR = str(DATA_DIR)
    moce.MODEL_NAME = args.model
    moce.N_CLUSTERS = args.clusters
    model = load_debiased_model(args.model)
    capture = moce.ActivationCapture(moce.TARGET_LAYERS[args.model](model))

    print(f'设备: {moce.DEVICE}')
    print(f'模型: {args.model}（v1.1裁剪重训练权重）')
    print(f'模式: {args.mode}')
    print(
        f'样本: {len(selected)}张/{selected["patient_key"].nunique()}人, '
        f'标签={selected["label"].value_counts().sort_index().to_dict()}'
    )
    print(f'输出: {output_dir}')

    try:
        for label in (0, 1):
            class_frame = selected[selected['label'] == label].copy()
            class_frame['图片名字'] = class_frame['processed_relpath']
            class_frame['patient_id'] = class_frame['patient_key']
            class_dir = output_dir / f'class_{label}'
            class_dir.mkdir(parents=True, exist_ok=True)
            print(
                f'\n===== 类别{label}: {moce.CLASS_NAMES[label]}, '
                f'{len(class_frame)}张/{class_frame["patient_key"].nunique()}人 ====='
            )
            features, records = moce.extract_class_features(
                model, capture, label, class_frame, str(class_dir),
            )
            enrich_records(records, class_frame)
            assignment, summary = moce.cluster_features(
                features, records, str(class_dir),
            )
            if assignment.empty:
                raise RuntimeError(f'类别{label}候选区域不足，无法完成聚类')
            moce.save_cluster_overview(
                label, assignment, str(class_dir / 'concept_clusters.png'),
            )
            importance = moce.evaluate_concept_importance(
                model, label, assignment, str(class_dir),
            )
            ssc_sdc = moce.evaluate_ssc_sdc(
                model, label, assignment, importance, str(class_dir),
            )
            print(f'候选区域总数: {len(features)}')
            print(f'平均每簇患者数: {summary["patient_count"].mean():.2f}')
            print(f'最高重要性概念簇: {int(importance.iloc[0]["cluster_id"])}')
            print(f'SSC/SDC评估步数: {len(ssc_sdc) - 1}')
    finally:
        capture.close()


def main():
    args = parse_args()
    if args.clusters <= 1:
        raise ValueError('--clusters必须大于1')
    run(args)


if __name__ == '__main__':
    main()

