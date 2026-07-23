#!/usr/bin/env python3
"""Attach the 2026-07-15 patient split to the v1.1 full cropped manifest."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OLD_SPLIT = PROJECT_DIR / '结果/数据划分/image_split_seed42.csv'
DEFAULT_FULL = PROJECT_DIR / '数据整理记录/第二批/匹配_v1/full_manifest.csv'
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / '数据整理记录/第二批/训练划分_v1/full_legacy_compatible_split_seed42.csv'
)


def parse_args():
    parser = argparse.ArgumentParser(description='生成旧划分兼容的v1.1 full训练清单')
    parser.add_argument('--old-split', type=Path, default=DEFAULT_OLD_SPLIT)
    parser.add_argument('--full-manifest', type=Path, default=DEFAULT_FULL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'输出已存在: {args.output}')

    old = pd.read_csv(args.old_split, encoding='utf-8-sig').rename(
        columns={'瘤变标签': 'label'}
    )
    full = pd.read_csv(args.full_manifest, encoding='utf-8-sig')
    required_old = {'patient_id', 'label', 'sha256', 'split'}
    if not required_old.issubset(old.columns):
        raise ValueError(f'旧划分缺少字段: {sorted(required_old - set(old.columns))}')
    if len(old) != 5229 or len(full) != 5229:
        raise ValueError(f'图片数异常: old={len(old)}, full={len(full)}')
    if old['sha256'].duplicated().any() or full['sha256'].duplicated().any():
        raise ValueError('存在重复sha256，无法一对一连接')
    if set(old['sha256']) != set(full['sha256']):
        raise ValueError('旧划分与full裁剪清单图片集合不一致')

    assignment = old[['sha256', 'patient_id', 'label', 'split']]
    result = full.merge(
        assignment,
        on=['sha256', 'patient_id', 'label'],
        how='left',
        validate='one_to_one',
    )
    if result['split'].isna().any():
        raise ValueError('存在未连接到旧划分的图片')
    if not set(result['split']).issubset({'train', 'val', 'test'}):
        raise ValueError('发现未知split')
    if result.groupby('patient_id')['split'].nunique().gt(1).any():
        raise ValueError('患者跨split')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding='utf-8-sig')
    summary = []
    for split, group in result.groupby('split', sort=True):
        summary.append({
            'split': split,
            'patients': int(group['patient_id'].nunique()),
            'images': int(len(group)),
            'case_patients': int(group.loc[group['label'].eq(1), 'patient_id'].nunique()),
            'control_patients': int(group.loc[group['label'].eq(0), 'patient_id'].nunique()),
            'case_images': int(group['label'].eq(1).sum()),
            'control_images': int(group['label'].eq(0).sum()),
        })
    summary_path = args.output.with_name(f'{args.output.stem}_summary.csv')
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding='utf-8-sig')
    hash_path = args.output.with_name(f'{args.output.stem}_sha256.txt')
    hash_path.write_text(f'{file_sha256(args.output)}  {args.output.name}\n', encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'清单: {args.output.resolve()}')


if __name__ == '__main__':
    main()
