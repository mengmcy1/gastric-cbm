"""审计医学生整理后的胃早癌概念提取训练集，不修改原始数据。"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_DIR / '数据' / '归档' / '原始数据_只读'
    / '胃早癌概念提取训练集 - 平衡+整理后'
)
OUTPUT_ROOT = PROJECT_DIR / '数据整理记录' / '概念提取训练集_v1' / '原始审计'
LABELS = {'非癌': 0, '早癌': 1}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def classify_aspect(width, height):
    ratio = width / height
    if 0.95 <= ratio <= 1.05:
        return 'square'
    if 1.70 <= ratio <= 1.85:
        return 'landscape_16x9'
    return 'landscape_other' if ratio > 1.05 else 'portrait'


def classify_size(width, height):
    longest = max(width, height)
    if longest <= 640:
        return 'small_le640'
    if longest <= 1024:
        return 'medium_641_1024'
    if longest <= 1600:
        return 'large_1025_1600'
    return 'xlarge_gt1600'


def distribution_tv(first, second):
    keys = set(first) | set(second)
    first_total = sum(first.values())
    second_total = sum(second.values())
    return 0.5 * sum(
        abs(first[key] / first_total - second[key] / second_total)
        for key in keys
    )


def scan_images():
    records = []
    decode_errors = []
    for class_name, label in LABELS.items():
        class_dir = DATA_ROOT / class_name
        for source_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for patient_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
                patient_key = f'{source_dir.name}/{patient_dir.name}'
                for image_path in sorted(patient_dir.rglob('*')):
                    if not image_path.is_file() or image_path.suffix.lower() not in VALID_EXTENSIONS:
                        continue
                    try:
                        with Image.open(image_path) as image:
                            width, height = image.size
                            image_format = image.format
                        records.append({
                            'label': label,
                            'class_name': class_name,
                            'source': source_dir.name,
                            'patient_id': patient_dir.name,
                            'patient_key': patient_key,
                            'image_name': image_path.name,
                            'original_relpath': str(image_path.relative_to(DATA_ROOT)),
                            'original_path': str(image_path.resolve()),
                            'original_width': width,
                            'original_height': height,
                            'resolution': f'{width}x{height}',
                            'aspect_group': classify_aspect(width, height),
                            'size_group': classify_size(width, height),
                            'image_format': image_format,
                            'original_sha256': file_sha256(image_path),
                        })
                    except Exception as error:
                        decode_errors.append({
                            'path': str(image_path.resolve()),
                            'error': str(error),
                        })
    return pd.DataFrame(records), pd.DataFrame(decode_errors)


def add_audit_flags(manifest):
    result = manifest.copy()
    patient_label_count = result.groupby('patient_key')['label'].transform('nunique')
    result['cross_label_patient'] = patient_label_count > 1

    hash_groups = {
        sha256: f'dup_{index:04d}'
        for index, (sha256, group) in enumerate(
            sorted(result.groupby('original_sha256')), start=1
        )
        if len(group) > 1
    }
    result['duplicate_group'] = result['original_sha256'].map(hash_groups).fillna('')
    duplicate_patient_count = result.groupby('original_sha256')['patient_key'].transform('nunique')
    result['cross_patient_duplicate'] = (
        result['duplicate_group'].ne('') & duplicate_patient_count.gt(1)
    )
    result['review_status'] = 'pending'
    result['review_flags'] = ''
    result.loc[result['cross_label_patient'], 'review_flags'] = 'cross_label_patient'
    cross_duplicate = result['cross_patient_duplicate']
    result.loc[cross_duplicate, 'review_flags'] = result.loc[
        cross_duplicate, 'review_flags'
    ].map(lambda value: '|'.join(filter(None, (value, 'cross_patient_duplicate'))))
    result.insert(0, 'image_id', [f'concept_{index:05d}' for index in range(1, len(result) + 1)])
    return result


def build_patient_summary(manifest):
    return manifest.groupby(['patient_key', 'source'], as_index=False).agg(
        patient_id=('patient_id', 'first'),
        label_count=('label', 'nunique'),
        labels=('class_name', lambda values: '|'.join(sorted(set(values)))),
        image_count=('image_id', 'size'),
        early_cancer_images=('label', lambda values: int((values == 1).sum())),
        noncancer_images=('label', lambda values: int((values == 0).sum())),
        cross_label_patient=('cross_label_patient', 'max'),
    )


def build_summary(manifest, decode_errors):
    by_label = {}
    for class_name, label in LABELS.items():
        subset = manifest[manifest['label'] == label]
        patient_counts = subset.groupby('patient_key').size()
        by_label[class_name] = {
            'images': len(subset),
            'patients': int(subset['patient_key'].nunique()),
            'images_per_patient_min': int(patient_counts.min()),
            'images_per_patient_median': float(patient_counts.median()),
            'images_per_patient_mean': float(patient_counts.mean()),
            'images_per_patient_max': int(patient_counts.max()),
            'source_images': subset['source'].value_counts().to_dict(),
            'source_patients': subset.groupby('source')['patient_key'].nunique().to_dict(),
            'top_resolutions': subset['resolution'].value_counts().head(12).to_dict(),
        }
    class_subsets = {
        label: manifest[manifest['label'] == label] for label in (0, 1)
    }
    tv = {}
    for column in ('source', 'resolution', 'aspect_group', 'size_group'):
        counts = [Counter(class_subsets[label][column]) for label in (0, 1)]
        tv[column] = distribution_tv(*counts)

    cancer = class_subsets[1]
    noncancer = class_subsets[0]
    cancer_1920 = int((cancer['resolution'] == '1920x1080').sum())
    noncancer_not_1920 = int((noncancer['resolution'] != '1920x1080').sum())
    resolution_rule = {
        'sensitivity': cancer_1920 / len(cancer),
        'specificity': noncancer_not_1920 / len(noncancer),
        'accuracy': (cancer_1920 + noncancer_not_1920) / len(manifest),
    }
    duplicate_rows = manifest[manifest['duplicate_group'] != '']
    return {
        'total_images': len(manifest),
        'decode_errors': len(decode_errors),
        'by_label': by_label,
        'distribution_tv': tv,
        'resolution_1920x1080_rule': resolution_rule,
        'cross_label_patients': int(
            manifest.loc[manifest['cross_label_patient'], 'patient_key'].nunique()
        ),
        'duplicate_groups': int(duplicate_rows['duplicate_group'].nunique()),
        'duplicate_files': len(duplicate_rows),
        'cross_patient_duplicate_groups': int(
            manifest.loc[
                manifest['cross_patient_duplicate'], 'duplicate_group'
            ].nunique()
        ),
    }


def write_markdown(summary):
    cancer = summary['by_label']['早癌']
    noncancer = summary['by_label']['非癌']
    lines = [
        '# 医学生整理概念集原始审计 v1',
        '',
        f'- 有效图片：{summary["total_images"]}；解码失败：{summary["decode_errors"]}',
        f'- 早癌：{cancer["images"]}张/{cancer["patients"]}人；每患者均值{cancer["images_per_patient_mean"]:.2f}张',
        f'- 非癌：{noncancer["images"]}张/{noncancer["patients"]}人；每患者均值{noncancer["images_per_patient_mean"]:.2f}张',
        f'- 来源图片分布TV：{summary["distribution_tv"]["source"]:.4f}',
        f'- 精确分辨率TV：{summary["distribution_tv"]["resolution"]:.4f}',
        f'- 宽高比分组TV：{summary["distribution_tv"]["aspect_group"]:.4f}',
        f'- 尺寸分组TV：{summary["distribution_tv"]["size_group"]:.4f}',
        f'- 仅用1920x1080规则：Sensitivity {summary["resolution_1920x1080_rule"]["sensitivity"]:.3f}，Specificity {summary["resolution_1920x1080_rule"]["specificity"]:.3f}，Accuracy {summary["resolution_1920x1080_rule"]["accuracy"]:.3f}',
        f'- 跨标签患者：{summary["cross_label_patients"]}',
        f'- 完全重复：{summary["duplicate_groups"]}组/{summary["duplicate_files"]}张；跨患者{summary["cross_patient_duplicate_groups"]}组',
        '',
        '结论：图片数量和来源构成基本平衡，但患者数量、每患者图片数、分辨率、尺寸和宽高比明显不平衡，不能直接作为去偏分类训练集。',
    ]
    (OUTPUT_ROOT / '审计报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest, decode_errors = scan_images()
    manifest = add_audit_flags(manifest)
    patient_summary = build_patient_summary(manifest)
    summary = build_summary(manifest, decode_errors)

    manifest.to_csv(OUTPUT_ROOT / 'original_manifest.csv', index=False, encoding='utf-8-sig')
    patient_summary.to_csv(
        OUTPUT_ROOT / 'patient_summary.csv', index=False, encoding='utf-8-sig'
    )
    manifest[manifest['duplicate_group'] != ''].to_csv(
        OUTPUT_ROOT / 'exact_duplicates.csv', index=False, encoding='utf-8-sig'
    )
    manifest[manifest['cross_label_patient']].to_csv(
        OUTPUT_ROOT / 'cross_label_patients.csv', index=False, encoding='utf-8-sig'
    )
    decode_errors.to_csv(OUTPUT_ROOT / 'decode_errors.csv', index=False, encoding='utf-8-sig')
    with open(OUTPUT_ROOT / 'audit_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_markdown(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'审计结果: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
