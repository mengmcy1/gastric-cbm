"""按已确认的医学规则，将第二批数据整理为“癌/患者/图片、非癌/患者/图片”。"""

import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict

from PIL import Image


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(PROJECT_DIR, '数据', '第二批')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '数据', '第二批整理后')
RECORD_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '最终整理')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 已确认：该患者全部归为非癌。
MOVE_TO_NONCANCER = {'01.0000000126950'}

# 已确认：该患者在整批数据中只保留两张真正的癌图。
PATIENT_KEEP = {
    '01.0000000129422': {
        '01.0000000129422_6_2019-09-18_14_27_45',
        '01.0000000129422_43_2019-09-18_14_31_18',
    },
}

# 已确认：这些患者保留癌侧，删除非癌侧图片。
DROP_NONCANCER = {
    '01.0000000138634',
    '01.0000000167252',
    '02武汉市中心白光__陈桂林',
}

# 已确认：该图片属于丁晶，从江伟坚目录排除。
WRONG_OWNER_PATIENT = '06孝感分类汇总白光__NJ2019015174(江伟坚)'
WRONG_OWNER_IMAGE = 'FF53718EC5884FDBB7402FABF4A1EE3_hrisk_c96.jpg'


ANATOMICAL_SUFFIX = re.compile(
    r'\s*(?:胃体小弯|胃窦前壁|胃窦（萎缩）|胃体|胃窦|胃角|胃底|贲门)$'
)

def file_sha256(path):
    """计算文件内容哈希，用于删除完全相同的图片副本。"""
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def patient_id_from_path(relative_parts):
    """以最内层患者文件夹为准；外院患者加医院前缀避免同名。"""
    source = relative_parts[1]
    patient_folder = relative_parts[-2]
    if source == '武大省人民数据':
        return patient_folder
    if relative_parts[0] == '非癌':
        patient_folder = ANATOMICAL_SUFFIX.sub('', patient_folder).strip()
    return f'{relative_parts[2]}__{patient_folder}'


def save_as_jpg(source, destination):
    """输出为 RGB JPG；已是 RGB JPG 时直接复制，避免重复压缩。"""
    with Image.open(source) as image:
        if source.lower().endswith('.jpg') and image.mode == 'RGB':
            shutil.copy2(source, destination)
        else:
            image.convert('RGB').save(destination, 'JPEG', quality=95, subsampling=0)


def unique_destination(folder, filename):
    """合并目录后如有同名不同内容图片，为后者增加编号。"""
    destination = os.path.join(folder, filename)
    if not os.path.exists(destination):
        return destination

    stem, extension = os.path.splitext(filename)
    number = 2
    while os.path.exists(os.path.join(folder, f'{stem}__{number}{extension}')):
        number += 1
    return os.path.join(folder, f'{stem}__{number}{extension}')


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if os.path.exists(RECORD_DIR):
        shutil.rmtree(RECORD_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RECORD_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '癌'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '非癌'), exist_ok=True)

    source_paths = []
    for root, _, filenames in os.walk(SOURCE_DIR):
        if os.path.commonpath([root, OUTPUT_DIR]) == OUTPUT_DIR:
            continue
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                source_paths.append(os.path.join(root, filename))
    source_paths.sort()

    seen_hashes = defaultdict(set)
    manifest = []
    cleaning_log = []
    patient_sources = defaultdict(set)

    for source_path in source_paths:
        relative_path = os.path.relpath(source_path, SOURCE_DIR)
        parts = relative_path.split(os.sep)
        original_label = 1 if parts[0] == '早癌与高级别' else 0
        patient_id = patient_id_from_path(parts)
        label = 0 if patient_id in MOVE_TO_NONCANCER else original_label
        image_name = os.path.basename(source_path)
        image_stem = os.path.splitext(image_name)[0]

        if patient_id in PATIENT_KEEP:
            if image_stem not in PATIENT_KEEP[patient_id]:
                cleaning_log.append({
                    'source_path': relative_path,
                    'patient_id': patient_id,
                    'action': '删除',
                    'reason': '0129422按确认结果只保留指定的两张图片',
                    'destination': '',
                })
                continue

        if original_label == 0 and patient_id in DROP_NONCANCER:
            cleaning_log.append({
                'source_path': relative_path,
                'patient_id': patient_id,
                'action': '删除',
                'reason': '同患者含癌和非癌病灶，按确认结果删除非癌侧',
                'destination': '',
            })
            continue

        if patient_id == WRONG_OWNER_PATIENT and image_name == WRONG_OWNER_IMAGE:
            cleaning_log.append({
                'source_path': relative_path,
                'patient_id': patient_id,
                'action': '删除',
                'reason': '图片属于丁晶，误放在江伟坚目录',
                'destination': '',
            })
            continue

        patient_label = (patient_id, label)

        class_name = '癌' if label == 1 else '非癌'
        patient_dir = os.path.join(OUTPUT_DIR, class_name, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        jpg_name = f'{image_stem}.jpg'
        destination = unique_destination(patient_dir, jpg_name)
        save_as_jpg(source_path, destination)
        digest = file_sha256(destination)
        if digest in seen_hashes[patient_label]:
            os.remove(destination)
            cleaning_log.append({
                'source_path': relative_path,
                'patient_id': patient_id,
                'action': '去重',
                'reason': '同一患者同一标签下转换后图片内容完全相同',
                'destination': '',
            })
            continue
        seen_hashes[patient_label].add(digest)

        output_relative = os.path.relpath(destination, OUTPUT_DIR)
        action = '转为非癌' if original_label == 1 and label == 0 else '保留'
        if action != '保留' or os.path.basename(destination) != jpg_name:
            cleaning_log.append({
                'source_path': relative_path,
                'patient_id': patient_id,
                'action': action,
                'reason': '0126950病理确认为非癌' if action == '转为非癌' else '合并后同名不同内容，自动重命名',
                'destination': output_relative,
            })

        source = parts[1]
        hospital = parts[2] if source == '外院数据' else source
        patient_sources[patient_id].add((source, hospital, parts[-2]))
        manifest.append({
            'patient_id': patient_id,
            'label': label,
            'class_name': class_name,
            'image_name': os.path.basename(destination),
            'image_path': output_relative,
            'source_path': relative_path,
            'sha256': digest,
        })

    patient_rows = []
    for patient_id in sorted({row['patient_id'] for row in manifest}):
        patient_images = [row for row in manifest if row['patient_id'] == patient_id]
        sources = sorted(patient_sources[patient_id])
        patient_rows.append({
            'patient_id': patient_id,
            'labels': '/'.join(map(str, sorted({row['label'] for row in patient_images}))),
            'image_count': len(patient_images),
            'original_sources': ' || '.join('/'.join(item) for item in sources),
        })

    write_csv(
        os.path.join(OUTPUT_DIR, 'dataset_manifest.csv'),
        manifest,
        ['patient_id', 'label', 'class_name', 'image_name', 'image_path', 'source_path', 'sha256'],
    )
    write_csv(
        os.path.join(RECORD_DIR, '患者清单.csv'),
        patient_rows,
        ['patient_id', 'labels', 'image_count', 'original_sources'],
    )
    write_csv(
        os.path.join(RECORD_DIR, '清理记录.csv'),
        cleaning_log,
        ['source_path', 'patient_id', 'action', 'reason', 'destination'],
    )

    cancer_images = sum(row['label'] == 1 for row in manifest)
    noncancer_images = sum(row['label'] == 0 for row in manifest)
    cancer_patients = len({row['patient_id'] for row in manifest if row['label'] == 1})
    noncancer_patients = len({row['patient_id'] for row in manifest if row['label'] == 0})
    mixed_patients = len({row['patient_id'] for row in manifest if row['label'] == 1}
                         & {row['patient_id'] for row in manifest if row['label'] == 0})
    summary = (
        f'原始图片: {len(source_paths)}\n'
        f'整理后图片: {len(manifest)}\n'
        f'癌: {cancer_images} 张，{cancer_patients} 位患者\n'
        f'非癌: {noncancer_images} 张，{noncancer_patients} 位患者\n'
        f'同时具有两类图片的患者: {mixed_patients}\n'
        f'删除或去重: {sum(row["action"] in {"删除", "去重"} for row in cleaning_log)}\n'
    )
    with open(os.path.join(RECORD_DIR, '整理汇总.txt'), 'w', encoding='utf-8') as file:
        file.write(summary)
    print(summary)
    print(f'数据目录: {OUTPUT_DIR}')
    print(f'整理记录: {RECORD_DIR}')


if __name__ == '__main__':
    main()
