"""第二批裁剪数据 P2 v1.1：dHash 候选经 SSIM 确认后去重并限图。"""

import argparse
import os
import re
import shutil
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
MANIFEST_PATH = os.path.join(DATA_DIR, 'processed_manifest.csv')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1_1')

DHASH_SIZE = 8
DHASH_THRESHOLD = 4
SSIM_THRESHOLD = 0.95
MAX_IMAGES_PER_PATIENT = 3
MIN_TIME_GAP_SECONDS = 2
RANDOM_SEED = 42
DEFAULT_DEBUG_PATIENTS_PER_GROUP = 5
DEFAULT_DEBUG_LARGE_PATIENTS = 10

UNIX_TIMESTAMP_PATTERN = re.compile(r'(?<!\d)(1[4-9]\d{8})(?!\d)')
DATE_TIME_PATTERN = re.compile(
    r'(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?([0-2]\d|3[01])'
    r'[_ .-]?([01]\d|2[0-3])[_ .-]?([0-5]\d)[_ .-]?([0-5]\d)'
)
COMPACT_DATE_TIME_PATTERN = re.compile(r'(?<!\d)(20\d{12})(?!\d)')
SHORT_DATE_TIME_PATTERN = re.compile(r'(?<!\d)(\d{12})(?!\d)')


def parse_args():
    parser = argparse.ArgumentParser(description='第二批数据患者内去重与限图 v1.1')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    parser.add_argument(
        '--debug-patients-per-group',
        type=int,
        default=DEFAULT_DEBUG_PATIENTS_PER_GROUP,
    )
    parser.add_argument(
        '--debug-large-patients',
        type=int,
        default=DEFAULT_DEBUG_LARGE_PATIENTS,
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV 无法解码图片: {path}')
    return image


def dhash64(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (DHASH_SIZE + 1, DHASH_SIZE), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return (left ^ right).bit_count()


def ssim_feature(path):
    image = read_image(os.path.join(DATA_DIR, path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float64)


def ssim_score(left, right):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    variance_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    variance_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    numerator = (2 * mu_left * mu_right + c1) * (2 * covariance + c2)
    denominator = (
        (mu_left * mu_left + mu_right * mu_right + c1)
        * (variance_left + variance_right + c2)
    )
    return float((numerator / denominator).mean())


def parse_capture_time(image_name):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]

    match = UNIX_TIMESTAMP_PATTERN.search(stem)
    if match:
        value = int(match.group(1))
        try:
            parsed = datetime.fromtimestamp(value)
            return value, parsed.isoformat(timespec='seconds'), 'unix_timestamp'
        except (OSError, OverflowError, ValueError):
            pass

    match = DATE_TIME_PATTERN.search(stem)
    if match:
        values = tuple(map(int, match.groups()))
        try:
            parsed = datetime(*values)
            return int(parsed.timestamp()), parsed.isoformat(timespec='seconds'), 'datetime_text'
        except ValueError:
            pass

    match = COMPACT_DATE_TIME_PATTERN.search(stem)
    if match:
        try:
            parsed = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
            return int(parsed.timestamp()), parsed.isoformat(timespec='seconds'), 'datetime_compact'
        except ValueError:
            pass

    match = SHORT_DATE_TIME_PATTERN.search(stem)
    if match:
        try:
            parsed = datetime.strptime(match.group(1), '%y%m%d%H%M%S')
            if 2010 <= parsed.year <= 2030:
                return int(parsed.timestamp()), parsed.isoformat(timespec='seconds'), 'datetime_short'
        except ValueError:
            pass

    return None, '', 'unparsed'


def aspect_group(width, height):
    ratio = width / height
    if ratio < 0.90:
        return 'portrait'
    if ratio <= 1.10:
        return 'square'
    if abs(ratio - 4 / 3) <= 0.08:
        return 'landscape_4x3'
    if abs(ratio - 16 / 9) <= 0.10:
        return 'landscape_16x9'
    return 'landscape_other'


def size_group(width, height):
    long_side = max(width, height)
    if long_side <= 512:
        return 'small_le512'
    if long_side <= 1024:
        return 'medium_513_1024'
    if long_side <= 1600:
        return 'large_1025_1600'
    return 'xlarge_gt1600'


def frame_profile(row):
    width = max(1, int(row.original_width))
    left_ratio = float(row.edge_trim_left) / width
    right_ratio = float(row.edge_trim_right) / width
    if bool(row.progress_bar_detected):
        return 'progress_bar_ui'
    if max(left_ratio, right_ratio) > 0.03:
        return 'side_ui'
    if row.crop_status == 'cropped':
        return 'contour_frame'
    if row.crop_status == 'fallback_full_frame':
        return 'fallback_full'
    return 'full_frame'


def add_style_fields(manifest):
    manifest = manifest.copy()
    manifest['aspect_group'] = [
        aspect_group(int(row.original_width), int(row.original_height))
        for row in manifest.itertuples(index=False)
    ]
    manifest['size_group'] = [
        size_group(int(row.original_width), int(row.original_height))
        for row in manifest.itertuples(index=False)
    ]
    manifest['frame_profile'] = [
        frame_profile(row) for row in manifest.itertuples(index=False)
    ]
    manifest['style_group'] = [
        f'{row.size_group}|{row.aspect_group}|{row.frame_profile}'
        for row in manifest.itertuples(index=False)
    ]
    return manifest


def select_debug_patients(manifest, per_group, large_count):
    patient_rows = (
        manifest.groupby('patient_id', as_index=False)
        .agg(label=('label', 'first'), center=('center', 'first'), image_count=('image_path', 'size'))
    )
    selected = set(
        patient_rows.nlargest(min(large_count, len(patient_rows)), 'image_count')['patient_id']
    )
    for offset, (_, group) in enumerate(patient_rows.groupby(['center', 'label'], sort=True)):
        count = min(per_group, len(group))
        sampled = group.sample(n=count, random_state=RANDOM_SEED + offset)
        selected.update(sampled['patient_id'])
    return selected


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def duplicate_components(hashes, paths):
    union_find = UnionFind(len(hashes))
    feature_cache = {}
    candidate_scores = {}
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hamming_distance(hashes[left], hashes[right]) <= DHASH_THRESHOLD:
                if left not in feature_cache:
                    feature_cache[left] = ssim_feature(paths[left])
                if right not in feature_cache:
                    feature_cache[right] = ssim_feature(paths[right])
                score = ssim_score(feature_cache[left], feature_cache[right])
                candidate_scores[(left, right)] = score
                if score >= SSIM_THRESHOLD:
                    union_find.union(left, right)
    components = {}
    for index in range(len(hashes)):
        components.setdefault(union_find.find(index), []).append(index)
    return sorted(components.values(), key=lambda indexes: min(indexes)), candidate_scores


def time_gap_is_valid(candidate, selected, timestamps):
    candidate_time = timestamps[candidate]
    if candidate_time is None:
        return True
    for selected_index in selected:
        selected_time = timestamps[selected_index]
        if selected_time is not None and abs(candidate_time - selected_time) < MIN_TIME_GAP_SECONDS:
            return False
    return True


def choose_patient_images(group):
    group = group.sort_values('processed_path', kind='stable').copy()
    indexes = list(group.index)
    hashes = [int(group.at[index, 'dhash_int']) for index in indexes]
    sharpness = [float(group.at[index, 'sharpness']) for index in indexes]
    timestamps = [group.at[index, 'capture_time_epoch'] for index in indexes]
    timestamps = [None if pd.isna(value) else int(value) for value in timestamps]
    paths = [str(group.at[index, 'processed_path']) for index in indexes]

    components, candidate_scores = duplicate_components(hashes, paths)
    representatives = []
    duplicate_meta = {}
    patient_id = str(group.iloc[0]['patient_id'])
    for number, component in enumerate(components, start=1):
        representative = sorted(component, key=lambda i: (-sharpness[i], paths[i]))[0]
        representatives.append(representative)
        group_id = f'{patient_id}__dup{number:03d}'
        confirmed_scores = [
            score for (left, right), score in candidate_scores.items()
            if left in component and right in component and score >= SSIM_THRESHOLD
        ]
        minimum_confirmed_ssim = min(confirmed_scores) if confirmed_scores else np.nan
        for local_index in component:
            duplicate_meta[local_index] = (
                group_id,
                len(component),
                representative,
                minimum_confirmed_ssim,
            )

    first = sorted(representatives, key=lambda i: (-sharpness[i], paths[i]))[0]
    selected = [first]
    selection_reason = {first: 'highest_sharpness'}
    time_relaxed = {first: False}
    while len(selected) < min(MAX_IMAGES_PER_PATIENT, len(representatives)):
        remaining = [index for index in representatives if index not in selected]
        time_valid = [
            index for index in remaining if time_gap_is_valid(index, selected, timestamps)
        ]
        pool = time_valid if time_valid else remaining
        relaxed = not bool(time_valid)

        def candidate_key(index):
            minimum_distance = min(
                hamming_distance(hashes[index], hashes[selected_index])
                for selected_index in selected
            )
            return (-minimum_distance, -sharpness[index], paths[index])

        chosen = sorted(pool, key=candidate_key)[0]
        selected.append(chosen)
        selection_reason[chosen] = (
            'max_dhash_diversity_time_relaxed' if relaxed else 'max_dhash_diversity'
        )
        time_relaxed[chosen] = relaxed

    rows = []
    for local_index, dataframe_index in enumerate(indexes):
        group_id, group_size, representative, minimum_confirmed_ssim = duplicate_meta[local_index]
        is_representative = local_index == representative
        is_selected = local_index in selected
        nearest_selected = min(
            (hamming_distance(hashes[local_index], hashes[value]) for value in selected if value != local_index),
            default=np.nan,
        )
        if not is_representative:
            exclusion_reason = 'near_duplicate'
        elif not is_selected:
            exclusion_reason = 'patient_image_limit'
        else:
            exclusion_reason = ''
        rows.append({
            'dataframe_index': dataframe_index,
            'duplicate_group': group_id,
            'duplicate_group_size': group_size,
            'duplicate_group_min_confirmed_ssim': minimum_confirmed_ssim,
            'duplicate_confirmation_method': 'dhash_le4_and_ssim_ge0.95',
            'duplicate_representative_path': paths[representative],
            'is_duplicate_representative': is_representative,
            'selected': is_selected,
            'selection_rank': selected.index(local_index) + 1 if is_selected else np.nan,
            'selection_reason': selection_reason.get(local_index, ''),
            'time_gap_relaxed': time_relaxed.get(local_index, False),
            'nearest_selected_dhash_distance': nearest_selected,
            'exclusion_reason': exclusion_reason,
        })
    return rows


def build_selection_manifest(manifest):
    manifest = add_style_fields(manifest)
    dhashes = []
    capture_times = []
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        image = read_image(os.path.join(DATA_DIR, row.processed_path))
        dhashes.append(dhash64(image))
        capture_times.append(parse_capture_time(row.image_name))
        if position % 500 == 0 or position == len(manifest):
            print(f'dHash 进度: {position}/{len(manifest)}')

    manifest['dhash_int'] = dhashes
    manifest['dhash_hex'] = [f'{value:016x}' for value in dhashes]
    manifest['capture_time_epoch'] = [value[0] for value in capture_times]
    manifest['capture_time'] = [value[1] for value in capture_times]
    manifest['capture_time_source'] = [value[2] for value in capture_times]
    manifest['quality_status'] = 'usable'
    manifest['quality_notes'] = np.where(
        manifest['crop_status'].eq('fallback_full_frame'),
        'fallback_full_frame',
        '',
    )

    selection_rows = []
    for _, patient_group in manifest.groupby('patient_id', sort=True):
        selection_rows.extend(choose_patient_images(patient_group))
    selection = pd.DataFrame(selection_rows).set_index('dataframe_index')
    result = manifest.join(selection).sort_index()
    result['selection_rank'] = result['selection_rank'].astype('Int64')
    result['review_status'] = 'pending'
    return result


def patient_summary(selection_manifest):
    rows = []
    for patient_id, group in selection_manifest.groupby('patient_id', sort=True):
        representatives = group['is_duplicate_representative'].sum()
        rows.append({
            'patient_id': patient_id,
            'label': int(group['label'].iloc[0]),
            'center': group['center'].iloc[0],
            'original_image_count': len(group),
            'duplicate_group_count': int(group['duplicate_group'].nunique()),
            'near_duplicate_count': int(len(group) - representatives),
            'selected_image_count': int(group['selected'].sum()),
            'time_parsed_count': int(group['capture_time'].astype(bool).sum()),
            'time_gap_relaxed_count': int(group['time_gap_relaxed'].sum()),
            'review_status': 'pending',
        })
    return pd.DataFrame(rows)


def write_outputs(selection_manifest, record_dir):
    os.makedirs(record_dir, exist_ok=True)
    selection_path = os.path.join(record_dir, 'selection_manifest.csv')
    limited_path = os.path.join(record_dir, 'patient_limited_manifest.csv')
    patient_path = os.path.join(record_dir, 'patient_selection_summary.csv')
    summary_path = os.path.join(record_dir, '筛图汇总.txt')

    selection_manifest.to_csv(selection_path, index=False, encoding='utf-8-sig')
    limited = selection_manifest.loc[selection_manifest['selected']].copy()
    limited = limited.sort_values(['patient_id', 'selection_rank'], kind='stable')
    limited.to_csv(limited_path, index=False, encoding='utf-8-sig')
    patients = patient_summary(selection_manifest)
    patients.to_csv(patient_path, index=False, encoding='utf-8-sig')

    duplicate_images = int((~selection_manifest['is_duplicate_representative']).sum())
    parsed_times = int(selection_manifest['capture_time'].astype(bool).sum())
    summary = (
        f'输入图片: {len(selection_manifest)}\n'
        f'输入患者: {selection_manifest.patient_id.nunique()}\n'
        f'近重复图片: {duplicate_images}\n'
        f'入选图片: {len(limited)}\n'
        f'时间戳可解析图片: {parsed_times}\n'
        f'使用时间间隔放宽的入选图片: {int(selection_manifest.time_gap_relaxed.sum())}\n'
        f'每患者最大入选数: {int(patients.selected_image_count.max())}\n'
        f'待人工审核状态: pending\n'
    )
    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write(summary)
    print('\n' + summary)
    print(f'选择记录: {selection_path}')
    print(f'入选清单: {limited_path}')


def main():
    args = parse_args()
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    if os.path.exists(record_dir):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {record_dir}')
        shutil.rmtree(record_dir)

    manifest = pd.read_csv(MANIFEST_PATH, encoding='utf-8-sig')
    if args.mode == 'debug':
        patient_ids = select_debug_patients(
            manifest,
            args.debug_patients_per_group,
            args.debug_large_patients,
        )
        manifest = manifest.loc[manifest['patient_id'].isin(patient_ids)].copy()
    manifest = manifest.sort_values(['patient_id', 'processed_path'], kind='stable').reset_index(drop=True)
    selection_manifest = build_selection_manifest(manifest)
    write_outputs(selection_manifest, record_dir)


if __name__ == '__main__':
    main()
