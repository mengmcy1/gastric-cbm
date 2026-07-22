"""生成 P2 患者内去重与限图的风险导向复核图。"""

import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
RECORD_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1')
RANDOM_SEED = 42
LARGE_PATIENT_COUNT = 10
PANEL_SIZE = 224
TITLE_HEIGHT = 58
HEADER_HEIGHT = 42
GRID_COLUMNS = 4


def parse_args():
    parser = argparse.ArgumentParser(description='生成 P2 筛图复核图')
    parser.add_argument('--mode', choices=('debug', 'full'), default='debug')
    return parser.parse_args()


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({'true': True, 'false': False})


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'OpenCV 无法解码图片: {path}')
    return image


def select_patients(manifest):
    reasons = {}

    def add(patient_ids, reason):
        for patient_id in patient_ids:
            reasons.setdefault(patient_id, set()).add(reason)

    duplicate_patients = manifest.loc[
        manifest['duplicate_group_size'].gt(1), 'patient_id'
    ].unique()
    relaxed_patients = manifest.loc[
        bool_series(manifest['time_gap_relaxed']), 'patient_id'
    ].unique()
    add(duplicate_patients, 'duplicate')
    add(relaxed_patients, 'time_relaxed')

    counts = manifest.groupby('patient_id').size().sort_values(ascending=False)
    add(counts.head(LARGE_PATIENT_COUNT).index, 'large_patient')

    patient_rows = (
        manifest.groupby('patient_id', as_index=False)
        .agg(label=('label', 'first'), center=('center', 'first'))
    )
    for offset, (_, group) in enumerate(patient_rows.groupby(['center', 'label'], sort=True)):
        sample = group.sample(n=1, random_state=RANDOM_SEED + offset)
        add(sample['patient_id'], 'stratified_sample')
    return reasons


def panel_for_row(row):
    image = read_image(os.path.join(DATA_DIR, row.processed_path))
    image = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    selected = bool(row.selected)
    representative = bool(row.is_duplicate_representative)
    if selected:
        color = (40, 180, 40)
        status = f'SEL#{int(row.selection_rank)}'
    elif not representative:
        color = (0, 140, 255)
        status = 'DROP:DUP'
    else:
        color = (150, 150, 150)
        status = 'DROP:LIMIT'
    cv2.rectangle(image, (1, 1), (PANEL_SIZE - 2, PANEL_SIZE - 2), color, 4)

    canvas = np.full((PANEL_SIZE + TITLE_HEIGHT, PANEL_SIZE, 3), 255, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = image
    duplicate_number = str(row.duplicate_group).rsplit('dup', 1)[-1]
    line1 = (
        f'{status} dup={duplicate_number}/{int(row.duplicate_group_size)} '
        f'sharp={float(row.sharpness):.0f}'
    )
    line2 = os.path.basename(str(row.processed_path))[:31]
    cv2.putText(canvas, line1, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    cv2.putText(canvas, line2, (4, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1)
    return canvas


def patient_sheet(number, group, reasons):
    group = group.sort_values(
        ['selected', 'selection_rank', 'processed_path'],
        ascending=[False, True, True],
        kind='stable',
        na_position='last',
    )
    panels = [panel_for_row(row) for row in group.itertuples(index=False)]
    panel_height = PANEL_SIZE + TITLE_HEIGHT
    rows = (len(panels) + GRID_COLUMNS - 1) // GRID_COLUMNS
    sheet = np.full(
        (HEADER_HEIGHT + rows * panel_height, GRID_COLUMNS * PANEL_SIZE, 3),
        245,
        dtype=np.uint8,
    )
    header = (
        f'patient={number:03d} y={int(group.label.iloc[0])} n={len(group)} '
        f'selected={int(bool_series(group.selected).sum())} reasons={"|".join(sorted(reasons))}'
    )
    cv2.putText(sheet, header, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 1)
    for index, panel in enumerate(panels):
        row = index // GRID_COLUMNS
        column = index % GRID_COLUMNS
        y1 = HEADER_HEIGHT + row * panel_height
        x1 = column * PANEL_SIZE
        sheet[y1:y1 + panel_height, x1:x1 + PANEL_SIZE] = panel
    return sheet


def main():
    args = parse_args()
    record_dir = os.path.join(RECORD_ROOT, args.mode)
    manifest_path = os.path.join(record_dir, 'selection_manifest.csv')
    output_dir = os.path.join(record_dir, '患者筛图对照')
    review_path = os.path.join(record_dir, '患者筛图复核清单.csv')
    manifest = pd.read_csv(manifest_path, encoding='utf-8-sig')
    manifest['selected'] = bool_series(manifest['selected'])
    manifest['is_duplicate_representative'] = bool_series(
        manifest['is_duplicate_representative']
    )
    patient_reasons = select_patients(manifest)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    review_rows = []
    for number, patient_id in enumerate(sorted(patient_reasons), start=1):
        group = manifest.loc[manifest['patient_id'].eq(patient_id)].copy()
        filename = f'患者筛图_{number:03d}.jpg'
        sheet = patient_sheet(number, group, patient_reasons[patient_id])
        success, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise ValueError(f'无法编码复核图: {patient_id}')
        encoded.tofile(os.path.join(output_dir, filename))
        review_rows.append({
            'review_number': number,
            'patient_id': patient_id,
            'label': int(group['label'].iloc[0]),
            'center': group['center'].iloc[0],
            'image_count': len(group),
            'selected_count': int(group['selected'].sum()),
            'review_reason': '|'.join(sorted(patient_reasons[patient_id])),
            'review_image': filename,
        })
    pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding='utf-8-sig')
    print(f'复核患者: {len(review_rows)}')
    print(f'复核清单: {review_path}')
    print(f'患者对照图: {output_dir}')


if __name__ == '__main__':
    main()
