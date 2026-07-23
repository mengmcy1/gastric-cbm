"""生成 P5 严格共同支持集全量与残余风险分层联系图。"""

import argparse
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
IMAGE_ROOT = os.path.join(PROJECT_DIR, '数据', '第二批裁剪后_v1_1')
STRICT_PATH = os.path.join(
    MATCH_ROOT, 'matched_weak_strict_common_support_v1_seed42.csv'
)
PIXEL_PATH = os.path.join(MATCH_ROOT, '像素统计基线_v1', '图像级低层统计.csv')
OUTPUT_DIR = os.path.join(MATCH_ROOT, 'P5视觉复核_v1')
RANDOM_SEED = 42
PANEL_SIZE = 256
TITLE_HEIGHT = 68
HEADER_HEIGHT = 42
GRID_COLUMNS = 4
PANELS_PER_SHEET = 16
EXTREME_COUNT_PER_LABEL = 8

RISK_FEATURES = [
    'roi_area_ratio',
    'contour_area_ratio',
    'black_pixel_ratio',
    'edge_density',
    'border_center_std_difference',
    'jpeg_blockiness',
    'saturation_mean',
]


def parse_args():
    parser = argparse.ArgumentParser(description='生成 P5 风险分层视觉复核材料 v1')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def read_image(relative_path):
    path = os.path.join(IMAGE_ROOT, relative_path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法解码图像: {path}')
    return image


def panel(row, review_number):
    image = read_image(row.processed_path)
    image = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    color = (20, 40, 210) if int(row.label) == 1 else (40, 160, 40)
    cv2.rectangle(image, (1, 1), (PANEL_SIZE - 2, PANEL_SIZE - 2), color, 4)
    canvas = np.full((PANEL_SIZE + TITLE_HEIGHT, PANEL_SIZE, 3), 255, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = image
    lines = [
        f'#{review_number:03d} {row.match_group_id} y={int(row.label)} {row.match_role}',
        f'roi={row.roi_area_ratio:.2f} contour={row.contour_area_ratio:.2f} black={row.black_pixel_ratio:.2f}',
        f'edge={row.edge_density:.3f} borderStd={row.border_center_std_difference:.1f}',
    ]
    for line_number, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (4, 18 + line_number * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (25, 25, 25),
            1,
        )
    return canvas


def write_sheets(frame, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    panel_height = PANEL_SIZE + TITLE_HEIGHT
    rows_per_sheet = PANELS_PER_SHEET // GRID_COLUMNS
    for page, start in enumerate(range(0, len(frame), PANELS_PER_SHEET), start=1):
        page_frame = frame.iloc[start:start + PANELS_PER_SHEET]
        sheet = np.full(
            (HEADER_HEIGHT + rows_per_sheet * panel_height, GRID_COLUMNS * PANEL_SIZE, 3),
            245,
            dtype=np.uint8,
        )
        header = f'{prefix} page={page:03d} images={start + 1}-{start + len(page_frame)}'
        cv2.putText(sheet, header, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 1)
        for offset, row in enumerate(page_frame.itertuples(index=False)):
            item = panel(row, start + offset + 1)
            grid_row = offset // GRID_COLUMNS
            grid_column = offset % GRID_COLUMNS
            y1 = HEADER_HEIGHT + grid_row * panel_height
            x1 = grid_column * PANEL_SIZE
            sheet[y1:y1 + panel_height, x1:x1 + PANEL_SIZE] = item
        path = os.path.join(output_dir, f'{prefix}_{page:03d}.jpg')
        success, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise ValueError(f'无法编码联系图: {path}')
        encoded.tofile(path)


def risk_selection(frame):
    reasons = defaultdict(set)
    for feature in RISK_FEATURES:
        for label, group in frame.groupby('label', sort=True):
            for index in group.nlargest(EXTREME_COUNT_PER_LABEL, feature).index:
                reasons[index].add(f'{feature}:high:y{label}')
            for index in group.nsmallest(EXTREME_COUNT_PER_LABEL, feature).index:
                reasons[index].add(f'{feature}:low:y{label}')
    strata = frame[['center', 'label', 'strict_style_group']].drop_duplicates()
    for offset, stratum in strata.reset_index(drop=True).iterrows():
        candidates = frame.loc[
            frame['center'].eq(stratum.center)
            & frame['label'].eq(stratum.label)
            & frame['strict_style_group'].eq(stratum.strict_style_group)
        ]
        selected = candidates.sample(n=1, random_state=RANDOM_SEED + offset)
        reasons[selected.index[0]].add('center_label_style_sample')
    selected = frame.loc[sorted(reasons)].copy()
    selected['review_reason'] = ['|'.join(sorted(reasons[index])) for index in selected.index]
    return selected


def add_review_order(frame):
    result = frame.reset_index(drop=True).copy()
    result.insert(0, 'review_number', np.arange(1, len(result) + 1))
    return result


def main():
    args = parse_args()
    if os.path.exists(OUTPUT_DIR):
        if not args.overwrite:
            raise FileExistsError(f'输出目录已存在，请使用 --overwrite: {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    strict = pd.read_csv(STRICT_PATH, encoding='utf-8-sig')
    pixel = pd.read_csv(PIXEL_PATH, encoding='utf-8-sig')
    strict = strict.merge(pixel, on='processed_path', how='left', validate='one_to_one')
    if strict[RISK_FEATURES].isna().any().any():
        raise ValueError('严格集存在缺失的风险统计')
    strict['role_order'] = strict['match_role'].map({'case': 0, 'control': 1})
    strict = strict.sort_values(
        ['match_group_id', 'role_order', 'selection_rank', 'processed_path'], kind='stable'
    ).drop(columns='role_order')
    strict = add_review_order(strict)

    all_dir = os.path.join(OUTPUT_DIR, '严格集全量联系图')
    write_sheets(strict, all_dir, 'strict_all')
    strict.to_csv(
        os.path.join(OUTPUT_DIR, '严格集全量复核清单.csv'), index=False, encoding='utf-8-sig'
    )

    risk = risk_selection(strict).sort_values(
        ['match_group_id', 'match_role', 'selection_rank'], kind='stable'
    )
    risk = add_review_order(risk.drop(columns='review_number'))
    risk_dir = os.path.join(OUTPUT_DIR, '风险分层联系图')
    write_sheets(risk, risk_dir, 'risk_stratified')
    risk.to_csv(
        os.path.join(OUTPUT_DIR, '风险分层复核清单.csv'), index=False, encoding='utf-8-sig'
    )
    summary = (
        f'严格集全量图像: {len(strict)}\n'
        f'严格集全量联系图: {(len(strict) + PANELS_PER_SHEET - 1) // PANELS_PER_SHEET}\n'
        f'风险分层图像: {len(risk)}\n'
        f'风险分层联系图: {(len(risk) + PANELS_PER_SHEET - 1) // PANELS_PER_SHEET}\n'
    )
    with open(os.path.join(OUTPUT_DIR, '视觉复核汇总.txt'), 'w', encoding='utf-8') as file:
        file.write(summary)
    print(summary)
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
