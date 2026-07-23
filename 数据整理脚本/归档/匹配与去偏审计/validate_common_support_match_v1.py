"""独立验收 P4 严格共同支持匹配集。"""

import os

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_PATH = os.path.join(
    PROJECT_DIR,
    '数据整理记录',
    '第二批',
    '筛图_v1_1',
    'full',
    'patient_limited_manifest.csv',
)
RECORD_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')


def main():
    limited = pd.read_csv(SELECTION_PATH, encoding='utf-8-sig')
    pairs = pd.read_csv(
        os.path.join(RECORD_DIR, 'strict_common_support_pairs_1to1_seed42.csv'),
        encoding='utf-8-sig',
    )
    manifest = pd.read_csv(
        os.path.join(RECORD_DIR, 'matched_weak_strict_common_support_v1_seed42.csv'),
        encoding='utf-8-sig',
    )
    failures = []
    if pairs.case_patient_id.duplicated().any():
        failures.append('reused_case')
    if pairs.control_patient_id.duplicated().any():
        failures.append('reused_control')
    if set(pairs.case_patient_id) & set(pairs.control_patient_id):
        failures.append('patient_in_both_roles')
    if pairs.match_group_id.duplicated().any():
        failures.append('duplicate_match_group')
    if manifest.processed_path.duplicated().any():
        failures.append('duplicate_image')

    source_paths = set(limited.processed_path)
    if not set(manifest.processed_path).issubset(source_paths):
        failures.append('image_not_in_patient_limited')
    source_meta = limited.set_index('processed_path')
    for row in manifest.itertuples(index=False):
        source = source_meta.loc[row.processed_path]
        if source.patient_id != row.patient_id or int(source.label) != int(row.label):
            failures.append('image_metadata_mismatch')
        if row.style_group != row.strict_style_group:
            failures.append('image_style_mismatch')

    for pair in pairs.itertuples(index=False):
        group = manifest.loc[manifest.match_group_id.eq(pair.match_group_id)]
        cases = group.loc[group.match_role.eq('case')]
        controls = group.loc[group.match_role.eq('control')]
        if set(cases.patient_id) != {pair.case_patient_id}:
            failures.append('wrong_case_patient')
        if set(controls.patient_id) != {pair.control_patient_id}:
            failures.append('wrong_control_patient')
        if not cases.label.eq(1).all() or not controls.label.eq(0).all():
            failures.append('wrong_labels')
        if len(cases) != len(controls) or len(cases) != int(pair.matched_images_per_role):
            failures.append('unequal_pair_image_count')
        if not group.style_group.eq(pair.shared_style_group).all():
            failures.append('pair_style_mismatch')
        if group.center.nunique() != 1 or group.center.iloc[0] != pair.center:
            failures.append('cross_center')

    failures = sorted(set(failures))
    conclusion = 'PASS' if not failures else 'FAIL'
    summary = (
        f'严格共同支持集验收: {conclusion}\n'
        f'匹配对: {len(pairs)}\n'
        f'患者数: {manifest.patient_id.nunique()}\n'
        f'图像数: {len(manifest)}\n'
        f'失败项: {len(failures)}\n'
    )
    with open(
        os.path.join(RECORD_DIR, '严格共同支持集自动验收.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(summary)
    print(summary)
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
