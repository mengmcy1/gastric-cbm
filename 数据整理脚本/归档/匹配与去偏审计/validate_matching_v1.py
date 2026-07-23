"""独立验收 P4 患者级级联匹配及冻结清单。"""

import argparse
import os

import pandas as pd


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '筛图_v1_1', 'full')
RECORD_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
YEAR_TOLERANCE = 2.0
SIZE_ORDER = {
    'small_le512': 0,
    'medium_513_1024': 1,
    'large_1025_1600': 2,
    'xlarge_gt1600': 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description='验收第二批患者级匹配 v1')
    return parser.parse_args()


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({'true': True, 'false': False})


def expected_level(case, control):
    same_style = case.patient_style_group == control.patient_style_group
    year_missing = pd.isna(case.patient_year) or pd.isna(control.patient_year)
    year_distance = 25.0 if year_missing else min(abs(case.patient_year - control.patient_year), 25.0)
    if same_style and not year_missing and year_distance <= YEAR_TOLERANCE:
        return 1, same_style, year_missing, year_distance
    if same_style:
        return 2, same_style, year_missing, year_distance
    return 3, same_style, year_missing, year_distance


def validate_pairs(pairs, signatures, ratio):
    failures = []
    signature_map = signatures.set_index('patient_id')
    expected_cases = set(signatures.loc[signatures.label.eq(1), 'patient_id'])
    if set(pairs['case_patient_id']) != expected_cases:
        failures.append(f'1to{ratio}:case_coverage')
    case_counts = pairs.groupby('case_patient_id').size()
    if not case_counts.eq(ratio).all():
        failures.append(f'1to{ratio}:case_pair_count')
    expected_slots = set(range(1, ratio + 1))
    if any(set(group['control_slot']) != expected_slots for _, group in pairs.groupby('case_patient_id')):
        failures.append(f'1to{ratio}:control_slots')
    if pairs['control_patient_id'].duplicated().any():
        failures.append(f'1to{ratio}:reused_control')
    if pairs['pair_id'].duplicated().any() or pairs['match_group_id'].nunique() != len(expected_cases):
        failures.append(f'1to{ratio}:duplicate_match_ids')

    for row in pairs.itertuples(index=False):
        case = signature_map.loc[row.case_patient_id]
        control = signature_map.loc[row.control_patient_id]
        if int(case.label) != 1 or int(control.label) != 0:
            failures.append(f'1to{ratio}:wrong_labels')
        if case.center != control.center or row.center != case.center:
            failures.append(f'1to{ratio}:cross_center')
        level, same_style, year_missing, distance_year = expected_level(case, control)
        size_distance = abs(
            SIZE_ORDER[case.patient_size_group] - SIZE_ORDER[control.patient_size_group]
        )
        if int(row.match_level) != level:
            failures.append(f'1to{ratio}:wrong_match_level')
        if bool(row.same_style_group) != same_style:
            failures.append(f'1to{ratio}:wrong_same_style')
        if bool(row.year_missing_in_pair) != year_missing:
            failures.append(f'1to{ratio}:wrong_year_missing')
        if abs(float(row.year_distance) - distance_year) > 1e-8:
            failures.append(f'1to{ratio}:wrong_year_distance')
        if int(row.size_group_distance) != size_distance:
            failures.append(f'1to{ratio}:wrong_size_distance')
        if int(row.aspect_group_mismatch) != int(
            case.patient_aspect_group != control.patient_aspect_group
        ):
            failures.append(f'1to{ratio}:wrong_aspect_mismatch')
        if int(row.frame_profile_mismatch) != int(
            case.patient_frame_profile != control.patient_frame_profile
        ):
            failures.append(f'1to{ratio}:wrong_frame_mismatch')
    return sorted(set(failures))


def validate_matched_images(manifest, pairs, limited, ratio):
    failures = []
    expected_patients = set(pairs.case_patient_id) | set(pairs.control_patient_id)
    if set(manifest.patient_id) != expected_patients:
        failures.append(f'1to{ratio}:image_patient_coverage')
    if manifest['processed_path'].duplicated().any():
        failures.append(f'1to{ratio}:duplicate_image_paths')
    expected_paths = set(limited.loc[limited.patient_id.isin(expected_patients), 'processed_path'])
    if set(manifest.processed_path) != expected_paths:
        failures.append(f'1to{ratio}:image_path_coverage')
    roles = manifest.groupby('patient_id')['match_role'].nunique()
    if not roles.eq(1).all():
        failures.append(f'1to{ratio}:mixed_match_role')
    case_roles = set(manifest.loc[manifest.match_role.eq('case'), 'patient_id'])
    control_roles = set(manifest.loc[manifest.match_role.eq('control'), 'patient_id'])
    if case_roles != set(pairs.case_patient_id) or control_roles != set(pairs.control_patient_id):
        failures.append(f'1to{ratio}:wrong_match_roles')
    if not manifest.review_status.eq('pending').all():
        failures.append(f'1to{ratio}:invalid_review_status')
    return failures


def main():
    parse_args()
    selection = pd.read_csv(
        os.path.join(SELECTION_ROOT, 'selection_manifest.csv'), encoding='utf-8-sig'
    )
    selection['selected'] = bool_series(selection['selected'])
    source_limited = pd.read_csv(
        os.path.join(SELECTION_ROOT, 'patient_limited_manifest.csv'), encoding='utf-8-sig'
    )
    signatures = pd.read_csv(
        os.path.join(RECORD_DIR, 'patient_signatures.csv'), encoding='utf-8-sig'
    )
    full = pd.read_csv(os.path.join(RECORD_DIR, 'full_manifest.csv'), encoding='utf-8-sig')
    limited = pd.read_csv(
        os.path.join(RECORD_DIR, 'patient_limited_manifest.csv'), encoding='utf-8-sig'
    )
    unmatched = pd.read_csv(
        os.path.join(RECORD_DIR, 'usable_unmatched_manifest.csv'), encoding='utf-8-sig'
    )
    excluded = pd.read_csv(
        os.path.join(RECORD_DIR, 'excluded_manifest.csv'), encoding='utf-8-sig'
    )

    failures = []
    if signatures.patient_id.duplicated().any() or len(signatures) != 2321:
        failures.append('signature_patient_coverage')
    if set(full.processed_path) != set(selection.processed_path):
        failures.append('full_manifest_coverage')
    if set(limited.processed_path) != set(source_limited.processed_path):
        failures.append('limited_manifest_coverage')
    if set(excluded.processed_path) != set(selection.loc[~selection.selected, 'processed_path']):
        failures.append('excluded_manifest_coverage')

    for ratio, pair_name, manifest_name in [
        (1, 'matched_pairs_1to1_seed42.csv', 'matched_weak_relaxed_all_cases_1to1_seed42.csv'),
        (2, 'matched_pairs_1to2_seed42.csv', 'matched_weak_relaxed_all_cases_1to2_seed42.csv'),
    ]:
        pairs = pd.read_csv(os.path.join(RECORD_DIR, pair_name), encoding='utf-8-sig')
        matched = pd.read_csv(os.path.join(RECORD_DIR, manifest_name), encoding='utf-8-sig')
        failures.extend(validate_pairs(pairs, signatures, ratio))
        failures.extend(validate_matched_images(matched, pairs, limited, ratio))
        if ratio == 1:
            matched_ids = set(matched.patient_id)
            expected_unmatched = set(limited.patient_id) - matched_ids
            if set(unmatched.patient_id) != expected_unmatched:
                failures.append('unmatched_patient_partition')
            if set(unmatched.processed_path) & set(matched.processed_path):
                failures.append('matched_unmatched_image_overlap')

    failures = sorted(set(failures))
    checks = pd.DataFrame({
        'check': ['matching_validation'],
        'status': ['PASS' if not failures else 'FAIL'],
        'details': ['' if not failures else '|'.join(failures)],
    })
    checks.to_csv(
        os.path.join(RECORD_DIR, '自动验收检查.csv'), index=False, encoding='utf-8-sig'
    )
    conclusion = 'PASS' if not failures else 'FAIL'
    summary = (
        f'自动验收结论: {conclusion}\n'
        f'患者签名: {len(signatures)}\n'
        f'1:1 匹配患者: 830\n'
        f'1:2 匹配患者: 1245\n'
        f'失败项: {len(failures)}\n'
    )
    with open(os.path.join(RECORD_DIR, '自动验收汇总.txt'), 'w', encoding='utf-8') as file:
        file.write(summary)
    print(summary)
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
