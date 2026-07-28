"""Build, verify, and package the project's private model assets."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_DIR / '结果'
ASSET_DIR = PROJECT_DIR / '模型资产'
RELEASE_DIR = PROJECT_DIR / '发布资产' / '模型权重'

RUNTIME_PATHS = {
    '结果/模型权重/resnet50_transfer_best.pth',
    '结果/模型权重/efficientnet_b0_best.pth',
    (
        '结果/去偏重训练_v1/expA_full_resnet50_seed42/'
        'resnet50_debiased_best.pth'
    ),
    (
        '结果/去偏重训练_v1/expA_full_efficientnet_b0_seed42/'
        'efficientnet_b0_debiased_best.pth'
    ),
    (
        '结果/SAE/去偏重训练_v1/resnet50/'
        'formal_expA_resnet50_sae_l1_0005_20260724/SAE模型/sae_best.pth'
    ),
}

COMPANION_NAMES = {
    'config.json',
    'locked_threshold_summary.csv',
    'locked_threshold_summary.json',
    'metrics.json',
    'pruning_summary.json',
    'training_history.csv',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package_name(relative_path):
    text = relative_path.as_posix()
    if text in RUNTIME_PATHS:
        return 'runtime_v1'
    if text.startswith('结果/去偏重训练_v1/'):
        return 'classifier_reproducibility_v1'
    return 'sae_experiments_v1'


def role(relative_path):
    text = relative_path.as_posix()
    if text in RUNTIME_PATHS:
        return 'runtime'
    if '早期实现_不可与修正后比较' in text:
        return 'historical_incompatible'
    if '/归档/' in text or '/debug_' in text:
        return 'historical'
    return 'research_reproducibility'


def discover_assets():
    paths = sorted(RESULT_DIR.rglob('*.pth'))
    records = []
    for path in paths:
        relative = path.relative_to(PROJECT_DIR)
        records.append({
            'path': relative.as_posix(),
            'size_bytes': path.stat().st_size,
            'sha256': sha256(path),
            'package': package_name(relative),
            'role': role(relative),
        })
    return records


def write_manifest(records):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ASSET_DIR / '模型资产清单.json'
    csv_path = ASSET_DIR / '模型资产清单.csv'
    total_bytes = sum(item['size_bytes'] for item in records)
    payload = {
        'schema_version': 1,
        'weight_count': len(records),
        'total_bytes': total_bytes,
        'assets': records,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['path', 'size_bytes', 'sha256', 'package', 'role'],
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def companion_paths(weight_path):
    run_dir = weight_path.parent
    if run_dir.name == 'SAE模型':
        run_dir = run_dir.parent
    candidates = []
    for path in run_dir.rglob('*'):
        if not path.is_file():
            continue
        if path.name in COMPANION_NAMES:
            candidates.append(path)
    return sorted(candidates)


def create_packages(records, selected):
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    package_names = sorted({item['package'] for item in records})
    if selected != 'all':
        package_names = [selected]

    outputs = []
    for name in package_names:
        members = [
            PROJECT_DIR / item['path']
            for item in records
            if item['package'] == name
        ]
        extra = {
            companion
            for weight in members
            for companion in companion_paths(weight)
        }
        archive_path = RELEASE_DIR / f'{name}.tar'
        with tarfile.open(archive_path, 'w') as archive:
            for path in sorted(set(members) | extra):
                archive.add(path, arcname=path.relative_to(PROJECT_DIR))
            for manifest in ASSET_DIR.glob('模型资产清单.*'):
                archive.add(manifest, arcname=manifest.relative_to(PROJECT_DIR))
        outputs.append(archive_path)
    checksum_path = RELEASE_DIR / 'SHA256SUMS'
    checksum_path.write_text(
        ''.join(
            f'{sha256(path)}  {path.name}\n'
            for path in sorted(outputs)
        ),
        encoding='ascii',
    )
    return outputs


def verify_manifest():
    manifest_path = ASSET_DIR / '模型资产清单.json'
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures = []
    for item in payload['assets']:
        path = PROJECT_DIR / item['path']
        if not path.is_file():
            failures.append(f'缺失: {item["path"]}')
        elif path.stat().st_size != item['size_bytes']:
            failures.append(f'大小不一致: {item["path"]}')
        elif sha256(path) != item['sha256']:
            failures.append(f'SHA256不一致: {item["path"]}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'验证通过: {len(payload["assets"])} 个权重')


def main():
    parser = argparse.ArgumentParser(description='管理私有模型权重资产')
    parser.add_argument(
        '--verify', action='store_true',
        help='根据已有清单校验权重，不重写清单',
    )
    parser.add_argument(
        '--package',
        choices=[
            'all',
            'runtime_v1',
            'classifier_reproducibility_v1',
            'sae_experiments_v1',
        ],
        help='生成指定的GitHub Release归档',
    )
    args = parser.parse_args()

    if args.verify:
        verify_manifest()
        return

    records = discover_assets()
    json_path, csv_path = write_manifest(records)
    print(f'权重清单: {len(records)} 个, {sum(x["size_bytes"] for x in records)} bytes')
    print(f'JSON: {json_path}')
    print(f'CSV:  {csv_path}')

    if args.package:
        for path in create_packages(records, args.package):
            print(f'Release归档: {path} ({path.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
