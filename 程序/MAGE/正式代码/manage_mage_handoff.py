#!/usr/bin/env python3
"""校验交接权重，或为受控数据目录恢复原项目路径。不会删除或覆盖文件。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify-assets', 'restore-data', 'check-data'])
    parser.add_argument('--data-root', type=Path, default=ROOT / '数据/教师学生模型交接数据')
    parser.add_argument('--catalog', type=Path, default=ROOT / '模型资产/mage_handoff.json',
                        help='verify-assets所用资产清单；SAE可指定sae_handoff.json')
    args = parser.parse_args()
    if args.action == 'verify-assets':
        catalog = json.loads(args.catalog.read_text())
        for asset in catalog['assets']:
            path = ROOT / asset['path']
            if path.stat().st_size != asset['size_bytes']:
                raise ValueError(f'权重大小不符: {path}')
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest != asset['sha256']:
                raise ValueError(f'权重校验不符: {path}')
            print(f"通过: {asset['role']}")
        return
    data_root = args.data_root.resolve()
    if args.action == 'restore-data':
        mapping = pd.read_csv(data_root / '原路径兼容映射.csv')
        prefix = Path('数据/教师学生模型交接数据')
        planned = []
        for row in mapping.itertuples():
            old = Path(row.old_path)
            relative_new = Path(row.new_path).relative_to(prefix)
            if old.is_absolute() or '..' in old.parts or '..' in relative_new.parts:
                raise ValueError('映射不允许绝对路径或上级路径')
            target = data_root / relative_new
            if not target.exists():
                raise FileNotFoundError(target)
            old = ROOT / old
            if old.exists() or old.is_symlink():
                if old.resolve() != target.resolve():
                    raise FileExistsError(f'原路径已有不同文件，未覆盖: {old}')
            else:
                planned.append((old, target, row.kind))
        for old, target, kind in planned:
            old.parent.mkdir(parents=True, exist_ok=True)
            old.symlink_to(os.path.relpath(target, old.parent), target_is_directory=kind == 'directory')
        print(f'映射检查通过，新增兼容链接 {len(planned)} 个；图片未复制')
    else:
        total = 0
        for folder in ['01_训练集', '02_验证集', '03_留出测试集', '04_内部时间外测试集', '05_外部多中心测试集']:
            frame = pd.read_csv(data_root / folder / '图片与标注清单.csv', dtype={'patient_id': str})
            for path in frame.image_path:
                if not (data_root / folder / path).is_file():
                    raise FileNotFoundError(path)
            print(f'{folder}: {len(frame)}图 / {frame.patient_id.nunique()}人；路径全部存在')
            total += len(frame)
        print(f'合计 {total} 张正式输入图；没有运行测试集预测')


if __name__ == '__main__':
    main()
