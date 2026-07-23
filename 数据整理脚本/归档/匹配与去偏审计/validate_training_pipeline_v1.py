"""独立验收 P6 冻结清单、增强和双模型阶段配置。"""

import os
import sys

import pandas as pd
import torch
from PIL import Image


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_DIR, '程序', '模型训练', '正式代码')
SPLIT_DIR = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '训练划分_v1')
sys.path.insert(0, CODE_DIR)

import efficientnet_train_debiased as efficientnet_entry  # noqa: E402
import resnet_train_debiased as resnet_entry  # noqa: E402
from train_utils import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    build_transforms,
    load_frozen_manifest,
    seed_everything,
)


MANIFESTS = {
    'full': ('full_split_seed42.csv', None),
    'relaxed_1to1': ('relaxed_all_cases_1to1_split_seed42.csv', None),
    'relaxed_1to2': ('relaxed_all_cases_1to2_split_seed42.csv', None),
    'strict': ('strict_common_support_1to1_folds5_seed42.csv', 0),
}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    checks = []
    loaded = {}
    for name, (filename, fold) in MANIFESTS.items():
        try:
            frame, validation_fold = load_frozen_manifest(
                os.path.join(SPLIT_DIR, filename),
                DEFAULT_IMAGE_ROOT,
                fold=fold,
            )
            loaded[name] = frame
            checks.extend([
                (f'{name}:load', True, ''),
                (f'{name}:three_splits', set(frame.split.unique()) == {'train', 'val', 'test'}, ''),
                (f'{name}:validation_fold', name != 'strict' or validation_fold == 1, ''),
            ])
        except Exception as error:
            checks.append((f'{name}:load', False, str(error)))

    seed_everything(42)
    sample_path = os.path.join(
        DEFAULT_IMAGE_ROOT,
        loaded['strict'].iloc[0]['processed_path'],
    )
    with Image.open(sample_path) as source:
        image = source.convert('RGB')
    train_transform, eval_transform = build_transforms()
    train_a = train_transform(image.copy())
    train_b = train_transform(image.copy())
    eval_a = eval_transform(image.copy())
    eval_b = eval_transform(image.copy())
    checks.extend([
        ('augmentation:shape', tuple(train_a.shape) == (3, 224, 224), ''),
        ('augmentation:finite', bool(torch.isfinite(train_a).all()), ''),
        ('augmentation:train_stochastic', not torch.equal(train_a, train_b), ''),
        ('augmentation:eval_deterministic', torch.equal(eval_a, eval_b), ''),
    ])

    for name, entry in [('resnet50', resnet_entry), ('efficientnet_b0', efficientnet_entry)]:
        model = entry.build_model(pretrained=False)
        entry.set_trainable_stage1(model)
        stage1 = parameter_count(model)
        entry.set_trainable_stage2(model)
        stage2 = parameter_count(model)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        checks.extend([
            (f'{name}:stage1_trainable', stage1 > 0, f'count={stage1}'),
            (f'{name}:stage2_expands_trainable', stage2 > stage1, f'{stage1}->{stage2}'),
            (f'{name}:forward_shape', tuple(output.shape) == (1, 2), str(tuple(output.shape))),
            (f'{name}:forward_finite', bool(torch.isfinite(output).all()), ''),
        ])

    result = pd.DataFrame(checks, columns=['check', 'passed', 'detail'])
    result.to_csv(
        os.path.join(SPLIT_DIR, 'P6训练管线自动验收.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    passed = bool(result.passed.all())
    text = (
        f'自动验收结论: {"PASS" if passed else "FAIL"}\n'
        f'检查项: {len(result)}\n'
        f'失败项: {int((~result.passed).sum())}\n'
    )
    with open(
        os.path.join(SPLIT_DIR, 'P6训练管线自动验收.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(text)
    print(text)
    if not passed:
        print(result.loc[~result.passed].to_string(index=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
