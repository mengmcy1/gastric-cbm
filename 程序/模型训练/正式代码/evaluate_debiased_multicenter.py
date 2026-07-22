"""使用 v1.1 裁剪和重训练权重评估独立多中心癌数据。"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SCRIPT_DIR = PROJECT_DIR / '数据整理脚本'
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_SCRIPT_DIR))

from efficientnet_train_debiased import build_model as build_efficientnet  # noqa: E402
from preprocess_second_batch_v1_1 import (  # noqa: E402
    OUTPUT_SIZE,
    detect_roi,
    draw_preview,
    read_image,
    refine_roi_by_edge_scan,
    resize_square,
    save_contact_sheets,
    write_jpeg,
)
from resnet_train_debiased import build_model as build_resnet  # noqa: E402
from train_utils import build_transforms  # noqa: E402


DEFAULT_CANCER_DIR = PROJECT_DIR / '数据' / '胃镜多中心早癌图片挑选'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / '结果' / '去偏重训练_v1' / '外部多中心验证_v1'
DEFAULT_RUNS = {
    'resnet50': PROJECT_DIR / '结果' / '去偏重训练_v1' / 'expA_full_resnet50_seed42',
    'efficientnet_b0': (
        PROJECT_DIR / '结果' / '去偏重训练_v1'
        / 'expA_full_efficientnet_b0_seed42'
    ),
}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExternalProcessedDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        with Image.open(self.dataframe.iloc[index]['processed_path']) as source:
            image = source.convert('RGB')
        return self.transform(image), index


def parse_args():
    parser = argparse.ArgumentParser(description='裁剪重训练模型的多中心外部验证')
    parser.add_argument('--cancer-dir', default=str(DEFAULT_CANCER_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--model', choices=('all', 'resnet50', 'efficientnet_b0'),
                        default='all')
    parser.add_argument('--resnet-run-dir', default=str(DEFAULT_RUNS['resnet50']))
    parser.add_argument('--efficientnet-run-dir',
                        default=str(DEFAULT_RUNS['efficientnet_b0']))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--overwrite-preprocessed', action='store_true')
    return parser.parse_args()


def scan_cancer_directory(root):
    root = Path(root)
    records = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image_paths = sorted(
            path for path in patient_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        for image_index, image_path in enumerate(image_paths, start=1):
            records.append({
                'patient_id': patient_dir.name,
                'label': 1,
                'image_name': image_path.name,
                'image_index': image_index,
                'original_path': str(image_path.resolve()),
            })
    return pd.DataFrame(records)


def preprocess_images(manifest, output_dir, overwrite):
    processed_dir = output_dir / '裁剪后图像_v1_1'
    manifest_path = output_dir / 'preprocess_manifest.csv'
    if manifest_path.exists() and not overwrite:
        cached = pd.read_csv(manifest_path, encoding='utf-8-sig', dtype={'patient_id': str})
        same_sources = (
            len(cached) == len(manifest)
            and cached['original_path'].tolist() == manifest['original_path'].tolist()
        )
        outputs_exist = cached['processed_path'].map(
            lambda path: Path(path).exists()
        ).all()
        if same_sources and outputs_exist:
            print(f'复用已裁剪外部数据: {len(cached)} 张')
            return cached

    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        original = read_image(row.original_path)
        original_h, original_w = original.shape[:2]
        roi = detect_roi(original)
        bbox, refine = refine_roi_by_edge_scan(original, roi['bbox'])
        x1, y1, x2, y2 = bbox
        cropped = original[y1:y2, x1:x2]
        processed = resize_square(cropped, OUTPUT_SIZE)
        destination = (
            processed_dir / row.patient_id
            / f'{row.image_index:04d}_{Path(row.image_name).stem}.jpg'
        )
        write_jpeg(str(destination), processed)
        rows.append({
            **row._asdict(),
            'processed_path': str(destination.resolve()),
            'original_width': original_w,
            'original_height': original_h,
            'crop_x1': x1,
            'crop_y1': y1,
            'crop_x2': x2,
            'crop_y2': y2,
            'crop_status': 'cropped' if refine['edge_refined'] else roi['crop_status'],
            'crop_method': (
                roi['crop_method'] + '+edge_scan_v1_1'
                if refine['edge_refined'] else roi['crop_method']
            ),
            'failure_reason': roi['failure_reason'],
            **refine,
        })
        if number % 100 == 0 or number == len(manifest):
            print(f'外部数据裁剪: {number}/{len(manifest)}')

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    save_crop_review(result, output_dir / '裁剪复核')
    return result


def save_crop_review(manifest, output_dir):
    dimensions = np.minimum(manifest['original_width'], manifest['original_height'])
    trim_ratio = (
        manifest['edge_trim_left'] + manifest['edge_trim_right']
        + manifest['edge_trim_top'] + manifest['edge_trim_bottom']
    ) / dimensions
    risk_indices = trim_ratio.nlargest(32).index.tolist()
    sample_indices = manifest.sample(
        n=min(32, len(manifest)), random_state=42
    ).index.tolist()
    selected = manifest.loc[list(dict.fromkeys(risk_indices + sample_indices))]
    previews = []
    for row in selected.itertuples(index=False):
        original = read_image(row.original_path)
        processed = read_image(row.processed_path)
        bbox = (row.crop_x1, row.crop_y1, row.crop_x2, row.crop_y2)
        title = f'{row.patient_id} edge={int(row.edge_refined)} bar={int(row.progress_bar_detected)}'
        previews.append(draw_preview(original, processed, bbox, title))
    save_contact_sheets(previews, str(output_dir))


def collect_probabilities(model, manifest, batch_size, num_workers):
    _, eval_transform = build_transforms()
    dataset = ExternalProcessedDataset(manifest, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == 'cuda',
    )
    probabilities = np.zeros(len(dataset), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for images, indices in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            probabilities[indices.numpy()] = (
                torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            )
    return probabilities


def wilson_interval(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)
    ) / denominator
    return center - margin, center + margin


def sensitivity_metrics(probabilities, threshold):
    probabilities = np.asarray(probabilities)
    tp = int((probabilities >= threshold).sum())
    total = len(probabilities)
    low, high = wilson_interval(tp, total)
    return {
        'threshold': threshold,
        'sample_count': total,
        'tp': tp,
        'fn': total - tp,
        'sensitivity': tp / total,
        'sensitivity_ci_low': low,
        'sensitivity_ci_high': high,
    }


def load_thresholds(run_dir):
    with open(run_dir / 'locked_threshold_summary.json', encoding='utf-8') as file:
        summary = json.load(file)
    return {
        level: float(summary['levels'][level]['threshold'])
        for level in ('image', 'patient')
    }


def load_trained_model(model_name, run_dir):
    builders = {
        'resnet50': (build_resnet, 'resnet50_debiased_best.pth'),
        'efficientnet_b0': (build_efficientnet, 'efficientnet_b0_debiased_best.pth'),
    }
    builder, weight_name = builders[model_name]
    checkpoint = torch.load(run_dir / weight_name, map_location=DEVICE, weights_only=False)
    model = builder(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(DEVICE)


def evaluate_model(model_name, run_dir, manifest, args, output_dir):
    thresholds = load_thresholds(run_dir)
    model = load_trained_model(model_name, run_dir)
    image_results = manifest.copy()
    image_results['cancer_probability'] = collect_probabilities(
        model, manifest, args.batch_size, args.num_workers
    )
    patient_results = image_results.groupby('patient_id', as_index=False).agg(
        image_count=('cancer_probability', 'size'),
        max_probability=('cancer_probability', 'max'),
        mean_probability=('cancer_probability', 'mean'),
        median_probability=('cancer_probability', 'median'),
    )
    image_metrics = sensitivity_metrics(
        image_results['cancer_probability'], thresholds['image']
    )
    patient_metrics = sensitivity_metrics(
        patient_results['max_probability'], thresholds['patient']
    )
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    image_results.to_csv(
        model_dir / 'image_predictions.csv', index=False, encoding='utf-8-sig'
    )
    patient_results.to_csv(
        model_dir / 'patient_predictions.csv', index=False, encoding='utf-8-sig'
    )
    pd.DataFrame([
        {'level': 'image', **image_metrics},
        {'level': 'patient', **patient_metrics},
    ]).to_csv(model_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
    patient_results[
        patient_results['max_probability'] < thresholds['patient']
    ].sort_values('max_probability').to_csv(
        model_dir / 'missed_cancer_patients.csv', index=False, encoding='utf-8-sig'
    )
    print(
        f'{model_name}: image Sens={image_metrics["sensitivity"]:.3f} '
        f'({image_metrics["tp"]}/{image_metrics["sample_count"]}), '
        f'patient Sens={patient_metrics["sensitivity"]:.3f} '
        f'({patient_metrics["tp"]}/{patient_metrics["sample_count"]})'
    )
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest = scan_cancer_directory(args.cancer_dir)
    if manifest.empty:
        raise ValueError('外部癌数据目录中未扫描到图片')
    print(f'设备: {DEVICE}; 外部数据: {manifest.patient_id.nunique()}人/{len(manifest)}张')
    processed = preprocess_images(manifest, output_dir, args.overwrite_preprocessed)
    run_dirs = {
        'resnet50': Path(args.resnet_run_dir),
        'efficientnet_b0': Path(args.efficientnet_run_dir),
    }
    model_names = list(run_dirs) if args.model == 'all' else [args.model]
    evaluation_config = {
        'device': str(DEVICE),
        'cancer_dir': str(Path(args.cancer_dir).resolve()),
        'image_count': len(processed),
        'patient_count': int(processed['patient_id'].nunique()),
        'preprocess': 'preprocess_second_batch_v1_1',
        'model_run_dirs': {
            name: str(run_dirs[name].resolve()) for name in model_names
        },
        'thresholds': {
            name: load_thresholds(run_dirs[name]) for name in model_names
        },
    }
    with open(output_dir / 'evaluation_config.json', 'w', encoding='utf-8') as file:
        json.dump(evaluation_config, file, ensure_ascii=False, indent=2)
    for model_name in model_names:
        evaluate_model(
            model_name, run_dirs[model_name], processed, args, output_dir
        )
    print(f'结果已保存至: {output_dir}')


if __name__ == '__main__':
    main()
