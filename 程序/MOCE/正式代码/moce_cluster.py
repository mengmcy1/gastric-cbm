"""按类别提取 MOCE 概念簇，并评估概念重要性。"""

import csv
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import KMeans

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DEMO_DIR = os.path.join(PROJECT_DIR, '程序', 'MOCE', 'demo测试')
sys.path.insert(0, DEMO_DIR)

from moce_single_demo import (
    ActivationCapture,
    CONCEPT_TRANSFORM,
    DATA_DIR,
    DEVICE,
    MODEL_TRANSFORM,
    MODEL_REGISTRY,
    TARGET_LAYERS,
    crop_and_resize,
    extract_candidate_masks,
    load_model,
)


CSV_PATH = os.path.join(DATA_DIR, 'dataset_manifest.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, '结果', 'MOCE聚类', '第二批')

MODEL_NAME = 'efficientnet_b0'  # 当前聚类使用的模型resnet50/efficientnet_b0
PATIENTS_PER_CLASS = None  # 每类抽取的患者数；None 表示使用全部患者
N_CLUSTERS = 25  # 每个类别的概念簇数量，沿用 MOCE 默认值
REPRESENTATIVES = 5  # 每个概念簇展示的代表区域数
ENCODE_BATCH_SIZE = 32  # 候选区域分批编码，控制显存占用
ALPHA = 1.0  # S_h 中移除分数排名的权重
BETA = 0.5  # S_h 中保留分数排名的权重
SSC_SDC_TOP_K = 5  # SSC/SDC 依次加入或移除的重要概念数量
RANDOM_SEED = 42  # 患者抽样和聚类随机种子

CLASS_NAMES = {0: '非癌', 1: '癌/高级别'}
FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=18,
)
SMALL_FONT = ImageFont.truetype(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=14,
)


def load_patient_samples():
    '''从整理清单读取图片，每个类别的每位患者随机保留一张。'''
    dataframe = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    dataframe = dataframe.rename(columns={
        'image_path': '图片名字',
        'label': '瘤变标签',
    })
    dataframe = dataframe.sample(frac=1, random_state=RANDOM_SEED)
    dataframe = dataframe.drop_duplicates(['patient_id'])

    samples = {}
    for label in [0, 1]:
        class_data = dataframe[dataframe['瘤变标签'] == label].copy()
        if PATIENTS_PER_CLASS is not None:
            class_data = class_data.head(PATIENTS_PER_CLASS)
        samples[label] = class_data.reset_index(drop=True)
    return samples


@torch.no_grad()
def encode_candidates(model, capture, original, candidates):
    """把候选区域重新输入模型，返回候选图片和目标层 GAP 特征。"""
    concept_images = []
    for item in candidates:
        concept, bbox = crop_and_resize(original, item['mask'])
        item['bbox'] = bbox
        concept_images.append(concept)

    features = []
    for start in range(0, len(concept_images), ENCODE_BATCH_SIZE):
        images = concept_images[start:start + ENCODE_BATCH_SIZE]
        batch = torch.stack([CONCEPT_TRANSFORM(image) for image in images]).to(DEVICE)
        model(batch)
        features.append(capture.output.mean(dim=(2, 3)).cpu().numpy())
    return concept_images, np.concatenate(features)


def extract_class_features(model, capture, label, dataframe, class_dir):
    """提取一个类别中不同患者的候选区域及其特征。"""
    patch_dir = os.path.join(class_dir, '候选区域')
    mask_dir = os.path.join(class_dir, '候选掩码')
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    features_all = []
    records = []
    patch_index = 0

    for order, row in enumerate(dataframe.itertuples(index=False), 1):
        image_name = getattr(row, '图片名字')
        patient_id = getattr(row, 'patient_id')
        image_path = os.path.join(DATA_DIR, image_name)
        original = Image.open(image_path).convert('RGB')
        original = np.asarray(original)
        class_probability, candidates = extract_candidate_masks(
            model, capture, original, label,
        )
        if not candidates:
            print(f'{image_name} 无候选区域，跳过')
            continue
        concept_images, features = encode_candidates(model, capture, original, candidates)

        for candidate, concept_image, feature in zip(candidates, concept_images, features):
            patch_name = f'{patch_index:06d}.jpg'
            mask_name = f'{patch_index:06d}.png'
            patch_path = os.path.join(patch_dir, patch_name)
            mask_path = os.path.join(mask_dir, mask_name)
            concept_image.save(patch_path, quality=92)
            Image.fromarray(candidate['mask'] * 255).save(mask_path)
            features_all.append(feature)
            records.append({
                'patch_index': patch_index,
                'image_name': image_name,
                'patient_id': patient_id,
                'class_label': label,
                'class_probability': class_probability,
                'channel': candidate['channel'],
                'channel_score': candidate['channel_score'],
                'area_ratio': candidate['area_ratio'],
                'bbox': candidate['bbox'],
                'patch_file': patch_path,
                'mask_file': mask_path,
            })
            patch_index += 1

        print(
            f'类别{label} [{order:02d}/{len(dataframe):02d}] '
            f'{patient_id}  候选区域={len(candidates)}'
        )

    return np.asarray(features_all, dtype=np.float32), records


def cluster_features(features, records, class_dir):
    """执行 K-Means，保存模型、特征、分配结果和簇统计。"""
    if len(records) < N_CLUSTERS:
        print(f'候选区域数少于聚类数 {N_CLUSTERS}，跳过聚类')
        return pd.DataFrame(), pd.DataFrame()

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(features)
    distances = kmeans.transform(features)

    for index, record in enumerate(records):
        cluster_id = int(cluster_ids[index])
        record['cluster_id'] = cluster_id
        record['distance_to_center'] = float(distances[index, cluster_id])

    joblib.dump(kmeans, os.path.join(class_dir, 'kmeans_model.joblib'))
    np.savez_compressed(
        os.path.join(class_dir, 'candidate_features.npz'),
        features=features,
        cluster_ids=cluster_ids,
        centers=kmeans.cluster_centers_,
    )

    assignment_path = os.path.join(class_dir, 'cluster_assignments.csv')
    fields = list(records[0].keys())
    with open(assignment_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

    assignment = pd.DataFrame(records)
    summary = assignment.groupby('cluster_id').agg(
        candidate_count=('patch_index', 'size'),
        patient_count=('patient_id', 'nunique'),
        mean_distance=('distance_to_center', 'mean'),
        mean_area_ratio=('area_ratio', 'mean'),
    ).reset_index()
    summary.to_csv(
        os.path.join(class_dir, 'cluster_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    return assignment, summary


@torch.no_grad()
def predict_target_probabilities(model, tensors, target_class):
    """分批计算一组输入属于目标类别的概率。"""
    probabilities = []
    for start in range(0, len(tensors), ENCODE_BATCH_SIZE):
        batch = torch.stack(tensors[start:start + ENCODE_BATCH_SIZE]).to(DEVICE)
        logits = model(batch)
        probabilities.extend(
            torch.softmax(logits, dim=1)[:, target_class].cpu().numpy()
        )
    return np.asarray(probabilities)


def select_cluster_masks(image_assignment, image_shape):
    """同一图片的每个概念簇只取距离簇中心最近的候选掩码。"""
    cluster_masks = {}
    for cluster_id, cluster_rows in image_assignment.groupby('cluster_id'):
        nearest = cluster_rows.nsmallest(1, 'distance_to_center').iloc[0]
        mask = np.asarray(Image.open(nearest['mask_file'])) > 0
        cluster_masks[int(cluster_id)] = mask.reshape(image_shape)
    return cluster_masks


def relative_ranks(values, skip_negative=False):
    """按作者实现把同图概念排名转换到 [0, 1)，较大值排名更高。"""
    ranks = np.zeros(len(values), dtype=np.float32)
    for position, index in enumerate(np.argsort(values)):
        if skip_negative and values[index] < 0:
            continue
        ranks[index] = position / len(values)
    return ranks


def evaluate_concept_importance(model, label, assignment, class_dir):
    """按图片计算概念级 S^R、S^E，并汇总全局重要性 S_h。"""
    per_image_records = []
    total_images = assignment['image_name'].nunique()

    for image_name, image_assignment in assignment.groupby('image_name', sort=False):
        original = np.asarray(
            Image.open(os.path.join(DATA_DIR, image_name)).convert('RGB')
        )
        cluster_masks = select_cluster_masks(
            image_assignment, original.shape[:2],
        )
        cluster_ids = sorted(cluster_masks)
        keep_tensors = []
        removed_tensors = []

        for cluster_id in cluster_ids:
            mask = cluster_masks[cluster_id]
            concept, _ = crop_and_resize(original, mask.astype(np.uint8))
            removed = original.copy()
            removed[mask] = 0
            keep_tensors.append(CONCEPT_TRANSFORM(concept))
            removed_tensors.append(MODEL_TRANSFORM(Image.fromarray(removed)))

        keep_probabilities = predict_target_probabilities(
            model, keep_tensors, label,
        )
        removed_probabilities = predict_target_probabilities(
            model, removed_tensors, label,
        )
        full_probability = float(image_assignment.iloc[0]['class_probability'])
        removal_drops = full_probability - removed_probabilities
        removal_total = removal_drops.sum()
        if np.isclose(removal_total, 0):
            removal_scores = np.zeros_like(removal_drops)
        else:
            removal_scores = removal_drops / removal_total
        extraction_scores = keep_probabilities / keep_probabilities.sum()

        image_scores = pd.DataFrame({
            'cluster_id': cluster_ids,
            'keep_probability': keep_probabilities,
            'removed_probability': removed_probabilities,
            'probability_drop': removal_drops,
            'S_R': removal_scores,
            'S_E': extraction_scores,
        })
        image_scores['rank_R'] = relative_ranks(
            image_scores['S_R'].to_numpy(), skip_negative=True,
        )
        image_scores['rank_E'] = relative_ranks(
            image_scores['S_E'].to_numpy(),
        )
        image_scores['weighted_rank'] = (
            ALPHA * image_scores['rank_R'] + BETA * image_scores['rank_E']
        )

        patient_id = image_assignment.iloc[0]['patient_id']
        for row in image_scores.itertuples(index=False):
            per_image_records.append({
                'image_name': image_name,
                'patient_id': patient_id,
                'class_label': label,
                'cluster_id': row.cluster_id,
                'candidate_count': int(
                    (image_assignment['cluster_id'] == row.cluster_id).sum()
                ),
                'full_probability': full_probability,
                'keep_probability': row.keep_probability,
                'removed_probability': row.removed_probability,
                'probability_drop': row.probability_drop,
                'S_R': row.S_R,
                'S_E': row.S_E,
                'rank_R': row.rank_R,
                'rank_E': row.rank_E,
                'weighted_rank': row.weighted_rank,
            })

    per_image = pd.DataFrame(per_image_records)
    importance = per_image.groupby('cluster_id').agg(
        image_count=('image_name', 'nunique'),
        patient_count=('patient_id', 'nunique'),
        mean_S_R=('S_R', 'mean'),
        mean_S_E=('S_E', 'mean'),
        mean_probability_drop=('probability_drop', 'mean'),
        weighted_rank_sum=('weighted_rank', 'sum'),
    ).reset_index()
    importance['image_coverage'] = importance['image_count'] / total_images
    importance['S_h'] = importance['weighted_rank_sum'] / total_images
    importance['importance_rank'] = importance['S_h'].rank(
        method='min', ascending=False,
    ).astype(int)
    importance = importance.sort_values(
        ['importance_rank', 'cluster_id'],
    ).reset_index(drop=True)

    per_image.to_csv(
        os.path.join(class_dir, 'concept_scores_per_image.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    importance.to_csv(
        os.path.join(class_dir, 'concept_importance.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    return importance


def evaluate_ssc_sdc(model, label, assignment, importance, class_dir):
    """按全局重要性依次保留或移除概念，生成 SSC/SDC 曲线数据。"""
    top_clusters = importance.nsmallest(
        SSC_SDC_TOP_K, 'importance_rank',
    )['cluster_id'].tolist()
    target_threshold = 0.5
    records = []

    for image_name, image_assignment in assignment.groupby('image_name', sort=False):
        original = np.asarray(
            Image.open(os.path.join(DATA_DIR, image_name)).convert('RGB')
        )
        cluster_masks = select_cluster_masks(
            image_assignment, original.shape[:2],
        )
        cumulative = np.zeros(original.shape[:2], dtype=bool)
        ssc_tensors = []
        sdc_tensors = []

        for step in range(SSC_SDC_TOP_K + 1):
            if step:
                cluster_id = top_clusters[step - 1]
                if cluster_id in cluster_masks:
                    cumulative |= cluster_masks[cluster_id]
            kept = original * cumulative[..., None]
            removed = original.copy()
            removed[cumulative] = 0
            ssc_tensors.append(MODEL_TRANSFORM(Image.fromarray(kept)))
            sdc_tensors.append(MODEL_TRANSFORM(Image.fromarray(removed)))

        ssc_probabilities = predict_target_probabilities(
            model, ssc_tensors, label,
        )
        sdc_probabilities = predict_target_probabilities(
            model, sdc_tensors, label,
        )
        patient_id = image_assignment.iloc[0]['patient_id']

        for step, (ssc_probability, sdc_probability) in enumerate(
            zip(ssc_probabilities, sdc_probabilities)
        ):
            records.append({
                'image_name': image_name,
                'patient_id': patient_id,
                'class_label': label,
                'step': step,
                'cluster_added_or_removed': (
                    '' if step == 0 else top_clusters[step - 1]
                ),
                'ssc_target_probability': ssc_probability,
                'sdc_target_probability': sdc_probability,
                'ssc_correct': int(ssc_probability >= target_threshold),
                'sdc_correct': int(sdc_probability >= target_threshold),
            })

    per_image = pd.DataFrame(records)
    summary = per_image.groupby('step').agg(
        ssc_accuracy=('ssc_correct', 'mean'),
        sdc_accuracy=('sdc_correct', 'mean'),
        mean_ssc_probability=('ssc_target_probability', 'mean'),
        mean_sdc_probability=('sdc_target_probability', 'mean'),
    ).reset_index()
    summary['cluster_added_or_removed'] = [
        '' if step == 0 else top_clusters[step - 1]
        for step in summary['step']
    ]
    per_image.to_csv(
        os.path.join(class_dir, 'ssc_sdc_per_image.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    summary.to_csv(
        os.path.join(class_dir, 'ssc_sdc_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    return summary


def fit_patch(image, size=(170, 150)):
    """保持比例，将代表区域居中放入固定展示面板。"""
    image = ImageOps.contain(image.convert('RGB'), size, Image.Resampling.LANCZOS)
    panel = Image.new('RGB', size)
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def save_cluster_overview(label, assignment, output_path):
    """为每个概念簇展示距离中心最近的多个代表区域。"""
    panel_width, panel_height = 170, 150
    label_width, row_height = 180, 195
    canvas = Image.new(
        'RGB',
        (label_width + panel_width * REPRESENTATIVES, row_height * N_CLUSTERS),
        'white',
    )
    draw = ImageDraw.Draw(canvas)

    for cluster_id in range(N_CLUSTERS):
        cluster = assignment[assignment['cluster_id'] == cluster_id]
        cluster = cluster.nsmallest(REPRESENTATIVES, 'distance_to_center')
        y = cluster_id * row_height
        draw.text((8, y + 10), f'概念簇 {cluster_id:02d}', fill=(20, 20, 20), font=FONT)
        draw.text(
            (8, y + 40),
            f'区域 {len(assignment[assignment["cluster_id"] == cluster_id])} 个',
            fill=(60, 60, 60),
            font=SMALL_FONT,
        )
        draw.text(
            (8, y + 64),
            f'患者 {assignment[assignment["cluster_id"] == cluster_id]["patient_id"].nunique()} 位',
            fill=(60, 60, 60),
            font=SMALL_FONT,
        )

        for column, row in enumerate(cluster.itertuples(index=False)):
            patch = fit_patch(Image.open(row.patch_file))
            x = label_width + column * panel_width
            canvas.paste(patch, (x, y))
            draw.text(
                (x + 4, y + panel_height + 5),
                str(row.patient_id),
                fill=(30, 30, 30),
                font=SMALL_FONT,
            )

    title = f'{MODEL_NAME} 类别{label}（{CLASS_NAMES[label]}）MOCE概念聚类'
    canvas.save(output_path, pnginfo=None)
    print(title)


def main():
    """按类别完成候选提取、概念聚类、重要性和 SSC/SDC 评估。"""
    weight_file = MODEL_REGISTRY[MODEL_NAME][0]
    weight_path = os.path.join(PROJECT_DIR, '结果', '模型权重', weight_file)
    model = load_model(MODEL_NAME, weight_path)
    capture = ActivationCapture(TARGET_LAYERS[MODEL_NAME](model))
    samples = load_patient_samples()

    for label, dataframe in samples.items():
        class_dir = os.path.join(OUTPUT_DIR, MODEL_NAME, f'class_{label}')
        os.makedirs(class_dir, exist_ok=True)
        print(f'\n===== 类别{label}：{CLASS_NAMES[label]}，患者数={len(dataframe)} =====')
        features, records = extract_class_features(
            model, capture, label, dataframe, class_dir,
        )
        assignment, summary = cluster_features(features, records, class_dir)
        if assignment.empty:
            continue
        save_cluster_overview(
            label,
            assignment,
            os.path.join(class_dir, 'concept_clusters.png'),
        )
        importance = evaluate_concept_importance(
            model, label, assignment, class_dir,
        )
        ssc_sdc = evaluate_ssc_sdc(
            model, label, assignment, importance, class_dir,
        )
        print(f'候选区域总数: {len(features)}')
        print(f'平均每簇患者数: {summary["patient_count"].mean():.2f}')
        print(f'最高重要性概念簇: {int(importance.iloc[0]["cluster_id"])}')
        print(f'SSC/SDC 评估步数: {len(ssc_sdc) - 1}')
        print(f'输出目录: {class_dir}')

    capture.close()


if __name__ == '__main__':
    main()
