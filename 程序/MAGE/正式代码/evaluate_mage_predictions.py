#!/usr/bin/env python3
"""用推理目录和含真值标签的清单计算图片/患者指标，不重新选阈值。"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def metrics(labels, probabilities, threshold):
    """按给定阈值计算混淆矩阵；单一类别队列的AUC保留为空。"""
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {'n': len(labels), 'threshold': threshold,
            'auc': float(roc_auc_score(labels, probabilities)) if labels.nunique() == 2 else None,
            'accuracy': float((prediction == labels).mean()),
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='与推理相同image_path，另有patient_id,label')
    parser.add_argument('--predictions', type=Path, required=True, help='infer_mage输出目录')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.manifest, dtype={'patient_id': str})
    predictions = pd.read_csv(args.predictions / 'image_predictions.csv', dtype={'patient_id': str})
    config = json.loads((args.predictions / 'inference_config.json').read_text())
    joined = predictions.merge(truth[['image_path', 'patient_id', 'label']],
                               on=['image_path', 'patient_id'], validate='one_to_one', how='outer', indicator=True)
    if joined.empty or not joined['_merge'].eq('both').all() or not set(joined.label) <= {0, 1}:
        raise ValueError('预测与标签清单必须完整一一匹配，且标签为0/1')
    if joined.groupby('patient_id').label.nunique().max() > 1:
        raise ValueError('同一患者标签不一致')
    patients = joined.groupby('patient_id', as_index=False).agg(
        label=('label', 'first'), probability=('cancer_probability', 'mean'))
    result = {'image': metrics(joined.label, joined.cancer_probability, config['image_threshold']),
              'patient': metrics(patients.label, patients.probability, config['patient_threshold']),
              'threshold_reselected': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
