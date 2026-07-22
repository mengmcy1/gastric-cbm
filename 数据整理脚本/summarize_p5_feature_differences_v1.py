"""汇总 P5 患者级数值特征的标签间 SMD 与单变量 AUC。"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_ROOT = os.path.join(PROJECT_DIR, '数据整理记录', '第二批', '匹配_v1')
SOURCES = {
    'metadata': os.path.join(MATCH_ROOT, '元数据基线_v1', '患者级元数据表.csv'),
    'pixel_statistics': os.path.join(MATCH_ROOT, '像素统计基线_v1', '患者级低层统计.csv'),
}
EXCLUDED = {'label'}


def feature_row(dataset, source, feature, frame):
    values = pd.to_numeric(frame[feature], errors='coerce')
    valid = values.notna()
    labels = frame.loc[valid, 'label'].astype(int)
    values = values.loc[valid]
    case = values.loc[labels.eq(1)]
    control = values.loc[labels.eq(0)]
    pooled_std = np.sqrt((case.var(ddof=1) + control.var(ddof=1)) / 2)
    smd = (case.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0.0
    auc = roc_auc_score(labels, values) if values.nunique() > 1 else 0.5
    return {
        'dataset': dataset,
        'source': source,
        'feature': feature,
        'case_count': len(case),
        'control_count': len(control),
        'case_mean': case.mean(),
        'control_mean': control.mean(),
        'smd': smd,
        'absolute_smd': abs(smd),
        'univariate_auc': auc,
        'oriented_univariate_auc': max(auc, 1 - auc),
    }


def main():
    rows = []
    for source, path in SOURCES.items():
        table = pd.read_csv(path, encoding='utf-8-sig')
        identifier_columns = {'dataset', 'patient_id', 'cv_group', 'label'}
        features = [
            column for column in table.select_dtypes(include='number').columns
            if column not in identifier_columns | EXCLUDED
        ]
        for dataset, frame in table.groupby('dataset', sort=True):
            rows.extend(feature_row(dataset, source, feature, frame) for feature in features)
    result = pd.DataFrame(rows).sort_values(
        ['dataset', 'absolute_smd'], ascending=[True, False], kind='stable'
    )
    result.to_csv(
        os.path.join(MATCH_ROOT, 'P5患者级数值特征差异.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    strict = result.loc[result['dataset'].eq('strict_common_support_1to1')]
    lines = ['严格共同支持集绝对SMD最高的数值特征：']
    for row in strict.nlargest(15, 'absolute_smd').itertuples(index=False):
        lines.append(
            f'{row.source}/{row.feature}: SMD={row.smd:.3f}, '
            f'单变量定向AUC={row.oriented_univariate_auc:.3f}'
        )
    text = '\n'.join(lines) + '\n'
    with open(
        os.path.join(MATCH_ROOT, 'P5严格集残余特征摘要.txt'), 'w', encoding='utf-8'
    ) as file:
        file.write(text)
    print(text)


if __name__ == '__main__':
    main()
