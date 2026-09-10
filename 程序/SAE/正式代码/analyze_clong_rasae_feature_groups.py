#!/usr/bin/env python3
"""固定RA-SAE全字典描述性分组：train拟合，val检查，癌/非癌共同分组。

不训练、不删除Feature，不把技术分组解释为医学概念。完整链接要求组内任意
两项的decoder余弦和类内中心化患者响应相关均>=0.5；阈值仅作探索性展示约定。
输出逐Feature/组统计、完整层次树和可离线查看的跨癌/非癌代表图册。
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch

from run_clong_rasae_pilot import read_subset
from clong_rasae_core import ArchetypalMatryoshkaSAE

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '结果/SAE/RA_SAE_Pilot_20260908'
CACHE = ROOT / '结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache'


def patient_array(values, frame):
    """图像值按患者内平均；返回与患者标签表同序的二维数组。"""
    table = pd.DataFrame(values).assign(patient_id=frame.patient_id.to_numpy())
    means = table.groupby('patient_id', sort=True).mean()
    labels = frame.groupby('patient_id', sort=True).label.first().to_numpy()
    return means.to_numpy(copy=True), labels


def response_similarity(peak, frame):
    """log1p峰值先患者平均、类别内中心化，两类等权后计算相关矩阵。

    类别只用于消除癌/非癌均值差这一共同因素，不作为聚类目标。
    常量项没有响应相关依据，保留为单项。
    """
    x, labels = patient_array(np.log1p(peak), frame)
    for label in (0, 1):
        mask = labels == label
        x[mask] = (x[mask] - x[mask].mean(0)) / np.sqrt(2 * mask.sum())
    norm = np.linalg.norm(x, axis=0)
    good = norm > 1e-12
    x[:, good] /= norm[good]
    x[:, ~good] = 0
    sim = np.clip(x.T @ x, -1, 1)
    np.fill_diagonal(sim, 1)
    return sim, good


def group_features(decoder_similarity, response, good):
    """完整链接同时约束方向/响应相似度，返回全部Feature的组和完整树。"""
    sim = np.minimum(decoder_similarity, response)
    sim[~good, :] = -1
    sim[:, ~good] = -1
    distance = np.clip(1 - sim, 0, 2)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method='complete')
    assignment = fcluster(tree, t=0.5, criterion='distance')
    groups = [np.flatnonzero(assignment == k).tolist() for k in np.unique(assignment)]
    groups.sort(key=lambda members: (-len(members), members[0]))
    return groups, tree, sim


def select_images(score, frame, label, count=2):
    """每类按分数取不同患者的两例；零响应不展示为有效代表。"""
    selected, patients = [], set()
    for i in np.argsort(-score, kind='stable'):
        row = frame.iloc[i]
        if row.label != label or row.patient_id in patients or score[i] <= 0:
            continue
        selected.append(int(i))
        patients.add(row.patient_id)
        if len(selected) == count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_csv(CACHE / f'{s}_metadata.csv').reset_index().rename(
        columns={'index': 'source_row'}) for s in ('train', 'val')}
    assert not (set(frames['train'].patient_id) & set(frames['val'].patient_id))
    assert all(f.groupby('patient_id').label.nunique().max() == 1 for f in frames.values())
    definition = {
        'scope': 'fixed_RA_duration100_K256_all2560_features_full_available_train_val',
        'cohort': {s: {'images': len(f), 'patients': f.patient_id.nunique(),
                       'patients_by_label': f.groupby('label').patient_id.nunique().to_dict()}
                   for s, f in frames.items()},
        'sae_training_unchanged': 'original 196-image pilot weights, not retrained on full train',
        'scale': 'existing pilot-train positive Q99; no val fitted scale',
        'response': 'mean_over_patient_images(log1p(max_patch_h/Q99)); within-label centering; equal class total weight',
        'clustering': 'complete linkage on max(1-decoder_cosine,1-response_correlation); train only; cut distance 0.5',
        'threshold_status': 'descriptive display convention, not significance or medical identity; not tuned on val',
        'constant_features': 'singleton, no reliable response correlation',
        'validation': 'fixed train groups; val response correlation and coactive spatial similarity; no regrouping',
        'group_activation': 'image max over members of peak/Q99; positive if any member active; patient mean then class mean',
        'representative': 'member maximizing mean joint similarity to group; tie smallest feature_id',
        'atlas': 'every feature: two distinct patients per label per split ranked peak/Q99; raw heatmaps; no medical names',
        'no_group_causal_claim': True, 'test_read': False, 'external_read': False,
    }
    (out / 'analysis_definition.json').write_text(json.dumps(definition, ensure_ascii=False, indent=2))
    ck = torch.load(BASE / 'duration100/ra_final.pth', map_location='cuda', weights_only=False)
    state, cfg = ck['state_dict'], ck['config']
    model = ArchetypalMatryoshkaSAE(state['points'], state['decoder_bias'], cfg['hidden_dim'],
        tuple(cfg['k_list']), cfg['delta'], True, cfg['seed'], cfg['initialization'])
    model.load_state_dict(state)
    model.eval()
    q99 = np.load(BASE / 'decision_alignment/ra_train_q99.npy')
    if args.check:
        batch = read_subset(frames['train'].iloc[:2], 'train', torch.device('cuda'))
        with torch.no_grad():
            h = model.encode(batch['spatial'], 256)
        assert h.shape == (2, 49, 2560) and torch.isfinite(h).all()
        a = np.array([[1, .8, .1], [.8, 1, .2], [.1, .2, 1.]])
        groups, _, _ = group_features(a, a, np.ones(3, bool))
        assert groups == [[0, 1], [2]]
        toy_frame = pd.DataFrame({'patient_id': ['a', 'b', 'c', 'd'], 'label': [0, 0, 1, 1]})
        corr, variable = response_similarity(np.array([[1., 1.], [2., 2.], [4., 4.], [5., 5.]]), toy_frame)
        assert variable.all() and abs(corr[0, 1] - 1) < 1e-10
        print('CHECK PASSED: real forward and complete-link grouping', flush=True)
        return
    maps, peaks, response, good = {}, {}, {}, {}
    for split, frame in frames.items():
        # 空间图只在本轮内存中使用，避免落盘约1.4GB可再生成的中间缓存。
        maps[split] = np.empty((len(frame), 49, 2560), dtype='float32')
        peaks[split] = np.empty((len(frame), 2560), np.float32)
        with torch.no_grad():
            for start in range(0, len(frame), 16):
                batch = read_subset(frame.iloc[start:start + 16], split, torch.device('cuda'))
                h = model.encode(batch['spatial'], 256).cpu().numpy()
                maps[split][start:start + len(h)] = h
                peaks[split][start:start + len(h)] = h.max(1) / q99
        np.save(out / f'{split}_peak_q99.npy', peaks[split])
        response[split], good[split] = response_similarity(peaks[split], frame)
        np.save(out / f'{split}_response_similarity.npy', response[split])
        print(f'{split}: projected {len(frame)} images / {frame.patient_id.nunique()} patients', flush=True)
    # Existing subset values must be reproduced when projecting the larger cohort.
    small = pd.read_csv(BASE / 'duration100/val_subset.csv')
    np.testing.assert_allclose(peaks['val'][small.source_row],
        np.load(BASE / 'decision_alignment/ra_val_raw_score.npy'), atol=1e-5, rtol=1e-4)
    with torch.no_grad():
        decoder = model.decoder_weight.cpu().numpy().astype(np.float64)
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    dsim = np.clip(decoder @ decoder.T, -1, 1)
    groups, tree, joint = group_features(dsim, response['train'], good['train'])
    np.save(out / 'train_linkage.npy', tree)
    group_rows, member_rows, pair_rows = [], [], []
    for number, members in enumerate(groups, 1):
        gid = f'G{number:04d}'
        sub = joint[np.ix_(members, members)]
        representative = members[int(np.argmax(sub.mean(1)))]
        group_rows.append({'group': gid, 'size': len(members), 'representative': representative,
                           'members': members})
        for j in members:
            member_rows.append({'group': gid, 'feature_id': j, 'representative': representative,
                                'train_variable': bool(good['train'][j]), 'val_variable': bool(good['val'][j])})
        for n, j in enumerate(members):
            for k in members[n + 1:]:
                assert joint[j, k] >= .5 - 1e-10
                row = {'group': gid, 'feature1': j, 'feature2': k, 'decoder_cosine': dsim[j, k],
                       'train_response_correlation': response['train'][j, k],
                       'val_response_correlation': response['val'][j, k] if good['val'][j] and good['val'][k] else np.nan}
                for split, frame in frames.items():
                    a, b = maps[split][:, :, j], maps[split][:, :, k]
                    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                    valid = denom > 0
                    values = (a[valid] * b[valid]).sum(1) / denom[valid]
                    t = pd.DataFrame({'patient': frame.loc[valid, 'patient_id'].to_numpy(), 'value': values})
                    row[f'{split}_coactive_images'] = int(valid.sum())
                    row[f'{split}_coactive_patients'] = t.patient.nunique()
                    row[f'{split}_spatial_cosine'] = t.groupby('patient').value.mean().mean()
                pair_rows.append(row)
    members_df = pd.DataFrame(member_rows).sort_values('feature_id')
    assert members_df.feature_id.tolist() == list(range(2560))
    members_df.to_csv(out / 'feature_groups.csv', index=False)
    pair_columns = ['group', 'feature1', 'feature2', 'decoder_cosine', 'train_response_correlation', 'val_response_correlation',
                    'train_coactive_images', 'train_coactive_patients', 'train_spatial_cosine',
                    'val_coactive_images', 'val_coactive_patients', 'val_spatial_cosine']
    pairs = pd.DataFrame(pair_rows, columns=pair_columns)
    pairs.to_csv(out / 'within_group_pairs.csv', index=False)
    stats, feature_stats = [], []
    for split, frame in frames.items():
        patient_peak, labels = patient_array(peaks[split], frame)
        patient_presence, _ = patient_array((peaks[split] > 0).astype(float), frame)
        for label in (0, 1):
            for j in range(2560):
                feature_stats.append({'split': split, 'label': label, 'feature_id': j,
                    'mean_peak_q99': patient_peak[labels == label, j].mean(),
                    'mean_image_active_fraction': patient_presence[labels == label, j].mean()})
        for g in group_rows:
            group_peak = peaks[split][:, g['members']].max(1)
            pp, labels = patient_array(group_peak[:, None], frame)
            present, _ = patient_array((group_peak > 0)[:, None].astype(float), frame)
            for label in (0, 1):
                stats.append({'group': g['group'], 'size': g['size'], 'representative': g['representative'],
                              'split': split, 'label': label, 'patients': int((labels == label).sum()),
                              'mean_max_peak_q99': float(pp[labels == label, 0].mean()),
                              'mean_image_active_fraction': float(present[labels == label, 0].mean())})
    pd.DataFrame(stats).to_csv(out / 'group_class_statistics.csv', index=False)
    pd.DataFrame(feature_stats).to_csv(out / 'feature_class_statistics.csv', index=False)
    pd.DataFrame([{**g, 'members': ','.join(map(str, g['members']))} for g in group_rows]).to_csv(
        out / 'groups.csv', index=False)
    summary = {**definition['cohort'], 'features': 2560, 'groups': len(groups),
               'multi_feature_groups': sum(len(g) > 1 for g in groups),
               'features_in_multi_groups': sum(len(g) for g in groups if len(g) > 1),
               'singletons': sum(len(g) == 1 for g in groups), 'largest_group': max(map(len, groups)),
               'within_group_pairs': len(pairs),
               'median_train_response': float(pairs.train_response_correlation.median()) if len(pairs) else None,
               'median_val_response': float(pairs.val_response_correlation.median()) if len(pairs) else None,
               'val_pairs_response_ge_05': int((pairs.val_response_correlation >= .5).sum()),
               'val_evaluable_pairs': int(pairs.val_response_correlation.notna().sum()),
               'median_train_spatial': float(pairs.train_spatial_cosine.median()) if len(pairs) else None,
               'median_val_spatial': float(pairs.val_spatial_cosine.median()) if len(pairs) else None,
               'all_features_assigned_once': True, 'previous_val98_reproduced': True}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('GROUPING', json.dumps(summary, ensure_ascii=False), flush=True)
    atlas = out / '离线图册'
    (atlas / 'images').mkdir(parents=True, exist_ok=True)
    examples, source_rows, used = {}, [], set()
    for j in range(2560):
        records = []
        for split, frame in frames.items():
            for label in (1, 0):
                selected = select_images(peaks[split][:, j], frame, label)
                assert frame.iloc[selected].patient_id.nunique() == len(selected)
                for i in selected:
                    code = f'{split}_{i:04d}'
                    used.add((split, i))
                    records.append({'image': code, 'split': split, 'label': label,
                                    'peak': round(float(peaks[split][i, j]), 4),
                                    'heat': np.round(maps[split][i, :, j] / q99[j], 4).tolist()})
                    source_rows.append({'feature_id': j, 'split': split, 'label': label,
                                        'source_row': i, 'image_code': code})
        examples[str(j)] = records
    for split, i in sorted(used):
        im = Image.open(ROOT / frames[split].iloc[i].image_relpath).convert('RGB')
        im.thumbnail((480, 480))
        im.save(atlas / 'images' / f'{split}_{i:04d}.jpg', quality=88)
    pd.DataFrame(source_rows).to_csv(out / 'atlas_selection.csv', index=False)
    bundle = {'groups': group_rows, 'examples': examples, 'stats': stats, 'summary': summary}
    (atlas / 'data.js').write_text('const DATA=' + json.dumps(bundle, ensure_ascii=False, separators=(',', ':')) + ';')
    (atlas / '打开图册.html').write_text(HTML, encoding='utf-8')
    (atlas / '阅读说明.txt').write_text(
        '解压后用浏览器打开“打开图册.html”，无需联网。左侧选择技术分组，右侧可切换每一个Feature。\n'
        '每项尽可能展示训练/验证、癌/非癌各2位患者，共8例；缺少正激活候选时不补零值图片。\n'
        '亮色是原始激活强度，不代表促癌。每项使用既有训练Q99固定尺度；高于1部分显示饱和。\n'
        '分组不是医学概念，没有合并或删除模型特征；癌/非癌共有不等于没有分类作用。\n'
        '可直接回复组号、Feature号、图像代号和观察，不要求填表。\n', encoding='utf-8')
    with zipfile.ZipFile(out / '全字典跨癌非癌图册.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(atlas.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(atlas))
    summary.update(atlas_features=len(examples), atlas_panels=len(source_rows), atlas_images=len(used))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('COMPLETE: full dictionary statistics and offline atlas', flush=True)


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RA-SAE全字典跨癌/非癌观察</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#182333}header{max-width:1100px}main{display:grid;grid-template-columns:250px 1fr;gap:20px}select,input{padding:8px;max-width:100%}#groups{width:100%;height:70vh}.cards{display:grid;grid-template-columns:repeat(2,minmax(250px,1fr));gap:16px}.card{background:white;padding:12px;border-radius:8px}.pair{display:flex;align-items:center;gap:5px}.pair img,.pair canvas{width:49%;height:auto}#stats{white-space:pre-line;font-size:14px;background:#fff;padding:12px;margin:12px 0}small{color:#555}button{padding:8px;margin:4px}@media(max-width:750px){main{display:block}#groups{height:150px}.cards{grid-template-columns:1fr}}</style>
<header><h1>RA-SAE：全字典跨癌 / 非癌观察</h1><p id="summary"></p><p>这些是技术分组，尚未命名为医学概念。请看同组不同Feature是否表达相似模式，以及癌与非癌例子中是否有共同点或反例。</p><small>原始7×7激活上采样；左原图、右热图。亮色表示激活强，不表示促癌。图像按激活选择，不代表总体发生率。</small></header>
<main><aside><p><input id="search" placeholder="查组号或Feature号，如1089"></p><select id="groups" size="25"></select></aside><section><h2 id="title"></h2><label>查看Feature：<select id="feature"></select></label><div id="stats"></div><div class="cards" id="cards"></div></section></main>
<script src="data.js"></script><script>
const groups=document.getElementById('groups'),feature=document.getElementById('feature');
document.getElementById('summary').textContent=`全部2560项Feature；训练${DATA.summary.train.images}图、验证${DATA.summary.val.images}图。共${DATA.summary.groups}组，其中${DATA.summary.multi_feature_groups}组包含多项，其余保留单项。`;
function list(){let q=document.getElementById('search').value.trim().toLowerCase();groups.replaceChildren();DATA.groups.filter(g=>!q||(q.startsWith('g')&&g.group.toLowerCase().includes(q))||g.members.some(j=>String(j)===q||('ra-f'+String(j).padStart(4,'0'))===q)).forEach(g=>groups.add(new Option(`${g.group} · ${g.size}项 · F${g.representative}`,g.group)));if(groups.options.length){groups.selectedIndex=0;showGroup()}}
function showGroup(){let g=DATA.groups.find(g=>g.group===groups.value);document.getElementById('title').textContent=g.group+'｜'+g.size+'项Feature';feature.replaceChildren();g.members.forEach(j=>feature.add(new Option('RA-F'+String(j).padStart(4,'0')+(j===g.representative?'（技术代表）':''),j)));feature.value=g.representative;let rows=DATA.stats.filter(r=>r.group===g.group);document.getElementById('stats').textContent=rows.map(r=>`${r.split==='train'?'训练':'验证'} · ${r.label?'癌':'非癌'}：${r.patients}位患者；组最大激活均值 ${r.mean_max_peak_q99.toFixed(3)}；患者平均图像激活率 ${(r.mean_image_active_fraction*100).toFixed(1)}%`).join('\n')+'\n组激活率指至少一项Feature激活；大组更容易激活。共有不代表中性，不用于医学诊断。';showFeature()}
function showFeature(){const cards=document.getElementById('cards');cards.replaceChildren();DATA.examples[feature.value].forEach(r=>{let card=document.createElement('div');card.className='card';let title=document.createElement('p');title.textContent=`${r.split==='train'?'训练':'验证'} / ${r.label?'癌':'非癌'} / ${r.image} / 峰值 ${r.peak}`;let pair=document.createElement('div');pair.className='pair';let im=new Image(),canvas=document.createElement('canvas');im.onload=()=>{canvas.width=im.naturalWidth;canvas.height=im.naturalHeight;let c=canvas.getContext('2d');c.drawImage(im,0,0);let map=document.createElement('canvas');map.width=7;map.height=7;let mc=map.getContext('2d'),d=mc.createImageData(7,7);for(let i=0;i<49;i++){let t=Math.max(0,Math.min(1,r.heat[i])),stops=[[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],v=t*4,k=Math.min(3,Math.floor(v)),u=v-k;for(let j=0;j<3;j++)d.data[4*i+j]=Math.round(stops[k][j]*(1-u)+stops[k+1][j]*u);d.data[4*i+3]=255}mc.putImageData(d,0,0);c.globalAlpha=.52;c.imageSmoothingEnabled=true;c.drawImage(map,0,0,canvas.width,canvas.height)};im.src='images/'+r.image+'.jpg';pair.append(im,canvas);card.append(title,pair);cards.append(card)})}
groups.onchange=showGroup;feature.onchange=showFeature;document.getElementById('search').oninput=list;list();
</script></html>'''


if __name__ == '__main__':
    main()
