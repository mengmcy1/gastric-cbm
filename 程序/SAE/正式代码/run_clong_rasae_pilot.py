#!/usr/bin/env python3
"""运行隔离的小规模RA-SAE可行性对照，不产生正式科学PASS/FAIL。

正式pilot固定128 train/64 val患者、每人最多2图、2560字典、K=64/128/256、
256个train代表点、20轮、seed1701。仅比较同初始化的自由/受约束字典。
继承原损失公式及gamma，不使用单位decoder投影，不挑选最佳epoch或K。
--debug缩小规模，仅检查运行链路。训练、评估均只读现有train/val缓存。
--epochs 100用于一次固定时长敏感性比较，核对前20轮与修正版记录/权重一致。
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from clong_rasae_core import ArchetypalMatryoshkaSAE, calibrate_initial_encoder
from clong_s2b_core import attention_from_features, pooled_from_features
from clong_s2c_matryoshka import evaluate_loss, joint_layer_losses
from clong_sae_discovery import (
    CLONG_CHECKPOINT_SHA256, FROZEN_PATIENT_THRESHOLD, patient_class_weights,
)
from build_mage_teacher_roi_manifest import file_sha256

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / '结果/SAE/CLong_S2b结构重构_20260820/frozen_spatial_cache'
SEED = 1701


def select_subset(frame: pd.DataFrame, patients_per_label: int) -> pd.DataFrame:
    """参数frame为单split清单，patients_per_label为每类人数。返回最多2图/患者的子集。

    患者按固定随机序列选取；同患者图按relative_path排序，不读取特征/概率选例。
    source_row保存原缓存行号。
    """
    frame = frame.reset_index().rename(columns={'index': 'source_row'})
    if frame.groupby('patient_id').label.nunique().max() != 1:
        raise ValueError('存在跨标签患者')
    rng = np.random.default_rng(SEED)
    patients = frame[['patient_id', 'label']].drop_duplicates()
    selected = []
    for label in (0, 1):
        ids = np.sort(patients.loc[patients.label.eq(label), 'patient_id'].to_numpy())
        selected.extend(rng.choice(ids, patients_per_label, replace=False))
    return (frame[frame.patient_id.isin(selected)].sort_values(['patient_id', 'relative_path'])
            .groupby('patient_id', sort=False).head(2).reset_index(drop=True))


def read_subset(frame: pd.DataFrame, split: str, device: torch.device) -> dict:
    """参数frame为选例表，split限train/val，device为计算设备；返回三个缓存Tensor。

    spatial [N,49,1280]、pooled [N,1280]、attention [N,49]，保持清单行序。
    """
    if split not in ('train', 'val'):
        raise ValueError('pilot仅允许train/val')
    indices = frame.source_row.to_numpy(int)
    suffixes = {'spatial': 'spatial_features', 'pooled': 'pooled_features', 'attention': 'attention'}
    arrays = {}
    for key, suffix in suffixes.items():
        array = np.load(CACHE / f'{split}_{suffix}.npy', mmap_mode='r')
        arrays[key] = torch.from_numpy(np.asarray(array[indices]).copy()).to(device)
    return arrays


@torch.no_grad()
def evaluate(sae: ArchetypalMatryoshkaSAE, data: dict, frame: pd.DataFrame,
             head: nn.Module, weight: torch.Tensor, bias: torch.Tensor,
             margin_std: float, gamma: float, batch: int) -> dict:
    """返回每K重构保真与描述性特征-注意力对齐指标，不重新确定阈值。

    参数sae为当前字典；data为spatial/pooled/attention Tensor缓存；frame为同序清单；
    head为冻结空间头；weight [2,1280]/bias [2]为分类头；margin_std/gamma为train常量；
    batch为图像批量。输出dict包含患者AUC/一致率、图像cosine和活跃Feature统计。
    """
    sae.eval()
    margin = weight[1] - weight[0]
    arrays = {key: value.cpu().numpy() for key, value in data.items()}
    loss = evaluate_loss(sae, arrays['spatial'], arrays['pooled'], arrays['attention'],
                         frame, SimpleNamespace(attention_head=head), margin, margin_std,
                         gamma, batch, data['spatial'].device)
    baseline_prob = (data['pooled'] @ weight.T + bias).softmax(1)[:, 1].cpu().numpy()
    result = {'joint_loss': loss['total'], 'per_k': {}}
    for k in sae.k_list:
        collected = {key: [] for key in ('probability', 'pool_cosine', 'attention_cosine',
                                         'mean_l0', 'feature_attention_cosine')}
        counts = torch.zeros(sae.hidden_dim, device=weight.device, dtype=torch.long)
        dictionary = sae.decoder_weight
        for start in range(0, len(frame), batch):
            x = data['spatial'][start:start + batch]
            a = data['attention'][start:start + batch]
            hidden = sae.encode(x, k)
            rebuilt = hidden @ dictionary + sae.decoder_bias
            rebuilt_attention = attention_from_features(rebuilt, head)
            pooled = pooled_from_features(rebuilt, rebuilt_attention)
            live = hidden.gt(1e-8)
            counts += live.sum((0, 1))
            active = live.any(1)
            cosine = F.cosine_similarity(hidden, a.unsqueeze(-1), dim=1)
            per_image_cos = (cosine * active).sum(1) / active.sum(1).clamp_min(1)
            values = {
                'probability': (pooled @ weight.T + bias).softmax(1)[:, 1],
                'pool_cosine': F.cosine_similarity(pooled, data['pooled'][start:start + batch], dim=1),
                'attention_cosine': F.cosine_similarity(rebuilt_attention, a, dim=1),
                'mean_l0': live.float().sum(2).mean(1),
                'feature_attention_cosine': per_image_cos,
            }
            for key, value in values.items():
                collected[key].extend(value.cpu().numpy().tolist())
        patient = frame[['patient_id', 'label']].copy()
        patient['original'] = baseline_prob
        patient['rebuilt'] = collected['probability']
        patient = patient.groupby('patient_id').agg(label=('label', 'first'),
                                                    original=('original', 'mean'),
                                                    rebuilt=('rebuilt', 'mean'))
        result['per_k'][str(k)] = {
            'patient_original_auc': float(roc_auc_score(patient.label, patient.original)),
            'patient_rebuilt_auc': float(roc_auc_score(patient.label, patient.rebuilt)),
            'patient_agreement': float(((patient.original >= FROZEN_PATIENT_THRESHOLD)
                                       == (patient.rebuilt >= FROZEN_PATIENT_THRESHOLD)).mean()),
            'patient_mean_abs_probability_change': float((patient.rebuilt - patient.original).abs().mean()),
            'pool_cosine': float(np.mean(collected['pool_cosine'])),
            'attention_cosine': float(np.mean(collected['attention_cosine'])),
            'feature_attention_cosine': float(np.mean(collected['feature_attention_cosine'])),
            'mean_l0': float(np.mean(collected['mean_l0'])),
            'inactive_fraction_on_this_subset': float((counts == 0).float().mean()),
        }
    return result


def main() -> None:
    """无函数参数；读取CLI设备、debug及新输出路径，返回None，落盘日志关联实验产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, choices=[20, 100], default=20)
    parser.add_argument('--initialization', choices=['legacy', 'unique'], default='legacy')
    parser.add_argument('--match-encoder-scale', action='store_true',
                        help='encoder步长乘train解析初始化scale；仅unique可用')
    args = parser.parse_args()
    if args.match_encoder_scale and args.initialization != 'unique':
        raise ValueError('encoder步长匹配仅用于unique解析校准初始化')
    if args.epochs == 100 and (args.initialization != 'unique' or not args.match_encoder_scale):
        raise ValueError('100轮时长对照只延长已修正的unique+encoder尺度方案')
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    params = {'seed': SEED, 'train_patients_per_label': 4 if args.debug else 64,
              'val_patients_per_label': 4 if args.debug else 32,
              'max_images_per_patient': 2, 'hidden_dim': 64 if args.debug else 2560,
              'k_list': [4, 8, 16] if args.debug else [64, 128, 256],
              'centroids': 8 if args.debug else 256, 'delta': 0.2,
              'epochs': 2 if args.debug else args.epochs, 'batch': 4 if args.debug else 16,
              'learning_rate': 1e-4, 'weight_decay': 0.0, 'warmup_epochs': 2,
              'checkpoint_rule': 'fixed_final_epoch_no_model_or_k_selection',
              'scientific_pass_fail': False, 'scope': 'exploratory_pilot',
              'internal_test_read': False, 'external_read': False,
              'normalization': 'train_subset_fixed_center_no_per_dimension_standardization',
              'control': 'same_centroids_initialization_and_encoder_free_unnormalized_dictionary',
              'reference_fit': 'MiniBatchKMeans_train_only_patient_class_sample_weights',
              'feature_attention_metric': 'image_mean_of_cosine(h_j_over_49_positions,original_attention)_over_active_features',
              'stability_evaluated': False, 'medical_semantics_evaluated': False}
    params['initialization'] = args.initialization
    reference = None
    reference_root = ROOT / '结果/SAE/RA_SAE_Pilot_20260908/pilot_unique_step'
    if args.epochs == 100 and not args.debug:
        reference = pd.read_csv(reference_root / 'history.csv').set_index(['arm', 'epoch'])
        params['scope'] = 'exploratory_duration_sensitivity'
        params['duration_reference'] = str(reference_root)
    params['encoder_step_policy'] = ('base_lr_times_train_initial_scale'
                                     if args.match_encoder_scale else 'base_lr')
    if args.initialization == 'unique':
        params['centroids'] = params['hidden_dim']
        params['initial_encoder_scale'] = 'train_only_closed_form_shared_across_k_once'
    (args.output / 'config.json').write_text(json.dumps(params, ensure_ascii=False, indent=2))
    frames, data = {}, {}
    for split in ('train', 'val'):
        full = pd.read_csv(CACHE / f'{split}_metadata.csv')
        if not full.split.eq(split).all():
            raise ValueError('缓存metadata split异常')
        frames[split] = select_subset(full, params[f'{split}_patients_per_label'])
        frames[split].to_csv(args.output / f'{split}_subset.csv', index=False)
        data[split] = read_subset(frames[split], split, device)
    if set(frames['train'].patient_id) & set(frames['val'].patient_id):
        raise ValueError('train/val患者重叠')
    summary_path = ROOT / '结果/MAGE/MG2L训练轮数敏感性_20260818/mg2l_attention_sae_summary_seed42.json'
    ck_path = Path(json.loads(summary_path.read_text())['selected_checkpoint'])
    if file_sha256(ck_path) != CLONG_CHECKPOINT_SHA256:
        raise ValueError('C-long checkpoint不等于冻结模型')
    state = torch.load(ck_path, map_location='cpu', weights_only=False)['model_state_dict']
    head = nn.Conv2d(1280, 1, 1).to(device)
    head.load_state_dict({'weight': state['attention_head.weight'], 'bias': state['attention_head.bias']})
    head.requires_grad_(False).eval()
    weight = state['classifier.1.weight'].to(device)
    bias = state['classifier.1.bias'].to(device)
    for split in ('train', 'val'):
        rebuilt_a = attention_from_features(data[split]['spatial'], head)
        torch.testing.assert_close(rebuilt_a, data[split]['attention'], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(pooled_from_features(data[split]['spatial'], rebuilt_a),
                                   data[split]['pooled'], atol=1e-5, rtol=1e-4)
    patient_weights = patient_class_weights(frames['train']).astype(np.float64)
    patient_weights /= patient_weights.sum()
    x_cpu = data['train']['spatial'].cpu().numpy()
    center = np.average(x_cpu.mean(1), axis=0, weights=patient_weights).astype(np.float32)
    print('Fit train-only representative points', flush=True)
    km = MiniBatchKMeans(n_clusters=params['centroids'], random_state=SEED, n_init=1,
                        batch_size=1024, max_iter=20, reassignment_ratio=0)
    km.fit((x_cpu - center).reshape(-1, 1280), sample_weight=np.repeat(patient_weights / 49, 49))
    points = torch.from_numpy(km.cluster_centers_).to(device)
    center = torch.from_numpy(center).to(device)
    margin = weight[1] - weight[0]
    train_margin = (data['train']['pooled'] @ margin).cpu().numpy()
    mean_margin = np.average(train_margin, weights=patient_weights)
    margin_std = float(np.sqrt(np.average((train_margin - mean_margin)**2, weights=patient_weights)))
    gamma_path = ROOT / '结果/SAE/CLong_S2c_Matryoshka_20260821/gamma_pool_calibration_seed42.json'
    gamma_config = json.loads(gamma_path.read_text())
    gamma = float(gamma_config['gamma_pool'])
    params.update({'gamma_pool': gamma, 'margin_std': margin_std,
                   'centroid_fit_steps': int(km.n_steps_),
                   'device': str(device), 'debug': args.debug,
                   'train_images': len(frames['train']), 'val_images': len(frames['val']),
                   'checkpoint_sha256': CLONG_CHECKPOINT_SHA256})
    (args.output / 'config.json').write_text(json.dumps(params, ensure_ascii=False, indent=2))
    orders_rng = np.random.default_rng(SEED)
    orders = [orders_rng.choice(len(patient_weights), len(patient_weights), replace=True,
                               p=patient_weights) for _ in range(params['epochs'])]
    results, history = {}, []
    initial = None
    replay_checks = {}
    for arm in ('free', 'ra'):
        sae = ArchetypalMatryoshkaSAE(points, center, params['hidden_dim'], tuple(params['k_list']),
                                     params['delta'], arm == 'ra', SEED,
                                     args.initialization).to(device)
        initialization_check = None
        if args.initialization == 'unique':
            initialization_check = calibrate_initial_encoder(
                sae, data['train']['spatial'], data['train']['attention'],
                torch.as_tensor(patient_weights, device=device), params['batch'])
            with torch.no_grad():
                normalized = F.normalize(sae.decoder_weight, dim=1)
                similarities = normalized @ normalized.T
                similarities.fill_diagonal_(-1)
                initialization_check['fraction_atoms_with_neighbor_cosine_gt_0p999'] = float(
                    (similarities.max(1).values > .999).float().mean())
                del similarities
            (args.output / f'{arm}_initialization.json').write_text(
                json.dumps(initialization_check, indent=2, allow_nan=False))
            print(json.dumps({'arm': arm, 'initialization': initialization_check}), flush=True)
            # 本轮已观察到近复制与大幅放大；仅在train排除同一失败再训练。
            if initialization_check['fraction_atoms_with_neighbor_cosine_gt_0p999'] >= .05:
                raise RuntimeError('train初始化仍有至少5%方向近复制，停止本次修正试验')
            checks = initialization_check['per_k'].values()
            if sum(x['after'] for x in checks) >= sum(x['center_only'] for x in checks):
                raise RuntimeError('train联合重构不优于均值参照，停止本次修正试验')
        if initial is None:
            initial = {name: value.detach().clone() for name, value in
                       [('dictionary', sae.decoder_weight), ('encoder', sae.encoder.weight)]}
        else:
            torch.testing.assert_close(sae.decoder_weight, initial['dictionary'], rtol=0, atol=0)
            torch.testing.assert_close(sae.encoder.weight, initial['encoder'], rtol=0, atol=0)
        encoder_lr_scale = initialization_check['scale'] if args.match_encoder_scale else 1.0
        optimizer = torch.optim.AdamW([
            {'params': list(sae.encoder.parameters()), 'lr_scale': encoder_lr_scale},
            {'params': [p for name, p in sae.named_parameters() if not name.startswith('encoder.')],
             'lr_scale': 1.0},
        ], lr=params['learning_rate'], weight_decay=0)
        initial_eval = evaluate(sae, data['val'], frames['val'], head, weight, bias,
                                margin_std, gamma, params['batch'])
        trajectory = [{'epoch': 0, **initial_eval}]
        for epoch, order in enumerate(orders, 1):
            sae.train()
            warmup = min(1.0, epoch / params['warmup_epochs'])
            for group in optimizer.param_groups:
                group['lr'] = params['learning_rate'] * warmup * group['lr_scale']
            total_loss = 0.
            for start in range(0, len(order), params['batch']):
                indices = torch.as_tensor(order[start:start + params['batch']], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss_k, terms = joint_layer_losses(
                    sae, data['train']['spatial'][indices], data['train']['pooled'][indices],
                    data['train']['attention'][indices], SimpleNamespace(attention_head=head),
                    margin, margin_std, gamma)
                loss = torch.stack(list(loss_k.values())).mean()
                patch = torch.stack([t['patch'] for t in terms.values()]).mean()
                loss = patch + warmup * (loss - patch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{arm} epoch{epoch}: nonfinite loss')
                loss.backward()
                optimizer.step()
                sae.normalize_decoder()
                total_loss += float(loss.detach()) * len(indices)
            row = {'arm': arm, 'epoch': epoch, 'train_loss': total_loss / len(order)}
            if epoch % 5 == 0 or epoch == params['epochs']:
                metrics = evaluate(sae, data['val'], frames['val'], head, weight, bias,
                                   margin_std, gamma, params['batch'])
                row['val_joint_loss'] = metrics['joint_loss']
                trajectory.append({'epoch': epoch, **metrics})
                (args.output / f'{arm}_val_trajectory.json').write_text(
                    json.dumps(trajectory, indent=2, allow_nan=False))
            if reference is not None and epoch <= 20:
                old = reference.loc[(arm, epoch)]
                for key in ('train_loss', 'val_joint_loss'):
                    if key in row and not np.isclose(row[key], old[key], atol=1e-6, rtol=1e-5):
                        raise RuntimeError(f'{arm} epoch{epoch} {key}未复现20轮参照')
                if epoch == 20:
                    old_state = torch.load(reference_root / f'{arm}_final.pth', map_location='cpu',
                                           weights_only=False)['state_dict']
                    for key, value in sae.state_dict().items():
                        torch.testing.assert_close(value.detach().cpu(), old_state[key], atol=1e-6, rtol=1e-5)
                    replay_checks[arm] = {'first_20_history_matches': True, 'epoch20_state_matches': True}
                    (args.output / 'epoch20_replay.json').write_text(json.dumps(replay_checks, indent=2))
                    print(f'{arm}: first 20 epochs and checkpoint reproduced', flush=True)
            history.append(row)
            pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
            print(json.dumps(row), flush=True)
        results[arm] = {'initial_val': initial_eval, 'final_val': metrics,
                        'initialization': initialization_check,
                        'final_train': evaluate(sae, data['train'], frames['train'], head, weight,
                                                bias, margin_std, gamma, params['batch']),
                        'max_relax_norm': float(sae.relaxation.detach().norm(dim=1).max()) if arm == 'ra' else None,
                        'dictionary_rms_change_from_initial': float((sae.decoder_weight.detach() - initial['dictionary']).square().mean().sqrt())}
        torch.save({'state_dict': sae.state_dict(), 'config': params, 'arm': arm},
                   args.output / f'{arm}_final.pth')
        del sae, optimizer
    (args.output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    print('Completed exploratory pilot; no scientific qualification decision.', flush=True)


if __name__ == '__main__':
    main()
