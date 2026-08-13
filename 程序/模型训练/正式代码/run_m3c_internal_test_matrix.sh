#!/usr/bin/env bash
# 先生成冻结M1框+q_region，再一次性评价M3c-B内部test三seed。
set -euo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
FREEZE="程序/模型训练/正式代码/freeze_m3c_internal_test_protocol.py"
BUILD="数据整理脚本/build_m3c_internal_test_predictions.py"
EVALUATE="程序/模型训练/正式代码/evaluate_m3c_locked_internal_test.py"
PROTOCOL="$ROOT/结果/M3c区域门控_0804/冻结内部test协议/m3c_internal_test_protocol.json"
PREDICTIONS="$ROOT/结果/M3c区域门控_0804/冻结内部test预测"
RESULTS="$ROOT/结果/M3c区域门控_0804/锁定内部test评估"
LOGS="$RESULTS/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"

if [[ $# -ne 1 ]]; then
    echo "用法: CUDA_DEVICE=<空闲GPU> $0 42,202,503" >&2; exit 2
fi
if [[ -z "$CUDA_DEVICE" ]]; then
    echo "必须先检查GPU并设置CUDA_DEVICE。" >&2; exit 2
fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "未找到nvidia-smi。" >&2; exit 1; }
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | awk -v target="$CUDA_DEVICE" '$1 == target {found=1} END {exit !found}'; then
    echo "CUDA_DEVICE无效: $CUDA_DEVICE" >&2; exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
mkdir -p "$LOGS"
cd "$ROOT"

if [[ ! -f "$PROTOCOL" ]]; then
    "$PYTHON" -u "$FREEZE"
fi

IFS=',' read -r -a seeds <<< "$1"
failed=0
for seed in "${seeds[@]}"; do
    case "$seed" in 42|202|503) ;; *) echo "非法seed: $seed" >&2; exit 2;; esac
    pred_csv="$PREDICTIONS/m3c_internal_test_predictions_seed${seed}.csv"
    pred_cfg="$PREDICTIONS/m3c_internal_test_predictions_seed${seed}.json"
    result_cfg="$RESULTS/m3c_locked_internal_test_seed${seed}/config.json"
    if [[ -f "$result_cfg" ]]; then
        echo "[$(date '+%F %T')] SKIP seed${seed}: 锁定test结果已存在。"; continue
    fi
    if [[ -e "$RESULTS/m3c_locked_internal_test_seed${seed}" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE seed${seed}评价目录；不自动覆盖。" >&2
        failed=1; continue
    fi
    if [[ ! -f "$pred_csv" || ! -f "$pred_cfg" ]]; then
        if [[ -e "$pred_csv" || -e "$pred_cfg" ]]; then
            echo "[$(date '+%F %T')] INCOMPLETE seed${seed}预测产物；不自动覆盖。" >&2
            failed=1; continue
        fi
        echo "[$(date '+%F %T')] START seed${seed} M1框+q_region"
        set +e
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
            "$PYTHON" -u "$BUILD" --seed "$seed" 2>&1 \
            | tee "$LOGS/seed${seed}_build.log"
        status=("${PIPESTATUS[@]}")
        set -e
        if [[ "${status[0]}" -ne 0 || "${status[1]}" -ne 0 ]]; then
            echo "[$(date '+%F %T')] FAILED seed${seed}预测；保留现场。" >&2
            failed=1; continue
        fi
    fi
    echo "[$(date '+%F %T')] START seed${seed} 锁定test评价"
    set +e
    "$PYTHON" -u "$EVALUATE" --seed "$seed" 2>&1 \
        | tee "$LOGS/seed${seed}_evaluate.log"
    status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${status[0]}" -ne 0 || "${status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED seed${seed}评价；保留现场。" >&2
        failed=1
    else
        echo "[$(date '+%F %T')] DONE seed${seed} M3c-B内部test"
    fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path('/home/mcy/gastric-cbm/结果/M3c区域门控_0804/锁定内部test评估')
rows = []
for seed in (42, 202, 503):
    path = root / f'm3c_locked_internal_test_seed{seed}/config.json'
    if not path.is_file():
        continue
    d = json.loads(path.read_text(encoding='utf-8'))
    rows.append(d)
    print(
        f"seed{seed}: 癌召回 {d['image_m1']['cancer_recall']:.4f} -> "
        f"{d['image_m3c']['cancer_recall']:.4f}; 非癌FP "
        f"{d['image_m1']['noncancer_fp']:.4f} -> {d['image_m3c']['noncancer_fp']:.4f}; "
        f"通过={d['checks']['passed_seed_confirmation']}"
    )
if len(rows) == 3:
    passed = all(d['checks']['passed_seed_confirmation'] for d in rows)
    print(f"M3c-B内部test总判定: {'确认通过' if passed else '未确认'}")
PY
echo "M3c-B内部test矩阵完成: $RESULTS"
