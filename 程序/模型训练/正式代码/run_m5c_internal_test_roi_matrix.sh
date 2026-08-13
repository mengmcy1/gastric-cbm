#!/usr/bin/env bash
# 用冻结M1为内部test构建三seed预测ROI；标签/真值框不参与ROI几何。
set -euo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="数据整理脚本/build_m5c_internal_test_roi.py"
OUTPUT="$ROOT/结果/M5c概率融合_0804/冻结内部test_ROI清单"
LOGS="$OUTPUT/logs"
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
IFS=',' read -r -a seeds <<< "$1"
failed=0
for seed in "${seeds[@]}"; do
    case "$seed" in 42|202|503) ;; *) echo "非法seed: $seed" >&2; exit 2;; esac
    csv="$OUTPUT/m5c_internal_test_roi_seed${seed}.csv"
    cfg="$OUTPUT/m5c_internal_test_roi_seed${seed}.json"
    if [[ -f "$csv" && -f "$cfg" ]]; then
        echo "[$(date '+%F %T')] SKIP seed${seed}: 完整test ROI已存在。"; continue
    fi
    if [[ -e "$csv" || -e "$cfg" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE seed${seed}: 不自动覆盖。" >&2
        failed=1; continue
    fi
    echo "[$(date '+%F %T')] START seed${seed} test ROI"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" --seed "$seed" 2>&1 \
        | tee "$LOGS/seed${seed}.log"
    status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${status[0]}" -ne 0 || "${status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED seed${seed}; 保留现场。" >&2; failed=1
    else
        echo "[$(date '+%F %T')] DONE seed${seed} test ROI"
    fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi
echo "M5c内部test ROI矩阵完成: $OUTPUT"
