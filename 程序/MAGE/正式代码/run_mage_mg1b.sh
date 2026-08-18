#!/usr/bin/env bash
set -uo pipefail

# Run the preregistered MG1b attention teacher on one explicitly selected GPU.
PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/MAGE/正式代码/train_mage_mg1b_attention_teacher.py"
RESULT_ROOT="$PROJECT_ROOT/结果/MAGE/MG1b注意力池化教师_20260818/正式验证集筛选"
RUN_NAME="mg1b_attention_efficientnet_b0_seed42"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"
LOG_ROOT="$RESULT_ROOT/logs"
LOG="$LOG_ROOT/$RUN_NAME.log"
CUDA_DEVICE="${CUDA_DEVICE:-}"

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "请先用nvidia-smi选择空闲GPU，再设置 CUDA_DEVICE=<物理编号>。" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi不可用，拒绝启动MG1b。" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    tr -d ' ' | grep -qx "$CUDA_DEVICE"; then
    echo "GPU $CUDA_DEVICE 不存在。" >&2
    exit 2
fi

config="$RUN_DIR/config.json"
product="$RUN_DIR/mg1b_best_teacher.pth"
diagnostic="$RUN_DIR/mg1b_best_diagnostic_ineligible.pth"
if [[ -f "$config" && ( -f "$product" || -f "$diagnostic" ) ]]; then
    echo "[SKIP] $RUN_NAME 已有完整正式或诊断产物: $RUN_DIR"
    exit 0
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "[FAILED] $RUN_NAME 存在残缺目录，请人工归档后再运行。" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"
echo "[$(date '+%F %T')] START $RUN_NAME GPU=$CUDA_DEVICE" | tee "$LOG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
    --device cuda \
    --seed 42 \
    --run-name "$RUN_NAME" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] DONE $RUN_NAME" | tee -a "$LOG"
else
    echo "[$(date '+%F %T')] FAILED_OR_INELIGIBLE $RUN_NAME status=$status" | tee -a "$LOG"
fi
exit "$status"
