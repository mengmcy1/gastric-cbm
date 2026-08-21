#!/usr/bin/env bash
# S2c Matryoshka patch SAE固定顺序：gamma_pool校准 -> seed42正式训练+逐K评价。
# seed202/503只有在seed42选出合格K之后才允许启动，本脚本不自动运行它们。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="$PROJECT_ROOT/程序/SAE/正式代码/clong_s2c_matryoshka.py"
OUTPUT="$PROJECT_ROOT/结果/SAE/CLong_S2c_Matryoshka_20260821"
LOG_DIR="$OUTPUT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "未找到nvidia-smi，拒绝启动S2c正式运行。"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

GAMMA_JSON="$OUTPUT/gamma_pool_calibration_seed42.json"
GAMMA_LOG="$LOG_DIR/gamma_pool_calibration_seed42.log"
if [[ ! -f "$GAMMA_JSON" ]]; then
  echo "[$(date '+%F %T')] START gamma_pool calibration" | tee -a "$GAMMA_LOG"
  "$PYTHON" -u "$ENTRY" --seed 42 --experiment s2c_clong_seed42 \
    --device cuda --calibrate-gamma 2>&1 | tee -a "$GAMMA_LOG"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED gamma_pool status=$status" | tee -a "$GAMMA_LOG"
    exit "$status"
  fi
  echo "[$(date '+%F %T')] DONE gamma_pool calibration" | tee -a "$GAMMA_LOG"
else
  echo "[$(date '+%F %T')] SKIP gamma_pool（JSON已存在，正式入口会复核血缘）"
fi

RUN_DIR="$OUTPUT/s2c_clong_seed42"
LOG="$LOG_DIR/s2c_clong_seed42.log"
if [[ -f "$RUN_DIR/config.json" && -f "$RUN_DIR/sae_best.pth" && -f "$RUN_DIR/training_history.csv" ]]; then
  echo "[$(date '+%F %T')] SKIP s2c_clong_seed42（完整产物已存在）"
  exit 0
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "发现残缺产物，请人工归档后再运行，不自动覆盖: $RUN_DIR"
  exit 1
fi
echo "[$(date '+%F %T')] START s2c_clong_seed42" | tee -a "$LOG"
"$PYTHON" -u "$ENTRY" --seed 42 --experiment s2c_clong_seed42 \
  --device cuda 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[$(date '+%F %T')] FAILED s2c_clong_seed42 status=$status" | tee -a "$LOG"
  exit "$status"
fi
echo "[$(date '+%F %T')] S2c seed42 DONE（selected_k见config.json）" | tee -a "$LOG"
