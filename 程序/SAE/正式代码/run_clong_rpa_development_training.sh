#!/usr/bin/env bash
# RP-A development字典训练：固定顺序seed43 -> seed44；不运行matching/bootstrap。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="$PROJECT_ROOT/程序/SAE/正式代码/clong_rpa_train_development.py"
TEST="$PROJECT_ROOT/程序/SAE/正式代码/test_clong_rpa_development.py"
OUTPUT="${RPA_DEVELOPMENT_ROOT:-$PROJECT_ROOT/结果/SAE/RP_A_Development_20260824}"
LOG_DIR="$OUTPUT/logs"
CUDA_DEVICE="${CUDA_DEVICE:?必须在启动前检查GPU并显式设置CUDA_DEVICE}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "未找到nvidia-smi，拒绝启动。"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

"$PYTHON" "$TEST" || exit $?

for seed in 43 44; do
  run="rpa_development_clong_seed${seed}"
  directory="$OUTPUT/$run"
  log="$LOG_DIR/${run}.log"
  if [[ -f "$directory/config.json" && -f "$directory/sae_best.pth" && \
        -f "$directory/training_history.csv" ]]; then
    echo "[$(date '+%F %T')] SKIP $run（完整训练产物已存在）"
    continue
  fi
  if [[ -e "$directory" ]]; then
    echo "发现残缺产物，请人工归档后再运行，不自动覆盖: $directory"
    exit 1
  fi
  echo "[$(date '+%F %T')] START $run" | tee -a "$log"
  "$PYTHON" -u "$ENTRY" --seed "$seed" --device cuda 2>&1 | tee -a "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED $run status=$status" | tee -a "$log"
    exit "$status"
  fi
  echo "[$(date '+%F %T')] DONE $run" | tee -a "$log"
done

echo "RP-A development字典训练完成；尚未执行matching或bootstrap。"
