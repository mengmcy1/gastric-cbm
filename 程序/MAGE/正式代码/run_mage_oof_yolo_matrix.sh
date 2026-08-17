#!/usr/bin/env bash
# Train MG0b cross-fitted detectors and predict each unseen holdout fold.

set -uo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
CODE="$ROOT/程序/MAGE/正式代码/train_mage_oof_yolo.py"
OUTPUT="$ROOT/结果/MAGE/MG0b_OOF_YOLO_20260817"
LOG_DIR="$OUTPUT/logs"
CUDA_DEVICE="${CUDA_DEVICE:?先检查nvidia-smi，再用CUDA_DEVICE指定空闲物理GPU}"
FOLDS="${1:-0,1,2,3,4}"

cd "$ROOT" || exit 1
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
mkdir -p "$LOG_DIR"

status=0
IFS=',' read -r -a fold_array <<< "$FOLDS"
for fold in "${fold_array[@]}"; do
  if [[ ! "$fold" =~ ^[0-4]$ ]]; then
    echo "非法fold: $fold" >&2
    exit 2
  fi
  name="mg0b_oof_yolo26s_fold${fold}"
  run_dir="$OUTPUT/$name"
  log="$LOG_DIR/$name.log"
  if [[ -f "$run_dir/mg0b_config.json" && -f "$run_dir/holdout_top1_predictions.csv" ]]; then
    echo "SKIP $name: 正式产物已完整存在"
    continue
  fi
  resume_args=()
  if [[ -e "$run_dir" ]]; then
    if [[ -f "$run_dir/weights/last.pt" && -f "$run_dir/results.csv" \
          && ! -e "$run_dir/mg0b_config.json" ]]; then
      resume_args=(--resume)
      echo "RESUME $name: 从last.pt恢复"
    else
      echo "FAILED $name: 残缺目录不可安全恢复，请人工归档" >&2
      status=1
      continue
    fi
  fi
  echo "[$(date '+%F %T')] START $name" | tee "$log"
  PYTHONUNBUFFERED=1 "$PYTHON" -u "$CODE" \
    --fold "$fold" --device "$CUDA_DEVICE" "${resume_args[@]}" 2>&1 | tee -a "$log"
  code=${PIPESTATUS[0]}
  if [[ $code -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED $name status=$code" | tee -a "$log"
    status=1
    continue
  fi
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$log"
done

exit "$status"
