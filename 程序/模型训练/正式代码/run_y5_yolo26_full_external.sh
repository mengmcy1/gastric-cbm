#!/usr/bin/env bash
# 运行Y4后探索性Full三seed外部诊断；启动前必须人工检查nvidia-smi。
set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON_BIN="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/模型训练/正式代码/evaluate_y5_yolo26_full_external.py"
OUTPUT="$PROJECT_ROOT/结果/YOLO26定位_0804/Y5外部诊断_Full_20260814"
LOG_DIR="$PROJECT_ROOT/结果/YOLO26定位_0804/Y5外部诊断_Full_20260814_logs"
CUDA_DEVICE="${CUDA_DEVICE:?请显式设置CUDA_DEVICE，并先检查nvidia-smi}"

if [[ -e "$OUTPUT" ]]; then
  echo "Y5输出已存在，拒绝重复外部投影: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/y5_full_external_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] START Y5 Full external, GPU=$CUDA_DEVICE" | tee "$LOG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$SCRIPT" \
  --device 0 \
  --bootstrap 2000 \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] DONE Y5 Full external" | tee -a "$LOG"
