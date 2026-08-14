#!/usr/bin/env bash
# 一次性执行Y4锁定内部test揭盲；调用前必须先人工检查nvidia-smi。
set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON_BIN="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/模型训练/正式代码/evaluate_y4_yolo26_locked_internal_test.py"
OUTPUT="$PROJECT_ROOT/结果/YOLO26定位_0804/Y4锁定内部测试_20260814"
LOG_DIR="$PROJECT_ROOT/结果/YOLO26定位_0804/Y4锁定内部测试_20260814_logs"
CUDA_DEVICE="${CUDA_DEVICE:?请显式设置CUDA_DEVICE，并先检查nvidia-smi}"

if [[ -e "$OUTPUT" ]]; then
  echo "Y4输出已存在，拒绝重复揭盲: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/y4_locked_internal_test_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] START Y4 locked internal test, GPU=$CUDA_DEVICE" | tee "$LOG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$SCRIPT" \
  --device 0 \
  --bootstrap 5000 \
  2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] DONE Y4 locked internal test" | tee -a "$LOG"
