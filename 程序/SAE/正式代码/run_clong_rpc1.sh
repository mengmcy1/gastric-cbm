#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/mcy/gastric-cbm
CODE_ROOT="$PROJECT_ROOT/程序/SAE/正式代码"
RESULT_ROOT="$PROJECT_ROOT/结果/SAE/RP_C1_Effect_Screen_20260825"
LOG_ROOT="$RESULT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/mcy/miniconda3/envs/gastric-cbm/bin/python}"

mkdir -p "$LOG_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$CODE_ROOT/test_clong_rpc.py" \
  2>&1 | tee "$LOG_ROOT/tests.log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$CODE_ROOT/run_clong_rpc1_effect_screen.py" \
  --device cuda 2>&1 | tee "$LOG_ROOT/effect_screen.log"
"$PYTHON_BIN" "$CODE_ROOT/summarize_clong_rpc1.py" \
  2>&1 | tee "$LOG_ROOT/summary.log"

echo "RP-C1完成: $RESULT_ROOT/formal"
