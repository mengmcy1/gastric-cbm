#!/usr/bin/env bash
# RP-C2正式五档计算；Gate A失败时不启动controls，不自动覆盖或续跑。
set -Eeuo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
CODE_ROOT="$PROJECT_ROOT/程序/SAE/正式代码"
RESULT_ROOT="$PROJECT_ROOT/结果/SAE/RP_C2_Intervention_20260825"
FORMAL_ROOT="$RESULT_ROOT/formal"
LOG_ROOT="$RESULT_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/home/mcy/miniconda3/envs/gastric-cbm/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

[[ ! -e "$FORMAL_ROOT" ]] || {
  echo "RP-C2 formal目录已存在，禁止覆盖或自动续跑: $FORMAL_ROOT" >&2
  exit 1
}
mkdir -p "$LOG_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$CODE_ROOT/test_clong_rpc2.py" \
  2>&1 | tee "$LOG_ROOT/formal_tests.log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -u "$CODE_ROOT/run_clong_rpc2_formal.py" \
  --device cuda 2>&1 | tee "$LOG_ROOT/formal.log"

echo "RP-C2 formal完成: $FORMAL_ROOT"
