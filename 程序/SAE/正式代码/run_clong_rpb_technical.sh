#!/usr/bin/env bash
# 构建RP-B train-only技术主表和Feature families；不读取val或测试集。
set -Eeuo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
CODE="$PROJECT_ROOT/程序/SAE/正式代码"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/结果/SAE/RP_B_Technical_20260825"
LOG_DIR="$OUTPUT_ROOT/logs"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "RP-B输出根已存在，禁止覆盖: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

"$PYTHON" "$CODE/test_clong_rpb.py" 2>&1 | tee "$LOG_DIR/tests.log"
"$PYTHON" -u "$CODE/build_clong_rpb_anchor_master.py" 2>&1 | tee "$LOG_DIR/rpb_anchor_master.log"
"$PYTHON" -u "$CODE/build_clong_rpb_families.py" 2>&1 | tee "$LOG_DIR/rpb_families.log"
"$PYTHON" -u "$CODE/summarize_clong_rpb.py" 2>&1 | tee "$LOG_DIR/rpb_summary.log"
echo "[$(date '+%F %T')] RP-B technical analysis completed" | tee "$LOG_DIR/completed.log"
