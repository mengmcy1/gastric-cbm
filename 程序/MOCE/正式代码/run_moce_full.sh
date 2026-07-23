#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="$PROJECT_DIR/程序/MOCE/正式代码/moce_curated_concept.py"
RESULT_ROOT="$PROJECT_DIR/结果/MOCE聚类/概念严格平衡_v1"
LOG_DIR="$RESULT_ROOT/logs"

export CUDA_VISIBLE_DEVICES=1
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

run_model() {
    local model="$1"
    local log_file="$LOG_DIR/${model}_full.log"
    echo "[$(date '+%F %T')] 开始运行 $model" | tee -a "$log_file"
    "$PYTHON" "$ENTRY" --model "$model" --mode full 2>&1 | tee -a "$log_file"
    echo "[$(date '+%F %T')] $model 运行完成" | tee -a "$log_file"
}

echo "[$(date '+%F %T')] MOCE全量任务启动，GPU=$CUDA_VISIBLE_DEVICES"
run_model resnet50
run_model efficientnet_b0
echo "[$(date '+%F %T')] 两个模型全部运行完成"
