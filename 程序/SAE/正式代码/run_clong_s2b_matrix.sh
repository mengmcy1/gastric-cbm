#!/usr/bin/env bash
# S2b seed42固定顺序：B -> gamma_pool校准 -> C -> D -> N -> 自动汇总。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="$PROJECT_ROOT/程序/SAE/正式代码/clong_s2b_discovery.py"
SUMMARY="$PROJECT_ROOT/程序/SAE/正式代码/summarize_clong_s2b.py"
OUTPUT="$PROJECT_ROOT/结果/SAE/CLong_S2b结构重构_20260820"
LOG_DIR="$OUTPUT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || exit 1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "未找到nvidia-smi，拒绝启动S2b正式矩阵。"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

run_arm() {
  local arm="$1"
  local lower
  lower="$(printf '%s' "$arm" | tr '[:upper:]' '[:lower:]')"
  local experiment="s4${lower}_clong_seed42"
  local run_dir="$OUTPUT/$experiment"
  local log="$LOG_DIR/${experiment}.log"
  if [[ -f "$run_dir/config.json" && -f "$run_dir/sae_best.pth" && -f "$run_dir/training_history.csv" ]]; then
    echo "[$(date '+%F %T')] SKIP $experiment（完整产物已存在）"
    return 0
  fi
  if [[ -f "$run_dir/protocol_failure.json" && -f "$run_dir/sae_best.pth" && -f "$run_dir/training_history.csv" ]]; then
    echo "[$(date '+%F %T')] SKIP $experiment（已记录预注册协议失败）"
    return 0
  fi
  if [[ "$arm" =~ ^(B|D)$ && -f "$run_dir/sae_best.pth" && -f "$run_dir/training_history.csv" && ! -e "$run_dir/config.json" ]]; then
    echo "[$(date '+%F %T')] RESUME_POSTPROCESS $experiment" | tee -a "$log"
    "$PYTHON" "$ENTRY" --arm "$arm" --seed 42 --experiment "$experiment" \
      --device cuda --postprocess-existing 2>&1 | tee -a "$log"
    local resume_status="${PIPESTATUS[0]}"
    if [[ -f "$run_dir/protocol_failure.json" ]]; then
      echo "[$(date '+%F %T')] PROTOCOL_FAILURE $experiment（已记录，继续后续独立臂）" | tee -a "$log"
      return 0
    fi
    if [[ "$resume_status" -ne 0 ]]; then
      echo "[$(date '+%F %T')] POSTPROCESS_FAILED $experiment status=$resume_status" | tee -a "$log"
      return "$resume_status"
    fi
    echo "[$(date '+%F %T')] DONE $experiment（后处理恢复）" | tee -a "$log"
    return 0
  fi
  if [[ -e "$run_dir" ]]; then
    echo "发现残缺产物，请人工归档后再运行，不自动覆盖: $run_dir"
    return 1
  fi
  echo "[$(date '+%F %T')] START $experiment" | tee -a "$log"
  "$PYTHON" "$ENTRY" --arm "$arm" --seed 42 --experiment "$experiment" \
    --device cuda 2>&1 | tee -a "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    if [[ -f "$run_dir/protocol_failure.json" ]]; then
      echo "[$(date '+%F %T')] PROTOCOL_FAILURE $experiment（已记录，继续后续独立臂）" | tee -a "$log"
      return 0
    fi
    echo "[$(date '+%F %T')] FAILED $experiment status=$status" | tee -a "$log"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE $experiment" | tee -a "$log"
}

run_arm B || exit $?

GAMMA_JSON="$OUTPUT/gamma_pool_calibration_seed42.json"
GAMMA_LOG="$LOG_DIR/gamma_pool_calibration_seed42.log"
if [[ ! -f "$GAMMA_JSON" ]]; then
  echo "[$(date '+%F %T')] START gamma_pool calibration" | tee -a "$GAMMA_LOG"
  "$PYTHON" "$ENTRY" --arm C --seed 42 --experiment s4c_clong_seed42 \
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

run_arm C || exit $?
run_arm D || exit $?
run_arm N || exit $?

SUMMARY_LOG="$LOG_DIR/s2b_summary_seed42.log"
echo "[$(date '+%F %T')] SUMMARIZE" | tee -a "$SUMMARY_LOG"
"$PYTHON" "$SUMMARY" --seed 42 2>&1 | tee -a "$SUMMARY_LOG"
status="${PIPESTATUS[0]}"
if [[ "$status" -ne 0 ]]; then
  echo "[$(date '+%F %T')] SUMMARIZE FAILED status=$status" | tee -a "$SUMMARY_LOG"
  exit "$status"
fi
echo "[$(date '+%F %T')] S2b matrix DONE" | tee -a "$SUMMARY_LOG"
