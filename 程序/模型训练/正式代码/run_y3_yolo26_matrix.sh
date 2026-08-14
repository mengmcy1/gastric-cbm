#!/usr/bin/env bash
# Run fixed-640 Y3 Balanced or Full replicates sequentially on one inspected GPU.

set -uo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
CODE_DIR="$ROOT/程序/模型训练/正式代码"
OUTPUT_ROOT="$ROOT/结果/YOLO26定位_0804/Y3固定640三种子"
LOG_DIR="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:?运行前先检查nvidia-smi，再通过CUDA_DEVICE指定目标GPU}"
ROLE="${1:?Usage: run_y3_yolo26_matrix.sh <balanced|full|all>}"

if [[ "$ROLE" != "balanced" && "$ROLE" != "full" && "$ROLE" != "all" ]]; then
  echo "Invalid role: $ROLE" >&2
  exit 2
fi

cd "$ROOT" || exit 1
nvidia-smi
mkdir -p "$LOG_DIR"

run_replicate() {
  local role="$1"
  local seed="$2"
  local run_name="y3_${role}_yolo26s_640_seed${seed}"
  local run_dir="$OUTPUT_ROOT/$run_name"
  local log_path="$LOG_DIR/${run_name}.log"
  if [[ -f "$run_dir/y3_geometry_config.json" ]]; then
    echo "SKIP $run_name: 完整训练与几何结果已存在"
    return 0
  fi
  local resume_args=()
  if [[ -e "$run_dir" ]]; then
    if [[ -f "$run_dir/weights/last.pt" && -f "$run_dir/results.csv" \
          && ! -e "$run_dir/y3_train_config.json" ]]; then
      resume_args=(--resume)
      echo "RESUME $run_name: 从现有last.pt继续正式训练"
    else
      echo "ERROR $run_name: 存在不可安全续训的残缺产物，请人工检查并归档" >&2
      return 1
    fi
  fi

  echo "[$(date '+%F %T')] START $run_name" | tee "$log_path"
  "$PYTHON" "$CODE_DIR/train_y3_yolo26.py" \
    --role "$role" --seed "$seed" --device "$CUDA_DEVICE" "${resume_args[@]}" \
    2>&1 | tee -a "$log_path"
  local train_status=${PIPESTATUS[0]}
  if [[ $train_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED train $run_name" | tee -a "$log_path"
    return "$train_status"
  fi
  "$PYTHON" "$CODE_DIR/evaluate_y3_yolo26.py" \
    --role "$role" --seed "$seed" --device "$CUDA_DEVICE" 2>&1 | tee -a "$log_path"
  local eval_status=${PIPESTATUS[0]}
  if [[ $eval_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED evaluation $run_name" | tee -a "$log_path"
    return "$eval_status"
  fi
  echo "[$(date '+%F %T')] DONE $run_name" | tee -a "$log_path"
}

run_role() {
  local role="$1"
  local status=0
  local seeds=(42 202 503)
  if [[ "$role" == "balanced" ]]; then
    seeds=(202 503)
    if [[ ! -f "$ROOT/结果/YOLO26定位_0804/Y2平衡分辨率预筛/y2b_yolo26s_640_seed42/y2_geometry_config.json" ]]; then
      echo "ERROR: 缺少Balanced seed42的Y2-B 640冻结结果" >&2
      return 1
    fi
  fi
  for seed in "${seeds[@]}"; do
    run_replicate "$role" "$seed" || status=1
  done
  if [[ $status -ne 0 ]]; then
    echo "Y3 $role存在失败seed，保留现场且不执行汇总" >&2
    return 1
  fi
  if [[ -f "$OUTPUT_ROOT/自动汇总_${role}/y3_summary.json" ]]; then
    echo "SKIP Y3 $role汇总: 完整汇总已存在"
    return 0
  fi
  if [[ -e "$OUTPUT_ROOT/自动汇总_${role}" ]]; then
    echo "ERROR Y3 $role汇总目录残缺，请人工检查并归档；脚本不覆盖" >&2
    return 1
  fi
  "$PYTHON" "$CODE_DIR/summarize_y3_yolo26.py" --role "$role" 2>&1 \
    | tee "$LOG_DIR/y3_${role}_summary.log"
}

status=0
if [[ "$ROLE" == "balanced" || "$ROLE" == "all" ]]; then
  run_role balanced || status=1
fi
if [[ "$ROLE" == "full" || "$ROLE" == "all" ]]; then
  run_role full || status=1
fi
exit "$status"
