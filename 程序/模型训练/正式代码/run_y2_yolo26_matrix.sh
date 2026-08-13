#!/usr/bin/env bash
# Run the frozen Y2-B 640/960 screen sequentially on one inspected GPU.

set -uo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
CODE_DIR="$ROOT/程序/模型训练/正式代码"
OUTPUT_ROOT="$ROOT/结果/YOLO26定位_0804/Y2平衡分辨率预筛"
LOG_DIR="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:?运行前先检查nvidia-smi，并通过CUDA_DEVICE指定目标GPU}"

cd "$ROOT" || exit 1
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "$LOG_DIR"

run_candidate() {
  local imgsz="$1"
  local run_name="y2b_yolo26s_${imgsz}_seed42"
  local run_dir="$OUTPUT_ROOT/$run_name"
  local log_path="$LOG_DIR/${run_name}.log"
  if [[ -f "$run_dir/y2_geometry_config.json" ]]; then
    echo "SKIP $run_name: 完整训练与几何结果已存在"
    return 0
  fi
  if [[ -e "$run_dir" ]]; then
    echo "ERROR $run_name: 存在残缺产物，请人工检查并归档；脚本不会覆盖" >&2
    return 1
  fi

  echo "[$(date '+%F %T')] START $run_name" | tee "$log_path"
  "$PYTHON" "$CODE_DIR/train_y2_yolo26.py" \
    --imgsz "$imgsz" --seed 42 --device "$CUDA_DEVICE" 2>&1 | tee -a "$log_path"
  local train_status=${PIPESTATUS[0]}
  if [[ $train_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED train $run_name" | tee -a "$log_path"
    return "$train_status"
  fi
  "$PYTHON" "$CODE_DIR/evaluate_y2_yolo26.py" \
    --imgsz "$imgsz" --seed 42 --device "$CUDA_DEVICE" 2>&1 | tee -a "$log_path"
  local eval_status=${PIPESTATUS[0]}
  if [[ $eval_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED evaluation $run_name" | tee -a "$log_path"
    return "$eval_status"
  fi
  echo "[$(date '+%F %T')] DONE $run_name" | tee -a "$log_path"
}

status=0
for imgsz in 640 960; do
  run_candidate "$imgsz" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "Y2-B存在失败候选，保留现场且不执行分辨率选择" >&2
  exit 1
fi

"$PYTHON" "$CODE_DIR/summarize_y2_yolo26.py" 2>&1 | tee "$LOG_DIR/y2_resolution_summary.log"
