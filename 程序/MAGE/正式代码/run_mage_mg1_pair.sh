#!/usr/bin/env bash
set -uo pipefail

# Run the preregistered MG1 real/shuffle pair on one explicitly selected GPU.
PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/MAGE/正式代码/train_mage_mg1_teacher.py"
SUMMARY_SCRIPT="$PROJECT_ROOT/程序/MAGE/正式代码/summarize_mage_mg1.py"
RESULT_ROOT="$PROJECT_ROOT/结果/MAGE/MG1灰度局部教师_20260817/正式验证集筛选"
LOG_ROOT="$RESULT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "请先用nvidia-smi选择空闲GPU，再设置 CUDA_DEVICE=<物理编号>。" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi不可用，拒绝启动MG1。" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    tr -d ' ' | grep -qx "$CUDA_DEVICE"; then
    echo "GPU $CUDA_DEVICE 不存在。" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
failures=()
for mode in real patch_shuffle; do
    run_name="mg1_${mode}_efficientnet_b0_seed42"
    run_dir="$RESULT_ROOT/$run_name"
    config="$run_dir/config.json"
    checkpoint="$run_dir/mg1_best_teacher.pth"
    log="$LOG_ROOT/${run_name}.log"
    if [[ -f "$config" && -f "$checkpoint" ]]; then
        echo "[SKIP] $run_name 已有完整产物"
        continue
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[FAILED] $run_name 存在残缺目录，请人工归档后再运行。" >&2
        failures+=("$run_name:partial")
        continue
    fi
    echo "[$(date '+%F %T')] START $run_name" | tee "$log"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" "$SCRIPT" \
        --input-mode "$mode" \
        --seed 42 \
        --run-name "$run_name" 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE $run_name" | tee -a "$log"
    else
        echo "[$(date '+%F %T')] FAILED $run_name status=$status" | tee -a "$log"
        failures+=("$run_name:$status")
    fi
done

if (( ${#failures[@]} > 0 )); then
    echo "MG1失败项: ${failures[*]}" >&2
    exit 1
fi

"$PYTHON" "$SUMMARY_SCRIPT" --result-root "$RESULT_ROOT" 2>&1 | \
    tee "$LOG_ROOT/mg1_pair_summary.log"
summary_status=${PIPESTATUS[0]}
if [[ $summary_status -ne 0 ]]; then
    echo "MG1统一汇总失败，status=$summary_status" >&2
    exit "$summary_status"
fi
