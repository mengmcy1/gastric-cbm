#!/usr/bin/env bash
set -uo pipefail

# Run the frozen MG2-L A-long/C-long duration sensitivity matrix, then select
# the attention-first SAE handoff checkpoint.  This script never reads test,
# internal test or external data and never modifies the original MG2 products.
# Usage: CUDA_DEVICE=<physical GPU index> bash run_mage_mg2l_duration_matrix.sh
PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
CODE_ROOT="$PROJECT_ROOT/程序/MAGE/正式代码"
TRAIN="$CODE_ROOT/train_mage_mg2_student.py"
SUMMARIZER="$CODE_ROOT/summarize_mage_mg2l_duration.py"
RESULT_ROOT="$PROJECT_ROOT/结果/MAGE/MG2L训练轮数敏感性_20260818"
RUN_ROOT="$RESULT_ROOT/正式验证集筛选"
LOG_ROOT="$RUN_ROOT/logs"
BETA_JSON="$PROJECT_ROOT/结果/MAGE/MG2全图学生蒸馏_20260818/beta_calibration_seed42.json"
SUMMARY_JSON="$RESULT_ROOT/mg2l_attention_sae_summary_seed42.json"
CUDA_DEVICE="${CUDA_DEVICE:-}"
DRY_RUN="${MG2L_DRY_RUN:-0}"

arm_complete() {
    local arm="$1" lower run_name dir
    lower=$(echo "$arm" | tr 'A-Z' 'a-z')
    run_name="mg2l_arm${lower}_efficientnet_b0_seed42"
    dir="$RUN_ROOT/$run_name"
    [[ -f "$dir/config.json" && -f "$dir/training_history.csv" \
        && -f "$dir/val_image_predictions.csv" \
        && -f "$dir/val_patient_predictions.csv" \
        && -f "$dir/mg2_arm${lower}_best_student.pth" ]]
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] MG2-L: A-long -> C-long -> SAE交接汇总"
    for ARM in A C; do
        if arm_complete "$ARM"; then
            echo "[DRY-RUN] $ARM-long已完整，将跳过"
        else
            echo "[DRY-RUN] $ARM-long将从头运行: stageA=10, stageB<=40"
        fi
    done
    exit 0
fi

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "请先用nvidia-smi选择空闲GPU，再设置CUDA_DEVICE。" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi不可用，拒绝启动。" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    tr -d ' ' | grep -qx "$CUDA_DEVICE"; then
    echo "GPU $CUDA_DEVICE不存在。" >&2
    exit 2
fi
if [[ ! -f "$BETA_JSON" ]]; then
    echo "冻结beta校准JSON不存在: $BETA_JSON" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
status_all=0
for ARM in A C; do
    LOWER=$(echo "$ARM" | tr 'A-Z' 'a-z')
    RUN_NAME="mg2l_arm${LOWER}_efficientnet_b0_seed42"
    RUN_DIR="$RUN_ROOT/$RUN_NAME"
    LOG="$LOG_ROOT/$RUN_NAME.log"
    if arm_complete "$ARM"; then
        echo "[SKIP] $ARM-long已有完整产物: $RUN_DIR"
        continue
    fi
    if [[ -e "$RUN_DIR" ]]; then
        echo "[FAILED] $ARM-long存在残缺目录，保留现场并拒绝覆盖。" >&2
        status_all=1
        continue
    fi
    EXTRA=()
    if [[ "$ARM" == "C" ]]; then
        EXTRA=(--beta-calibration-json "$BETA_JSON")
    fi
    echo "[$(date '+%F %T')] START $ARM-long GPU=$CUDA_DEVICE" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" -u "$TRAIN" \
        --arm "$ARM" --device cuda --seed 42 \
        --stage-a-epochs 10 --stage-b-epochs 40 \
        --output-root "$RUN_ROOT" --run-name "$RUN_NAME" \
        "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE $ARM-long" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAILED $ARM-long status=$status" | tee -a "$LOG"
        status_all=1
    fi
done

if [[ $status_all -ne 0 ]]; then
    echo "MG2-L存在失败项，不执行SAE交接汇总。" >&2
    exit "$status_all"
fi
for ARM in A C; do
    if ! arm_complete "$ARM"; then
        echo "$ARM-long产物不完整，拒绝汇总。" >&2
        exit 1
    fi
done
if [[ -f "$SUMMARY_JSON" ]]; then
    echo "[SKIP] MG2-L汇总已存在: $SUMMARY_JSON"
    exit 0
fi
SUMMARY_LOG="$LOG_ROOT/mg2l_attention_sae_summary.log"
echo "[$(date '+%F %T')] SUMMARIZE MG2-L" | tee -a "$SUMMARY_LOG"
PYTHONUNBUFFERED=1 "$PYTHON" -u "$SUMMARIZER" 2>&1 | tee -a "$SUMMARY_LOG"
status=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] SUMMARIZE status=$status" | tee -a "$SUMMARY_LOG"
exit "$status"
