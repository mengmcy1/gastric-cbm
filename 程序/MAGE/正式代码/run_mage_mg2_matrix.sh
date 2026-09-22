#!/usr/bin/env bash
set -uo pipefail

# Run the frozen MG2 A/B/C student matrix on one explicitly selected GPU,
# then auto-summarize the eight release gates when all three arms complete.
# Arms run in order A -> B -> C with identical seed/init/sampling/budget;
# only the loss differs. Failed arms keep their partial state untouched and
# are never summarized as success; complete arms are skipped idempotently.
# If all three arm products are complete and the gate summary already exists,
# the summarize step is skipped idempotently (never reported as failure).
#
# Usage:
#   CUDA_DEVICE=<物理GPU编号> [BETA_CALIBRATION_JSON=<路径>] bash run_mage_mg2_matrix.sh
#   MG2_MATRIX_DRY_RUN=1 bash run_mage_mg2_matrix.sh   # 只打印计划，不执行
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${MAGE_PYTHON:-python}"
SCRIPT="$SCRIPT_DIR/train_mage_mg2_student.py"
SUMMARIZER="$SCRIPT_DIR/summarize_mage_mg2.py"
RESULT_ROOT="$PROJECT_ROOT/结果/MAGE/MG2全图学生蒸馏_20260818"
RUN_ROOT="$RESULT_ROOT/正式验证集筛选"
LOG_ROOT="$RUN_ROOT/logs"
TEACHER_CACHE="$RESULT_ROOT/teacher_cache_mg1b_v3.pt"
BETA_JSON="${BETA_CALIBRATION_JSON:-$RESULT_ROOT/beta_calibration_seed42.json}"
SUMMARY_JSON="$RESULT_ROOT/mg2_gate_summary_seed42.json"
SUMMARY_LOG="$LOG_ROOT/mg2_gate_summary.log"
CUDA_DEVICE="${CUDA_DEVICE:-}"
DRY_RUN="${MG2_MATRIX_DRY_RUN:-0}"

# 返回0=完整，1=缺失/残缺；$1=arm大写字母。
arm_complete() {
    local lower
    lower=$(echo "$1" | tr 'A-Z' 'a-z')
    local dir="$RUN_ROOT/mg2_arm${lower}_efficientnet_b0_seed42"
    [[ -f "$dir/config.json" && -f "$dir/training_history.csv" \
        && -f "$dir/val_image_predictions.csv" \
        && -f "$dir/val_patient_predictions.csv" \
        && -f "$dir/mg2_arm${lower}_best_student.pth" ]]
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 只打印计划，不执行任何训练或汇总。"
    for ARM in A B C; do
        if arm_complete "$ARM"; then
            echo "[DRY-RUN] arm $ARM: 产物完整，将幂等跳过"
        else
            extra=""
            [[ "$ARM" == "C" ]] && extra=" --beta-calibration-json $BETA_JSON"
            echo "[DRY-RUN] arm $ARM: 将执行 $SCRIPT --arm $ARM --device cuda --seed 42$extra"
        fi
    done
    if [[ -f "$SUMMARY_JSON" ]]; then
        echo "[DRY-RUN] 汇总已存在，三组全部成功时将幂等跳过: $SUMMARY_JSON"
    else
        echo "[DRY-RUN] 三组全部成功后将自动执行: $SUMMARIZER -> $SUMMARY_JSON"
    fi
    exit 0
fi

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "请先用nvidia-smi选择空闲GPU，再设置 CUDA_DEVICE=<物理编号>。" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi不可用，拒绝启动MG2矩阵。" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits | \
    tr -d ' ' | grep -qx "$CUDA_DEVICE"; then
    echo "GPU $CUDA_DEVICE 不存在。" >&2
    exit 2
fi
if [[ ! -f "$TEACHER_CACHE" ]]; then
    echo "教师缓存不存在，请先运行build_mage_mg2_teacher_cache.py: $TEACHER_CACHE" >&2
    exit 2
fi
if [[ ! -f "$BETA_JSON" ]]; then
    echo "beta校准JSON不存在，请先运行--calibrate-beta: $BETA_JSON" >&2
    exit 2
fi

mkdir -p "$LOG_ROOT"
status_all=0
for ARM in A B C; do
    LOWER=$(echo "$ARM" | tr 'A-Z' 'a-z')
    RUN_NAME="mg2_arm${LOWER}_efficientnet_b0_seed42"
    RUN_DIR="$RUN_ROOT/$RUN_NAME"
    LOG="$LOG_ROOT/$RUN_NAME.log"
    if arm_complete "$ARM"; then
        echo "[SKIP] $RUN_NAME 已有完整正式产物: $RUN_DIR"
        continue
    fi
    if [[ -e "$RUN_DIR" ]]; then
        echo "[FAILED] $RUN_NAME 存在残缺目录，保留现场并请人工归档后再运行。" >&2
        status_all=1
        continue
    fi
    EXTRA=()
    if [[ "$ARM" == "C" ]]; then
        EXTRA=(--beta-calibration-json "$BETA_JSON")
    fi
    # 追加写入，保留服务器/终端中断前的训练记录。
    echo "[$(date '+%F %T')] START $RUN_NAME GPU=$CUDA_DEVICE" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" -u "$SCRIPT" \
        --arm "$ARM" --device cuda --seed 42 "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE $RUN_NAME" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAILED $RUN_NAME status=$status（保留现场）" | tee -a "$LOG"
        status_all=1
    fi
done

if [[ $status_all -ne 0 ]]; then
    echo "[FAILED] 存在失败或残缺组，不执行八门槛汇总，保留现场。" >&2
    exit "$status_all"
fi
for ARM in A B C; do
    if ! arm_complete "$ARM"; then
        echo "[FAILED] arm $ARM 产物不完整，拒绝汇总。" >&2
        exit 1
    fi
done
if [[ -f "$SUMMARY_JSON" ]]; then
    echo "[SKIP] 八门槛汇总已存在，幂等跳过: $SUMMARY_JSON"
    exit 0
fi
echo "[$(date '+%F %T')] SUMMARIZE 八门槛判定" | tee -a "$SUMMARY_LOG"
PYTHONUNBUFFERED=1 "$PYTHON" -u "$SUMMARIZER" 2>&1 | tee -a "$SUMMARY_LOG"
status=${PIPESTATUS[0]}
if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] SUMMARIZE DONE -> $SUMMARY_JSON" | tee -a "$SUMMARY_LOG"
else
    echo "[$(date '+%F %T')] SUMMARIZE FAILED status=$status" | tee -a "$SUMMARY_LOG"
fi
exit "$status"
