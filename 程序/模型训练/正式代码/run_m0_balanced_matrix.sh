#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
MANIFEST="$PROJECT_ROOT/数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/07_M0来源内平衡清单_20260805/m0_balanced_keep_primary_1to1p3_split_seed42.csv"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M0平衡_0804/正式验证集筛选"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "未找到nvidia-smi，拒绝启动正式GPU矩阵。" >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

for model in efficientnet_b0 resnet50; do
    if [[ "$model" == "efficientnet_b0" ]]; then
        entry="程序/模型训练/正式代码/efficientnet_train_debiased.py"
    else
        entry="程序/模型训练/正式代码/resnet_train_debiased.py"
    fi
    for seed in 42 202 503; do
        run_name="m0_balanced_keep_${model}_seed${seed}"
        run_dir="$OUTPUT_ROOT/$run_name"
        log_file="$LOG_ROOT/${run_name}.log"
        if [[ -e "$run_dir" ]]; then
            echo "正式输出已存在，拒绝覆盖: $run_dir" >&2
            exit 1
        fi
        echo "[$(date '+%F %T')] START $run_name"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
            "$PYTHON" -u "$entry" \
            --manifest "$MANIFEST" \
            --image-root "$PROJECT_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --run-name "$run_name" \
            --seed "$seed" \
            --batch-size 32 \
            --num-workers 4 \
            --stage1-epochs 10 \
            --stage2-epochs 20 \
            --early-stop-patience 8 \
            --defer-test 2>&1 | tee "$log_file"
        echo "[$(date '+%F %T')] DONE $run_name"
    done
done

echo "全部6组平衡M0训练完成。日志目录: $LOG_ROOT"
