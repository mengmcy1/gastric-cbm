#!/usr/bin/env bash
# M1正式矩阵启动器：统一运行来源内平衡主实验和全量诊断对照。
# 安全边界：脚本不传--evaluate-test，因此所有运行只用train/val选择模型。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m1_localization.py"
BALANCED_MANIFEST="$PROJECT_ROOT/数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/08_M1辅助定位清单_20260810/m1_balanced_keep_primary_1to1p3_split_seed42.csv"
FULL_MANIFEST="$PROJECT_ROOT/数据整理记录/图像裁剪/胃早癌概念提取训练集0804_预处理_v1/09_M1全量诊断清单_20260810/m1_full_keep_split_seed42.csv"
BALANCED_M0_ROOT="$PROJECT_ROOT/结果/M0平衡_0804/正式验证集筛选"
FULL_M0_ROOT="$PROJECT_ROOT/结果/M0全量诊断_0804/正式验证集筛选"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M1辅助定位_0804/正式验证集筛选"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-1}"


usage() {
    # 要求显式选择数据范围和种子，防止误启动完整六组长任务。
    cat <<'EOF'
用法:
  run_m1_matrix.sh <balanced|full|all> <42|202|503|逗号分隔列表>

示例:
  # 首先只跑平衡主实验seed42并验收
  CUDA_DEVICE=1 run_m1_matrix.sh balanced 42

  # 若尚未运行任何正式组，可一次运行完整六组
  CUDA_DEVICE=1 run_m1_matrix.sh all 42,202,503

  # 若平衡seed42已完成并验收，继续剩余五组
  CUDA_DEVICE=1 run_m1_matrix.sh balanced 202,503
  CUDA_DEVICE=1 run_m1_matrix.sh full 42,202,503

实时监测（另开终端）:
  tail -F /home/mcy/gastric-cbm/结果/M1辅助定位_0804/正式验证集筛选/logs/m1_balanced_keep_efficientnet_b0_seed42.log
  watch -n 2 nvidia-smi

说明:
  - 默认不评估internal test；本脚本不会传--evaluate-test。
  - 每组使用同数据角色、同种子的M0最佳权重初始化。
  - Ctrl+C退出tail/watch只停止监视；前台运行矩阵的终端中Ctrl+C会终止训练。
EOF
}


validate_seed() {
    # M1正式矩阵只允许使用M0已经完成并冻结的三个随机种子。
    case "$1" in
        42|202|503) ;;
        *)
            echo "非法seed: $1；只允许42/202/503。" >&2
            exit 2
            ;;
    esac
}


run_one() {
    # 运行一个数据角色和种子的M1，保存独立目录与实时日志。
    local dataset_role="$1"
    local seed="$2"
    local manifest
    local m0_root
    local run_name

    if [[ "$dataset_role" == "balanced" ]]; then
        manifest="$BALANCED_MANIFEST"
        m0_root="$BALANCED_M0_ROOT"
        run_name="m1_balanced_keep_efficientnet_b0_seed${seed}"
    else
        manifest="$FULL_MANIFEST"
        m0_root="$FULL_M0_ROOT"
        run_name="m1_full_keep_efficientnet_b0_seed${seed}"
    fi

    local m0_checkpoint="$m0_root/m0_${dataset_role}_keep_efficientnet_b0_seed${seed}/efficientnet_b0_debiased_best.pth"
    if [[ "$dataset_role" == "full" ]]; then
        m0_checkpoint="$m0_root/m0_full_keep_efficientnet_b0_seed${seed}/efficientnet_b0_debiased_best.pth"
    fi
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"

    [[ -f "$manifest" ]] || { echo "缺少M1 manifest: $manifest" >&2; exit 1; }
    [[ -f "$m0_checkpoint" ]] || { echo "缺少对应M0权重: $m0_checkpoint" >&2; exit 1; }
    if [[ -e "$run_dir" ]]; then
        echo "正式输出已存在，拒绝覆盖: $run_dir" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] START $run_name"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --manifest "$manifest" \
        --image-root "$PROJECT_ROOT" \
        --m0-checkpoint "$m0_checkpoint" \
        --output-root "$OUTPUT_ROOT" \
        --run-name "$run_name" \
        --seed "$seed" \
        --batch-size 32 \
        --num-workers 4 \
        --warmup-epochs 5 \
        --joint-epochs 20 \
        --warmup-lr 1e-3 \
        --joint-lr 1e-4 \
        --weight-decay 1e-4 \
        --lambda-loc 1.0 \
        --lambda-size 0.1 \
        --lambda-offset 1.0 \
        --classification-noninferiority 0.005 \
        --crop-min-bbox-retention 0.80 \
        --early-stop-patience 8 2>&1 | tee "$log_file"
    echo "[$(date '+%F %T')] DONE $run_name"
}


main() {
    # 校验显卡、范围和种子后，按明确顺序串行执行所选矩阵。
    if [[ $# -ne 2 ]]; then
        usage
        exit 2
    fi
    local scope="$1"
    local seed_spec="$2"
    local datasets=()
    case "$scope" in
        balanced) datasets=(balanced) ;;
        full) datasets=(full) ;;
        all) datasets=(balanced full) ;;
        *) usage; exit 2 ;;
    esac

    IFS=',' read -r -a seeds <<< "$seed_spec"
    if [[ ${#seeds[@]} -eq 0 ]]; then
        echo "至少指定一个seed。" >&2
        exit 2
    fi
    for seed in "${seeds[@]}"; do
        validate_seed "$seed"
    done

    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "未找到nvidia-smi，拒绝启动正式GPU矩阵。" >&2
        exit 1
    }
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    mkdir -p "$LOG_ROOT"
    cd "$PROJECT_ROOT"

    for dataset_role in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            run_one "$dataset_role" "$seed"
        done
    done
    echo "所选M1矩阵运行完成。日志目录: $LOG_ROOT"
}


main "$@"
