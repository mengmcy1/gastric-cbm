#!/usr/bin/env bash
set -uo pipefail

# 运行S2预注册冻结的C-long SAE矩阵（2026-08-19冻结）：
#   15组 L1+margin 主矩阵（5宽度 x 3 lambda，seed42）
#   1组 无margin诊断（w10240/lambda=5e-4/gamma=0，seed42）
#   1组 Top-K备选（w10240/K=1024/gamma=0.1，seed42，始终运行）
# 17组共用同一份冻结特征缓存（首组提取，其余复用，血缘逐组校验）。
# 失败组保留现场并跳过；完整组幂等跳过。全部17组成功后自动执行汇总器；
# 任一失败则不汇总并返回非零。
#
# 用法：
#   CUDA_DEVICE=<物理GPU编号> bash run_clong_sae_matrix.sh
#   CLONG_MATRIX_DRY_RUN=1 bash run_clong_sae_matrix.sh   # 只打印计划
PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT_DIR="$PROJECT_ROOT/程序/SAE/正式代码"
SCRIPT="$SCRIPT_DIR/clong_sae_discovery.py"
SUMMARIZER="$SCRIPT_DIR/summarize_clong_sae_matrix.py"
RUN_ROOT="$PROJECT_ROOT/结果/SAE/CLong文献重构_20260819"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY_JSON="$RUN_ROOT/clong_sae_matrix_summary_seed42.json"
SUMMARY_LOG="$LOG_ROOT/clong_sae_matrix_summary.log"
FIRST_EXPERIMENT="clong_w512_l2e-4_seed42"
CUDA_DEVICE="${CUDA_DEVICE:-}"
DRY_RUN="${CLONG_MATRIX_DRY_RUN:-0}"

# 17组固定顺序：名称|机制|宽度|lambda|gamma|topk
RUNS=(
    "clong_w512_l2e-4_seed42|relu_l1|512|2e-4|0.1|"
    "clong_w512_l5e-4_seed42|relu_l1|512|5e-4|0.1|"
    "clong_w512_l1e-3_seed42|relu_l1|512|1e-3|0.1|"
    "clong_w1280_l2e-4_seed42|relu_l1|1280|2e-4|0.1|"
    "clong_w1280_l5e-4_seed42|relu_l1|1280|5e-4|0.1|"
    "clong_w1280_l1e-3_seed42|relu_l1|1280|1e-3|0.1|"
    "clong_w2560_l2e-4_seed42|relu_l1|2560|2e-4|0.1|"
    "clong_w2560_l5e-4_seed42|relu_l1|2560|5e-4|0.1|"
    "clong_w2560_l1e-3_seed42|relu_l1|2560|1e-3|0.1|"
    "clong_w5120_l2e-4_seed42|relu_l1|5120|2e-4|0.1|"
    "clong_w5120_l5e-4_seed42|relu_l1|5120|5e-4|0.1|"
    "clong_w5120_l1e-3_seed42|relu_l1|5120|1e-3|0.1|"
    "clong_w10240_l2e-4_seed42|relu_l1|10240|2e-4|0.1|"
    "clong_w10240_l5e-4_seed42|relu_l1|10240|5e-4|0.1|"
    "clong_w10240_l1e-3_seed42|relu_l1|10240|1e-3|0.1|"
    "clong_w10240_l5e-4_gamma0_seed42|relu_l1|10240|5e-4|0.0|"
    "clong_w10240_topk1024_seed42|topk|10240|0|0.1|1024"
)

# 返回0=完整，1=缺失/残缺；$1=实验名。
run_complete() {
    local dir="$RUN_ROOT/$1"
    [[ -f "$dir/config.json" && -f "$dir/metrics.json" \
        && -f "$dir/SAE模型/sae_best.pth" \
        && -f "$dir/特征缓存/cache_config.json" \
        && -f "$dir/feature_summary.csv" \
        && -f "$dir/feature筛选/pruning_summary.json" ]]
}

build_command() {
    # $1..$6 = 名称|机制|宽度|lambda|gamma|topk；输出CMD数组。
    local name="$1" mode="$2" width="$3" lambda="$4" gamma="$5" topk="$6"
    CMD=("$SCRIPT" --experiment "$name" --seed 42 --hidden-dim "$width" \
        --activation-mode "$mode" --lambda-l1 "$lambda" \
        --margin-loss-weight "$gamma" --device cuda)
    if [[ "$mode" == "topk" ]]; then
        CMD+=(--top-k "$topk")
    fi
    if [[ "$name" != "$FIRST_EXPERIMENT" ]]; then
        CMD+=(--feature-cache-from "$RUN_ROOT/$FIRST_EXPERIMENT/特征缓存")
    fi
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 只打印计划，不执行任何训练或汇总。"
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r name mode width lambda gamma topk <<< "$entry"
        if run_complete "$name"; then
            echo "[DRY-RUN] $name: 产物完整，将幂等跳过"
        else
            build_command "$name" "$mode" "$width" "$lambda" "$gamma" "$topk"
            echo "[DRY-RUN] $name: ${CMD[*]}"
        fi
    done
    if [[ -f "$SUMMARY_JSON" ]]; then
        echo "[DRY-RUN] 汇总已存在，17组全部成功时将幂等跳过: $SUMMARY_JSON"
    else
        echo "[DRY-RUN] 17组全部成功后将自动执行: $SUMMARIZER -> $SUMMARY_JSON"
    fi
    exit 0
fi

if [[ -z "$CUDA_DEVICE" ]]; then
    echo "请先用nvidia-smi选择空闲GPU，再设置 CUDA_DEVICE=<物理编号>。" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi不可用，拒绝启动C-long SAE矩阵。" >&2
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
status_all=0
for entry in "${RUNS[@]}"; do
    IFS='|' read -r name mode width lambda gamma topk <<< "$entry"
    RUN_DIR="$RUN_ROOT/$name"
    LOG="$LOG_ROOT/$name.log"
    if run_complete "$name"; then
        echo "[SKIP] $name 已有完整正式产物: $RUN_DIR"
        continue
    fi
    if [[ -e "$RUN_DIR" ]]; then
        echo "[FAILED] $name 存在残缺目录，保留现场并请人工归档后再运行。" >&2
        status_all=1
        continue
    fi
    if [[ "$name" != "$FIRST_EXPERIMENT" ]]; then
        if [[ ! -f "$RUN_ROOT/$FIRST_EXPERIMENT/特征缓存/cache_config.json" ]]; then
            echo "[FAILED] 首组特征缓存缺失，无法复用: $FIRST_EXPERIMENT" >&2
            status_all=1
            continue
        fi
    fi
    build_command "$name" "$mode" "$width" "$lambda" "$gamma" "$topk"
    # 追加写入，保留服务器/终端中断前的训练记录。
    echo "[$(date '+%F %T')] START $name GPU=$CUDA_DEVICE" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 "$PYTHON" -u \
        "${CMD[@]}" 2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%F %T')] DONE $name" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAILED $name status=$status（保留现场）" | tee -a "$LOG"
        status_all=1
    fi
done

if [[ $status_all -ne 0 ]]; then
    echo "[FAILED] 存在失败或残缺组，不执行汇总，保留现场。" >&2
    exit "$status_all"
fi
for entry in "${RUNS[@]}"; do
    IFS='|' read -r name _ <<< "$entry"
    if ! run_complete "$name"; then
        echo "[FAILED] $name 产物不完整，拒绝汇总。" >&2
        exit 1
    fi
done
if [[ -f "$SUMMARY_JSON" ]]; then
    echo "[SKIP] 汇总已存在，幂等跳过: $SUMMARY_JSON"
    exit 0
fi
echo "[$(date '+%F %T')] SUMMARIZE 矩阵汇总与正式配置选择" | tee -a "$SUMMARY_LOG"
PYTHONUNBUFFERED=1 "$PYTHON" -u "$SUMMARIZER" 2>&1 | tee -a "$SUMMARY_LOG"
status=${PIPESTATUS[0]}
if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] SUMMARIZE DONE -> $SUMMARY_JSON" | tee -a "$SUMMARY_LOG"
else
    echo "[$(date '+%F %T')] SUMMARIZE FAILED status=$status" | tee -a "$SUMMARY_LOG"
fi
exit "$status"
