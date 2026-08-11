#!/usr/bin/env bash
# M3c-B正式矩阵：冻结M1的分类、定位和特征提取，只训练ROI区域门控头。
# 安全边界：入口只建立train/val Dataset，不读取internal test或external图像。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m3c_region_gate.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M3c区域门控_0804/正式验证集筛选"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"


usage() {
    cat <<'EOF'
用法:
  run_m3c_region_gate_matrix.sh <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  CUDA_DEVICE=<空闲GPU> bash \
    程序/模型训练/正式代码/run_m3c_region_gate_matrix.sh 42,202,503

实时监测（另开终端）:
  tail -F /home/mcy/gastric-cbm/结果/M3c区域门控_0804/正式验证集筛选/logs/*.log
  watch -n 2 nvidia-smi

说明:
  - 每个seed自动加载对应的M1 warmup-only产品。
  - 只使用balanced train/val训练和选择；不读test/external。
  - 某个seed失败时保留失败config并继续后续seed。
  - 矩阵最终以3/3 passed_seed_success=True作为正式成功。
EOF
}


validate_seed() {
    case "$1" in
        42|202|503) ;;
        *) echo "非法seed: $1；只允许42/202/503。" >&2; exit 2 ;;
    esac
}


run_one() {
    local seed="$1"
    local run_name="m3c_balanced_keep_efficientnet_b0_seed${seed}"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"

    if [[ -f "$run_dir/config.json" ]]; then
        local status
        status="$($PYTHON -c "import json; print(json.load(open('$run_dir/config.json')).get('status', 'unknown'))")"
        echo "[$(date '+%F %T')] SKIP $run_name：已有config（status=$status）。"
        [[ "$status" == "success" ]]
        return
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE $run_name：目录存在但无config，保留现场并继续。" >&2
        return 10
    fi

    echo "[$(date '+%F %T')] START $run_name seed=$seed"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --seed "$seed" \
        --output-root "$OUTPUT_ROOT" \
        --run-name "$run_name" 2>&1 | tee "$log_file"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] LOG_FAILED $run_name tee退出码=${pipeline_status[1]}" >&2
        return "${pipeline_status[1]}"
    fi
    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name python退出码=${pipeline_status[0]}；继续后续seed。" >&2
        return "${pipeline_status[0]}"
    fi
    echo "[$(date '+%F %T')] DONE $run_name"
}


main() {
    if [[ $# -ne 1 ]]; then
        usage
        exit 2
    fi
    IFS=',' read -r -a seeds <<< "$1"
    for seed in "${seeds[@]}"; do
        validate_seed "$seed"
    done
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "未找到nvidia-smi，拒绝启动正式GPU矩阵。" >&2
        exit 1
    }
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "请先用nvidia-smi确认当前空闲GPU，再显式设置CUDA_DEVICE。" >&2
        exit 2
    fi
    if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
        | awk -v target="$CUDA_DEVICE" '$1 == target { found=1 } END { exit !found }'; then
        echo "CUDA_DEVICE不是当前主机上的有效GPU编号: $CUDA_DEVICE" >&2
        exit 2
    fi

    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    mkdir -p "$LOG_ROOT"
    cd "$PROJECT_ROOT"

    local completed=()
    local failed=()
    for seed in "${seeds[@]}"; do
        if run_one "$seed"; then
            completed+=("seed${seed}")
        else
            local status=$?
            failed+=("seed${seed}(exit=${status})")
        fi
    done
    echo "M3c-B矩阵结束。成功/已存在: ${completed[*]:-无}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "失败组: ${failed[*]}" >&2
        echo "后续seed已继续执行；矩阵返回非零以提示未达到3/3。" >&2
        return 1
    fi
    echo "3个seed均完成。日志目录: $LOG_ROOT"
}


main "$@"
