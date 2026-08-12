#!/usr/bin/env bash
# M5冻结ROI清单矩阵：用冻结M1+M3c-B为三种子train/val确定性推理（两类都用预测框ROI）。
# 安全边界：只为train/val生成；internal test/external不读图、不生成框、不算q_region。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
INFERENCE_BATCH_SIZE=32
ENTRY="数据整理脚本/build_m5_roi_manifest.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M5预测ROI融合_0804/冻结ROI清单"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"


usage() {
    cat <<'EOF'
用法:
  run_m5_roi_manifest_matrix.sh <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  CUDA_DEVICE=<空闲GPU> bash 程序/模型训练/正式代码/run_m5_roi_manifest_matrix.sh 42,202,503

说明:
  - 每个seed自动定位对应的M1 warmup-only产品与M3c-B门控；门控阈值同源读取。
  - 幂等跳过只接受debug=false的正式config且manifest文件存在。
  - 任何seed失败，矩阵最终返回非零；不静默吞错。
  - M5训练前的必做步骤；三个seed的正式ROI清单都生成后才可启动M5训练。
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
    local config_file="$OUTPUT_ROOT/m5_roi_config_seed${seed}.json"
    local manifest_file="$OUTPUT_ROOT/m5_roi_manifest_seed${seed}.csv"
    local log_file="$LOG_ROOT/m5_roi_manifest_seed${seed}.log"

    if [[ -f "$config_file" ]]; then
        # 幂等验证：config必须为正式（debug=false）且manifest存在。
        local debug_flag
        debug_flag="$("$PYTHON" -c "
import json
print(json.load(open('$config_file')).get('debug', True))
")"
        if [[ "$debug_flag" == "False" ]] && [[ -f "$manifest_file" ]]; then
            echo "[$(date '+%F %T')] SKIP seed${seed}：正式ROI清单已存在（debug=false）。"
            return 0
        fi
        echo "[$(date '+%F %T')] INCOMPLETE seed${seed}：config存在但为debug或缺manifest。" >&2
        echo "  脚本不自动覆盖；请人工归档/删除后重建正式清单。" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] START seed${seed} ROI清单"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" --seed "$seed" \
        --batch-size "$INFERENCE_BATCH_SIZE" 2>&1 | tee "$log_file"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] LOG_FAILED seed${seed} tee退出码=${pipeline_status[1]}" >&2
        return "${pipeline_status[1]}"
    fi
    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED seed${seed} python退出码=${pipeline_status[0]}" >&2
        return "${pipeline_status[0]}"
    fi
    echo "[$(date '+%F %T')] DONE seed${seed} ROI清单"
}


main() {
    if [[ $# -ne 1 ]]; then
        usage
        exit 2
    fi
    IFS=',' read -r -a seeds <<< "$1"
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
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "必须先用nvidia-smi确认空闲GPU，再显式设置CUDA_DEVICE。" >&2
        exit 2
    fi
    if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
        | awk -v target="$CUDA_DEVICE" '$1 == target { found=1 } END { exit !found }'; then
        echo "CUDA_DEVICE不是当前主机上的有效物理GPU编号: $CUDA_DEVICE" >&2
        exit 2
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    mkdir -p "$LOG_ROOT"
    cd "$PROJECT_ROOT"

    local failed=0
    for seed in "${seeds[@]}"; do
        run_one "$seed" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "存在失败seed，M5 ROI清单矩阵返回非零。" >&2
        exit 1
    fi
    echo "M5 ROI清单矩阵运行完成。日志目录: $LOG_ROOT"
}


main "$@"
