#!/usr/bin/env bash
# M3正式矩阵启动器：运行冻结M0特征下的非癌假阳性抑制训练。
# 安全边界：M3脚本不提供--evaluate-test，internal test全程锁定。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m3_suppression.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M3定位抑制_0804/正式验证集筛选"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
W_NEG_MAX="${W_NEG_MAX:-1.0}"
NEG_TOPK="${NEG_TOPK:-0}"
W_NEG_RAMP_EPOCHS="${W_NEG_RAMP_EPOCHS:-5}"
MAX_CANCER_DETECTION_LOSS="${MAX_CANCER_DETECTION_LOSS:--1}"
RESOLUTION_GATE_MIN_CANCER="${RESOLUTION_GATE_MIN_CANCER:-0}"
RESOLUTION_GATE_MAX_DETECTION_LOSS="${RESOLUTION_GATE_MAX_DETECTION_LOSS:-1}"


usage() {
    cat <<'EOF'
用法:
  run_m3_matrix.sh <balanced|full|all> <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  # 先跑平衡主实验三组并逐组核验passed_seed_success
  CUDA_DEVICE=<空闲GPU> run_m3_matrix.sh balanced 42,202,503
  # 平衡3/3通过后，再启动全量诊断（脚本会自动门控）
  CUDA_DEVICE=<空闲GPU> run_m3_matrix.sh full 42,202,503
  # 或一次all：脚本内部先跑平衡，再检查3/3门控后才跑全量
  CUDA_DEVICE=<空闲GPU> run_m3_matrix.sh all 42,202,503

M3b示例（只在内部balanced val选择，禁止运行中查看外部集）:
  # A：温和权重、原全网格负损失
  CUDA_DEVICE=<空闲GPU> RUN_SUFFIX=_m3bA_w05_r10 \
    W_NEG_MAX=0.5 NEG_TOPK=0 W_NEG_RAMP_EPOCHS=10 \
    MAX_CANCER_DETECTION_LOSS=1 RESOLUTION_GATE_MIN_CANCER=15 \
    RESOLUTION_GATE_MAX_DETECTION_LOSS=1 \
    run_m3_matrix.sh balanced 42,202,503
  # B：温和权重、仅最高5个非癌响应位置
  CUDA_DEVICE=<空闲GPU> RUN_SUFFIX=_m3bB_top5_w05_r10 \
    W_NEG_MAX=0.5 NEG_TOPK=5 W_NEG_RAMP_EPOCHS=10 \
    MAX_CANCER_DETECTION_LOSS=1 RESOLUTION_GATE_MIN_CANCER=15 \
    RESOLUTION_GATE_MAX_DETECTION_LOSS=1 \
    run_m3_matrix.sh balanced 42,202,503

实时监测（另开终端）:
  tail -F /home/mcy/gastric-cbm/结果/M3定位抑制_0804/正式验证集筛选/logs/m3_balanced_keep_efficientnet_b0_seed42.log
  watch -n 2 nvidia-smi

说明:
  - M3不评估internal test；脚本不会传--evaluate-test。
  - 每组使用同数据角色、同种子的M1 warmup-only产品初始化（入口按--role自动定位）。
  - RUN_SUFFIX会追加到结果目录和日志名；用于受控复跑时禁止留空。
  - M3默认参数为W_NEG_MAX=1、NEG_TOPK=0、ramp=5，严格保留原协议。
  - M3b必须使用非空RUN_SUFFIX，并显式传入癌检出计数及分辨率门槛参数。
  - 全量腿启动前强制检查平衡3种子config的passed_seed_success全为True，否则拒绝。
EOF
}


validate_seed() {
    case "$1" in
        42|202|503) ;;
        *)
            echo "非法seed: $1；只允许42/202/503。" >&2
            exit 2
            ;;
    esac
}


check_balanced_gate() {
    # 全量腿门控：平衡3种子必须全部 passed_seed_success=True，否则拒绝启动。
    for seed in 42 202 503; do
        local cfg="$OUTPUT_ROOT/m3_balanced_keep_efficientnet_b0_seed${seed}${RUN_SUFFIX}/config.json"
        if [[ ! -f "$cfg" ]]; then
            echo "缺少平衡seed${seed}的M3 config: $cfg；不能启动全量腿。" >&2
            return 1
        fi
        local passed
        if ! passed="$("$PYTHON" -c "
import json, sys
try:
    c = json.load(open('$cfg'))
    print(c['m3_selection']['passed_seed_success'])
except (KeyError, json.JSONDecodeError) as error:
    print('False', file=sys.stderr)
    sys.exit(1)
")"; then
            echo "无法读取平衡seed${seed}的门控状态: $cfg" >&2
            return 1
        fi
        if [[ "$passed" != "True" ]]; then
            echo "平衡seed${seed} passed_seed_success=False，3/3未达成，拒绝启动全量腿。" >&2
            return 1
        fi
    done
    echo "平衡3种子passed_seed_success均=True，允许启动全量腿。"
}


run_one() {
    local dataset_role="$1"
    local seed="$2"
    local run_name="m3_${dataset_role}_keep_efficientnet_b0_seed${seed}${RUN_SUFFIX}"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"

    if [[ -e "$run_dir" ]]; then
        if [[ -f "$run_dir/config.json" ]]; then
            echo "[$(date '+%F %T')] SKIP $run_name：已有完整config，视为已完成。"
            return 0
        fi
        echo "[$(date '+%F %T')] KNOWN_FAILED $run_name：已有目录但无config，保留失败现场并继续。" >&2
        return 10
    fi
    echo "[$(date '+%F %T')] START $run_name role=$dataset_role seed=$seed "\
"w_neg_max=$W_NEG_MAX neg_topk=$NEG_TOPK ramp=$W_NEG_RAMP_EPOCHS "\
"max_cancer_loss=$MAX_CANCER_DETECTION_LOSS resolution_min=$RESOLUTION_GATE_MIN_CANCER"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --role "$dataset_role" \
        --seed "$seed" \
        --w-neg-max "$W_NEG_MAX" \
        --neg-topk "$NEG_TOPK" \
        --w-neg-ramp-epochs "$W_NEG_RAMP_EPOCHS" \
        --max-cancer-detection-loss "$MAX_CANCER_DETECTION_LOSS" \
        --resolution-gate-min-cancer "$RESOLUTION_GATE_MIN_CANCER" \
        --resolution-gate-max-detection-loss "$RESOLUTION_GATE_MAX_DETECTION_LOSS" \
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
    if [[ $# -ne 2 ]]; then
        usage
        exit 2
    fi
    local scope="$1"
    local seed_spec="$2"
    local datasets=()
    case "$scope" in
        balanced) datasets=(balanced) ;;
        full)
            if ! check_balanced_gate; then
                exit 1
            fi
            datasets=(full)
            ;;
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
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "必须先用nvidia-smi确认空闲GPU，再显式设置CUDA_DEVICE。" >&2
        exit 2
    fi
    if [[ -z "$RUN_SUFFIX" ]] && {
        [[ "$W_NEG_MAX" != "1.0" ]] || [[ "$NEG_TOPK" != "0" ]] \
            || [[ "$W_NEG_RAMP_EPOCHS" != "5" ]] \
            || [[ "$MAX_CANCER_DETECTION_LOSS" != "-1" ]] \
            || [[ "$RESOLUTION_GATE_MIN_CANCER" != "0" ]];
    }; then
        echo "非默认M3参数必须设置非空RUN_SUFFIX，防止与原M3结果混淆。" >&2
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

    local completed_runs=()
    local failed_runs=()
    for dataset_role in "${datasets[@]}"; do
        # all模式下，平衡跑完、全量启动前再次门控。
        if [[ "$dataset_role" == "full" ]]; then
            if ! check_balanced_gate; then
                failed_runs+=("full_gate")
                echo "平衡腿未达到3/3，按协议跳过全部full任务。" >&2
                break
            fi
        fi
        for seed in "${seeds[@]}"; do
            if run_one "$dataset_role" "$seed"; then
                completed_runs+=("${dataset_role}:seed${seed}")
            else
                local status=$?
                failed_runs+=("${dataset_role}:seed${seed}(exit=${status})")
            fi
        done
    done
    echo "矩阵执行结束。完成/已存在: ${completed_runs[*]:-无}"
    if [[ ${#failed_runs[@]} -gt 0 ]]; then
        echo "失败/跳过: ${failed_runs[*]}" >&2
        echo "其余任务已继续执行；矩阵最终返回非零，提示仍有失败组。" >&2
        return 1
    fi
    echo "所选M3矩阵全部完成。日志目录: $LOG_ROOT"
}


main "$@"
