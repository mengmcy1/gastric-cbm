#!/usr/bin/env bash
# M5b正式矩阵：残差锚定融合，M5b-A(硬门控) / M5b-B(门控×局部置信) × 三种子。
# 复用M5冻结ROI清单（batch32，已验收）；产品保护=argmax(全局, 候选融合)，Δ≥0。
# 安全边界：internal test/external不读取；患者聚合用top2_mean；test_evaluated恒False。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m5b_fusion.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M5b残差融合_0804/正式验证集筛选"
ROI_ROOT="$PROJECT_ROOT/结果/M5预测ROI融合_0804/冻结ROI清单"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"


usage() {
    cat <<'EOF'
用法:
  run_m5b_fusion_matrix.sh <gate|gate_local> <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  CUDA_DEVICE=<空闲GPU> bash 程序/模型训练/正式代码/run_m5b_fusion_matrix.sh gate 42,202,503
  CUDA_DEVICE=<空闲GPU> bash 程序/模型训练/正式代码/run_m5b_fusion_matrix.sh gate_local 42,202,503

前置:
  先运行 run_m5_roi_manifest_matrix.sh 42,202,503（batch32）生成正式冻结ROI清单。

说明:
  - alpha-mode: gate=M5b-A(α=gate_pass)；gate_local=M5b-B(α=gate_pass×p_local，主方案)。
  - 幂等跳过只接受config含m5b_selection且产品/候选/预测/历史齐全的完整产物。
  - 产品保护判据：最终产品=argmax(全局基线, 候选融合)，Δ≥0；回退全局记fell_back。
  - 分级结论（预注册）：正式成功/强成功/提示性收益/失败；矩阵结束只报告，不补救。
EOF
}


validate_seed() {
    case "$1" in
        42|202|503) ;;
        *) echo "非法seed: $1；只允许42/202/503。" >&2; exit 2 ;;
    esac
}


validate_alpha_mode() {
    case "$1" in
        gate|gate_local) ;;
        *) echo "非法alpha-mode: $1；只允许gate/gate_local。" >&2; exit 2 ;;
    esac
}


run_one() {
    local seed="$1"
    local run_name="m5b_${ALPHA_MODE}_balanced_keep_efficientnet_b0_seed${seed}"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"

    if [[ -f "$run_dir/config.json" ]]; then
        local complete
        complete="$("$PYTHON" -c "
import json, os
run_dir = '$run_dir'
c = json.load(open(os.path.join(run_dir, 'config.json')))
sel = c.get('m5b_selection', {})
required = ['delta_patient_auc', 'product_patient_auc', 'fell_back_to_global']
ok_files = all(os.path.exists(os.path.join(run_dir, f)) for f in
               ['m5b_best.pth', 'm5b_best_fusion_candidate.pth',
                'val_image_predictions.csv', 'val_patient_predictions.csv',
                'val_patient_candidate_predictions.csv',
                'training_history.csv', 'stage_l_history.csv'])
print('1' if bool(sel) and all(k in sel for k in required) and ok_files else '0')
")"
        if [[ "$complete" == "1" ]]; then
            echo "[$(date '+%F %T')] SKIP $run_name：完整产物已存在。"
            return 0
        fi
        echo "[$(date '+%F %T')] INCOMPLETE $run_name：config存在但缺关键字段或产物文件。" >&2
        echo "  脚本不自动覆盖；请人工归档/删除 $run_dir 后重跑。" >&2
        return 1
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE $run_name：输出目录存在但缺config。" >&2
        echo "  脚本不自动覆盖；请人工归档/删除 $run_dir 后重跑。" >&2
        return 1
    fi
    if [[ ! -f "$ROI_ROOT/m5_roi_manifest_seed${seed}.csv" ]] \
        || [[ ! -f "$ROI_ROOT/m5_roi_config_seed${seed}.json" ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name：缺少正式ROI清单 seed${seed}，先跑ROI清单矩阵。" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] START $run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --seed "$seed" \
        --alpha-mode "$ALPHA_MODE" \
        --roi-manifest "$ROI_ROOT/m5_roi_manifest_seed${seed}.csv" \
        --output-root "$OUTPUT_ROOT" \
        --run-name "$run_name" 2>&1 | tee "$log_file"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] LOG_FAILED $run_name tee退出码=${pipeline_status[1]}" >&2
        return "${pipeline_status[1]}"
    fi
    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name python退出码=${pipeline_status[0]}；保留现场。" >&2
        return "${pipeline_status[0]}"
    fi
    echo "[$(date '+%F %T')] DONE $run_name"
}


report_summary() {
    echo ""
    echo "=== M5b-${ALPHA_MODE} 三种子汇总 ==="
    local all_no_fallback=1
    local all_strong=1
    local sum=0
    local count=0
    for seed in 42 202 503; do
        local cfg="$OUTPUT_ROOT/m5b_${ALPHA_MODE}_balanced_keep_efficientnet_b0_seed${seed}/config.json"
        if [[ ! -f "$cfg" ]]; then
            echo "seed${seed}: 无config（未完成或失败）"
            all_no_fallback=0
            all_strong=0
            continue
        fi
        local delta fellback raw
        read -r delta fellback raw <<< "$("$PYTHON" -c "
import json
c = json.load(open('$cfg'))
sel = c.get('m5b_selection', {})
print(sel.get('delta_patient_auc'), sel.get('fell_back_to_global'), sel.get('raw_delta_patient_auc'))
")"
        if [[ -z "$delta" || -z "$fellback" ]]; then
            echo "seed${seed}: config不完整（缺少Δ/回退标记）"
            all_no_fallback=0
            all_strong=0
            continue
        fi
        echo "seed${seed}: 产品Δ=$delta (raw融合Δ=$raw) 回退=$fellback"
        count=$((count + 1))
        sum="$("$PYTHON" -c "print($sum + $delta)")"
        delta_strong="$("$PYTHON" -c "print(1 if $delta >= 0.02 else 0)")"
        if [[ "$fellback" == "True" ]]; then
            all_no_fallback=0
        fi
        if [[ "$delta_strong" == "0" ]]; then
            all_strong=0
        fi
    done
    if [[ "$count" -gt 0 ]]; then
        local mean
        mean="$("$PYTHON" -c "print($sum / $count)")"
        echo "平均Δ患者AUC=$mean"
        if [[ "$all_strong" == "1" ]]; then
            echo "结论: 强成功（3/3 seed各自 Δ>=0.02）"
        elif [[ "$all_no_fallback" == "1" ]] \
            && "$PYTHON" -c "exit(0 if $mean >= 0.02 else 1)"; then
            echo "结论: 正式成功（3/3均由真实融合epoch胜出，平均>=0.02）"
        elif [[ "$all_no_fallback" == "1" ]] \
            && "$PYTHON" -c "exit(0 if 0.01 <= $mean < 0.02 else 1)"; then
            echo "结论: 提示性收益（3/3均由真实融合epoch胜出，平均Δ∈[0.01,0.02)）"
        else
            echo "结论: 失败（存在seed回退全局，或平均Δ<0.01）"
        fi
    fi
}


main() {
    if [[ $# -ne 2 ]]; then
        usage
        exit 2
    fi
    ALPHA_MODE="$1"
    validate_alpha_mode "$ALPHA_MODE"
    IFS=',' read -r -a seeds <<< "$2"
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
    report_summary
    if [[ "$failed" -ne 0 ]]; then
        echo "存在失败seed，M5b训练矩阵返回非零。" >&2
        exit 1
    fi
    echo "M5b训练矩阵运行完成。日志目录: $LOG_ROOT"
}


main "$@"
