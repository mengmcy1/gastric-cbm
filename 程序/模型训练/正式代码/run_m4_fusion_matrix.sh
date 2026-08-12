#!/usr/bin/env bash
# M4正式矩阵：冻结M1全局/定位基座，训练局部Encoder副本+局部头+融合头。
# 安全边界：internal test/external不读取；患者聚合用top2_mean；test_evaluated恒False。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m4_fusion.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M4真值ROI融合_0804/正式验证集筛选"
ROI_ROOT="$PROJECT_ROOT/结果/M4真值ROI融合_0804/冻结ROI清单"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"


usage() {
    cat <<'EOF'
用法:
  run_m4_fusion_matrix.sh <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  CUDA_DEVICE=<空闲GPU> bash 程序/模型训练/正式代码/run_m4_fusion_matrix.sh 42,202,503

前置:
  先运行 run_m4_roi_manifest_matrix.sh 42,202,503 生成三个seed的正式冻结ROI清单。

说明:
  - 幂等跳过只接受config含m4_selection的完整产物。
  - 任何seed失败，矩阵最终返回非零；不静默吞错。
  - 正式成功按预注册：3/3 seed的Δ患者AUC均>0且平均>=0.02；矩阵结束只报告，不补救。
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
    local run_name="m4_balanced_keep_efficientnet_b0_seed${seed}"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"

    if [[ -f "$run_dir/config.json" ]]; then
        # 幂等验证：config须含非空m4_selection及关键字段，且产品文件齐全。
        local complete
        complete="$("$PYTHON" -c "
import json, os
run_dir = '$run_dir'
c = json.load(open(os.path.join(run_dir, 'config.json')))
sel = c.get('m4_selection', {})
required = ['delta_patient_auc', 'patient_fusion_auc', 'metadata_audit']
ok_files = all(os.path.exists(os.path.join(run_dir, f)) for f in
               ['m4_best.pth', 'val_image_predictions.csv', 'val_patient_predictions.csv'])
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
    if [[ ! -f "$ROI_ROOT/m4_roi_manifest_seed${seed}.csv" ]] \
        || [[ ! -f "$ROI_ROOT/m4_roi_config_seed${seed}.json" ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name：缺少正式ROI清单 seed${seed}，先跑ROI清单矩阵。" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] START $run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --seed "$seed" \
        --roi-manifest "$ROI_ROOT/m4_roi_manifest_seed${seed}.csv" \
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
    echo "=== M4三种子汇总 ==="
    local all_positive=1
    local sum=0
    local count=0
    for seed in 42 202 503; do
        local cfg="$OUTPUT_ROOT/m4_balanced_keep_efficientnet_b0_seed${seed}/config.json"
        if [[ ! -f "$cfg" ]]; then
            echo "seed${seed}: 无config（未完成或失败）"
            all_positive=0
            continue
        fi
        local delta
        delta="$("$PYTHON" -c "
import json
c = json.load(open('$cfg'))
value = c.get('m4_selection', {}).get('delta_patient_auc')
print('' if value is None else value)
")"
        if [[ -z "$delta" ]]; then
            echo "seed${seed}: config不完整（缺少Δ患者AUC）"
            all_positive=0
            continue
        fi
        echo "seed${seed}: Δ患者AUC=$delta"
        count=$((count + 1))
        sum="$("$PYTHON" -c "print($sum + $delta)")"
        delta_positive="$("$PYTHON" -c "print(1 if $delta > 0 else 0)")"
        if [[ "$delta_positive" == "0" ]]; then
            all_positive=0
        fi
    done
    if [[ "$count" -gt 0 ]]; then
        local mean
        mean="$("$PYTHON" -c "print($sum / $count)")"
        echo "平均Δ患者AUC=$mean"
        if [[ "$all_positive" == "1" ]] && "$PYTHON" -c "exit(0 if $mean >= 0.02 else 1)"; then
            echo "结论: 正式成功（3/3均>0且平均>=0.02）"
        else
            echo "结论: 未达预注册成功判据（3/3均>0且平均>=0.02）"
        fi
    fi
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
    report_summary
    if [[ "$failed" -ne 0 ]]; then
        echo "存在失败seed，M4训练矩阵返回非零。" >&2
        exit 1
    fi
    echo "M4训练矩阵运行完成。日志目录: $LOG_ROOT"
}


main "$@"
