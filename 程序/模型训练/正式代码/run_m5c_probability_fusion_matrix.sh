#!/usr/bin/env bash
# M5c正式矩阵：冻结M5b-B全局/局部头，患者概率固定均值为主，train-LR/rank均值为辅助。
# internal test/external不读取；三种方法预先固定，禁止按val择优后冒充单一正式产品。

set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/efficientnet_m5c_probability_fusion.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/M5c概率融合_0804/正式验证集筛选"
ROI_ROOT="$PROJECT_ROOT/结果/M5预测ROI融合_0804/冻结ROI清单"
M5B_ROOT="$PROJECT_ROOT/结果/M5b残差融合_0804/正式验证集筛选"
LOG_ROOT="$OUTPUT_ROOT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"


usage() {
    cat <<'EOF'
用法:
  run_m5c_probability_fusion_matrix.sh <42|202|503|逗号分隔列表>

示例:
  nvidia-smi
  CUDA_DEVICE=<空闲GPU> bash 程序/模型训练/正式代码/run_m5c_probability_fusion_matrix.sh 42,202,503

说明:
  - 主检验A=fixed mean；B=train患者LR、C=train ECDF rank mean仅作辅助。
  - 复用同seed M5b-B(gate_local)冻结局部头，不训练CNN，不使用门控或几何变量。
  - internal test/external保持锁定。
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
    local run_name="m5c_balanced_keep_efficientnet_b0_seed${seed}"
    local run_dir="$OUTPUT_ROOT/$run_name"
    local log_file="$LOG_ROOT/${run_name}.log"
    if [[ -f "$run_dir/config.json" ]]; then
        local complete
        complete="$("$PYTHON" -c "
import json, os
d = '$run_dir'
c = json.load(open(os.path.join(d, 'config.json')))
methods = c.get('methods', {})
files = ['train_patient_predictions.csv', 'val_patient_predictions.csv', 'source_entry.py']
print('1' if c.get('primary_method') == 'A_fixed_mean_primary' and len(methods) == 3
      and all(os.path.isfile(os.path.join(d, f)) for f in files) else '0')
")"
        if [[ "$complete" == "1" ]]; then
            echo "[$(date '+%F %T')] SKIP $run_name：完整产物已存在。"
            return 0
        fi
        echo "[$(date '+%F %T')] INCOMPLETE $run_name：不自动覆盖，请人工归档。" >&2
        return 1
    fi
    if [[ -e "$run_dir" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE $run_name：目录存在但缺config。" >&2
        return 1
    fi
    local roi="$ROI_ROOT/m5_roi_manifest_seed${seed}.csv"
    local m5b="$M5B_ROOT/m5b_gate_local_balanced_keep_efficientnet_b0_seed${seed}"
    if [[ ! -f "$roi" || ! -f "$m5b/config.json" || ! -f "$m5b/m5b_best.pth" ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name：缺ROI或正式M5b-B冻结产物。" >&2
        return 1
    fi
    echo "[$(date '+%F %T')] START $run_name"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" \
        --seed "$seed" --roi-manifest "$roi" --m5b-run "$m5b" \
        --output-root "$OUTPUT_ROOT" --run-name "$run_name" 2>&1 | tee "$log_file"
    local status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${status[1]}" -ne 0 ]]; then return "${status[1]}"; fi
    if [[ "${status[0]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED $run_name python退出码=${status[0]}；保留现场。" >&2
        return "${status[0]}"
    fi
    echo "[$(date '+%F %T')] DONE $run_name"
}


report_summary() {
    "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
methods = [
    "A_fixed_mean_primary",
    "B_train_lr_auxiliary",
    "C_train_ecdf_rank_mean_auxiliary",
]
print("\n=== M5c 三种子汇总 ===")
records = {name: [] for name in methods}
for seed in (42, 202, 503):
    path = root / f"m5c_balanced_keep_efficientnet_b0_seed{seed}" / "config.json"
    if not path.is_file():
        print(f"seed{seed}: 未完成")
        continue
    config = json.loads(path.read_text(encoding="utf-8"))
    print(f"seed{seed}: global={config['global_patient_auc']:.4f}")
    for name in methods:
        result = config["methods"][name]
        records[name].append(result["delta_vs_global"])
        print(f"  {name}: AUC={result['patient_auc']:.4f} Δ={result['delta_vs_global']:+.4f}")
for name in methods:
    values = records[name]
    if len(values) == 3:
        mean = sum(values) / 3
        all_positive = all(value > 0 for value in values)
        print(f"{name}: 平均Δ={mean:+.4f}; 3/3提升={all_positive}")
        if name == "A_fixed_mean_primary":
            if all_positive and mean >= 0.01:
                print("主检验结论: 验证成功（3/3提升且平均Δ>=0.01）")
            elif all_positive and mean >= 0.005:
                print("主检验结论: 提示性收益（3/3提升且平均Δ在[0.005,0.01)）")
            else:
                print("主检验结论: 失败")
PY
}


main() {
    if [[ $# -ne 1 ]]; then usage; exit 2; fi
    IFS=',' read -r -a seeds <<< "$1"
    for seed in "${seeds[@]}"; do validate_seed "$seed"; done
    command -v nvidia-smi >/dev/null 2>&1 || { echo "未找到nvidia-smi。" >&2; exit 1; }
    if [[ -z "$CUDA_DEVICE" ]]; then
        echo "请先检查GPU，再显式设置CUDA_DEVICE。" >&2
        exit 2
    fi
    if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
        | awk -v target="$CUDA_DEVICE" '$1 == target { found=1 } END { exit !found }'; then
        echo "CUDA_DEVICE不是有效GPU编号: $CUDA_DEVICE" >&2
        exit 2
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    mkdir -p "$LOG_ROOT"
    cd "$PROJECT_ROOT"
    local failed=0
    for seed in "${seeds[@]}"; do run_one "$seed" || failed=1; done
    report_summary
    if [[ "$failed" -ne 0 ]]; then exit 1; fi
    echo "M5c概率融合矩阵完成。日志目录: $LOG_ROOT"
}


main "$@"
