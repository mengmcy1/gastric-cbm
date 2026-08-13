#!/usr/bin/env bash
# M5c-A内部test一次性锁定评价；不允许从test调权重、聚合、阈值或ROI。
set -euo pipefail

ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="程序/模型训练/正式代码/evaluate_m5c_locked_internal_test.py"
OUTPUT="$ROOT/结果/M5c概率融合_0804/锁定内部test评估"
ROI_ROOT="$ROOT/结果/M5c概率融合_0804/冻结内部test_ROI清单"
LOGS="$OUTPUT/logs"
CUDA_DEVICE="${CUDA_DEVICE:-}"

if [[ $# -ne 1 ]]; then
    echo "用法: CUDA_DEVICE=<空闲GPU> $0 42,202,503" >&2; exit 2
fi
if [[ -z "$CUDA_DEVICE" ]]; then
    echo "必须先检查GPU并设置CUDA_DEVICE。" >&2; exit 2
fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "未找到nvidia-smi。" >&2; exit 1; }
if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | awk -v target="$CUDA_DEVICE" '$1 == target {found=1} END {exit !found}'; then
    echo "CUDA_DEVICE无效: $CUDA_DEVICE" >&2; exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
mkdir -p "$LOGS"
cd "$ROOT"
IFS=',' read -r -a seeds <<< "$1"
failed=0
for seed in "${seeds[@]}"; do
    case "$seed" in 42|202|503) ;; *) echo "非法seed: $seed" >&2; exit 2;; esac
    roi="$ROI_ROOT/m5c_internal_test_roi_seed${seed}.csv"
    run="$OUTPUT/m5c_locked_internal_test_seed${seed}"
    if [[ ! -f "$roi" ]]; then
        echo "缺少seed${seed} test ROI，先运行ROI矩阵。" >&2; failed=1; continue
    fi
    if [[ -f "$run/config.json" && -f "$run/test_patient_predictions.csv" \
          && -f "$run/test_image_predictions.csv" ]]; then
        echo "[$(date '+%F %T')] SKIP seed${seed}: 完整test结果已存在。"; continue
    fi
    if [[ -e "$run" ]]; then
        echo "[$(date '+%F %T')] INCOMPLETE seed${seed}: 不自动覆盖。" >&2
        failed=1; continue
    fi
    echo "[$(date '+%F %T')] START seed${seed} locked test"
    set +e
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
        "$PYTHON" -u "$ENTRY" --seed "$seed" --roi-manifest "$roi" \
        2>&1 | tee "$LOGS/seed${seed}.log"
    status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${status[0]}" -ne 0 || "${status[1]}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED seed${seed}; 保留现场。" >&2; failed=1
    else
        echo "[$(date '+%F %T')] DONE seed${seed} locked test"
    fi
done

"$PYTHON" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
print("\n=== M5c-A锁定内部test三种子汇总 ===")
for seed in (42, 202, 503):
    path = root / f"m5c_locked_internal_test_seed{seed}" / "config.json"
    if not path.is_file():
        print(f"seed{seed}: 未完成"); continue
    r = json.loads(path.read_text(encoding="utf-8"))["results"]
    rows.append(r["delta_patient_auc"])
    m = r["m5c_patient_at_frozen_sens90_threshold"]
    print(f"seed{seed}: global={r['global_patient_auc']:.4f} "
          f"M5c={r['m5c_patient_auc']:.4f} Δ={r['delta_patient_auc']:+.4f} "
          f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")
if len(rows) == 3:
    mean = sum(rows) / 3
    print(f"平均Δ={mean:+.4f}; 3/3提升={all(x > 0 for x in rows)}")
    if all(x > 0 for x in rows) and mean >= 0.01:
        print("确认结论: 正式确认")
    elif all(x > 0 for x in rows) and mean >= 0.005:
        print("确认结论: 提示性收益")
    else:
        print("确认结论: 未确认")
PY
if [[ "$failed" -ne 0 ]]; then exit 1; fi
echo "M5c锁定内部test矩阵完成: $OUTPUT"
