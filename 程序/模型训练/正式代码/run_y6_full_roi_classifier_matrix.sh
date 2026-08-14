#!/usr/bin/env bash
set -uo pipefail

# Y6探索性Full三seed串行入口。调用前必须由用户显式选择空闲物理GPU。
PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/模型训练/正式代码/efficientnet_y6_full_roi_classifier.py"
OUTPUT="$PROJECT_ROOT/结果/YOLO26定位_0804/Y6预测ROI局部分类_Full_20260814"
LOG_DIR="$OUTPUT/logs"
SEEDS_CSV="${1:-42,202,503}"
CUDA_DEVICE="${CUDA_DEVICE:?请先用nvidia-smi检查后设置CUDA_DEVICE物理卡号}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
mkdir -p "$LOG_DIR"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
failures=()
for seed in "${SEEDS[@]}"; do
  run="y6_full_efficientnet_b0_seed${seed}"
  config="$OUTPUT/$run/config.json"
  if [[ -f "$config" ]]; then
    echo "SKIP $run：正式config已存在"
    continue
  fi
  if [[ -e "$OUTPUT/$run" ]]; then
    echo "ERROR $run：存在残缺目录，请人工归档后再运行" >&2
    failures+=("$run:partial")
    continue
  fi
  log="$LOG_DIR/${run}.log"
  echo "[$(date '+%F %T')] START $run" | tee -a "$log"
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$SCRIPT" \
    --seed "$seed" --device 0 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  set -e
  if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] DONE $run" | tee -a "$log"
  else
    echo "[$(date '+%F %T')] FAILED $run status=$status" | tee -a "$log"
    failures+=("$run:$status")
  fi
done

"$PYTHON" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for seed in (42,202,503):
    path=root/f"y6_full_efficientnet_b0_seed{seed}"/"config.json"
    if not path.is_file():
        continue
    c=json.loads(path.read_text(encoding="utf-8")); s=c["selection"]
    rows.append((seed,s["global_patient_auc"],s["candidate_patient_auc"],
                 s["raw_delta_patient_auc"],s["fell_back_to_global"]))
print("\n=== Y6 Full三种子阶段汇总 ===")
for seed,g,c,d,f in rows:
    print(f"seed{seed}: global={g:.4f} candidate={c:.4f} Δ={d:+.4f} 回退={f}")
if len(rows)==3:
    mean=sum(row[3] for row in rows)/3
    passed=all(row[3]>0 for row in rows) and mean>=0.01
    print(f"平均raw Δ={mean:+.4f}; 3/3提升={all(row[3]>0 for row in rows)}")
    print("Y6结论:", "验证成功" if passed else "未达到冻结成功判据")
PY

if (( ${#failures[@]} )); then
  printf 'Y6失败项: %s\n' "${failures[*]}" >&2
  exit 1
fi
echo "Y6 Full矩阵完成。日志目录: $LOG_DIR"
