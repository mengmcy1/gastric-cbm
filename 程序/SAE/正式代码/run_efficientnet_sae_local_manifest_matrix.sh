#!/usr/bin/env bash
# 为指定seed串行生成局部SAE冻结ROI清单；单seed失败会保留日志并继续后续seed。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/build_efficientnet_sae_local_manifest.py"
OUTPUT="$PROJECT_ROOT/数据整理记录/SAE_EfficientNet_0804/冻结局部ROI清单"
SEEDS="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:?请先设置CUDA_DEVICE；运行前按nvidia-smi实际占用选择}"

cd "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "$OUTPUT/logs"
failures=()

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}"; do
  csv="$OUTPUT/efficientnet_local_roi_seed${seed}.csv"
  config="$OUTPUT/efficientnet_local_roi_seed${seed}.json"
  log="$OUTPUT/logs/seed${seed}.log"
  if [[ -f "$csv" && -f "$config" ]]; then
    echo "SKIP seed${seed}: 正式CSV和config均已存在"
    continue
  fi
  if [[ -e "$csv" || -e "$config" ]]; then
    echo "FAILED seed${seed}: 发现残缺产物，请人工归档后再运行" | tee -a "$log"
    failures+=("seed${seed}:partial")
    continue
  fi
  echo "[$(date '+%F %T')] START local ROI seed${seed}" | tee "$log"
  if CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$SCRIPT" \
      --seed "$seed" --device 0 2>&1 | tee -a "$log"; then
    echo "[$(date '+%F %T')] DONE local ROI seed${seed}" | tee -a "$log"
  else
    status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] FAILED local ROI seed${seed} status=$status" | tee -a "$log"
    failures+=("seed${seed}:$status")
  fi
done

if ((${#failures[@]})); then
  printf '局部SAE清单失败项: %s\n' "${failures[*]}"
  exit 1
fi
echo "局部SAE清单完成: $OUTPUT"
