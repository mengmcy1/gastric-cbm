#!/usr/bin/env bash
# 运行seed42 S0工程验证；支持global、local或all，正式输出存在时幂等跳过。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/efficientnet_sae_discovery.py"
OUTPUT="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-A0_S0工程验证"
MODE="${1:-all}"
CUDA_DEVICE="${CUDA_DEVICE:?请先设置CUDA_DEVICE；运行前按nvidia-smi实际占用选择}"

if [[ "$MODE" != "global" && "$MODE" != "local" && "$MODE" != "all" ]]; then
  echo "用法: CUDA_DEVICE=N $0 <global|local|all>" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "$OUTPUT/logs"
branches=("$MODE")
[[ "$MODE" == "all" ]] && branches=(global local)
failures=()

for branch in "${branches[@]}"; do
  experiment="a0_${branch}_efficientnet_b0_seed42_s0_h512_l1_0005"
  run_dir="$OUTPUT/$experiment"
  log="$OUTPUT/logs/${experiment}.log"
  if [[ -f "$run_dir/config.json" && -f "$run_dir/metrics.json" && -f "$run_dir/SAE模型/sae_best.pth" ]]; then
    echo "SKIP $experiment: 正式产物完整"
    continue
  fi
  if [[ -e "$run_dir" ]]; then
    echo "FAILED $experiment: 发现残缺目录，请人工归档后再运行" | tee -a "$log"
    failures+=("$experiment:partial")
    continue
  fi
  echo "[$(date '+%F %T')] START $experiment" | tee "$log"
  if CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$SCRIPT" \
      --branch "$branch" --seed 42 --hidden-dim 512 --lambda-l1 5e-4 \
      --epochs 1000 --patience 50 --warmup-fraction 0.05 \
      --image-batch-size 32 --sae-batch-size 32 --num-workers 4 \
      --output-root "$OUTPUT" --experiment "$experiment" 2>&1 | tee -a "$log"; then
    echo "[$(date '+%F %T')] DONE $experiment" | tee -a "$log"
  else
    status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] FAILED $experiment status=$status" | tee -a "$log"
    failures+=("$experiment:$status")
  fi
done

if ((${#failures[@]})); then
  printf 'SAE-A0失败项: %s\n' "${failures[*]}"
  exit 1
fi
echo "SAE-A0完成: $OUTPUT"
