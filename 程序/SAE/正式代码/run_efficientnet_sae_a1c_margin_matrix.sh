#!/usr/bin/env bash
# seed42 A1c：固定8倍Top-K字典，比较三档冻结分类margin保真权重。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/efficientnet_sae_discovery.py"
A0="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-A0_S0工程验证"
OUTPUT="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-A1c_Margin保真"
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
gammas=(0.1 0.5 1.0)
gamma_names=(01 05 10)
failures=()

for branch in "${branches[@]}"; do
  cache="$A0/a0_${branch}_efficientnet_b0_seed42_s0_h512_l1_0005/特征缓存"
  if [[ ! -f "$cache/train_gap_features.npy" || ! -f "$cache/val_gap_features.npy" ]]; then
    echo "缺少$branch A0冻结特征缓存: $cache" >&2
    exit 1
  fi
  for index in "${!gammas[@]}"; do
    gamma="${gammas[$index]}"
    gamma_name="${gamma_names[$index]}"
    experiment="a1c_${branch}_efficientnet_b0_seed42_h10240_topk1024_margin${gamma_name}"
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
    echo "[$(date '+%F %T')] START $experiment gamma=$gamma" | tee "$log"
    if CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$SCRIPT" \
        --branch "$branch" --seed 42 --hidden-dim 10240 \
        --activation-mode topk --top-k 1024 --lambda-l1 0 \
        --margin-loss-weight "$gamma" \
        --epochs 1000 --patience 50 --warmup-fraction 0.05 \
        --image-batch-size 32 --sae-batch-size 32 --num-workers 4 \
        --feature-cache-from "$cache" --output-root "$OUTPUT" \
        --experiment "$experiment" 2>&1 | tee -a "$log"; then
      echo "[$(date '+%F %T')] DONE $experiment" | tee -a "$log"
    else
      status=${PIPESTATUS[0]}
      echo "[$(date '+%F %T')] FAILED $experiment status=$status" | tee -a "$log"
      failures+=("$experiment:$status")
    fi
  done
done

if ((${#failures[@]})); then
  printf 'SAE-A1c失败项: %s\n' "${failures[*]}"
  exit 1
fi
echo "SAE-A1c矩阵完成: $OUTPUT"
