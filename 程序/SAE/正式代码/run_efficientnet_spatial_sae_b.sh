#!/usr/bin/env bash
# SAE-B seed42逐级容量验证：局部通过后才验证全局，K=128失败才升到256。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/efficientnet_spatial_sae_discovery.py"
OUTPUT="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-B空间Token"
CUDA_DEVICE="${CUDA_DEVICE:?请先根据nvidia-smi设置CUDA_DEVICE}"

cd "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "$OUTPUT/logs"

run_one() {
  local branch="$1"
  local top_k="$2"
  local experiment="sae_b_${branch}_efficientnet_b0_seed42_h10240_topk${top_k}_margin01"
  local run_dir="$OUTPUT/$experiment"
  local log="$OUTPUT/logs/${experiment}.log"
  local cache_args=()
  if [[ "$top_k" == "256" ]]; then
    local source="$OUTPUT/sae_b_${branch}_efficientnet_b0_seed42_h10240_topk128_margin01/空间特征缓存"
    cache_args=(--feature-cache-from "$source")
  fi
  if [[ -f "$run_dir/config.json" && -f "$run_dir/metrics.json" && -f "$run_dir/SAE模型/spatial_sae_best.pth" ]]; then
    echo "SKIP $experiment: 正式产物完整"
    return 0
  fi
  if [[ -e "$run_dir" ]]; then
    echo "FAILED $experiment: 发现残缺目录，请人工归档后再运行" >&2
    return 2
  fi
  echo "[$(date '+%F %T')] START $experiment" | tee "$log"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$SCRIPT" \
    --branch "$branch" --seed 42 --hidden-dim 10240 --top-k "$top_k" \
    --margin-loss-weight 0.1 --epochs 1000 --patience 50 --warmup-fraction 0.05 \
    --image-batch-size 32 --sae-image-batch-size 16 --num-workers 4 \
    --output-root "$OUTPUT" --experiment "$experiment" "${cache_args[@]}" >> "$log" 2>&1
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED $experiment status=$status" | tee -a "$log"
    return "$status"
  fi
  echo "[$(date '+%F %T')] DONE $experiment" | tee -a "$log"
}

passed() {
  local branch="$1"
  local top_k="$2"
  local metrics="$OUTPUT/sae_b_${branch}_efficientnet_b0_seed42_h10240_topk${top_k}_margin01/metrics.json"
  "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["quality_gate"]["passed"] else 1)' "$metrics"
}

selected_k=""
for top_k in 128 256; do
  run_one local "$top_k" || exit $?
  if passed local "$top_k"; then
    selected_k="$top_k"
    echo "局部空间SAE通过: K=$top_k"
    break
  fi
done
if [[ -z "$selected_k" ]]; then
  echo "SAE-B停止: 局部K=128/256均未通过预注册门槛；不启动全局分支。"
  exit 0
fi

selected_k=""
for top_k in 128 256; do
  run_one global "$top_k" || exit $?
  if passed global "$top_k"; then
    selected_k="$top_k"
    echo "全局空间SAE通过: K=$top_k"
    break
  fi
done
if [[ -z "$selected_k" ]]; then
  echo "SAE-B停止: 全局K=128/256均未通过预注册门槛。"
  exit 0
fi
echo "SAE-B seed42双分支均通过，可进入SAE-B1复现。"
