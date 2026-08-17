#!/usr/bin/env bash
# 固定复现seed42已通过的全局A1c配置；不扫描超参数，不读取test或external。
set -uo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/efficientnet_sae_discovery.py"
SUMMARY="$PROJECT_ROOT/程序/SAE/正式代码/summarize_efficientnet_sae_global_replication.py"
OUTPUT="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-C全局多种子复现"
SEED42_RUN="$PROJECT_ROOT/结果/SAE/EfficientNet全局局部_0804/SAE-A1c_Margin保真/a1c_global_efficientnet_b0_seed42_h10240_topk1024_margin01"
CUDA_DEVICE="${CUDA_DEVICE:?请先设置CUDA_DEVICE；运行前按nvidia-smi实际占用选择}"
SEED_LIST="${1:-202,503}"

cd "$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
mkdir -p "$OUTPUT/logs"

if [[ ! -f "$SEED42_RUN/config.json" || ! -f "$SEED42_RUN/metrics.json" ]]; then
  echo "缺少作为预注册依据的seed42全局A1c正式结果: $SEED42_RUN" >&2
  exit 1
fi

IFS=',' read -r -a seeds <<< "$SEED_LIST"
failures=()
for seed in "${seeds[@]}"; do
  if [[ "$seed" != "202" && "$seed" != "503" ]]; then
    echo "本启动器只允许固定复现seed202/503，收到: $seed" >&2
    exit 2
  fi
  experiment="sae_c_global_efficientnet_b0_seed${seed}_h10240_topk1024_margin01"
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
  # Python直接追加日志，监控终端断开时训练输出仍由当前shell持有。
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$SCRIPT" \
    --branch global --seed "$seed" --hidden-dim 10240 \
    --activation-mode topk --top-k 1024 --lambda-l1 0 \
    --margin-loss-weight 0.1 \
    --epochs 1000 --patience 50 --warmup-fraction 0.05 \
    --image-batch-size 32 --sae-batch-size 32 --num-workers 4 \
    --output-root "$OUTPUT" --experiment "$experiment" >> "$log" 2>&1
  status=$?
  if ((status == 0)); then
    echo "[$(date '+%F %T')] DONE $experiment" | tee -a "$log"
  else
    echo "[$(date '+%F %T')] FAILED $experiment status=$status" | tee -a "$log"
    failures+=("$experiment:$status")
  fi
done

"$PYTHON" "$SUMMARY"
if ((${#failures[@]})); then
  printf '全局SAE复现失败项: %s\n' "${failures[*]}"
  exit 1
fi
echo "全局SAE复现完成: $OUTPUT"
