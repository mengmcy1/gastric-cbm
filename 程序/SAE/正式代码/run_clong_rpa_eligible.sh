#!/usr/bin/env bash
set -uo pipefail

# RP-A eligible 正式聚合审计与活跃频率校准启动器。
# STAGE=all 首次串行执行两阶段；若审计已完成而校准中断，可仅用
# STAGE=calibrate 复用已绑定 SHA 的患者缓存。

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
ENTRY="$PROJECT_ROOT/程序/SAE/正式代码/clong_rpa_eligible.py"
STAGE="${STAGE:-all}"

if [[ -z "${CUDA_DEVICE:-}" ]]; then
  echo "错误: 必须显式设置 CUDA_DEVICE" >&2
  exit 2
fi
if [[ "$STAGE" != "all" && "$STAGE" != "audit" && "$STAGE" != "calibrate" ]]; then
  echo "错误: STAGE只能是all/audit/calibrate" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

cd "$PROJECT_ROOT" || exit 1
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$ENTRY" \
  --stage "$STAGE" \
  --device cuda \
  --batch-size 8
