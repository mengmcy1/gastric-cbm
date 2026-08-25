#!/usr/bin/env bash
# 顺序执行RP-C2工程preflight与四对象probe；不启动正式五档科学运行。
set -Eeuo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
CODE_ROOT="$PROJECT_ROOT/程序/SAE/正式代码"
RESULT_ROOT="$PROJECT_ROOT/结果/SAE/RP_C2_Intervention_20260825"
LOG_ROOT="$RESULT_ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-/home/mcy/miniconda3/envs/gastric-cbm/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$LOG_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$CODE_ROOT/test_clong_rpc2.py" \
  2>&1 | tee "$LOG_ROOT/tests.log"

if [[ -f "$RESULT_ROOT/preflight/config.json" ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("/home/mcy/gastric-cbm/结果/SAE/RP_C2_Intervention_20260825/preflight/config.json")
if json.loads(path.read_text(encoding="utf-8")).get("status") != "rpc2_preflight_passed":
    raise SystemExit("preflight config状态不完整，拒绝跳过")
PY
  echo "preflight已完整通过，幂等跳过"
elif [[ -e "$RESULT_ROOT/preflight" ]]; then
  echo "preflight存在残缺现场，拒绝覆盖: $RESULT_ROOT/preflight" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -u \
    "$CODE_ROOT/run_clong_rpc2_preflight_probe.py" preflight --device cuda \
    2>&1 | tee "$LOG_ROOT/preflight.log"
fi

if [[ -f "$RESULT_ROOT/probe/config.json" ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("/home/mcy/gastric-cbm/结果/SAE/RP_C2_Intervention_20260825/probe/config.json")
if json.loads(path.read_text(encoding="utf-8")).get("status") != "rpc2_engineering_probe_complete":
    raise SystemExit("probe config状态不完整，拒绝跳过")
PY
  echo "probe已完整通过，幂等跳过"
elif [[ -e "$RESULT_ROOT/probe" ]]; then
  echo "probe存在残缺现场，拒绝覆盖: $RESULT_ROOT/probe" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -u \
    "$CODE_ROOT/run_clong_rpc2_preflight_probe.py" probe --device cuda \
    2>&1 | tee "$LOG_ROOT/probe.log"
fi

echo "RP-C2工程preflight/probe完成；未生成科学结论。"
