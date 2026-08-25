#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SCRIPT="$PROJECT_ROOT/程序/SAE/正式代码/render_clong_rpd_atlas.py"
OUTPUT_ROOT="$PROJECT_ROOT/结果/SAE/RP_D_Technical_Atlas_20260825/render_v1"

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T')] START RP-D v1 formal render"
"$PYTHON" "$SCRIPT" --output-root "$OUTPUT_ROOT"
echo "[$(date '+%F %T')] DONE RP-D v1 formal render"
