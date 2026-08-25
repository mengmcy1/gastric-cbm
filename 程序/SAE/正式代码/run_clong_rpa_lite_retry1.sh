#!/usr/bin/env bash
# RP-A-lite CUDA timeout工程恢复：保留replicate56，以32x64只补剩余14条。
set -Eeuo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
CODE="$PROJECT_ROOT/程序/SAE/正式代码"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
SOURCE_ROOT="$PROJECT_ROOT/结果/SAE/RP_A_Development_20260824"
OUTPUT_ROOT="$PROJECT_ROOT/结果/SAE/RP_A_Lite_Exploratory_20260825"
LOG_DIR="$OUTPUT_ROOT/logs_retry1"
CUDA_DEVICES="${CUDA_DEVICES:?Lite retry1启动前必须显式设置3张CUDA_DEVICES}"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"

[[ "${#GPUS[@]}" -eq 3 ]] || { echo "Lite retry1固定使用3个worker" >&2; exit 1; }
[[ -f "$OUTPUT_ROOT/missing_workers/worker_00_of_03.partial.jsonl" ]] \
  || { echo "缺少首次恢复现场" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT/missing_retry1" && ! -e "$OUTPUT_ROOT/summary" ]] \
  || { echo "Lite retry1或汇总产物已存在，禁止覆盖" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "非法GPU编号: $gpu" >&2; exit 1; }
done

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export RPA_DEVELOPMENT_ROOT="$SOURCE_ROOT"
nvidia-smi | tee "$LOG_DIR/nvidia_smi_preflight.log"
"$PYTHON" "$CODE/test_clong_rpa_lite.py" 2>&1 | tee "$LOG_DIR/rpa_lite_tests.log"

if [[ ! -f "$OUTPUT_ROOT/equivalence/replicate53_32x64.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" -u "$CODE/clong_rpa_lite_worker.py" \
    --mode equivalence64 --worker-rank 0 --worker-count 1 --device cuda \
    --source-block 32 --target-block 64 >"$LOG_DIR/equivalence64.log" 2>&1
fi

PYTHONPATH="$CODE" "$PYTHON" - <<'PY' | tee "$LOG_DIR/equivalence64_check.log"
from pathlib import Path
from clong_rpa_lite_core import collect_source_records, load_jsonl, max_metric_difference
source = Path("/home/mcy/gastric-cbm/结果/SAE/RP_A_Development_20260824/bootstrap_workers/formal")
old, _ = collect_source_records([source / f"worker_{rank:02d}_of_03.jsonl" for rank in range(3)])
old53 = next(row for row in old if int(row["replicate_index"]) == 53)
new53 = load_jsonl(Path("/home/mcy/gastric-cbm/结果/SAE/RP_A_Lite_Exploratory_20260825/equivalence/replicate53_32x64.jsonl"))[0]
difference = max_metric_difference(old53, new53)
print(f"replicate53_32x64_max_absolute_difference={difference}")
if difference != 0.0:
    raise SystemExit("32x64等价性未通过")
PY

pids=()
for rank in 0 1 2; do
  log="$LOG_DIR/worker${rank}.log"
  CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PYTHON" -u "$CODE/clong_rpa_lite_worker.py" \
    --mode retry1 --worker-rank "$rank" --worker-count 3 --device cuda \
    --source-block 32 --target-block 64 >"$log" 2>&1 &
  pids+=("$!")
  echo "Lite retry1 worker $rank -> GPU ${GPUS[$rank]}, log=$log"
done

remaining=("${pids[@]}")
while [[ "${#remaining[@]}" -gt 0 ]]; do
  finished=""
  if wait -n -p finished "${remaining[@]}"; then status=0; else status="$?"; fi
  next=()
  for pid in "${remaining[@]}"; do [[ "$pid" != "$finished" ]] && next+=("$pid"); done
  remaining=("${next[@]}")
  if [[ "$status" -ne 0 ]]; then
    for pid in "${remaining[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${remaining[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo "Lite retry1 worker失败，保留现场" >&2
    exit "$status"
  fi
done

"$PYTHON" -u "$CODE/summarize_clong_rpa_lite.py" \
  2>&1 | tee "$LOG_DIR/summary.log"
echo "[$(date '+%F %T')] RP-A-lite retry1 completed"
