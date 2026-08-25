#!/usr/bin/env bash
# RP-A 42/43/44 development-calibration完整runner；202/503/911不在本入口中。
set -Eeuo pipefail

PROJECT_ROOT="/home/mcy/gastric-cbm"
CODE="$PROJECT_ROOT/程序/SAE/正式代码"
PYTHON="/home/mcy/miniconda3/envs/gastric-cbm/bin/python"
OUTPUT="${RPA_DEVELOPMENT_ROOT:-$PROJECT_ROOT/结果/SAE/RP_A_Development_20260824}"
LOG_DIR="$OUTPUT/logs"
CUDA_DEVICES="${CUDA_DEVICES:?启动前检查GPU并以逗号分隔显式设置CUDA_DEVICES}"
SOURCE_BLOCK="${RPA_SOURCE_BLOCK:-32}"
TARGET_BLOCK="${RPA_TARGET_BLOCK:-128}"
export RPA_DEVELOPMENT_ROOT="$OUTPUT" RPA_SOURCE_BLOCK="$SOURCE_BLOCK" RPA_TARGET_BLOCK="$TARGET_BLOCK"
IFS=',' read -r -a GPUS <<< "$CUDA_DEVICES"
CURRENT_STAGE="startup"
CURRENT_COMMAND="$0"
CURRENT_LOG="$LOG_DIR/rpa_development_runner.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

record_failure() {
  local status="$?"
  trap - ERR
  "$PYTHON" "$CODE/clong_rpa_record_implementation_failure.py" \
    --stage "$CURRENT_STAGE" --command "$CURRENT_COMMAND" \
    --log "$CURRENT_LOG" --exit-code "$status" || true
  exit "$status"
}
trap record_failure ERR

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "CUDA_DEVICES至少包含一张卡" | tee -a "$CURRENT_LOG"
  exit 1
fi
CURRENT_STAGE="gpu_preflight"
CURRENT_COMMAND="nvidia-smi"
nvidia-smi | tee -a "$CURRENT_LOG"
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "非法GPU编号: $gpu" | tee -a "$CURRENT_LOG"
    exit 1
  fi
done
if ! [[ "$SOURCE_BLOCK" =~ ^[1-9][0-9]*$ && "$TARGET_BLOCK" =~ ^[1-9][0-9]*$ ]]; then
  echo "RPA_SOURCE_BLOCK和RPA_TARGET_BLOCK必须为正整数" | tee -a "$CURRENT_LOG"
  exit 1
fi

CODE_SNAPSHOT="$OUTPUT/run_code_snapshot.json"
CURRENT_STAGE="code_provenance"
CURRENT_COMMAND="$PYTHON $CODE/clong_rpa_provenance.py"
if [[ -f "$CODE_SNAPSHOT" ]]; then
  "$PYTHON" "$CODE/clong_rpa_provenance.py" verify --output "$CODE_SNAPSHOT" \
    2>&1 | tee -a "$CURRENT_LOG"
else
  "$PYTHON" "$CODE/clong_rpa_provenance.py" create --output "$CODE_SNAPSHOT" \
    2>&1 | tee -a "$CURRENT_LOG"
fi

run_logged() {
  local stage="$1"
  local log="$2"
  shift 2
  CURRENT_STAGE="$stage"
  CURRENT_LOG="$log"
  CURRENT_COMMAND="$*"
  echo "[$(date '+%F %T')] START $stage" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] DONE $stage" | tee -a "$log"
}

run_logged "unit_tests" "$LOG_DIR/rpa_development_tests.log" \
  "$PYTHON" "$CODE/test_clong_rpa_development.py"
run_logged "development_training" "$LOG_DIR/rpa_development_training_matrix.log" \
  env CUDA_DEVICE="${GPUS[0]}" bash "$CODE/run_clong_rpa_development_training.sh"

for seed in 42 43 44; do
  cache="$OUTPUT/analysis_cache/seed${seed}/config.json"
  if [[ -f "$cache" ]]; then
    echo "SKIP prepare seed${seed}（完整config存在）" | tee -a "$CURRENT_LOG"
  else
    run_logged "prepare_seed${seed}" "$LOG_DIR/rpa_prepare_seed${seed}.log" \
      env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" -u \
      "$CODE/clong_rpa_prepare_seed.py" --seed "$seed" --device cuda
  fi
done

MATCHING="$OUTPUT/full_train_matching/formal"
if [[ -f "$MATCHING/full_train_metrics.json" ]]; then
  echo "SKIP full_train_matching（完整metrics存在）" | tee -a "$CURRENT_LOG"
else
  run_logged "full_train_matching" "$LOG_DIR/rpa_full_train_matching.log" \
    env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" -u \
    "$CODE/clong_rpa_match_development.py" --device cuda \
    --source-block "$SOURCE_BLOCK" --target-block "$TARGET_BLOCK"
fi

anchor_count="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["strict_anchor_count"])' "$MATCHING/full_train_metrics.json")"
if [[ "$anchor_count" -lt 100 ]]; then
  run_logged "finalize_scientific_failure" "$LOG_DIR/rpa_finalize.log" \
    "$PYTHON" -u "$CODE/clong_rpa_finalize_development.py"
  echo "RP-A按协议停止：strict anchors=$anchor_count < 100" | tee -a "$CURRENT_LOG"
  exit 0
fi

WORKER_ROOT="$OUTPUT/bootstrap_workers/formal"
worker_count="${#GPUS[@]}"
complete_workers=0
for rank in "${!GPUS[@]}"; do
  file="$WORKER_ROOT/worker_$(printf '%02d' "$rank")_of_$(printf '%02d' "$worker_count").jsonl"
  [[ -f "$file" ]] && complete_workers=$((complete_workers + 1))
done
if [[ "$complete_workers" -eq "$worker_count" ]]; then
  echo "SKIP bootstrap workers（全部输出存在）" | tee -a "$CURRENT_LOG"
elif [[ "$complete_workers" -ne 0 || -d "$WORKER_ROOT" ]]; then
  echo "bootstrap worker存在残缺现场，请人工审阅，不自动覆盖" | tee -a "$CURRENT_LOG"
  exit 1
else
  CURRENT_STAGE="bootstrap_workers"
  CURRENT_COMMAND="parallel static bootstrap workers"
  pids=()
  worker_logs=()
  for rank in "${!GPUS[@]}"; do
    log="$LOG_DIR/rpa_bootstrap_worker${rank}.log"
    env CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PYTHON" -u \
      "$CODE/clong_rpa_bootstrap_worker.py" \
      --worker-rank "$rank" --worker-count "$worker_count" --device cuda \
      --source-block "$SOURCE_BLOCK" --target-block "$TARGET_BLOCK" \
      >"$log" 2>&1 &
    pids+=("$!")
    worker_logs+=("$log")
    echo "bootstrap worker $rank -> GPU ${GPUS[$rank]}, log=$log" | tee -a "$CURRENT_LOG"
  done
  remaining=("${pids[@]}")
  while [[ "${#remaining[@]}" -gt 0 ]]; do
    finished_pid=""
    if wait -n -p finished_pid "${remaining[@]}"; then
      status=0
    else
      status="$?"
    fi
    next=()
    for pid in "${remaining[@]}"; do
      [[ "$pid" != "$finished_pid" ]] && next+=("$pid")
    done
    remaining=("${next[@]}")
    if [[ "$status" -ne 0 ]]; then
      for index in "${!pids[@]}"; do
        if [[ "${pids[$index]}" == "$finished_pid" ]]; then
          CURRENT_LOG="${worker_logs[$index]}"
          break
        fi
      done
      for pid in "${remaining[@]}"; do
        kill "$pid" 2>/dev/null || true
      done
      for pid in "${remaining[@]}"; do
        wait "$pid" 2>/dev/null || true
      done
      false
    fi
  done
fi

BOOTSTRAP="$OUTPUT/bootstrap_calibration/formal/bootstrap_thresholds.json"
if [[ -f "$BOOTSTRAP" ]]; then
  echo "SKIP bootstrap coordinator（结果存在）" | tee -a "$CURRENT_LOG"
else
  run_logged "bootstrap_coordinator" "$LOG_DIR/rpa_bootstrap_coordinator.log" \
    "$PYTHON" -u "$CODE/clong_rpa_bootstrap_coordinator.py" \
    --worker-count "$worker_count"
fi
bootstrap_status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$BOOTSTRAP")"
if [[ "$bootstrap_status" == "bootstrap_calibration_infeasible" ]]; then
  run_logged "finalize_scientific_failure" "$LOG_DIR/rpa_finalize.log" \
    "$PYTHON" -u "$CODE/clong_rpa_finalize_development.py"
  exit 0
fi

VAL="$OUTPUT/val_reproduction/formal/val_reproduction.json"
if [[ -f "$VAL" ]]; then
  echo "SKIP val_reproduction（结果存在）" | tee -a "$CURRENT_LOG"
else
  run_logged "val_reproduction" "$LOG_DIR/rpa_val_reproduction.log" \
    env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" -u \
    "$CODE/clong_rpa_validate_development.py" --device cuda \
    --source-block "$SOURCE_BLOCK" --target-block "$TARGET_BLOCK"
fi

run_logged "finalize" "$LOG_DIR/rpa_finalize.log" \
  "$PYTHON" -u "$CODE/clong_rpa_finalize_development.py"
echo "RP-A development-calibration runner完成。" | tee -a "$CURRENT_LOG"
