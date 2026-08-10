#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="PLUS-WAVE/InfiniSplat",
    allow_patterns="checkpoints/*.ckpt",
    local_dir=".",
)
PY
