#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:?usage: bash scripts/eval_lba60.sh <ckpt> [split] [batch_size] [calibrate_on|none]}"
SPLIT="${2:-test}"
BATCH_SIZE="${3:-32}"
CALIBRATE_ON="${4:-val}"

cd "$REPO_ROOT"

CMD=(
  python
  binding_affinity/scripts/eval_affinity_model.py
  --ckpt "$CKPT"
  --split "$SPLIT"
  --batch_size "$BATCH_SIZE"
)

if [[ "$CALIBRATE_ON" != "none" ]]; then
  CMD+=(--calibrate_on "$CALIBRATE_ON")
fi

"${CMD[@]}"
