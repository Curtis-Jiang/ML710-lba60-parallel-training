#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/lba60_quick.yaml}"
RUN_NAME="${2:-lba60_single_seed0}"
SEED="${3:-0}"

cd "$REPO_ROOT"

python binding_affinity/scripts/train_affinity_model.py \
  --config "$CONFIG" \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  "${@:4}"
