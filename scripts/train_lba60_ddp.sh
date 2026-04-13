#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/lba60_quick.yaml}"
RUN_NAME="${2:-lba60_ddp_seed0}"
SEED="${3:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

cd "$REPO_ROOT"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  binding_affinity/scripts/train_affinity_model.py \
  --config "$CONFIG" \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  "${@:4}"
