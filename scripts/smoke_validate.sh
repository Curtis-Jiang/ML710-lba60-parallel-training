#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-lba60_smoke_validation_$(date -u +%Y%m%d_%H%M%S)}"
SEED="${2:-0}"
CKPT_PATH="$REPO_ROOT/runs/affinity/lba60/$RUN_NAME/ckpt_best.pt"

cd "$REPO_ROOT"

echo "[1/4] Forward sanity check"
python scripts/forward_sanity.py --config configs/lba60_quick.yaml --split train --batch_size 2

echo "[2/4] Packed launcher dry run"
python scripts/launch_lba60_jobs.py --spec configs/packed_seed_sweep.yaml --dry_run

echo "[3/4] Single-GPU smoke training"
bash scripts/train_lba60_single.sh configs/lba60_smoke.yaml "$RUN_NAME" "$SEED"

echo "[4/4] Checkpoint evaluation"
bash scripts/eval_lba60.sh "$CKPT_PATH" test 32 val

echo "Smoke validation completed successfully."
echo "Run directory: $REPO_ROOT/runs/affinity/lba60/$RUN_NAME"
