#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_compact_dataset.py
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_binding.py \
  --config configs/mamba_course.yaml \
  --strategy ddp \
  --run-name mamba_ddp_course
