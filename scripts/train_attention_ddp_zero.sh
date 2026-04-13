#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_compact_dataset.py
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_binding.py \
  --config configs/attention_course.yaml \
  --strategy ddp_zero \
  --run-name attention_ddp_zero_course
