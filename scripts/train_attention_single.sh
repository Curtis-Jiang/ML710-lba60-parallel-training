#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_compact_dataset.py
python scripts/train_binding.py \
  --config configs/attention_course.yaml \
  --strategy single \
  --run-name attention_single_course
