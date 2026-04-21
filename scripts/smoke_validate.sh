#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_compact_dataset.py

rm -rf \
  runs/attention_smoke_single \
  runs/attention_smoke_ddp \
  runs/attention_smoke_ddp_zero \
  runs/mamba_smoke_single \
  runs/mamba_smoke_ddp

python scripts/train_binding.py \
  --config configs/attention_smoke.yaml \
  --strategy single \
  --run-name attention_smoke_single
python scripts/eval_binding.py \
  --config configs/attention_smoke.yaml \
  --checkpoint runs/attention_smoke_single/best.pt \
  --output report/artifacts/attention_smoke_single_eval.json

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_binding.py \
  --config configs/attention_smoke.yaml \
  --strategy ddp \
  --run-name attention_smoke_ddp
python scripts/eval_binding.py \
  --config configs/attention_smoke.yaml \
  --checkpoint runs/attention_smoke_ddp/best.pt \
  --output report/artifacts/attention_smoke_ddp_eval.json

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_binding.py \
  --config configs/attention_smoke.yaml \
  --strategy ddp_zero \
  --run-name attention_smoke_ddp_zero
python scripts/eval_binding.py \
  --config configs/attention_smoke.yaml \
  --checkpoint runs/attention_smoke_ddp_zero/best.pt \
  --output report/artifacts/attention_smoke_ddp_zero_eval.json

python scripts/check_mamba_install.py

python scripts/train_binding.py \
  --config configs/mamba_smoke.yaml \
  --strategy single \
  --run-name mamba_smoke_single
python scripts/eval_binding.py \
  --config configs/mamba_smoke.yaml \
  --checkpoint runs/mamba_smoke_single/best.pt \
  --output report/artifacts/mamba_smoke_single_eval.json

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_binding.py \
  --config configs/mamba_smoke.yaml \
  --strategy ddp \
  --run-name mamba_smoke_ddp
python scripts/eval_binding.py \
  --config configs/mamba_smoke.yaml \
  --checkpoint runs/mamba_smoke_ddp/best.pt \
  --output report/artifacts/mamba_smoke_ddp_eval.json

python scripts/build_report.py --runs-dir runs --report-dir report
echo "smoke validation finished"
