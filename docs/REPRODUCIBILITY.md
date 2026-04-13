# Reproducibility

This file gives the shortest reliable path from a working CUDA environment to a validated ML710 run.

## Assumptions

- CUDA GPUs are visible
- PyTorch with CUDA support is installed
- the current working directory is the repo root

## Step 1: Run The Smoke Validation

```bash
bash scripts/smoke_validate.sh
```

The smoke script checks four things:

1. config plus dataset plus model wiring via `forward_sanity.py`
2. packed-job launcher parsing via `--dry_run`
3. a short 2-epoch single-GPU training run
4. checkpoint evaluation on the smoke run

If the script finishes successfully, the core training and evaluation path is intact.

## Step 2: Reproduce The Main Comparison

Single-GPU baseline:

```bash
bash scripts/train_lba60_single.sh configs/lba60_quick.yaml lba60_single_seed0 0
```

2-GPU DDP:

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_seed0 0
```

Evaluate:

```bash
bash scripts/eval_lba60.sh runs/affinity/lba60/lba60_ddp_seed0/ckpt_best.pt test 32 val
```

## Step 3: Rebuild The Comparison Report

```bash
python scripts/build_report.py \
  --runs \
  runs/affinity/lba60/lba60_single_seed0 \
  runs/affinity/lba60/lba60_ddp_seed0
```

This regenerates:

- `report/RUN_COMPARISON.md`
- `report/artifacts/run_comparison.json`

## Optional Step 4: Goodput-Oriented Packed Runs

Dry-run first:

```bash
python scripts/launch_lba60_jobs.py --spec configs/packed_seed_sweep.yaml --dry_run
```

Launch for real when enough GPUs are available:

```bash
python scripts/launch_lba60_jobs.py --spec configs/packed_seed_sweep.yaml
```

## Expected Output Locations

- run outputs: `runs/affinity/lba60/<run_name>/`
- evaluation metrics: `test_metrics.json`
- run summary: `summary.json`
- report copies: `report/artifacts/`

## Common Overrides

Reduce batch size if memory is tight:

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_bs6 0 --set train.batch_size=6
```

Match the single-GPU global batch inside DDP:

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_gb8 0 --set train.batch_size=4
```

## What Counts As Success

A run is in good shape when:

- `summary.json` exists
- `ckpt_best.pt` exists
- `metrics.jsonl` contains epoch-level throughput lines
- evaluation completes without an import or checkpoint-path error

## Reporting Note

The existing quick-run numbers in `report/` were produced on H100 hardware. If you re-run on A100, report the new wall-clock times explicitly instead of reusing estimates.
