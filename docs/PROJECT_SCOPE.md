# Project Scope

This ML710 version is intentionally narrower than the original research workspace.

## Kept

- one task: `lba60`
- one model family: a single graph-based affinity regressor
- one copied processed dataset snapshot
- one training script
- one evaluation script
- one packed multi-run launcher for goodput experiments

## Removed From The Main Story

- `lba30`
- `lep`
- expert ensembling
- MoE routing
- teacher stacking
- cross-project paper infrastructure

## Why This Scope Fits ML710 Better

The course project is about parallelizing and analyzing an ML workload, not about reproducing a paper's full benchmark suite. This repo therefore treats `lba60` as the workload and focuses on:

- throughput
- scaling efficiency
- time-to-accuracy
- goodput under a fixed GPU budget

## Recommended Experiment Ladder

1. Single-GPU baseline
   Measure throughput and final validation/test quality from `configs/lba60_quick.yaml`.

2. Two-GPU DDP
   Use `scripts/train_lba60_ddp.sh` with the same config to study data-parallel speedup and memory behavior.

3. Packed concurrent runs
   Use `scripts/launch_lba60_jobs.py` with `configs/packed_seed_sweep.yaml` to compare goodput against a single larger run under the same total GPU budget.

## Optional Extension

If the team wants a third non-trivial parallel angle beyond plain 2-GPU DDP, the cleanest extension is:

- strong-scaling or weak-scaling sweeps by changing `NPROC_PER_NODE`
- packed multi-job scheduling versus one wider DDP job

That keeps the entire story on one workload instead of drifting back into multi-dataset paper mode.

## Primary Metrics To Report

- `train/examples_per_sec`
- wall-clock time per run
- best validation Pearson correlation
- test Pearson and RMSE
- memory usage from `cuda/peak_reserved_gb`
- goodput under the same total GPU allocation
