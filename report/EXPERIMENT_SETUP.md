# Experiment Setup

## Objective

Measure how the same `lba60` workload behaves under:

1. a single-GPU baseline
2. a 2-GPU DDP run

This directly supports the ML710 course focus on parallel throughput and scaling behavior.

## Shared Configuration

Both runs use:

- config: `configs/lba60_quick.yaml`
- task: `lba60`
- seed: `0`
- epochs: `20`
- per-GPU batch size: `8`
- optimizer: `AdamW`
- scheduler: warmup + cosine decay
- loss: MSE

## Commands

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_lba60_single.sh configs/lba60_quick.yaml lba60_single_quick_seed0 0
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=1,2 NPROC_PER_NODE=2 bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_quick_seed0 0
```

## Output Locations

Single GPU run:

`runs/affinity/lba60/lba60_single_quick_seed0/`

DDP run:

`runs/affinity/lba60/lba60_ddp_quick_seed0/`

The report uses saved summaries and metrics instead of raw terminal logs.

## Notes

- The runs were launched on H100 GPUs in the current environment.
- The same config is used for both runs so throughput and convergence can be compared cleanly.
- Final report tables should be generated from `summary.json`, `test_metrics.json`, and `metrics.jsonl`.
