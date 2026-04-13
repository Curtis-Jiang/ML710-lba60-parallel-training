# ML710 Summary

## Executive Summary

This repository packages one ML710 workload:

- `lba60` protein-ligand binding-affinity regression

The original research workspace was intentionally simplified so the project can focus on parallel training behavior, not on reproducing a multi-benchmark paper pipeline.

## Completed Comparison

Two quick-profile runs have already been completed with the same core setup:

- config: `configs/lba60_quick.yaml`
- epochs: `20`
- seed: `0`
- optimizer: `AdamW`
- scheduler: warmup plus cosine decay
- loss: MSE

Compared runs:

1. `lba60_single_quick_seed0`
2. `lba60_ddp_quick_seed0`

Important caveat:

- single GPU used global batch `8`
- DDP used global batch `16`

So the current comparison is strong for throughput analysis and directionally useful for quality analysis, but it is not a perfectly controlled statistical-efficiency study.

## Main Results

| run | GPUs | global batch | approx wall min | avg ex/s | best val pearson | test pearson | test rmse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lba60_single_quick_seed0` | 1 | 8 | 35.96 | 34.64 | 0.7747 | 0.7758 | 1.3115 |
| `lba60_ddp_quick_seed0` | 2 | 16 | 18.47 | 70.49 | 0.7546 | 0.7684 | 1.3335 |

Derived takeaways:

- DDP average throughput speedup: about `2.04x`
- DDP approximate wall-clock speedup: about `1.95x`

## ML710 Interpretation

### Throughput And Runtime

This is the strongest result in the current submission package:

- throughput improves from `34.64` to `70.49` examples/s
- end-to-end runtime drops from about `36.0` to about `18.5` minutes

That is close to linear scaling for a 2-GPU DDP run.

### Quality Tradeoff

The single-GPU run finished with slightly better validation and test quality. The correct interpretation is not that DDP failed, but that the global batch changed from `8` to `16`, which changes the optimization path.

### Why This Is Still A Good Course Result

This is exactly the kind of result worth presenting in an ML systems class:

1. the parallel version clearly improves throughput
2. the faster configuration does not automatically dominate quality
3. systems gains and optimization behavior need to be discussed together

## Submission Recommendation

For a final ML710 submission, the current evidence supports the following recommendation:

- use single GPU as the accuracy-oriented baseline
- use 2-GPU DDP as the speed-oriented configuration
- present packed jobs as the next goodput-oriented extension under a fixed GPU budget

## Highest-Value Follow-Up

If the team runs one more experiment, the best next step is:

- 2-GPU DDP with per-GPU batch `4` so the global batch matches the single-GPU baseline

That produces a much cleaner statistical-efficiency comparison. The repo already also supports packed concurrent jobs through `scripts/launch_lba60_jobs.py`.

## Supporting Files

- `report/RUN_COMPARISON.md`
- `report/artifacts/run_comparison.json`
- `report/artifacts/lba60_single_quick_seed0.summary.json`
- `report/artifacts/lba60_ddp_quick_seed0.summary.json`
- `runs/affinity/lba60/lba60_single_quick_seed0/`
- `runs/affinity/lba60/lba60_ddp_quick_seed0/`
