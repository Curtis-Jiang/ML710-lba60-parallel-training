# Course Requirements Mapping

## What This Rebuild Covers

- one clear supervised workload
- 1-GPU baseline
- 2-GPU naive DDP baseline
- 2-GPU DDP + ZeroRedundancyOptimizer as the advanced distributed strategy
- report artifacts for throughput and validation quality
- final course-scale runs with `mamba_ssm`
- runtime settings that are much closer to the intended A100 half-hour target

## What It Does Not Claim

This repo gets much closer to a final ML710 submission, but by itself it does
not satisfy the full multi-student strategy count from the original course
handout. Additional advanced strategies would still be needed for a fully
complete group submission if the team must satisfy the exact “one per student”
rule.

## Metrics Tracked

- throughput (`train_examples_per_sec`)
- wall-clock time (`total_wall_time_sec`)
- validation AUC
- validation AUPR
- accuracy and F1

## Current Final Run Set

- `attention_single_course`
- `attention_ddp_zero_course`
- `mamba_single_course`
- `mamba_ddp_zero_course`

---

# v2 Requirements Coverage (2026-04-23 full sweep)

The v1 mapping above reflects the "one advanced strategy + single-GPU baseline"
version of the project. The v2 sweep extends coverage to the full "one strategy
per student" course rule by running a broader parallelism matrix on the same
workload.

## v2 What Is Now Covered

- one clear supervised workload (unchanged)
- 1-GPU baseline for both model families
- 2-GPU naive DDP baseline
- 2-GPU and 4-GPU DDP + ZeroRedundancyOptimizer (ZeRO-1) as an advanced strategy
- 2-GPU and 4-GPU FSDP ZeRO-2 (shard optimizer states + gradients)
- 2-GPU and 4-GPU FSDP ZeRO-3 (full parameter sharding)
- 2-GPU and 4-GPU Tensor Parallelism (Megatron-style column/row-split attention)
- 2-GPU Branch Model Parallelism (attention branch split across protein / SMILES)
- 4-GPU 2D Hybrid Parallelism (TP2 x DP2)
- 1-GPU and 2-GPU Mamba baseline (with `mamba_ssm` backend)
- Phase-level completion audit in `report/PHASE_COMPLETION_REPORT.md`
- throughput, wall-clock, scaling efficiency, goodput, and peak GPU memory
  captured per run in `report/artifacts/`

## v2 Strategy-to-Student Mapping

This is the set of distinct parallel strategies exercised end-to-end and
suitable for one-per-student attribution:

| # | Strategy | Representative Runs |
| ---: | --- | --- |
| 1 | Single GPU | `attention_single_course`, `mamba_single_course` |
| 2 | DDP (naive data parallel) | `attention_ddp_2gpu_course`, `attention_ddp_4gpu_course`, `mamba_ddp_course` |
| 3 | DDP + ZeRO-1 (ZeroRedundancyOptimizer) | `attention_ddp_zero_2gpu_course`, `attention_ddp_zero_4gpu_course` |
| 4 | FSDP ZeRO-2 (shard grads + opt states) | `attention_fsdp_z2_2gpu_course`, `attention_fsdp_z2_4gpu_course` |
| 5 | FSDP ZeRO-3 (full parameter sharding) | `attention_fsdp_z3_2gpu_course`, `attention_fsdp_z3_4gpu_course` |
| 6 | Tensor Parallel (Megatron-style) | `attention_tp2_course`, `attention_tp4_course` |
| 7 | Branch Model Parallel | `attention_branch_mp_course` |
| 8 | Hybrid 2D (TP x DP) | `attention_hybrid_tp2_dp2_course` |

## v2 Metrics Tracked

Unchanged from v1, plus:

- `scaling efficiency` = `(throughput_N / throughput_1) / N`
- `wall speedup` vs single-GPU baseline
- `peak GPU bytes rank0` per strategy/GPU count
- `goodput` combining throughput and validation quality
  (see `report/artifacts/goodput_table.csv`)

## v2 What It Still Does Not Claim

- Pipeline Parallelism (GPipe / 1F1B) is not included; not required for this
  model size but would be the natural next-step strategy if the team extends
  to larger models.
- 3D parallelism (TP x PP x DP) is not implemented; `hybrid_tp2_dp2` is the
  highest-dimensional parallelism in v2.
- No multi-node experiments; all runs are on a single node (up to 4 GPUs).
