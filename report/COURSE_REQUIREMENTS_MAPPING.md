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
