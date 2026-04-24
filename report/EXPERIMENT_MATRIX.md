# Experiment Matrix

## Recommended Final Course Runs

| Run | Model | Backend | GPUs | Strategy | Global Batch | H100 Wall Time |
| --- | --- | --- | ---: | --- | ---: | ---: |
| attention_single_course | attention | `-` | 1 | single | 64 | `1203.80 s` |
| attention_ddp_zero_course | attention | `-` | 2 | ddp_zero | 64 | `990.25 s` |
| mamba_single_course | mamba | `mamba_ssm` | 1 | single | 64 | `1954.46 s` |
| mamba_ddp_zero_course | mamba | `mamba_ssm` | 2 | ddp_zero | 64 | `1338.40 s` |

## Optional Additional Baselines

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_ddp_course | attention | 2 | ddp | 64 |
| mamba_ddp_course | mamba | 2 | ddp | 64 |

## Smoke Runs

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_smoke_single | attention | 1 | single | 32 |
| attention_smoke_ddp | attention | 2 | ddp | 32 |
| attention_smoke_ddp_zero | attention | 2 | ddp_zero | 32 |
| mamba_smoke_single | mamba | 1 | single | 32 |
| mamba_smoke_ddp | mamba | 2 | ddp | 32 |

---

# v2 Matrix – Full Phase B/C/D Sweep (2026-04-23)

v1 above is kept as the "original course target" plan. v2 below is the actual
parallel-strategy matrix that was executed end-to-end across three phases, with
measured H100 80GB wall times filled in where runs have completed.

## v2 Phase B – Single-GPU Baselines (parallel on 2 GPUs)

| Run | Model | GPUs | Strategy | Global Batch | Script | H100 Wall Time |
| --- | --- | ---: | --- | ---: | --- | ---: |
| attention_single_course | attention | 1 | single | 64 | `scripts/train_attention_single.sh` | `1985.27 s` |
| mamba_single_course | mamba | 1 | single | 64 | `scripts/train_mamba_single.sh` | `3257.71 s` |

## v2 Phase C – 2-GPU Strategy Sweep (paired in 3 rounds)

| Round | Run | Model | GPUs | Strategy | Global Batch | Script | H100 Wall Time |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: |
| C1 | attention_ddp_2gpu_course | attention | 2 | ddp | 64 | `scripts/train_attention_ddp_scaling.sh` (NGPU=2) | `1289.58 s` |
| C1 | attention_ddp_zero_2gpu_course | attention | 2 | ddp_zero (ZeRO-1) | 64 | `scripts/train_attention_ddp_zero_scaling.sh` (NGPU=2) | `1383.06 s` |
| C2 | attention_fsdp_z2_2gpu_course | attention | 2 | fsdp_z2 | 64 | `scripts/train_attention_fsdp_z2.sh` (NGPU=2) | `1656.88 s` |
| C2 | attention_fsdp_z3_2gpu_course | attention | 2 | fsdp_z3 | 64 | `scripts/train_attention_fsdp_z3.sh` (NGPU=2) | `1821.41 s` |
| C3 | attention_tp2_course | attention | 2 | tp | 64 | `scripts/train_attention_tp.sh` (NGPU=2, tp2 config) | `3031.30 s` |
| C3 | attention_branch_mp_course | attention | 2 | branch_mp | 64 | `scripts/train_attention_branch_mp.sh` | `1836.79 s` |
| C-opt | mamba_ddp_course | mamba | 2 | ddp | 64 | `scripts/train_mamba_ddp.sh` | `1858.84 s` |

## v2 Phase D – 4-GPU Serial Sweep

| Step | Run | Model | GPUs | Strategy | Global Batch | Script | H100 Wall Time |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: |
| D1 | attention_ddp_4gpu_course | attention | 4 | ddp | 64 | `scripts/train_attention_ddp_scaling.sh` (NGPU=4) | `1042.39 s` |
| D2 | attention_ddp_zero_4gpu_course | attention | 4 | ddp_zero (ZeRO-1) | 64 | `scripts/train_attention_ddp_zero_scaling.sh` (NGPU=4) | `1236.86 s` |
| D3 | attention_fsdp_z2_4gpu_course | attention | 4 | fsdp_z2 | 64 | `scripts/train_attention_fsdp_z2.sh` (NGPU=4) | `1486.21 s` |
| D4 | attention_fsdp_z3_4gpu_course | attention | 4 | fsdp_z3 | 64 | `scripts/train_attention_fsdp_z3.sh` (NGPU=4) | `1632.88 s` |
| D5 | attention_tp4_course | attention | 4 | tp | 64 | `scripts/train_attention_tp.sh` (NGPU=4, tp4 config) | `3053.15 s` |
| D6 | attention_hybrid_tp2_dp2_course | attention | 4 | hybrid_tp_dp (TP2xDP2) | 64 | `scripts/train_attention_hybrid_tp2_dp2.sh` | `3378.14 s` |

## v2 Completion Summary

- Planned course runs: **15** (Phase B = 2, Phase C = 7 including optional
  mamba DDP, Phase D = 6).
- Completed cleanly (25/25 epochs, summary + history written): **15**.
- Pending / running: **0**.
- Crashed / partial: **0**.

Full per-run numerical details (best val metrics, throughput, memory, etc.)
live in `report/COURSE_EXPERIMENT_SUMMARY.md` (v2 section) and the auto-generated
`report/artifacts/course_run_table.{csv,json}`.
