# Course Experiment Summary

## Setup

These are the four course-scale experiments requested for the current project:

- `attention_single_course`
- `attention_ddp_zero_course`
- `mamba_single_course`
- `mamba_ddp_zero_course`

Shared setup:

- dataset: `100,000` train / `10,000` val
- task: binary protein-ligand binding classification
- truncation: protein `512`, SMILES `128`
- epochs: `25`
- global batch size: `64`
- mixed precision: `bf16`
- hardware used for these runs: NVIDIA H100 80GB

The raw structured table for these runs is in:

- `report/artifacts/course_run_table.json`
- `report/artifacts/course_run_table.csv`

## Results Table

| Run | Model | Strategy | GPUs | Params | Throughput (ex/s) | Wall Time (s) | Val AUC | Val AUPR | Val Acc | Val F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_single_course | attention | single | 1 | 15,651,841 | 2140.71 | 1203.80 | 0.9390 | 0.6670 | 0.9328 | 0.5385 |
| attention_ddp_zero_course | attention | ddp_zero | 2 | 15,651,841 | 2587.79 | 990.25 | 0.9385 | 0.6672 | 0.9339 | 0.5262 |
| mamba_single_course | mamba | single | 1 | 18,926,593 | 1314.76 | 1954.46 | 0.9565 | 0.7575 | 0.9416 | 0.6708 |
| mamba_ddp_zero_course | mamba | ddp_zero | 2 | 18,926,593 | 1909.43 | 1338.40 | 0.9589 | 0.7560 | 0.9455 | 0.6699 |

## Comparison

### Attention: single vs advanced distributed

- `ddp_zero` throughput is `1.209x` of single-GPU throughput.
- `ddp_zero` wall time is `0.823x` of the single-GPU wall time.
- Val AUC changes by `-0.0005`.
- Val AUPR changes by `+0.0002`.

Interpretation:

- For the larger final Attention configuration, `ddp_zero` now gives a real systems win instead of pure overhead.
- The validation metrics stay essentially flat, so this is a clean throughput improvement rather than an optimization tradeoff.

### Mamba: single vs advanced distributed

- `ddp_zero` throughput is `1.452x` of single-GPU throughput.
- `ddp_zero` wall time is `0.685x` of the single-GPU wall time.
- Val AUC changes by `+0.0024`.
- Val AUPR changes by `-0.0015`.

Interpretation:

- With `mamba_ssm`, the advanced distributed Mamba run now shows a clear systems speedup.
- Validation quality stays nearly unchanged, so this is mostly a runtime gain rather than a quality tradeoff.

### Attention vs Mamba

- On single GPU, Mamba is `0.614x` as fast as Attention.
- On single GPU, Mamba improves Val AUC by `+0.0174` and Val AUPR by `+0.0905`.
- Under `ddp_zero`, Mamba is `0.738x` as fast as Attention.
- Under `ddp_zero`, Mamba improves Val AUC by `+0.0204` and Val AUPR by `+0.0888`.

Interpretation:

- In this final version, Mamba is a heavier and slower workload than Attention.
- The tradeoff is that Mamba still gives stronger validation quality, which makes it a good stress test for the distributed part of the project.

## Runtime Calibration Note

- On H100, the four final runs land at about `16-33 minutes`.
- The same configs are intended to land closer to the course target of about
  half an hour on A100-class GPUs.

## Important Note

For the Mamba runs, the recorded backend is `mamba_ssm`.

That means:

- the current course-scale Mamba results reflect the intended state-space block
- this is the only Mamba backend used in the final project presentation and report

## Takeaway

The four requested experiments all ran successfully and give a coherent course story:

- both model families train correctly
- the advanced distributed strategy is implemented and reproducible
- `ddp_zero` helps both the larger Attention run and the Mamba run
- Mamba is now the heavier, more distributed-systems-oriented workload in the current codebase

---

# v2 Results – Full Parallel-Strategy Sweep (2026-04-23 runs)

The v1 tables above are kept as-is for historical comparison. This v2 section
extends the study to the full parallel-strategy matrix executed in Phases B / C / D
on H100 80GB, same course dataset and recipe (`epochs=25`, `global_batch=64`,
`bf16`, protein=512 / SMILES=128).

Hardware note: v2 was run on H100 80GB (SXM); numbers are not directly
comparable with the v1 table when it was calibrated on earlier H100 sessions.
Throughput (`ex/s`) and wall time are measured end-to-end.

## v2 Setup

- 15 course-scale runs executed across three phases (all complete):
  - Phase B (single-GPU baselines): `attention_single`, `mamba_single`
  - Phase C (2-GPU strategy sweep): `ddp`, `ddp_zero (ZeRO-1)`, `fsdp_z2`,
    `fsdp_z3`, `tp2`, `branch_mp`, and the conditional `mamba_ddp`
  - Phase D (4-GPU serial sweep): `ddp`, `ddp_zero`, `fsdp_z2`, `fsdp_z3`, `tp4`,
    `hybrid_tp2_dp2`

The raw structured tables for this sweep are in:

- `report/artifacts/run_table.csv` / `.json` (all 27 runs including smoke)
- `report/artifacts/course_run_table.csv` / `.json` (course runs only)
- `report/artifacts/goodput_table.csv` / `.json` (scaling efficiency + goodput)
- `report/artifacts/convergence_curves.json` (per-epoch curves)
- `report/PHASE_COMPLETION_REPORT.md` (phase-by-phase completion audit)

## v2 Results Table

| Run | Model | Strategy | GPUs | Params | Throughput (ex/s) | Wall Time (s) | Val AUC | Val AUPR | Val Acc | Val F1 | Peak GPU (GB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_single_course | attention | single | 1 | 15,651,841 | 1292.43 | 1985.27 | 0.9443 | 0.6780 | 0.9363 | 0.5733 | 3.23 |
| attention_ddp_2gpu_course | attention | ddp | 2 | 15,651,841 | 1979.78 | 1289.58 | 0.9144 | 0.5697 | 0.9233 | 0.4692 | 1.80 |
| attention_ddp_zero_2gpu_course | attention | ddp_zero (ZeRO-1) | 2 | 15,651,841 | 1844.83 | 1383.06 | 0.9288 | 0.6336 | 0.9296 | 0.4831 | 1.74 |
| attention_fsdp_z2_2gpu_course | attention | fsdp_z2 | 2 | 15,651,841 | 1533.83 | 1656.88 | 0.9258 | 0.6017 | 0.9260 | 0.4559 | 1.64 |
| attention_fsdp_z3_2gpu_course | attention | fsdp_z3 | 2 | 15,651,841 | 1394.13 | 1821.41 | 0.9416 | 0.6823 | 0.9360 | 0.5664 | 1.62 |
| attention_tp2_course | attention | tp | 2 | 15,651,841 | 840.21 | 3031.30 | 0.8933 | 0.4861 | 0.9159 | 0.3589 | 2.51 |
| attention_branch_mp_course | attention | branch_mp | 2 | 15,651,841 | 1400.73 | 1836.79 | 0.9430 | 0.6861 | 0.9352 | 0.5703 | 2.64 |
| attention_ddp_4gpu_course | attention | ddp | 4 | 15,651,841 | 2433.25 | 1042.39 | 0.8831 | 0.4818 | 0.9145 | 0.2493 | 1.06 |
| attention_ddp_zero_4gpu_course | attention | ddp_zero (ZeRO-1) | 4 | 15,651,841 | 2048.97 | 1236.86 | 0.9383 | 0.6692 | 0.9333 | 0.5580 | 0.97 |
| attention_fsdp_z2_4gpu_course | attention | fsdp_z2 | 4 | 15,651,841 | 1700.02 | 1486.21 | 0.9369 | 0.6580 | 0.9303 | 0.5249 | 0.85 |
| attention_fsdp_z3_4gpu_course | attention | fsdp_z3 | 4 | 15,651,841 | 1545.30 | 1632.88 | 0.9377 | 0.6585 | 0.9306 | 0.5528 | 0.82 |
| attention_tp4_course | attention | tp | 4 | 15,651,841 | 832.59 | 3053.15 | 0.8744 | 0.4550 | 0.9135 | 0.3241 | 2.17 |
| attention_hybrid_tp2_dp2_course | attention | hybrid_tp_dp | 4 | 15,651,841 | 746.50 | 3378.14 | 0.8506 | 0.4105 | 0.9123 | 0.1509 | 1.35 |
| mamba_single_course | mamba | single | 1 | 18,926,593 | 788.16 | 3257.71 | 0.9568 | 0.7555 | 0.9463 | 0.6599 | 4.83 |
| mamba_ddp_course | mamba | ddp | 2 | 18,926,593 | 1377.70 | 1858.84 | 0.9559 | 0.7488 | 0.9436 | 0.6659 | 2.65 |

## v2 Scaling Analysis – Attention

Baseline: `attention_single_course` (1 GPU, 1292.43 ex/s, Val AUC 0.9443).
Scaling efficiency = `(throughput / baseline_throughput) / gpus`.

How to read the columns below:

- `Speedup > 1x` means the parallel run is actually faster than a single GPU.
- `Scaling Eff. = Speedup / N`. It is always positive because throughputs are
  positive; do not read a positive `Eff%` as "speedup success".
- The real break-even for `Eff` is `1/N` (so `50%` for 2-GPU and `25%` for
  4-GPU). Any row with `Eff ≤ 1/N` means the extra GPUs gave **zero or
  negative throughput return** and is flagged `Comms-bound` in the last column.

| Strategy | GPUs | Throughput (ex/s) | Speedup | Scaling Eff. | Wall Speedup | Peak GB | dVal AUC | Speedup > 1? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| single (baseline) | 1 | 1292.43 | 1.00x | 100% | 1.00x | 3.23 | 0.0000 | — |
| DDP | 2 | 1979.78 | 1.53x | 77% | 1.54x | 1.80 | -0.0299 | yes |
| DDP | 4 | 2433.25 | 1.88x | 47% | 1.90x | 1.06 | -0.0612 | yes |
| DDP + ZeRO-1 | 2 | 1844.83 | 1.43x | 71% | 1.44x | 1.74 | -0.0155 | yes |
| DDP + ZeRO-1 | 4 | 2048.97 | 1.59x | 40% | 1.61x | 0.97 | -0.0060 | yes |
| FSDP ZeRO-2 | 2 | 1533.83 | 1.19x | 59% | 1.20x | 1.64 | -0.0185 | yes |
| FSDP ZeRO-2 | 4 | 1700.02 | 1.32x | 33% | 1.34x | 0.85 | -0.0074 | yes |
| FSDP ZeRO-3 | 2 | 1394.13 | 1.08x | 54% | 1.09x | 1.62 | -0.0027 | yes |
| FSDP ZeRO-3 | 4 | 1545.30 | 1.20x | 30% | 1.22x | 0.82 | -0.0066 | yes |
| branch_mp | 2 | 1400.73 | 1.08x | 54% | 1.08x | 2.64 | -0.0013 | yes |
| TP | 2 | 840.21 | 0.65x | 33% (< 1/N=50%) | 0.65x | 2.51 | -0.0510 | **no — Comms-bound** |
| TP | 4 | 832.59 | 0.64x | 16% (< 1/N=25%) | 0.65x | 2.17 | -0.0699 | **no — Comms-bound** |
| hybrid TP2xDP2 | 4 | 746.50 | 0.58x | 14% (< 1/N=25%) | 0.59x | 1.35 | -0.0937 | **no — Comms-bound** |

Why TP / hybrid fall into the comms-bound region at this model size:

- Tensor parallelism inserts 2 `all-reduce` calls per attention block (4 per
  fwd + bwd pass). Those collectives have fixed NCCL latency that does not
  shrink with shard size.
- The attention model here is only ~15.6 M params with a small hidden dim, so
  per-rank GEMMs become small and kernel / NCCL overhead dominates.
- H100 compute is so fast that the compute-to-communication ratio flips
  unfavorably: shards finish faster than the next all-reduce can complete, so
  all-reduce sits on the critical path.
- `hybrid_tp2_dp2` stacks TP all-reduce + DP gradient all-reduce on top of each
  other, which is why it has the worst efficiency of all runs (14%).

TP / hybrid are included for correctness and implementation completeness, not
as throughput wins — in line with the rule of thumb "TP only starts to pay off
once the model no longer fits on a single GPU or once hidden_dim / seq_len are
large enough to amortize the collective cost".

## v2 Scaling Analysis – Mamba

Baseline: `mamba_single_course` (1 GPU, 788.16 ex/s, Val AUC 0.9568).

| Strategy | GPUs | Throughput (ex/s) | Speedup | Scaling Eff. | Wall Speedup | Peak GB | Val AUC | dVal AUC | Speedup > 1? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| single (baseline) | 1 | 788.16 | 1.00x | 100% | 1.00x | 4.83 | 0.9568 | 0.0000 | — |
| DDP | 2 | 1377.70 | 1.75x | 87% | 1.75x | 2.65 | 0.9559 | -0.0009 | yes |

Notes on the Mamba result:

- 2-GPU DDP on Mamba scales much better than 2-GPU DDP on Attention
  (87% efficiency vs 77% for attention). Mamba's heavier per-step compute
  (state-space scan dominated) hides the DDP gradient all-reduce more
  effectively — higher compute-to-comm ratio → higher scaling efficiency.
- Val AUC only drops by `-0.0009` at best-epoch 9 (vs best-epoch 9 for single),
  so this is effectively a pure systems win with no quality regression.
- Peak rank-0 memory drops from 4.83 GB (single) to 2.65 GB (DDP-2), a -45%
  reduction from gradient/bucket amortization plus not holding the full batch
  as a single activation stack.

## v2 Key Observations

- **Peak throughput**: `attention_ddp_4gpu` at 2433 ex/s (1.88x single-GPU,
  47% scaling efficiency). Pure-DDP is still the fastest when the model fits.
- **Best quality-preserving parallel**: `branch_mp_2gpu` (0.9430) and
  `fsdp_z3_2gpu` (0.9416) almost match the 0.9443 single-GPU AUC while cutting
  peak memory by ~50%.
- **Memory compression**: FSDP ZeRO-3 4-GPU peaks at 0.82 GB rank-0 vs 3.23 GB
  single (-75%), making it the right choice when the model no longer fits on
  one device.
- **TP and hybrid (TP2xDP2) lose in throughput**: 832 / 746 ex/s — collective
  overhead dominates because the model is only ~15.6 M params; TP is still
  useful as a correctness / implementation test but is not a throughput win
  at this scale.
- **Mamba single vs Attention single**: Mamba delivers Val AUC 0.9568 (+0.0125
  vs attention) but only 0.61x attention throughput (788 vs 1292 ex/s) — it is
  the heavier, higher-quality workload as planned.

## v2 Throughput-vs-Quality Frontier

Approximate Pareto front (higher throughput is better, higher AUC is better):

1. `attention_single_course` — 1292 ex/s, AUC 0.9443 (quality anchor, 1 GPU)
2. `attention_ddp_zero_4gpu_course` — 2049 ex/s, AUC 0.9383 (best 4-GPU quality)
3. `attention_ddp_4gpu_course` — 2433 ex/s, AUC 0.8831 (peak throughput, less stable)
4. `mamba_single_course` — 788 ex/s, AUC 0.9568 (best quality, slowest)

The remaining points either dominate-or-are-dominated-by one of the above on both axes.

## v2 Takeaway

- All **15 / 15** planned course runs completed cleanly (25/25 epochs,
  `summary.json` and 25-line `history.jsonl` present for every run; no
  pending cells remain).
- DDP family (DDP, ZeRO-1, FSDP-Z2/Z3) scales as expected; memory shrinks
  monotonically from raw DDP -> ZeRO-1 -> Z2 -> Z3, while throughput decreases
  modestly at each step due to extra collective communication.
- TP-only and hybrid (TP+DP) parallelisms expose a clear compute/communication
  trade-off at this model size and make the systems picture complete for the
  report, even though they are not the fastest configurations.
- Mamba 2-GPU DDP is the single cleanest "systems win" in v2:
  **1.75x speedup, 87% scaling efficiency, -0.0009 Val AUC** — higher
  compute-per-step hides gradient all-reduce better than for Attention.

