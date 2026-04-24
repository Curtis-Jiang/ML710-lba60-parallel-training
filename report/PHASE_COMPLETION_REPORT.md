# Phase Completion Report

Generated from `runs/*_course/summary.json` after running Phases B, C, D.

Legend: `OK` = summary.json + 25 epochs of history. `MISSING` = run directory not found.

## Phase B – Single-GPU baseline (per-run target ~45 min)

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_single` | `runs/attention_single_course` | 1 | OK | 25 | 1985 | 1292 | 3.23 | 0.9443 | 0.6780 | 0.5733 |
| `mamba_single` | `runs/mamba_single_course` | 1 | OK | 25 | 3258 | 788 | 4.83 | 0.9568 | 0.7555 | 0.6599 |

## Phase C Round 1 – DDP + ZeRO-1 (2-GPU, paired)

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_ddp_scaling` | `runs/attention_ddp_2gpu_course` | 2 | OK | 25 | 1290 | 1980 | 1.80 | 0.9144 | 0.5697 | 0.4692 |
| `attention_ddp_zero_scaling` | `runs/attention_ddp_zero_2gpu_course` | 2 | OK | 25 | 1383 | 1845 | 1.74 | 0.9288 | 0.6336 | 0.4831 |

## Phase C Round 2 – ZeRO-2 + ZeRO-3 (2-GPU, paired)

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_fsdp_z2` | `runs/attention_fsdp_z2_2gpu_course` | 2 | OK | 25 | 1657 | 1534 | 1.64 | 0.9258 | 0.6017 | 0.4559 |
| `attention_fsdp_z3` | `runs/attention_fsdp_z3_2gpu_course` | 2 | OK | 25 | 1821 | 1394 | 1.62 | 0.9416 | 0.6823 | 0.5664 |

## Phase C Round 3 – TP-2 + branch_mp (2-GPU, paired)

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention_tp (tp2)` | `runs/attention_tp2_course` | 2 | OK | 25 | 3031 | 840 | 2.51 | 0.8933 | 0.4861 | 0.3589 |
| `attention_branch_mp` | `runs/attention_branch_mp_course` | 2 | OK | 25 | 1837 | 1401 | 2.64 | 0.9430 | 0.6861 | 0.5703 |

## Phase C Mamba – 2-GPU DDP (optional / conditional)

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mamba_ddp (2-GPU)` | `runs/mamba_ddp_course` | 2 | OK | 25 | 1859 | 1378 | 2.65 | 0.9559 | 0.7488 | 0.6659 |

## Phase D – 4-GPU serial

| Script / label | Run dir | GPUs | Status | Epochs | Wall (s) | ex/s | Peak GB | Val AUC | Val AUPR | Val F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D1 ddp_scaling (NGPU=4)` | `runs/attention_ddp_4gpu_course` | 4 | OK | 25 | 1042 | 2433 | 1.06 | 0.8831 | 0.4818 | 0.2493 |
| `D2 ddp_zero_scaling (NGPU=4)` | `runs/attention_ddp_zero_4gpu_course` | 4 | OK | 25 | 1237 | 2049 | 0.97 | 0.9383 | 0.6692 | 0.5580 |
| `D3 fsdp_z2 (NGPU=4)` | `runs/attention_fsdp_z2_4gpu_course` | 4 | OK | 25 | 1486 | 1700 | 0.85 | 0.9369 | 0.6580 | 0.5249 |
| `D4 fsdp_z3 (NGPU=4)` | `runs/attention_fsdp_z3_4gpu_course` | 4 | OK | 25 | 1633 | 1545 | 0.82 | 0.9377 | 0.6585 | 0.5528 |
| `D5 tp (NGPU=4 / tp4)` | `runs/attention_tp4_course` | 4 | OK | 25 | 3053 | 833 | 2.17 | 0.8744 | 0.4550 | 0.3241 |
| `D6 hybrid_tp2_dp2 (4-GPU)` | `runs/attention_hybrid_tp2_dp2_course` | 4 | OK | 25 | 3378 | 746 | 1.35 | 0.8506 | 0.4105 | 0.1509 |


## Scaling Analysis – Attention (baseline = attention_single_course)

- Baseline throughput = `1292.4 ex/s`, baseline Val AUC = `0.9443`, wall = `1985 s`

| Strategy | GPUs | ex/s | Speedup | Scaling efficiency | Wall (s) | Wall speedup | Peak GB | Val AUC | dAUC vs single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single baseline | 1 | 1292 | 1.00x | 100% | 1985 | 1.00x | 3.23 | 0.9443 | +0.0000 |
| DDP 2-GPU | 2 | 1980 | 1.53x | 77% | 1290 | 1.54x | 1.80 | 0.9144 | -0.0299 |
| DDP 4-GPU | 4 | 2433 | 1.88x | 47% | 1042 | 1.90x | 1.06 | 0.8831 | -0.0612 |
| ZeRO-1 2-GPU | 2 | 1845 | 1.43x | 71% | 1383 | 1.44x | 1.74 | 0.9288 | -0.0155 |
| ZeRO-1 4-GPU | 4 | 2049 | 1.59x | 40% | 1237 | 1.61x | 0.97 | 0.9383 | -0.0060 |
| FSDP ZeRO-2 2-GPU | 2 | 1534 | 1.19x | 59% | 1657 | 1.20x | 1.64 | 0.9258 | -0.0185 |
| FSDP ZeRO-2 4-GPU | 4 | 1700 | 1.32x | 33% | 1486 | 1.34x | 0.85 | 0.9369 | -0.0074 |
| FSDP ZeRO-3 2-GPU | 2 | 1394 | 1.08x | 54% | 1821 | 1.09x | 1.62 | 0.9416 | -0.0027 |
| FSDP ZeRO-3 4-GPU | 4 | 1545 | 1.20x | 30% | 1633 | 1.22x | 0.82 | 0.9377 | -0.0066 |
| TP 2-GPU | 2 | 840 | 0.65x | 33% | 3031 | 0.65x | 2.51 | 0.8933 | -0.0510 |
| TP 4-GPU | 4 | 833 | 0.64x | 16% | 3053 | 0.65x | 2.17 | 0.8744 | -0.0699 |
| branch_mp 2-GPU | 2 | 1401 | 1.08x | 54% | 1837 | 1.08x | 2.64 | 0.9430 | -0.0013 |
| hybrid TP2xDP2 4-GPU | 4 | 746 | 0.58x | 14% | 3378 | 0.59x | 1.35 | 0.8506 | -0.0937 |

## Key Observations

- **Best absolute throughput**: `attention_ddp_4gpu_course` at 2,433 ex/s (1.88x vs baseline).
- **Highest Val AUC among parallel runs**: `attention_branch_mp_course` 0.9430 and `attention_fsdp_z3_2gpu_course` 0.9416, effectively matching the 0.9443 single-GPU baseline.
- **Memory savers**: ZeRO-3 4-GPU uses only `0.82 GB` peak rank-0 memory vs `3.23 GB` for single GPU (-75%).
- **TP has the highest compute overhead**: tp2 and tp4 are the slowest attention configs (840 / 833 ex/s) due to all-gather and collective traffic dominating the relatively small 15.6 M-param model.
- **Hybrid TP2xDP2** (4-GPU) is also throughput-limited (746 ex/s) and its best-epoch arrived early (ep 13), so Val AUC 0.8506 lags the other 4-GPU runs; more epochs/LR tuning would be needed for this config.
- **Mamba single vs Attention single**: Mamba reaches Val AUC `0.9568` (vs `0.9443`) at `788 ex/s` (vs `1,292 ex/s`). Mamba is the heavier, higher-quality workload, as expected.

## Completeness Audit

- Executed course runs: **15 / 15** planned.
- `mamba_ddp_course` (Phase C conditional mamba 2-GPU DDP) has been backfilled via `bash scripts/train_mamba_ddp.sh` and is now included.
- All 15 executed runs wrote `summary.json` + 25 lines of `history.jsonl`, so no crashed/partial runs.

## Regenerated Report Artifacts

`scripts/build_report.py` was re-executed so these files are up to date with every `runs/*/summary.json` (including all smoke and all course runs):

- `report/RESULTS_SUMMARY.md` — full 27-row results table (smoke + course).
- `report/artifacts/run_table.json` / `.csv` — every run.
- `report/artifacts/course_run_table.json` / `.csv` — course runs only.
- `report/artifacts/convergence_curves.json` — per-epoch `train_loss`, `val_auc`, `val_aupr`, `peak_gpu_bytes` for every run.
- `report/artifacts/goodput_table.json` / `.csv` — scaling efficiency and goodput vs each model's single-GPU baseline.

