# ML710 Final Project: Parallel Training For LBA60 Affinity Prediction

This repository is a submission-oriented ML710 project built from a larger research workspace and intentionally reduced to one clear workload:

- one task: `lba60`
- one model family: a single graph-based binding-affinity regressor
- one copied processed dataset snapshot
- no MoE, no expert routing, no paper-style multi-dataset benchmark story

For GitHub publication, the large processed tensor files are kept out of git. The repo includes the code, configs, reports, and dataset metadata; the `.pt` split files should be staged locally under `data/processed/lba60_existing_pocket/` before running training.

The repo is packaged around ML systems questions rather than paper reproduction. The main comparison is:

1. single-GPU baseline
2. 2-GPU DDP with `torchrun`
3. packed concurrent jobs under a fixed GPU budget

## Why This Fits ML710

The course project asks for a parallelized ML workload plus analysis of throughput, scaling, and quality tradeoffs. This repo matches that framing because it already includes:

- a single well-defined workload
- one clean single-GPU baseline
- a reproducible DDP path
- a packed multi-job launcher for goodput experiments
- logged throughput, time, memory, validation, and test metrics
- a small processed dataset snapshot so teammates can run the project without rebuilding data

## Current Submission Status

This repo is already organized so it can be shared as a course submission package.

Included in the current folder:

- cleaned training engine under `binding_affinity/`
- course configs under `configs/`
- launch helpers under `scripts/`
- processed `lba60` tensors under `data/processed/lba60_existing_pocket/`
- onboarding and architecture docs under `docs/`
- ML710-facing summaries under `report/`
- copied course reference PDFs under `docs/course_reference/`

Completed experimental results already stored in the repo:

| run | GPUs | global batch | approx wall min | avg ex/s | best val pearson | test pearson | test rmse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lba60_single_quick_seed0` | 1 | 8 | 35.96 | 34.64 | 0.7747 | 0.7758 | 1.3115 |
| `lba60_ddp_quick_seed0` | 2 | 16 | 18.47 | 70.49 | 0.7546 | 0.7684 | 1.3335 |

These results support the core ML710 story:

- DDP improves throughput by about `2.04x`
- DDP reduces total wall time by about `1.95x`
- accuracy remains competitive, but the quick comparison changes the global batch and therefore also changes convergence behavior

## Repo Layout

```text
ML710/
├── configs/                 # smoke / quick / course / packed-job configs
├── data/processed/          # copied lba60 processed tensors
├── binding_affinity/        # minimal affinity-training engine
├── docs/                    # onboarding, architecture, submission guidance
├── outputs/                 # generated evaluation artifacts
├── report/                  # experiment summaries and course-facing notes
├── runs/                    # training runs and checkpoints
└── scripts/                 # single-GPU, DDP, eval, report, smoke tools
```

`lba60` split sizes in the copied dataset:

- train: `3563`
- val: `448`
- test: `452`

## Quick Start

Fastest sanity check:

```bash
python scripts/forward_sanity.py --config configs/lba60_quick.yaml --split train --batch_size 2
```

One-command smoke validation:

```bash
bash scripts/smoke_validate.sh
```

Recommended single-GPU baseline:

```bash
bash scripts/train_lba60_single.sh configs/lba60_quick.yaml lba60_single_seed0 0
```

Recommended 2-GPU DDP run:

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_seed0 0
```

Evaluate the best checkpoint:

```bash
bash scripts/eval_lba60.sh runs/affinity/lba60/lba60_ddp_seed0/ckpt_best.pt test 32 val
```

Dry-run the packed launcher:

```bash
python scripts/launch_lba60_jobs.py --spec configs/packed_seed_sweep.yaml --dry_run
```

## Reproducibility Path

If a teammate wants the shortest route from clone to usable results:

1. Read `docs/SUBMISSION_GUIDE.md`
2. Run `bash scripts/smoke_validate.sh`
3. Reproduce the single-GPU and DDP quick runs
4. Rebuild the comparison report with `python scripts/build_report.py --runs ...`
5. Use `report/ML710_SUMMARY.md` and `report/RUN_COMPARISON.md` in the final presentation/report

## Main Artifacts

Each run writes to `runs/affinity/lba60/<run_name>/` and produces:

- `config_snapshot.yaml`
- `metrics.jsonl`
- `ckpt_best.pt`
- `summary.json`
- `test_metrics.json`
- `stdout.log` for packed-launcher runs

Important logged metrics already available per epoch:

- `train/examples_per_sec`
- `time_sec`
- `cuda/peak_alloc_gb`
- `cuda/peak_reserved_gb`

That means the repo is already instrumented for throughput, scaling, and goodput analysis without adding a second logging system.

## Submission-Oriented Docs

Start here for team sharing and final packaging:

- `docs/SUBMISSION_GUIDE.md`
- `docs/REPRODUCIBILITY.md`
- `docs/TEAM_GUIDE.md`
- `docs/DATA_GUIDE.md`
- `docs/CODE_ARCHITECTURE.md`
- `report/ML710_SUMMARY.md`
- `report/COURSE_REQUIREMENTS_MAPPING.md`
- `docs/course_reference/README.md`

## Environment

This repo assumes a working PyTorch CUDA environment already exists on the cluster. The dependency list is in `requirements.txt`.

For the training path on processed tensors, the key runtime pieces are:

- `torch`
- `numpy`
- `scipy`
- `pyyaml`

Preprocessing dependencies remain listed only because the vendored engine still contains a small amount of related utility code, even though preprocessing is not part of the ML710 workflow.
