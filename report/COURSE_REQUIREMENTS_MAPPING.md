# ML710 Requirements Mapping

## Workload Requirement

Course need:

- one real ML workload that can be parallelized and analyzed

Repo response:

- this project uses one workload, `lba60` protein-ligand affinity regression
- the original research workspace was narrowed to one dataset, one model family, and one processed data snapshot

Status:

- satisfied

## Parallelization Requirement

Course need:

- compare meaningful parallel execution strategies rather than only reporting a final model score

Repo response:

The repo directly supports:

1. single-GPU baseline
2. 2-GPU DDP with `torchrun`
3. packed concurrent jobs on disjoint GPU groups for goodput-style analysis

Status:

- baseline and DDP already run
- packed-job launcher implemented and smoke-checked

## Measurement Requirement

Course need:

- report throughput, runtime, and model-quality consequences of the parallel strategy

Repo response:

Each run already records:

- wall-clock proxy via run timestamps
- `train/examples_per_sec`
- epoch `time_sec`
- peak GPU memory
- best validation Pearson
- test Pearson and RMSE

Status:

- satisfied

## Reproducibility Requirement

Course need:

- teammates and graders should be able to follow the workflow without reverse-engineering a research codebase

Repo response:

This repo now includes:

- `README.md`
- `docs/SUBMISSION_GUIDE.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATA_GUIDE.md`
- `docs/CODE_ARCHITECTURE.md`
- `scripts/smoke_validate.sh`

Status:

- satisfied

## Runtime Budget Requirement

Course need:

- practical runs on A100-class hardware, ideally around one hour or less

Repo response:

- the default submission workload is `configs/lba60_quick.yaml`
- completed H100 runs show about `36` minutes on 1 GPU and about `18.5` minutes on 2 GPUs
- `docs/RUNTIME_NOTES.md` documents the expected A100 slowdown range and how to reduce batch size if needed

Status:

- satisfied for a practical course configuration

## Strongest Submission Framing

The cleanest final story for this repo is:

1. establish a single-GPU control run
2. demonstrate near-linear throughput scaling with 2-GPU DDP
3. discuss the quality tradeoff introduced by a changed global batch
4. present packed concurrent jobs as the goodput-oriented extension under a fixed GPU budget

## If The Team Needs Three Distinct Angles

For a three-person team, the cleanest split is:

- baseline plus instrumentation
- DDP scaling
- packed-job goodput

If the instructor insists that all three angles must be non-trivial beyond baseline, the next best extension is:

- fixed-global-batch DDP by setting `train.batch_size=4` on 2 GPUs

That keeps the project on one workload while making the statistical-efficiency comparison cleaner.

## Current Readiness

This repository is ready to support a course submission now. The baseline and DDP evidence are already present, and the packed-job path is implemented for an optional extension.
