# Team Guide

## Goal

This repo is a simplified ML710 project centered on one workload:

- task: `lba60`
- objective: regression
- comparison: single GPU vs DDP vs packed concurrent jobs

If you are new to the repo, start here instead of reading the whole codebase.

## First Things To Read

1. `README.md`
2. `docs/SUBMISSION_GUIDE.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/DATA_GUIDE.md`
5. `docs/CODE_ARCHITECTURE.md`
6. `docs/RUNTIME_NOTES.md`

## First Command To Run

Use the fastest sanity check:

```bash
bash scripts/smoke_validate.sh
```

If that works, the model, dataset, launcher, short training path, and evaluation path are all wired correctly.

## Typical Workflow

### 1. Run a single-GPU baseline

```bash
bash scripts/train_lba60_single.sh configs/lba60_quick.yaml lba60_single_seed0 0
```

### 2. Run a DDP comparison

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_seed0 0
```

### 3. Evaluate a checkpoint

```bash
bash scripts/eval_lba60.sh runs/affinity/lba60/lba60_ddp_seed0/ckpt_best.pt test 32 val
```

### 4. Compare results

Look at:

- `summary.json`
- `test_metrics.json`
- `metrics.jsonl`

inside each run directory.

## Where Things Live

- configs: `configs/`
- launch scripts: `scripts/`
- training core: `binding_affinity/scripts/train_affinity_model.py`
- model code: `binding_affinity/src/binding_affinity/models/affinity/`
- data copy: `data/processed/lba60_existing_pocket/`
- outputs: `runs/affinity/lba60/`
- final shared writeups: `report/`

## Common Questions

### Why only one dataset?

Because ML710 is about parallelizing an ML workload, not reproducing every benchmark from the original research project.

### Why is preprocessing not part of the workflow?

Because the processed snapshot is already copied into this repo. That keeps the project easy to run and explain.

### What should we report?

At minimum:

- runtime
- throughput
- validation Pearson
- test Pearson
- test RMSE
- GPU memory usage

### Which config should we use?

Use `configs/lba60_quick.yaml` unless you are intentionally doing a smoke test or a longer final run.

### Which docs matter most for the final deliverable?

Use these first:

- `docs/SUBMISSION_GUIDE.md`
- `docs/REPRODUCIBILITY.md`
- `report/ML710_SUMMARY.md`
- `report/COURSE_REQUIREMENTS_MAPPING.md`

## What To Avoid

- do not add back the MoE/expert framing
- do not mix in other datasets unless the team explicitly decides to expand scope
- do not edit the processed tensors in place
- do not compare runs with different configs unless you clearly document the difference

## Course Reference Material

The original ML710 course documents are copied here:

- `docs/course_reference/ML710 Project Instructions.pdf`
- `docs/course_reference/ML710 Project Grading.pdf`
- `docs/course_reference/ML710_project_presentation_Maksym_Bekuzarov_Dec_2022.pdf`

These are useful when deciding what results and figures belong in the final presentation.
