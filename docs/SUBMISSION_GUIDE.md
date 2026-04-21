# Submission Guide

## What This Repo Is For

This repository is the implementation side of the ML710 course project.

The final course deliverable is still slides, but this repo is structured to
make the slide content easy to reproduce:

- one workload
- two model families
- one baseline distributed strategy
- one advanced distributed strategy

## Recommended Submission Narrative

1. Workload:
   binary protein-ligand binding from protein sequence and SMILES inputs
2. Architectures:
   attention and mamba
3. Parallel strategies:
   single GPU, naive DDP, DDP + ZeroRedundancyOptimizer
4. Evaluation:
   throughput, wall-clock time, validation AUC, validation AUPR

The current final results use:

- Attention on `single` and `ddp_zero`
- Mamba with `mamba_ssm` on `single` and `ddp_zero`

## Key Output Files

- `report/RESULTS_SUMMARY.md`
- `report/COURSE_EXPERIMENT_SUMMARY.md`
- `report/COURSE_REQUIREMENTS_MAPPING.md`
- `report/SLIDE_OUTLINE.md`
- `report/artifacts/run_table.csv`

## Reproducibility Entry Point

```bash
pip install -r requirements.txt
bash scripts/install_mamba.sh
bash scripts/smoke_validate.sh
```

`build_compact_dataset.py` is safe to run in a fresh clone because it exits
early when the tracked compact TSV dataset is already present.
