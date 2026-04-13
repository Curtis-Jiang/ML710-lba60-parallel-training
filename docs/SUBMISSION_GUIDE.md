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

## Key Output Files

- `report/RESULTS_SUMMARY.md`
- `report/COURSE_REQUIREMENTS_MAPPING.md`
- `report/SLIDE_OUTLINE.md`
- `report/artifacts/run_table.csv`

## Reproducibility Entry Point

```bash
python scripts/build_compact_dataset.py
bash scripts/smoke_validate.sh
```
