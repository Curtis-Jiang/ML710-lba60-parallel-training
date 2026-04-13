# Submission Guide

This file explains how to present this repository as an ML710 final project rather than as a research-side code dump.

## One-Sentence Project Description

This project studies how parallel training changes throughput, runtime, and model quality for one protein-ligand affinity regression workload: `lba60`.

## Recommended Submission Story

Use the following framing in the report or presentation:

1. We intentionally reduced a larger research workspace to one clean workload.
2. We kept only the code required to train and evaluate a single binding-affinity model on `lba60`.
3. We compare single-GPU training, 2-GPU DDP, and packed concurrent jobs under a fixed GPU budget.
4. We analyze throughput, wall-clock time, and accuracy tradeoffs instead of trying to reproduce a full paper benchmark suite.

## What To Submit

The cleanest submission bundle is this folder as-is, plus the final slides/report built from the files below.

Core files to cite:

- `README.md`
- `docs/DATA_GUIDE.md`
- `docs/CODE_ARCHITECTURE.md`
- `docs/REPRODUCIBILITY.md`
- `report/ML710_SUMMARY.md`
- `report/RUN_COMPARISON.md`
- `report/COURSE_REQUIREMENTS_MAPPING.md`

Reference material included for the team:

- `docs/course_reference/ML710 Project Instructions.pdf`
- `docs/course_reference/ML710 Project Grading.pdf`
- `docs/course_reference/ML710_project_presentation_Maksym_Bekuzarov_Dec_2022.pdf`

## Minimum Experiments For A Credible Submission

Already completed in this repo:

- one single-GPU quick run
- one 2-GPU DDP quick run

Strong optional extension:

- one packed-jobs comparison under the same total GPU budget

If time is limited, the current baseline plus DDP evidence is enough to support a solid ML710 story. The packed-jobs launcher remains the best next extension for goodput analysis.

## What Results To Show

At minimum, include:

- one table with runtime and throughput
- one table or figure with best validation Pearson and test Pearson/RMSE
- one short discussion of why DDP speeds training up but can also change convergence when the global batch changes
- one note on goodput or packed-job scheduling, even if only as a planned extension

## Recommended Slide Or Report Structure

1. Problem and workload
2. Why the repo was simplified to one workload
3. Baseline training path
4. DDP training path
5. Runtime and throughput results
6. Accuracy and statistical-efficiency discussion
7. Goodput angle via packed runs
8. Final recommendation under a fixed GPU budget

## Suggested Team Split

If three teammates need clean responsibilities, this repo supports a natural split:

- teammate 1: workload explanation, dataset, and single-GPU baseline
- teammate 2: DDP implementation, scaling, and memory analysis
- teammate 3: packed jobs, goodput discussion, and final report packaging

If the instructor insists on a third clearly non-trivial angle beyond plain DDP, the cleanest extension is a fixed-global-batch DDP control or a small strong-scaling sweep.

## Final Checklist

- README points to the right commands
- smoke validation passes
- run summaries exist in `report/artifacts/`
- final narrative stays on one workload
- report explicitly discusses both throughput and quality
- final slides mention the global-batch caveat in the current quick comparison
