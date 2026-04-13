# Code Architecture

## High-Level View

This repository has four layers:

1. top-level configs and launch scripts
2. a small binding-affinity training engine under `binding_affinity/`
3. the copied processed `lba60` dataset
4. generated run artifacts and reports

## Main Entry Points

Use these first:

- `scripts/train_lba60_single.sh`
  Single-GPU training wrapper.

- `scripts/train_lba60_ddp.sh`
  Multi-GPU DDP training wrapper using `torchrun`.

- `scripts/eval_lba60.sh`
  Evaluate a saved checkpoint.

- `scripts/forward_sanity.py`
  Fast forward-only validation that checks config, data, and model wiring.

- `scripts/launch_lba60_jobs.py`
  Launch multiple independent jobs on disjoint GPU groups for goodput experiments.

## Config Flow

The recommended config is:

- `configs/lba60_quick.yaml`

It inherits from:

- `configs/lba60_base.yaml`
- `binding_affinity/configs/affinity/lba60.yaml`
- `binding_affinity/configs/affinity/train.yaml`
- `binding_affinity/configs/affinity/model_affinity.yaml`

The top-level config decides the course-facing choices:

- processed data location
- run/output directories
- number of epochs
- batch size
- optimizer settings
- regression loss mode

## Training Pipeline

The main training script is:

- `binding_affinity/scripts/train_affinity_model.py`

The pipeline is:

1. load YAML config
2. load processed `train/val/test` tensors
3. build data loaders
4. construct `AffinityModel`
5. train with either single GPU or DDP
6. evaluate on validation every epoch
7. save the best checkpoint by validation Pearson
8. run final test evaluation from the best checkpoint
9. write summaries and metrics to disk

## Model Components

The model is centered around:

- `binding_affinity/src/binding_affinity/models/affinity/model.py`

Important pieces:

- `FAENetEncoder`
  Encodes atom-level graph structure with message passing.

- `EnergyHead`
  Produces per-node contributions and sums them into graph-level energies.

- `HierTokenizer`
  Produces interaction tokens between ligand and protein representations.

- `AffinityRegressor`
  Combines energy difference and token information into the final scalar prediction.

## Forward Logic

For each sample in a batch, the model computes:

- energy of the whole complex
- energy of the protein part
- energy of the ligand part

Then it forms:

- `dE = E_complex - E_protein - E_ligand`

This `dE` term is the physics-inspired core signal. The tokenizer adds an interaction summary on top of it.

## Data Structures

The batched graph container is:

- `binding_affinity/src/binding_affinity/types.py`

The collate logic is:

- `binding_affinity/src/binding_affinity/data/collate.py`

The dataset loader is:

- `binding_affinity/src/binding_affinity/data/processed_dataset.py`

## Utility Modules

- `utils/config.py`
  Recursive config loading with `includes`.

- `utils/ckpt.py`
  Checkpoint save/load helpers.

- `utils/logger.py`
  Writes `metrics.jsonl`.

- `utils/metrics.py`
  Pearson, Spearman, RMSE, and affine calibration helpers.

- `utils/seed.py`
  Determinism and random seed setup.

## Output Layout

Each run produces a directory like:

`runs/affinity/lba60/<run_name>/`

Expected files:

- `config_snapshot.yaml`
- `metrics.jsonl`
- `ckpt_best.pt`
- `summary.json`
- `test_metrics.json`

These files are enough to build the course report without re-parsing stdout.

## What Was Simplified

Compared with the original research project, this standalone repo removes:

- alternate datasets
- preprocessing code paths
- ensemble logic
- MoE logic
- extra benchmark branches

That keeps the codebase small enough for teammates to understand in one sitting.
