# Data Guide

## What The Dataset Is

This project uses a preprocessed copy of the `ATOM3D-LBA` sequence-identity-60 split, referred to in the repo as `lba60`.

The prediction target is a real-valued protein-ligand binding affinity score:

- one sample = one protein-ligand complex
- one label = one scalar affinity target `y`
- one split = `train`, `val`, or `test`

In this standalone ML710 project, we do **not** read raw LMDB data during training. We only use the copied processed tensors under:

`data/processed/lba60_existing_pocket/`

For GitHub publication, the large `.pt` split files are intentionally not tracked in git because they exceed practical repository limits. The metadata JSON remains in the repo, and teammates should place the three split tensors back under the same directory when running experiments locally.

## Split Sizes

- train: `3563`
- val: `448`
- test: `452`

## Why Processed Data Is Used

The original research workspace contained preprocessing code and several alternate dataset variants. For the course project, that would add unnecessary complexity. We therefore keep only one processed snapshot so teammates can focus on:

- training
- scaling
- throughput
- result analysis

instead of graph construction and data conversion.

## File Format

The key files are:

- `lba60_train.pt`
- `lba60_val.pt`
- `lba60_test.pt`
- `lba60_preprocess_meta.json`

Each `.pt` file stores a dictionary with:

- `samples`: a Python list of examples
- `meta`: metadata about how the split was prepared

The JSON file stores a compact summary for all three splits.

## Example Sample Structure

Each sample contains:

- `id`: complex identifier such as `5ey0`
- `y`: scalar regression label
- `complex`: graph for the full complex
- `protein`: graph for the protein pocket
- `ligand`: graph for the ligand

Each graph contains at least:

- `z`: integer node identifiers
- `pos`: node coordinates
- `edge_index`: graph edges
- `fa_t`: frame-averaging translation
- `fa_R`: frame-averaging rotations

The `complex` graph also includes:

- `is_ligand`: boolean mask that marks ligand nodes inside the combined graph

## Size Statistics

From `lba60_preprocess_meta.json`:

- average complex nodes in train: about `244`
- average protein nodes in train: about `219`
- average ligand nodes in train: about `26`
- average complex edges in train: about `4408`

This is large enough to make batching, data loading, and multi-GPU scaling meaningful for ML710.

## How The Training Code Uses The Data

The active data path is:

1. `ProcessedLBADataset` reads one `.pt` split file.
2. `collate_lba_samples` converts a list of samples into a batched dictionary.
3. The model consumes three batched graphs: `complex`, `protein`, and `ligand`.
4. The script writes outputs to `runs/affinity/lba60/<run_name>/`.

Relevant files:

- `binding_affinity/src/binding_affinity/data/processed_dataset.py`
- `binding_affinity/src/binding_affinity/data/collate.py`
- `binding_affinity/src/binding_affinity/types.py`

## Sanity Check Commands

Fast structural sanity check:

```bash
python scripts/forward_sanity.py --config configs/lba60_quick.yaml --split train --batch_size 2
```

Training smoke test:

```bash
bash scripts/train_lba60_single.sh configs/lba60_smoke.yaml lba60_smoke 0
```

## What Was Intentionally Removed

For clarity, this course repo no longer includes raw-dataset training paths or alternate processed dataset families from the original research workspace. That is intentional. The repo is designed to be easy to use for one workload, not to preserve every original experiment branch.
