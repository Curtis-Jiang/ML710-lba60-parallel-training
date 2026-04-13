# ML710 Sequence Binding Parallel Training

This repository is a course-focused rebuild of the ML710 project around a
single supervised workload: binary protein-ligand binding classification from
protein sequences and SMILES strings.

The project is optimized for distributed training clarity rather than model
novelty. It provides:

- one clean workload based on a compact PubChem course subset
- two model families: `attention` and `mamba`
- three training strategies: `single`, `ddp`, and `ddp_zero`
- deterministic compact-dataset building
- smoke validation and report artifact generation

## Project Layout

- `src/sequence_binding/`: data pipeline, models, training engine
- `configs/`: course and smoke configs for attention and mamba
- `scripts/`: dataset builder, train/eval entrypoints, launcher shells
- `data/datasets/pubchem_course/`: compact tracked TSV dataset
- `docs/`: teammate-facing guides and course-reference PDFs
- `report/`: report markdowns and generated artifacts

## Workload

- Input:
  - protein sequence characters
  - SMILES characters
- Label:
  - binary bind / non-bind
- Main dataset:
  - `data/datasets/pubchem_course`
- Course split sizes:
  - `100,000` train rows
  - `10,000` val rows
- Default truncation:
  - protein: `512`
  - SMILES: `128`
- Tokenization:
  - character-level for both branches
  - `PAD=0`, `UNK=1`

The compact dataset is intended to stay Git-friendly:

- `train.tsv` is about `63 MB`
- `val.tsv` is about `6 MB`
- vocab and metadata JSON files are tiny
- each file stays below GitHub's `100 MB` hard upload limit

## Quick Start

1. Install the default dependencies:

```bash
pip install -r requirements.txt
```

2. Build the compact dataset:

```bash
python scripts/build_compact_dataset.py
```

3. Train the attention baseline on one GPU:

```bash
bash scripts/train_attention_single.sh
```

4. Train the attention model with 2-GPU DDP:

```bash
bash scripts/train_attention_ddp.sh
```

5. Train the advanced 2-GPU DDP+ZeRO run:

```bash
bash scripts/train_attention_ddp_zero.sh
```

6. Train the mamba model:

```bash
bash scripts/train_mamba_single.sh
bash scripts/train_mamba_ddp.sh
```

7. Run the end-to-end smoke suite:

```bash
bash scripts/smoke_validate.sh
```

## Notes

- Runtime training and evaluation use only the compact TSV dataset. The
  original large source bank is used only as a one-time external conversion
  source by `scripts/build_compact_dataset.py`.
- The compact dataset is committed in this repository, so a teammate can clone
  or pull the repo and run training without fetching any external bank files.
- The mamba model uses `mamba-ssm` when it is available. If the package is not
  installed, the implementation falls back to a lightweight gated sequence
  block so the project remains runnable for smoke validation.
- If you want the official Mamba backend instead of the fallback block, install
  it separately with `pip install mamba-ssm`.
- `naive DDP` is included as a baseline because it is required for the course
  comparisons. `ddp_zero` is included as the advanced distributed strategy in
  this version of the project.
