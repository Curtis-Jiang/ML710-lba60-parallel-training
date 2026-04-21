# Data Guide

## Task

The rebuilt ML710 project uses binary protein-ligand binding classification.

- input A: protein sequence
- input B: SMILES string
- label: `0` or `1`

## Compact Dataset

The runtime dataset lives at:

- `data/datasets/pubchem_course`

It contains:

- `train.tsv`
- `val.tsv`
- `protein_vocab.json`
- `smiles_vocab.json`
- `dataset_meta.json`

Each TSV row has:

- `sample_id`
- `protein_sequence`
- `smiles`
- `label`

## Representation

- Protein and SMILES are stored as plain strings in TSV rows.
- Training tokenizes both branches at character level.
- The runtime vocabulary uses:
  - `PAD=0`
  - `UNK=1`

## Split Sizes

- Course split:
  - `100,000` train rows
  - `10,000` val rows
- Smoke runs use the same files but limit the number of train/eval batches in
  config.

The compact dataset is already tracked in this repository, so teammates do not
need to build it before running experiments.

If you want to rebuild it from the original external source files, use:

```bash
python scripts/build_compact_dataset.py
```

For the current final project version, the tracked TSV dataset is the default
and recommended path.
