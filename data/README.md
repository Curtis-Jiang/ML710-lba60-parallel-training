# Data

The active runtime data for this project lives under:

- `data/datasets/pubchem_course/train.tsv`
- `data/datasets/pubchem_course/val.tsv`
- `data/datasets/pubchem_course/protein_vocab.json`
- `data/datasets/pubchem_course/smiles_vocab.json`
- `data/datasets/pubchem_course/dataset_meta.json`

These files define the actual course dataset used by training and evaluation.
They are compact enough to share with teammates directly.

The TSV schema is:

- `sample_id`
- `protein_sequence`
- `smiles`
- `label`

To rebuild the compact dataset from the external source files:

```bash
python scripts/build_compact_dataset.py
```
