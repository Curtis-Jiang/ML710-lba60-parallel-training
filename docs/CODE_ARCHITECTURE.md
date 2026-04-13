# Code Architecture

## Package Layout

- `src/sequence_binding/data/`
  - TSV dataset loading
  - char tokenization
  - dataset and collate logic
- `src/sequence_binding/models/`
  - attention encoder
  - mamba encoder
  - shared fusion classifier
- `src/sequence_binding/engine/`
  - distributed setup
  - training loop
  - evaluation metrics
- `scripts/`
  - compact dataset build
  - train
  - eval
  - smoke validation

## Main Flow

1. `build_compact_dataset.py` materializes compact `train.tsv` and `val.tsv`.
2. `train_binding.py` loads config, vocab JSON files, and TSV rows.
3. The dataset tokenizes strings on the fly and reconstructs normalized
   `PairBatch` objects.
4. The selected model computes one binary logit per pair.
5. The trainer runs `single`, `ddp`, or `ddp_zero`.
6. `eval_binding.py` writes evaluation JSON.
7. `build_report.py` converts run summaries into report artifacts.

## Model Separation

- `attention.py` keeps the transformer-style branch encoders.
- `mamba.py` keeps the state-space branch encoders.
- `common.py` keeps shared embedding, pooling, and fusion behavior.
