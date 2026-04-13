# Team Guide

## First Things To Read

1. `README.md`
2. `docs/DATA_GUIDE.md`
3. `docs/CODE_ARCHITECTURE.md`
4. `report/EXPERIMENT_MATRIX.md`

## Core Commands

Build the compact dataset:

```bash
python scripts/build_compact_dataset.py
```

Run one-GPU attention baseline:

```bash
bash scripts/train_attention_single.sh
```

Run two-GPU attention DDP:

```bash
bash scripts/train_attention_ddp.sh
```

Run the smoke suite:

```bash
bash scripts/smoke_validate.sh
```

## What To Compare

For ML710, the most important comparisons are:

- attention single vs attention ddp
- attention ddp vs attention ddp_zero
- attention vs mamba under the same batch budget

Track:

- throughput
- wall-clock time
- validation AUC / AUPR
- memory behavior
