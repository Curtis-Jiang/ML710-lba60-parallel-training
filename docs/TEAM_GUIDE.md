# Team Guide

## First Things To Read

1. `README.md`
2. `report/COURSE_EXPERIMENT_SUMMARY.md`
3. `docs/DATA_GUIDE.md`
4. `docs/CODE_ARCHITECTURE.md`
5. `report/EXPERIMENT_MATRIX.md`

## Core Commands

The dataset is already committed in the repo. This command is optional and only
rebuilds the TSV files if you have the external source files:

```bash
python scripts/build_compact_dataset.py
```

Install the Mamba dependency used by the final reported results:

```bash
bash scripts/install_mamba.sh
```

Run one-GPU attention baseline:

```bash
bash scripts/train_attention_single.sh
```

Run the recommended advanced two-GPU Attention experiment:

```bash
bash scripts/train_attention_ddp_zero.sh
```

Run the recommended Mamba pair:

```bash
bash scripts/train_mamba_single.sh
bash scripts/train_mamba_ddp_zero.sh
```

Run the smoke suite:

```bash
bash scripts/smoke_validate.sh
```

## What To Compare

For ML710, the most important comparisons are:

- attention single vs attention ddp_zero
- mamba single vs mamba ddp_zero
- attention vs mamba under the same global batch budget

The current final reported Mamba runs use the `mamba_ssm` backend.

Track:

- throughput
- wall-clock time
- validation AUC / AUPR
- memory behavior
