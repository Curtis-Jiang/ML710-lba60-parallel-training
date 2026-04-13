# Report Folder

This folder stores ML710-facing summaries for completed experiments.

Expected contents:

- run comparison notes
- course requirement mapping
- copied metrics or parsed summaries
- final presentation/report-ready writeups

The raw training artifacts still live under:

`runs/affinity/lba60/`

This separation keeps the report readable while preserving the original run outputs.

Key files generated in this pass:

- `RUN_COMPARISON.md`
- `ML710_SUMMARY.md`
- `EXPERIMENT_SETUP.md`
- `artifacts/run_comparison.json`
- `artifacts/lba60_single_quick_seed0.summary.json`
- `artifacts/lba60_ddp_quick_seed0.summary.json`

Raw terminal logs were intentionally omitted after the naming cleanup so the shared project folder only contains the renamed `binding_affinity` terminology.
