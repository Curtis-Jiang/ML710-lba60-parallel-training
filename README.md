# ML710 Sequence Binding Parallel Training

This repository is the final course-facing version of the ML710 project around
one supervised workload: binary protein-ligand binding classification from
protein sequences and SMILES strings.

The project is optimized for distributed training clarity rather than model
novelty. It provides:

- one clean workload based on a compact PubChem course subset
- two model families: `attention` and `mamba`
- three training strategies: `single`, `ddp`, and `ddp_zero`
- deterministic compact-dataset building
- smoke validation and report artifact generation

The current `*_course.yaml` configs are calibrated as the final course-scale
setting:

- target runtime: roughly the `A100 half-hour` course target
- measured runtime on H100: about `16-33 minutes` for the four final runs
- recommended final comparison: `single` vs `ddp_zero` for both Attention and Mamba

## Final Submission Snapshot

The canonical four course runs are:

| Run | Model | Backend | Strategy | H100 Wall Time | Throughput (ex/s) | Val AUC | Val AUPR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `attention_single_course` | attention | `-` | `single` | `1203.80 s` | `2140.71` | `0.9390` | `0.6670` |
| `attention_ddp_zero_course` | attention | `-` | `ddp_zero` | `990.25 s` | `2587.79` | `0.9385` | `0.6672` |
| `mamba_single_course` | mamba | `mamba_ssm` | `single` | `1954.46 s` | `1314.76` | `0.9565` | `0.7575` |
| `mamba_ddp_zero_course` | mamba | `mamba_ssm` | `ddp_zero` | `1338.40 s` | `1909.43` | `0.9589` | `0.7560` |

These are the numbers currently reflected in:

- `report/COURSE_EXPERIMENT_SUMMARY.md`
- `report/RESULTS_SUMMARY.md`
- `report/artifacts/course_run_table.csv`

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

1. Install the base dependencies:

```bash
pip install -r requirements.txt
```

2. Install the Mamba dependency used by the final reported results:

```bash
bash scripts/install_mamba.sh
```

3. The compact dataset is already committed in this repo. This command is
safe to run and will simply no-op unless you want to rebuild from the
external source files:

```bash
python scripts/build_compact_dataset.py
```

4. Train the attention baseline on one GPU:

```bash
bash scripts/train_attention_single.sh
```

5. Train the recommended advanced 2-GPU Attention run:

```bash
bash scripts/train_attention_ddp_zero.sh
```

6. Train the Mamba baseline and advanced run:

```bash
bash scripts/train_mamba_single.sh
bash scripts/train_mamba_ddp_zero.sh
```

7. Optional naive DDP baselines:

```bash
bash scripts/train_attention_ddp.sh
bash scripts/train_mamba_ddp.sh
```

8. Run the end-to-end smoke suite:

```bash
bash scripts/smoke_validate.sh
```

## Reproducing The Final Four Runs

To match the current course report as closely as possible:

```bash
pip install -r requirements.txt
bash scripts/install_mamba.sh
bash scripts/train_attention_single.sh
bash scripts/train_attention_ddp_zero.sh
bash scripts/train_mamba_single.sh
bash scripts/train_mamba_ddp_zero.sh
python scripts/build_report.py --runs-dir runs --report-dir report
```

## Notes

- Runtime training and evaluation use only the compact TSV dataset. The
  original large source bank is used only as a one-time external conversion
  source by `scripts/build_compact_dataset.py`.
- The compact dataset is committed in this repository, so a teammate can clone
  or pull the repo and run training without fetching any external bank files.
- The final project version uses `mamba_ssm` for all Mamba experiments.
- `src/sequence_binding/models/mamba.py` loads the core Mamba module directly,
  so the training path is not blocked by unrelated optional generation imports.
- The helper script `scripts/check_mamba_install.py` is used by the Mamba
  training scripts and the smoke suite to fail fast with a clear setup message.
- `naive DDP` is included as a baseline because it is required for the course
  comparisons. `ddp_zero` is included as the advanced distributed strategy in
  this version of the project.
