# Data Folder

This repository is published without the large processed tensor files in order to keep the GitHub repo within practical size limits.

Expected local path for the workload data:

`data/processed/lba60_existing_pocket/`

Expected local files:

- `lba60_train.pt`
- `lba60_val.pt`
- `lba60_test.pt`
- `lba60_preprocess_meta.json`

The code and configs in this repo assume that the three `.pt` files are present locally before training or evaluation starts.
