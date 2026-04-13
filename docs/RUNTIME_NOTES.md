# Runtime Notes

## Dataset Size

The copied `lba60` processed snapshot in `data/processed/lba60_existing_pocket/` contains:

- train: `3563`
- val: `448`
- test: `452`

## Config Profiles

- `configs/lba60_smoke.yaml`
  Intended for sanity checks only. Uses 2 epochs and a very small batch.

- `configs/lba60_quick.yaml`
  Recommended default for the course project. Uses 20 epochs and is the main target for scaling comparisons.

- `configs/lba60_course.yaml`
  A slightly longer variant for final report runs if the runtime budget still looks safe.

## Practical Runtime Expectations

These are planning estimates, not new A100 measurements from this standalone directory.

- The upstream workload family ran on H100 hardware in the original research environment.
- A practical A100 slowdown versus H100 is likely in the rough range of `1.5x` to `2.2x`, depending on whether the run is compute-bound or input-bound.
- Based on that and the current dataset size, `configs/lba60_quick.yaml` is designed to be a reasonable candidate for a course run on `2 x A100` within roughly `30` to `60` minutes.

## Suggested Planning Budget

- 1 GPU smoke run: a few minutes
- 1 GPU quick run: tens of minutes
- 2 GPU quick DDP run: around the course target budget
- 2 GPU course run: potentially around or a bit above 1 hour depending on GPU type, dataloader throughput, and memory headroom

## Memory Caution

If the available hardware is `A100 40GB` instead of `A100 80GB`, start with the provided batch sizes first and reduce further if needed:

- default per-GPU batch in quick/course configs: `8`
- first fallback: set `train.batch_size=6`

Example override:

```bash
bash scripts/train_lba60_ddp.sh configs/lba60_quick.yaml lba60_ddp_bs6 0 --set train.batch_size=6
```

## Honest Reporting Note

If this project is submitted before new A100 timing data is collected, the report should explicitly say that the A100 numbers were estimated from upstream H100 behavior plus hardware-based slowdown expectations, not directly benchmarked in this repo.
