# Course Experiment Summary

## Setup

These are the four course-scale experiments requested for the current project:

- `attention_single_course`
- `attention_ddp_zero_course`
- `mamba_single_course`
- `mamba_ddp_zero_course`

Shared setup:

- dataset: `100,000` train / `10,000` val
- task: binary protein-ligand binding classification
- truncation: protein `512`, SMILES `128`
- epochs: `25`
- global batch size: `64`
- mixed precision: `bf16`
- hardware used for these runs: NVIDIA H100 80GB

The raw structured table for these runs is in:

- `report/artifacts/course_run_table.json`
- `report/artifacts/course_run_table.csv`

## Results Table

| Run | Model | Strategy | GPUs | Params | Throughput (ex/s) | Wall Time (s) | Val AUC | Val AUPR | Val Acc | Val F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_single_course | attention | single | 1 | 15,651,841 | 2140.71 | 1203.80 | 0.9390 | 0.6670 | 0.9328 | 0.5385 |
| attention_ddp_zero_course | attention | ddp_zero | 2 | 15,651,841 | 2587.79 | 990.25 | 0.9385 | 0.6672 | 0.9339 | 0.5262 |
| mamba_single_course | mamba | single | 1 | 18,926,593 | 1314.76 | 1954.46 | 0.9565 | 0.7575 | 0.9416 | 0.6708 |
| mamba_ddp_zero_course | mamba | ddp_zero | 2 | 18,926,593 | 1909.43 | 1338.40 | 0.9589 | 0.7560 | 0.9455 | 0.6699 |

## Comparison

### Attention: single vs advanced distributed

- `ddp_zero` throughput is `1.209x` of single-GPU throughput.
- `ddp_zero` wall time is `0.823x` of the single-GPU wall time.
- Val AUC changes by `-0.0005`.
- Val AUPR changes by `+0.0002`.

Interpretation:

- For the larger final Attention configuration, `ddp_zero` now gives a real systems win instead of pure overhead.
- The validation metrics stay essentially flat, so this is a clean throughput improvement rather than an optimization tradeoff.

### Mamba: single vs advanced distributed

- `ddp_zero` throughput is `1.452x` of single-GPU throughput.
- `ddp_zero` wall time is `0.685x` of the single-GPU wall time.
- Val AUC changes by `+0.0024`.
- Val AUPR changes by `-0.0015`.

Interpretation:

- With `mamba_ssm`, the advanced distributed Mamba run now shows a clear systems speedup.
- Validation quality stays nearly unchanged, so this is mostly a runtime gain rather than a quality tradeoff.

### Attention vs Mamba

- On single GPU, Mamba is `0.614x` as fast as Attention.
- On single GPU, Mamba improves Val AUC by `+0.0174` and Val AUPR by `+0.0905`.
- Under `ddp_zero`, Mamba is `0.738x` as fast as Attention.
- Under `ddp_zero`, Mamba improves Val AUC by `+0.0204` and Val AUPR by `+0.0888`.

Interpretation:

- In this final version, Mamba is a heavier and slower workload than Attention.
- The tradeoff is that Mamba still gives stronger validation quality, which makes it a good stress test for the distributed part of the project.

## Runtime Calibration Note

- On H100, the four final runs land at about `16-33 minutes`.
- The same configs are intended to land closer to the course target of about
  half an hour on A100-class GPUs.

## Important Note

For the Mamba runs, the recorded backend is `mamba_ssm`.

That means:

- the current course-scale Mamba results reflect the intended state-space block
- this is the only Mamba backend used in the final project presentation and report

## Takeaway

The four requested experiments all ran successfully and give a coherent course story:

- both model families train correctly
- the advanced distributed strategy is implemented and reproducible
- `ddp_zero` helps both the larger Attention run and the Mamba run
- Mamba is now the heavier, more distributed-systems-oriented workload in the current codebase
