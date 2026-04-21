# Experiment Matrix

## Recommended Final Course Runs

| Run | Model | Backend | GPUs | Strategy | Global Batch | H100 Wall Time |
| --- | --- | --- | ---: | --- | ---: | ---: |
| attention_single_course | attention | `-` | 1 | single | 64 | `1203.80 s` |
| attention_ddp_zero_course | attention | `-` | 2 | ddp_zero | 64 | `990.25 s` |
| mamba_single_course | mamba | `mamba_ssm` | 1 | single | 64 | `1954.46 s` |
| mamba_ddp_zero_course | mamba | `mamba_ssm` | 2 | ddp_zero | 64 | `1338.40 s` |

## Optional Additional Baselines

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_ddp_course | attention | 2 | ddp | 64 |
| mamba_ddp_course | mamba | 2 | ddp | 64 |

## Smoke Runs

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_smoke_single | attention | 1 | single | 32 |
| attention_smoke_ddp | attention | 2 | ddp | 32 |
| attention_smoke_ddp_zero | attention | 2 | ddp_zero | 32 |
| mamba_smoke_single | mamba | 1 | single | 32 |
| mamba_smoke_ddp | mamba | 2 | ddp | 32 |
