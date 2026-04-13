# Experiment Matrix

## Default Course Runs

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_single_course | attention | 1 | single | 64 |
| attention_ddp_course | attention | 2 | ddp | 64 |
| attention_ddp_zero_course | attention | 2 | ddp_zero | 64 |
| mamba_single_course | mamba | 1 | single | 64 |
| mamba_ddp_course | mamba | 2 | ddp | 64 |

## Smoke Runs

| Run | Model | GPUs | Strategy | Global Batch |
| --- | --- | ---: | --- | ---: |
| attention_smoke_single | attention | 1 | single | 32 |
| attention_smoke_ddp | attention | 2 | ddp | 32 |
| attention_smoke_ddp_zero | attention | 2 | ddp_zero | 32 |
| mamba_smoke_single | mamba | 1 | single | 32 |
| mamba_smoke_ddp | mamba | 2 | ddp | 32 |
