# Results Summary

This file is generated from `runs/*/summary.json`.

| Run | Model | Backend | Strategy | Val AUC | Val AUPR | Throughput (ex/s) | Wall Time (s) | Params |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention_ddp_zero_course | attention | - | ddp_zero | 0.9385 | 0.6672 | 2587.79 | 990.25 | 15,651,841 |
| attention_single_course | attention | - | single | 0.9390 | 0.6670 | 2140.71 | 1203.80 | 15,651,841 |
| attention_smoke_ddp | attention | - | ddp | 0.5970 | 0.1670 | 1123.39 | 1.17 | 979,457 |
| attention_smoke_ddp_zero | attention | - | ddp_zero | 0.5970 | 0.1670 | 1133.97 | 1.25 | 979,457 |
| attention_smoke_single | attention | - | single | 0.5717 | 0.1518 | 1245.10 | 1.15 | 979,457 |
| mamba_ddp_zero_course | mamba | mamba_ssm | ddp_zero | 0.9589 | 0.7560 | 1909.43 | 1338.40 | 18,926,593 |
| mamba_single_course | mamba | mamba_ssm | single | 0.9565 | 0.7575 | 1314.76 | 1954.46 | 18,926,593 |
| mamba_smoke_ddp | mamba | mamba_ssm | ddp | 0.5719 | 0.1549 | 992.12 | 1.30 | 1,181,185 |
| mamba_smoke_single | mamba | mamba_ssm | single | 0.5255 | 0.1220 | 1114.24 | 1.19 | 1,181,185 |
