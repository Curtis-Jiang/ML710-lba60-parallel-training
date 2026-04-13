# Results Summary

This file is generated from `runs/*/summary.json`.

| Run | Model | Strategy | Val AUC | Val AUPR | Throughput (ex/s) | Wall Time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| attention_smoke_ddp | attention | ddp | 0.5970 | 0.1670 | 1098.06 | 1.29 |
| attention_smoke_ddp_zero | attention | ddp_zero | 0.5970 | 0.1670 | 867.35 | 1.54 |
| attention_smoke_single | attention | single | 0.5717 | 0.1518 | 1042.80 | 1.33 |
| mamba_smoke_ddp | mamba | ddp | 0.5887 | 0.1580 | 974.48 | 1.42 |
| mamba_smoke_single | mamba | single | 0.4949 | 0.0886 | 1018.47 | 1.38 |
