# Results Summary

This file is generated from `runs/*/summary.json`.

| Run | Model | Backend | Strategy | Val AUC | Val AUPR | Throughput (ex/s) | Wall Time (s) | Peak GPU (GB) | Params |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_branch_mp_course | attention | - | branch_mp | 0.9430 | 0.6861 | 1400.73 | 1836.79 | 2.46 | 15,651,841 |
| attention_branch_mp_smoke | attention | - | branch_mp | 0.5734 | 0.1443 | 1134.38 | 0.96 | 0.14 | 979,457 |
| attention_ddp_2gpu_course | attention | - | ddp | 0.9144 | 0.5697 | 1979.78 | 1289.58 | 1.68 | 15,651,841 |
| attention_ddp_2gpu_smoke | attention | - | ddp | 0.5980 | 0.1675 | 616.62 | 1.73 | 0.10 | 979,457 |
| attention_ddp_4gpu_course | attention | - | ddp | 0.8831 | 0.4818 | 2433.25 | 1042.39 | 0.99 | 15,651,841 |
| attention_ddp_zero_2gpu_course | attention | - | ddp_zero | 0.9288 | 0.6336 | 1844.83 | 1383.06 | 1.62 | 15,651,841 |
| attention_ddp_zero_2gpu_smoke | attention | - | ddp_zero | 0.5980 | 0.1675 | 470.49 | 2.25 | 0.10 | 979,457 |
| attention_ddp_zero_4gpu_course | attention | - | ddp_zero | 0.9383 | 0.6692 | 2048.97 | 1236.86 | 0.90 | 15,651,841 |
| attention_fsdp_z2_2gpu_course | attention | - | fsdp_z2 | 0.9258 | 0.6017 | 1533.83 | 1656.88 | 1.53 | 15,651,841 |
| attention_fsdp_z2_2gpu_smoke | attention | - | fsdp_z2 | 0.5995 | 0.1676 | 755.10 | 2.18 | 0.09 | 979,457 |
| attention_fsdp_z2_4gpu_course | attention | - | fsdp_z2 | 0.9369 | 0.6580 | 1700.02 | 1486.21 | 0.79 | 15,651,841 |
| attention_fsdp_z3_2gpu_course | attention | - | fsdp_z3 | 0.9416 | 0.6823 | 1394.13 | 1821.41 | 1.51 | 15,651,841 |
| attention_fsdp_z3_2gpu_smoke | attention | - | fsdp_z3 | 0.5995 | 0.1676 | 310.22 | 4.12 | 0.09 | 979,457 |
| attention_fsdp_z3_4gpu_course | attention | - | fsdp_z3 | 0.9377 | 0.6585 | 1545.30 | 1632.88 | 0.77 | 15,651,841 |
| attention_hybrid_tp2_dp2_course | attention | - | hybrid_tp_dp | 0.8506 | 0.4105 | 746.50 | 3378.14 | 1.26 | 15,651,841 |
| attention_hybrid_tp2_dp2_smoke | attention | - | hybrid_tp_dp | 0.5955 | 0.1590 | 301.91 | 4.71 | 0.08 | 979,457 |
| attention_single_course | attention | - | single | 0.9443 | 0.6780 | 1292.43 | 1985.27 | 3.01 | 15,651,841 |
| attention_single_smoke | attention | - | single | 0.5668 | 0.1420 | 1184.72 | 0.93 | 0.17 | 979,457 |
| attention_smoke_ddp | attention | - | ddp | 0.5981 | 0.1686 | 1082.52 | 1.02 | 0.10 | 979,457 |
| attention_smoke_ddp_zero | attention | - | ddp_zero | 0.5981 | 0.1686 | 975.50 | 1.13 | 0.10 | 979,457 |
| attention_smoke_single | attention | - | single | 0.5668 | 0.1420 | 833.70 | 1.30 | 0.17 | 979,457 |
| attention_tp2_course | attention | - | tp | 0.8933 | 0.4861 | 840.21 | 3031.30 | 2.34 | 15,651,841 |
| attention_tp2_smoke | attention | - | tp | 0.5675 | 0.1421 | 276.12 | 3.85 | 0.14 | 979,457 |
| attention_tp4_course | attention | - | tp | 0.8744 | 0.4550 | 832.59 | 3053.15 | 2.02 | 15,651,841 |
| mamba_ddp_course | mamba | mamba_ssm | ddp | 0.9559 | 0.7488 | 1377.70 | 1858.84 | 2.46 | 18,926,593 |
| mamba_single_course | mamba | mamba_ssm | single | 0.9568 | 0.7555 | 788.16 | 3257.71 | 4.49 | 18,926,593 |
| mamba_smoke_ddp | mamba | mamba_ssm | ddp | 0.5762 | 0.1568 | 907.50 | 1.21 | 0.14 | 1,181,185 |
| mamba_smoke_single | mamba | mamba_ssm | single | 0.5177 | 0.1176 | 985.04 | 1.11 | 0.24 | 1,181,185 |
