# Run Comparison

This file was generated from `summary.json`, `test_metrics.json`, and `metrics.jsonl` in the run directories.

## Summary Table

| run | world size | global batch | avg ex/s | max ex/s | approx total wall sec | approx epoch wall sec sum | peak reserved GB | best val pearson | test pearson | test rmse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lba60_single_quick_seed0 | 1 | 8 | 34.640 | 35.286 | 0.266 | 2145.603 | 7.416 | 0.775 | 0.776 | 1.311 |
| lba60_ddp_quick_seed0 | 2 | 16 | 70.486 | 73.255 | 0.100 | 1097.116 | 7.736 | 0.755 | 0.768 | 1.333 |

## Relative Throughput

- `lba60_ddp_quick_seed0` vs `lba60_single_quick_seed0` average throughput speedup: `2.035x`

## Notes

- `avg ex/s` and `max ex/s` come from epoch-level `train/examples_per_sec`.
- `approx epoch wall sec sum` is the sum of epoch-level `time_sec` and is a useful training-time proxy.
- Final course writeups may still want to add external wall-clock timing from process launch/finish timestamps.
