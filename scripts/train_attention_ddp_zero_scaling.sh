#!/usr/bin/env bash
# Scaling wrapper: DDP + ZeroRedundancyOptimizer (ZeRO-1) at any GPU count.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s ddp_zero -n "${NGPU:-2}" "$@"
