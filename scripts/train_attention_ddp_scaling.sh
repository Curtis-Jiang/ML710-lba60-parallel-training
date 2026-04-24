#!/usr/bin/env bash
# Scaling wrapper: naive DDP baseline at any GPU count.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s ddp -n "${NGPU:-2}" "$@"
