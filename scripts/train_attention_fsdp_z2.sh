#!/usr/bin/env bash
# Thin wrapper: attention + ZeRO-2 via FSDP SHARD_GRAD_OP.
# Override GPU count with NGPU=<n>.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s fsdp_z2 -n "${NGPU:-2}" "$@"
