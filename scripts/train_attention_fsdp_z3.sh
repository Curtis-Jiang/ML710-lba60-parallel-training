#!/usr/bin/env bash
# Thin wrapper: attention + ZeRO-3 via FSDP FULL_SHARD.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s fsdp_z3 -n "${NGPU:-2}" "$@"
