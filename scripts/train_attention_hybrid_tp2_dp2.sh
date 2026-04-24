#!/usr/bin/env bash
# Thin wrapper: 4-GPU hybrid parallel with 2-way TP x 2-way DDP.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s hybrid_tp_dp -n 4 --tp-size 2 --dp-size 2 "$@"
