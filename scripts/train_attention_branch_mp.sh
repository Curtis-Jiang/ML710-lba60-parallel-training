#!/usr/bin/env bash
# Thin wrapper: branch model-parallel (single process, cuda:0 + cuda:1).
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s branch_mp "$@"
