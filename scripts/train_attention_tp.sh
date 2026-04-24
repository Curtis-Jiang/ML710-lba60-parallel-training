#!/usr/bin/env bash
# Thin wrapper: tensor parallel on the attention FFN. Default tp=2; set NGPU=4 for tp=4.
set -euo pipefail
exec bash "$(dirname "$0")/run_strategy.sh" -s tp -n "${NGPU:-2}" "$@"
