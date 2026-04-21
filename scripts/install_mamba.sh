#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pip install "transformers==4.47.1" "tokenizers<0.22"
pip install --no-build-isolation "mamba-ssm==2.2.4"

echo "mamba_ssm installation complete"
