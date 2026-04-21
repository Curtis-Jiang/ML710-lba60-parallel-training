#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_binding.models.mamba import MambaSSM


def main() -> int:
    if MambaSSM is None:
        print("mamba_ssm is not available. Run `bash scripts/install_mamba.sh` first.", file=sys.stderr)
        return 1
    print("mamba_ssm is available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
