#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_binding import load_experiment_config
from sequence_binding.engine import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a sequence-binding model.")
    parser.add_argument("--config", required=True, help="Config YAML relative to the repo root.")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=[
            "single",
            "ddp",
            "ddp_zero",
            "fsdp_z2",
            "fsdp_z3",
            "branch_mp",
            "tp",
            "hybrid_tp_dp",
        ],
        help="Optional override for the distributed strategy.",
    )
    parser.add_argument("--run-name", default=None, help="Optional output run directory name.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    run_dir = run_training(config, strategy=args.strategy, run_name=args.run_name)
    # Only the launcher-visible rank should print the run directory: under
    # torchrun every child process reaches this line, so a bare ``print``
    # would emit N duplicate lines and confuse downstream parsers.
    if int(os.environ.get("RANK", "0")) == 0:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
