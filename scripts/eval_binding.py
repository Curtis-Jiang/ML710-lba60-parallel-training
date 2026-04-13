#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_binding import load_experiment_config
from sequence_binding.engine import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained sequence-binding run.")
    parser.add_argument("--config", required=True, help="Config YAML relative to the repo root.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument(
        "--strategy",
        default="single",
        choices=["single", "ddp", "ddp_zero"],
        help="Distributed strategy used for evaluation.",
    )
    parser.add_argument("--output", required=True, help="JSON output path.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    metrics = run_evaluation(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        strategy=args.strategy,
        output_path=args.output,
    )
    if metrics:
        print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
