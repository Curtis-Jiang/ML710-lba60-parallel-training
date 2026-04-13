#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build report artifacts from run summaries.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    report_dir = Path(args.report_dir)
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = _load_json(summary_path)
        row = {
            "run_name": summary["run_name"],
            "model": summary["model"],
            "strategy": summary["strategy"],
            "mamba_backend": summary.get("mamba_backend"),
            "global_batch_size": summary["global_batch_size"],
            "per_device_batch_size": summary["per_device_batch_size"],
            "best_epoch": summary["best_epoch"],
            "best_val_auc": summary["best_val_metrics"].get("auc"),
            "best_val_aupr": summary["best_val_metrics"].get("aupr"),
            "best_val_accuracy": summary["best_val_metrics"].get("accuracy"),
            "best_val_f1": summary["best_val_metrics"].get("f1"),
            "mean_train_examples_per_sec": summary["mean_train_examples_per_sec"],
            "total_wall_time_sec": summary["total_wall_time_sec"],
        }
        rows.append(row)

    json_path = artifact_dir / "run_table.json"
    csv_path = artifact_dir / "run_table.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Results Summary",
        "",
        "This file is generated from `runs/*/summary.json`.",
        "",
        "| Run | Model | Strategy | Val AUC | Val AUPR | Throughput (ex/s) | Wall Time (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['model']} | {row['strategy']} | "
            f"{row['best_val_auc']:.4f} | {row['best_val_aupr']:.4f} | "
            f"{row['mean_train_examples_per_sec']:.2f} | {row['total_wall_time_sec']:.2f} |"
        )
    (report_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
