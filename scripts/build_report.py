#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sequence_binding.engine.metrics import goodput as goodput_fn


STRATEGY_GPU_COUNT = {
    "single": 1,
    "branch_mp": 2,
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_rows(json_path: Path, csv_path: Path, rows: list[dict]) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _infer_gpus(summary: dict) -> int:
    """Recover the GPU count used by a run from its summary."""
    strategy = summary.get("strategy", "")
    if strategy in STRATEGY_GPU_COUNT:
        return STRATEGY_GPU_COUNT[strategy]
    global_bs = int(summary.get("global_batch_size") or 0)
    per_device = int(summary.get("per_device_batch_size") or 0)
    if per_device > 0 and global_bs > 0:
        return max(global_bs // per_device, 1)
    return 1


def _baseline_throughput(summaries: dict[str, dict], model: str) -> float | None:
    """Throughput of the matching ``<model>_single_*`` run, if present."""
    for summary in summaries.values():
        if summary.get("model") != model:
            continue
        if summary.get("strategy") != "single":
            continue
        return float(summary.get("mean_train_examples_per_sec") or 0.0) or None
    return None


def _baseline_val_auc(summaries: dict[str, dict], model: str) -> float | None:
    for summary in summaries.values():
        if summary.get("model") != model or summary.get("strategy") != "single":
            continue
        auc = summary.get("best_val_metrics", {}).get("auc")
        return float(auc) if auc is not None else None
    return None


def _row_for_summary(summary: dict) -> dict:
    return {
        "run_name": summary["run_name"],
        "model": summary["model"],
        "strategy": summary["strategy"],
        "mamba_backend": summary.get("mamba_backend"),
        "parameter_count": summary["parameter_count"],
        "global_batch_size": summary["global_batch_size"],
        "per_device_batch_size": summary["per_device_batch_size"],
        "best_epoch": summary["best_epoch"],
        "best_val_auc": summary["best_val_metrics"].get("auc"),
        "best_val_aupr": summary["best_val_metrics"].get("aupr"),
        "best_val_accuracy": summary["best_val_metrics"].get("accuracy"),
        "best_val_f1": summary["best_val_metrics"].get("f1"),
        "mean_train_examples_per_sec": summary["mean_train_examples_per_sec"],
        "total_wall_time_sec": summary["total_wall_time_sec"],
        "peak_gpu_gb": round(
            float(summary.get("peak_gpu_bytes_rank0", 0)) / (1024 ** 3), 3
        ),
    }


def _build_convergence_curves(summaries: dict[str, dict]) -> list[dict]:
    curves: list[dict] = []
    for summary in summaries.values():
        history = summary.get("history") or []
        curves.append(
            {
                "run_name": summary["run_name"],
                "model": summary["model"],
                "strategy": summary["strategy"],
                "history": [
                    {
                        "epoch": item.get("epoch"),
                        "cumulative_wall_time_sec": item.get("cumulative_wall_time_sec"),
                        "train_loss": item.get("train_loss"),
                        "val_auc": (item.get("val_metrics") or {}).get("auc"),
                        "val_aupr": (item.get("val_metrics") or {}).get("aupr"),
                        "peak_gpu_bytes": item.get("peak_gpu_bytes"),
                    }
                    for item in history
                ],
            }
        )
    return curves


def _build_goodput_rows(summaries: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for summary in summaries.values():
        model = summary["model"]
        gpus = _infer_gpus(summary)
        throughput = float(summary.get("mean_train_examples_per_sec") or 0.0)
        val_auc = float(summary.get("best_val_metrics", {}).get("auc") or 0.0)
        base_thr = _baseline_throughput(summaries, model)
        base_auc = _baseline_val_auc(summaries, model)
        scaling_eff: float | None
        if base_thr and gpus > 0:
            scaling_eff = throughput / (base_thr * gpus)
        else:
            scaling_eff = None
        goodput_val: float | None
        if scaling_eff is not None and base_auc:
            goodput_val = goodput_fn(
                examples_per_sec=throughput,
                val_auc=val_auc,
                baseline_val_auc=base_auc,
                scaling_efficiency=scaling_eff,
            )
        else:
            goodput_val = None
        rows.append(
            {
                "run_name": summary["run_name"],
                "model": model,
                "strategy": summary["strategy"],
                "gpus": gpus,
                "throughput": round(throughput, 3),
                "best_val_auc": round(val_auc, 6) if val_auc else None,
                "scaling_efficiency": round(scaling_eff, 4) if scaling_eff is not None else None,
                "goodput": round(goodput_val, 3) if goodput_val is not None else None,
                "peak_gpu_gb": round(
                    float(summary.get("peak_gpu_bytes_rank0", 0)) / (1024 ** 3), 3
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build report artifacts from run summaries.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    report_dir = Path(args.report_dir)
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = _load_json(summary_path)
        summaries[summary["run_name"]] = summary

    rows = [_row_for_summary(summaries[name]) for name in summaries]

    json_path = artifact_dir / "run_table.json"
    csv_path = artifact_dir / "run_table.csv"
    _write_rows(json_path, csv_path, rows)

    course_rows = [row for row in rows if "_course" in row["run_name"]]
    course_json_path = artifact_dir / "course_run_table.json"
    course_csv_path = artifact_dir / "course_run_table.csv"
    _write_rows(course_json_path, course_csv_path, course_rows)

    curves = _build_convergence_curves(summaries)
    (artifact_dir / "convergence_curves.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8"
    )

    goodput_rows = _build_goodput_rows(summaries)
    goodput_json_path = artifact_dir / "goodput_table.json"
    goodput_csv_path = artifact_dir / "goodput_table.csv"
    _write_rows(goodput_json_path, goodput_csv_path, goodput_rows)

    lines = [
        "# Results Summary",
        "",
        "This file is generated from `runs/*/summary.json`.",
        "",
        "| Run | Model | Backend | Strategy | Val AUC | Val AUPR | Throughput (ex/s) | Wall Time (s) | Peak GPU (GB) | Params |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        backend = row["mamba_backend"] or "-"
        lines.append(
            f"| {row['run_name']} | {row['model']} | {backend} | {row['strategy']} | "
            f"{row['best_val_auc']:.4f} | {row['best_val_aupr']:.4f} | "
            f"{row['mean_train_examples_per_sec']:.2f} | {row['total_wall_time_sec']:.2f} | "
            f"{row['peak_gpu_gb']:.2f} | "
            f"{row.get('parameter_count', 0):,} |"
        )
    (report_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(csv_path)
    print(course_json_path)
    print(course_csv_path)
    print(artifact_dir / "convergence_curves.json")
    print(goodput_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
