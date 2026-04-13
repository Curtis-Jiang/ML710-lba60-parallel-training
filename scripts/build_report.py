#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


def _collect_run(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    test_metrics_path = run_dir / "test_metrics.json"
    metrics_path = run_dir / "metrics.jsonl"
    config_snapshot_path = run_dir / "config_snapshot.yaml"

    summary = _read_json(summary_path)
    test_metrics = _read_json(test_metrics_path)
    rows = _read_jsonl(metrics_path)

    throughput = [float(r["train/examples_per_sec"]) for r in rows if "train/examples_per_sec" in r]
    epoch_time = [float(r["time_sec"]) for r in rows if "time_sec" in r]
    peak_reserved = [float(r["cuda/peak_reserved_gb"]) for r in rows if "cuda/peak_reserved_gb" in r]
    peak_alloc = [float(r["cuda/peak_alloc_gb"]) for r in rows if "cuda/peak_alloc_gb" in r]

    wall_clock_sec_approx = None
    if config_snapshot_path.exists() and summary_path.exists():
        wall_clock_sec_approx = float(summary_path.stat().st_mtime - config_snapshot_path.stat().st_mtime)

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "world_size": int(summary["world_size"]),
        "batch_per_gpu": int(summary["batch_per_gpu"]),
        "global_batch": int(summary["global_batch"]),
        "best_epoch": summary["selection"]["best_epoch"],
        "val_best_raw": summary["selection"]["val@best_raw"],
        "val_best_calib": summary["selection"]["val@best_calib"],
        "test_raw": summary["results"]["test_raw"],
        "test_calib": summary["results"]["test_calib"],
        "test_metrics_file": test_metrics,
        "epochs_logged": len(rows),
        "train_examples_per_sec_avg": _safe_mean(throughput),
        "train_examples_per_sec_max": _safe_max(throughput),
        "train_examples_per_sec_last": None if not throughput else float(throughput[-1]),
        "epoch_time_sec_avg": _safe_mean(epoch_time),
        "epoch_time_sec_sum": None if not epoch_time else float(sum(epoch_time)),
        "wall_clock_sec_approx": wall_clock_sec_approx,
        "peak_reserved_gb_max": _safe_max(peak_reserved),
        "peak_alloc_gb_max": _safe_max(peak_alloc),
    }


def _speedup(single: Dict[str, Any], other: Dict[str, Any]) -> Optional[float]:
    a = single.get("train_examples_per_sec_avg")
    b = other.get("train_examples_per_sec_avg")
    if a in (None, 0) or b is None:
        return None
    return float(b) / float(a)


def _md_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _metric_value(d: Optional[Dict[str, Any]], key: str) -> str:
    if not isinstance(d, dict) or key not in d or d[key] is None:
        return "n/a"
    return _md_float(float(d[key]))


def _build_markdown(runs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Run Comparison")
    lines.append("")
    lines.append("This file was generated from `summary.json`, `test_metrics.json`, and `metrics.jsonl` in the run directories.")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| run | world size | global batch | avg ex/s | max ex/s | approx total wall sec | approx epoch wall sec sum | peak reserved GB | best val pearson | test pearson | test rmse |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        lines.append(
            "| "
            + f"{run['run_name']} | "
            + f"{run['world_size']} | "
            + f"{run['global_batch']} | "
            + f"{_md_float(run['train_examples_per_sec_avg'])} | "
            + f"{_md_float(run['train_examples_per_sec_max'])} | "
            + f"{_md_float(run['wall_clock_sec_approx'])} | "
            + f"{_md_float(run['epoch_time_sec_sum'])} | "
            + f"{_md_float(run['peak_reserved_gb_max'])} | "
            + f"{_metric_value(run['val_best_raw'], 'pearson')} | "
            + f"{_metric_value(run['test_raw'], 'pearson')} | "
            + f"{_metric_value(run['test_raw'], 'rmse')} |"
        )

    if len(runs) >= 2:
        base = runs[0]
        lines.append("")
        lines.append("## Relative Throughput")
        lines.append("")
        for run in runs[1:]:
            s = _speedup(base, run)
            if s is not None:
                lines.append(f"- `{run['run_name']}` vs `{base['run_name']}` average throughput speedup: `{s:.3f}x`")
            else:
                lines.append(f"- `{run['run_name']}` vs `{base['run_name']}` average throughput speedup: `n/a`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `avg ex/s` and `max ex/s` come from epoch-level `train/examples_per_sec`.")
    lines.append("- `approx epoch wall sec sum` is the sum of epoch-level `time_sec` and is a useful training-time proxy.")
    lines.append("- Final course writeups may still want to add external wall-clock timing from process launch/finish timestamps.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparison report from completed run directories.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories such as runs/affinity/lba60/<run_name>")
    parser.add_argument("--json_out", type=str, default="report/artifacts/run_comparison.json")
    parser.add_argument("--md_out", type=str, default="report/RUN_COMPARISON.md")
    args = parser.parse_args()

    run_dirs = [Path(p).resolve() for p in args.runs]
    results = [_collect_run(run_dir) for run_dir in run_dirs]

    json_out = Path(args.json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_out = Path(args.md_out).resolve()
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_build_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
