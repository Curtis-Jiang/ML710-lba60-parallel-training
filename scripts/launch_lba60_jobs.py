#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "binding_affinity" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from binding_affinity.utils.config import load_config  # noqa: E402
from binding_affinity.utils.paths import ensure_dir, resolve_ws_path  # noqa: E402


@dataclass(frozen=True)
class JobSpec:
    name: str
    config: str
    gpus: str
    nproc_per_node: int
    seed: int
    overrides: Dict[str, Any]


@dataclass
class RunningJob:
    spec: JobSpec
    task: str
    run_name: str
    run_dir: Path
    log_path: Path
    proc: subprocess.Popen
    t0: float
    returncode: Optional[int] = None
    t1: Optional[float] = None


def _yaml_scalar(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip()


def _unique_run_name(task_dir: Path, base_name: str) -> str:
    base_name = str(base_name)[:180]
    if not (task_dir / base_name).exists():
        return base_name
    for idx in range(2, 10_000):
        candidate = f"{base_name}_v{idx}"
        if not (task_dir / candidate).exists():
            return candidate
    raise RuntimeError(f"failed to allocate unique run name for {base_name}")


def _load_jobs(path: str) -> List[JobSpec]:
    spec_path = resolve_ws_path(path)
    raw = yaml.safe_load(spec_path.read_text()) or {}
    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError(f"spec must contain a non-empty jobs list: {spec_path}")

    jobs: List[JobSpec] = []
    for idx, item in enumerate(jobs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{idx}] must be a mapping in {spec_path}")
        overrides = item.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"jobs[{idx}].overrides must be a mapping in {spec_path}")
        jobs.append(
            JobSpec(
                name=str(item["name"]),
                config=str(item["config"]),
                gpus=str(item["gpus"]),
                nproc_per_node=int(item.get("nproc_per_node", 2)),
                seed=int(item.get("seed", 0)),
                overrides=overrides,
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multiple LBA60 jobs in parallel on disjoint GPU groups.")
    parser.add_argument("--spec", type=str, default="configs/packed_seed_sweep.yaml")
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--summary_out", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    jobs = _load_jobs(args.spec)
    running: List[RunningJob] = []

    for spec in jobs:
        cfg = load_config(spec.config)
        task = str(cfg["task"]["name"])
        runs_root = ensure_dir(cfg["io"]["runs_root"])
        task_dir = ensure_dir(Path(runs_root) / task)
        run_name = _unique_run_name(task_dir, spec.name)
        run_dir = ensure_dir(task_dir / run_name)
        log_path = run_dir / "stdout.log"

        visible = [gpu.strip() for gpu in spec.gpus.split(",") if gpu.strip()]
        if len(visible) != spec.nproc_per_node:
            raise ValueError(
                f"job {spec.name}: gpus={spec.gpus!r} must have the same count as nproc_per_node={spec.nproc_per_node}"
            )

        train_script = str(resolve_ws_path("binding_affinity/scripts/train_affinity_model.py"))
        if spec.nproc_per_node == 1:
            cmd = [
                "python",
                "-u",
                train_script,
                "--config",
                str(resolve_ws_path(spec.config)),
                "--run_name",
                run_name,
                "--seed",
                str(spec.seed),
            ]
        else:
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={spec.nproc_per_node}",
                train_script,
                "--config",
                str(resolve_ws_path(spec.config)),
                "--run_name",
                run_name,
                "--seed",
                str(spec.seed),
            ]
        for key, value in spec.overrides.items():
            cmd += ["--set", f"{key}={_yaml_scalar(value)}"]

        payload = {
            "name": spec.name,
            "task": task,
            "gpus": spec.gpus,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "cmd": cmd,
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            continue

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = spec.gpus
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        log_file.close()

        running.append(
            RunningJob(
                spec=spec,
                task=task,
                run_name=run_name,
                run_dir=run_dir,
                log_path=log_path,
                proc=proc,
                t0=time.time(),
            )
        )
        print(json.dumps({"event": "started", "name": spec.name, "pid": proc.pid, "run_dir": str(run_dir)}))

    if args.dry_run:
        return

    while True:
        unfinished = 0
        for job in running:
            if job.returncode is not None:
                continue
            rc = job.proc.poll()
            if rc is None:
                unfinished += 1
                continue
            job.returncode = int(rc)
            job.t1 = time.time()
            print(
                json.dumps(
                    {
                        "event": "finished",
                        "name": job.spec.name,
                        "returncode": job.returncode,
                        "elapsed_sec": round(float(job.t1 - job.t0), 3),
                        "run_dir": str(job.run_dir),
                    }
                )
            )
        if unfinished == 0:
            break
        time.sleep(float(args.poll_seconds))

    if args.summary_out:
        summary_path = resolve_ws_path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = []
        for job in running:
            elapsed = None if job.t1 is None else round(float(job.t1 - job.t0), 3)
            summary.append(
                {
                    "name": job.spec.name,
                    "config": job.spec.config,
                    "gpus": job.spec.gpus,
                    "nproc_per_node": job.spec.nproc_per_node,
                    "seed": job.spec.seed,
                    "task": job.task,
                    "run_name": job.run_name,
                    "run_dir": str(job.run_dir),
                    "log_path": str(job.log_path),
                    "returncode": job.returncode,
                    "elapsed_sec": elapsed,
                }
            )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
