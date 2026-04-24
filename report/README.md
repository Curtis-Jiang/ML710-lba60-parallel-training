# Report

This directory contains the course-facing writeups and generated experiment
artifacts for the rebuilt sequence-binding ML710 project.

Recommended reading order:

1. `COURSE_EXPERIMENT_SUMMARY.md`
2. `RESULTS_SUMMARY.md`
3. `COURSE_REQUIREMENTS_MAPPING.md`
4. `EXPERIMENT_MATRIX.md`

- `WORKLOAD_OVERVIEW.md`
- `ARCHITECTURES.md`
- `EXPERIMENT_MATRIX.md`
- `COURSE_EXPERIMENT_SUMMARY.md`
- `COURSE_REQUIREMENTS_MAPPING.md`
- `SLIDE_OUTLINE.md`
- `RESULTS_SUMMARY.md` (generated)
- `artifacts/` (generated CSV/JSON tables)

---

# v2 Reading Guide (2026-04-23 sweep)

Every v1 markdown above is kept intact for historical comparison. v2 content
has been **appended** to each file as a new section (after a `---` separator)
rather than rewriting the original text. The recommended v2 reading order is:

1. `PHASE_COMPLETION_REPORT.md` — phase-by-phase completion audit for Phases
   B / C / D (15/15 runs done)
2. `EXPERIMENT_MATRIX.md` — v2 section: actual scripts + wall times per run
3. `COURSE_EXPERIMENT_SUMMARY.md` — v2 section: results table + scaling
   analysis + throughput-vs-quality Pareto
4. `COURSE_REQUIREMENTS_MAPPING.md` — v2 section: strategy-to-student mapping
5. `WORKLOAD_OVERVIEW.md` — v2 section: what is held constant vs what varies
6. `ARCHITECTURES.md` — v2 section: where parallelism hooks into each model
7. `SLIDE_OUTLINE.md` — v2 section: slide structure for the full sweep
8. `RESULTS_SUMMARY.md` — auto-regenerated table over all 27 runs
9. `artifacts/` — auto-regenerated CSV / JSON tables including:
   - `run_table.{csv,json}` — all runs
   - `course_run_table.{csv,json}` — course runs only
   - `goodput_table.{csv,json}` — scaling efficiency + goodput
   - `convergence_curves.json` — per-epoch curves for every run

To regenerate the auto-generated pieces after new runs land in `runs/*/`:

```bash
python scripts/build_report.py --runs-dir runs --report-dir report
```
