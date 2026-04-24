#!/usr/bin/env bash
# Unified smoke validator: runs one or all strategies in smoke mode and
# asserts that summary.json exists with best_val_metrics.auc > 0.55.
#
# Examples:
#   bash scripts/validate_strategy.sh -s tp
#   bash scripts/validate_strategy.sh -s all
#   bash scripts/validate_strategy.sh -s all -m attention

set -euo pipefail
cd "$(dirname "$0")/.."

STRATEGY=""
MODEL="attention"

usage() {
    cat >&2 <<EOF
usage: $0 -s <strategy|all> [-m <attention|mamba>]

  strategies: single ddp ddp_zero fsdp_z2 fsdp_z3 tp branch_mp hybrid_tp_dp
              or "all" for a safe serial sweep
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--strategy) STRATEGY="$2"; shift 2 ;;
        -m|--model)    MODEL="$2";    shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$STRATEGY" ]]; then usage; exit 2; fi

ALL_ORDER=(single ddp ddp_zero fsdp_z2 fsdp_z3 tp branch_mp hybrid_tp_dp)

if [[ "$STRATEGY" == "all" ]]; then
    LIST=("${ALL_ORDER[@]}")
else
    LIST=("$STRATEGY")
fi

check_summary() {
    local run_dir="$1"
    python - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
summary_path = run_dir / "summary.json"
history_path = run_dir / "history.jsonl"
if not summary_path.exists():
    print(f"FAIL: {summary_path} missing"); sys.exit(1)
if not history_path.exists() or history_path.stat().st_size == 0:
    print(f"FAIL: {history_path} empty"); sys.exit(1)
summary = json.loads(summary_path.read_text())
auc = (summary.get("best_val_metrics") or {}).get("auc")
if auc is None or auc <= 0.55:
    print(f"FAIL: best_val_auc={auc}"); sys.exit(1)
print(f"OK   auc={auc:.4f} peak_gpu_gb={summary.get('peak_gpu_bytes_rank0', 0) / (1024**3):.2f}")
PY
}

PASS=()
FAIL=()
for strat in "${LIST[@]}"; do
    echo ""
    echo "===== smoke: $strat ($MODEL) ====="
    set +e
    bash scripts/run_strategy.sh -s "$strat" -m "$MODEL" --smoke
    rc=$?
    set -e
    if (( rc != 0 )); then
        FAIL+=("$strat (run_rc=$rc)")
        continue
    fi
    # Must mirror the default RUN_NAME computed inside run_strategy.sh.
    case "$strat" in
        single)       run_name="${MODEL}_single_smoke" ;;
        branch_mp)    run_name="${MODEL}_branch_mp_smoke" ;;
        hybrid_tp_dp) run_name="${MODEL}_hybrid_tp2_dp2_smoke" ;;
        tp)           run_name="${MODEL}_tp2_smoke" ;;
        *)            run_name="${MODEL}_${strat}_2gpu_smoke" ;;
    esac
    set +e
    out=$(check_summary "runs/$run_name")
    rc=$?
    set -e
    echo "  $out"
    if (( rc == 0 )); then
        PASS+=("$strat")
    else
        FAIL+=("$strat ($out)")
    fi
done

echo ""
echo "===== summary ====="
printf "  pass: %s\n" "${PASS[*]:-none}"
printf "  fail: %s\n" "${FAIL[*]:-none}"

if (( ${#FAIL[@]} > 0 )); then exit 1; fi
