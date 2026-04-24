#!/usr/bin/env bash
# Unified dispatcher for every parallel strategy in the project.
# Picks the right config YAML, decides python vs torchrun, and forwards to
# scripts/train_binding.py.
#
# Examples:
#   bash scripts/run_strategy.sh -s fsdp_z3 -n 2
#   bash scripts/run_strategy.sh -s hybrid_tp_dp --tp-size 2 --dp-size 2
#   bash scripts/run_strategy.sh -s branch_mp --smoke
#   NGPU=4 bash scripts/run_strategy.sh -s tp

set -euo pipefail
cd "$(dirname "$0")/.."

STRATEGY=""
MODEL="attention"
NGPU="${NGPU:-}"
SMOKE=0
RUN_NAME=""
TP_SIZE=""
DP_SIZE=""
EXTRA=""

usage() {
    cat >&2 <<EOF
usage: $0 -s <strategy> [options]

  -s, --strategy <name>          one of: single ddp ddp_zero fsdp_z2 fsdp_z3
                                 branch_mp tp hybrid_tp_dp
  -n, --ngpu <N>                 number of GPUs (default depends on strategy)
  -m, --model <attention|mamba>  workload (default: attention)
      --smoke                    use the *_smoke.yaml config and add _smoke suffix
  -r, --run-name <name>          override the run directory name
      --tp-size <N>              tp degree (hybrid_tp_dp only; default 2)
      --dp-size <N>              dp degree (hybrid_tp_dp only; default 2)
      --extra "..."              extra args forwarded to train_binding.py
  -h, --help                     show this help

Env vars:
  CUDA_VISIBLE_DEVICES  restricts visible GPUs (honored for all strategies)
  NGPU                  alternative way to set --ngpu

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--strategy)   STRATEGY="$2"; shift 2 ;;
        -m|--model)      MODEL="$2";    shift 2 ;;
        -n|--ngpu)       NGPU="$2";     shift 2 ;;
        -r|--run-name)   RUN_NAME="$2"; shift 2 ;;
        --tp-size)       TP_SIZE="$2";  shift 2 ;;
        --dp-size)       DP_SIZE="$2";  shift 2 ;;
        --extra)         EXTRA="$2";    shift 2 ;;
        --smoke)         SMOKE=1;       shift 1 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$STRATEGY" ]]; then
    echo "error: --strategy is required" >&2
    usage
    exit 2
fi

# ---- default GPU count per strategy ------------------------------------------
case "$STRATEGY" in
    single)         NGPU="${NGPU:-1}" ;;
    branch_mp)      NGPU="${NGPU:-2}" ;;
    tp)             NGPU="${NGPU:-2}" ;;
    hybrid_tp_dp)   NGPU="${NGPU:-4}"; TP_SIZE="${TP_SIZE:-2}"; DP_SIZE="${DP_SIZE:-2}" ;;
    ddp|ddp_zero|fsdp_z2|fsdp_z3) NGPU="${NGPU:-2}" ;;
    *) echo "error: unknown strategy '$STRATEGY'" >&2; exit 2 ;;
esac

# ---- config suffix per strategy ---------------------------------------------
# Maps strategy (+ optional tp degree) to the config file stem.
case "$STRATEGY" in
    single)         SUFFIX="course" ;;
    ddp|ddp_zero)   SUFFIX="course" ;;
    fsdp_z2)        SUFFIX="fsdp_z2_course" ;;
    fsdp_z3)        SUFFIX="fsdp_z3_course" ;;
    branch_mp)      SUFFIX="branch_mp_course" ;;
    tp)             SUFFIX="tp${NGPU}_course" ;;
    hybrid_tp_dp)   SUFFIX="hybrid_tp${TP_SIZE}_dp${DP_SIZE}_course" ;;
esac

if (( SMOKE == 1 )); then
    # attention_course.yaml lives next to attention_smoke.yaml; FSDP/TP/etc
    # have matching *_smoke.yaml siblings.
    case "$STRATEGY" in
        single|ddp|ddp_zero) CFG_STEM="${MODEL}_smoke" ;;
        *) CFG_STEM="${MODEL}_${SUFFIX/_course/_smoke}" ;;
    esac
else
    case "$STRATEGY" in
        single|ddp|ddp_zero) CFG_STEM="${MODEL}_course" ;;
        *) CFG_STEM="${MODEL}_${SUFFIX}" ;;
    esac
fi
CONFIG="configs/${CFG_STEM}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "error: missing config $CONFIG" >&2
    exit 2
fi

# ---- default run-name --------------------------------------------------------
if [[ -z "$RUN_NAME" ]]; then
    case "$STRATEGY" in
        single)       RUN_NAME="${MODEL}_single_course" ;;
        branch_mp)    RUN_NAME="${MODEL}_branch_mp_course" ;;
        hybrid_tp_dp) RUN_NAME="${MODEL}_hybrid_tp${TP_SIZE}_dp${DP_SIZE}_course" ;;
        tp)           RUN_NAME="${MODEL}_tp${NGPU}_course" ;;
        *)            RUN_NAME="${MODEL}_${STRATEGY}_${NGPU}gpu_course" ;;
    esac
    if (( SMOKE == 1 )); then
        RUN_NAME="${RUN_NAME%_course}_smoke"
    fi
fi

# ---- validations -------------------------------------------------------------
if [[ "$STRATEGY" == "hybrid_tp_dp" ]]; then
    if (( TP_SIZE * DP_SIZE != NGPU )); then
        echo "error: tp_size($TP_SIZE) * dp_size($DP_SIZE) != ngpu($NGPU)" >&2
        exit 2
    fi
fi
if [[ "$STRATEGY" == "branch_mp" && "$NGPU" != "2" ]]; then
    echo "error: branch_mp requires exactly 2 GPUs (got $NGPU)" >&2
    exit 2
fi

# ---- refresh compact dataset (no-op if already built) -----------------------
python scripts/build_compact_dataset.py >/dev/null

echo "[run_strategy] strategy=$STRATEGY model=$MODEL ngpu=$NGPU config=$CONFIG run=$RUN_NAME smoke=$SMOKE"

TRAIN_CMD=(scripts/train_binding.py --config "$CONFIG" --strategy "$STRATEGY" --run-name "$RUN_NAME")
if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA )
    TRAIN_CMD+=("${EXTRA_ARR[@]}")
fi

case "$STRATEGY" in
    single|branch_mp)
        # Single Python process; for branch_mp it must see both GPUs.
        if [[ "$STRATEGY" == "branch_mp" ]]; then
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
        else
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        fi
        exec python "${TRAIN_CMD[@]}"
        ;;
    *)
        exec torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "${TRAIN_CMD[@]}"
        ;;
esac
