#!/usr/bin/env bash
# Reproduce every headline result from a clean checkout, in order.
#
#   ./scripts/reproduce_headline.sh quick   # wiring check, short time limits
#   ./scripts/reproduce_headline.sh full    # the honest solve times (hours)
#
# Outputs land under experiments/results/repro_<mode>_<stamp>/. Every script
# already emits agentbible provenance. Run from workspace/ with the stable
# env active and GRB_LICENSE_FILE set. QUICK numbers are a wiring check only
# and must never be quoted; FULL reproduces the numbers in the proposal.

set -euo pipefail

MODE="${1:-quick}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="experiments/results/repro_${MODE}_${STAMP}"
mkdir -p "${OUT}"

case "${MODE}" in
  quick)
    LAYER_A_LIMIT=60
    GLASS_LIMIT=60
    GLASS_SIZES="80"
    DEPTH=2
    NM_EVALS=10
    MPS_SIZES="40"
    MPS_CHI="4,8"
    WITH_MPS_FLAG=""
    ;;
  full)
    LAYER_A_LIMIT=1800
    GLASS_LIMIT=3600
    GLASS_SIZES="80,120,160"
    DEPTH=3
    NM_EVALS=60
    MPS_SIZES="80,120,160"
    MPS_CHI="4,8,16,32,64"
    WITH_MPS_FLAG="--with-mps"
    ;;
  *)
    echo "usage: $0 [quick|full]" >&2
    exit 2
    ;;
esac

PY="${PYTHON:-python}"
echo "== reproduce_headline mode=${MODE} -> ${OUT}"

echo "== 1/6 feeder hardness (reconfiguration ON vs OFF, IEEE 33)"
${PY} scripts/reconfiguration_sweep.py --feeders ieee33 \
  --families community_bridging --seeds 7,24 --time-limit "${LAYER_A_LIMIT}" \
  ${WITH_MPS_FLAG} --qaoa-rounds 1,2,3 --out "${OUT}/feeder_on.jsonl"
${PY} scripts/reconfiguration_sweep.py --feeders ieee33 \
  --families community_bridging --seeds 7,24 --time-limit "${LAYER_A_LIMIT}" \
  --no-reconfiguration --out "${OUT}/feeder_off.jsonl"

echo "== 2/6 Path B scale tier (dense glasses)"
${PY} scripts/pathb_spine.py --time-limit "${GLASS_LIMIT}" --seeds 7,24 \
  --fused-sizes "" --glass-sizes "${GLASS_SIZES}" --glass-mean-degree 0 \
  --glass-coefficient-range=-10,10 --out "${OUT}/pathb_dense.jsonl"

echo "== 3/6 QAOA landscape (cop vs vanilla vs warm)"
${PY} scripts/qaoa_landscape.py --time-limit "${LAYER_A_LIMIT}" \
  --depth "${DEPTH}" --shots 1024 --nm-evals "${NM_EVALS}" \
  --out "${OUT}/landscape.jsonl"

echo "== 4/6 decomposition + D14 certificate"
${PY} scripts/qaoa_decomposition_run.py --time-limit "${LAYER_A_LIMIT}" \
  --block-size 10 --p 1 --out "${OUT}/decomposition.jsonl"

echo "== 5/6 scale-tier MPS sweep"
${PY} scripts/scale_mps_sweep.py --sizes "${MPS_SIZES}" --seeds 7 \
  --chi-values "${MPS_CHI}" --gurobi-time-limit "${GLASS_LIMIT}" \
  --out "${OUT}/scale_mps.jsonl"

echo "== 6/6 congestion metrics + figures"
${PY} scripts/congestion_metrics.py --time-limit "${LAYER_A_LIMIT}" \
  --eval-time-limit "${LAYER_A_LIMIT}" --seed 7 --out "${OUT}/congestion.json"
${PY} scripts/make_landscape_figure.py --results "${OUT}/landscape.jsonl" \
  --out-dir "${OUT}/figures"

echo "== done: ${OUT}"
