#!/usr/bin/env bash
# Reproduce every headline result from a clean checkout, in order.
#
#   ./scripts/reproduce_headline.sh quick   # wiring check, short time limits
#   ./scripts/reproduce_headline.sh full    # the honest solve times (hours)
#
# Outputs land under experiments/results/repro_<mode>_<stamp>/. Every script
# already emits provenance records. Run from workspace/ with the stable
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

# Preflight. Without this the script prints "== 1/11" and then dies on
# "python: command not found" -- eleven steps of work abandoned after the
# banner, which is exactly the failure a reproduction script must not have.
if ! command -v "${PY}" >/dev/null 2>&1; then
  echo "ERROR: interpreter '${PY}' not found." >&2
  echo "  Activate the project environment, or set PYTHON=/path/to/python." >&2
  exit 1
fi
if ! "${PY}" -c "import eon" >/dev/null 2>&1; then
  echo "ERROR: '${PY}' cannot import the eon package." >&2
  echo "  Install it first:  pip install -e '.[dev]'" >&2
  exit 1
fi
echo "== interpreter: $(command -v "${PY}")"
echo "== reproduce_headline mode=${MODE} -> ${OUT}"

echo "== 1/11 feeder hardness (reconfiguration ON vs OFF, IEEE 33)"
${PY} scripts/reconfiguration_sweep.py --feeders ieee33 \
  --families community_bridging --seeds 7,24 --time-limit "${LAYER_A_LIMIT}" \
  ${WITH_MPS_FLAG} --qaoa-rounds 1,2,3 --out "${OUT}/feeder_on.jsonl"
${PY} scripts/reconfiguration_sweep.py --feeders ieee33 \
  --families community_bridging --seeds 7,24 --time-limit "${LAYER_A_LIMIT}" \
  --no-reconfiguration --out "${OUT}/feeder_off.jsonl"

echo "== 2/11 Path B scale tier (dense glasses)"
${PY} scripts/pathb_spine.py --time-limit "${GLASS_LIMIT}" --seeds 7,24 \
  --fused-sizes "" --glass-sizes "${GLASS_SIZES}" --glass-mean-degree 0 \
  --glass-coefficient-range=-10,10 --out "${OUT}/pathb_dense.jsonl"

echo "== 3/11 QAOA landscape (cop vs vanilla vs warm)"
${PY} scripts/qaoa_landscape.py --time-limit "${LAYER_A_LIMIT}" \
  --depth "${DEPTH}" --shots 1024 --nm-evals "${NM_EVALS}" \
  --out "${OUT}/landscape.jsonl"

echo "== 4/11 decomposition + D14 certificate"
${PY} scripts/qaoa_decomposition_run.py --time-limit "${LAYER_A_LIMIT}" \
  --block-size 10 --p 1 --out "${OUT}/decomposition.jsonl"

echo "== 5/11 scale-tier MPS sweep"
${PY} scripts/scale_mps_sweep.py --sizes "${MPS_SIZES}" --seeds 7,24 \
  --chi-values "${MPS_CHI}" --gurobi-time-limit 600 \
  --out "${OUT}/scale_mps.jsonl"

echo "== 6/11 congestion metrics"
${PY} scripts/congestion_metrics.py --time-limit "${LAYER_A_LIMIT}" \
  --eval-time-limit 600 --seed 7 --out "${OUT}/congestion.json"

echo "== 7/11 random-feasible control (S3), both subspace regimes"
# The finding needs BOTH runs. At the derived weight the subspace is 190 states
# against 1024 shots, so sampling is near-exhaustive and cop-QAOA ties a random
# draw. At weight 6 the subspace is 38,760 states, and the comparison is about
# search quality rather than coverage. Reporting only one regime misstates the
# result in whichever direction that regime happens to favor.
${PY} scripts/random_feasible_control.py --time-limit "${LAYER_A_LIMIT}" \
  --shots 1024 --repeats 25 --seeds 7 --depth 2 \
  --out "${OUT}/s3_control.jsonl"
${PY} scripts/random_feasible_control.py --time-limit "${LAYER_A_LIMIT}" \
  --shots 1024 --repeats 25 --seeds 7 --depth 2 --hamming-weight 6 \
  --out "${OUT}/s3_control_w6.jsonl"

echo "== 8/11 logical-vs-physical transpile table (S4)"
${PY} scripts/transpile_table.py --depth 2 --levels 0,1 \
  --out "${OUT}/s4_transpile.json"

${PY} scripts/gate_count_floor.py --table "${OUT}/s4_transpile.json" \
  --out "${OUT}/gate_floor.json"

echo "== 9/11 Qiskit cross-check (D5)"
${PY} scripts/qiskit_validation.py --shots 4096 --depth 2 \
  --out "${OUT}/qiskit_validation.json"

echo "== 10/11 hardness figure (S1 family)"
# The instance family is a SEPARATE sweep -- 12 Layer A solves at 1800 s each,
# far too slow to sit inside this pipeline -- so its records live in their
# canonical location, not this run's output directory. Skip cleanly when they
# are absent; the previous version pointed at "${OUT}"/s1_proper and printed a
# FileNotFoundError traceback on every clean run, which reads as breakage.
S1_ON="experiments/results/s1_proper"
S1_OFF="experiments/results/s1_proper_off"
if compgen -G "${S1_ON}/*.jsonl" > /dev/null; then
  OFF_ARGS=""
  if compgen -G "${S1_OFF}/*.jsonl" > /dev/null; then
    OFF_ARGS="--off ${S1_OFF}/*.jsonl"
  fi
  # shellcheck disable=SC2086
  ${PY} scripts/make_hardness_figure.py \
    --on ${S1_ON}/*.jsonl ${OFF_ARGS} --out-dir "${OUT}/figures"
else
  echo "   skipped: no records in ${S1_ON}"
  echo "   generate them with scripts/reconfiguration_sweep.py (see docs/results_index.md)"
fi

echo "== 11/11 figures"
${PY} scripts/make_coverage_figure.py \
  --records "${OUT}"/s3_control.jsonl "${OUT}"/s3_control_w6.jsonl \
  --out-dir "${OUT}/figures"
${PY} scripts/make_landscape_figure.py --results "${OUT}/landscape.jsonl" \
  --out-dir "${OUT}/figures"
${PY} scripts/make_plan_figure.py --metrics "${OUT}/congestion.json" \
  --out-dir "${OUT}/figures"
${PY} scripts/make_pipeline_figure.py --out-dir "${OUT}/figures"

echo "== done: ${OUT}"
