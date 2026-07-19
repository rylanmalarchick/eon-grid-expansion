"""Coupling robustness at honest solve time (PLAN.txt section 6 item A).

The 2026-06-28 item-2 run found the IEEE 123 reconfiguration eff_tw flipped 9->1
between two 180s runs of the SAME instance -- the Layer A solve was suboptimal and
its incumbent varied, so the reduced-QUBO coupling varied. This re-builds the same
instance several times at a longer solve time and reports the eff_tw / J/h / Layer A
gap each time, so the spread tells us whether honest solve time gives robust
load-bearing coupling on IEEE 123 (IEEE 33 is the robust control).

Run from workspace/:
    python scripts/coupling_robustness.py [--time-limit S] [--repeats N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import _compute_baseline_stress, build_instance_surrogate
from eon.instances.scenarios import build_scenario_set
from eon.instances.treewidth import coupling_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("coupling_robustness")

CANDIDATE_COUNT = 24
NEIGHBORHOOD_SIZE = 20
MAX_NEW_LINES = 3
COST_PER_KM = 10_000.0
SCENARIO_KIND = "stressed_five_point"
# (feeder, family, seed, repeats): the IEEE 123 fragility under test + the IEEE 33
# control that was robust at 180s.
GRID = [
    ("ieee123", "community_bridging", 7, 3),
    ("ieee33", "community_bridging", 7, 2),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/coupling_robustness_{stamp}/runs.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenario_set(SCENARIO_KIND)
    config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES,
        time_limit_s=args.time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=True,
    )
    baseline_config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES, time_limit_s=args.time_limit, n_scenarios_aggregated=3
    )
    ok = 0
    failed = 0
    with out_path.open("w") as handle:
        for feeder, family, seed, repeats in GRID:
            net = load_distribution_feeder(feeder)
            baseline_stress = _compute_baseline_stress(net, scenarios, baseline_config)
            for rep in range(repeats):
                logger.info("%s %s seed=%d rep=%d/%d", feeder, family, seed, rep + 1, repeats)
                try:
                    _candidates, layer_a, surrogate = build_instance_surrogate(
                        net,
                        scenarios,
                        config,
                        family=family,
                        candidate_count=CANDIDATE_COUNT,
                        neighborhood_size=NEIGHBORHOOD_SIZE,
                        seed=seed,
                        time_limit=args.time_limit,
                        cost_per_km=COST_PER_KM,
                        baseline_stress=baseline_stress,
                    )
                    coupling = coupling_diagnostics(surrogate)
                    record = {
                        "feeder": feeder,
                        "family": family,
                        "seed": seed,
                        "rep": rep,
                        "time_limit_s": args.time_limit,
                        "layer_a_status": layer_a.termination_status,
                        "layer_a_mip_gap": layer_a.mip_gap,
                        "n_builds": len(layer_a.selected_candidates),
                        "variable_count": coupling.variable_count,
                        "structural_treewidth": coupling.structural_treewidth,
                        "effective_treewidth": coupling.effective_treewidth,
                        "coupling_field_ratio": coupling.coupling_field_ratio,
                        "mps_easy": coupling.mps_easy,
                    }
                    ok += 1
                # Broad catch: a long run must record the failure and continue.
                except Exception as exc:
                    logger.exception("run failed: %s %s seed=%d rep=%d", feeder, family, seed, rep)
                    record = {
                        "feeder": feeder,
                        "family": family,
                        "seed": seed,
                        "rep": rep,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failed += 1
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                logger.info(
                    "  -> eff_tw=%s J/h=%s gap=%s",
                    record.get("effective_treewidth"),
                    record.get("coupling_field_ratio"),
                    record.get("layer_a_mip_gap"),
                )

    logger.info("done: %d ok, %d failed -> %s", ok, failed, out_path)
    if ok == 0:
        raise SystemExit(f"coupling_robustness: all {failed} runs failed; see {out_path}")


if __name__ == "__main__":
    main()
