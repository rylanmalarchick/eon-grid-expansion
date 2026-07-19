"""Congestion-reduction metrics for the headline IEEE 33 plan (brief outcome 3).

Builds the headline instance (IEEE 33, community_bridging, seed 7,
reconfiguration ON, honest Layer A at --time-limit) and evaluates three
configurations with fixed-builds LinDistFlow solves:

  1. status_quo        -- no new lines, reconfiguration OFF
  2. reconfiguration   -- no new lines, reconfiguration ON
  3. plan              -- the Layer A incumbent's line set, reconfiguration ON

reporting the scenario-weighted congestion (total over-limit MW, the D3
primary axis), voltage violation, curtailment, and loss proxy for each, plus
the deltas. The 2 vs 3 split attributes the benefit between switching and
new lines. The plan is the TIME_LIMIT incumbent, not proven optimal -- the
record carries the Layer A gap.

Run from workspace/:
    python scripts/congestion_metrics.py [--time-limit 1800] [--seed 7]
        [--eval-time-limit 600] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    ExpansionResult,
    solve_lindistflow_expansion,
)
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("congestion_metrics")

SCENARIO_KIND = "stressed_five_point"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config(time_limit: float, reconfiguration: bool) -> ExpansionProblemConfig:
    return ExpansionProblemConfig(
        max_new_lines=3,
        time_limit_s=time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=reconfiguration,
    )


def _evaluation_block(result: ExpansionResult) -> dict[str, object]:
    return {
        "termination_status": result.termination_status,
        "objective_value": result.objective_value,
        "aggregate_metrics": result.aggregate_metrics,
        "selected_candidates": list(result.selected_candidates),
        "closed_line_count": result.metadata.get("closed_line_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--eval-time-limit", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/congestion_{stamp}/metrics.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = load_distribution_feeder("ieee33")
    scenarios = build_scenario_set(SCENARIO_KIND)
    plan_config = _config(args.time_limit, reconfiguration=True)
    baseline_stress = _compute_baseline_stress(net, scenarios, plan_config)

    logger.info("headline Layer A solve (ieee33 community_bridging seed=%d, %.0fs)",
                args.seed, args.time_limit)
    candidates, layer_a, _surrogate = build_instance_surrogate(
        net,
        scenarios,
        plan_config,
        family="community_bridging",
        candidate_count=24,
        neighborhood_size=20,
        seed=args.seed,
        time_limit=args.time_limit,
        cost_per_km=10_000.0,
        baseline_stress=baseline_stress,
    )
    plan_builds = {c.name: int(c.name in layer_a.selected_candidates) for c in candidates}
    no_builds = {c.name: 0 for c in candidates}
    logger.info("incumbent plan: %s (layer A %s gap=%s)",
                sorted(layer_a.selected_candidates), layer_a.termination_status,
                layer_a.mip_gap)

    evaluations: dict[str, ExpansionResult] = {}
    for label, builds, reconfiguration in (
        ("status_quo", no_builds, False),
        ("reconfiguration_only", no_builds, True),
        ("plan", plan_builds, True),
    ):
        logger.info("evaluating %s", label)
        evaluations[label] = solve_lindistflow_expansion(
            net,
            scenarios,
            candidates,
            _config(args.eval_time_limit, reconfiguration),
            fixed_builds=builds,
        )

    def congestion(label: str) -> float:
        return float(evaluations[label].aggregate_metrics["weighted_congestion_mw"])

    record = {
        "instance": f"ieee33:community_bridging:seed{args.seed}",
        "layer_a": {
            "termination_status": layer_a.termination_status,
            "mip_gap": layer_a.mip_gap,
            "objective_value": layer_a.objective_value,
            "selected_candidates": list(layer_a.selected_candidates),
        },
        "evaluations": {label: _evaluation_block(r) for label, r in evaluations.items()},
        "congestion_reduction_mw": {
            "plan_vs_status_quo": congestion("status_quo") - congestion("plan"),
            "plan_vs_reconfiguration_only": congestion("reconfiguration_only")
            - congestion("plan"),
            "reconfiguration_vs_status_quo": congestion("status_quo")
            - congestion("reconfiguration_only"),
        },
        "run": {
            "time_limit_s": args.time_limit,
            "eval_time_limit_s": args.eval_time_limit,
            "git_commit": _git_commit(),
            "timestamp_utc": stamp,
            "note": "plan = Layer A TIME_LIMIT incumbent, not proven optimal",
        },
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    logger.info(
        "congestion MW: status_quo=%.4f reconfig_only=%.4f plan=%.4f -> %s",
        congestion("status_quo"),
        congestion("reconfiguration_only"),
        congestion("plan"),
        out_path,
    )


if __name__ == "__main__":
    main()
