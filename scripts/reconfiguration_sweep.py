"""Joint expansion+reconfiguration hardness sweep at n>=20 (PLAN.txt section 6).

Builds instances with reconfiguration on (--no-reconfiguration for the OFF
baseline), records the gate signals per instance -- Layer A status + MIP gap +
builds, effective tree-width + J/h, GTN contraction width, Layer B gap, and (with
--with-mps) the MPS sweep -- and appends one JSON line per instance as it completes.
Read the Path-A signal from the effective (load-bearing) tree-width and J/h (compare
ON vs OFF) and the Layer A gap; the GTN contraction width is penalty-saturated at
n=20 and does not discriminate.

Run from workspace/:
    python scripts/reconfiguration_sweep.py [--time-limit S] [--feeders a,b]
        [--families a,b] [--seeds 7,24] [--neighborhood 20] [--with-mps]
        [--no-reconfiguration] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eon.formulations.layer_b import solve_layer_b_surrogate
from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set
from eon.instances.treewidth import coupling_diagnostics
from eon.mps.protocol import run_mps_protocol, run_tree_tn_control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reconfiguration_sweep")

CANDIDATE_COUNT = 24
MAX_NEW_LINES = 3
COST_PER_KM = 10_000.0
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


def run_instance(
    feeder: str,
    family: str,
    seed: int,
    *,
    neighborhood_size: int,
    time_limit: float,
    with_mps: bool,
    reconfiguration: bool,
    net: object,
    scenarios: list,
    baseline_stress: dict[int, float],
    run_meta: dict[str, object],
) -> dict[str, object]:
    config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES,
        time_limit_s=time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=reconfiguration,
    )
    candidates, layer_a, surrogate = build_instance_surrogate(
        net,
        scenarios,
        config,
        family=family,
        candidate_count=CANDIDATE_COUNT,
        neighborhood_size=neighborhood_size,
        seed=seed,
        time_limit=time_limit,
        cost_per_km=COST_PER_KM,
        baseline_stress=baseline_stress,
    )
    coupling = coupling_diagnostics(surrogate)
    layer_b = solve_layer_b_surrogate(surrogate, time_limit_s=time_limit, mip_gap=0.01)
    instance_id = f"{feeder}:{family}:seed{seed}:n{coupling.variable_count}"

    # The full MPS sweep (D5) is expensive; the structural Path-A question (does
    # reconfiguration push the contraction width above the chi budget?) needs only
    # the exact-TN control. Run the full sweep only when --with-mps is set.
    mps_section: dict[str, object] | None = None
    if with_mps:
        mps = run_mps_protocol(
            surrogate,
            instance_name=instance_id,
            reference_energy=layer_b.objective_value,
            seed=seed,
        )
        tree = mps.tree_tn_result
        ordering_gaps = [o.best_energy - mps.reference_energy for o in mps.ordering_results]
        mps_section = {
            "backend": mps.backend,
            "chi_max_reached": mps.chi_max_reached,
            "reference_energy": mps.reference_energy,
            "exact_ground_energy": mps.exact_ground_energy,
            "ordering_best_energies": [o.best_energy for o in mps.ordering_results],
            "ordering_gaps_vs_reference": ordering_gaps,
        }
    else:
        tree = run_tree_tn_control(
            surrogate, seed=seed, reference_energy=layer_b.objective_value
        )

    # Tree-TN-negative = the strong exact-TN control could NOT contract within budget
    # (within_budget=False). Contraction width alone is penalty-saturated at n=20 (the
    # same for reconfiguration ON and OFF), so it does NOT discriminate; the honest
    # Path-A signals to read in the record are effective_treewidth + J/h (load-bearing
    # coupling, ON vs OFF) and the Layer A Gurobi gap, not this flag.
    tree_negative_candidate = bool(
        tree is not None
        and tree.backend == "generic_tn_tropical"
        and not coupling.mps_easy
        and not tree.within_budget
    )

    return {
        "instance_id": instance_id,
        "feeder": feeder,
        "family": family,
        "seed": seed,
        "candidate_count": CANDIDATE_COUNT,
        "neighborhood_size": neighborhood_size,
        "enable_reconfiguration": reconfiguration,
        "with_mps": with_mps,
        "layer_a": {
            "termination_status": layer_a.termination_status,
            "mip_gap": layer_a.mip_gap,
            "best_bound": layer_a.best_bound,
            "objective_value": layer_a.objective_value,
            "runtime_s": layer_a.runtime_s,
            "n_builds": len(layer_a.selected_candidates),
            "selected_candidates": list(layer_a.selected_candidates),
            "closed_line_count": layer_a.metadata.get("closed_line_count"),
        },
        "coupling": {
            "variable_count": coupling.variable_count,
            "structural_treewidth": coupling.structural_treewidth,
            "effective_treewidth": coupling.effective_treewidth,
            # J/h reported next to the Layer A gap above: a J/h from a high-gap solve
            # is suboptimality-inflated (gap-robust reading).
            "coupling_field_ratio": coupling.coupling_field_ratio,
            "mps_easy": coupling.mps_easy,
        },
        "tree_tn": None
        if tree is None
        else {
            "backend": tree.backend,
            "contraction_width": tree.contraction_width,
            "within_budget": tree.within_budget,
            "exact_ground_energy": tree.best_energy if tree.within_budget else None,
            "chi_max_reached": tree.chi_max_reached,
        },
        "layer_b": {
            "objective_value": layer_b.objective_value,
            "mip_gap": layer_b.mip_gap,
            "status": layer_b.status,
        },
        "mps": mps_section,
        "signals": {
            "tree_negative_candidate": tree_negative_candidate,
            # The exact Gurobi terminal status is in layer_a.termination_status; this
            # flag is just "not proven optimal" (TIME_LIMIT in practice, but could be
            # INFEASIBLE / NUMERIC -- read the status, not this label, to be sure).
            "layer_a_not_optimal": layer_a.termination_status != "OPTIMAL",
            "layer_b_gap_open": (layer_b.mip_gap or 0.0) > 0.01,
        },
        "run": run_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--neighborhood", type=int, default=20)
    parser.add_argument("--feeders", default="ieee33,ieee123")
    parser.add_argument("--families", default="community_bridging,useful_adversarial")
    parser.add_argument("--seeds", default="7,24")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--with-mps",
        action="store_true",
        help="Also run the full MPS sweep (slow). Default: GTN exact-TN control only.",
    )
    parser.add_argument(
        "--no-reconfiguration",
        dest="reconfiguration",
        action="store_false",
        help="Build with reconfiguration OFF (the baseline for the ON/OFF comparison).",
    )
    args = parser.parse_args()

    feeders = [f.strip() for f in args.feeders.split(",") if f.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/reconfig_sweep_{stamp}/sweep.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta: dict[str, object] = {
        "time_limit_s": args.time_limit,
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
        "scenario_kind": SCENARIO_KIND,
    }

    grid = [(f, fam, s) for f in feeders for fam in families for s in seeds]
    logger.info("sweep: %d instances -> %s", len(grid), out_path)

    scenarios = build_scenario_set(SCENARIO_KIND)
    baseline_config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES, time_limit_s=args.time_limit, n_scenarios_aggregated=3
    )
    nets: dict[str, object] = {}
    baseline_stress: dict[str, dict[int, float]] = {}
    for feeder in feeders:
        nets[feeder] = load_distribution_feeder(feeder)
        baseline_stress[feeder] = _compute_baseline_stress(
            nets[feeder], scenarios, baseline_config
        )

    ok = 0
    failed = 0
    with out_path.open("w") as handle:
        for index, (feeder, family, seed) in enumerate(grid, start=1):
            logger.info("[%d/%d] %s %s seed=%d", index, len(grid), feeder, family, seed)
            try:
                record = run_instance(
                    feeder,
                    family,
                    seed,
                    neighborhood_size=args.neighborhood,
                    time_limit=args.time_limit,
                    with_mps=args.with_mps,
                    reconfiguration=args.reconfiguration,
                    net=nets[feeder],
                    scenarios=scenarios,
                    baseline_stress=baseline_stress[feeder],
                    run_meta=run_meta,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001 -- long sweep must record and continue
                logger.exception("instance failed: %s %s seed=%d", feeder, family, seed)
                record = {
                    "instance_id": f"{feeder}:{family}:seed{seed}",
                    "feeder": feeder,
                    "family": family,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "run": run_meta,
                }
                failed += 1
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            tn = record.get("signals", {}).get("tree_negative_candidate")  # type: ignore[union-attr]
            cw = (record.get("tree_tn") or {}).get("contraction_width")  # type: ignore[union-attr]
            logger.info(
                "  -> contraction_width=%s tree_negative_candidate=%s", cw, tn
            )

    logger.info("done: %d ok, %d failed of %d -> %s", ok, failed, len(grid), out_path)
    if ok == 0:
        raise SystemExit(
            f"reconfiguration_sweep: all {failed} instances failed; see {out_path}"
        )


if __name__ == "__main__":
    main()
