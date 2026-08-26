"""Joint expansion+reconfiguration hardness sweep at n>=20 (PLAN.txt section 6).

Builds instances with reconfiguration on (--no-reconfiguration for the OFF
baseline), records the gate signals per instance -- Layer A status + MIP gap +
builds, effective tree-width + J/h, GTN contraction width, Layer B gap, and (with
--with-mps) the MPS sweep -- and appends one JSON line per instance as it completes.
Read the Path-A signal from the effective (load-bearing) tree-width and J/h (compare
ON vs OFF) and the Layer A gap; the GTN contraction width is penalty-saturated at
n=20 and does not discriminate.

With --with-mps, --qaoa-rounds takes a comma list (e.g. 1,2,3): Layer A is solved
ONCE per instance and the MPS sweep runs once per p on that single surrogate, one
JSON line per (instance, p). Separate runs are NOT a controlled p-comparison --
the Gurobi TIME_LIMIT incumbent varies run-to-run and the reduced QUBO with it.

Run from workspace/:
    python scripts/reconfiguration_sweep.py [--time-limit S] [--feeders a,b]
        [--families a,b] [--seeds 7,24] [--neighborhood 20] [--with-mps]
        [--no-reconfiguration] [--qaoa-rounds 1,2,3] [--out PATH] [--log-file PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from eon.formulations.layer_b import LayerBSolution, solve_layer_b_surrogate
from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    ExpansionResult,
    solve_lindistflow_expansion,
    suggested_flow_big_m_mva,
)
from eon.instances.candidate_lines import generate_candidate_lines, require_seed_diversification
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set
from eon.instances.treewidth import CouplingDiagnostics, coupling_diagnostics
from eon.mps.protocol import TreeTensorControlResult, run_mps_protocol, run_tree_tn_control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reconfiguration_sweep")

DEFAULT_CANDIDATE_COUNT = 24
MAX_NEW_LINES = 3
COST_PER_KM = 10_000.0
SCENARIO_KIND = "stressed_five_point"


def _finite(value: float | None) -> float | None:
    """JSON has no Infinity/NaN; serialize a non-finite energy as null."""
    return value if value is not None and math.isfinite(value) else None


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
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    time_limit: float,
    with_mps: bool,
    reconfiguration: bool,
    qaoa_rounds_list: list[int],
    angle_iterations: int,
    rank_jitter: float,
    layer_a_only: bool = False,
    net: object = None,
    scenarios: list,
    baseline_stress: dict[int, float],
    run_meta: dict[str, object],
) -> Iterator[dict[str, object]]:
    """Yield one record per QAOA depth p, all on ONE Layer A solve / surrogate."""
    config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES,
        time_limit_s=time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=reconfiguration,
        # Sized to the feeder: the head line carries all demand, and the
        # IEEE-33-scale default cannot represent an MV feeder's flows at all
        # (silent INFEASIBLE, 2026-08-01). No-op for IEEE 33/123.
        flow_big_m_mva=suggested_flow_big_m_mva(net, scenarios),
    )
    if layer_a_only:
        # Solve Layer A and stop. The surrogate build below is ~140 further MILP
        # evaluations per instance, and its only consumer is the coupling
        # diagnostic this project has withdrawn as unsound, so paying for it
        # when the question is solver hardness buys nothing.
        candidates = generate_candidate_lines(
            net,
            family,
            candidate_count,
            seed=seed,
            cost_per_km=COST_PER_KM,
            stress_info=baseline_stress if family == "useful_adversarial" else None,
            rank_jitter=rank_jitter,
        )
        layer_a = solve_lindistflow_expansion(net, scenarios, candidates, config)
        yield {
            "instance_id": f"{feeder}:{family}:seed{seed}:layerA",
            "feeder": feeder,
            "family": family,
            "seed": seed,
            "candidate_count": candidate_count,
            "neighborhood_size": neighborhood_size,
            "enable_reconfiguration": reconfiguration,
            "with_mps": False,
            "layer_a_only": True,
            "layer_a": {
                "termination_status": layer_a.termination_status,
                "mip_gap": layer_a.mip_gap,
                "best_bound": layer_a.best_bound,
                "objective_value": layer_a.objective_value,
                "runtime_s": layer_a.runtime_s,
                "n_builds": len(layer_a.selected_candidates),
                "selected_candidates": list(layer_a.selected_candidates),
            },
            "coupling": None,
            "layer_b": None,
            "tree_tn": None,
            "mps": None,
            "run": run_meta,
        }
        return

    candidates, layer_a, surrogate = build_instance_surrogate(
        net,
        scenarios,
        config,
        family=family,
        candidate_count=candidate_count,
        neighborhood_size=neighborhood_size,
        seed=seed,
        time_limit=time_limit,
        cost_per_km=COST_PER_KM,
        baseline_stress=baseline_stress,
        rank_jitter=rank_jitter,
    )
    coupling = coupling_diagnostics(surrogate)
    layer_b = solve_layer_b_surrogate(surrogate, time_limit_s=time_limit, mip_gap=0.01)
    instance_id = f"{feeder}:{family}:seed{seed}:n{coupling.variable_count}"

    if not with_mps:
        # The structural Path-A question needs only the exact-TN control; p is
        # meaningless without the MPS sweep (recorded as qaoa_rounds=None).
        tree = run_tree_tn_control(
            surrogate, seed=seed, reference_energy=layer_b.objective_value
        )
        yield _make_record(
            instance_id,
            feeder,
            family,
            seed,
            neighborhood_size=neighborhood_size,
            candidate_count=candidate_count,
            reconfiguration=reconfiguration,
            with_mps=with_mps,
            layer_a=layer_a,
            coupling=coupling,
            layer_b=layer_b,
            tree=tree,
            mps_section=None,
            qaoa_rounds=None,
            run_meta=run_meta,
        )
        return

    for qaoa_rounds in qaoa_rounds_list:
        mps = run_mps_protocol(
            surrogate,
            instance_name=instance_id,
            reference_energy=layer_b.objective_value,
            seed=seed,
            qaoa_rounds=qaoa_rounds,
            angle_iterations=angle_iterations,
        )
        tree = mps.tree_tn_result
        ordering_gaps = [o.best_energy - mps.reference_energy for o in mps.ordering_results]
        mps_section: dict[str, object] = {
            "backend": mps.backend,
            "chi_max_reached": mps.chi_max_reached,
            "reference_energy": _finite(mps.reference_energy),
            "exact_ground_energy": _finite(mps.exact_ground_energy),
            # The MPS-negativity signal: best chi=64 MPS energy vs the exact ground
            # (from the GTN control, tree_tn.exact_ground_energy). A positive gap means
            # bounded-chi MPS misses the optimum -- the brief's outcome 4.
            "best_energy_over_orderings": _finite(
                min((o.best_energy for o in mps.ordering_results), default=float("inf"))
            ),
            "orderings": [
                {
                    "ordering": o.ordering,
                    "best_energy": _finite(o.best_energy),
                    "chi_curve": {
                        str(chi): _finite(energy) for chi, energy in sorted(o.chi_curve.items())
                    },
                    "max_entropy": _finite(max(o.entropy_curve.values(), default=0.0)),
                }
                for o in mps.ordering_results
            ],
            "ordering_gaps_vs_reference": [_finite(g) for g in ordering_gaps],
            # The reference is the UNPENALIZED Layer-B objective; MPS/TN
            # energies are of the Hess-PENALIZED compiled QUBO, whose exact
            # ground sits one penalty unit above the reference. Quote gaps vs
            # tree_tn.exact_ground_energy (same compiled object), not this.
            "reference_caveat": "reference is unpenalized Layer B; compare vs "
            "tree_tn.exact_ground_energy for same-object gaps",
        }
        yield _make_record(
            instance_id,
            feeder,
            family,
            seed,
            neighborhood_size=neighborhood_size,
            candidate_count=candidate_count,
            reconfiguration=reconfiguration,
            with_mps=with_mps,
            layer_a=layer_a,
            coupling=coupling,
            layer_b=layer_b,
            tree=tree,
            mps_section=mps_section,
            qaoa_rounds=qaoa_rounds,
            run_meta=run_meta,
        )


def _make_record(
    instance_id: str,
    feeder: str,
    family: str,
    seed: int,
    *,
    neighborhood_size: int,
    reconfiguration: bool,
    with_mps: bool,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    layer_a: ExpansionResult,
    coupling: CouplingDiagnostics,
    layer_b: LayerBSolution,
    tree: TreeTensorControlResult | None,
    mps_section: dict[str, object] | None,
    qaoa_rounds: int | None,
    run_meta: dict[str, object],
) -> dict[str, object]:
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
        "candidate_count": candidate_count,
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
        "run": {**run_meta, "qaoa_rounds": qaoa_rounds},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--neighborhood", type=int, default=20)
    # Must exceed --neighborhood, or the surrogate is silently clamped to the
    # candidate count and the run duplicates a smaller instance size.
    parser.add_argument(
        "--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT
    )
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
    parser.add_argument(
        "--qaoa-rounds",
        default="1",
        help="QAOA p (MPS sweep depth); comma list sweeps p on ONE Layer A solve.",
    )
    parser.add_argument(
        "--angle-iterations", type=int, default=10, help="QAOA angle-optimization iterations."
    )
    parser.add_argument(
        "--layer-a-only",
        action="store_true",
        help="Record the Layer A solve and skip the Layer B surrogate. The "
        "surrogate build is ~140 extra MILP evaluations per instance and its "
        "only consumer is the coupling diagnostic, which this project has "
        "withdrawn as unsound. Use when the question is solver hardness.",
    )
    parser.add_argument(
        "--rank-jitter",
        type=float,
        default=0.0,
        help="Perturb the candidate ranking's tie-breaks so seeds genuinely "
        "diversify (0.0 = deterministic families, seeds are a NO-OP).",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Log file path (default: <out>.log next to the JSONL) so a killed run "
        "leaves a forensic trail.",
    )
    args = parser.parse_args()

    feeders = [f.strip() for f in args.feeders.split(",") if f.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    # Fail at launch, not after hours of compute whose records cannot honestly
    # be labelled. Every family in the sweep has to be able to diversify.
    for family in families:
        require_seed_diversification(seeds, family, rank_jitter=args.rank_jitter)
    qaoa_rounds_list = [int(p) for p in str(args.qaoa_rounds).split(",") if p.strip()]
    if not qaoa_rounds_list or any(p < 1 for p in qaoa_rounds_list):
        raise SystemExit(f"--qaoa-rounds must be positive ints, got {args.qaoa_rounds!r}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/reconfig_sweep_{stamp}/sweep.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    run_meta: dict[str, object] = {
        "time_limit_s": args.time_limit,
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
        "scenario_kind": SCENARIO_KIND,
        "rank_jitter": args.rank_jitter,
        "layer_a_only": args.layer_a_only,
        "seed_diversifies": args.rank_jitter > 0.0,
    }

    grid = [(f, fam, s) for f in feeders for fam in families for s in seeds]
    logger.info(
        "sweep: %d instances x p in %s -> %s (log: %s)",
        len(grid),
        qaoa_rounds_list,
        out_path,
        log_path,
    )

    scenarios = build_scenario_set(SCENARIO_KIND)
    baseline_config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES, time_limit_s=args.time_limit, n_scenarios_aggregated=3
    )
    nets: dict[str, object] = {}
    baseline_stress: dict[str, dict[int, float]] = {}
    for feeder in feeders:
        nets[feeder] = load_distribution_feeder(feeder)
        baseline_stress[feeder] = _compute_baseline_stress(
            nets[feeder],
            scenarios,
            replace(
                baseline_config,
                flow_big_m_mva=suggested_flow_big_m_mva(nets[feeder], scenarios),
            ),
        )

    ok = 0
    failed = 0
    with out_path.open("w") as handle:
        for index, (feeder, family, seed) in enumerate(grid, start=1):
            logger.info("[%d/%d] %s %s seed=%d", index, len(grid), feeder, family, seed)
            try:
                # Generator: records stream out per p; a crash mid-sweep keeps the
                # p-values already written (the empty-p2.jsonl lesson).
                for record in run_instance(
                    feeder,
                    family,
                    seed,
                    neighborhood_size=args.neighborhood,
                    candidate_count=args.candidate_count,
                    time_limit=args.time_limit,
                    with_mps=args.with_mps,
                    reconfiguration=args.reconfiguration,
                    qaoa_rounds_list=qaoa_rounds_list,
                    angle_iterations=args.angle_iterations,
                    rank_jitter=args.rank_jitter,
                    layer_a_only=args.layer_a_only,
                    net=nets[feeder],
                    scenarios=scenarios,
                    baseline_stress=baseline_stress[feeder],
                    run_meta=run_meta,
                ):
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    run_block = cast(dict[str, object], record.get("run") or {})
                    tree_block = cast(dict[str, object], record.get("tree_tn") or {})
                    signal_block = cast(dict[str, object], record.get("signals") or {})
                    logger.info(
                        "  -> p=%s contraction_width=%s tree_negative_candidate=%s",
                        run_block.get("qaoa_rounds"),
                        tree_block.get("contraction_width"),
                        signal_block.get("tree_negative_candidate"),
                    )
                ok += 1
            # Broad catch: a long sweep must record the failure and continue.
            except Exception as exc:
                logger.exception("instance failed: %s %s seed=%d", feeder, family, seed)
                error_record = {
                    "instance_id": f"{feeder}:{family}:seed{seed}",
                    "feeder": feeder,
                    "family": family,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "run": run_meta,
                }
                handle.write(json.dumps(error_record, sort_keys=True) + "\n")
                handle.flush()
                failed += 1

    logger.info("done: %d ok, %d failed of %d -> %s", ok, failed, len(grid), out_path)
    if ok == 0:
        raise SystemExit(
            f"reconfiguration_sweep: all {failed} instances failed; see {out_path}"
        )


if __name__ == "__main__":
    main()
