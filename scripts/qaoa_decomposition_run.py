"""P3 decomposition-wrapper runs with the D14 certificate.

Job 1 (feeder, n>=30): IEEE 33, candidate_count=40, neighborhood 32,
reconfiguration ON, Layer A at --time-limit -> n~32 surrogate -> Gurobi
Layer B (bound) -> per-block cop-QAOA -> merge/repair -> certificate
(LB vs rescored UB on the surrogate objective) -> LinDistFlow fixed-builds
solve of the merged plan (the feasible Layer A plan the brief asks for).

Job 2 (external, n=50): fused planted instance; the certificate must
sandwich the known planted optimum (recorded as a check, not assumed).

The certified gap is the decomposition/integrality gap of the classical
wrapper, NOT classical-vs-quantum advantage.

Run from workspace/:
    python scripts/qaoa_decomposition_run.py [--time-limit 1800]
        [--block-size 10] [--p 1] [--out PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eon.formulations.layer_b import solve_layer_b_surrogate
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set
from eon.quantum.decomposition import solve_decomposed_qaoa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qaoa_decomposition_run")

SCENARIO_KIND = "stressed_five_point"
DISCLAIMER = (
    "certified gap = decomposition/integrality gap of the classical wrapper; "
    "NOT classical-vs-quantum advantage"
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/qaoa_decomp_{stamp}/decomposition.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    run_meta: dict[str, object] = {
        "time_limit_s": args.time_limit,
        "block_size": args.block_size,
        "p": args.p,
        "shots": args.shots,
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
        "note": DISCLAIMER,
    }

    with out_path.open("w") as out:
        # --- Job 1: IEEE 33 n>=30 -----------------------------------------
        logger.info("job 1: ieee33 candidate_count=40 neighborhood=32 (Layer A %.0fs)",
                    args.time_limit)
        net = load_distribution_feeder("ieee33")
        scenarios = build_scenario_set(SCENARIO_KIND)
        config = ExpansionProblemConfig(
            max_new_lines=3,
            time_limit_s=args.time_limit,
            n_scenarios_aggregated=3,
            enable_reconfiguration=True,
        )
        baseline_stress = _compute_baseline_stress(net, scenarios, config)
        candidates, layer_a, surrogate = build_instance_surrogate(
            net,
            scenarios,
            config,
            family="community_bridging",
            candidate_count=40,
            neighborhood_size=32,
            seed=7,
            time_limit=args.time_limit,
            cost_per_km=10_000.0,
            baseline_stress=baseline_stress,
        )
        n = len(surrogate.variables)
        logger.info("  layer A %s gap=%s; surrogate n=%d",
                    layer_a.termination_status, layer_a.mip_gap, n)
        layer_b = solve_layer_b_surrogate(surrogate, time_limit_s=600.0, mip_gap=0.0)
        solution, certificate = solve_decomposed_qaoa(
            surrogate,
            block_size=args.block_size,
            p=args.p,
            shots=args.shots,
            gurobi_best_bound=layer_b.best_bound,
        )
        merged_builds = solution.actual_builds
        rescore = solve_lindistflow_expansion(
            net, scenarios, candidates, config, fixed_builds=merged_builds
        )
        record = {
            "job": "ieee33_n32_decomposition",
            "variable_count": n,
            "layer_a": {
                "termination_status": layer_a.termination_status,
                "mip_gap": layer_a.mip_gap,
            },
            "layer_b_gurobi": {
                "objective_value": layer_b.objective_value,
                "best_bound": layer_b.best_bound,
                "status": layer_b.status,
            },
            "decomposed_plan": {
                "selected_candidates": list(solution.selected_candidates),
                "surrogate_objective": solution.objective_value,
            },
            "certificate": dataclasses.asdict(certificate),
            "layer_a_rescore": {
                "termination_status": rescore.termination_status,
                "objective_value": rescore.objective_value,
                "feasible": rescore.termination_status != "INFEASIBLE"
                and rescore.objective_value != float("inf"),
            },
            "run": run_meta,
        }
        out.write(json.dumps(record, sort_keys=True) + "\n")
        out.flush()
        logger.info(
            "  -> LB=%.4g UB=%.4g gap=%.4g (%s); layer A rescore %s obj=%.6g",
            certificate.lower_bound,
            certificate.upper_bound,
            certificate.gap,
            certificate.lower_bound_source,
            rescore.termination_status,
            rescore.objective_value,
        )

        # --- Job 2: fused planted n=50 -------------------------------------
        logger.info("job 2: fused_planted n=50 seed=7")
        instance = generate_fused_planted(50, 7, block_size=10, alpha=0.1)
        ext_surrogate = build_external_surrogate(instance)
        ext_layer_b = solve_layer_b_surrogate(ext_surrogate, time_limit_s=600.0, mip_gap=0.0)
        ext_solution, ext_certificate = solve_decomposed_qaoa(
            ext_surrogate,
            block_size=args.block_size,
            p=args.p,
            shots=args.shots,
            gurobi_best_bound=ext_layer_b.best_bound,
        )
        assert instance.planted_energy is not None  # fused instances always plant
        planted = float(instance.planted_energy)
        sandwich_ok = (
            ext_certificate.lower_bound <= planted + 1e-6
            and planted <= ext_certificate.upper_bound + 1e-6
        )
        record = {
            "job": "fused_planted_n50_decomposition",
            "variable_count": 50,
            "planted_energy": planted,
            "layer_b_gurobi": {
                "objective_value": ext_layer_b.objective_value,
                "best_bound": ext_layer_b.best_bound,
                "status": ext_layer_b.status,
            },
            "decomposed_plan": {
                "surrogate_objective": ext_solution.objective_value,
            },
            "certificate": dataclasses.asdict(ext_certificate),
            "sandwich_holds": sandwich_ok,
            "run": run_meta,
        }
        out.write(json.dumps(record, sort_keys=True) + "\n")
        out.flush()
        logger.info(
            "  -> LB=%.4g planted=%.4g UB=%.4g sandwich=%s",
            ext_certificate.lower_bound,
            planted,
            ext_certificate.upper_bound,
            sandwich_ok,
        )
        if not sandwich_ok:
            raise SystemExit("planted sandwich violated -- certificate bug, investigate")

    logger.info("done -> %s", out_path)


if __name__ == "__main__":
    main()
