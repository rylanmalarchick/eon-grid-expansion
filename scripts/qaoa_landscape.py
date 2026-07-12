"""P3 landscape runs: cop-QAOA vs vanilla penalty-QAOA (+ Egger warm start).

Instances:
- Feeders: IEEE 33 community_bridging seeds 7/24, n=20, reconfiguration ON,
  1800 s Layer A (the same build protocol as experiments/results/mps_higherp).
  All three algorithms run here (the at-most-K constraint is what the cop
  mixer preserves).
- Planted anchors: fused_planted n in {20, 24} seed 7 (external, no
  cardinality constraint -> vanilla + warm start only; using the weight
  mixer there would leak the planted Hamming weight).

Both algorithms share the cost vector and optimizer budget; the exact
reference for approximation is min(energy vector) -- the optimum of the SAME
scored object, no cross-encoding offset. One JSONL record per (instance,
algorithm, depth); the p=1 7x7 grid is recorded per (feeder instance,
algorithm) for the landscape figure. Simulator only (D7).

Run from workspace/:
    python scripts/qaoa_landscape.py [--time-limit 1800] [--depth 3]
        [--shots 1024] [--nm-evals 60] [--out PATH] [--log-file PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set
from eon.quantum.cop_qaoa import (
    QAOARunResult,
    _target_hamming_weight,
    _uniform_weight_state,
    run_constrained_qaoa_depths,
)
from eon.quantum.energy import build_energy_vector
from eon.quantum.qaoa import run_penalty_qaoa
from eon.quantum.simulator import (
    expected_energy_of_state,
    simulate_qaoa,
    uniform_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qaoa_landscape")

CANDIDATE_COUNT = 24
MAX_NEW_LINES = 3
COST_PER_KM = 10_000.0
SCENARIO_KIND = "stressed_five_point"
GRID_BETAS = tuple(np.linspace(0.1, math.pi / 2, 7))
GRID_GAMMAS = tuple(np.linspace(0.1, math.pi, 7))


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


def _feasible_fraction(surrogate: LayerBSurrogate, counts: dict[str, int]) -> float:
    from eon.quantum.postprocess import decode_counts

    decoded = decode_counts(surrogate, counts)
    return float(sum(r.sampling_prob for r in decoded if r.feasible))


def _p1_grid(
    surrogate: LayerBSurrogate, energies: np.ndarray, *, algorithm: str
) -> dict[str, object]:
    # numpy engine only: the Qiskit Statevector path is intractable at n=20
    # (DiagonalGate synthesis; and initialize triggers an O(4^n) reset matrix).
    n = len(surrogate.variables)
    if algorithm == "cop":
        initial = _uniform_weight_state(n, _target_hamming_weight(surrogate))
        mixer = "xy_ring"
    else:
        initial = uniform_state(n)
        mixer = "x"
    rows: list[list[float]] = []
    for beta in GRID_BETAS:
        row = []
        for gamma in GRID_GAMMAS:
            state = simulate_qaoa(initial, energies, (beta,), (gamma,), mixer=mixer)
            row.append(expected_energy_of_state(state, energies))
        rows.append(row)
    return {"betas": list(GRID_BETAS), "gammas": list(GRID_GAMMAS), "energies": rows}


def _result_record(
    instance_id: str,
    algorithm: str,
    result: QAOARunResult,
    *,
    exact_optimum: float,
    feasible_fraction: float,
    run_meta: dict[str, object],
) -> dict[str, object]:
    best = result.best_sample
    scale = max(abs(exact_optimum), 1.0)
    return {
        "instance_id": instance_id,
        "algorithm": algorithm,
        "p": result.p,
        "betas": list(result.betas),
        "gammas": list(result.gammas),
        "expected_energy": result.expected_energy,
        "exact_optimum": exact_optimum,
        "best_sample": {
            "objective": best.objective,
            "feasible": best.feasible,
            "sampling_prob": best.sampling_prob,
            "selected_candidates": list(best.selected_candidates),
        },
        # Best FEASIBLE sampled energy relative to the exact optimum of the
        # same scored cost object (0 = optimal).
        "best_sample_excess": (best.objective - exact_optimum) / scale
        if best.feasible
        else None,
        "feasible_fraction": feasible_fraction,
        "run": run_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--nm-evals", type=int, default=60)
    parser.add_argument("--seeds", default="7,24")
    parser.add_argument("--out", default="")
    parser.add_argument("--log-file", default="")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of truncating (crash resume; rerun only "
        "the seeds/instances not yet recorded).",
    )
    parser.add_argument(
        "--skip-planted",
        action="store_true",
        help="Skip the fused planted anchors (resume helper).",
    )
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else Path(f"experiments/results/qaoa_{stamp}/landscape.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    run_meta: dict[str, object] = {
        "time_limit_s": args.time_limit,
        "shots": args.shots,
        "nm_evals": args.nm_evals,
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
    }
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # --- assemble instances -------------------------------------------------
    jobs: list[tuple[str, LayerBSurrogate, bool]] = []  # (id, surrogate, constrained)
    scenarios = build_scenario_set(SCENARIO_KIND)
    net = load_distribution_feeder("ieee33")
    baseline_stress = _compute_baseline_stress(
        net,
        scenarios,
        ExpansionProblemConfig(
            max_new_lines=MAX_NEW_LINES, time_limit_s=args.time_limit, n_scenarios_aggregated=3
        ),
    )
    for seed in seeds:
        config = ExpansionProblemConfig(
            max_new_lines=MAX_NEW_LINES,
            time_limit_s=args.time_limit,
            n_scenarios_aggregated=3,
            enable_reconfiguration=True,
        )
        logger.info("building ieee33 community_bridging seed=%d (Layer A %.0fs)",
                    seed, args.time_limit)
        _, layer_a, surrogate = build_instance_surrogate(
            net,
            scenarios,
            config,
            family="community_bridging",
            candidate_count=CANDIDATE_COUNT,
            neighborhood_size=20,
            seed=seed,
            time_limit=args.time_limit,
            cost_per_km=COST_PER_KM,
            baseline_stress=baseline_stress,
        )
        logger.info("  layer A %s gap=%s", layer_a.termination_status, layer_a.mip_gap)
        jobs.append((f"ieee33:community_bridging:seed{seed}:n20", surrogate, True))
    if not args.skip_planted:
        for n in (20, 24):
            instance = generate_fused_planted(n, 7, block_size=10, alpha=0.1)
            jobs.append((instance.name, build_external_surrogate(instance), False))

    # --- run ---------------------------------------------------------------
    with out_path.open("a" if args.append else "w") as out:
        for instance_id, surrogate, constrained in jobs:
            logger.info("instance %s: building energy vector", instance_id)
            energies = build_energy_vector(surrogate)
            exact_optimum = float(energies.min())
            algorithms: list[tuple[str, list[QAOARunResult]]] = []
            if constrained:
                logger.info("  cop-QAOA depths 1..%d", args.depth)
                algorithms.append(
                    (
                        "cop",
                        run_constrained_qaoa_depths(
                            surrogate,
                            p=args.depth,
                            shots=args.shots,
                            nelder_mead_evals=args.nm_evals,
                            energy_vector=energies,
                        ),
                    )
                )
            logger.info("  vanilla depths 1..%d", args.depth)
            algorithms.append(
                (
                    "vanilla",
                    run_penalty_qaoa(
                        surrogate,
                        p=args.depth,
                        shots=args.shots,
                        nelder_mead_evals=args.nm_evals,
                        energies=energies,
                    ),
                )
            )
            logger.info("  warm-started vanilla depths 1..%d", args.depth)
            algorithms.append(
                (
                    "vanilla_warm",
                    run_penalty_qaoa(
                        surrogate,
                        p=args.depth,
                        shots=args.shots,
                        warm_start=True,
                        nelder_mead_evals=args.nm_evals,
                        energies=energies,
                    ),
                )
            )
            for algorithm, results in algorithms:
                for result in results:
                    record = _result_record(
                        instance_id,
                        algorithm,
                        result,
                        exact_optimum=exact_optimum,
                        feasible_fraction=_feasible_fraction(surrogate, result.counts),
                        run_meta=run_meta,
                    )
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    logger.info(
                        "  -> %s %s p=%d excess=%s feas=%.3f",
                        instance_id,
                        algorithm,
                        result.p,
                        record["best_sample_excess"],
                        record["feasible_fraction"],
                    )
            if constrained:
                for algorithm in ("cop", "vanilla"):
                    grid_record = {
                        "instance_id": instance_id,
                        "algorithm": algorithm,
                        "p1_grid": _p1_grid(surrogate, energies, algorithm=algorithm),
                        "exact_optimum": exact_optimum,
                        "run": run_meta,
                    }
                    out.write(json.dumps(grid_record, sort_keys=True) + "\n")
                    out.flush()
                logger.info("  p=1 grids recorded for %s", instance_id)

    logger.info("done -> %s", out_path)


if __name__ == "__main__":
    main()
