"""S3: does cop-QAOA beat UNIFORM RANDOM SAMPLING of its own feasible subspace?

The cop-QAOA mixer (XY ring) preserves Hamming weight, so its entire search
space is the fixed-weight subspace C(n, w). If that subspace is small relative
to the shot budget, "cop-QAOA found the optimum" is explained by near-
exhaustive sampling, NOT by search quality -- the control a technical reviewer
asks for first. This script measures it exactly:

  |subspace| = C(n, w), coverage = expected distinct states in `shots` draws,
  random-feasible best-of-shots (median over repeats) vs cop-QAOA best sample
  vs the exact optimum of the same compiled cost object.

Reports the honest verdict either way; the claim it protects is
"feasibility-by-construction", which does not depend on winning this control.

Run from workspace/:
    python scripts/random_feasible_control.py [--time-limit 1800]
        [--shots 1024] [--repeats 25] [--seeds 7] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.candidate_lines import require_seed_diversification
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
)
from eon.instances.scenarios import build_scenario_set
from eon.quantum.cop_qaoa import (
    _target_hamming_weight,
    run_constrained_qaoa_depths,
)
from eon.quantum.energy import build_energy_vector
from eon.quantum.postprocess import state_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("random_feasible_control")

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


def _fixed_weight_states(n: int, weight: int) -> np.ndarray:
    """Indices of all basis states with exactly `weight` set bits."""
    from itertools import combinations

    return np.asarray(
        [sum(1 << position for position in combo) for combo in combinations(range(n), weight)],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument(
        "--rank-jitter",
        type=float,
        default=0.0,
        help="Perturb the candidate ranking's tie-breaks so seeds genuinely "
        "diversify (0.0 = deterministic families, seeds are a NO-OP and every "
        "seed rebuilds the SAME Layer B instance).",
    )
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--neighborhood", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--max-new-lines",
        type=int,
        default=3,
        help="Layer A build budget K. NOTE: raising this does NOT enlarge the "
        "cop subspace. The derived Hamming weight is min(n, K, "
        "base_selected_count), and the incumbent's selected count binds. Use "
        "--hamming-weight to set the subspace directly.",
    )
    parser.add_argument(
        "--hamming-weights",
        default=None,
        help="Comma-separated weights swept in ONE Layer A solve, e.g. 2,3,4,6,8. "
        "The Layer A solve dominates runtime and does not depend on the weight, "
        "so a sweep is nearly free and traces cop-QAOA's advantage as a function "
        "of shot coverage instead of asserting it at a single point.",
    )
    parser.add_argument(
        "--hamming-weight",
        type=int,
        default=None,
        help="Set the cop subspace weight directly, so |subspace| = C(n, w). "
        "This is how to test whether cop-QAOA beats random-feasible when the "
        "subspace is NOT near-exhaustible by the shot budget. The cardinality "
        "is then imposed, not inherited from the Layer A incumbent -- say so "
        "when reporting.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    # Fail at launch, not after hours of compute that cannot be labelled.
    require_seed_diversification(
        [int(s) for s in args.seeds.split(",") if s.strip()],
        "community_bridging",
        rank_jitter=args.rank_jitter,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/s3_control_{stamp}/control.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_path.with_suffix(".log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    net = load_distribution_feeder("ieee33")
    scenarios = build_scenario_set(SCENARIO_KIND)
    config = ExpansionProblemConfig(
        max_new_lines=args.max_new_lines,
        time_limit_s=args.time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=True,
    )
    baseline_stress = _compute_baseline_stress(net, scenarios, config)

    with out_path.open("w") as out:
        for seed in (int(s) for s in args.seeds.split(",") if s.strip()):
            logger.info("building ieee33 community_bridging seed=%d", seed)
            _, layer_a, surrogate = build_instance_surrogate(
                net,
                scenarios,
                config,
                family="community_bridging",
                candidate_count=24,
                neighborhood_size=args.neighborhood,
                seed=seed,
                time_limit=args.time_limit,
                cost_per_km=10_000.0,
                baseline_stress=baseline_stress,
                rank_jitter=args.rank_jitter,
            )
            energies = build_energy_vector(surrogate)
            n = len(surrogate.variables)
            if args.hamming_weights:
                sweep = [int(w) for w in args.hamming_weights.split(",") if w.strip()]
            else:
                sweep = [args.hamming_weight]
            # The Layer A solve above is weight-independent, so every weight
            # in the sweep reuses it.
            for weight_override in sweep:
                weight = _target_hamming_weight(surrogate, override=weight_override)
                subspace = _fixed_weight_states(n, weight)
                subspace_energies = energies[subspace]
                subspace_size = len(subspace)
                exact_optimum = float(energies.min())
                subspace_optimum = float(subspace_energies.min())

                # Expected distinct states drawn in `shots` uniform draws with
                # replacement: N * (1 - (1 - 1/N)^shots).
                coverage = 1.0 - (1.0 - 1.0 / subspace_size) ** args.shots
                logger.info(
                    "n=%d weight=%d |subspace|=C(%d,%d)=%d shots=%d expected coverage=%.1f%%",
                    n,
                    weight,
                    n,
                    weight,
                    subspace_size,
                    args.shots,
                    100 * coverage,
                )

                rng = np.random.default_rng(seed)
                random_best = [
                    float(subspace_energies[rng.integers(0, subspace_size, size=args.shots)].min())
                    for _ in range(args.repeats)
                ]

                logger.info("running cop-QAOA depths 1..%d", args.depth)
                cop_results = run_constrained_qaoa_depths(
                    surrogate,
                    p=args.depth,
                    shots=args.shots,
                    energy_vector=energies,
                    seed=seed,
                    # MUST be the swept weight, not the flag: passing the
                    # flag made cop search the derived w=2 subspace while
                    # the record reported w=6 statistics beside it.
                    hamming_weight=weight_override,
                )

                # Score cop on the SAME vector random_best is drawn from.
                # best_sample.objective is the UNPENALIZED surrogate objective,
                # and at weights above the build budget every cop sample is
                # infeasible -- so cop was credited with an unpenalized score
                # while random paid the penalty, and cop "won" by construction.
                # It even reported an excess BELOW the exact optimum, which is
                # impossible against a true minimum.
                def _scored(result: object, vector: np.ndarray = energies) -> float:
                    return min(
                        float(vector[state_index(key)])
                        for key in result.counts  # type: ignore[attr-defined]
                    )

                cop_best = {r.p: _scored(r) for r in cop_results}

                scale = max(abs(exact_optimum), 1.0)
                record = {
                    "instance_id": f"ieee33:community_bridging:seed{seed}:n{n}",
                    "layer_a": {
                        "termination_status": layer_a.termination_status,
                        "mip_gap": layer_a.mip_gap,
                    },
                    "subspace": {
                        "variable_count": n,
                        "hamming_weight": weight,
                        "weight_source": ("imposed" if weight_override is not None else "derived"),
                        "size": subspace_size,
                        "shots": args.shots,
                        "expected_coverage_fraction": coverage,
                        "shots_exceed_subspace": args.shots >= subspace_size,
                    },
                    "exact_optimum": exact_optimum,
                    "subspace_optimum": subspace_optimum,
                    "random_feasible": {
                        "repeats": args.repeats,
                        "median_best": float(np.median(random_best)),
                        "worst_best": float(np.max(random_best)),
                        "found_subspace_optimum_fraction": float(
                            np.mean([abs(b - subspace_optimum) < 1e-9 for b in random_best])
                        ),
                    },
                    "cop_qaoa_best_by_depth": cop_best,
                    # The verdict: does cop-QAOA beat a uniform draw from its own
                    # search space at the same shot budget?
                    "cop_beats_random_median": {
                        str(p): bool(v < float(np.median(random_best)) - 1e-9)
                        for p, v in cop_best.items()
                    },
                    "excess_over_exact": {
                        "random_median": (float(np.median(random_best)) - exact_optimum) / scale,
                        **{f"cop_p{p}": (v - exact_optimum) / scale for p, v in cop_best.items()},
                    },
                    "run": {
                        "time_limit_s": args.time_limit,
                        "rank_jitter": args.rank_jitter,
                        "max_new_lines": args.max_new_lines,
                        "hamming_weight_override": weight_override,
                        "git_commit": _git_commit(),
                        "timestamp_utc": stamp,
                        "note": "cop-QAOA searches ONLY the fixed-weight subspace; if "
                        "shots >= |subspace| its optimum-finding is near-exhaustive "
                        "sampling, not search quality",
                    },
                }
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
            logger.info(
                "  -> |subspace|=%d random_median_excess=%.4f cop_p%d_excess=%.4f",
                subspace_size,
                record["excess_over_exact"]["random_median"],  # type: ignore[index]
                args.depth,
                record["excess_over_exact"][f"cop_p{args.depth}"],  # type: ignore[index]
            )

    logger.info("done -> %s", out_path)


if __name__ == "__main__":
    main()
