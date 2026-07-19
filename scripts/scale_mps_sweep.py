"""Grind 2.5: direct scale-tier MPS-negativity on the dense-glass instances.

For each HARDNESS-tier dense glass (complete graph, +-10 integer couplings;
the 2026-07-12 six: n in {80,120,160} x seeds {7,24}), run the D5 chi-swept
MPS QAOA protocol (JuliQAOA, penalty-free) and score the best MPS-sampled
energy against the classical incumbent (Gurobi at --gurobi-time-limit; neal
SA recorded alongside -- at the scale tier there is NO exact ground truth,
and every record says so).

HONEST CAVEAT carried in every record: above max_exact_qubits (24) the Julia
driver runs FIXED canonical angles (no angle optimization is implemented at
scale), so "MPS gives a poor objective" here means poor at fixed angles and
bounded chi; the chi-curve (flat vs improving) is the truncation diagnostic.

Run from workspace/:
    python scripts/scale_mps_sweep.py [--sizes 80,120,160] [--seeds 7,24]
        [--qaoa-rounds 2] [--chi-values 4,8,16,32,64]
        [--gurobi-time-limit 600] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from eon.formulations.layer_b import solve_layer_b_surrogate
from eon.formulations.qubo import compile_external_qubo, solve_qubo_with_neal
from eon.instances.external import build_external_surrogate, generate_longrange_spin_glass
from eon.mps.protocol import run_mps_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scale_mps_sweep")

CAVEAT = (
    "no exact ground truth at this scale; reference = Gurobi incumbent "
    "(TIME_LIMIT); MPS runs FIXED canonical angles above max_exact_qubits=24 "
    "(no angle optimization at scale), so poorness is at-fixed-angles; read "
    "the chi-curve for the truncation signal"
)


def _finite(value: float | None) -> float | None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="80,120,160")
    parser.add_argument("--seeds", default="7,24")
    parser.add_argument("--qaoa-rounds", type=int, default=2)
    parser.add_argument("--chi-values", default="4,8,16,32,64")
    parser.add_argument("--gurobi-time-limit", type=float, default=600.0)
    parser.add_argument("--angle-iterations", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    chi_values = tuple(int(c) for c in args.chi_values.split(",") if c.strip())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/scale_mps_{stamp}/sweep.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    run_meta: dict[str, object] = {
        "qaoa_rounds": args.qaoa_rounds,
        "chi_values": list(chi_values),
        "gurobi_time_limit_s": args.gurobi_time_limit,
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
        "caveat": CAVEAT,
    }

    grid = [(n, seed) for n in sizes for seed in seeds]
    logger.info("scale MPS sweep: %d instances -> %s", len(grid), out_path)
    ok = 0
    failed = 0
    with out_path.open("w") as out:
        for index, (n, seed) in enumerate(grid, start=1):
            instance = generate_longrange_spin_glass(
                n, seed, mean_degree=0.0, coefficient_range=(-10, 10)
            )
            logger.info("[%d/%d] %s", index, len(grid), instance.name)
            try:
                surrogate = build_external_surrogate(instance)
                gurobi = solve_layer_b_surrogate(
                    surrogate, time_limit_s=args.gurobi_time_limit, mip_gap=0.0
                )
                sa = solve_qubo_with_neal(
                    surrogate, compile_external_qubo(surrogate), num_reads=256
                )
                incumbent = gurobi.objective_value
                mps = run_mps_protocol(
                    surrogate,
                    instance_name=instance.name,
                    chi_values=chi_values,
                    chi_limit=max(chi_values),
                    qaoa_rounds=args.qaoa_rounds,
                    angle_iterations=args.angle_iterations,
                    seed=seed,
                    reference_energy=incumbent,
                    penalty_free=True,
                )
                best_mps = min(
                    (o.best_energy for o in mps.ordering_results), default=float("inf")
                )
                scale = max(abs(incumbent), 1.0)
                record = {
                    "instance_id": instance.name,
                    "variable_count": n,
                    "seed": seed,
                    "gurobi": {
                        "objective_value": _finite(gurobi.objective_value),
                        "best_bound": _finite(gurobi.best_bound),
                        "mip_gap": _finite(gurobi.mip_gap),
                        "status": gurobi.status,
                    },
                    "simulated_annealing": {"best_energy": _finite(sa.objective_value)},
                    "mps": {
                        "backend": mps.backend,
                        "chi_max_reached": mps.chi_max_reached,
                        "best_energy_over_orderings": _finite(best_mps),
                        "excess_over_incumbent": _finite((best_mps - incumbent) / scale),
                        "orderings": [
                            {
                                "ordering": o.ordering,
                                "best_energy": _finite(o.best_energy),
                                "chi_curve": {
                                    str(chi): _finite(e)
                                    for chi, e in sorted(o.chi_curve.items())
                                },
                                "max_entropy": _finite(
                                    max(o.entropy_curve.values(), default=0.0)
                                ),
                            }
                            for o in mps.ordering_results
                        ],
                    },
                    "run": run_meta,
                }
                ok += 1
            # Broad catch: a long sweep must record the failure and continue.
            except Exception as exc:
                logger.exception("instance failed: %s", instance.name)
                record = {
                    "instance_id": instance.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "run": run_meta,
                }
                failed += 1
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            mps_block = record.get("mps") or {}
            logger.info(
                "  -> excess_over_incumbent=%s chi_max=%s",
                mps_block.get("excess_over_incumbent"),
                mps_block.get("chi_max_reached"),
            )

    logger.info("done: %d ok, %d failed of %d -> %s", ok, failed, len(grid), out_path)
    if ok == 0:
        raise SystemExit(f"scale_mps_sweep: all {failed} instances failed; see {out_path}")


if __name__ == "__main__":
    main()
