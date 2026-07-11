"""Path B hardness spine: score external/synthetic QUBOs (PLAN.txt section 6).

Instances: fused planted (R34, unique known optimum, tunable hardness via
--alpha) and long-range spin glasses (no planted optimum, tree-TN workhorse).
Per instance, one JSONL record: coupling diagnostics (effective tree-width,
J/h), the raw-QUBO Gurobi solve (gap/bound at --time-limit; for planted
instances whether Gurobi found the planted optimum), a neal SA baseline, the
GTN exact-TN control (within_budget=False => tree-TN-negative at scale), and
optionally (--with-mps) the penalty-free MPS sweep on NISQ-sized (<=30 var)
instances.

Everything runs PENALTY-FREE: external instances carry no cardinality
constraint, and the Hess penalty would fake the tree-TN signal.

Anti-cherry-pick: every generated instance is recorded, including the ones
Gurobi closes (labeled easy), per the evaluation protocol (PLAN.txt section 7).

Run from workspace/:
    python scripts/pathb_spine.py [--time-limit S] [--seeds 7,24]
        [--fused-sizes 20,30,50,100] [--glass-sizes 50,100,200] [--alpha 0.1]
        [--with-mps] [--qaoa-rounds 2] [--out PATH] [--log-file PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from eon.formulations.layer_b import solve_layer_b_surrogate
from eon.formulations.qubo import compile_external_qubo, solve_qubo_with_neal
from eon.instances.external import (
    ExternalQuboInstance,
    build_external_surrogate,
    generate_fused_planted,
    generate_longrange_spin_glass,
)
from eon.instances.treewidth import coupling_diagnostics
from eon.mps.protocol import run_mps_protocol, run_tree_tn_control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pathb_spine")

_NISQ_QUBIT_LIMIT = 30
_PLANTED_ATOL = 1e-6


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


def score_instance(
    instance: ExternalQuboInstance,
    *,
    time_limit: float,
    with_mps: bool,
    qaoa_rounds: int,
    angle_iterations: int,
    seed: int,
    run_meta: dict[str, object],
) -> dict[str, object]:
    n = len(instance.variable_names)
    surrogate = build_external_surrogate(instance)
    coupling = coupling_diagnostics(surrogate)

    layer_b = solve_layer_b_surrogate(surrogate, time_limit_s=time_limit, mip_gap=0.0)
    gurobi_found_planted = None
    if instance.planted_energy is not None and layer_b.objective_value is not None:
        gurobi_found_planted = bool(
            abs(layer_b.objective_value - instance.planted_energy) <= _PLANTED_ATOL
        )

    compilation = compile_external_qubo(surrogate)
    sa = solve_qubo_with_neal(surrogate, compilation, num_reads=256)

    reference = (
        instance.planted_energy
        if instance.planted_energy is not None
        else layer_b.objective_value
    )

    mps_section: dict[str, object] | None = None
    if with_mps and n <= _NISQ_QUBIT_LIMIT:
        mps = run_mps_protocol(
            surrogate,
            instance_name=instance.name,
            reference_energy=reference,
            seed=seed,
            qaoa_rounds=qaoa_rounds,
            angle_iterations=angle_iterations,
            penalty_free=True,
        )
        tree = mps.tree_tn_result
        mps_section = {
            "backend": mps.backend,
            "chi_max_reached": mps.chi_max_reached,
            "qaoa_rounds": qaoa_rounds,
            "reference_energy": _finite(mps.reference_energy),
            "exact_ground_energy": _finite(mps.exact_ground_energy),
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
        }
    else:
        tree = run_tree_tn_control(
            surrogate, seed=seed, reference_energy=reference, penalty_free=True
        )

    gap_open = layer_b.status != "OPTIMAL"
    tree_negative = bool(tree is not None and not tree.within_budget)
    return {
        "instance_id": instance.name,
        "generator": instance.metadata.get("generator"),
        "variable_count": n,
        "seed": seed,
        "metadata": instance.metadata,
        "planted": None
        if instance.planted_energy is None
        else {
            "energy": instance.planted_energy,
            "gurobi_found_planted": gurobi_found_planted,
        },
        "coupling": {
            "structural_treewidth": coupling.structural_treewidth,
            "effective_treewidth": coupling.effective_treewidth,
            "coupling_field_ratio": coupling.coupling_field_ratio,
            "mps_easy": coupling.mps_easy,
        },
        "gurobi": {
            "objective_value": _finite(layer_b.objective_value),
            "mip_gap": _finite(layer_b.mip_gap),
            "status": layer_b.status,
        },
        "simulated_annealing": {
            "best_energy": _finite(sa.objective_value),
            "num_reads": 256,
        },
        "tree_tn": None
        if tree is None
        else {
            "backend": tree.backend,
            "contraction_width": tree.contraction_width,
            "within_budget": tree.within_budget,
            "exact_ground_energy": _finite(tree.best_energy) if tree.within_budget else None,
        },
        "mps": mps_section,
        "signals": {
            # HARDNESS-tier signal (Wave 5 split): Gurobi cannot close AND the
            # strong exact-TN control cannot contract within the 2^28 budget.
            "gurobi_gap_open": gap_open,
            "tree_tn_negative": tree_negative,
            "hardness_tier_candidate": bool(gap_open and tree_negative),
            "nisq_runnable": n <= _NISQ_QUBIT_LIMIT,
        },
        "run": run_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--seeds", default="7,24")
    parser.add_argument("--fused-sizes", default="20,30,50,100")
    parser.add_argument("--glass-sizes", default="50,100,200")
    parser.add_argument(
        "--glass-mean-degree",
        type=float,
        default=6.0,
        help="Spin-glass mean degree; <= 0 means complete graph (dense SK-style).",
    )
    parser.add_argument(
        "--glass-coefficient-range",
        default="",
        help="Integer coupling range 'low,high' for the spin glass (default +-1).",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--with-mps", action="store_true")
    parser.add_argument("--qaoa-rounds", type=int, default=2)
    parser.add_argument("--angle-iterations", type=int, default=10)
    parser.add_argument("--out", default="")
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fused_sizes = [int(s) for s in args.fused_sizes.split(",") if s.strip()]
    glass_sizes = [int(s) for s in args.glass_sizes.split(",") if s.strip()]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else Path(f"experiments/results/pathb_{stamp}/spine.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    glass_range: tuple[int, int] | None = None
    if args.glass_coefficient_range:
        low, high = (int(v) for v in args.glass_coefficient_range.split(","))
        glass_range = (low, high)

    run_meta: dict[str, object] = {
        "time_limit_s": args.time_limit,
        "alpha": args.alpha,
        "block_size": args.block_size,
        "glass_mean_degree": args.glass_mean_degree,
        "glass_coefficient_range": None if glass_range is None else list(glass_range),
        "git_commit": _git_commit(),
        "timestamp_utc": stamp,
    }

    jobs: list[tuple[ExternalQuboInstance, int]] = []
    for seed in seeds:
        for n in fused_sizes:
            jobs.append(
                (
                    generate_fused_planted(
                        n, seed, block_size=args.block_size, alpha=args.alpha
                    ),
                    seed,
                )
            )
        for n in glass_sizes:
            jobs.append(
                (
                    generate_longrange_spin_glass(
                        n,
                        seed,
                        mean_degree=args.glass_mean_degree,
                        coefficient_range=glass_range,
                    ),
                    seed,
                )
            )

    logger.info("pathb spine: %d instances -> %s (log: %s)", len(jobs), out_path, log_path)
    ok = 0
    failed = 0
    with out_path.open("w") as handle:
        for index, (instance, seed) in enumerate(jobs, start=1):
            logger.info("[%d/%d] %s", index, len(jobs), instance.name)
            try:
                record = score_instance(
                    instance,
                    time_limit=args.time_limit,
                    with_mps=args.with_mps,
                    qaoa_rounds=args.qaoa_rounds,
                    angle_iterations=args.angle_iterations,
                    seed=seed,
                    run_meta=run_meta,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001 -- long sweep must record and continue
                logger.exception("instance failed: %s", instance.name)
                record = {
                    "instance_id": instance.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "run": run_meta,
                }
                failed += 1
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            signals = cast(dict[str, object], record.get("signals") or {})
            gurobi_block = cast(dict[str, object], record.get("gurobi") or {})
            logger.info(
                "  -> gurobi=%s tree_negative=%s hardness_tier=%s",
                gurobi_block.get("status"),
                signals.get("tree_tn_negative"),
                signals.get("hardness_tier_candidate"),
            )

    logger.info("done: %d ok, %d failed of %d -> %s", ok, failed, len(jobs), out_path)
    if ok == 0:
        raise SystemExit(f"pathb_spine: all {failed} instances failed; see {out_path}")


if __name__ == "__main__":
    main()
