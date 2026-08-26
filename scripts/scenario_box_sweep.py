"""Does our five-scenario set actually find the worst case it stands in for?

The headline congestion numbers are scenario-weighted over five hand-picked
operating points. Those five points are a SAMPLE of a three-dimensional
uncertainty box -- load scale, generation scale, line-capacity scale -- and a
sample of a continuous box is a lower bound on the worst case over it, never a
guarantee. A DSO plans against the worst case, not the weighted mean, so if the
sample misses badly the headline overstates how well the plan holds up.

This script measures the miss. It evaluates the plan (and the status quo, for
the reduction claim) on a dense grid over the same box the five points span,
and compares:

  weighted_five_point   what we publish -- the probability-weighted mean
  sampled_worst         max over the five points, each evaluated on its own
  grid_worst            max over the dense grid

HONESTY: grid_worst is itself a dense-sample LOWER bound on the true worst case
over the continuous box, not a certificate. Reported as such. It can only widen
the gap it measures, never narrow it, so it is the right side to err on.

The budget sweep costs no extra solves: random N-point samplers are simulated
by drawing subsets of the already-evaluated grid, which is exactly what a
smaller scenario set would have seen.

Run from workspace/:
    python scripts/scenario_box_sweep.py [--grid 9,9,5] [--workers 10] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    solve_lindistflow_expansion,
)
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import Scenario, build_scenario_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scenario_box_sweep")

SCENARIO_KIND = "stressed_five_point"
FAMILY = "community_bridging"
CANDIDATE_COUNT = 24
COST_PER_KM = 10_000.0
# The headline plan (scripts/congestion_metrics.py, seed 7). Passed as a fixed
# build set so no Layer A solve is needed and the sweep is cheap to reproduce.
# The plan under test is READ from the headline artifact, never hard-coded.
# A literal here silently went stale across the 2026-08-08 model correction: it
# named the old TWO-line plan while the corrected headline plan is one line, so
# the whole robustness section described a plan the document does not report.
PLAN_BUILDS: tuple[str, ...] = ()


def _load_plan_builds(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(
            f"cannot read the plan under test from {path}. The box sweep must "
            "evaluate the SAME build set the headline reports; pass "
            "--plan-from to point at the congestion metrics artifact."
        )
    record = json.loads(path.read_text())
    builds = tuple(record["layer_a"]["selected_candidates"])
    if not builds:
        raise ValueError(f"{path} records no selected candidates; nothing to evaluate")
    return builds
SAMPLE_BUDGETS = (5, 10, 25, 50, 100)
SAMPLE_SEEDS = 200


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


def _box(scenarios: list[Scenario]) -> dict[str, tuple[float, float]]:
    """The axis-aligned hull of the five points -- the box they claim to cover.

    Deliberately not wider: the question is whether five points find the worst
    case inside their OWN span, not whether the span itself is too narrow.
    """
    return {
        "load_scale": (
            min(s.load_scale for s in scenarios),
            max(s.load_scale for s in scenarios),
        ),
        "generation_scale": (
            min(s.generation_scale for s in scenarios),
            max(s.generation_scale for s in scenarios),
        ),
        "line_capacity_scale": (
            min(s.line_capacity_scale for s in scenarios),
            max(s.line_capacity_scale for s in scenarios),
        ),
    }


_WORKER: dict[str, object] = {}


def _init_worker(eval_time_limit: float) -> None:
    """Each worker rebuilds the feeder and candidates once, not per point."""
    net = load_distribution_feeder("ieee33")
    _WORKER["net"] = net
    _WORKER["candidates"] = generate_candidate_lines(
        net, FAMILY, CANDIDATE_COUNT, seed=7, cost_per_km=COST_PER_KM
    )
    # Threads=1: these run as parallel processes, and letting each solve claim
    # every core makes all of them slower.
    _WORKER["config"] = lambda reconfiguration: ExpansionProblemConfig(
        max_new_lines=3,
        time_limit_s=eval_time_limit,
        n_scenarios_aggregated=3,
        enable_reconfiguration=reconfiguration,
        threads=1,
    )


# Must match scripts/congestion_metrics.py exactly, or the sweep measures a
# different object than the headline: status_quo is reconfiguration OFF, and
# running it ON silently reproduces the reconfiguration_only number instead.
CONFIGURATIONS = {
    "plan": {"builds": True, "reconfiguration": True},
    "reconfiguration_only": {"builds": False, "reconfiguration": True},
    "status_quo": {"builds": False, "reconfiguration": False},
}


def _evaluate(job: tuple[str, float, float, float]) -> dict[str, object]:
    label, load, generation, capacity = job
    spec = CONFIGURATIONS[label]
    candidates = _WORKER["candidates"]
    builds = {
        candidate.name: int(spec["builds"] and candidate.name in PLAN_BUILDS)
        for candidate in candidates
    }
    scenario = [
        Scenario(
            name="point",
            load_scale=load,
            generation_scale=generation,
            probability=1.0,
            line_capacity_scale=capacity,
        )
    ]
    config = _WORKER["config"](spec["reconfiguration"])
    result = solve_lindistflow_expansion(
        _WORKER["net"], scenario, candidates, config, fixed_builds=builds
    )
    return {
        "configuration": label,
        "load_scale": load,
        "generation_scale": generation,
        "line_capacity_scale": capacity,
        "congestion_mw": float(result.aggregate_metrics["weighted_congestion_mw"]),
        "voltage_violation_pu": float(
            result.aggregate_metrics["weighted_voltage_violation_pu"]
        ),
        "termination_status": result.termination_status,
    }


def _budget_curve(grid_values: np.ndarray, grid_worst: float) -> list[dict[str, float]]:
    """What a random N-point scenario set would have reported, drawn from the
    grid we already evaluated. Median over SAMPLE_SEEDS draws."""
    rng = np.random.default_rng(7)
    curve = []
    for budget in SAMPLE_BUDGETS:
        if budget > grid_values.size:
            continue
        worsts = [
            grid_values[rng.choice(grid_values.size, size=budget, replace=False)].max()
            for _ in range(SAMPLE_SEEDS)
        ]
        median = float(np.median(worsts))
        curve.append(
            {
                "budget": budget,
                "median_sampled_worst_mw": median,
                "median_under_report_fraction": (
                    (grid_worst - median) / grid_worst if grid_worst > 0 else 0.0
                ),
            }
        )
    return curve


def _coordinate(record: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(record["load_scale"]),
        float(record["generation_scale"]),
        float(record["line_capacity_scale"]),
    )


def _monotonicity(records: list[dict[str, object]]) -> dict[str, object]:
    """Congestion should rise with load and fall with generation and capacity.

    If it does, the worst case over the box sits at a corner and any scenario
    set containing that corner finds it exactly -- a structural statement the
    grid alone cannot make. Violations are reported with their magnitude so a
    MIP-gap artifact is not mistaken for real non-monotonicity.
    """
    by_point = {_coordinate(r): float(r["congestion_mw"]) for r in records}
    axes = [sorted({point[index] for point in by_point}) for index in range(3)]
    expectations = [("load_scale", 1), ("generation_scale", -1), ("line_capacity_scale", -1)]
    report: dict[str, object] = {}
    for index, (name, direction) in enumerate(expectations):
        others = [axes[other] for other in range(3) if other != index]
        other_indices = [other for other in range(3) if other != index]
        violations, comparisons, largest = 0, 0, 0.0
        for combination in product(*others):
            series = []
            for value in axes[index]:
                key = [0.0, 0.0, 0.0]
                key[index] = value
                for slot, other_value in zip(other_indices, combination, strict=True):
                    key[slot] = other_value
                point = tuple(key)
                if point in by_point:
                    series.append(by_point[point])
            for left, right in zip(series, series[1:], strict=False):
                comparisons += 1
                change = (right - left) * direction
                if change < -1e-9:
                    violations += 1
                    largest = max(largest, -change)
        report[name] = {
            "expected": "increasing" if direction > 0 else "decreasing",
            "violations": violations,
            "comparisons": comparisons,
            "largest_violation_mw": largest,
        }
    return report


def _summarize(
    records: list[dict[str, object]], scenarios: list[Scenario]
) -> dict[str, object]:
    """Grid worst vs five-point worst, per configuration.

    Deduplicates by coordinate first: a scenario point can coincide with a grid
    node, and the duplicate silently double-counts that scenario's probability
    in the weighted mean (it moved status_quo from 26.459 to 26.679 before this
    was caught).
    """
    five_point_keys = {
        (s.load_scale, s.generation_scale, s.line_capacity_scale) for s in scenarios
    }
    summary: dict[str, object] = {}
    for label in CONFIGURATIONS:
        deduplicated: dict[tuple[float, float, float], dict[str, object]] = {}
        for record in records:
            if record["configuration"] == label:
                deduplicated[_coordinate(record)] = record
        subset = list(deduplicated.values())

        grid_records = [r for r in subset if _coordinate(r) not in five_point_keys]
        grid_values = np.array([float(r["congestion_mw"]) for r in grid_records])
        grid_worst = float(grid_values.max())
        worst_point = max(grid_records, key=lambda r: float(r["congestion_mw"]))

        # One lookup per scenario, so each probability is counted exactly once.
        by_point = {_coordinate(r): float(r["congestion_mw"]) for r in subset}
        five_point_values = {
            s.name: by_point[(s.load_scale, s.generation_scale, s.line_capacity_scale)]
            for s in scenarios
        }
        weighted = sum(s.probability * five_point_values[s.name] for s in scenarios)
        sampled_worst = max(five_point_values.values())

        not_optimal = sum(1 for r in subset if r["termination_status"] != "OPTIMAL")
        summary[label] = {
            "weighted_five_point_mw": weighted,
            "sampled_worst_of_five_mw": sampled_worst,
            "grid_worst_mw": grid_worst,
            "grid_points": int(grid_values.size),
            "under_report_vs_five_point_worst": (
                (grid_worst - sampled_worst) / grid_worst if grid_worst > 0 else 0.0
            ),
            "under_report_vs_weighted": (
                (grid_worst - weighted) / grid_worst if grid_worst > 0 else 0.0
            ),
            "grid_worst_at": dict(
                zip(
                    ("load_scale", "generation_scale", "line_capacity_scale"),
                    _coordinate(worst_point),
                    strict=True,
                )
            ),
            # Load-bearing caveat: a TIME_LIMIT evaluation returns a feasible
            # incumbent, so its congestion is an upper bound on what the
            # operator could achieve at that point. It can overstate the worst
            # case, never understate it.
            "solves_not_proven_optimal": not_optimal,
            "solves_total": len(subset),
            "monotonicity": _monotonicity(subset),
            "budget_curve": _budget_curve(grid_values, grid_worst),
        }
        logger.info(
            "%-20s weighted=%.3f  five-point worst=%.3f  grid worst=%.3f (%d pts) "
            "-> under-report %.1f%%  [%d/%d solves not proven optimal]",
            label,
            weighted,
            sampled_worst,
            grid_worst,
            grid_values.size,
            100.0 * float(summary[label]["under_report_vs_five_point_worst"]),
            not_optimal,
            len(subset),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="9,9,5", help="points per axis: load,gen,capacity")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--eval-time-limit", type=float, default=120.0)
    parser.add_argument(
        "--resummarize",
        default="",
        help="recompute the summary from an existing record's evaluations, no solves",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--plan-from",
        default="experiments/results/congestion_seed7_fixed/metrics.json",
        help="artifact whose Layer A plan this sweep evaluates; read, never assumed",
    )
    args = parser.parse_args()
    global PLAN_BUILDS
    PLAN_BUILDS = _load_plan_builds(Path(args.plan_from))
    logger.info("plan under test (from %s): %s", args.plan_from, list(PLAN_BUILDS))

    if args.resummarize:
        source = Path(args.resummarize)
        existing = json.loads(source.read_text())
        scenarios = build_scenario_set(existing["scenario_kind"])
        existing["summary"] = _summarize(existing["evaluations"], scenarios)
        existing["run"]["resummarized_utc"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        destination = Path(args.out) if args.out else source
        destination.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        logger.info("rewrote summary in %s", destination)
        return

    counts = [int(part) for part in args.grid.split(",")]
    if len(counts) != 3 or any(count < 2 for count in counts):
        raise SystemExit("--grid needs three axis counts, each >= 2")

    scenarios = build_scenario_set(SCENARIO_KIND)
    box = _box(scenarios)
    axes = {
        name: np.linspace(low, high, count)
        for (name, (low, high)), count in zip(box.items(), counts, strict=True)
    }
    logger.info("box: %s", {k: (round(v[0], 3), round(v[1], 3)) for k, v in box.items()})

    jobs: list[tuple[str, float, float, float]] = []
    for label in CONFIGURATIONS:
        for load, generation, capacity in product(*axes.values()):
            jobs.append((label, float(load), float(generation), float(capacity)))
        # The five points themselves, evaluated singly so they are directly
        # comparable with grid points (the published number weights them).
        for scenario in scenarios:
            jobs.append(
                (
                    label,
                    scenario.load_scale,
                    scenario.generation_scale,
                    scenario.line_capacity_scale,
                )
            )
    logger.info("%d evaluations on %d workers", len(jobs), args.workers)

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.eval_time_limit,)
    ) as pool:
        records = list(pool.map(_evaluate, jobs, chunksize=4))
    logger.info("evaluations complete")

    summary = _summarize(records, scenarios)

    record = {
        "feeder": "ieee33",
        "scenario_kind": SCENARIO_KIND,
        "plan_builds": list(PLAN_BUILDS),
        "box": {name: list(bounds) for name, bounds in box.items()},
        "grid_counts": counts,
        "summary": summary,
        "evaluations": records,
        "run": {
            "eval_time_limit_s": args.eval_time_limit,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "note": (
                "grid_worst is a dense-sample LOWER bound on the true worst case "
                "over the continuous box, not a certificate; it can only widen the "
                "measured gap, never narrow it"
            ),
        },
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
