"""Re-score the instance zoo by structural coupling tree-width.

Rebuilds each instance's Layer B surrogate via the shared zoo schedule and
reports the tree-width of its physics-derived coupling graph -- the leading
MPS-hardness indicator (MPS contraction cost is exp(tree-width); PLAN.txt D13,
reading.txt R30). Answers: was any current instance ever in the hard
(high-tree-width) regime, or are they MPS-easy by construction?

Run from workspace/:  python scripts/rescore_treewidth.py
"""

from __future__ import annotations

from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import (
    _compute_baseline_stress,
    build_instance_surrogate,
    default_feeder_schedule,
)
from eon.instances.scenarios import build_scenario_set
from eon.instances.treewidth import MPS_EASY_TREEWIDTH, coupling_diagnostics

# Mirrors classify_default_instance_zoo's defaults so the rebuilt instances
# match the committed zoo.
CANDIDATE_COUNT = 20
NEIGHBORHOOD_SIZE = 20
LARGE_CANDIDATE_COUNT = 50
MAX_NEW_LINES = 3
TIME_LIMIT = 10.0
COST_PER_KM = 10_000.0
SCENARIO_KIND = "stressed_five_point"


def main() -> None:
    config = ExpansionProblemConfig(
        max_new_lines=MAX_NEW_LINES,
        time_limit_s=TIME_LIMIT,
        n_scenarios_aggregated=3,
    )
    scenarios = build_scenario_set(SCENARIO_KIND)
    schedule = default_feeder_schedule(
        candidate_count=CANDIDATE_COUNT,
        neighborhood_size=NEIGHBORHOOD_SIZE,
        large_candidate_count=LARGE_CANDIDATE_COUNT,
    )

    rows: list[dict[str, object]] = []
    for feeder, feeder_candidate_count, feeder_neighborhood_size, family_schedule in schedule:
        net = load_distribution_feeder(feeder)
        baseline_stress = _compute_baseline_stress(net, scenarios, config)
        for family, repeat_count in family_schedule:
            for repeat_index in range(repeat_count):
                seed = 7 + 17 * repeat_index
                instance_id = (
                    f"{feeder}:{family}:seed{seed}:n{feeder_candidate_count}:"
                    f"k{MAX_NEW_LINES}:rep{repeat_index}"
                )
                try:
                    _, layer_a, surrogate = build_instance_surrogate(
                        net,
                        scenarios,
                        config,
                        family=family,
                        candidate_count=feeder_candidate_count,
                        neighborhood_size=feeder_neighborhood_size,
                        seed=seed,
                        time_limit=TIME_LIMIT,
                        cost_per_km=COST_PER_KM,
                        baseline_stress=baseline_stress,
                    )
                    diag = coupling_diagnostics(surrogate)
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "feeder": feeder,
                            "family": family,
                            "n": len(surrogate.variables),
                            "builds": len(layer_a.selected_candidates),
                            "struct_tw": diag.structural_treewidth,
                            "eff_tw": diag.effective_treewidth,
                            "ratio": diag.coupling_field_ratio,
                            "edges": diag.structural_edges,
                            "eff_edges": diag.effective_edges,
                            "mps_easy": diag.mps_easy,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    # Re-score is a sweep; record the failure, keep going.
                    rows.append(
                        {"instance_id": instance_id, "error": f"{type(exc).__name__}: {exc}"}
                    )

    scored = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    header = (
        f"{'instance_id':52} {'n':>3} {'bld':>3} {'s_tw':>4} {'e_tw':>4} "
        f"{'J/h':>8} {'edg':>4} {'eedg':>4} mps_easy"
    )
    print(header)
    print("-" * len(header))
    for r in scored:
        ratio = r["ratio"]
        ratio_s = f"{float(ratio):8.3f}" if ratio is not None else f"{'None':>8}"
        print(
            f"{r['instance_id']:52} {r['n']:>3} {r['builds']:>3} {r['struct_tw']:>4} "
            f"{r['eff_tw']:>4} {ratio_s} {r['edges']:>4} {r['eff_edges']:>4} {r['mps_easy']}"
        )

    print()
    if scored:
        struct = sorted(int(r["struct_tw"]) for r in scored)
        effective = sorted(int(r["eff_tw"]) for r in scored)
        hard = sum(1 for t in effective if t > MPS_EASY_TREEWIDTH)
        print(f"scored: {len(scored)}   failed: {len(failed)}")
        print(
            f"structural tree-width (all couplings): "
            f"min={struct[0]} median={struct[len(struct) // 2]} max={struct[-1]}"
        )
        print(
            f"effective tree-width (load-bearing):   "
            f"min={effective[0]} median={effective[len(effective) // 2]} max={effective[-1]}"
        )
        easy = len(scored) - hard
        print(f"MPS-easy (effective tw <= {MPS_EASY_TREEWIDTH}): {easy}/{len(scored)}")
        print(f"potentially-hard (effective tw > {MPS_EASY_TREEWIDTH}): {hard}/{len(scored)}")
        ratios = sorted(float(r["ratio"]) for r in scored if r["ratio"] is not None)
        if ratios:
            print(
                f"coupling/field ratio J/h: min={ratios[0]:.4f} "
                f"median={ratios[len(ratios) // 2]:.4f} max={ratios[-1]:.4f} "
                f"(of {len(ratios)}/{len(scored)} with nonzero fields)"
            )
        by_family: dict[str, list[int]] = {}
        for r in scored:
            by_family.setdefault(str(r["family"]), []).append(int(r["eff_tw"]))
        print("max effective tree-width by family:", {f: max(v) for f, v in by_family.items()})
    for r in failed:
        print("FAILED", r["instance_id"], "--", r["error"])


if __name__ == "__main__":
    main()
