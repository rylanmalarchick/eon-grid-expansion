from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from eon.formulations.layer_b import (
    build_layer_b_surrogate,
    solve_layer_b_surrogate,
    validate_layer_b_surrogate,
)
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import Scenario, build_scenario_set
from eon.mps.protocol import run_mps_protocol
from eon.validation import validate_finite_array

app = typer.Typer(add_completion=False, help="Generate Layer A/Layer B hardness records.")

# Classification thresholds. An instance is "hard" only when the MPS and tree-tensor
# backends both flag it as negative, Layer B validates, and the Layer B MIP gap is open.
# tree-TN reference gap above which the tree backend is "negative":
_TREE_REFERENCE_GAP_THRESHOLD = 0.5
# per-ordering energy gap above which MPS is "negative":
_MPS_ORDERING_GAP_THRESHOLD = 0.5
# min top-k agreement for Layer B to count as validated:
_LAYER_B_AGREEMENT_THRESHOLD = 0.7
# max feasibility-violation rate for Layer B validation:
_FEASIBILITY_VIOLATION_THRESHOLD = 0.1
# min Layer B MIP gap for an instance to qualify as hard:
_HARD_MIP_GAP_THRESHOLD = 0.01
# min ordering results before a chi plateau is meaningful:
_MIN_ORDERINGS_FOR_PLATEAU = 3


@app.command()
def main(
    output_path: Annotated[
        Path,
        typer.Option(help="JSONL output path."),
    ] = Path("instances/hard_instances.jsonl"),
    candidate_count: Annotated[
        int,
        typer.Option(help="Candidate lines per instance."),
    ] = 20,
    neighborhood_size: Annotated[
        int,
        typer.Option(help="Layer B neighborhood size."),
    ] = 20,
    max_new_lines: Annotated[
        int,
        typer.Option(help="Build budget."),
    ] = 3,
    time_limit: Annotated[
        float,
        typer.Option(help="Per-solve time limit in seconds."),
    ] = 10.0,
    scenario_kind: Annotated[
        str,
        typer.Option(help="Scenario set identifier."),
    ] = "stressed_five_point",
    large_candidate_count: Annotated[
        int,
        typer.Option(help="Candidate count for the explicit large-instance IEEE 123 pass."),
    ] = 50,
) -> None:
    records = classify_default_instance_zoo(
        candidate_count=candidate_count,
        neighborhood_size=neighborhood_size,
        max_new_lines=max_new_lines,
        time_limit=time_limit,
        scenario_kind=scenario_kind,
        large_candidate_count=large_candidate_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    typer.echo(json.dumps({"count": len(records), "output_path": str(output_path)}, indent=2))


def classify_default_instance_zoo(
    *,
    candidate_count: int = 20,
    neighborhood_size: int = 20,
    max_new_lines: int = 3,
    time_limit: float = 10.0,
    scenario_kind: str = "stressed_five_point",
    large_candidate_count: int = 50,
    cost_per_km: float = 10_000.0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    config = ExpansionProblemConfig(
        max_new_lines=max_new_lines,
        time_limit_s=time_limit,
        n_scenarios_aggregated=3,
    )
    scenarios = build_scenario_set(scenario_kind)
    feeder_schedule = (
        (
            "ieee123",
            candidate_count,
            neighborhood_size,
            (
                ("adversarial_long_range", 10),
                ("community_bridging", 6),
                ("useful_adversarial", 2),
                ("distance_weighted", 2),
            ),
        ),
        (
            "ieee123",
            large_candidate_count,
            large_candidate_count,
            (
                ("adversarial_long_range", 4),
                ("community_bridging", 2),
                ("useful_adversarial", 1),
            ),
        ),
        (
            "ieee33",
            min(candidate_count, 20),
            min(neighborhood_size, 20),
            (
                ("adversarial_long_range", 1),
                ("community_bridging", 1),
                ("useful_adversarial", 1),
                ("distance_weighted", 1),
                ("random_uniform", 1),
            ),
        ),
    )

    for (
        feeder,
        feeder_candidate_count,
        feeder_neighborhood_size,
        family_schedule,
    ) in feeder_schedule:
        net = load_distribution_feeder(feeder)
        # Compute baseline stress for stress-aware candidate generation
        baseline_stress = _compute_baseline_stress(
            net, scenarios, config,
        )
        for family, repeat_count in family_schedule:
            for repeat_index in range(repeat_count):
                seed = 7 + 17 * repeat_index
                candidates = generate_candidate_lines(
                    net,
                    family,
                    feeder_candidate_count,
                    seed=seed,
                    cost_per_km=cost_per_km,
                    stress_info=baseline_stress if family == "useful_adversarial" else None,
                )
                instance_id = (
                    f"{feeder}:{family}:seed{seed}:n{feeder_candidate_count}:"
                    f"k{max_new_lines}:rep{repeat_index}"
                )
                try:
                    layer_a = solve_lindistflow_expansion(net, scenarios, candidates, config)
                    surrogate = build_layer_b_surrogate(
                        net,
                        scenarios,
                        candidates,
                        layer_a,
                        config,
                        neighborhood_size=min(feeder_neighborhood_size, len(candidates)),
                        evaluation_time_limit_s=max(3.0, time_limit / 3.0),
                        pair_sample_limit=min(250, feeder_neighborhood_size * 6),
                    )
                    validation = validate_layer_b_surrogate(
                        net,
                        scenarios,
                        candidates,
                        surrogate,
                        config,
                        top_k=min(5, max(2, len(surrogate.variables))),
                        evaluation_time_limit_s=max(3.0, time_limit / 3.0),
                    )
                    layer_b = solve_layer_b_surrogate(
                        surrogate,
                        time_limit_s=time_limit,
                        mip_gap=config.mip_gap,
                    )
                    mps = run_mps_protocol(
                        surrogate,
                        instance_name=f"{feeder}_{family}_{feeder_candidate_count}_{seed}",
                        reference_energy=layer_b.objective_value,
                    )
                    ordering_gaps = [
                        ordering.best_energy - mps.reference_energy
                        for ordering in mps.ordering_results
                    ]
                    expected_energy_gaps = [
                        min(ordering.chi_curve.values()) - mps.exact_qaoa_energy
                        for ordering in mps.ordering_results
                        if ordering.chi_curve and math.isfinite(mps.exact_qaoa_energy)
                    ]
                    best_mps_energy = min(
                        (ordering.best_energy for ordering in mps.ordering_results),
                        default=float("inf"),
                    )
                    best_expected_energy = min(
                        (
                            min(ordering.chi_curve.values())
                            for ordering in mps.ordering_results
                            if ordering.chi_curve
                        ),
                        default=float("inf"),
                    )
                    tree_result = mps.tree_tn_result
                    tree_negative = bool(
                        tree_result is not None
                        and tree_result.reference_gap > _TREE_REFERENCE_GAP_THRESHOLD
                    )
                    chi_flat = _chi_plateau_across_orderings(mps)
                    entropy_persistent = _entropy_persists_to_chi_max(mps)
                    layer_b_validated = (
                        validation.top_k_agreement >= _LAYER_B_AGREEMENT_THRESHOLD
                        and validation.feasibility_violation_rate
                        <= _FEASIBILITY_VIOLATION_THRESHOLD
                    )
                    mps_negative = bool(
                        mps.backend == "juliqaoa_mps"
                        and ordering_gaps
                        and all(gap > _MPS_ORDERING_GAP_THRESHOLD for gap in ordering_gaps)
                        and chi_flat
                        and entropy_persistent
                    )
                    final_classification = (
                        "hard"
                        if (
                            mps_negative
                            and tree_negative
                            and layer_b_validated
                            and (layer_b.mip_gap or 0.0) > _HARD_MIP_GAP_THRESHOLD
                        )
                        else "easy"
                        if not mps_negative and not tree_negative
                        else "ambiguous"
                    )
                    validate_finite_array(
                        np.asarray(list(layer_a.aggregate_metrics.values()), dtype=float),
                        name=f"{feeder}_{family}_{repeat_index}_layer_a_aggregate_metrics",
                    )
                    objective_vector = [
                        layer_a.objective_value,
                        layer_b.objective_value,
                        best_mps_energy,
                        best_expected_energy,
                        mps.reference_energy,
                    ]
                    if math.isfinite(mps.exact_ground_energy):
                        objective_vector.append(mps.exact_ground_energy)
                    if math.isfinite(mps.exact_qaoa_energy):
                        objective_vector.append(mps.exact_qaoa_energy)
                    validate_finite_array(
                        np.asarray(objective_vector, dtype=float),
                        name=f"{feeder}_{family}_{repeat_index}_objective_vector",
                    )
                except Exception as exc:  # noqa: BLE001
                    # Deliberate broad catch: this is a sweep over many independent
                    # solver invocations (Gurobi, agentbible, MPS, numpy), any of which
                    # can fail in unrelated ways. A missed exception type would abort the
                    # whole zoo and lose every instance computed so far. The failure is
                    # recorded (not silently dropped) so the output reflects what was
                    # skipped and why. KeyboardInterrupt/SystemExit still propagate.
                    logging.exception("Failed to evaluate instance %s", instance_id)
                    records.append(
                        {
                            "instance_id": instance_id,
                            "feeder": feeder,
                            "candidate_family": family,
                            "candidate_count": feeder_candidate_count,
                            "candidate_seed": seed,
                            "max_new_lines": max_new_lines,
                            "scenario_kind": scenario_kind,
                            "final_classification": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                records.append(
                    {
                        "instance_id": instance_id,
                        "feeder": feeder,
                        "candidate_family": family,
                        "candidate_count": feeder_candidate_count,
                        "candidate_seed": seed,
                        "max_new_lines": max_new_lines,
                        "scenario_kind": scenario_kind,
                        "layer_a": {
                            "objective_value": layer_a.objective_value,
                            "runtime_s": layer_a.runtime_s,
                            "mip_gap": layer_a.mip_gap,
                            "status": layer_a.termination_status,
                            "selected_candidates": list(layer_a.selected_candidates),
                            "aggregate_metrics": layer_a.aggregate_metrics,
                        },
                        "layer_b": {
                            "variable_count": len(surrogate.variables),
                            "objective_value": layer_b.objective_value,
                            "runtime_s": layer_b.runtime_s,
                            "mip_gap": layer_b.mip_gap,
                            "best_bound": layer_b.best_bound,
                            "status": layer_b.status,
                            "selected_candidates": list(layer_b.selected_candidates),
                            "top_k_agreement": validation.top_k_agreement,
                            "feasibility_violation_rate": validation.feasibility_violation_rate,
                            "objective_distortion": validation.objective_distortion,
                            "feasible_plan_count": validation.feasible_plan_count,
                            "evaluated_plan_count": validation.evaluated_plan_count,
                        },
                        "mps_results": {
                            "backend": mps.backend,
                            "chi_max_reached": mps.chi_max_reached,
                            "ordering_sensitivity": mps.ordering_sensitivity,
                            "best_energy": best_mps_energy,
                            "best_expected_energy": best_expected_energy,
                            "reference_energy": mps.reference_energy,
                            "exact_ground_energy": mps.exact_ground_energy,
                            "exact_qaoa_energy": mps.exact_qaoa_energy,
                            "vs_gurobi_gap": best_mps_energy - mps.reference_energy,
                            "vs_qaoa_gap": max(expected_energy_gaps, default=float("nan")),
                            "optimized_angles": list(mps.optimized_angles),
                            "worst_order_gap": max(ordering_gaps, default=float("inf")),
                            "ordering_results": [
                                {
                                    "ordering": ordering.ordering,
                                    "best_energy": ordering.best_energy,
                                    "chi_curve": ordering.chi_curve,
                                    "sampled_energy_curve": ordering.sampled_energy_curve,
                                    "entropy_curve": ordering.entropy_curve,
                                    "entanglement_entropy_profiles": (
                                        ordering.entanglement_entropy_profiles
                                    ),
                                    "energy_variance": ordering.energy_variance,
                                    "stopped_reason": ordering.stopped_reason,
                                }
                                for ordering in mps.ordering_results
                            ],
                        },
                        "tree_tn_results": (
                            None
                            if tree_result is None
                            else {
                                "backend": tree_result.backend,
                                "chi_max_reached": tree_result.chi_max_reached,
                                "chi_curve": tree_result.chi_curve,
                                "sampled_energy_curve": tree_result.sampled_energy_curve,
                                "sample_variance_curve": tree_result.sample_variance_curve,
                                "stopped_reason": tree_result.stopped_reason,
                                "best_energy": tree_result.best_energy,
                                "reference_gap": tree_result.reference_gap,
                                "variable_order": list(tree_result.variable_order),
                                "max_bag_size": tree_result.max_bag_size,
                            }
                        ),
                        "classification": {
                            "layer_a_nontrivial": (
                                layer_a.aggregate_metrics["weighted_voltage_violation_pu"] > 0.0
                                or layer_a.aggregate_metrics["weighted_congestion_mw"] > 0.0
                            ),
                            "layer_b_validated": layer_b_validated,
                            "chi_flat_across_orderings": chi_flat,
                            "entropy_persistent_to_chi_max": entropy_persistent,
                            "mps_negative_candidate": mps_negative,
                            "tree_tn_negative_candidate": tree_negative,
                            "final_classification": final_classification,
                        },
                    }
                )
    return records


def _chi_plateau_across_orderings(
    mps: object,
    *,
    tolerance: float = 1e-3,
    tail_length: int = 3,
) -> bool:
    ordering_results = getattr(mps, "ordering_results", [])
    if len(ordering_results) < _MIN_ORDERINGS_FOR_PLATEAU:
        return False

    for ordering in ordering_results:
        if len(ordering.sampled_energy_curve) < tail_length:
            return False
        tail = [
            energy
            for _, energy in sorted(
                ordering.sampled_energy_curve.items(),
                key=lambda item: item[0],
            )[-tail_length:]
        ]
        if max(tail) - min(tail) > tolerance:
            return False
    return True


def _entropy_persists_to_chi_max(
    mps: object,
    *,
    min_entropy: float = 0.5,
) -> bool:
    ordering_results = getattr(mps, "ordering_results", [])
    if not ordering_results:
        return False

    for ordering in ordering_results:
        if not ordering.entropy_curve:
            return False
        chi_max = max(ordering.entropy_curve)
        if ordering.entropy_curve[chi_max] < min_entropy:
            return False
    return True


def _compute_baseline_stress(
    net: object,
    scenarios: list[Scenario],
    config: ExpansionProblemConfig,
) -> dict[int, float]:
    """Run a no-candidate baseline solve and extract per-bus stress scores."""
    baseline = solve_lindistflow_expansion(
        net,
        scenarios,
        [],
        config,
    )
    stress: dict[int, float] = {}
    for _scenario_name, voltage_map in baseline.bus_voltage_pu.items():
        for bus, voltage in voltage_map.items():
            bus_stress = max(config.voltage_min_pu - voltage, 0.0) + max(
                voltage - config.voltage_max_pu, 0.0,
            )
            stress[bus] = stress.get(bus, 0.0) + bus_stress
    for metric in baseline.scenario_metrics:
        # Spread thermal stress evenly across buses as a proxy
        bus_count = max(len(baseline.bus_voltage_pu.get(metric.scenario, {})), 1)
        per_bus_thermal = metric.thermal_overlimit_mw / bus_count
        for bus in baseline.bus_voltage_pu.get(metric.scenario, {}):
            stress[bus] = stress.get(bus, 0.0) + per_bus_thermal
    return stress


if __name__ == "__main__":
    app()
