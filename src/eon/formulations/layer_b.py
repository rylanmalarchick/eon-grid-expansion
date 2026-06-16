from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    ExpansionResult,
    solve_lindistflow_expansion,
)
from eon.instances.candidate_lines import CandidateLine
from eon.instances.scenarios import Scenario
from eon.validation import validate_finite_array, validate_normalized_l1


@dataclass(frozen=True, slots=True)
class LayerBVariable:
    name: str
    candidate: CandidateLine
    default_value: int
    stress_score: float


@dataclass(frozen=True, slots=True)
class LayerBSurrogate:
    variables: tuple[LayerBVariable, ...]
    offset: float
    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    fixed_builds: dict[str, int]
    max_new_lines: int
    base_objective: float
    base_selected_count: int

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def actual_build_count(self, toggle_decisions: dict[str, int]) -> int:
        return sum(self.actual_builds(toggle_decisions).values())

    def actual_builds(self, toggle_decisions: dict[str, int]) -> dict[str, int]:
        builds = dict(self.fixed_builds)
        for variable in self.variables:
            toggle = int(toggle_decisions.get(variable.name, 0))
            builds[variable.name] = variable.default_value ^ toggle
        return builds

    def surrogate_objective(self, toggle_decisions: dict[str, int]) -> float:
        value = self.offset
        for variable in self.variables:
            toggle = int(toggle_decisions.get(variable.name, 0))
            value += self.linear[variable.name] * toggle
        for (left, right), coefficient in self.quadratic.items():
            value += (
                coefficient
                * int(toggle_decisions.get(left, 0))
                * int(toggle_decisions.get(right, 0))
            )
        return float(value)


@dataclass(frozen=True, slots=True)
class LayerBPlanEvaluation:
    toggle_decisions: dict[str, int]
    actual_builds: dict[str, int]
    surrogate_objective: float
    parent_objective: float
    parent_status: str
    parent_feasible: bool


@dataclass(frozen=True, slots=True)
class LayerBValidationResult:
    evaluated_plan_count: int
    feasible_plan_count: int
    top_k: int
    top_k_agreement: float
    feasibility_violation_rate: float
    objective_distortion: float
    best_parent_objective: float
    best_surrogate_parent_objective: float


@dataclass(frozen=True, slots=True)
class LayerBSolution:
    objective_value: float
    toggle_decisions: dict[str, int]
    actual_builds: dict[str, int]
    selected_candidates: tuple[str, ...]
    status: str
    runtime_s: float = 0.0
    mip_gap: float | None = None
    best_bound: float | None = None


def build_layer_b_surrogate(
    net: Any,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    layer_a_result: ExpansionResult,
    problem_config: ExpansionProblemConfig,
    *,
    neighborhood_size: int = 6,
    evaluation_time_limit_s: float = 5.0,
    pair_sample_limit: int | None = None,
) -> LayerBSurrogate:
    validate_normalized_l1(
        np.asarray([scenario.probability for scenario in scenarios], dtype=float),
        name="layer_b_scenario_probabilities",
    )
    default_builds = _default_builds(candidates, layer_a_result)
    selected_candidates = _select_neighborhood(
        candidates,
        layer_a_result,
        problem_config,
        neighborhood_size=neighborhood_size,
        default_builds=default_builds,
    )
    variables = tuple(selected_candidates)
    evaluation_config = replace(problem_config, time_limit_s=evaluation_time_limit_s)
    offset = layer_a_result.objective_value
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    infeasibility_penalty = layer_a_result.objective_value + 10_000_000.0

    for variable in variables:
        toggles = {variable.name: 1}
        parent_objective = _evaluate_parent_objective(
            net,
            scenarios,
            candidates,
            evaluation_config,
            default_builds,
            variables,
            toggles,
            base_objective=layer_a_result.objective_value,
            infeasibility_penalty=infeasibility_penalty,
        )
        linear[variable.name] = parent_objective - offset

    sampled_pairs = _select_pair_candidates(
        variables,
        max_pairs=pair_sample_limit or min(200, len(variables) * max(len(variables) - 1, 0) // 2),
    )
    for left, right in sampled_pairs:
        toggles = {left.name: 1, right.name: 1}
        parent_objective = _evaluate_parent_objective(
            net,
            scenarios,
            candidates,
            evaluation_config,
            default_builds,
            variables,
            toggles,
            base_objective=layer_a_result.objective_value,
            infeasibility_penalty=infeasibility_penalty,
        )
        quadratic[(left.name, right.name)] = (
            parent_objective - offset - linear[left.name] - linear[right.name]
        )

    validate_finite_array(
        np.asarray([offset, *linear.values(), *quadratic.values()], dtype=float),
        name="layer_b_surrogate_coefficients",
    )

    return LayerBSurrogate(
        variables=variables,
        offset=offset,
        linear=linear,
        quadratic=quadratic,
        fixed_builds=default_builds,
        max_new_lines=problem_config.max_new_lines,
        base_objective=layer_a_result.objective_value,
        base_selected_count=sum(default_builds.values()),
    )


def validate_layer_b_surrogate(
    net: Any,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    surrogate: LayerBSurrogate,
    problem_config: ExpansionProblemConfig,
    *,
    top_k: int = 5,
    evaluation_time_limit_s: float = 5.0,
    violation_tolerance: float = 1e-6,
) -> LayerBValidationResult:
    evaluation_config = replace(problem_config, time_limit_s=evaluation_time_limit_s)
    if len(surrogate.variables) <= 16:
        plans = enumerate_layer_b_plans(net, scenarios, candidates, surrogate, evaluation_config)
    else:
        plans = evaluate_layer_b_validation_sample(
            net,
            scenarios,
            candidates,
            surrogate,
            evaluation_config,
            top_k=top_k,
        )
    feasible_plans = [
        plan
        for plan in plans
        if sum(plan.actual_builds.values()) <= surrogate.max_new_lines
        and plan.parent_status != "INFEASIBLE"
    ]
    sorted_by_parent = sorted(feasible_plans, key=lambda plan: plan.parent_objective)
    sorted_by_surrogate = sorted(feasible_plans, key=lambda plan: plan.surrogate_objective)
    effective_k = min(top_k, len(feasible_plans))

    if not feasible_plans:
        return LayerBValidationResult(
            evaluated_plan_count=len(plans),
            feasible_plan_count=0,
            top_k=0,
            top_k_agreement=0.0,
            feasibility_violation_rate=1.0,
            objective_distortion=float("inf"),
            best_parent_objective=float("inf"),
            best_surrogate_parent_objective=float("inf"),
        )

    top_parent = {
        _plan_key(plan.toggle_decisions) for plan in sorted_by_parent[:effective_k]
    }
    top_surrogate = {
        _plan_key(plan.toggle_decisions) for plan in sorted_by_surrogate[:effective_k]
    }
    agreement = len(top_parent & top_surrogate) / effective_k
    surrogate_top = sorted_by_surrogate[:effective_k]
    feasibility_violation_rate = (
        sum(not plan.parent_feasible for plan in surrogate_top) / effective_k
        if effective_k
        else 0.0
    )
    best_parent = sorted_by_parent[0]
    best_surrogate = sorted_by_surrogate[0]
    objective_distortion = abs(
        best_surrogate.parent_objective - best_parent.parent_objective
    ) / max(
        abs(best_parent.parent_objective),
        1.0,
    )
    if (
        best_parent.parent_objective == float("inf")
        or best_surrogate.parent_objective == float("inf")
    ):
        objective_distortion = float("inf")

    return LayerBValidationResult(
        evaluated_plan_count=len(plans),
        feasible_plan_count=len(feasible_plans),
        top_k=effective_k,
        top_k_agreement=agreement,
        feasibility_violation_rate=feasibility_violation_rate,
        objective_distortion=objective_distortion,
        best_parent_objective=best_parent.parent_objective,
        best_surrogate_parent_objective=best_surrogate.parent_objective,
    )


def enumerate_layer_b_plans(
    net: Any,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    surrogate: LayerBSurrogate,
    problem_config: ExpansionProblemConfig,
) -> list[LayerBPlanEvaluation]:
    plans: list[LayerBPlanEvaluation] = []
    for toggle_values in product((0, 1), repeat=len(surrogate.variables)):
        toggle_decisions = {
            variable.name: toggle
            for variable, toggle in zip(surrogate.variables, toggle_values, strict=False)
        }
        actual_builds = surrogate.actual_builds(toggle_decisions)
        result = solve_lindistflow_expansion(
            net,
            scenarios,
            candidates,
            problem_config,
            fixed_builds=actual_builds,
        )
        parent_feasible = (
            result.termination_status != "INFEASIBLE"
            and result.objective_value != float("inf")
        )
        plans.append(
            LayerBPlanEvaluation(
                toggle_decisions=toggle_decisions,
                actual_builds=actual_builds,
                surrogate_objective=surrogate.surrogate_objective(toggle_decisions),
                parent_objective=result.objective_value,
                parent_status=result.termination_status,
                parent_feasible=parent_feasible,
            )
        )
    return plans


def evaluate_layer_b_validation_sample(
    net: Any,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    surrogate: LayerBSurrogate,
    problem_config: ExpansionProblemConfig,
    *,
    top_k: int,
    seed: int = 7,
) -> list[LayerBPlanEvaluation]:
    sampled_toggles = _sample_validation_toggle_plans(surrogate, top_k=top_k, seed=seed)
    plans: list[LayerBPlanEvaluation] = []
    for toggle_decisions in sampled_toggles:
        actual_builds = surrogate.actual_builds(toggle_decisions)
        result = solve_lindistflow_expansion(
            net,
            scenarios,
            candidates,
            problem_config,
            fixed_builds=actual_builds,
        )
        parent_feasible = (
            result.termination_status != "INFEASIBLE"
            and result.objective_value != float("inf")
        )
        plans.append(
            LayerBPlanEvaluation(
                toggle_decisions=toggle_decisions,
                actual_builds=actual_builds,
                surrogate_objective=surrogate.surrogate_objective(toggle_decisions),
                parent_objective=result.objective_value,
                parent_status=result.termination_status,
                parent_feasible=parent_feasible,
            )
        )
    return plans


def solve_layer_b_surrogate(
    surrogate: LayerBSurrogate,
    *,
    time_limit_s: float | None = None,
    mip_gap: float | None = None,
) -> LayerBSolution:
    model = gp.Model("layer_b_surrogate")
    model.Params.OutputFlag = 0
    if time_limit_s is not None:
        model.Params.TimeLimit = time_limit_s
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    toggle_vars = {
        variable.name: model.addVar(vtype=GRB.BINARY, name=f"toggle[{variable.name}]")
        for variable in surrogate.variables
    }

    actual_build_expr = gp.quicksum(
        (1 - toggle_vars[variable.name]) if variable.default_value else toggle_vars[variable.name]
        for variable in surrogate.variables
    ) + sum(
        value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    )
    model.addConstr(actual_build_expr <= surrogate.max_new_lines, name="actual_build_limit")

    objective = gp.QuadExpr(surrogate.offset)
    for variable in surrogate.variables:
        objective += surrogate.linear[variable.name] * toggle_vars[variable.name]
    for (left, right), coefficient in surrogate.quadratic.items():
        objective += coefficient * toggle_vars[left] * toggle_vars[right]
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    toggle_decisions = {name: int(round(var.X)) for name, var in toggle_vars.items()}
    actual_builds = surrogate.actual_builds(toggle_decisions)
    selected_candidates = tuple(
        sorted(name for name, selected in actual_builds.items() if selected)
    )
    return LayerBSolution(
        objective_value=float(model.ObjVal),
        toggle_decisions=toggle_decisions,
        actual_builds=actual_builds,
        selected_candidates=selected_candidates,
        status=_status_name(model.Status),
        runtime_s=float(model.Runtime),
        mip_gap=float(model.MIPGap) if model.SolCount else None,
        best_bound=float(model.ObjBound) if model.SolCount else None,
    )


def _select_neighborhood(
    candidates: list[CandidateLine],
    layer_a_result: ExpansionResult,
    problem_config: ExpansionProblemConfig,
    *,
    neighborhood_size: int,
    default_builds: dict[str, int],
) -> list[LayerBVariable]:
    bus_stress = _weighted_bus_stress(layer_a_result, problem_config)
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            default_builds[candidate.name],
            bus_stress.get(candidate.from_bus, 0.0) + bus_stress.get(candidate.to_bus, 0.0),
            _family_priority(candidate.family),
            -candidate.build_cost,
        ),
        reverse=True,
    )
    variables = []
    for candidate in ranked[:neighborhood_size]:
        variables.append(
            LayerBVariable(
                name=candidate.name,
                candidate=candidate,
                default_value=default_builds[candidate.name],
                stress_score=bus_stress.get(candidate.from_bus, 0.0)
                + bus_stress.get(candidate.to_bus, 0.0),
            )
        )
    return variables


def _weighted_bus_stress(
    layer_a_result: ExpansionResult,
    problem_config: ExpansionProblemConfig,
) -> dict[int, float]:
    stress: dict[int, float] = {}
    scenario_scores = {
        metric.scenario: metric.voltage_violation_pu + metric.thermal_overlimit_mw
        for metric in layer_a_result.scenario_metrics
    }
    selected_scenarios = {
        scenario_name
        for scenario_name, _ in sorted(
            scenario_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: max(problem_config.n_scenarios_aggregated, 1)]
    }
    probability_lookup = {
        metric.scenario: metric.probability for metric in layer_a_result.scenario_metrics
    }
    for scenario_name, bus_map in layer_a_result.bus_voltage_pu.items():
        if selected_scenarios and scenario_name not in selected_scenarios:
            continue
        probability = probability_lookup.get(scenario_name, 1.0)
        for bus, voltage in bus_map.items():
            bus_stress = max(problem_config.voltage_min_pu - voltage, 0.0) + max(
                voltage - problem_config.voltage_max_pu,
                0.0,
            )
            stress[bus] = stress.get(bus, 0.0) + probability * bus_stress
    return stress


def _select_pair_candidates(
    variables: tuple[LayerBVariable, ...],
    *,
    max_pairs: int,
) -> list[tuple[LayerBVariable, LayerBVariable]]:
    if max_pairs <= 0:
        return []

    ranked_pairs = sorted(
        combinations(variables, 2),
        key=lambda pair: (
            pair[0].stress_score + pair[1].stress_score,
            _pair_family_bonus(pair[0], pair[1]),
            -abs(pair[0].candidate.build_cost - pair[1].candidate.build_cost),
        ),
        reverse=True,
    )
    return ranked_pairs[:max_pairs]


def _sample_validation_toggle_plans(
    surrogate: LayerBSurrogate,
    *,
    top_k: int,
    seed: int,
) -> list[dict[str, int]]:
    rng = random.Random(seed)
    plans: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()

    def add_plan(toggle_decisions: dict[str, int]) -> None:
        key = _plan_key(toggle_decisions)
        if key in seen:
            return
        seen.add(key)
        plans.append(toggle_decisions)

    zero = {name: 0 for name in surrogate.variable_names}
    add_plan(zero)

    ranked_variables = sorted(
        surrogate.variables,
        key=lambda variable: (variable.default_value, variable.stress_score),
        reverse=True,
    )
    for variable in ranked_variables[: min(len(ranked_variables), max(4, top_k * 2))]:
        toggles = dict(zero)
        toggles[variable.name] = 1
        add_plan(toggles)

    pair_candidates = _select_pair_candidates(
        surrogate.variables,
        max_pairs=min(20, len(surrogate.variables) * max(len(surrogate.variables) - 1, 0) // 2),
    )
    for left, right in pair_candidates:
        toggles = dict(zero)
        toggles[left.name] = 1
        toggles[right.name] = 1
        add_plan(toggles)

    max_random = min(32, 2 * len(surrogate.variables))
    budget = max(
        0,
        surrogate.max_new_lines
        - sum(
            value
            for name, value in surrogate.fixed_builds.items()
            if name not in surrogate.variable_names
        ),
    )
    variable_names = list(surrogate.variable_names)
    while len(plans) < max(12, max_random) and variable_names:
        sampled = dict(zero)
        flip_count = min(
            len(variable_names),
            rng.randint(1, max(1, min(3, budget if budget > 0 else 3))),
        )
        for name in rng.sample(variable_names, k=flip_count):
            sampled[name] = 1
        add_plan(sampled)

    return plans


def _family_priority(family: str) -> int:
    priority = {
        "adversarial_long_range": 4,
        "useful_adversarial": 3,
        "community_bridging": 3,
        "distance_weighted": 2,
        "random_uniform": 1,
    }
    return priority.get(family, 0)


def _pair_family_bonus(left: LayerBVariable, right: LayerBVariable) -> int:
    return _family_priority(left.candidate.family) + _family_priority(right.candidate.family)


def _default_builds(
    candidates: list[CandidateLine],
    layer_a_result: ExpansionResult,
) -> dict[str, int]:
    return {
        candidate.name: int(round(layer_a_result.build_decisions.get(candidate.name, 0.0)))
        for candidate in candidates
    }


def _evaluate_parent_objective(
    net: Any,
    scenarios: list[Scenario],
    candidates: list[CandidateLine],
    problem_config: ExpansionProblemConfig,
    default_builds: dict[str, int],
    variables: tuple[LayerBVariable, ...],
    toggles: dict[str, int],
    *,
    base_objective: float,
    infeasibility_penalty: float,
) -> float:
    actual_builds = dict(default_builds)
    for variable in variables:
        actual_builds[variable.name] = variable.default_value ^ int(toggles.get(variable.name, 0))
    selected_count = sum(actual_builds.values())
    if selected_count > problem_config.max_new_lines:
        return infeasibility_penalty + 100_000.0 * selected_count
    result = solve_lindistflow_expansion(
        net,
        scenarios,
        candidates,
        problem_config,
        fixed_builds=actual_builds,
    )
    if result.objective_value == float("inf") or result.termination_status == "INFEASIBLE":
        return infeasibility_penalty + 100_000.0 * selected_count
    return result.objective_value


def _plan_key(toggle_decisions: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(toggle_decisions.items()))


def _status_name(status_code: int) -> str:
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return status_map.get(status_code, str(status_code))
