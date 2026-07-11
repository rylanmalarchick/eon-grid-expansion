from __future__ import annotations

from dataclasses import dataclass

import neal
import numpy as np

from eon.formulations.layer_b import LayerBSolution, LayerBSurrogate
from eon.validation import validate_symmetric_matrix


@dataclass(frozen=True, slots=True)
class QuboCompilation:
    qubo: dict[tuple[str, str], float]
    offset: float
    penalty_strength: float
    target_selected_count: int
    method: str


def compile_layer_b_qubo(
    surrogate: LayerBSurrogate,
    *,
    penalty_strength: float = 50_000.0,
    method: str = "hess_slack_free",
) -> QuboCompilation:
    labels = tuple(f"toggle[{i}]" for i in range(len(surrogate.variables)))
    label_by_name = {
        variable.name: labels[index] for index, variable in enumerate(surrogate.variables)
    }
    qubo: dict[tuple[str, str], float] = {}
    offset = float(surrogate.offset)

    for variable in surrogate.variables:
        _add_linear_term(qubo, label_by_name[variable.name], surrogate.linear[variable.name])
    for (left, right), coefficient in surrogate.quadratic.items():
        _add_quadratic_term(qubo, label_by_name[left], label_by_name[right], coefficient)

    actual_builds = {
        label_by_name[variable.name]: _actual_build_affine_coefficient(variable.default_value)
        for variable in surrogate.variables
    }
    fixed_outside = sum(
        value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    )
    remaining_budget = surrogate.max_new_lines - fixed_outside

    if method == "hess_slack_free" and remaining_budget <= 1:
        offset = _apply_hess_at_most_one_penalty(
            qubo,
            offset,
            actual_builds,
            fixed_outside=fixed_outside,
            remaining_budget=remaining_budget,
            penalty_strength=penalty_strength,
        )
    elif method == "hess_slack_free" and remaining_budget <= 5:
        # Hess-style at-most-K: dense quadratic cardinality penalty
        # Penalizes (sum_i actual_build_i - K)_+^2 via a shifted (sum - K)^2 penalty
        # that only activates when sum > K. This produces O(n^2) pairwise couplings
        # which is what makes the QUBO hard for MPS on non-1D graphs.
        offset = _apply_hess_at_most_k_penalty(
            qubo,
            offset,
            actual_builds,
            max_builds=remaining_budget,
            penalty_strength=penalty_strength,
        )
    else:
        target_count = max(0, min(remaining_budget, surrogate.base_selected_count - fixed_outside))
        offset = _apply_centered_count_penalty(
            qubo,
            offset,
            actual_builds,
            target_count=target_count,
            penalty_strength=penalty_strength,
        )
        if method == "hess_slack_free":
            method = "hess_slack_free_fallback"

    qubo = {key: value for key, value in qubo.items() if abs(value) > 1e-12}
    _validate_qubo_matrix(qubo, len(labels))
    return QuboCompilation(
        qubo=qubo,
        offset=offset,
        penalty_strength=penalty_strength,
        target_selected_count=surrogate.base_selected_count,
        method=method,
    )


def compile_layer_b_qubo_hess(
    surrogate: LayerBSurrogate,
    *,
    penalty_strength: float = 50_000.0,
) -> QuboCompilation:
    return compile_layer_b_qubo(
        surrogate,
        penalty_strength=penalty_strength,
        method="hess_slack_free",
    )


def compile_external_qubo(surrogate: LayerBSurrogate) -> QuboCompilation:
    """Compile WITHOUT any cardinality penalty. External / Path B instances carry
    no at-most-K constraint; the Hess penalty adds O(n^2) all-pairs couplings that
    densify the coupling graph (compiled tree-width ~ n-1) and would fake the
    tree-TN hardness signal."""
    labels = tuple(f"toggle[{i}]" for i in range(len(surrogate.variables)))
    label_by_name = {
        variable.name: labels[index] for index, variable in enumerate(surrogate.variables)
    }
    qubo: dict[tuple[str, str], float] = {}
    for variable in surrogate.variables:
        _add_linear_term(qubo, label_by_name[variable.name], surrogate.linear[variable.name])
    for (left, right), coefficient in surrogate.quadratic.items():
        _add_quadratic_term(qubo, label_by_name[left], label_by_name[right], coefficient)
    qubo = {key: value for key, value in qubo.items() if abs(value) > 1e-12}
    _validate_qubo_matrix(qubo, len(labels))
    return QuboCompilation(
        qubo=qubo,
        offset=float(surrogate.offset),
        penalty_strength=0.0,
        target_selected_count=surrogate.base_selected_count,
        method="penalty_free",
    )


def solve_qubo_with_neal(
    surrogate: LayerBSurrogate,
    compilation: QuboCompilation,
    *,
    num_reads: int = 256,
) -> LayerBSolution:
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(compilation.qubo, num_reads=num_reads)
    best_solution: LayerBSolution | None = None

    for sample, energy in sampleset.data(fields=["sample", "energy"]):
        toggle_decisions = {
            name: int(sample[f"toggle[{i}]"])
            for i, name in enumerate(surrogate.variable_names)
        }
        actual_builds = surrogate.actual_builds(toggle_decisions)
        if sum(actual_builds.values()) > surrogate.max_new_lines:
            continue
        selected_candidates = tuple(
            sorted(name for name, selected in actual_builds.items() if selected)
        )
        candidate_solution = LayerBSolution(
            objective_value=float(energy + compilation.offset),
            toggle_decisions=toggle_decisions,
            actual_builds=actual_builds,
            selected_candidates=selected_candidates,
            status="SAMPLED",
        )
        if (
            best_solution is None
            or candidate_solution.objective_value < best_solution.objective_value
        ):
            best_solution = candidate_solution

    if best_solution is None:
        zero_toggles = {name: 0 for name in surrogate.variable_names}
        actual_builds = surrogate.actual_builds(zero_toggles)
        best_solution = LayerBSolution(
            objective_value=surrogate.surrogate_objective(zero_toggles),
            toggle_decisions=zero_toggles,
            actual_builds=actual_builds,
            selected_candidates=tuple(
                sorted(name for name, selected in actual_builds.items() if selected)
            ),
            status="REPAIRED_ZERO",
        )
    return best_solution


def _actual_build_affine_coefficient(default_value: int) -> tuple[float, float]:
    constant = float(default_value)
    linear = float(1 - 2 * default_value)
    return constant, linear


def _apply_hess_at_most_one_penalty(
    qubo: dict[tuple[str, str], float],
    offset: float,
    actual_builds: dict[str, tuple[float, float]],
    *,
    fixed_outside: int,
    remaining_budget: int,
    penalty_strength: float,
) -> float:
    labels = tuple(actual_builds)
    if remaining_budget <= 0:
        for label in labels:
            constant, linear = actual_builds[label]
            offset += penalty_strength * constant
            _add_linear_term(qubo, label, penalty_strength * linear)
        return offset

    for index, left in enumerate(labels):
        left_constant, left_linear = actual_builds[left]
        for right in labels[index + 1 :]:
            right_constant, right_linear = actual_builds[right]
            offset += penalty_strength * left_constant * right_constant
            _add_linear_term(
                qubo,
                left,
                penalty_strength * right_constant * left_linear,
            )
            _add_linear_term(
                qubo,
                right,
                penalty_strength * left_constant * right_linear,
            )
            _add_quadratic_term(
                qubo,
                left,
                right,
                penalty_strength * left_linear * right_linear,
            )
    if fixed_outside >= 1:
        for label in labels:
            constant, linear = actual_builds[label]
            offset += penalty_strength * constant
            _add_linear_term(qubo, label, penalty_strength * linear)
    return offset


def _apply_hess_at_most_k_penalty(
    qubo: dict[tuple[str, str], float],
    offset: float,
    actual_builds: dict[str, tuple[float, float]],
    *,
    max_builds: int,
    penalty_strength: float,
) -> float:
    """At-most-K penalty via Hess-style dense quadratic cardinality encoding.

    Encodes penalty_strength * max(0, sum(actual_builds) - K)^2 as a QUBO.
    Uses (sum_i a_i - K)^2 = sum_i a_i^2 + 2*sum_{i<j} a_i*a_j - 2K*sum_i a_i + K^2
    where a_i = constant_i + linear_i * x_i is the affine build expression.

    This produces O(n^2) pairwise couplings which densifies the Ising coupling graph.
    """
    labels = tuple(actual_builds)
    k = float(max_builds)

    # Compute (sum_i a_i - K)^2 contribution
    # a_i = c_i + l_i * x_i where c_i is constant part, l_i is linear part
    # (sum a_i - K)^2 = (sum(c_i) - K + sum(l_i * x_i))^2
    # = (C + sum(l_i * x_i))^2 where C = sum(c_i) - K
    # = C^2 + 2C * sum(l_i * x_i) + (sum(l_i * x_i))^2

    sum_constants = sum(c for c, _ in actual_builds.values())
    big_c = sum_constants - k

    # Constant term: C^2
    offset += penalty_strength * big_c * big_c

    # Linear terms: 2C * l_i * x_i + l_i^2 * x_i  (since x_i^2 = x_i for binary)
    for label in labels:
        _, linear = actual_builds[label]
        coefficient = penalty_strength * (2.0 * big_c * linear + linear * linear)
        _add_linear_term(qubo, label, coefficient)

    # Quadratic terms: 2 * l_i * l_j * x_i * x_j for all i < j
    for index, left in enumerate(labels):
        _, left_linear = actual_builds[left]
        for right in labels[index + 1:]:
            _, right_linear = actual_builds[right]
            _add_quadratic_term(
                qubo,
                left,
                right,
                2.0 * penalty_strength * left_linear * right_linear,
            )
    return offset


def _apply_centered_count_penalty(
    qubo: dict[tuple[str, str], float],
    offset: float,
    actual_builds: dict[str, tuple[float, float]],
    *,
    target_count: int,
    penalty_strength: float,
) -> float:
    constant_term = -float(target_count)
    for constant, _ in actual_builds.values():
        constant_term += constant
    offset += penalty_strength * constant_term * constant_term

    labels = tuple(actual_builds)
    for label in labels:
        constant, linear = actual_builds[label]
        coefficient = penalty_strength * (2.0 * constant_term * linear + linear * linear)
        _add_linear_term(qubo, label, coefficient)

    for index, left in enumerate(labels):
        _, left_linear = actual_builds[left]
        for right in labels[index + 1 :]:
            _, right_linear = actual_builds[right]
            _add_quadratic_term(
                qubo,
                left,
                right,
                2.0 * penalty_strength * left_linear * right_linear,
            )
    return offset


def _add_linear_term(
    qubo: dict[tuple[str, str], float],
    label: str,
    coefficient: float,
) -> None:
    if abs(coefficient) <= 1e-12:
        return
    key = (label, label)
    qubo[key] = qubo.get(key, 0.0) + coefficient


def _add_quadratic_term(
    qubo: dict[tuple[str, str], float],
    left: str,
    right: str,
    coefficient: float,
) -> None:
    if abs(coefficient) <= 1e-12:
        return
    if left == right:
        _add_linear_term(qubo, left, coefficient)
        return
    key = (left, right) if left <= right else (right, left)
    qubo[key] = qubo.get(key, 0.0) + coefficient


def _validate_qubo_matrix(
    qubo: dict[tuple[str, str], float],
    variable_count: int,
) -> None:
    matrix = np.zeros((variable_count, variable_count), dtype=float)
    for (left, right), coefficient in qubo.items():
        left_index = _toggle_label_index(left)
        right_index = _toggle_label_index(right)
        matrix[left_index, right_index] += coefficient
        if left_index != right_index:
            matrix[right_index, left_index] += coefficient
    validate_symmetric_matrix(matrix, name="layer_b_qubo_matrix")


def _toggle_label_index(label: str) -> int:
    prefix = "toggle["
    if not label.startswith(prefix) or not label.endswith("]"):
        raise ValueError(f"Unexpected toggle label: {label}")
    return int(label[len(prefix) : -1])
