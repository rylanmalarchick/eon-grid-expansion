"""Topology-partition decomposition wrapper with the D14 thin certificate.

The wrapper splits the surrogate's coupling graph into <=block_size blocks,
runs the constrained QAOA per block, merges, greedily repairs cardinality,
and rescores the merged plan on the FULL surrogate objective (the upper
bound). The certificate's lower bound is classical optimization duality only
(PLAN.txt D14): the dropped-coupling block bound -- each block minimized
exactly (cardinality relaxed), each dropped inter-block coupling J
contributing min(0, J) -- optionally tightened by a Gurobi dual bound the
caller provides. The certified gap is the DECOMPOSITION/INTEGRALITY gap of
the classical wrapper, NOT classical-vs-quantum advantage; tightening it is
a classical workstream.

The bound kinds are separate types (eon.quantum.bounds), so an incumbent
objective cannot be handed to the dual-bound slot: numerically it would look
like a perfectly ordinary bound and would silently make the certificate look
tight.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import networkx as nx

from eon.formulations.layer_b import LayerBSolution, LayerBSurrogate
from eon.quantum.bounds import (
    CertifiedLowerBound,
    ClassicalDecompositionGap,
    HeuristicIncumbent,
)
from eon.quantum.cop_qaoa import run_constrained_qaoa_subproblem


@dataclass(frozen=True, slots=True)
class DecompositionCertificate:
    # The typed pair is the certificate; the flat fields below are kept for the
    # JSON records that already reference them.
    gap_object: ClassicalDecompositionGap
    lower_bound: float
    upper_bound: float
    gap: float
    lower_bound_source: str
    block_sizes: tuple[int, ...]
    dropped_coupling_count: int
    dropped_coupling_bound: float
    gurobi_dual_bound: float | None
    # Transparency: the greedy cardinality repair can override block-QAOA
    # outputs; these fields say whether it fired and how many variable builds
    # it changed, so "reaggregated from per-block QAOA" is never overstated.
    repair_applied: bool = False
    builds_changed_by_repair: int = 0


def decompose_surrogate(
    surrogate: LayerBSurrogate,
    *,
    block_size: int = 5,
) -> list[LayerBSurrogate]:
    graph = nx.Graph()
    for variable in surrogate.variables:
        graph.add_node(variable.name)
    for left, right in surrogate.quadratic:
        graph.add_edge(left, right)

    blocks: list[list[str]] = []
    for component in nx.connected_components(graph):
        ordered = sorted(component)
        for start in range(0, len(ordered), block_size):
            blocks.append(ordered[start : start + block_size])

    subproblems: list[LayerBSurrogate] = []
    for block in blocks:
        variables = tuple(variable for variable in surrogate.variables if variable.name in block)
        quadratic = {
            (left, right): value
            for (left, right), value in surrogate.quadratic.items()
            if left in block and right in block
        }
        linear = {name: surrogate.linear[name] for name in block}
        subproblems.append(
            replace(
                surrogate,
                variables=variables,
                linear=linear,
                quadratic=quadratic,
            )
        )
    return subproblems


def _block_minimum_toggle_space(block: LayerBSurrogate) -> float:
    """Exact minimum of the block's linear+quadratic toggle objective, WITHOUT
    the shared offset and WITHOUT the cardinality constraint (relaxing only
    lowers the minimum, keeping the bound valid). Blocks are <= block_size
    variables, so exhaustive enumeration is exact and cheap."""
    names = [variable.name for variable in block.variables]
    best = float("inf")
    for toggles in product((0, 1), repeat=len(names)):
        assignment = dict(zip(names, toggles, strict=True))
        value = sum(block.linear[name] * assignment[name] for name in names)
        value += sum(
            coefficient * assignment[left] * assignment[right]
            for (left, right), coefficient in block.quadratic.items()
        )
        best = min(best, value)
    return best


def dropped_coupling_lower_bound(
    surrogate: LayerBSurrogate,
    blocks: list[LayerBSurrogate],
) -> tuple[float, int]:
    """LB = offset + sum_b exact block minimum + sum over dropped couplings of
    min(0, J). Valid because toggle variables are binary (each dropped term is
    at least min(0, J)) and dropping the cardinality constraint only lowers
    block minima. Returns (bound, dropped_coupling_count).

    MACHINE-CHECKED: lean/DroppedCouplingBound.lean proves this bound never
    exceeds the energy of any assignment (over Int; the argument is
    ordered-ring generic). tests/test_dropped_coupling_bound.py brute-forces
    the same statement against the true minimum on small instances, with a
    negative control showing the min(0, J) floor is load-bearing.
    """
    in_block: set[tuple[str, str]] = set()
    for block in blocks:
        in_block.update(block.quadratic.keys())
    dropped = [
        coefficient
        for key, coefficient in surrogate.quadratic.items()
        if key not in in_block
    ]
    bound = float(surrogate.offset)
    bound += sum(_block_minimum_toggle_space(block) for block in blocks)
    bound += sum(min(0.0, coefficient) for coefficient in dropped)
    return bound, len(dropped)


def solve_decomposed_qaoa(
    surrogate: LayerBSurrogate,
    *,
    block_size: int = 5,
    p: int = 1,
    shots: int = 1024,
    gurobi_dual_bound: CertifiedLowerBound | None = None,
) -> tuple[LayerBSolution, DecompositionCertificate]:
    """Per-block constrained QAOA -> merge -> greedy cardinality repair ->
    rescore on the FULL surrogate objective (upper bound) + D14 certificate.

    `gurobi_dual_bound` is a CertifiedLowerBound rather than a float on
    purpose: a solver's incumbent objective is numerically indistinguishable
    from its dual bound, and passing the wrong one produces a certificate that
    looks tight and proves nothing.
    """
    # Checked on arrival, not where it is consumed: a wrong-typed bound that
    # loses the max against the block bound would otherwise slip through
    # unexamined and only fail on some later instance where it wins.
    if gurobi_dual_bound is not None and not isinstance(
        gurobi_dual_bound, CertifiedLowerBound
    ):
        raise TypeError(
            "gurobi_dual_bound must be a CertifiedLowerBound, got "
            f"{type(gurobi_dual_bound).__name__}; only a proven bound may "
            "certify the decomposition gap"
        )
    blocks = decompose_surrogate(surrogate, block_size=block_size)
    outside_fixed = {
        name: value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    }
    variable_builds: dict[str, int] = {}
    for subproblem in blocks:
        run = run_constrained_qaoa_subproblem(subproblem, p=p, shots=shots)
        variable_builds.update(run.best_sample.actual_builds)

    # Greedy cardinality repair over VARIABLE builds only -- Layer-A-fixed
    # lines outside the surrogate are never touched (unbuilding one is not a
    # legal move, and the surrogate coefficients are conditional on them).
    # Ranking is in BUILD space: the objective delta of build=1 vs build=0 is
    # +linear for default 0 (toggle 0->1) and -linear for default 1 (toggle
    # 1->0); keep the most beneficial builds within the remaining budget.
    defaults = {variable.name: variable.default_value for variable in surrogate.variables}
    variable_budget = surrogate.max_new_lines - sum(outside_fixed.values())
    repair_applied = False
    builds_changed = 0
    if sum(variable_builds.values()) > variable_budget:
        repair_applied = True

        def build_delta(name: str) -> float:
            coefficient = surrogate.linear.get(name, 0.0)
            return coefficient if defaults.get(name, 0) == 0 else -coefficient

        built = sorted(
            (name for name, value in variable_builds.items() if value),
            key=build_delta,
        )
        keep = set(built[: max(0, variable_budget)])
        repaired = {name: int(name in keep) for name in variable_builds}
        builds_changed = sum(
            1 for name in variable_builds if repaired[name] != variable_builds[name]
        )
        variable_builds = repaired

    combined_builds = {**outside_fixed, **variable_builds}
    toggle_decisions = {
        variable.name: variable.default_value ^ combined_builds.get(variable.name, 0)
        for variable in surrogate.variables
    }
    upper_bound = surrogate.surrogate_objective(toggle_decisions)

    block_bound, dropped_count = dropped_coupling_lower_bound(surrogate, blocks)
    if gurobi_dual_bound is not None and gurobi_dual_bound.value > block_bound:
        certified = gurobi_dual_bound
    else:
        certified = CertifiedLowerBound(
            value=block_bound, source="dropped_coupling_block_bound"
        )
    # Construction is the check: mismatched kinds raise TypeError, and a lower
    # bound above the incumbent raises ValueError.
    gap_object = ClassicalDecompositionGap(
        lower=certified,
        upper=HeuristicIncumbent(
            value=upper_bound, source="rescored_decomposed_plan"
        ),
    )

    certificate = DecompositionCertificate(
        gap_object=gap_object,
        lower_bound=certified.value,
        upper_bound=upper_bound,
        gap=gap_object.value,
        lower_bound_source=certified.source,
        block_sizes=tuple(len(block.variables) for block in blocks),
        dropped_coupling_count=dropped_count,
        dropped_coupling_bound=block_bound,
        gurobi_dual_bound=(
            gurobi_dual_bound.value if gurobi_dual_bound is not None else None
        ),
        repair_applied=repair_applied,
        builds_changed_by_repair=builds_changed,
    )
    selected_candidates = tuple(
        sorted(name for name, selected in combined_builds.items() if selected)
    )
    solution = LayerBSolution(
        objective_value=upper_bound,
        toggle_decisions=toggle_decisions,
        actual_builds=combined_builds,
        selected_candidates=selected_candidates,
        status="DECOMPOSED_QAOA",
        best_bound=certified.value,
    )
    return solution, certificate
