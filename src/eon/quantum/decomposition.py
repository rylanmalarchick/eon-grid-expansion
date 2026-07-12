"""Topology-partition decomposition wrapper with the D14 thin certificate.

The wrapper splits the surrogate's coupling graph into <=block_size blocks,
runs the constrained QAOA per block, merges, greedily repairs cardinality,
and rescores the merged plan on the FULL surrogate objective (the upper
bound). The certificate's lower bound is classical optimization duality only
(PLAN.txt D14): the dropped-coupling block bound -- each block minimized
exactly (cardinality relaxed), each dropped inter-block coupling J
contributing min(0, J) -- optionally tightened by a Gurobi best_bound the
caller provides. The certified gap is the DECOMPOSITION/INTEGRALITY gap of
the classical wrapper, NOT classical-vs-quantum advantage; tightening it is
a classical workstream.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import networkx as nx

from eon.formulations.layer_b import LayerBSolution, LayerBSurrogate
from eon.quantum.cop_qaoa import run_constrained_qaoa_subproblem

_CERTIFICATE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class DecompositionCertificate:
    lower_bound: float
    upper_bound: float
    gap: float
    lower_bound_source: str
    block_sizes: tuple[int, ...]
    dropped_coupling_count: int
    dropped_coupling_bound: float
    gurobi_best_bound: float | None


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
    block minima. Returns (bound, dropped_coupling_count)."""
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
    gurobi_best_bound: float | None = None,
) -> tuple[LayerBSolution, DecompositionCertificate]:
    """Per-block constrained QAOA -> merge -> greedy cardinality repair ->
    rescore on the FULL surrogate objective (upper bound) + D14 certificate."""
    blocks = decompose_surrogate(surrogate, block_size=block_size)
    combined_builds = {
        name: value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    }
    for subproblem in blocks:
        run = run_constrained_qaoa_subproblem(subproblem, p=p, shots=shots)
        combined_builds.update(run.best_sample.actual_builds)

    if sum(combined_builds.values()) > surrogate.max_new_lines:
        ranked = sorted(
            combined_builds.items(),
            key=lambda item: surrogate.linear.get(item[0], 0.0),
        )
        keep = {name for name, _ in ranked[: surrogate.max_new_lines]}
        combined_builds = {name: int(name in keep) for name in combined_builds}

    toggle_decisions = {
        variable.name: variable.default_value ^ combined_builds.get(variable.name, 0)
        for variable in surrogate.variables
    }
    upper_bound = surrogate.surrogate_objective(toggle_decisions)

    block_bound, dropped_count = dropped_coupling_lower_bound(surrogate, blocks)
    if gurobi_best_bound is not None and gurobi_best_bound > block_bound:
        lower_bound = float(gurobi_best_bound)
        source = "gurobi_best_bound"
    else:
        lower_bound = block_bound
        source = "dropped_coupling_block_bound"
    if lower_bound > upper_bound + _CERTIFICATE_TOLERANCE:
        raise RuntimeError(
            f"certificate violation: lower bound {lower_bound} exceeds upper "
            f"bound {upper_bound} -- a bug in the bound or the rescore, not a result"
        )

    certificate = DecompositionCertificate(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap=upper_bound - lower_bound,
        lower_bound_source=source,
        block_sizes=tuple(len(block.variables) for block in blocks),
        dropped_coupling_count=dropped_count,
        dropped_coupling_bound=block_bound,
        gurobi_best_bound=gurobi_best_bound,
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
        best_bound=lower_bound,
    )
    return solution, certificate
