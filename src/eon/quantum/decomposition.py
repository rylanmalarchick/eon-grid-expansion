from __future__ import annotations

from dataclasses import replace

import networkx as nx

from eon.formulations.layer_b import LayerBSolution, LayerBSurrogate
from eon.quantum.cop_qaoa import run_constrained_qaoa_subproblem


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


def solve_decomposed_qaoa(
    surrogate: LayerBSurrogate,
    *,
    block_size: int = 5,
    p: int = 1,
    shots: int = 1024,
) -> LayerBSolution:
    combined_builds = {
        name: value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    }
    for subproblem in decompose_surrogate(surrogate, block_size=block_size):
        run = run_constrained_qaoa_subproblem(subproblem, p=p, shots=shots)
        combined_builds.update(run.best_sample.actual_builds)

    if sum(combined_builds.values()) > surrogate.max_new_lines:
        ranked = sorted(
            combined_builds.items(),
            key=lambda item: surrogate.linear.get(item[0], 0.0),
        )
        keep = {name for name, _ in ranked[: surrogate.max_new_lines]}
        combined_builds = {name: int(name in keep) for name in combined_builds}

    selected_candidates = tuple(
        sorted(name for name, selected in combined_builds.items() if selected)
    )
    return LayerBSolution(
        objective_value=0.0,
        toggle_decisions={},
        actual_builds=combined_builds,
        selected_candidates=selected_candidates,
        status="DECOMPOSED_QAOA",
    )
