from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.qubo import QuboCompilation

# Couplings below this magnitude are treated as absent edges.
_COUPLING_EPS = 1e-9

# The MPS protocol sweeps bond dimension up to chi_limit (eon.mps.protocol
# run_mps_protocol default chi_limit=64). An Ising/QUBO whose load-bearing
# coupling graph has effective tree-width w is EXACTLY representable by a tensor
# network of bond dimension 2**w (TN contraction cost ~ exp(tree-width);
# reading.txt R30/R18), so it is provably contractible within the sweep's budget
# whenever 2**w <= chi_limit, i.e. w <= log2(chi_limit). With chi_limit=64 the
# threshold is 6.  Keep _MPS_CHI_LIMIT in sync with protocol.run_mps_protocol.
_MPS_CHI_LIMIT = 64
MPS_EASY_TREEWIDTH = _MPS_CHI_LIMIT.bit_length() - 1  # = log2(64) = 6


@dataclass(frozen=True, slots=True)
class CouplingDiagnostics:
    variable_count: int
    # Structural = ALL physics-derived quadratic couplings (surrogate.quadratic),
    # unweighted. NOTE: this is confounded when surrogate.quadratic is a
    # fixed-size SAMPLE of pairs -- tree-width then reflects the sample cap, not
    # physics (the n=20 zoo showed tw~11 at uniform density yet was MPS-easy).
    structural_treewidth: int
    structural_edges: int
    structural_density: float
    structural_max_degree: int
    structural_components: int
    # Magnitude-aware view: only couplings strong enough to compete with the
    # linear fields are "load-bearing". This is the hardness-relevant signal.
    linear_scale: float  # median |h_i|
    coupling_scale: float  # median |J_ij|
    coupling_field_ratio: float | None  # coupling_scale / linear_scale
    effective_treewidth: int  # tree-width of the load-bearing coupling graph
    effective_edges: int
    # Compiled = the QUBO the MPS backend actually sees, including the cardinality
    # penalty. The at-most-K penalty couples all pairs, so this is ~complete
    # (tree-width ~ n-1) for every instance: an encoding artifact, not problem
    # hardness. Reported only to make that point explicit.
    compiled_treewidth: int | None
    compiled_edges: int | None

    @property
    def mps_easy(self) -> bool:
        # Easy if the LOAD-BEARING coupling is low tree-width. Topological
        # tree-width alone is confounded by weak / sampled couplings, so the
        # effective (magnitude-aware) tree-width is the criterion.
        return self.effective_treewidth <= MPS_EASY_TREEWIDTH


def structural_coupling_graph(
    surrogate: LayerBSurrogate, *, eps: float = _COUPLING_EPS
) -> nx.Graph:
    index = {name: position for position, name in enumerate(surrogate.variable_names)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(surrogate.variables)))
    for (left, right), coefficient in surrogate.quadratic.items():
        if left == right or abs(coefficient) <= eps:
            continue
        graph.add_edge(index[left], index[right])
    return graph


def effective_coupling_graph(
    surrogate: LayerBSurrogate,
    *,
    relative_threshold: float = 1.0,
    eps: float = _COUPLING_EPS,
) -> nx.Graph:
    """Coupling graph keeping only LOAD-BEARING edges: |J_ij| at least
    `relative_threshold` times the median field |h|. When there are no fields
    (all |h| ~ 0) every nonzero coupling is load-bearing."""
    index = {name: position for position, name in enumerate(surrogate.variable_names)}
    field_scale = _median_abs(surrogate.linear.values())
    threshold = max(relative_threshold * field_scale, eps)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(surrogate.variables)))
    for (left, right), coefficient in surrogate.quadratic.items():
        if left == right or abs(coefficient) < threshold:
            continue
        graph.add_edge(index[left], index[right])
    return graph


def qubo_coupling_graph(
    compilation: QuboCompilation, *, eps: float = _COUPLING_EPS
) -> nx.Graph:
    graph = nx.Graph()
    for (left, right), coefficient in compilation.qubo.items():
        if left == right or abs(coefficient) <= eps:
            continue
        graph.add_edge(_toggle_index(left), _toggle_index(right))
    return graph


def estimate_treewidth(graph: nx.Graph) -> int:
    if graph.number_of_edges() == 0:
        return 0
    width, _ = nx.algorithms.approximation.treewidth_min_fill_in(graph)
    return int(width)


def coupling_diagnostics(
    surrogate: LayerBSurrogate,
    compilation: QuboCompilation | None = None,
) -> CouplingDiagnostics:
    graph = structural_coupling_graph(surrogate)
    variable_count = len(surrogate.variables)
    edges = graph.number_of_edges()
    max_pairs = variable_count * (variable_count - 1) // 2
    density = edges / max_pairs if max_pairs else 0.0
    max_degree = max((degree for _, degree in graph.degree()), default=0)
    components = nx.number_connected_components(graph) if variable_count else 0

    field_scale = _median_abs(surrogate.linear.values())
    coupling_scale = _median_abs(surrogate.quadratic.values())
    coupling_field_ratio = coupling_scale / field_scale if field_scale > 0.0 else None
    effective_graph = effective_coupling_graph(surrogate)

    compiled_treewidth: int | None = None
    compiled_edges: int | None = None
    if compilation is not None:
        compiled_graph = qubo_coupling_graph(compilation)
        compiled_treewidth = estimate_treewidth(compiled_graph)
        compiled_edges = compiled_graph.number_of_edges()

    return CouplingDiagnostics(
        variable_count=variable_count,
        structural_treewidth=estimate_treewidth(graph),
        structural_edges=edges,
        structural_density=density,
        structural_max_degree=max_degree,
        structural_components=components,
        linear_scale=field_scale,
        coupling_scale=coupling_scale,
        coupling_field_ratio=coupling_field_ratio,
        effective_treewidth=estimate_treewidth(effective_graph),
        effective_edges=effective_graph.number_of_edges(),
        compiled_treewidth=compiled_treewidth,
        compiled_edges=compiled_edges,
    )


def _median_abs(values: Iterable[float]) -> float:
    magnitudes = sorted(abs(float(value)) for value in values)
    if not magnitudes:
        return 0.0
    middle = len(magnitudes) // 2
    if len(magnitudes) % 2 == 1:
        return magnitudes[middle]
    return 0.5 * (magnitudes[middle - 1] + magnitudes[middle])


def _toggle_index(label: str) -> int:
    prefix = "toggle["
    if not label.startswith(prefix) or not label.endswith("]"):
        raise ValueError(f"Unexpected toggle label: {label}")
    return int(label[len(prefix) : -1])
