from __future__ import annotations

import networkx as nx

from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
from eon.instances.treewidth import (
    MPS_EASY_TREEWIDTH,
    coupling_diagnostics,
    estimate_treewidth,
    structural_coupling_graph,
)


def _variable(index: int) -> LayerBVariable:
    # candidate is unused by the tree-width path; None keeps the fixture minimal.
    return LayerBVariable(name=f"c{index}", candidate=None, default_value=0, stress_score=0.0)


def _surrogate(
    n: int,
    edges: list[tuple[int, int]],
    *,
    field: float = 0.0,
    edge_weight: float = 1.0,
) -> LayerBSurrogate:
    variables = tuple(_variable(i) for i in range(n))
    quadratic = {(f"c{a}", f"c{b}"): edge_weight for a, b in edges}
    return LayerBSurrogate(
        variables=variables,
        offset=0.0,
        linear={f"c{i}": field for i in range(n)},
        quadratic=quadratic,
        fixed_builds={},
        max_new_lines=2,
        base_objective=0.0,
        base_selected_count=0,
    )


def test_estimate_treewidth_known_graphs() -> None:
    # Empty graph (no couplings) is MPS-trivial: tree-width 0.
    assert estimate_treewidth(nx.empty_graph(5)) == 0
    # A path / tree has tree-width 1 -- the radial-feeder regime.
    assert estimate_treewidth(nx.path_graph(6)) == 1
    # A single cycle has tree-width 2.
    assert estimate_treewidth(nx.cycle_graph(5)) == 2
    # A complete graph K_k has tree-width k-1 -- the dense-coupling regime.
    assert estimate_treewidth(nx.complete_graph(4)) == 3
    assert estimate_treewidth(nx.complete_graph(6)) == 5


def test_structural_graph_edges_match_quadratic_terms() -> None:
    # 4-cycle: c0-c1-c2-c3-c0.
    surrogate = _surrogate(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    graph = structural_coupling_graph(surrogate)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert estimate_treewidth(graph) == 2


def test_coupling_diagnostics_path_is_mps_easy() -> None:
    # A path-structured surrogate is the low-tree-width / MPS-easy regime.
    surrogate = _surrogate(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    diag = coupling_diagnostics(surrogate)
    assert diag.variable_count == 5
    assert diag.structural_treewidth == 1
    assert diag.structural_components == 1
    assert diag.mps_easy is True


def test_coupling_diagnostics_complete_is_not_mps_easy() -> None:
    # K_n has tree-width n-1. It is genuinely not MPS-easy only when the tree-width
    # exceeds the chi-budget threshold log2(chi_limit)=MPS_EASY_TREEWIDTH, i.e. exact
    # TN contraction needs bond dimension 2**tw > chi_limit. K6 (tw 5 -> chi 32 <= 64)
    # is in fact easy; size up so the dense regime really exceeds the budget.
    n = MPS_EASY_TREEWIDTH + 2  # tree-width n-1 = MPS_EASY_TREEWIDTH + 1 > threshold
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    surrogate = _surrogate(n, edges)
    diag = coupling_diagnostics(surrogate)
    assert diag.structural_treewidth == n - 1
    assert diag.structural_density == 1.0
    assert diag.effective_treewidth > MPS_EASY_TREEWIDTH
    assert diag.mps_easy is False


def _complete_edges(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def test_weak_couplings_are_not_load_bearing() -> None:
    # K6 is topologically tree-width 5, but the couplings are tiny next to the
    # linear fields, so effectively MPS-easy (the n=20 zoo failure mode).
    surrogate = _surrogate(6, _complete_edges(6), field=10.0, edge_weight=0.01)
    diag = coupling_diagnostics(surrogate)
    assert diag.structural_treewidth == 5
    assert diag.effective_treewidth == 0
    assert diag.effective_edges == 0
    assert diag.coupling_field_ratio == 0.001
    assert diag.mps_easy is True


def test_strong_couplings_are_load_bearing() -> None:
    # Strong couplings (J >> h) on a dense graph stay load-bearing; at a size whose
    # tree-width exceeds the chi-budget threshold the instance is genuinely not
    # MPS-easy (tree-width n-1 > MPS_EASY_TREEWIDTH => bond dim 2**(n-1) > chi_limit).
    n = MPS_EASY_TREEWIDTH + 2
    surrogate = _surrogate(n, _complete_edges(n), field=1.0, edge_weight=50.0)
    diag = coupling_diagnostics(surrogate)
    assert diag.effective_treewidth == n - 1
    assert diag.coupling_field_ratio == 50.0
    assert diag.mps_easy is False


def test_disconnected_structural_graph_components() -> None:
    # Two disjoint edges -> 2 components, plus isolated nodes counted.
    surrogate = _surrogate(5, [(0, 1), (2, 3)])
    diag = coupling_diagnostics(surrogate)
    assert diag.structural_components == 3  # {0,1}, {2,3}, {4}
    assert diag.structural_treewidth == 1
