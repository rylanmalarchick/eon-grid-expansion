"""How far above a PROVEN CNOT floor does each ansatz sit?

Cao et al. (R53, arXiv:2509.10070) prove that a graphic parity network for a
connected graph with n vertices and m edges needs at least m+n-1 CNOT gates. A
QAOA cost layer exp(-i gamma sum J_ij Z_i Z_j) is exactly such a network over
the coupling graph, so the bound applies once per layer.

This turns S4's claim -- that the constrained ansatz's cost is an
implementation artifact rather than a fundamental bound -- from a citation into
a measurement.

SCOPE, and it matters: the floor covers the COST LAYERS ONLY. It says nothing
about the mixer or the state preparation, so the whole circuit's true floor is
higher than the number computed here, and every ratio below is an UPPER BOUND on
how far from optimal the circuit is. The real gap is smaller.

The bound also requires a CONNECTED coupling graph; this refuses to report a
ratio if the graph is disconnected rather than quoting an inapplicable bound.

Run from workspace/:
    python scripts/gate_count_floor.py --table experiments/results/s4_transpile/table.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx

from eon.formulations.qubo import compile_external_qubo
from eon.instances.external import build_external_surrogate, generate_fused_planted

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gate_count_floor")

# The standard decomposition of RZZ(theta) is CNOT, RZ(theta), CNOT.
CNOTS_PER_RZZ = 2


def _coupling_graph(qubits: int, seed: int, block_size: int, alpha: float) -> nx.Graph:
    surrogate = build_external_surrogate(
        generate_fused_planted(qubits, seed=seed, block_size=block_size, alpha=alpha)
    )
    compilation = compile_external_qubo(surrogate)
    index = lambda label: int(label.split("[")[1].rstrip("]"))  # noqa: E731
    graph = nx.Graph()
    graph.add_nodes_from(range(len(surrogate.variables)))
    graph.add_edges_from(
        (index(left), index(right))
        for left, right in compilation.qubo
        if left != right
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="experiments/results/s4_transpile/table.json")
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    table = json.loads(Path(args.table).read_text())
    graph = _coupling_graph(args.qubits, args.seed, args.block_size, args.alpha)
    n, m = graph.number_of_nodes(), graph.number_of_edges()

    if not nx.is_connected(graph):
        logger.error(
            "coupling graph is NOT connected; the m+n-1 bound assumes connectivity "
            "and does not apply. Refusing to report a ratio."
        )
        sys.exit(1)
    if m != table.get("coupling_count"):
        logger.error(
            "rebuilt graph has %d edges but the table records %d -- the instance "
            "or the compilation path does not match, so the floor would describe a "
            "different circuit",
            m, table.get("coupling_count"),
        )
        sys.exit(1)

    floor_per_layer = m + n - 1
    logger.info("coupling graph n=%d m=%d connected; floor %d CNOT per cost layer",
                n, m, floor_per_layer)

    rows = []
    for row in table["rows"]:
        depth = int(row["qaoa_depth_p"])
        rzz = int(row["logical"]["ops"].get("rzz", 0))
        floor = floor_per_layer * depth
        logical = CNOTS_PER_RZZ * rzz
        transpiled = int(row["transpiled"]["1"]["two_qubit_gates"])
        rows.append({
            "algorithm": row["algorithm"],
            "qaoa_depth_p": depth,
            "cost_layer_cnot_floor": floor,
            "logical_cnot_equivalent": logical,
            "logical_over_floor": logical / floor,
            "transpiled_two_qubit": transpiled,
            "transpiled_over_floor": transpiled / floor,
        })
        logger.info(
            "  %-8s floor %d | logical %d (%.2fx) | transpiled %d (%.1fx)",
            row["algorithm"], floor, logical, logical / floor,
            transpiled, transpiled / floor,
        )

    record = {
        "instance_id": table["instance_id"],
        "coupling_graph": {"vertices": n, "edges": m, "connected": True},
        "floor_per_cost_layer": floor_per_layer,
        "bound": (
            "Cao et al. 2025 (arXiv:2509.10070): m+n-1 CNOT for a connected "
            "graphic parity network"
        ),
        "scope": (
            "covers the cost layers only; the mixer and state preparation are not "
            "bounded, so the whole-circuit floor is higher and these ratios are "
            "UPPER bounds on the distance from optimal"
        ),
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
