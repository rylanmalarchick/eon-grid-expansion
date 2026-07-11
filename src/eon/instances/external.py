"""External / synthetic QUBO instances for the Path B hardness spine.

Three generators (PLAN.txt section 6, refs R33/R34 in reading.txt):

- generate_posiform_planted: Hahn/Pelofske/Djidjev posiform planting
  (arXiv:2308.05859). Produces a QUBO with a UNIQUE planted optimum. On its
  own it is classically EASY (2-SAT-derived, solvable in linear time) -- it
  supplies ground truth, not hardness.
- generate_fused_planted: the arXiv:2411.03626 hardening -- disjoint random
  block QUBOs solved exactly, their minimizers concatenated into the planted
  bitstring, then Q = sum_i R_i + alpha * P with P posiform-planted on that
  bitstring. The unique optimum survives (R_i are block-minimal at the planted
  point; alpha * P breaks ties strictly); smaller alpha is harder.
- generate_longrange_spin_glass: random long-range +-J couplings, no planted
  optimum (heuristic-hardness workhorse for the tree-TN-negativity test).

All generators are deterministic in (n, seed). Instances carry no cardinality
constraint: build_external_surrogate sets max_new_lines = n so the Layer B
Gurobi solve is unconstrained, and the TN controls must be run penalty-free
(run_mps_protocol / run_tree_tn_control with penalty_free=True) -- the Hess
cardinality penalty would densify the coupling graph and fake the hardness
signal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product

from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
from eon.instances.candidate_lines import CandidateLine

# Literal encoding for the 2-SAT machinery: literal 2*i is x_i, 2*i+1 is NOT x_i.


@dataclass(frozen=True, slots=True)
class ExternalQuboInstance:
    """A QUBO over binary variables v000..v{n-1}, objective offset + h.x + x.J.x."""

    name: str
    variable_names: tuple[str, ...]
    offset: float
    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    planted_solution: dict[str, int] | None = None
    planted_energy: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def energy(self, assignment: dict[str, int]) -> float:
        total = self.offset
        for name, coefficient in self.linear.items():
            total += coefficient * assignment[name]
        for (left, right), coefficient in self.quadratic.items():
            total += coefficient * assignment[left] * assignment[right]
        return total


def _variable_names(n: int) -> tuple[str, ...]:
    return tuple(f"v{i:03d}" for i in range(n))


def build_external_surrogate(instance: ExternalQuboInstance) -> LayerBSurrogate:
    """Wrap an external QUBO as a LayerBSurrogate (stub candidates, no cardinality
    constraint: max_new_lines = n makes the Layer B at-most-K vacuous)."""
    candidates = [
        CandidateLine(
            name=name,
            from_bus=index,
            to_bus=(index + 1) % max(2, len(instance.variable_names)),
            family="external",
            length_km=1.0,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.3,
            max_i_ka=0.4,
            build_cost=0.0,
        )
        for index, name in enumerate(instance.variable_names)
    ]
    variables = tuple(
        LayerBVariable(name=c.name, candidate=c, default_value=0, stress_score=1.0)
        for c in candidates
    )
    planted_count = (
        sum(instance.planted_solution.values())
        if instance.planted_solution is not None
        else len(variables) // 2
    )
    return LayerBSurrogate(
        variables=variables,
        offset=instance.offset,
        linear={name: instance.linear.get(name, 0.0) for name in instance.variable_names},
        quadratic=dict(instance.quadratic),
        fixed_builds={},
        max_new_lines=len(variables),
        base_objective=instance.planted_energy if instance.planted_energy is not None else 0.0,
        base_selected_count=max(1, planted_count),
    )


# --------------------------------------------------------------------------
# 2-SAT via implication graph + SCC (Aspvall-Plass-Tarjan), iterative Kosaraju.
# --------------------------------------------------------------------------


def _two_sat_satisfiable(n: int, clauses: list[tuple[int, int]]) -> bool:
    """clauses are pairs of literals (2*i = x_i, 2*i+1 = NOT x_i); (a OR b)."""
    node_count = 2 * n
    forward: list[list[int]] = [[] for _ in range(node_count)]
    backward: list[list[int]] = [[] for _ in range(node_count)]
    for a, b in clauses:
        # (a OR b) => (NOT a -> b), (NOT b -> a)
        forward[a ^ 1].append(b)
        forward[b ^ 1].append(a)
        backward[b].append(a ^ 1)
        backward[a].append(b ^ 1)

    order: list[int] = []
    visited = [False] * node_count
    for start in range(node_count):
        if visited[start]:
            continue
        stack: list[tuple[int, int]] = [(start, 0)]
        visited[start] = True
        while stack:
            node, edge_index = stack[-1]
            if edge_index < len(forward[node]):
                stack[-1] = (node, edge_index + 1)
                nxt = forward[node][edge_index]
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, 0))
            else:
                order.append(node)
                stack.pop()

    component = [-1] * node_count
    label = 0
    for start in reversed(order):
        if component[start] != -1:
            continue
        component[start] = label
        stack2 = [start]
        while stack2:
            node = stack2.pop()
            for nxt in backward[node]:
                if component[nxt] == -1:
                    component[nxt] = label
                    stack2.append(nxt)
        label += 1

    return all(component[2 * i] != component[2 * i + 1] for i in range(n))


def _two_sat_unique(n: int, clauses: list[tuple[int, int]], planted: list[int]) -> bool:
    """The formula has planted as its UNIQUE solution iff it is satisfiable and,
    for every k, forcing x_k to the flipped value makes it unsatisfiable."""
    if not _two_sat_satisfiable(n, clauses):
        return False
    for k in range(n):
        flipped_literal = 2 * k + (1 if planted[k] == 1 else 0)  # literal asserting NOT planted_k
        if _two_sat_satisfiable(n, [*clauses, (flipped_literal, flipped_literal)]):
            return False
    return True


# --------------------------------------------------------------------------
# Posiform planting (R33, arXiv:2308.05859)
# --------------------------------------------------------------------------


def _plant_two_sat(
    n: int,
    planted: list[int],
    rng: random.Random,
    edge_candidates: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Add exclusion clauses over random pairs until planted is the unique
    2-SAT solution. Bounded: the 2-SAT phase transition is at O(n) clauses,
    so 80n + 1000 attempts is far beyond exhaustion; hitting it means the
    edge set is too sparse to isolate the planted bitstring, and we raise."""
    clauses: list[tuple[int, int]] = []
    max_attempts = 80 * n + 1000
    initial_batch = 3 * n
    for attempt in range(max_attempts):
        if edge_candidates is None:
            i, j = rng.sample(range(n), 2)
        else:
            i, j = edge_candidates[rng.randrange(len(edge_candidates))]
        wrong_tuples = [t for t in product((0, 1), repeat=2) if t != (planted[i], planted[j])]
        xi_hat, xj_hat = wrong_tuples[rng.randrange(3)]
        # Exclude (x_i, x_j) == (xi_hat, xj_hat): clause (l_i OR l_j) with
        # l_i = x_i if xi_hat == 0 else NOT x_i (eq. 3 of arXiv:2308.05859).
        literal_i = 2 * i + (1 if xi_hat == 1 else 0)
        literal_j = 2 * j + (1 if xj_hat == 1 else 0)
        clauses.append((literal_i, literal_j))
        if attempt >= initial_batch and _two_sat_unique(n, clauses, planted):
            return clauses
    raise RuntimeError(
        f"posiform planting: no unique 2-SAT solution after {max_attempts} clauses "
        f"(n={n}); the edge set is too sparse for this planted bitstring"
    )


def _posiform_to_qubo(
    n: int,
    clauses: list[tuple[int, int]],
    coefficients: list[float],
) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
    """Each clause (z OR z') becomes the posiform term b * NOT(z) * NOT(z'), which
    is 0 iff the clause is satisfied. Expanding complements (x-bar = 1 - x) yields
    QUBO terms; the resulting objective is >= 0 with value 0 exactly on 2-SAT
    solutions (the machine-checked planting lemma, lean/PosiformPlanting.lean)."""
    offset = 0.0
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    def _add_linear(index: int, value: float) -> None:
        linear[index] = linear.get(index, 0.0) + value

    def _add_quadratic(i: int, j: int, value: float) -> None:
        key = (i, j) if i < j else (j, i)
        quadratic[key] = quadratic.get(key, 0.0) + value

    for (lit_a, lit_b), b in zip(clauses, coefficients, strict=True):
        i, i_negated = lit_a // 2, bool(lit_a & 1)
        j, j_negated = lit_b // 2, bool(lit_b & 1)
        # Posiform term: b * (NOT of lit_a) * (NOT of lit_b). "NOT of literal x_i"
        # is x-bar_i (i.e. 1 - x_i); "NOT of literal NOT-x_i" is x_i itself.
        a_complemented = not i_negated  # term factor is x-bar_i when the literal was x_i
        b_complemented = not j_negated
        if i == j:
            raise ValueError("posiform clauses must pair distinct variables")
        if not a_complemented and not b_complemented:
            _add_quadratic(i, j, b)
        elif a_complemented and not b_complemented:
            _add_linear(j, b)
            _add_quadratic(i, j, -b)
        elif not a_complemented and b_complemented:
            _add_linear(i, b)
            _add_quadratic(i, j, -b)
        else:
            offset += b
            _add_linear(i, -b)
            _add_linear(j, -b)
            _add_quadratic(i, j, b)
    return offset, linear, quadratic


def generate_posiform_planted(
    n: int,
    seed: int,
    *,
    coefficient_choices: tuple[float, ...] = (1.0, 2.0),
    edge_candidates: list[tuple[int, int]] | None = None,
    name: str | None = None,
) -> ExternalQuboInstance:
    """R33 posiform planting: unique planted optimum, objective 0 at the optimum.

    NOTE: alone these are classically EASY (linear-time via 2-SAT); use
    generate_fused_planted for hardness with retained ground truth.
    """
    if n < 2:
        raise ValueError(f"posiform planting needs n >= 2, got {n}")
    rng = random.Random(seed)
    planted = [rng.randrange(2) for _ in range(n)]
    clauses = _plant_two_sat(n, planted, rng, edge_candidates)
    coefficients = [float(rng.choice(coefficient_choices)) for _ in clauses]
    offset, linear_idx, quadratic_idx = _posiform_to_qubo(n, clauses, coefficients)

    names = _variable_names(n)
    planted_solution = {names[i]: planted[i] for i in range(n)}
    instance = ExternalQuboInstance(
        name=name or f"posiform_planted_n{n}_s{seed}",
        variable_names=names,
        offset=offset,
        linear={names[i]: v for i, v in linear_idx.items() if v != 0.0},
        quadratic={(names[i], names[j]): v for (i, j), v in quadratic_idx.items() if v != 0.0},
        planted_solution=planted_solution,
        planted_energy=0.0,
        metadata={
            "generator": "posiform_planted",
            "reference": "arXiv:2308.05859",
            "clause_count": len(clauses),
            "seed": seed,
        },
    )
    _check_planted_energy(instance)
    return instance


# --------------------------------------------------------------------------
# Fused planted instances (R34, arXiv:2411.03626)
# --------------------------------------------------------------------------


def _random_block_qubo(
    block: list[int],
    rng: random.Random,
    coefficient_range: tuple[int, int],
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    low, high = coefficient_range
    linear = {i: float(rng.randint(low, high)) for i in block}
    quadratic: dict[tuple[int, int], float] = {}
    for a_pos in range(len(block)):
        for b_pos in range(a_pos + 1, len(block)):
            coefficient = float(rng.randint(low, high))
            if coefficient != 0.0:
                i, j = block[a_pos], block[b_pos]
                quadratic[(i, j) if i < j else (j, i)] = coefficient
    return linear, quadratic


def _block_minimum(
    block: list[int],
    linear: dict[int, float],
    quadratic: dict[tuple[int, int], float],
) -> tuple[dict[int, int], float]:
    """Exact brute force over the block (bounded by _MAX_BLOCK_SIZE); returns the
    first minimizer in enumeration order (deterministic)."""
    best_energy = float("inf")
    best_assignment: dict[int, int] = {}
    for bits in product((0, 1), repeat=len(block)):
        assignment = dict(zip(block, bits, strict=True))
        energy = sum(linear[i] * assignment[i] for i in block)
        energy += sum(v * assignment[i] * assignment[j] for (i, j), v in quadratic.items())
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment
    return best_assignment, best_energy


_MAX_BLOCK_SIZE = 16


def generate_fused_planted(
    n: int,
    seed: int,
    *,
    block_size: int = 10,
    alpha: float = 0.1,
    coefficient_range: tuple[int, int] = (-10, 10),
    name: str | None = None,
) -> ExternalQuboInstance:
    """R34: Q = sum_i R_i + alpha * P. Random block QUBOs R_i on a disjoint
    partition (each solved exactly by brute force), planted bitstring = the
    concatenated block minimizers, P posiform-planted on it with long-range
    (unrestricted) clause pairs. The planted point stays the UNIQUE optimum:
    each R_i is block-minimal there and alpha * P > 0 strictly elsewhere.
    Smaller alpha is harder (arXiv:2411.03626 Gurobi results)."""
    if not 2 <= block_size <= _MAX_BLOCK_SIZE:
        raise ValueError(f"block_size must be in [2, {_MAX_BLOCK_SIZE}], got {block_size}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0 to preserve uniqueness, got {alpha}")
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    blocks = [indices[start : start + block_size] for start in range(0, n, block_size)]

    total_linear: dict[int, float] = {}
    total_quadratic: dict[tuple[int, int], float] = {}
    planted = [0] * n
    blocks_energy = 0.0
    for block in blocks:
        if len(block) < 2:
            # A 1-variable tail block: linear term only, minimizer by sign.
            i = block[0]
            coefficient = float(rng.randint(*coefficient_range))
            total_linear[i] = total_linear.get(i, 0.0) + coefficient
            planted[i] = 1 if coefficient < 0 else 0
            blocks_energy += min(0.0, coefficient)
            continue
        linear, quadratic = _random_block_qubo(block, rng, coefficient_range)
        minimizer, block_energy = _block_minimum(block, linear, quadratic)
        blocks_energy += block_energy
        for i, v in linear.items():
            total_linear[i] = total_linear.get(i, 0.0) + v
        for key, v in quadratic.items():
            total_quadratic[key] = total_quadratic.get(key, 0.0) + v
        for i, bit in minimizer.items():
            planted[i] = bit

    clauses = _plant_two_sat(n, planted, rng, None)
    coefficients = [float(rng.choice((1.0, 2.0))) for _ in clauses]
    offset, posiform_linear, posiform_quadratic = _posiform_to_qubo(n, clauses, coefficients)
    for i, v in posiform_linear.items():
        total_linear[i] = total_linear.get(i, 0.0) + alpha * v
    for key, v in posiform_quadratic.items():
        total_quadratic[key] = total_quadratic.get(key, 0.0) + alpha * v

    names = _variable_names(n)
    instance = ExternalQuboInstance(
        name=name or f"fused_planted_n{n}_s{seed}_a{alpha:g}",
        variable_names=names,
        offset=alpha * offset,
        linear={names[i]: v for i, v in total_linear.items() if v != 0.0},
        quadratic={
            (names[i], names[j]): v for (i, j), v in total_quadratic.items() if v != 0.0
        },
        planted_solution={names[i]: planted[i] for i in range(n)},
        planted_energy=blocks_energy,
        metadata={
            "generator": "fused_planted",
            "reference": "arXiv:2411.03626",
            "block_size": block_size,
            "alpha": alpha,
            "clause_count": len(clauses),
            "coefficient_range": list(coefficient_range),
            "seed": seed,
        },
    )
    _check_planted_energy(instance)
    return instance


# --------------------------------------------------------------------------
# Long-range spin glass (no planted optimum; heuristic hardness)
# --------------------------------------------------------------------------


def generate_longrange_spin_glass(
    n: int,
    seed: int,
    *,
    mean_degree: float = 6.0,
    coupling_magnitude: float = 1.0,
    field_scale: float = 0.1,
    name: str | None = None,
) -> ExternalQuboInstance:
    """Random +-J couplings on ~mean_degree * n / 2 uniformly random long-range
    pairs, weak random fields. Non-geometric, so the coupling graph is an
    Erdos-Renyi-like expander: tree-width Theta(n) at fixed mean degree -- the
    tree-TN-negativity workhorse. No planted optimum (label heuristic)."""
    if n < 3:
        raise ValueError(f"spin glass needs n >= 3, got {n}")
    rng = random.Random(seed)
    edge_target = int(round(mean_degree * n / 2))
    edges: set[tuple[int, int]] = set()
    max_edges = n * (n - 1) // 2
    edge_target = min(edge_target, max_edges)
    while len(edges) < edge_target:
        i, j = rng.sample(range(n), 2)
        edges.add((i, j) if i < j else (j, i))

    names = _variable_names(n)
    quadratic = {
        (names[i], names[j]): coupling_magnitude * rng.choice((-1.0, 1.0)) for i, j in edges
    }
    linear = {
        names[i]: field_scale * rng.uniform(-1.0, 1.0)
        for i in range(n)
        if field_scale > 0.0
    }
    return ExternalQuboInstance(
        name=name or f"longrange_spin_glass_n{n}_s{seed}",
        variable_names=names,
        offset=0.0,
        linear=linear,
        quadratic=quadratic,
        planted_solution=None,
        planted_energy=None,
        metadata={
            "generator": "longrange_spin_glass",
            "mean_degree": mean_degree,
            "edge_count": len(edges),
            "seed": seed,
        },
    )


def _check_planted_energy(instance: ExternalQuboInstance) -> None:
    """The planted assignment must evaluate to the recorded planted energy exactly
    (integer/dyadic coefficients; any drift means the expansion is wrong)."""
    assert instance.planted_solution is not None
    assert instance.planted_energy is not None
    value = instance.energy(instance.planted_solution)
    if abs(value - instance.planted_energy) > 1e-9:
        raise AssertionError(
            f"{instance.name}: planted energy mismatch -- evaluated {value}, "
            f"recorded {instance.planted_energy}"
        )
