"""Path B external instance generators (R33 posiform planting, R34 fusion).

Uniqueness of the planted optimum is verified by exhaustive brute force at
small n (machine-checked-numeric); the posiform-to-QUBO expansion is anchored
on the worked example of arXiv:2308.05859 section 2.3. Uniqueness at large n
rests on the cited constructions, not on these tests.
"""

from __future__ import annotations

from itertools import product

import pytest

from eon.formulations.qubo import compile_external_qubo, compile_layer_b_qubo_hess
from eon.instances.external import (
    ExternalQuboInstance,
    _check_planted_energy,
    _posiform_to_qubo,
    build_external_surrogate,
    generate_fused_planted,
    generate_longrange_spin_glass,
    generate_posiform_planted,
)
from eon.instances.treewidth import coupling_diagnostics


def _brute_force_minima(instance: ExternalQuboInstance) -> tuple[float, list[dict[str, int]]]:
    best = float("inf")
    minima: list[dict[str, int]] = []
    for bits in product((0, 1), repeat=len(instance.variable_names)):
        assignment = dict(zip(instance.variable_names, bits, strict=True))
        energy = instance.energy(assignment)
        if energy < best - 1e-12:
            best = energy
            minima = [assignment]
        elif abs(energy - best) <= 1e-12:
            minima.append(assignment)
    return best, minima


@pytest.mark.parametrize("n,seed", [(6, 7), (8, 24), (10, 3)])
def test_posiform_planted_unique_optimum(n: int, seed: int) -> None:
    instance = generate_posiform_planted(n, seed)
    best, minima = _brute_force_minima(instance)
    assert abs(best - 0.0) <= 1e-12, "posiform planted optimum must be exactly 0"
    assert len(minima) == 1, "planted solution must be the UNIQUE optimum"
    assert minima[0] == instance.planted_solution
    assert instance.planted_energy == 0.0


def test_posiform_expansion_matches_paper_example() -> None:
    # arXiv:2308.05859 sec 2.3: clauses for planted (1,0,1) expand to
    # Q = x2 + x3 - 2*x1*x3 (+ offset 1), unique minimum at (1,0,1).
    # Literals: 2i = x_i, 2i+1 = NOT x_i, zero-indexed (x1 -> index 0).
    clauses = [
        (3, 5),  # (not-x2 or not-x3)
        (0, 3),  # (x1 or not-x2)
        (0, 5),  # (x1 or not-x3)
        (0, 2),  # (x1 or x2)
        (3, 4),  # (not-x2 or x3)
        (1, 4),  # (not-x1 or x3)
    ]
    offset, linear, quadratic = _posiform_to_qubo(3, clauses, [1.0] * 6)
    assert offset == 1.0
    assert {k: v for k, v in linear.items() if v != 0.0} == {1: 1.0, 2: 1.0}
    assert {k: v for k, v in quadratic.items() if v != 0.0} == {(0, 2): -2.0}


@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_fused_planted_unique_optimum(alpha: float) -> None:
    instance = generate_fused_planted(12, seed=7, block_size=4, alpha=alpha)
    best, minima = _brute_force_minima(instance)
    assert len(minima) == 1, "fusion must preserve the unique planted optimum"
    assert minima[0] == instance.planted_solution
    assert abs(best - float(instance.planted_energy)) <= 1e-9


def test_fused_planted_rejects_nonpositive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        generate_fused_planted(12, seed=7, alpha=0.0)


def test_generators_deterministic_in_seed() -> None:
    a = generate_fused_planted(20, seed=42, block_size=5)
    b = generate_fused_planted(20, seed=42, block_size=5)
    assert a == b
    p1 = generate_posiform_planted(12, seed=42)
    p2 = generate_posiform_planted(12, seed=42)
    assert p1 == p2
    g1 = generate_longrange_spin_glass(30, seed=11)
    g2 = generate_longrange_spin_glass(30, seed=11)
    assert g1 == g2


def test_spin_glass_shape() -> None:
    instance = generate_longrange_spin_glass(50, seed=5, mean_degree=6.0)
    assert instance.planted_solution is None
    assert len(instance.quadratic) == 150  # mean_degree * n / 2
    assert all(v in (-1.0, 1.0) for v in instance.quadratic.values())


def test_external_surrogate_has_no_cardinality_constraint() -> None:
    instance = generate_posiform_planted(10, seed=7)
    surrogate = build_external_surrogate(instance)
    assert surrogate.max_new_lines == 10, "at-most-K must be vacuous for external QUBOs"
    diagnostics = coupling_diagnostics(surrogate)
    assert diagnostics.variable_count == 10


def test_penalty_free_compilation_preserves_coupling_graph() -> None:
    instance = generate_fused_planted(16, seed=9, block_size=4)
    surrogate = build_external_surrogate(instance)

    free = compile_external_qubo(surrogate)
    assert free.method == "penalty_free"
    assert free.penalty_strength == 0.0
    free_pairs = {key for key in free.qubo if key[0] != key[1]}
    assert len(free_pairs) == len(instance.quadratic), (
        "penalty-free compilation must not add or drop couplings"
    )

    # The Hess path densifies -- the exact artifact penalty_free exists to avoid.
    hess = compile_layer_b_qubo_hess(surrogate)
    hess_pairs = {key for key in hess.qubo if key[0] != key[1]}
    assert len(hess_pairs) > len(free_pairs)


def test_planted_energy_mismatch_is_caught() -> None:
    instance = generate_posiform_planted(6, seed=7)
    corrupted = ExternalQuboInstance(
        name=instance.name,
        variable_names=instance.variable_names,
        offset=instance.offset + 0.5,  # drift the offset; planted energy now wrong
        linear=instance.linear,
        quadratic=instance.quadratic,
        planted_solution=instance.planted_solution,
        planted_energy=instance.planted_energy,
        metadata=instance.metadata,
    )
    with pytest.raises(AssertionError, match="planted energy mismatch"):
        _check_planted_energy(corrupted)
