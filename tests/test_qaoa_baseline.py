"""Vanilla penalty-QAOA baseline + shared angle machinery (P3 comparator).

Anchors: the vectorized energy vector against a hand-rolled per-state
computation (no shared code path), the WS-QAOA construction against the R46
formulas, and the optimizer against variational monotonicity in p.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
from eon.instances.candidate_lines import CandidateLine
from eon.quantum.angles import optimize_angles
from eon.quantum.energy import CARDINALITY_PENALTY, build_energy_vector
from eon.quantum.qaoa import (
    build_penalty_qaoa_circuit,
    expected_energy_from_vector,
    run_penalty_qaoa,
    warm_start_thetas,
)


def _toy_surrogate(n: int = 4, *, default_one: bool = False) -> LayerBSurrogate:
    candidates = [
        CandidateLine(
            name=f"t{i}",
            from_bus=i,
            to_bus=(i + 1) % n,
            family="toy",
            length_km=1.0,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.3,
            max_i_ka=0.4,
            build_cost=1.0,
        )
        for i in range(n)
    ]
    variables = tuple(
        LayerBVariable(
            name=c.name,
            candidate=c,
            default_value=1 if (default_one and i == 0) else 0,
            stress_score=1.0,
        )
        for i, c in enumerate(candidates)
    )
    linear = {v.name: -2.0 + 0.5 * i for i, v in enumerate(variables)}
    quadratic = {
        (variables[0].name, variables[2].name): 1.5,
        (variables[1].name, variables[3].name): -0.75,
    }
    return LayerBSurrogate(
        variables=variables,
        offset=3.0,
        linear=linear,
        quadratic=quadratic,
        fixed_builds={},
        max_new_lines=2,
        base_objective=3.0,
        base_selected_count=2,
    )


@pytest.mark.parametrize("default_one", [False, True])
def test_energy_vector_matches_per_state_hand_computation(default_one: bool) -> None:
    surrogate = _toy_surrogate(default_one=default_one)
    n = len(surrogate.variables)
    energies = build_energy_vector(surrogate)
    assert energies.shape == (2**n,)
    for state in range(2**n):
        builds = {v.name: (state >> i) & 1 for i, v in enumerate(surrogate.variables)}
        toggles = {v.name: v.default_value ^ builds[v.name] for v in surrogate.variables}
        expected = surrogate.surrogate_objective(toggles)
        if sum(builds.values()) > surrogate.max_new_lines:
            expected += surrogate.base_objective + CARDINALITY_PENALTY
        assert energies[state] == pytest.approx(expected, abs=1e-9)


def test_vanilla_p1_expected_energy_matches_numpy_simulation() -> None:
    """No-Qiskit anchor: |+>^n -> diagonal phase -> RX mixer, in raw numpy."""
    surrogate = _toy_surrogate()
    n = len(surrogate.variables)
    energies = build_energy_vector(surrogate)
    beta, gamma = 0.37, 0.91

    state = np.full(2**n, 1.0 / math.sqrt(2**n), dtype=complex)
    state = np.exp(-1j * gamma * energies) * state
    # RX(-2 beta) on each qubit = exp(+i beta X): amplitude mixing per bit.
    for qubit in range(n):
        stride = 1 << qubit
        new_state = state.copy()
        for index in range(2**n):
            partner = index ^ stride
            new_state[index] = (
                math.cos(beta) * state[index] + 1j * math.sin(beta) * state[partner]
            )
        state = new_state
    numpy_energy = float(np.real(np.dot(np.abs(state) ** 2, energies)))

    circuit = build_penalty_qaoa_circuit(energies, p=1, betas=(beta,), gammas=(gamma,))
    qiskit_energy = expected_energy_from_vector(energies, circuit)
    # Values sit at the big-M penalty scale (~1e6); associativity differences
    # make abs tolerance meaningless -- this is a deterministic computation, so
    # the relative tolerance is pinned tight.
    assert qiskit_energy == pytest.approx(numpy_energy, rel=1e-12)


def test_warm_start_thetas_formula_and_clamping() -> None:
    c_star = np.asarray([0.0, 0.25, 0.5, 1.0])
    thetas = warm_start_thetas(c_star, epsilon=0.25)
    assert thetas[0] == pytest.approx(2 * math.asin(math.sqrt(0.25)))
    assert thetas[1] == pytest.approx(2 * math.asin(math.sqrt(0.25)))
    assert thetas[2] == pytest.approx(2 * math.asin(math.sqrt(0.5)))
    assert thetas[3] == pytest.approx(2 * math.asin(math.sqrt(0.75)))
    with pytest.raises(ValueError, match="epsilon"):
        warm_start_thetas(c_star, epsilon=0.6)


def test_warm_start_at_epsilon_half_reduces_to_vanilla() -> None:
    """R46: epsilon = 0.5 recovers standard QAOA exactly (init and mixer)."""
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)
    thetas = warm_start_thetas(np.asarray([0.0, 1.0, 0.3, 0.9]), epsilon=0.5)
    betas, gammas = (0.4,), (0.8,)
    vanilla = Statevector.from_instruction(
        build_penalty_qaoa_circuit(energies, p=1, betas=betas, gammas=gammas)
    )
    warm = Statevector.from_instruction(
        build_penalty_qaoa_circuit(energies, p=1, betas=betas, gammas=gammas, thetas=thetas)
    )
    assert vanilla.equiv(warm)


def test_optimizer_monotone_in_depth_on_toy() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)

    def energy_fn(betas: tuple[float, ...], gammas: tuple[float, ...]) -> float:
        circuit = build_penalty_qaoa_circuit(
            energies, p=len(betas), betas=betas, gammas=gammas
        )
        return expected_energy_from_vector(energies, circuit)

    schedules = optimize_angles(energy_fn, 3, nelder_mead_evals=40)
    assert [s.p for s in schedules] == [1, 2, 3]
    # Monotone in depth BY CONSTRUCTION: the zero-padded start reproduces the
    # previous depth's state exactly and keep-best never regresses from it.
    assert schedules[1].expected_energy <= schedules[0].expected_energy + 1e-6
    assert schedules[2].expected_energy <= schedules[1].expected_energy + 1e-6


def test_run_penalty_qaoa_returns_per_depth_results() -> None:
    surrogate = _toy_surrogate()
    results = run_penalty_qaoa(surrogate, p=2, shots=256, nelder_mead_evals=20)
    assert [r.p for r in results] == [1, 2]
    assert all(len(r.best_sample.bitstring) == 4 for r in results)
    assert all(np.isfinite(r.expected_energy) for r in results)
    # The feasible-first decode ordering: if ANY feasible state was sampled the
    # best_sample must be feasible (with 256 shots on 4 qubits it always is).
    assert all(r.best_sample.feasible for r in results)
