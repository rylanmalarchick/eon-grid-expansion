"""The exact numpy QAOA engine must match Qiskit statevector simulation at
small n (all three mixers, multi-layer, including bit/axis conventions)."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from eon.quantum.energy import build_energy_vector
from eon.quantum.qaoa import build_penalty_qaoa_circuit, warm_start_thetas
from eon.quantum.simulator import (
    expected_energy_of_state,
    product_state,
    sample_counts,
    simulate_qaoa,
    uniform_state,
)
from tests.test_qaoa_baseline import _toy_surrogate

BETAS = (0.37, 0.81)
GAMMAS = (0.93, 0.24)


def _qiskit_state(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector.from_instruction(circuit).data)


def _assert_states_match(numpy_state: np.ndarray, qiskit_state: np.ndarray) -> None:
    # Global phase must also match: both paths implement identical gates.
    np.testing.assert_allclose(numpy_state, qiskit_state, atol=1e-10)


def test_vanilla_matches_qiskit_p2() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)
    numpy_state = simulate_qaoa(
        uniform_state(4), energies, BETAS, GAMMAS, mixer="x"
    )
    circuit = build_penalty_qaoa_circuit(energies, p=2, betas=BETAS, gammas=GAMMAS)
    _assert_states_match(numpy_state, _qiskit_state(circuit))


def test_warm_start_matches_qiskit_p2() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)
    thetas = warm_start_thetas(np.asarray([0.1, 0.7, 0.4, 0.95]), epsilon=0.2)
    numpy_state = simulate_qaoa(
        product_state(thetas), energies, BETAS, GAMMAS, mixer="warm_start", thetas=thetas
    )
    circuit = build_penalty_qaoa_circuit(
        energies, p=2, betas=BETAS, gammas=GAMMAS, thetas=thetas
    )
    _assert_states_match(numpy_state, _qiskit_state(circuit))


def test_xy_ring_matches_qiskit_p2() -> None:
    from eon.quantum.cop_qaoa import (
        _target_hamming_weight,
        _uniform_weight_state,
        build_constrained_qaoa_circuit,
    )

    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)
    initial = _uniform_weight_state(4, _target_hamming_weight(surrogate))
    numpy_state = simulate_qaoa(initial, energies, BETAS, GAMMAS, mixer="xy_ring")
    circuit = build_constrained_qaoa_circuit(
        surrogate, p=2, betas=BETAS, gammas=GAMMAS, energy_vector=energies
    )
    _assert_states_match(numpy_state, _qiskit_state(circuit))


def test_xy_ring_preserves_hamming_weight() -> None:
    from eon.quantum.cop_qaoa import _uniform_weight_state

    initial = _uniform_weight_state(5, 2)
    energies = np.arange(2**5, dtype=float)
    state = simulate_qaoa(initial, energies, (0.6, 0.3), (0.9, 0.2), mixer="xy_ring")
    weights = np.asarray([bin(index).count("1") for index in range(2**5)])
    off_weight_mass = float(np.sum(np.abs(state[weights != 2]) ** 2))
    assert off_weight_mass < 1e-20


def test_expected_energy_and_sampling_consistency() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate)
    state = simulate_qaoa(uniform_state(4), energies, (0.4,), (0.8,), mixer="x")
    energy = expected_energy_of_state(state, energies)
    assert np.isfinite(energy)
    counts = sample_counts(state, 4096, seed=11)
    assert sum(counts.values()) == 4096
    assert all(len(key) == 4 for key in counts)
    # Sampled mean energy converges toward the exact expectation.
    index_of = {format(i, "04b"): i for i in range(16)}
    sampled_mean = (
        sum(energies[index_of[key]] * count for key, count in counts.items()) / 4096
    )
    assert sampled_mean == pytest.approx(energy, rel=0.2)
