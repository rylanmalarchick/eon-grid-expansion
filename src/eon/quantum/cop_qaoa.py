from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, XXPlusYYGate
from qiskit.quantum_info import Statevector

from eon.formulations.layer_b import LayerBSurrogate
from eon.quantum.angles import optimize_angles
from eon.quantum.energy import build_energy_vector
from eon.quantum.postprocess import QuantumResult, decode_counts
from eon.validation import (
    validate_finite_array,
    validate_hermitian_matrix,
    validate_probability_array,
    validate_unitary_matrix,
)

# The dense-matrix validations (hermitian/unitary on diag(energies)) are O(4^n)
# memory; they are exhaustive checks for small blocks only. Above this qubit
# count only the O(2^n) vector checks run.
_DENSE_VALIDATION_MAX_QUBITS = 10


@dataclass(frozen=True, slots=True)
class QAOALandscape:
    betas: tuple[float, ...]
    gammas: tuple[float, ...]
    energies: tuple[tuple[float, ...], ...]
    best_beta: float
    best_gamma: float
    best_energy: float


@dataclass(frozen=True, slots=True)
class QAOARunResult:
    p: int
    betas: tuple[float, ...]
    gammas: tuple[float, ...]
    expected_energy: float
    best_sample: QuantumResult
    counts: dict[str, int]
    landscape: QAOALandscape | None


def build_p1_landscape(
    surrogate: LayerBSurrogate,
    *,
    beta_values: tuple[float, ...] | None = None,
    gamma_values: tuple[float, ...] | None = None,
    energy_vector: np.ndarray | None = None,
) -> QAOALandscape:
    betas = beta_values or tuple(np.linspace(0.1, math.pi / 2, 7))
    gammas = gamma_values or tuple(np.linspace(0.1, math.pi, 7))
    if energy_vector is None:
        energy_vector = _energy_vector(surrogate)
    energies: list[tuple[float, ...]] = []
    best = (float("inf"), betas[0], gammas[0])
    for beta in betas:
        row = []
        for gamma in gammas:
            circuit = build_constrained_qaoa_circuit(
                surrogate, p=1, betas=(beta,), gammas=(gamma,), energy_vector=energy_vector
            )
            energy = expected_energy(surrogate, circuit, energy_vector=energy_vector)
            row.append(energy)
            if energy < best[0]:
                best = (energy, beta, gamma)
        energies.append(tuple(row))
    return QAOALandscape(
        betas=betas,
        gammas=gammas,
        energies=tuple(energies),
        best_beta=best[1],
        best_gamma=best[2],
        best_energy=best[0],
    )


def run_constrained_qaoa_subproblem(
    surrogate: LayerBSurrogate,
    *,
    p: int = 1,
    shots: int = 1024,
    beta_values: tuple[float, ...] | None = None,
    gamma_values: tuple[float, ...] | None = None,
) -> QAOARunResult:
    energy_vector = _energy_vector(surrogate)
    landscape = build_p1_landscape(
        surrogate,
        beta_values=beta_values,
        gamma_values=gamma_values,
        energy_vector=energy_vector,
    )
    betas = tuple(landscape.best_beta for _ in range(p))
    gammas = tuple(landscape.best_gamma for _ in range(p))
    circuit = build_constrained_qaoa_circuit(
        surrogate, p=p, betas=betas, gammas=gammas, energy_vector=energy_vector
    )
    energy = expected_energy(surrogate, circuit, energy_vector=energy_vector)
    statevector = Statevector.from_instruction(circuit)
    counts = dict(statevector.sample_counts(shots))
    decoded = decode_counts(surrogate, counts, total_shots=shots)
    best_sample = min(
        decoded,
        key=lambda result: (not result.feasible, result.objective, -result.sampling_prob),
    )
    return QAOARunResult(
        p=p,
        betas=betas,
        gammas=gammas,
        expected_energy=energy,
        best_sample=best_sample,
        counts=counts,
        landscape=landscape if p == 1 else None,
    )


def build_constrained_qaoa_circuit(
    surrogate: LayerBSurrogate,
    *,
    p: int,
    betas: tuple[float, ...],
    gammas: tuple[float, ...],
    energy_vector: np.ndarray | None = None,
) -> QuantumCircuit:
    if len(betas) != p or len(gammas) != p:
        raise ValueError("QAOA parameter lengths must match p.")

    n = len(surrogate.variables)
    dense_checks = n <= _DENSE_VALIDATION_MAX_QUBITS
    circuit = QuantumCircuit(n)
    circuit.initialize(_uniform_weight_state(n, _target_hamming_weight(surrogate)), range(n))
    energies = energy_vector if energy_vector is not None else _energy_vector(surrogate)
    validate_finite_array(energies, name="qaoa_energy_vector")
    if dense_checks:
        validate_hermitian_matrix(np.diag(energies), name="qaoa_cost_hamiltonian")

    for layer in range(p):
        phase = np.exp(-1j * gammas[layer] * energies)
        phase_gate = DiagonalGate(phase)
        if dense_checks:
            validate_unitary_matrix(
                np.diag(phase),
                name=f"qaoa_phase_gate_layer{layer}",
            )
        circuit.append(phase_gate, range(n))
        if n > 1:
            for left in range(n - 1):
                mixer_gate = XXPlusYYGate(2.0 * betas[layer])
                if dense_checks:
                    validate_unitary_matrix(
                        mixer_gate.to_matrix(),
                        name=f"qaoa_mixer_gate_layer{layer}_{left}",
                    )
                circuit.append(mixer_gate, [left, left + 1])
            wrap_gate = XXPlusYYGate(2.0 * betas[layer])
            if dense_checks:
                validate_unitary_matrix(
                    wrap_gate.to_matrix(),
                    name=f"qaoa_mixer_gate_layer{layer}_wrap",
                )
            circuit.append(wrap_gate, [n - 1, 0])
    return circuit


def expected_energy(
    surrogate: LayerBSurrogate,
    circuit: QuantumCircuit,
    *,
    energy_vector: np.ndarray | None = None,
) -> float:
    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities()
    validate_probability_array(probabilities, name="qaoa_statevector_probabilities")
    if energy_vector is None:
        energy_vector = _energy_vector(surrogate)
    return float(np.dot(probabilities, energy_vector))


def run_constrained_qaoa_depths(
    surrogate: LayerBSurrogate,
    *,
    p: int,
    shots: int = 1024,
    nelder_mead_evals: int = 60,
    energy_vector: np.ndarray | None = None,
) -> list[QAOARunResult]:
    """cop-QAOA at depths 1..p with per-depth angle optimization (the shared
    grid+INTERP+Nelder-Mead schedule, same budget as the vanilla baseline).
    Simulation is the exact numpy engine (simulator.py; Qiskit's Statevector
    synthesizes the 2^n DiagonalGate and is intractable at n=20)."""
    from eon.quantum.simulator import (
        expected_energy_of_state,
        sample_counts,
        simulate_qaoa,
    )

    energies = energy_vector if energy_vector is not None else _energy_vector(surrogate)
    n = len(surrogate.variables)
    initial = _uniform_weight_state(n, _target_hamming_weight(surrogate))

    def energy_fn(betas: tuple[float, ...], gammas: tuple[float, ...]) -> float:
        state = simulate_qaoa(initial, energies, betas, gammas, mixer="xy_ring")
        return expected_energy_of_state(state, energies)

    results: list[QAOARunResult] = []
    for schedule in optimize_angles(energy_fn, p, nelder_mead_evals=nelder_mead_evals):
        state = simulate_qaoa(
            initial, energies, schedule.betas, schedule.gammas, mixer="xy_ring"
        )
        counts = sample_counts(state, shots, seed=7 + schedule.p)
        decoded = decode_counts(surrogate, counts, total_shots=shots)
        best_sample = min(
            decoded,
            key=lambda result: (not result.feasible, result.objective, -result.sampling_prob),
        )
        results.append(
            QAOARunResult(
                p=schedule.p,
                betas=schedule.betas,
                gammas=schedule.gammas,
                expected_energy=schedule.expected_energy,
                best_sample=best_sample,
                counts=counts,
                landscape=None,
            )
        )
    return results


def _energy_vector(surrogate: LayerBSurrogate) -> np.ndarray:
    return build_energy_vector(surrogate)


def _target_hamming_weight(surrogate: LayerBSurrogate) -> int:
    # Capped at the variable count: decomposition blocks inherit the parent's
    # max_new_lines / base_selected_count, which can exceed the block size (an
    # empty fixed-weight subspace otherwise).
    return min(
        len(surrogate.variables),
        surrogate.max_new_lines,
        max(1, surrogate.base_selected_count or 1),
    )


def _uniform_weight_state(num_qubits: int, hamming_weight: int) -> np.ndarray:
    amplitudes = np.zeros(2**num_qubits, dtype=complex)
    valid_states = [
        state
        for state in range(2**num_qubits)
        if format(state, f"0{num_qubits}b").count("1") == hamming_weight
    ]
    amplitude = 1.0 / math.sqrt(len(valid_states))
    for state in valid_states:
        amplitudes[state] = amplitude
    return amplitudes
