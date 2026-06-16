from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, XXPlusYYGate
from qiskit.quantum_info import Statevector

from eon.formulations.layer_b import LayerBSurrogate
from eon.quantum.postprocess import QuantumResult, decode_counts
from eon.validation import (
    validate_finite_array,
    validate_hermitian_matrix,
    validate_probability_array,
    validate_unitary_matrix,
)


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
) -> QAOALandscape:
    betas = beta_values or tuple(np.linspace(0.1, math.pi / 2, 7))
    gammas = gamma_values or tuple(np.linspace(0.1, math.pi, 7))
    energies: list[tuple[float, ...]] = []
    best = (float("inf"), betas[0], gammas[0])
    for beta in betas:
        row = []
        for gamma in gammas:
            circuit = build_constrained_qaoa_circuit(surrogate, p=1, betas=(beta,), gammas=(gamma,))
            energy = expected_energy(surrogate, circuit)
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
    landscape = build_p1_landscape(
        surrogate,
        beta_values=beta_values,
        gamma_values=gamma_values,
    )
    betas = tuple(landscape.best_beta for _ in range(p))
    gammas = tuple(landscape.best_gamma for _ in range(p))
    circuit = build_constrained_qaoa_circuit(surrogate, p=p, betas=betas, gammas=gammas)
    energy = expected_energy(surrogate, circuit)
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
) -> QuantumCircuit:
    if len(betas) != p or len(gammas) != p:
        raise ValueError("QAOA parameter lengths must match p.")

    n = len(surrogate.variables)
    circuit = QuantumCircuit(n)
    circuit.initialize(_uniform_weight_state(n, _target_hamming_weight(surrogate)), range(n))
    energies = _energy_vector(surrogate)
    validate_finite_array(energies, name="qaoa_energy_vector")
    validate_hermitian_matrix(np.diag(energies), name="qaoa_cost_hamiltonian")

    for layer in range(p):
        phase = np.exp(-1j * gammas[layer] * energies)
        phase_gate = DiagonalGate(phase)
        validate_unitary_matrix(
            np.diag(phase),
            name=f"qaoa_phase_gate_layer{layer}",
        )
        circuit.append(phase_gate, range(n))
        if n > 1:
            for left in range(n - 1):
                mixer_gate = XXPlusYYGate(2.0 * betas[layer])
                validate_unitary_matrix(
                    mixer_gate.to_matrix(),
                    name=f"qaoa_mixer_gate_layer{layer}_{left}",
                )
                circuit.append(mixer_gate, [left, left + 1])
            wrap_gate = XXPlusYYGate(2.0 * betas[layer])
            validate_unitary_matrix(
                wrap_gate.to_matrix(),
                name=f"qaoa_mixer_gate_layer{layer}_wrap",
            )
            circuit.append(wrap_gate, [n - 1, 0])
    return circuit


def expected_energy(surrogate: LayerBSurrogate, circuit: QuantumCircuit) -> float:
    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities()
    validate_probability_array(probabilities, name="qaoa_statevector_probabilities")
    return float(np.dot(probabilities, _energy_vector(surrogate)))


def _energy_vector(surrogate: LayerBSurrogate) -> np.ndarray:
    energies = []
    outside_selected = sum(
        value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    )
    penalty = surrogate.base_objective + 10_000_000.0
    for state in range(2 ** len(surrogate.variables)):
        actual_builds = {}
        bitstring = format(state, f"0{len(surrogate.variables)}b")[::-1]
        for variable, bit in zip(surrogate.variables, bitstring, strict=False):
            actual_builds[variable.name] = int(bit)
        toggle_decisions = {
            variable.name: variable.default_value ^ actual_builds[variable.name]
            for variable in surrogate.variables
        }
        energy = surrogate.surrogate_objective(toggle_decisions)
        if sum(actual_builds.values()) + outside_selected > surrogate.max_new_lines:
            energy += penalty
        energies.append(energy)
    return np.asarray(energies, dtype=float)


def _target_hamming_weight(surrogate: LayerBSurrogate) -> int:
    return min(surrogate.max_new_lines, max(1, surrogate.base_selected_count or 1))


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
