"""Vanilla penalty-QAOA baseline (+ Egger warm start), the D4 comparator.

Same cost vector and scoring as the constrained-mixer cop-QAOA (energy.py /
postprocess.py); only the initial state and mixer differ, so the comparison
isolates the ansatz:

- vanilla: |+>^n init, X mixer (RX(-2 beta), i.e. exp(+i beta X)).
- warm-started (WS-QAOA, R46 Egger/Marecek/Woerner, arXiv:2009.10095):
  init R_Y(theta_i)|0>, mixer R_Y(theta_i) R_Z(-2 beta) R_Y(-theta_i), with
  theta_i = 2 arcsin(sqrt(c_i*)) from the continuous relaxation of the
  cardinality-constrained surrogate, epsilon-regularized. At epsilon = 0.5 this
  reduces exactly to vanilla.

The Clifford warm-start variant remains OUT: its citation (2602.14327) is on
the reading.txt suspect list (D11).
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

from eon.formulations.layer_b import LayerBSurrogate
from eon.quantum.angles import AngleSchedule, optimize_angles
from eon.quantum.cop_qaoa import QAOARunResult
from eon.quantum.energy import build_energy_vector
from eon.quantum.postprocess import decode_counts
from eon.quantum.simulator import (
    expected_energy_of_state,
    product_state,
    sample_counts,
    simulate_qaoa,
    uniform_state,
)
from eon.validation import validate_finite_array


def solve_relaxation(surrogate: LayerBSurrogate, *, time_limit_s: float = 60.0) -> np.ndarray:
    """Continuous [0,1] relaxation of the cardinality-constrained surrogate
    (Gurobi nonconvex QP). Returns c* over ACTUAL BUILDS, in variable order."""
    import gurobipy as gp  # lazy: keep the package importable license-free
    from gurobipy import GRB

    model = gp.Model("surrogate_relaxation")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit_s
    model.Params.NonConvex = 2

    builds = {
        variable.name: model.addVar(lb=0.0, ub=1.0, name=variable.name)
        for variable in surrogate.variables
    }
    # toggle = default XOR build; for relaxed build b: t = d + (1 - 2d) b.
    toggles = {
        variable.name: (
            builds[variable.name]
            if variable.default_value == 0
            else 1.0 - builds[variable.name]
        )
        for variable in surrogate.variables
    }
    objective = gp.QuadExpr(surrogate.offset)
    for variable in surrogate.variables:
        objective += surrogate.linear[variable.name] * toggles[variable.name]
    for (left, right), coefficient in surrogate.quadratic.items():
        objective += coefficient * toggles[left] * toggles[right]
    outside_selected = sum(
        value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    )
    model.addConstr(
        gp.quicksum(builds.values()) + outside_selected <= surrogate.max_new_lines,
        name="cardinality",
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("surrogate relaxation produced no solution")
    return np.asarray([builds[v.name].X for v in surrogate.variables], dtype=float)


def warm_start_thetas(c_star: np.ndarray, *, epsilon: float = 0.25) -> np.ndarray:
    """theta_i = 2 arcsin(sqrt(c_i*)) with the R46 epsilon regularization
    (avoids frozen qubits at c* in {0,1}; epsilon = 0.5 recovers vanilla)."""
    if not 0.0 <= epsilon <= 0.5:
        raise ValueError(f"epsilon must be in [0, 0.5], got {epsilon}")
    clamped = np.clip(c_star, epsilon, 1.0 - epsilon)
    return 2.0 * np.arcsin(np.sqrt(clamped))


def build_penalty_qaoa_circuit(
    energies: np.ndarray,
    *,
    p: int,
    betas: tuple[float, ...],
    gammas: tuple[float, ...],
    thetas: np.ndarray | None = None,
) -> QuantumCircuit:
    """thetas=None -> vanilla (|+>^n, X mixer); thetas -> WS-QAOA (R46)."""
    if len(betas) != p or len(gammas) != p:
        raise ValueError("QAOA parameter lengths must match p.")
    n = int(math.log2(len(energies)))
    if 2**n != len(energies):
        raise ValueError("energy vector length must be a power of two")
    if thetas is not None and len(thetas) != n:
        raise ValueError("thetas length must match qubit count")

    circuit = QuantumCircuit(n)
    if thetas is None:
        circuit.h(range(n))
    else:
        for qubit in range(n):
            circuit.ry(float(thetas[qubit]), qubit)

    for layer in range(p):
        phase = np.exp(-1j * gammas[layer] * energies)
        circuit.append(DiagonalGate(phase), range(n))
        for qubit in range(n):
            if thetas is None:
                circuit.rx(-2.0 * betas[layer], qubit)
            else:
                # exp(-i beta H_M,i^ws) = R_Y(theta) R_Z(-2 beta) R_Y(-theta).
                circuit.ry(-float(thetas[qubit]), qubit)
                circuit.rz(-2.0 * betas[layer], qubit)
                circuit.ry(float(thetas[qubit]), qubit)
    return circuit


def expected_energy_from_vector(energies: np.ndarray, circuit: QuantumCircuit) -> float:
    statevector = Statevector.from_instruction(circuit)
    return float(np.real(np.dot(statevector.probabilities(), energies)))


def run_penalty_qaoa(
    surrogate: LayerBSurrogate,
    *,
    p: int = 1,
    shots: int = 1024,
    warm_start: bool = False,
    epsilon: float = 0.25,
    nelder_mead_evals: int = 60,
    relaxation_time_limit_s: float = 60.0,
    energies: np.ndarray | None = None,
    seed: int = 7,
) -> list[QAOARunResult]:
    """Run vanilla (or warm-started) penalty-QAOA at depths 1..p; one
    QAOARunResult per depth, angles from the shared optimizer. Simulation is
    the exact numpy engine (simulator.py) -- Qiskit's Statevector synthesizes
    the 2^n DiagonalGate and is intractable at n=20; the Qiskit circuit
    builder above stays as the exportable artifact and small-n cross-check."""
    if energies is None:
        energies = build_energy_vector(surrogate)
    validate_finite_array(energies, name="penalty_qaoa_energy_vector")
    n = len(surrogate.variables)
    thetas = None
    if warm_start:
        thetas = warm_start_thetas(
            solve_relaxation(surrogate, time_limit_s=relaxation_time_limit_s),
            epsilon=epsilon,
        )
    mixer = "x" if thetas is None else "warm_start"
    initial = uniform_state(n) if thetas is None else product_state(thetas)

    def energy_fn(betas: tuple[float, ...], gammas: tuple[float, ...]) -> float:
        state = simulate_qaoa(initial, energies, betas, gammas, mixer=mixer, thetas=thetas)
        return expected_energy_of_state(state, energies)

    schedules: list[AngleSchedule] = optimize_angles(
        energy_fn, p, nelder_mead_evals=nelder_mead_evals
    )

    results: list[QAOARunResult] = []
    for schedule in schedules:
        state = simulate_qaoa(
            initial, energies, schedule.betas, schedule.gammas, mixer=mixer, thetas=thetas
        )
        # Distinct sampling stream per (instance seed, algorithm, depth); cop
        # uses salt 0 in run_constrained_qaoa_depths.
        salt = 2 if warm_start else 1
        counts = sample_counts(state, shots, seed=(seed * 3 + salt) * 101 + schedule.p)
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
