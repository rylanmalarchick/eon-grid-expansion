"""The exported gate-level circuit must BE the benchmarked algorithm.

Everything we report about cop-QAOA / vanilla comes from the numpy engine,
which applies a single diagonal phase operator. The hardware artifact instead
applies RZ/RZZ gates. If those two disagree, the exported circuit is a
lookalike and every depth/gate-count claim describes a different algorithm.
These tests pin the equality (up to global phase) for all three mixers.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.energy import build_energy_vector
from eon.quantum.export import build_gate_level_qaoa_circuit
from eon.quantum.qaoa import warm_start_thetas
from eon.quantum.simulator import (
    product_state,
    simulate_qaoa,
    uniform_state,
)
from tests.test_qaoa_baseline import _toy_surrogate

BETAS = (0.37, 0.81)
GAMMAS = (0.93, 0.24)


def _equal_up_to_global_phase(a: np.ndarray, b: np.ndarray) -> bool:
    """b == e^{i phi} a. With <a|b> = e^{i phi} for normalized states, |overlap|
    == 1 is the criterion; rotating a by +phi must then reproduce b. (The
    penalty magnitude ~5e4 makes the absolute phase wrap ~1e4 times, so the
    phase itself is meaningless -- only the alignment is.)"""
    overlap = np.vdot(a, b)
    if not np.isclose(abs(overlap), 1.0, atol=1e-9):
        return False
    return bool(np.allclose(a * np.exp(1j * np.angle(overlap)), b, atol=1e-8))


def test_gate_level_matches_engine_vanilla() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate, penalty_mode="quadratic")
    engine = simulate_qaoa(uniform_state(4), energies, BETAS, GAMMAS, mixer="x")
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=BETAS, gammas=GAMMAS, mixer="x"
    )
    exported = np.asarray(Statevector.from_instruction(circuit).data)
    assert _equal_up_to_global_phase(engine, exported)


def test_gate_level_matches_engine_warm_start() -> None:
    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate, penalty_mode="quadratic")
    thetas = warm_start_thetas(np.asarray([0.1, 0.7, 0.4, 0.95]), epsilon=0.2)
    engine = simulate_qaoa(
        product_state(thetas), energies, BETAS, GAMMAS, mixer="warm_start", thetas=thetas
    )
    circuit = build_gate_level_qaoa_circuit(
        surrogate,
        betas=BETAS,
        gammas=GAMMAS,
        mixer="warm_start",
        thetas=[float(t) for t in thetas],
    )
    exported = np.asarray(Statevector.from_instruction(circuit).data)
    assert _equal_up_to_global_phase(engine, exported)


def test_gate_level_matches_engine_xy_ring() -> None:
    from eon.quantum.cop_qaoa import _target_hamming_weight, _uniform_weight_state

    surrogate = _toy_surrogate()
    energies = build_energy_vector(surrogate, penalty_mode="quadratic")
    initial = _uniform_weight_state(4, _target_hamming_weight(surrogate))
    engine = simulate_qaoa(initial, energies, BETAS, GAMMAS, mixer="xy_ring")
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=BETAS, gammas=GAMMAS, mixer="xy_ring"
    )
    exported = np.asarray(Statevector.from_instruction(circuit).data)
    assert _equal_up_to_global_phase(engine, exported)


def test_penalty_free_path_matches_engine() -> None:
    """penalty_free compiles WITHOUT a cardinality penalty, so the matching
    engine cost is penalty_mode='none' (external instances carry no at-most-K
    constraint at all)."""
    instance = generate_fused_planted(8, seed=7, block_size=4, alpha=0.5)
    surrogate = build_external_surrogate(instance)
    energies = build_energy_vector(surrogate, penalty_mode="none")
    engine = simulate_qaoa(uniform_state(8), energies, (0.3,), (0.5,), mixer="x")
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=(0.3,), gammas=(0.5,), mixer="x", penalty_free=True
    )
    exported = np.asarray(Statevector.from_instruction(circuit).data)
    assert _equal_up_to_global_phase(engine, exported)


def test_rejects_unknown_mixer_and_length_mismatch() -> None:
    surrogate = _toy_surrogate()
    with pytest.raises(ValueError, match="mixer"):
        build_gate_level_qaoa_circuit(
            surrogate, betas=(0.1,), gammas=(0.1,), mixer="nonsense"
        )
    with pytest.raises(ValueError, match="equal length"):
        build_gate_level_qaoa_circuit(
            surrogate, betas=(0.1, 0.2), gammas=(0.1,), mixer="x"
        )


def test_export_is_free_of_dense_diagonal_gates() -> None:
    """The whole point: no 2^n operator, so this transpiles and exports."""
    surrogate = _toy_surrogate()
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=BETAS, gammas=GAMMAS, mixer="x"
    )
    names = {instruction.operation.name for instruction in circuit.data}
    assert "diagonal" not in names
    assert names <= {"h", "rz", "rzz", "rx", "barrier"}
