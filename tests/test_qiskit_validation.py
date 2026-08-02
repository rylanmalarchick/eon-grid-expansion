"""The Aer-vs-numpy cross-check must have teeth.

scripts/qiskit_validation.py is the artifact E.ON runs to confirm the circuit we
hand over is the algorithm we benchmarked. A cross-check that passes no matter
what the circuit does would validate nothing, so the test below first breaks the
circuit and pins that the check FAILS, then pins that it passes when intact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.energy import build_energy_vector
from eon.quantum.export import build_gate_level_qaoa_circuit
from eon.quantum.postprocess import decode_counts
from eon.quantum.simulator import expected_energy_of_state, simulate_qaoa, uniform_state

BETAS = (0.42, 0.19)
GAMMAS = (0.31, 0.57)
SHOTS = 8192
SEED = 7


@pytest.fixture(scope="module")
def surrogate():
    # n=12 keeps the exact 2^n engine and Aer both fast; the property under
    # test (two engines, one circuit) does not depend on width.
    return build_external_surrogate(
        generate_fused_planted(12, seed=SEED, block_size=6, alpha=0.1)
    )


def _aer_expectation(surrogate, betas, gammas) -> tuple[float, float]:
    """Returns (expectation, standard error) from a shot-based Aer run."""
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=betas, gammas=gammas, mixer="x", penalty_free=True
    )
    circuit.measure_all()
    backend = AerSimulator()
    counts = (
        backend.run(transpile(circuit, backend), shots=SHOTS, seed_simulator=SEED)
        .result()
        .get_counts()
    )
    decoded = decode_counts(surrogate, counts, total_shots=SHOTS)
    mean = sum(r.sampling_prob * r.objective for r in decoded)
    spread = math.sqrt(sum(r.sampling_prob * (r.objective - mean) ** 2 for r in decoded))
    return mean, spread / math.sqrt(SHOTS)


def _exact_expectation(surrogate, betas, gammas) -> float:
    energies = build_energy_vector(surrogate, penalty_mode="none")
    n = len(surrogate.variables)
    state = simulate_qaoa(uniform_state(n), energies, betas, gammas, mixer="x")
    return expected_energy_of_state(state, energies)


def test_the_two_engines_agree_on_the_same_circuit(surrogate) -> None:
    aer, standard_error = _aer_expectation(surrogate, BETAS, GAMMAS)
    exact = _exact_expectation(surrogate, BETAS, GAMMAS)
    sigmas = abs(aer - exact) / standard_error
    assert sigmas < 5.0, f"{sigmas:.1f} sigma apart: aer={aer!r} exact={exact!r}"


def test_the_check_fails_when_the_circuit_is_wrong(surrogate) -> None:
    """The negative control. Run Aer on perturbed angles and score it against
    the exact engine's UNPERTURBED value: this is what a genuine
    circuit-construction bug looks like, and the check must reject it."""
    perturbed = (GAMMAS[0] + 0.25, GAMMAS[1])
    aer, standard_error = _aer_expectation(surrogate, BETAS, perturbed)
    exact = _exact_expectation(surrogate, BETAS, GAMMAS)
    sigmas = abs(aer - exact) / standard_error
    assert sigmas >= 5.0, (
        f"a circuit built from the wrong angles passed the cross-check at "
        f"{sigmas:.1f} sigma -- the check cannot detect a broken circuit"
    )


def test_the_exported_circuit_is_the_simulated_one_up_to_global_phase(surrogate) -> None:
    """Shot noise cannot see a phase error that a statevector comparison can, so
    also compare amplitudes directly -- no sampling involved."""
    from qiskit.quantum_info import Statevector

    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=BETAS, gammas=GAMMAS, mixer="x", penalty_free=True
    )
    from_qiskit = np.asarray(Statevector(circuit))
    energies = build_energy_vector(surrogate, penalty_mode="none")
    from_engine = simulate_qaoa(
        uniform_state(len(surrogate.variables)), energies, BETAS, GAMMAS, mixer="x"
    )
    overlap = abs(np.vdot(from_qiskit, from_engine))
    assert overlap == pytest.approx(1.0, abs=1e-9), f"|<qiskit|engine>| = {overlap}"
