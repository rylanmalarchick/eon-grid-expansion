"""Gate-level QAOA circuits: the exportable, Qiskit/Braket-validatable artifact.

The simulation path compiles the cost operator into a single DiagonalGate over
2^n amplitudes. That is fine for the exact numpy engine but it is NOT a
hardware artifact: synthesizing it explodes (the same wall that made
Statevector intractable at n=20), and no transpiler or device ingests it.

This module emits the standard gate-level QAOA circuit instead -- RZ per local
field, RZZ per coupling, then the mixer -- which is what actually runs on
hardware, exports to OpenQASM 3, and can be transpiled to a device basis. It is
equivalence-tested against the numpy engine (tests/test_export.py), so the
exported circuit is provably the same object we benchmarked, up to global
phase.

Convention note: the Ising energies used elsewhere are in BUILD space (bit i of
the basis index = build i). Here bit i maps to qubit i, matching
simulator.py and postprocess.decode_counts.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.qubo import compile_external_qubo, compile_layer_b_qubo_hess
from eon.quantum.cop_qaoa import _target_hamming_weight, _uniform_weight_state
from eon.quantum.mixers import MIXER_NAMES


def build_gate_level_qaoa_circuit(
    surrogate: LayerBSurrogate,
    *,
    betas: tuple[float, ...],
    gammas: tuple[float, ...],
    mixer: str,
    penalty_free: bool = False,
    thetas: list[float] | None = None,
) -> QuantumCircuit:
    """Standard QAOA circuit for the compiled cost operator.

    mixer: 'x' (vanilla), 'warm_start' (R46), or 'xy_ring' (cop-QAOA).
    Uses the SAME compilation as the simulated runs, so depth/gate counts
    describe the benchmarked algorithm rather than a lookalike.
    """
    if mixer not in MIXER_NAMES:
        raise ValueError(f"mixer must be one of {sorted(MIXER_NAMES)}, got {mixer!r}")
    if len(betas) != len(gammas):
        raise ValueError("betas and gammas must have equal length")

    compile_fn = compile_external_qubo if penalty_free else compile_layer_b_qubo_hess
    compilation = compile_fn(surrogate)
    n = len(surrogate.variables)

    # QUBO (over build bits) -> Ising angles. E(x) = sum_ii Q_ii x_i +
    # sum_{i<j} Q_ij x_i x_j; a Z-basis phase e^{-i gamma E} is RZ/RZZ.
    linear: dict[int, float] = dict.fromkeys(range(n), 0.0)
    quadratic: dict[tuple[int, int], float] = {}
    for (left, right), coefficient in compilation.qubo.items():
        i = _toggle_index(left)
        j = _toggle_index(right)
        if i == j:
            linear[i] += coefficient
        else:
            key = (i, j) if i < j else (j, i)
            quadratic[key] = quadratic.get(key, 0.0) + coefficient

    # The compiled QUBO is over TOGGLE variables; a qubit is a BUILD bit.
    # toggle_i = default_i XOR build_i, i.e. z-flip the default-1 qubits. We
    # fold that flip into the rotation signs rather than adding X gates.
    sign = {
        index: (1.0 if variable.default_value == 0 else -1.0)
        for index, variable in enumerate(surrogate.variables)
    }
    offset_shift = {
        index: (0.0 if variable.default_value == 0 else 1.0)
        for index, variable in enumerate(surrogate.variables)
    }
    _ = offset_shift  # constants are global phase; tracked for clarity only

    circuit = QuantumCircuit(n)
    if mixer == "xy_ring":
        circuit.initialize(
            _uniform_weight_state(n, _target_hamming_weight(surrogate)), range(n)
        )
    elif mixer == "warm_start":
        if thetas is None:
            raise ValueError("warm_start mixer requires thetas")
        for qubit, theta in enumerate(thetas):
            circuit.ry(float(theta), qubit)
    else:
        circuit.h(range(n))

    for layer, (beta, gamma) in enumerate(zip(betas, gammas, strict=True)):
        # x_i = (1 - z_i)/2 turns Q_ii x_i into a single-qubit Z rotation and
        # Q_ij x_i x_j into a ZZ rotation plus single-qubit terms.
        z_coefficient = dict.fromkeys(range(n), 0.0)
        for i, coefficient in linear.items():
            z_coefficient[i] -= coefficient / 2.0
        for (i, j), coefficient in quadratic.items():
            z_coefficient[i] -= coefficient / 4.0
            z_coefficient[j] -= coefficient / 4.0
        for (i, j), coefficient in quadratic.items():
            if coefficient != 0.0:
                circuit.rzz(gamma * coefficient * sign[i] * sign[j] / 2.0, i, j)
        for i, coefficient in z_coefficient.items():
            if coefficient != 0.0:
                circuit.rz(2.0 * gamma * coefficient * sign[i], i)

        if mixer == "x":
            for qubit in range(n):
                circuit.rx(-2.0 * beta, qubit)
        elif mixer == "warm_start":
            assert thetas is not None
            for qubit, theta in enumerate(thetas):
                circuit.ry(-float(theta), qubit)
                circuit.rz(-2.0 * beta, qubit)
                circuit.ry(float(theta), qubit)
        else:
            pairs = [(left, left + 1) for left in range(n - 1)] + [(n - 1, 0)]
            for a, b in pairs:
                circuit.rxx(beta, a, b)
                circuit.ryy(beta, a, b)
        circuit.barrier(label=f"layer{layer}")
    return circuit


def _toggle_index(label: str) -> int:
    return int(label.split("[")[1].rstrip("]"))
