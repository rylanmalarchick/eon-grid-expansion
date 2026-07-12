"""Exact numpy statevector simulation for the structured QAOA circuits.

Qiskit's Statevector.from_instruction synthesizes a 2^n-entry DiagonalGate
into elementary gates before applying it, which is intractable at n=20. The
QAOA circuits here are all (initial product/Dicke state) + per-layer
(diagonal phase, then a product mixer or an XY ring) -- each layer is O(2^n)
numpy work applied directly to the amplitude vector. Cross-validated against
Qiskit at small n in tests/test_qaoa_simulator.py; the Qiskit circuit
builders remain the exportable (Qiskit-validatable) artifact.

Qubit convention matches the rest of the stack: qubit i = bit i of the basis
state index (little-endian), i.e. axis i after reshaping to (2,)*n in
REVERSED axis order. Internally we reshape so that axis k addresses qubit
(n-1-k), matching numpy's C-order."""

from __future__ import annotations

import math

import numpy as np


def uniform_state(n: int) -> np.ndarray:
    return np.full(2**n, 1.0 / math.sqrt(2**n), dtype=complex)


def product_state(thetas: np.ndarray) -> np.ndarray:
    """|phi> = prod_i R_Y(theta_i)|0>: amplitude of basis state s is
    prod_i (cos(theta_i/2) if bit_i(s)=0 else sin(theta_i/2))."""
    n = len(thetas)
    state = np.ones(1, dtype=complex)
    # kron(a, b) makes b the fast axis; iterating qubit 0 first keeps qubit 0
    # the least significant bit (each later qubit becomes the new slow axis).
    for theta in thetas:
        qubit = np.asarray([math.cos(theta / 2.0), math.sin(theta / 2.0)], dtype=complex)
        state = np.kron(qubit, state)
    assert state.shape == (2**n,)
    return state


def apply_diagonal_phase(state: np.ndarray, energies: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-1j * gamma * energies) * state


def _apply_single_qubit(state: np.ndarray, qubit: int, matrix: np.ndarray) -> np.ndarray:
    n = int(math.log2(len(state)))
    tensor = state.reshape((2,) * n)
    axis = n - 1 - qubit  # C-order: last axis is qubit 0
    tensor = np.moveaxis(tensor, axis, 0)
    tensor = np.tensordot(matrix, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(2**n)


def apply_x_mixer(state: np.ndarray, beta: float) -> np.ndarray:
    """exp(+i beta X) on every qubit (= RX(-2 beta), the vanilla mixer)."""
    matrix = np.asarray(
        [
            [math.cos(beta), 1j * math.sin(beta)],
            [1j * math.sin(beta), math.cos(beta)],
        ],
        dtype=complex,
    )
    for qubit in range(int(math.log2(len(state)))):
        state = _apply_single_qubit(state, qubit, matrix)
    return state


def apply_warm_start_mixer(state: np.ndarray, beta: float, thetas: np.ndarray) -> np.ndarray:
    """R46 WS mixer per qubit: R_Y(theta) R_Z(-2 beta) R_Y(-theta)."""

    def ry(theta: float) -> np.ndarray:
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return np.asarray([[c, -s], [s, c]], dtype=complex)

    rz = np.asarray(
        [[np.exp(1j * beta), 0.0], [0.0, np.exp(-1j * beta)]], dtype=complex
    )  # R_Z(-2 beta) = diag(e^{i beta}, e^{-i beta})
    for qubit, theta in enumerate(thetas):
        matrix = ry(float(theta)) @ rz @ ry(-float(theta))
        state = _apply_single_qubit(state, qubit, matrix)
    return state


def apply_xy_ring_mixer(state: np.ndarray, beta: float) -> np.ndarray:
    """XXPlusYYGate(2 beta) on (0,1), (1,2), ..., (n-2, n-1), then (n-1, 0) --
    the cop-QAOA Hamming-weight-preserving ring, applied in the same order as
    build_constrained_qaoa_circuit."""
    n = int(math.log2(len(state)))
    if n <= 1:
        return state
    pairs = [(left, left + 1) for left in range(n - 1)] + [(n - 1, 0)]
    for a, b in pairs:
        state = _apply_xxplusyy(state, a, b, beta)
    return state


def _apply_xxplusyy(state: np.ndarray, a: int, b: int, beta: float) -> np.ndarray:
    """Qiskit XXPlusYYGate(2 beta, 0): acts on the {|01>, |10>} subspace as
    [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]] (|00>, |11> fixed).
    Qiskit order: qubit a is the LOW qubit of the pair (matrix basis |q_b q_a>)."""
    n = int(math.log2(len(state)))
    tensor = state.reshape((2,) * n)
    axis_a = n - 1 - a
    axis_b = n - 1 - b
    # Bring the two qubit axes to the front: (q_a, q_b, rest).
    tensor = np.moveaxis(tensor, (axis_a, axis_b), (0, 1))
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    # |01>: q_a=1, q_b=0 -> tensor[1,0]; |10>: q_a=0, q_b=1 -> tensor[0,1].
    amp_01 = tensor[1, 0].copy()
    amp_10 = tensor[0, 1].copy()
    tensor[1, 0] = cos_b * amp_01 - 1j * sin_b * amp_10
    tensor[0, 1] = -1j * sin_b * amp_01 + cos_b * amp_10
    tensor = np.moveaxis(tensor, (0, 1), (axis_a, axis_b))
    return tensor.reshape(2**n)


def simulate_qaoa(
    initial_state: np.ndarray,
    energies: np.ndarray,
    betas: tuple[float, ...],
    gammas: tuple[float, ...],
    *,
    mixer: str,
    thetas: np.ndarray | None = None,
) -> np.ndarray:
    """Run the layered QAOA circuit exactly; mixer in {x, warm_start, xy_ring}."""
    state = initial_state
    for beta, gamma in zip(betas, gammas, strict=True):
        state = apply_diagonal_phase(state, energies, gamma)
        if mixer == "x":
            state = apply_x_mixer(state, beta)
        elif mixer == "warm_start":
            if thetas is None:
                raise ValueError("warm_start mixer requires thetas")
            state = apply_warm_start_mixer(state, beta, thetas)
        elif mixer == "xy_ring":
            state = apply_xy_ring_mixer(state, beta)
        else:
            raise ValueError(f"unknown mixer {mixer!r}")
    return state


def expected_energy_of_state(state: np.ndarray, energies: np.ndarray) -> float:
    return float(np.real(np.dot(np.abs(state) ** 2, energies)))


def sample_counts(state: np.ndarray, shots: int, *, seed: int) -> dict[str, int]:
    """Sample basis states; keys are Qiskit-style bitstrings (qubit 0 = LAST
    character), matching Statevector.sample_counts / decode_counts."""
    n = int(math.log2(len(state)))
    probabilities = np.abs(state) ** 2
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(state), size=shots, p=probabilities)
    counts: dict[str, int] = {}
    for index in sampled:
        key = format(int(index), f"0{n}b")
        counts[key] = counts.get(key, 0) + 1
    return counts
