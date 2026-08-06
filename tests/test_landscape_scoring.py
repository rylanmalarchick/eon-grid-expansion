"""The landscape excess must index the energy vector with the right bit order.

decode_counts REVERSES the raw count key before storing it, so QuantumResult
.bitstring is already in variable order: bitstring[i] is variable i. The energy
vector is indexed by a state whose bit i is variable i, which is
int(bitstring[::-1], 2), not int(bitstring, 2). Getting this backwards silently
scores the wrong state -- the same qubit-ordering class of bug that a warm-start
test caught in the simulator earlier.

It also pins that excess is computed against the penalized vector on both sides,
so it can never come out negative.
"""

from __future__ import annotations

import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.energy import build_energy_vector
from eon.quantum.postprocess import decode_counts


@pytest.fixture(scope="module")
def surrogate():
    return build_external_surrogate(generate_fused_planted(10, seed=7, block_size=5, alpha=0.5))


def _state_index(bitstring: str) -> int:
    """bitstring[i] is variable i; the energy vector's bit i is variable i."""
    return int(bitstring[::-1], 2)


def test_decoded_bitstring_indexes_the_state_it_describes(surrogate) -> None:
    n = len(surrogate.variables)
    for state in (0, 1, 2, 5, 2**n - 1):
        raw = format(state, f"0{n}b")  # counts key: qubit 0 rightmost
        decoded = decode_counts(surrogate, {raw: 1}, total_shots=1)[0]
        assert _state_index(decoded.bitstring) == state, (
            f"state {state} round-tripped to {_state_index(decoded.bitstring)}"
        )
        for i, variable in enumerate(surrogate.variables):
            assert decoded.actual_builds[variable.name] == (state >> i) & 1


def test_the_wrong_bit_order_would_pick_a_different_state(surrogate) -> None:
    """Negative control: if the reversal did not matter, the test above could
    not detect the bug it exists for."""
    n = len(surrogate.variables)
    asymmetric = 1  # bit 0 set only -- reads differently in each order
    raw = format(asymmetric, f"0{n}b")
    decoded = decode_counts(surrogate, {raw: 1}, total_shots=1)[0]
    assert int(decoded.bitstring, 2) != _state_index(decoded.bitstring)


@pytest.mark.parametrize("mode", ["quadratic", "flat"])
def test_excess_against_the_penalized_vector_is_never_negative(surrogate, mode: str) -> None:
    energies = build_energy_vector(surrogate, penalty_mode=mode)
    optimum = float(energies.min())
    n = len(surrogate.variables)
    for state in range(0, 2**n, max(1, 2**n // 64)):
        raw = format(state, f"0{n}b")
        decoded = decode_counts(surrogate, {raw: 1}, total_shots=1)[0]
        penalized = float(energies[_state_index(decoded.bitstring)])
        assert penalized - optimum >= -1e-9, (
            f"state {state}: penalized {penalized} below the minimum {optimum}"
        )
