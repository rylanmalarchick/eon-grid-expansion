"""One conversion from a counts key to an energy-vector index.

Qiskit writes qubit 0 rightmost. build_energy_vector indexes state s so that
bit i is variable i. Those two agree, so the key is the index in base 2 and no
reversal belongs anywhere. Reversing once looks harmless and silently scores a
different state.
"""

from __future__ import annotations

import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.energy import build_energy_vector
from eon.quantum.postprocess import decode_counts, state_index


@pytest.fixture(scope="module")
def surrogate():
    return build_external_surrogate(generate_fused_planted(10, seed=7, block_size=5, alpha=0.5))


@pytest.mark.parametrize("state", [0, 1, 5, 6, 511, 1023])
def test_the_key_round_trips_to_its_own_state(surrogate, state: int) -> None:
    n = len(surrogate.variables)
    key = format(state, f"0{n}b")
    assert state_index(key) == state


@pytest.mark.parametrize("state", [1, 5, 6, 511])
def test_the_index_agrees_with_the_decoded_builds(surrogate, state: int) -> None:
    """The decoded per-variable builds must match the bits of the index."""
    n = len(surrogate.variables)
    key = format(state, f"0{n}b")
    decoded = decode_counts(surrogate, {key: 1}, total_shots=1)[0]
    index = state_index(key)
    for i, variable in enumerate(surrogate.variables):
        assert decoded.actual_builds[variable.name] == (index >> i) & 1


def test_reversing_the_key_scores_a_different_state(surrogate) -> None:
    """Negative control. If reversal were harmless this helper would not matter,
    and the bug it exists to prevent would be undetectable."""
    n = len(surrogate.variables)
    key = format(5, f"0{n}b")
    energies = build_energy_vector(surrogate, penalty_mode="none")
    assert state_index(key) != int(key[::-1], 2)
    assert energies[state_index(key)] != energies[int(key[::-1], 2)]
