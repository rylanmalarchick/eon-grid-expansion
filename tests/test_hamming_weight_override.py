"""The constrained subspace must be settable, not just derivable.

The S3-large experiment asks whether cop-QAOA beats random sampling when the
feasible subspace is much larger than the shot budget. That needs a bigger
Hamming weight. Raising the build budget does NOT give one: the target weight is
min(n, max_new_lines, base_selected_count), and base_selected_count is however
many lines the Layer A incumbent happened to select. A run with
--max-new-lines 6 therefore produced the identical 190-state weight-2 subspace
and answered nothing, twice, over several hours.
"""

from __future__ import annotations

import math

import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.cop_qaoa import _target_hamming_weight, _uniform_weight_state


@pytest.fixture(scope="module")
def surrogate():
    return build_external_surrogate(generate_fused_planted(12, seed=7, block_size=4, alpha=0.5))


def test_derived_weight_is_clamped_by_the_incumbent(surrogate) -> None:
    """Baseline behaviour, kept: without an override the weight comes from the
    Layer A incumbent, so a larger budget alone changes nothing."""
    derived = _target_hamming_weight(surrogate)
    from dataclasses import replace

    roomier = replace(surrogate, max_new_lines=surrogate.max_new_lines + 5)
    assert _target_hamming_weight(roomier) == derived


def test_an_explicit_override_sets_the_weight(surrogate) -> None:
    n = len(surrogate.variables)
    assert _target_hamming_weight(surrogate, override=4) == 4
    assert _target_hamming_weight(surrogate, override=n) == n


def test_the_override_is_clamped_to_the_variable_count(surrogate) -> None:
    """A weight above n has an EMPTY subspace, which would silently produce a
    zero state vector rather than an error."""
    n = len(surrogate.variables)
    assert _target_hamming_weight(surrogate, override=n + 10) == n


@pytest.mark.parametrize("weight", [1, 2, 4])
def test_the_subspace_has_the_size_the_weight_implies(surrogate, weight: int) -> None:
    n = len(surrogate.variables)
    state = _uniform_weight_state(n, weight)
    occupied = int((abs(state) > 0).sum())
    assert occupied == math.comb(n, weight), (
        f"weight {weight} should span C({n},{weight}) = {math.comb(n, weight)} states"
    )


def test_a_nonpositive_override_is_rejected(surrogate) -> None:
    with pytest.raises(ValueError, match="positive"):
        _target_hamming_weight(surrogate, override=0)
