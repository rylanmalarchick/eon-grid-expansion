"""The Lean theorem and the Python function must be about the same object.

lean/DroppedCouplingBound.lean proves that

    offset + sum(block minima) + sum(min(0, J)) <= energy(x)   for every x

so the computed value is a valid lower bound on the global minimum. A proof of
a theorem the code does not implement is worth nothing, so this test brute-forces
the other side: enumerate EVERY assignment of a small surrogate and check the
implementation's bound never exceeds the true minimum.

The negative control matters as much as the positive one. Replacing min(0, J)
with 0 -- i.e. assuming dropped couplings can only help -- is the plausible
wrong version of this bound, and on an instance with negative couplings it must
produce a value that exceeds the true minimum. If it does not, this test is not
sensitive enough to detect a broken bound.
"""

from __future__ import annotations

from itertools import product

import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.decomposition import decompose_surrogate, dropped_coupling_lower_bound


def _true_minimum(surrogate) -> float:
    """Exhaustive minimum over toggle space, cardinality relaxed -- the same
    relaxation the block minima use, so the comparison is like-for-like."""
    names = [variable.name for variable in surrogate.variables]
    best = float("inf")
    for assignment in product((0, 1), repeat=len(names)):
        toggles = dict(zip(names, assignment, strict=True))
        value = surrogate.offset
        value += sum(surrogate.linear[name] * toggles[name] for name in names)
        value += sum(
            coefficient * toggles[left] * toggles[right]
            for (left, right), coefficient in surrogate.quadratic.items()
        )
        best = min(best, value)
    return best


@pytest.mark.parametrize("seed", [7, 24, 42])
@pytest.mark.parametrize("block_size", [3, 4])
def test_bound_never_exceeds_the_true_minimum(seed: int, block_size: int) -> None:
    surrogate = build_external_surrogate(
        generate_fused_planted(12, seed=seed, block_size=4, alpha=0.5)
    )
    blocks = decompose_surrogate(surrogate, block_size=block_size)
    bound, dropped = dropped_coupling_lower_bound(surrogate, blocks)
    truth = _true_minimum(surrogate)
    assert bound <= truth + 1e-9, (
        f"bound {bound} exceeds the true minimum {truth}: the certificate would "
        f"claim a tighter gap than reality ({dropped} couplings dropped)"
    )


def test_dropping_the_negative_floor_breaks_the_bound() -> None:
    """Negative control: the min(0, J) term is load-bearing, not decorative."""
    surrogate = build_external_surrogate(
        generate_fused_planted(12, seed=7, block_size=4, alpha=0.5)
    )
    blocks = decompose_surrogate(surrogate, block_size=3)
    in_block: set[tuple[str, str]] = set()
    for block in blocks:
        in_block.update(block.quadratic.keys())
    negative_dropped = [
        coefficient
        for key, coefficient in surrogate.quadratic.items()
        if key not in in_block and coefficient < 0
    ]
    assert negative_dropped, "instance has no negative dropped coupling to test with"

    correct, _ = dropped_coupling_lower_bound(surrogate, blocks)
    naive = correct - sum(negative_dropped)  # the version that floors at 0
    truth = _true_minimum(surrogate)
    assert naive > truth + 1e-9, (
        "flooring dropped couplings at 0 still produced a valid bound, so this "
        "test cannot distinguish the correct bound from the broken one"
    )
