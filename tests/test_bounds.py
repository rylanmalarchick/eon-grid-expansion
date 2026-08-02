"""A heuristic value must not be usable where a proof is expected.

The D14 certificate pairs a lower bound that is PROVEN (classical optimization
duality: a Gurobi dual bound, or the dropped-coupling relaxation) against an
upper bound that is merely ACHIEVED (the objective of a plan the wrapper
found). Those two have different epistemic status and identical Python types,
so nothing stopped an incumbent objective being passed in as the dual bound --
which yields a certificate that looks tight and proves nothing. The existing
guard only fires when the lower bound exceeds the upper bound, so a wrong bound
that still sits below the incumbent passes silently.

These tests pin that the distinction lives in the type system.
"""

from __future__ import annotations

import pytest

from eon.quantum.bounds import (
    CertifiedLowerBound,
    ClassicalDecompositionGap,
    HeuristicIncumbent,
)


def test_a_certified_bound_and_an_incumbent_are_different_types() -> None:
    proven = CertifiedLowerBound(value=-10.0, source="gurobi_dual_bound")
    achieved = HeuristicIncumbent(value=-4.0, source="rescored_decomposed_plan")
    assert not isinstance(achieved, CertifiedLowerBound)
    assert not isinstance(proven, HeuristicIncumbent)


def test_an_incumbent_cannot_be_used_as_the_certified_lower_bound() -> None:
    """The failure this exists to prevent: someone passes the objective of a
    feasible solution where a dual bound is expected. It is numerically
    plausible -- below the incumbent, above the true bound -- so no value check
    catches it."""
    achieved = HeuristicIncumbent(value=-6.0, source="some_heuristic_run")
    with pytest.raises(TypeError, match="CertifiedLowerBound"):
        ClassicalDecompositionGap(
            lower=achieved,  # type: ignore[arg-type]
            upper=HeuristicIncumbent(value=-4.0, source="rescored_decomposed_plan"),
        )


def test_a_certified_bound_cannot_be_used_as_the_incumbent() -> None:
    proven = CertifiedLowerBound(value=-10.0, source="dropped_coupling_block_bound")
    with pytest.raises(TypeError, match="HeuristicIncumbent"):
        ClassicalDecompositionGap(
            lower=proven,
            upper=proven,  # type: ignore[arg-type]
        )


def test_the_gap_is_upper_minus_lower() -> None:
    gap = ClassicalDecompositionGap(
        lower=CertifiedLowerBound(value=-10.0, source="dropped_coupling_block_bound"),
        upper=HeuristicIncumbent(value=-4.0, source="rescored_decomposed_plan"),
    )
    assert gap.value == pytest.approx(6.0)
    assert gap.relative == pytest.approx(6.0 / 10.0)


def test_a_lower_bound_above_the_incumbent_is_rejected() -> None:
    """Still a bug, still caught -- the type split adds a guard, it does not
    replace the arithmetic one."""
    with pytest.raises(ValueError, match="exceeds"):
        ClassicalDecompositionGap(
            lower=CertifiedLowerBound(value=1.0, source="gurobi_dual_bound"),
            upper=HeuristicIncumbent(value=-4.0, source="rescored_decomposed_plan"),
        )


def test_the_gap_states_what_it_is_not() -> None:
    """The gap is a classical decomposition/integrality gap. The one thing a
    reader must never do is read it as a quantum-vs-classical result, so the
    type carries that sentence with it rather than relying on prose elsewhere."""
    assert "not" in ClassicalDecompositionGap.__doc__.lower()
    assert "advantage" in ClassicalDecompositionGap.__doc__.lower()
