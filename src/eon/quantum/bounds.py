"""Bound kinds, separated by type so a heuristic cannot pass for a proof.

A bound that is PROVEN and a bound that is merely ACHIEVED are both one float.
Nothing in the language stops the second being used where the first is
expected, and the result of that mistake is a certificate that looks tight and
guarantees nothing -- the one failure mode this project cannot afford, because
the certificate is what makes the decomposition claim checkable at all.

So the epistemic status lives in the type:

  CertifiedLowerBound   proven by classical optimization duality -- a solver's
                        dual bound, or the dropped-coupling relaxation. Valid
                        regardless of which algorithm produced the solution.
  HeuristicIncumbent    the objective of a feasible plan someone found. A real
                        upper bound on the optimum, but no statement about how
                        far from it.

Pairing them is the only way to build a gap, and the constructor refuses a
mismatch. Compare `separation.jl` in the LANL robust-compilation work, which
splits `HeuristicSeparation` from `CertifiedSeparation` for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CertifiedLowerBound:
    """A lower bound PROVEN by classical optimization duality.

    `source` names the derivation (e.g. "gurobi_dual_bound",
    "dropped_coupling_block_bound") so a record says which proof it rests on.
    """

    value: float
    source: str


@dataclass(frozen=True, slots=True)
class HeuristicIncumbent:
    """The objective of a feasible solution. An upper bound, never a guarantee.

    `source` names what produced it (e.g. "rescored_decomposed_plan").
    """

    value: float
    source: str


@dataclass(frozen=True, slots=True)
class ClassicalDecompositionGap:
    """The distance between what we proved and what we achieved.

    This is the decomposition and integrality gap of the CLASSICAL wrapper. It
    is NOT a quantum-versus-classical advantage measurement, and tightening it
    is a classical workstream. The sentence lives on the type because the gap
    travels into records, figures, and prose where a caption cannot follow it.
    """

    lower: CertifiedLowerBound
    upper: HeuristicIncumbent

    def __post_init__(self) -> None:
        if not isinstance(self.lower, CertifiedLowerBound):
            raise TypeError(
                f"lower must be a CertifiedLowerBound, got {type(self.lower).__name__}; "
                "a value that was achieved rather than proven cannot certify a gap"
            )
        if not isinstance(self.upper, HeuristicIncumbent):
            raise TypeError(
                f"upper must be a HeuristicIncumbent, got {type(self.upper).__name__}; "
                "the upper bound is the objective of a plan we found, not a proof"
            )
        if self.lower.value > self.upper.value + _TOLERANCE:
            raise ValueError(
                f"lower bound {self.lower.value} exceeds upper bound "
                f"{self.upper.value} -- a bug in the bound or the rescore, "
                "not a result"
            )

    @property
    def value(self) -> float:
        return self.upper.value - self.lower.value

    @property
    def relative(self) -> float:
        """Gap as a fraction of |lower bound|; 0.0 when the bound is zero."""
        scale = abs(self.lower.value)
        return self.value / scale if scale > 0 else 0.0


_TOLERANCE = 1e-6
