"""Shared, vectorized QAOA cost machinery.

One energy vector serves both QAOA variants (cop constrained-mixer and vanilla
penalty), so the comparison isolates the mixer/initial state: identical cost,
identical scoring. Semantics match LayerBSurrogate.surrogate_objective plus the
big-M cardinality penalty the constrained mixer never needs but the penalty
path relies on (qubit i = surrogate.variables[i] = bit i of the basis-state
index)."""

from __future__ import annotations

import numpy as np

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.qubo import compile_layer_b_qubo_hess

CARDINALITY_PENALTY = 10_000_000.0


def build_energy_vector(
    surrogate: LayerBSurrogate, *, penalty_mode: str = "flat"
) -> np.ndarray:
    """Energies of all 2^n basis states, vectorized (the per-state Python loop
    is unusable at n=20). Basis state s encodes actual builds: build_i = bit i
    of s; toggle_i = default_i XOR build_i.

    penalty_mode selects how the at-most-K constraint enters the cost:

      "flat"      surrogate_objective + (base_objective + CARDINALITY_PENALTY)
                  on every infeasible state. A PLATEAU: all violations cost the
                  same, so a variational landscape has no gradient toward
                  feasibility. This is the pre-2026-08-01 default and the
                  encoding every P3 number was produced under; kept as the
                  default so those records stay reproducible.
      "quadratic" the standard Hess (sum - K)^2 penalty -- the same object the
                  MPS/TN path scores (compile_layer_b_qubo_hess), and the FAIR
                  baseline for penalty-QAOA.
      "none"      no cardinality penalty (feasibility enforced elsewhere, e.g.
                  by the cop-QAOA mixer).
    """
    if penalty_mode not in {"flat", "quadratic", "none"}:
        raise ValueError(
            f"penalty_mode must be 'flat', 'quadratic' or 'none', got {penalty_mode!r}"
        )
    n = len(surrogate.variables)
    states = np.arange(2**n, dtype=np.uint64)
    builds = np.empty((n, 2**n), dtype=np.float64)
    toggles = np.empty((n, 2**n), dtype=np.float64)
    index = {variable.name: i for i, variable in enumerate(surrogate.variables)}
    for i, variable in enumerate(surrogate.variables):
        bit = ((states >> np.uint64(i)) & np.uint64(1)).astype(np.float64)
        builds[i] = bit
        toggles[i] = bit if variable.default_value == 0 else 1.0 - bit

    energies = np.full(2**n, float(surrogate.offset))
    for variable in surrogate.variables:
        coefficient = surrogate.linear[variable.name]
        if coefficient != 0.0:
            energies += coefficient * toggles[index[variable.name]]
    for (left, right), coefficient in surrogate.quadratic.items():
        if coefficient != 0.0:
            energies += coefficient * toggles[index[left]] * toggles[index[right]]

    outside_selected = sum(
        value
        for name, value in surrogate.fixed_builds.items()
        if name not in surrogate.variable_names
    )
    build_counts = builds.sum(axis=0) + outside_selected

    if penalty_mode == "none":
        return energies
    if penalty_mode == "flat":
        penalty = surrogate.base_objective + CARDINALITY_PENALTY
        energies[build_counts > surrogate.max_new_lines] += penalty
        return energies

    # "quadratic": reproduce the compiled Hess object exactly, so the QAOA and
    # MPS halves of the project score ONE cost function. The compiled QUBO is
    # over TOGGLE variables while a basis bit is a BUILD -- they coincide only
    # for default_value == 0, so substitute toggles explicitly.
    compilation = compile_layer_b_qubo_hess(surrogate)
    quadratic_energies = np.full(2**n, float(compilation.offset))
    for (left, right), coefficient in compilation.qubo.items():
        i = _toggle_index(left)
        j = _toggle_index(right)
        quadratic_energies += coefficient * toggles[i] * toggles[j]
    return quadratic_energies


def _toggle_index(label: str) -> int:
    return int(label.split("[")[1].rstrip("]"))
