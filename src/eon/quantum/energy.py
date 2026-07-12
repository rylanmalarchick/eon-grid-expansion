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

CARDINALITY_PENALTY = 10_000_000.0


def build_energy_vector(surrogate: LayerBSurrogate) -> np.ndarray:
    """Energies of all 2^n basis states, vectorized (the per-state Python loop
    is unusable at n=20). Basis state s encodes actual builds: build_i = bit i
    of s; toggle_i = default_i XOR build_i; energy = surrogate_objective(t)
    (+ penalty where the build count exceeds max_new_lines)."""
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
    penalty = surrogate.base_objective + CARDINALITY_PENALTY
    energies[build_counts > surrogate.max_new_lines] += penalty
    return energies
