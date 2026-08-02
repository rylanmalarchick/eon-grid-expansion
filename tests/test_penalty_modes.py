"""Two penalty encodings, and they are NOT interchangeable (2026-08-01).

'flat' adds a constant big-M to every infeasible state -- a plateau with no
gradient toward feasibility. 'quadratic' is the standard Hess (sum-K)^2
formulation used by the MPS/TN path. Every P3 number was produced under
'flat'; these tests pin both, and pin that the quadratic vector agrees with
the compiled QUBO the MPS path scores (so the two halves of the project can
finally be compared on one object)."""

from __future__ import annotations

import numpy as np
import pytest

from eon.formulations.qubo import compile_layer_b_qubo_hess
from eon.quantum.energy import CARDINALITY_PENALTY, build_energy_vector
from tests.test_qaoa_baseline import _toy_surrogate


def _compiled_energies(surrogate) -> np.ndarray:
    """Independent ground truth: the compiled QUBO is over TOGGLE variables,
    while a basis-state bit is a BUILD (decode_counts' convention). Substitute
    toggle_i = default_i XOR build_i explicitly -- do NOT assume they match."""
    compilation = compile_layer_b_qubo_hess(surrogate)
    n = len(surrogate.variables)
    states = np.arange(2**n)
    toggle = {}
    for i, variable in enumerate(surrogate.variables):
        build_bit = (states >> i) & 1
        toggle[i] = build_bit if variable.default_value == 0 else 1 - build_bit
    out = np.full(2**n, float(compilation.offset))
    for (left, right), coefficient in compilation.qubo.items():
        i = int(left.split("[")[1].rstrip("]"))
        j = int(right.split("[")[1].rstrip("]"))
        out = out + coefficient * toggle[i] * toggle[j]
    return out


def test_quadratic_mode_matches_the_compiled_qubo() -> None:
    surrogate = _toy_surrogate()
    quadratic = build_energy_vector(surrogate, penalty_mode="quadratic")
    np.testing.assert_allclose(quadratic, _compiled_energies(surrogate), atol=1e-9)


def test_flat_mode_is_the_pre_2026_08_01_default() -> None:
    surrogate = _toy_surrogate()
    default = build_energy_vector(surrogate)
    flat = build_energy_vector(surrogate, penalty_mode="flat")
    np.testing.assert_array_equal(default, flat)
    n = len(surrogate.variables)
    states = np.arange(2**n)
    builds = sum((states >> i) & 1 for i in range(n))
    infeasible = builds > surrogate.max_new_lines
    # Flat: every infeasible state carries the SAME penalty -> a plateau.
    penalties = flat[infeasible] - build_energy_vector(surrogate, penalty_mode="none")[infeasible]
    assert np.allclose(penalties, surrogate.base_objective + CARDINALITY_PENALTY)


def test_encodings_are_not_interchangeable() -> None:
    surrogate = _toy_surrogate()
    flat = build_energy_vector(surrogate, penalty_mode="flat")
    quadratic = build_energy_vector(surrogate, penalty_mode="quadratic")
    difference = flat - quadratic
    assert not np.allclose(difference, difference[0]), (
        "the two penalty encodings differ by more than a constant -- results "
        "under one do not transfer to the other"
    )


def test_feasible_states_agree_across_modes() -> None:
    """Where the constraint is satisfied, all encodings must coincide (up to
    the compiled offset), or the comparison would be meaningless."""
    surrogate = _toy_surrogate()
    n = len(surrogate.variables)
    states = np.arange(2**n)
    builds = sum((states >> i) & 1 for i in range(n))
    feasible = builds <= surrogate.max_new_lines
    flat = build_energy_vector(surrogate, penalty_mode="flat")[feasible]
    none = build_energy_vector(surrogate, penalty_mode="none")[feasible]
    np.testing.assert_allclose(flat, none, atol=1e-9)


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="penalty_mode"):
        build_energy_vector(_toy_surrogate(), penalty_mode="nonsense")


def test_quadratic_mode_correct_for_default_one_variables() -> None:
    """The compiled QUBO is over TOGGLE variables while basis bits are BUILDS;
    they coincide only when default_value == 0. A default-1 variable must not
    silently flip the cost."""
    surrogate = _toy_surrogate(default_one=True)
    quadratic = build_energy_vector(surrogate, penalty_mode="quadratic")
    np.testing.assert_allclose(quadratic, _compiled_energies(surrogate), atol=1e-9)
