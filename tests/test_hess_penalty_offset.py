"""The Hess at-most-K encoding is a plain (sum-K)^2 penalty -- equality-biased,
NOT max(0, sum-K)^2. Consequence: a constrained optimum with s* < K builds
carries exactly penalty * (K - s*)^2 in the compiled QUBO (the "+50000" offset
observed on every IEEE 33 record, root-caused 2026-07-31). These tests pin
that arithmetic so the offset stays a documented property, not an anomaly."""

from __future__ import annotations

from itertools import product

import pytest

from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
from eon.formulations.qubo import compile_layer_b_qubo_hess
from eon.instances.candidate_lines import CandidateLine


def _underbuild_surrogate() -> LayerBSurrogate:
    """K=2 but only one build is profitable: the constrained optimum has s*=1."""
    candidates = [
        CandidateLine(
            name=f"u{i}",
            from_bus=i,
            to_bus=(i + 1) % 4,
            family="toy",
            length_km=1.0,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.3,
            max_i_ka=0.4,
            build_cost=1.0,
        )
        for i in range(4)
    ]
    variables = tuple(
        LayerBVariable(name=c.name, candidate=c, default_value=0, stress_score=1.0)
        for c in candidates
    )
    return LayerBSurrogate(
        variables=variables,
        offset=3.0,
        linear={"u0": -10.0, "u1": 2.0, "u2": 5.0, "u3": 4.0},
        quadratic={("u0", "u2"): 1.5},
        fixed_builds={},
        max_new_lines=2,
        base_objective=3.0,
        base_selected_count=1,
    )


def _compiled_energy(compilation, bits: tuple[int, ...]) -> float:
    total = compilation.offset
    for (left, right), coefficient in compilation.qubo.items():
        li = int(left.split("[")[1].rstrip("]"))
        ri = int(right.split("[")[1].rstrip("]"))
        total += coefficient * bits[li] * bits[ri]
    return total


def test_compiled_ground_is_min_of_penalized_objective() -> None:
    """Compiled ground == min over ALL states of unpenalized + penalty*(K-s)^2.
    This is the exact semantics of the (sum-K)^2 encoding, and the source of
    the +50000 offset on the IEEE 33 records (their 2-build optimum carries
    (3-2)^2 * 50000 and no 3-build plan comes within 50000)."""
    surrogate = _underbuild_surrogate()
    penalty = 50_000.0
    compilation = compile_layer_b_qubo_hess(surrogate, penalty_strength=penalty)

    expected_ground = float("inf")
    for bits in product((0, 1), repeat=4):
        toggles = {v.name: b for v, b in zip(surrogate.variables, bits, strict=True)}
        value = surrogate.surrogate_objective(toggles)
        value += penalty * (surrogate.max_new_lines - sum(bits)) ** 2
        expected_ground = min(expected_ground, value)

    compiled_ground = min(
        _compiled_energy(compilation, bits) for bits in product((0, 1), repeat=4)
    )
    assert compiled_ground == pytest.approx(expected_ground, rel=1e-12)


def test_equality_bias_can_shift_the_argmin() -> None:
    """The encoding is NOT the at-most-K problem: on this toy the true
    constrained optimum builds 1 line (objective -7), but the compiled ground
    is the WORSE 2-build state (-5) because it dodges the (K-s)^2 penalty.
    Documented bias; comparisons must therefore stay within one compiled
    object (the quarantine rule in PLAN.txt, 2026-07-11)."""
    surrogate = _underbuild_surrogate()
    compilation = compile_layer_b_qubo_hess(surrogate, penalty_strength=50_000.0)

    constrained_optimum = min(
        surrogate.surrogate_objective(
            {v.name: b for v, b in zip(surrogate.variables, bits, strict=True)}
        )
        for bits in product((0, 1), repeat=4)
        if sum(bits) <= surrogate.max_new_lines
    )
    compiled_ground = min(
        _compiled_energy(compilation, bits) for bits in product((0, 1), repeat=4)
    )
    assert constrained_optimum == pytest.approx(-7.0)
    assert compiled_ground == pytest.approx(-5.0)
    assert compiled_ground != pytest.approx(constrained_optimum)
