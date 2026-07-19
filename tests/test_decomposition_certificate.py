"""D14 certificate: LB <= UB always; on a planted instance the pair must
sandwich the known exact optimum (LB <= planted <= UB)."""

from __future__ import annotations

import pytest

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.decomposition import (
    decompose_surrogate,
    dropped_coupling_lower_bound,
    solve_decomposed_qaoa,
)


def test_certificate_sandwiches_planted_optimum() -> None:
    instance = generate_fused_planted(12, seed=7, block_size=4, alpha=0.5)
    surrogate = build_external_surrogate(instance)
    solution, certificate = solve_decomposed_qaoa(surrogate, block_size=4, p=1, shots=256)

    assert certificate.lower_bound <= certificate.upper_bound + 1e-9
    planted = float(instance.planted_energy)
    assert certificate.lower_bound <= planted + 1e-9
    assert planted <= certificate.upper_bound + 1e-9
    assert certificate.gap == pytest.approx(
        certificate.upper_bound - certificate.lower_bound
    )
    assert solution.objective_value == pytest.approx(certificate.upper_bound)
    assert solution.status == "DECOMPOSED_QAOA"
    # The merged plan is a real assignment: rescoring it reproduces the UB.
    assert surrogate.surrogate_objective(solution.toggle_decisions) == pytest.approx(
        certificate.upper_bound
    )


def test_gurobi_bound_tightens_when_larger() -> None:
    instance = generate_fused_planted(12, seed=24, block_size=4, alpha=0.5)
    surrogate = build_external_surrogate(instance)
    blocks = decompose_surrogate(surrogate, block_size=4)
    block_bound, dropped = dropped_coupling_lower_bound(surrogate, blocks)
    assert dropped > 0  # fusion couples across blocks; some edges must be cut

    tighter = block_bound + 0.5
    _, certificate = solve_decomposed_qaoa(
        surrogate, block_size=4, p=1, shots=128, gurobi_best_bound=tighter
    )
    assert certificate.lower_bound == pytest.approx(tighter)
    assert certificate.lower_bound_source == "gurobi_best_bound"

    _, certificate_loose = solve_decomposed_qaoa(
        surrogate, block_size=4, p=1, shots=128, gurobi_best_bound=block_bound - 100.0
    )
    assert certificate_loose.lower_bound == pytest.approx(block_bound)
    assert certificate_loose.lower_bound_source == "dropped_coupling_block_bound"


def test_impossible_bound_raises_certificate_violation() -> None:
    instance = generate_fused_planted(12, seed=7, block_size=4, alpha=0.5)
    surrogate = build_external_surrogate(instance)
    with pytest.raises(RuntimeError, match="certificate violation"):
        solve_decomposed_qaoa(
            surrogate, block_size=4, p=1, shots=128, gurobi_best_bound=1e12
        )


def _surrogate_with_outside_fixed() -> object:
    from dataclasses import replace

    from eon.instances.external import build_external_surrogate, generate_fused_planted

    instance = generate_fused_planted(8, seed=7, block_size=4, alpha=0.5)
    surrogate = build_external_surrogate(instance)
    # Two lines OUTSIDE the variable set that Layer A fixed: one built, one not.
    # Tight budget so the greedy repair must fire.
    return replace(
        surrogate,
        fixed_builds={**surrogate.fixed_builds, "fx_built": 1, "fx_unbuilt": 0},
        max_new_lines=2,
    )


def test_repair_never_touches_outside_fixed_builds() -> None:
    surrogate = _surrogate_with_outside_fixed()
    solution, certificate = solve_decomposed_qaoa(surrogate, block_size=4, p=1, shots=128)
    assert solution.actual_builds["fx_built"] == 1, (
        "repair must not unbuild a line Layer A fixed outside the surrogate"
    )
    assert solution.actual_builds["fx_unbuilt"] == 0, (
        "repair must not build a line Layer A fixed to 0 outside the surrogate"
    )
    # The variable budget shrinks by the outside built count.
    variable_builds = sum(
        v for k, v in solution.actual_builds.items() if k not in ("fx_built", "fx_unbuilt")
    )
    assert variable_builds + 1 <= surrogate.max_new_lines
    assert certificate.lower_bound <= certificate.upper_bound + 1e-9


def test_repair_ranking_is_build_space_for_default_one() -> None:
    from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
    from eon.instances.candidate_lines import CandidateLine

    def _var(name: str, default: int) -> LayerBVariable:
        candidate = CandidateLine(
            name=name,
            from_bus=0,
            to_bus=1,
            family="toy",
            length_km=1.0,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.3,
            max_i_ka=0.4,
            build_cost=1.0,
        )
        return LayerBVariable(
            name=name, candidate=candidate, default_value=default, stress_score=1.0
        )

    # Building a (default 0, linear -10) lowers the objective by 10.
    # Building c (default 1, linear -50) RAISES it by 50 (toggle goes 1 -> 0).
    # The old toggle-space ranking picked c (most negative linear) -- wrong.
    surrogate = LayerBSurrogate(
        variables=(_var("a", 0), _var("b", 0), _var("c", 1)),
        offset=0.0,
        linear={"a": -10.0, "b": -1.0, "c": -50.0},
        quadratic={},
        fixed_builds={},
        max_new_lines=1,
        base_objective=0.0,
        base_selected_count=1,
    )
    solution, _ = solve_decomposed_qaoa(surrogate, block_size=3, p=1, shots=128)
    assert solution.actual_builds["a"] == 1, "repair must keep the beneficial build"
    assert solution.actual_builds["c"] == 0, (
        "repair must drop the default-1 build whose construction raises the objective"
    )


def test_certificate_records_repair_transparency() -> None:
    surrogate = _surrogate_with_outside_fixed()
    _, certificate = solve_decomposed_qaoa(surrogate, block_size=4, p=1, shots=128)
    assert certificate.repair_applied in (True, False)
    assert certificate.builds_changed_by_repair >= 0
