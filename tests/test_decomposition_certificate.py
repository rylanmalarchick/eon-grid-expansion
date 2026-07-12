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
