import pytest

from eon.mps.juliqaoa_smoke import _synthetic_surrogate
from eon.mps.protocol import run_mps_protocol


@pytest.mark.requires_gurobi
def test_mps_protocol_smoke_runs() -> None:
    surrogate = _synthetic_surrogate("easy_path_n8")
    result = run_mps_protocol(
        surrogate,
        instance_name="easy_path_n8",
        chi_values=(4,),
        chi_limit=4,
        angle_iterations=2,
        sample_count=2,
    )
    assert result.backend == "juliqaoa_mps"
    assert result.chi_max_reached == 4
    assert len(result.ordering_results) >= 2
    assert result.tree_tn_result is not None
    assert result.tree_tn_result.backend == "generic_tn_tropical"
    assert result.tree_tn_result.within_budget is True
    # Two independent exact methods on the same compiled QUBO must agree: JuliQAOA's
    # brute-force enumeration (exact_ground_energy) and GTN's tropical contraction
    # (tree control best_energy). This cross-validates the GTN backend end to end.
    # Both are exact computations of the same QUBO ground energy, so they must agree
    # to ~machine precision, not the loose default rel=1e-6 (CLAUDE.md tolerance rule).
    assert result.tree_tn_result.best_energy == pytest.approx(
        result.exact_ground_energy, rel=1e-9, abs=1e-9
    )
    assert result.exact_ground_energy <= min(
        ordering.best_energy for ordering in result.ordering_results
    )
