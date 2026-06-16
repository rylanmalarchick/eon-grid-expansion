from eon.mps.juliqaoa_smoke import _synthetic_surrogate
from eon.mps.protocol import run_mps_protocol


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
    assert result.tree_tn_result.backend == "treewidth_dp_control"
    assert result.tree_tn_result.chi_max_reached == 4
    assert result.exact_ground_energy <= min(
        ordering.best_energy for ordering in result.ordering_results
    )
