from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_phase1_scenarios


def test_small_lindistflow_solve_runs() -> None:
    net = load_distribution_feeder("ieee33")
    scenarios = build_phase1_scenarios()
    candidates = generate_candidate_lines(net, "distance_weighted", 4, seed=3)
    config = ExpansionProblemConfig(max_new_lines=1, time_limit_s=5.0)
    result = solve_lindistflow_expansion(net, scenarios, candidates, config)
    assert result.termination_status in {"OPTIMAL", "TIME_LIMIT"}
    assert result.objective_value < float("inf")
    assert len(result.build_decisions) == len(candidates)
