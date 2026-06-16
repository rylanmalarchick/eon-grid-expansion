from eon.formulations.layer_b import (
    build_layer_b_surrogate,
    solve_layer_b_surrogate,
    validate_layer_b_surrogate,
)
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.formulations.qubo import compile_layer_b_qubo, solve_qubo_with_neal
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_phase1_scenarios


def test_layer_b_surrogate_pipeline_runs() -> None:
    net = load_distribution_feeder("ieee33")
    scenarios = build_phase1_scenarios()
    candidates = generate_candidate_lines(net, "distance_weighted", 3, seed=2)
    config = ExpansionProblemConfig(max_new_lines=1, time_limit_s=3.0)
    layer_a_result = solve_lindistflow_expansion(net, scenarios, candidates, config)

    surrogate = build_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        layer_a_result,
        config,
        neighborhood_size=3,
        evaluation_time_limit_s=2.0,
    )
    assert len(surrogate.variables) == 3

    validation = validate_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        surrogate,
        config,
        top_k=2,
        evaluation_time_limit_s=2.0,
    )
    assert validation.evaluated_plan_count == 8
    assert 0.0 <= validation.top_k_agreement <= 1.0

    gurobi_solution = solve_layer_b_surrogate(surrogate)
    assert gurobi_solution.status in {"OPTIMAL", "TIME_LIMIT"}

    qubo = compile_layer_b_qubo(surrogate)
    annealed_solution = solve_qubo_with_neal(surrogate, qubo, num_reads=32)
    assert len(annealed_solution.toggle_decisions) == 3
