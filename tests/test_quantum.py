from eon.formulations.layer_b import build_layer_b_surrogate
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_phase1_scenarios
from eon.quantum.cop_qaoa import build_p1_landscape, run_constrained_qaoa_subproblem


def test_constrained_qaoa_smoke_runs() -> None:
    net = load_distribution_feeder("ieee33")
    scenarios = build_phase1_scenarios()
    candidates = generate_candidate_lines(net, "distance_weighted", 3, seed=4)
    config = ExpansionProblemConfig(max_new_lines=1, time_limit_s=3.0)
    layer_a = solve_lindistflow_expansion(net, scenarios, candidates, config)
    surrogate = build_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        layer_a,
        config,
        neighborhood_size=3,
        evaluation_time_limit_s=2.0,
    )
    landscape = build_p1_landscape(surrogate)
    assert len(landscape.energies) == len(landscape.betas)

    run = run_constrained_qaoa_subproblem(surrogate, p=1, shots=128)
    assert run.best_sample.sampling_prob > 0.0
