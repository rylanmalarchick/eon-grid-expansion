import gurobipy as gp
from gurobipy import GRB

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    _add_reconfiguration,
    _candidate_to_spec,
    _existing_line_specs,
    solve_lindistflow_expansion,
)
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder, slack_bus_index
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


def test_reconfiguration_enforces_radiality() -> None:
    # White-box: the radiality block alone (no power flow) must yield a spanning
    # tree -- exactly n_buses - 1 closed branches -- and a candidate may close
    # only if built. Power-flow-free so it is instant and deterministic.
    net = load_distribution_feeder("ieee33")
    buses = [int(bus) for bus in net.bus.index]
    slack = slack_bus_index(net)
    base_kv = {int(bus): float(vn) for bus, vn in net.bus["vn_kv"].items()}
    candidates = generate_candidate_lines(net, "community_bridging", 8, seed=7)
    all_lines = _existing_line_specs(net, base_kv) + [
        _candidate_to_spec(candidate, net.sn_mva, base_kv) for candidate in candidates
    ]

    model = gp.Model()
    model.Params.OutputFlag = 0
    build_vars = {
        line.name: model.addVar(vtype=GRB.BINARY, name=f"build[{line.name}]")
        for line in all_lines
        if line.is_candidate
    }
    closed = _add_reconfiguration(model, all_lines, buses, slack, build_vars)
    model.setObjective(0.0)
    model.optimize()

    assert model.SolCount >= 1
    closed_count = sum(1 for var in closed.values() if var.X > 0.5)
    assert closed_count == len(buses) - 1
    for line in all_lines:
        if line.is_candidate and closed[line.name].X > 0.5:
            assert build_vars[line.name].X > 0.5
