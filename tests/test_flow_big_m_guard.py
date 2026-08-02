"""flow_big_m_mva is a REPRESENTATIONAL bound on line-flow variables, sized
for IEEE-33-scale feeders (~4 MW). A real MV feeder carrying tens of MW is
then INFEASIBLE for a reason that has nothing to do with the grid -- which is
exactly how MV Oberrhein f1 (33.8 MW) failed on 2026-08-01, reporting a bare
INFEASIBLE with no hint. These tests pin the suggestion helper and the
diagnostic that must accompany such an infeasibility."""

from __future__ import annotations

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    solve_lindistflow_expansion,
    suggested_flow_big_m_mva,
)
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_scenario_set


def test_suggestion_scales_with_feeder_demand() -> None:
    scenarios = build_scenario_set("stressed_five_point")
    small = suggested_flow_big_m_mva(load_distribution_feeder("ieee33"), scenarios)
    large = suggested_flow_big_m_mva(load_distribution_feeder("mv_oberrhein_f1"), scenarios)
    assert large > small
    # Must cover the whole feeder's peak demand -- the head line carries it all.
    assert large >= 33.8


def test_infeasible_under_small_big_m_carries_a_hint() -> None:
    net = load_distribution_feeder("mv_oberrhein_f1")
    scenarios = build_scenario_set("stressed_five_point")
    config = ExpansionProblemConfig(
        max_new_lines=3,
        time_limit_s=60.0,
        n_scenarios_aggregated=3,
        enable_reconfiguration=False,
        flow_big_m_mva=25.0,
    )
    result = solve_lindistflow_expansion(net, scenarios, [], config)
    assert result.termination_status == "INFEASIBLE"
    hint = str(result.metadata.get("infeasibility_hint", ""))
    assert "flow_big_m_mva" in hint, "a big-M-induced infeasibility must say so"
    assert "25.0" in hint and "suggested" in hint.lower()


def test_adequate_big_m_solves_the_same_feeder() -> None:
    net = load_distribution_feeder("mv_oberrhein_f1")
    scenarios = build_scenario_set("stressed_five_point")
    config = ExpansionProblemConfig(
        max_new_lines=3,
        time_limit_s=120.0,
        n_scenarios_aggregated=3,
        enable_reconfiguration=False,
        flow_big_m_mva=suggested_flow_big_m_mva(net, scenarios),
    )
    result = solve_lindistflow_expansion(net, scenarios, [], config)
    assert result.termination_status == "OPTIMAL"
    assert "infeasibility_hint" not in result.metadata
