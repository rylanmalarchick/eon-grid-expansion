"""Voltage must FALL along a loaded radial feeder, not rise.

LinDistFlow is v_to = v_from - 2(r*P + x*Q) with P the power flowing FROM
from_bus TO to_bus, i.e. positive toward the load. This model's nodal balance
appends +p_flow at from_bus and -p_flow at to_bus, which makes p_flow NEGATIVE
when power moves toward the load. Substituting a negative P into a drop equation
written for a positive one flips the sign, and voltage climbs with distance from
the substation -- faster the more loaded the feeder is.

Ground truth: pandapower's AC power flow on the same network. IEEE 33's textbook
minimum is ~0.913 pu at the feeder end.
"""

from __future__ import annotations

import pandapower as pp
import pytest

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    solve_lindistflow_expansion,
)
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import Scenario


@pytest.mark.requires_gurobi
def test_voltage_falls_away_from_the_substation() -> None:
    net = load_distribution_feeder("ieee33")
    pp.runpp(net)
    ac_min = float(net.res_bus.vm_pu.min())
    assert ac_min < 0.95, "fixture check: the AC solution should show a real voltage drop"

    scenarios = [
        Scenario(
            name="nominal",
            load_scale=1.0,
            generation_scale=1.0,
            probability=1.0,
            line_capacity_scale=1.0,
        )
    ]
    candidates = generate_candidate_lines(
        net, "community_bridging", 24, seed=7, cost_per_km=10_000.0
    )
    config = ExpansionProblemConfig(
        max_new_lines=3, time_limit_s=120.0, enable_reconfiguration=False, threads=1
    )
    result = solve_lindistflow_expansion(
        net,
        scenarios,
        candidates,
        config,
        fixed_builds={c.name: 0 for c in candidates},
    )
    voltages = result.bus_voltage_pu["nominal"]
    model_min = min(voltages.values())
    model_max = max(voltages.values())

    assert model_max <= 1.05 + 1e-6, (
        f"model reports voltage RISING to {model_max:.4f} pu on a loaded radial "
        f"feeder; the AC solution peaks at {float(net.res_bus.vm_pu.max()):.4f}. "
        "The drop term's sign disagrees with the nodal-balance flow convention."
    )
    assert model_min < 1.0, (
        f"model reports no bus below nominal (min {model_min:.4f}); the AC "
        f"solution reaches {ac_min:.4f}"
    )


@pytest.mark.requires_gurobi
@pytest.mark.parametrize("load_scale, tol", [(1.0, 0.02), (1.7, 0.03)])
def test_agrees_with_ac_power_flow(load_scale: float, tol: float) -> None:
    """The relaxation has to track the AC solution it stands in for.

    LinDistFlow neglects losses, so it reads slightly optimistic (higher) than
    the AC answer; the tolerance is one-sided-ish but checked as a magnitude.
    This is the check that would have caught both the drop-term sign error and
    the per-unit scaling error the moment either was introduced.
    """
    reference = load_distribution_feeder("ieee33")
    reference.load.p_mw *= load_scale
    reference.load.q_mvar *= load_scale
    pp.runpp(reference)

    net = load_distribution_feeder("ieee33")
    scenarios = [
        Scenario(
            name="s",
            load_scale=load_scale,
            generation_scale=1.0,
            probability=1.0,
            line_capacity_scale=1.0,
        )
    ]
    result = solve_lindistflow_expansion(
        net,
        scenarios,
        [],
        ExpansionProblemConfig(
            max_new_lines=3, time_limit_s=120.0, enable_reconfiguration=False, threads=1
        ),
        fixed_builds={},
    )
    assert result.termination_status == "OPTIMAL"
    # The state variable is SQUARED voltage; compare magnitudes.
    modelled = {bus: value**0.5 for bus, value in result.bus_voltage_pu["s"].items()}
    worst = max(modelled, key=lambda b: abs(modelled[b] - reference.res_bus.vm_pu[b]))
    error = modelled[worst] - reference.res_bus.vm_pu[worst]
    assert abs(error) <= tol, (
        f"bus {worst}: model {modelled[worst]:.4f} pu vs AC "
        f"{reference.res_bus.vm_pu[worst]:.4f} pu (error {error:+.4f})"
    )
