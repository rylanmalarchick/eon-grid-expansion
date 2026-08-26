"""Voltage limits must bound MAGNITUDE, not squared voltage.

`v_bus` is squared voltage: the LinDistFlow drop carries the factor 2 and
tests/test_voltage_sign.py takes `value ** 0.5` before comparing to an AC
solution. But `voltage_min_pu = 0.95` and `voltage_max_pu = 1.05` were applied
directly to that variable, so the band actually enforced was
[sqrt(0.95), sqrt(1.05)] = [0.9747, 1.0247] -- a band less than half the width
of the one documented, on the wrong quantity.

This is the fourth defect in the same units family as the three fixed on
2026-08-08 (drop sign, per-unit scaling, gating, tie energisation), and the
sweep that found those did not find this one.
"""

from __future__ import annotations

import pytest

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    solve_lindistflow_expansion,
)
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import Scenario


@pytest.mark.requires_gurobi
def test_the_enforced_band_is_on_magnitude() -> None:
    """Drive the feeder until it violates, and check WHERE the band bites.

    At a load that pushes the feeder below nominal, the first bus to be
    penalised must be the first bus whose MAGNITUDE crosses voltage_min_pu, not
    the first whose SQUARED voltage does.
    """
    net = load_distribution_feeder("ieee33")
    scenarios = [
        Scenario(name="s", load_scale=1.3, generation_scale=1.0,
                 probability=1.0, line_capacity_scale=1.0)
    ]
    config = ExpansionProblemConfig(
        max_new_lines=3, time_limit_s=120.0, enable_reconfiguration=False, threads=1
    )
    result = solve_lindistflow_expansion(net, scenarios, [], config, fixed_builds={})
    assert result.termination_status == "OPTIMAL"

    squared = result.bus_voltage_pu["s"]
    magnitudes = {bus: value**0.5 for bus, value in squared.items()}
    reported = result.scenario_metrics[0].voltage_violation_pu

    # Sum of magnitude deficits against the documented band.
    expected = sum(
        max(0.0, config.voltage_min_pu - v) + max(0.0, v - config.voltage_max_pu)
        for v in magnitudes.values()
    )
    assert reported == pytest.approx(expected, rel=0.05, abs=1e-6), (
        f"reported violation {reported:.4f} does not match the magnitude "
        f"deficit {expected:.4f}. If the bound is applied to squared voltage "
        f"the reported figure is a squared deficit mislabelled as pu."
    )
