"""Branches that are not in service must not carry power.

Two ways a branch can fail to exist and still conduct:

  * an UNBUILT CANDIDATE -- flow gating used to be applied only under
    ``enable_reconfiguration``, so in the non-reconfiguration model an unbuilt
    candidate still entered the nodal balance. Its thermal row reads
    ``p_abs + q_abs <= cap*z + overload`` with z = 0, so 100 % of whatever it
    carried was booked as congestion on a line nobody built -- inflating the
    baseline the headline reduction is measured against.

  * a NORMALLY-OPEN TIE -- ``_existing_line_specs`` iterated ``net.line``
    without an ``in_service`` filter, energizing IEEE 33's five open ties and
    handing the optimizer a meshed network to plan on.
"""

from __future__ import annotations

import pytest

from eon.formulations.lindistflow import (
    ExpansionProblemConfig,
    solve_lindistflow_expansion,
)
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import Scenario

_TOL = 1e-6


def _solve(*, reconfiguration: bool):
    net = load_distribution_feeder("ieee33")
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
        max_new_lines=3,
        time_limit_s=120.0,
        enable_reconfiguration=reconfiguration,
        threads=1,
    )
    result = solve_lindistflow_expansion(
        net,
        scenarios,
        candidates,
        config,
        fixed_builds={c.name: 0 for c in candidates},
    )
    return net, result


@pytest.mark.requires_gurobi
@pytest.mark.parametrize("reconfiguration", [False, True])
def test_unbuilt_candidates_carry_no_flow(reconfiguration: bool) -> None:
    _, result = _solve(reconfiguration=reconfiguration)
    loading = result.line_loading_proxy_mva["nominal"]
    carried = {
        name: mva for name, mva in loading.items() if name.startswith("cand") and abs(mva) > _TOL
    }
    assert not carried, (
        f"{len(carried)} unbuilt candidate(s) carry flow totalling "
        f"{sum(carried.values()):.3f} MVA; every MVA is booked as congestion "
        f"on a line that was never built"
    )


@pytest.mark.requires_gurobi
def test_normally_open_ties_are_not_energized_without_switching() -> None:
    net, result = _solve(reconfiguration=False)
    open_ties = [f"line_{i}" for i in net.line.index[~net.line.in_service]]
    assert open_ties, "fixture check: IEEE 33 should ship with open tie lines"
    loading = result.line_loading_proxy_mva["nominal"]
    energized = {n: loading[n] for n in open_ties if abs(loading.get(n, 0.0)) > _TOL}
    assert not energized, f"normally-open tie(s) carrying flow with switching disabled: {energized}"
