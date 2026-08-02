"""Asking for more surrogate variables than there are candidates must be loud.

The clamp itself is right -- there is nothing to select beyond the candidates
that exist. But it used to be a bare min(), so a sweep that requested
neighborhood sizes 16/20/24/28 against 24 candidates produced FOUR result files
of which the last two were identical: n=28 silently became a second n=24. The
duplicate records read as a fourth instance size, which is a claim about
instance diversity that the data did not support.
"""

from __future__ import annotations

import logging

import pytest

from eon.formulations.lindistflow import ExpansionProblemConfig
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.hardness_classifier import build_instance_surrogate
from eon.instances.scenarios import build_scenario_set


@pytest.mark.parametrize(
    "requested,candidates,expect_warning",
    [(28, 12, True), (8, 12, False)],
)
def test_over_wide_neighborhood_warns_and_reports_the_real_size(
    caplog: pytest.LogCaptureFixture, requested: int, candidates: int, expect_warning: bool
) -> None:
    net = load_distribution_feeder("ieee33")
    scenarios = build_scenario_set("phase1")
    config = ExpansionProblemConfig(
        max_new_lines=2, time_limit_s=10.0, enable_reconfiguration=False, threads=1
    )
    with caplog.at_level(logging.WARNING, logger="eon.instances.hardness_classifier"):
        generated, _, surrogate = build_instance_surrogate(
            net,
            scenarios,
            config,
            family="community_bridging",
            candidate_count=candidates,
            neighborhood_size=requested,
            seed=7,
            time_limit=10.0,
            cost_per_km=10_000.0,
            baseline_stress={},
        )

    assert len(surrogate.variables) <= len(generated)
    warned = any("exceeds the" in record.message for record in caplog.records)
    assert warned is expect_warning, (
        f"requested {requested} against {len(generated)} candidates: "
        f"warning={warned}, expected {expect_warning}"
    )
