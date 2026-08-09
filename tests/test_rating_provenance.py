"""Invented branch ratings must be visible, because congestion is defined against them.

`case33bw` ships max_i_ka = 99999 on every branch -- pandapower's "no limit"
placeholder. Normalisation masks values above 1000 to NaN, takes the median of
an all-NaN series (NaN), and falls through to a hard-coded 0.4 kA. So every
rating on the feeder that produces every headline congestion number is a
modelling assumption, applied uniformly to the head section and the laterals
alike, and nothing said so.

This does not assert the value is right. It asserts the assumption is recorded,
so no number can rest on it silently.
"""

from __future__ import annotations

import logging

from eon.instances.distribution_feeders import load_distribution_feeder


def test_ieee33_records_that_every_rating_is_defaulted() -> None:
    net = load_distribution_feeder("ieee33")
    provenance = net["eon_metadata"]["rating_provenance"]
    assert provenance["total_branches"] == 37
    assert provenance["defaulted_branches"] == 37, (
        "case33bw ships no usable rating; if this changes, the congestion "
        "baseline changes with it"
    )
    assert provenance["default_ka"] == 0.4
    assert "fallback" in provenance["default_source"]


def test_defaulting_is_warned_about(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        load_distribution_feeder("ieee33")
    assert any("NOT feeder data" in record.message for record in caplog.records), (
        "a wholly invented rating set must warn: it is load-bearing for every "
        "congestion number on this feeder"
    )
