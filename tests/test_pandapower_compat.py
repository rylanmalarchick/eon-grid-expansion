"""Feeder loading must not depend on one pandapower release.

pandapower 3.5 removed select_subnet from the top-level namespace; it lives in
pandapower.toolbox in both 3.4 and 3.5. Calling pp.select_subnet worked on the
machine it was written on and raised AttributeError on a machine with a NEWER
release -- which is also what a reviewer on a current pandapower would hit.
"""

from __future__ import annotations

import pytest

from eon.instances.distribution_feeders import load_distribution_feeder


def test_select_subnet_is_imported_from_a_stable_location() -> None:
    """Pin the import path rather than the version."""
    from pandapower.toolbox import select_subnet

    assert callable(select_subnet)


@pytest.mark.parametrize("feeder", ["mv_oberrhein_f1", "mv_oberrhein_f2"])
def test_mv_oberrhein_feeders_load(feeder: str) -> None:
    net = load_distribution_feeder(feeder)
    assert len(net.bus) > 0
    assert len(net.line) > 0
    # One slack, and it is the substation busbar rather than the HV grid.
    assert len(net.ext_grid) == 1


def test_the_two_feeders_are_disjoint() -> None:
    """They are the two as-operated feeders of one network. Overlapping buses
    would mean the split silently kept the tie lines."""
    f1 = load_distribution_feeder("mv_oberrhein_f1")
    f2 = load_distribution_feeder("mv_oberrhein_f2")
    assert not (set(f1.bus.index) & set(f2.bus.index))
