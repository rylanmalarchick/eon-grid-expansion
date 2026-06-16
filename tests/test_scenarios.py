from __future__ import annotations

import numpy as np
import pytest

from eon.instances.scenarios import build_scenario_set

ALL_SCENARIO_KINDS = [
    "phase1",
    "three_point",
    "stressed_three_point",
    "stressed_five_point",
]


@pytest.mark.parametrize("kind", ALL_SCENARIO_KINDS)
def test_scenario_probabilities_sum_to_one(kind: str) -> None:
    scenarios = build_scenario_set(kind)
    total = sum(s.probability for s in scenarios)
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


@pytest.mark.parametrize("kind", ALL_SCENARIO_KINDS)
def test_scenario_fields_positive(kind: str) -> None:
    scenarios = build_scenario_set(kind)
    for s in scenarios:
        assert s.load_scale > 0
        assert s.generation_scale > 0
        assert s.probability > 0
        assert s.line_capacity_scale > 0


def test_stressed_five_point_has_capacity_reduction() -> None:
    scenarios = build_scenario_set("stressed_five_point")
    assert len(scenarios) == 5
    reduced = [s for s in scenarios if s.line_capacity_scale < 1.0]
    assert len(reduced) >= 3


def test_unknown_scenario_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scenario set"):
        build_scenario_set("nonexistent")
