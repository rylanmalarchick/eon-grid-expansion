"""Shared markers.

Tests that solve a Layer A model carry `@pytest.mark.requires_gurobi`. Without
a licence they SKIP, and the skip is deliberate and named: a reviewer sees
"skipped (needs gurobipy...)" in the summary rather than a green run that
quietly covered less than it appears to.

Auto-converting a ModuleNotFoundError into a skip would be tidier and worse --
it would also swallow a genuine import bug.
"""

from __future__ import annotations

import importlib.util

import pytest

GUROBI_AVAILABLE = importlib.util.find_spec("gurobipy") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_gurobi: solves a Layer A model; needs gurobipy"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if GUROBI_AVAILABLE:
        return
    skip = pytest.mark.skip(
        reason="needs gurobipy and a Gurobi licence (Layer A is a Gurobi model)"
    )
    for item in items:
        if "requires_gurobi" in item.keywords:
            item.add_marker(skip)
