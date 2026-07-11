"""The p-list sweep must reuse ONE Layer A solve / surrogate per instance."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconfiguration_sweep.py"
_spec = importlib.util.spec_from_file_location("reconfiguration_sweep", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def _fake_layer_a() -> SimpleNamespace:
    return SimpleNamespace(
        termination_status="TIME_LIMIT",
        mip_gap=0.5,
        best_bound=1.0,
        objective_value=2.0,
        runtime_s=1.0,
        selected_candidates=("c0",),
        metadata={"closed_line_count": 3},
    )


def _fake_coupling() -> SimpleNamespace:
    return SimpleNamespace(
        variable_count=20,
        structural_treewidth=11,
        effective_treewidth=11,
        coupling_field_ratio=70.0,
        mps_easy=False,
    )


def _fake_layer_b() -> SimpleNamespace:
    return SimpleNamespace(objective_value=5.0, mip_gap=0.0, status="OPTIMAL")


def _fake_tree(within_budget: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        backend="generic_tn_tropical",
        contraction_width=16.0,
        within_budget=within_budget,
        best_energy=4.5,
        chi_max_reached=64,
    )


def _fake_mps(p: int) -> SimpleNamespace:
    ordering = SimpleNamespace(
        ordering="natural",
        best_energy=6.0 + p,
        chi_curve={4: 8.0, 64: 6.0 + p},
        entropy_curve={4: 0.5, 64: 1.5},
    )
    return SimpleNamespace(
        backend="juliqaoa_mps",
        chi_max_reached=64,
        reference_energy=5.0,
        exact_ground_energy=4.5,
        ordering_results=[ordering],
        tree_tn_result=_fake_tree(),
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"build": [], "mps": [], "layer_b": [], "tree": []}

    def fake_build(net, scenarios, config, **kwargs):  # noqa: ANN001
        calls["build"].append(kwargs["seed"])
        return ([], _fake_layer_a(), object())

    def fake_layer_b(surrogate, **kwargs):  # noqa: ANN001
        calls["layer_b"].append(kwargs)
        return _fake_layer_b()

    def fake_mps_protocol(surrogate, **kwargs):  # noqa: ANN001
        calls["mps"].append(kwargs["qaoa_rounds"])
        return _fake_mps(kwargs["qaoa_rounds"])

    def fake_tree_control(surrogate, **kwargs):  # noqa: ANN001
        calls["tree"].append(kwargs)
        return _fake_tree()

    monkeypatch.setattr(sweep, "build_instance_surrogate", fake_build)
    monkeypatch.setattr(sweep, "coupling_diagnostics", lambda s: _fake_coupling())
    monkeypatch.setattr(sweep, "solve_layer_b_surrogate", fake_layer_b)
    monkeypatch.setattr(sweep, "run_mps_protocol", fake_mps_protocol)
    monkeypatch.setattr(sweep, "run_tree_tn_control", fake_tree_control)
    return calls


def _run(calls: dict[str, list], *, with_mps: bool, p_list: list[int]) -> list[dict]:
    return list(
        sweep.run_instance(
            "ieee33",
            "community_bridging",
            7,
            neighborhood_size=20,
            time_limit=180.0,
            with_mps=with_mps,
            reconfiguration=True,
            qaoa_rounds_list=p_list,
            angle_iterations=10,
            net=object(),
            scenarios=[],
            baseline_stress={},
            run_meta={"time_limit_s": 180.0},
        )
    )


def test_p_list_sweeps_on_one_surrogate(patched: dict[str, list]) -> None:
    records = _run(patched, with_mps=True, p_list=[1, 2, 3])

    assert len(records) == 3
    assert patched["build"] == [7], "Layer A must be solved exactly once per instance"
    assert len(patched["layer_b"]) == 1, "Layer B must be solved exactly once per instance"
    assert patched["mps"] == [1, 2, 3]
    assert [r["run"]["qaoa_rounds"] for r in records] == [1, 2, 3]
    # Same instance across p: identical instance_id and Layer A block.
    assert len({r["instance_id"] for r in records}) == 1
    assert all(r["layer_a"] == records[0]["layer_a"] for r in records)


def test_record_schema_matches_prior_runs(patched: dict[str, list]) -> None:
    (record,) = _run(patched, with_mps=True, p_list=[2])
    assert set(record) == {
        "instance_id",
        "feeder",
        "family",
        "seed",
        "candidate_count",
        "neighborhood_size",
        "enable_reconfiguration",
        "with_mps",
        "layer_a",
        "coupling",
        "tree_tn",
        "layer_b",
        "mps",
        "signals",
        "run",
    }
    assert record["run"]["qaoa_rounds"] == 2
    assert record["mps"]["orderings"][0]["chi_curve"] == {"4": 8.0, "64": 8.0}


def test_no_mps_yields_single_record_with_null_p(patched: dict[str, list]) -> None:
    records = _run(patched, with_mps=False, p_list=[1, 2, 3])

    assert len(records) == 1
    assert records[0]["mps"] is None
    assert records[0]["run"]["qaoa_rounds"] is None
    assert len(patched["tree"]) == 1
    assert patched["mps"] == []
