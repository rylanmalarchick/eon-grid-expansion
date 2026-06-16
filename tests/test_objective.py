from eon.metrics.objective import ScenarioMetrics, aggregate_metrics, canonical_congestion_mw


def test_canonical_congestion_metric() -> None:
    assert canonical_congestion_mw(3.5, 0.2) == 0.7


def test_aggregate_metrics() -> None:
    metrics = [
        ScenarioMetrics("a", 0.5, 1.0, 0.1, 0.0, 0.4),
        ScenarioMetrics("b", 0.5, 2.0, 0.2, 1.0, 0.6),
    ]
    aggregate = aggregate_metrics(metrics)
    assert aggregate["weighted_congestion_mw"] == 1.5
    assert aggregate["weighted_voltage_violation_pu"] == 0.15000000000000002
