from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    build_cost: float = 1.0
    thermal_violation: float = 500_000.0
    voltage_violation: float = 750_000.0
    curtailment: float = 250_000.0
    loss_proxy: float = 5_000.0


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    scenario: str
    probability: float
    thermal_overlimit_mw: float
    voltage_violation_pu: float
    curtailment_mw: float
    loss_proxy_mw: float


def canonical_congestion_mw(thermal_overlimit_mw: float, probability: float) -> float:
    return round(float(thermal_overlimit_mw) * float(probability), 12)


def aggregate_metrics(metrics: list[ScenarioMetrics]) -> dict[str, float]:
    return {
        "weighted_congestion_mw": sum(
            canonical_congestion_mw(metric.thermal_overlimit_mw, metric.probability)
            for metric in metrics
        ),
        "weighted_voltage_violation_pu": sum(
            metric.voltage_violation_pu * metric.probability for metric in metrics
        ),
        "weighted_curtailment_mw": sum(
            metric.curtailment_mw * metric.probability for metric in metrics
        ),
        "weighted_loss_proxy_mw": sum(
            metric.loss_proxy_mw * metric.probability for metric in metrics
        ),
    }
