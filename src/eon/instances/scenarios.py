from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    load_scale: float
    generation_scale: float
    probability: float = 1.0
    line_capacity_scale: float = 1.0


def build_phase1_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="stressed",
            load_scale=1.40,
            generation_scale=1.00,
            probability=1.0,
            line_capacity_scale=1.0,
        )
    ]


def build_scenario_set(kind: str) -> list[Scenario]:
    scenario_kind = kind.lower().strip()
    if scenario_kind == "phase1":
        return build_phase1_scenarios()
    if scenario_kind == "three_point":
        return [
            Scenario(name="peak_load", load_scale=1.45, generation_scale=0.75, probability=0.45),
            Scenario(name="balanced", load_scale=1.00, generation_scale=1.00, probability=0.30),
            Scenario(name="peak_dg", load_scale=0.80, generation_scale=1.35, probability=0.25),
        ]
    if scenario_kind == "stressed_three_point":
        return [
            Scenario(
                name="severe_peak",
                load_scale=1.65,
                generation_scale=0.60,
                probability=0.45,
                line_capacity_scale=0.70,
            ),
            Scenario(
                name="stressed_balanced",
                load_scale=1.20,
                generation_scale=0.85,
                probability=0.30,
                line_capacity_scale=0.85,
            ),
            Scenario(
                name="high_dg_curtail",
                load_scale=0.70,
                generation_scale=1.50,
                probability=0.25,
                line_capacity_scale=0.80,
            ),
        ]
    if scenario_kind == "stressed_five_point":
        return [
            Scenario(
                name="extreme_peak",
                load_scale=1.70,
                generation_scale=0.50,
                probability=0.25,
                line_capacity_scale=0.65,
            ),
            Scenario(
                name="severe_peak",
                load_scale=1.55,
                generation_scale=0.70,
                probability=0.25,
                line_capacity_scale=0.75,
            ),
            Scenario(
                name="stressed_balanced",
                load_scale=1.20,
                generation_scale=0.90,
                probability=0.20,
                line_capacity_scale=0.85,
            ),
            Scenario(
                name="high_dg_curtail",
                load_scale=0.70,
                generation_scale=1.50,
                probability=0.15,
                line_capacity_scale=0.80,
            ),
            Scenario(
                name="emergency_low_gen",
                load_scale=1.40,
                generation_scale=0.35,
                probability=0.15,
                line_capacity_scale=0.70,
            ),
        ]
    raise ValueError(f"Unknown scenario set '{kind}'.")
