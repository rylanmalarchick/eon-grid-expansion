from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eon.formulations.layer_b import LayerBSurrogate
from eon.validation import validate_probability_array


@dataclass(frozen=True, slots=True)
class QuantumResult:
    bitstring: str
    sampling_prob: float
    objective: float
    feasible: bool
    certified_gap: float | None
    actual_builds: dict[str, int]
    selected_candidates: tuple[str, ...]


def decode_counts(
    surrogate: LayerBSurrogate,
    counts: dict[str, int],
    *,
    total_shots: int | None = None,
) -> list[QuantumResult]:
    shots = total_shots or sum(counts.values()) or 1
    probabilities = np.asarray(
        [
            count / shots
            for _, count in sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        dtype=float,
    )
    validate_probability_array(probabilities, name="quantum_postprocess_probabilities")
    results: list[QuantumResult] = []
    for raw_bitstring, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        bitstring = raw_bitstring[::-1]
        actual_builds = {
            variable.name: int(bit)
            for variable, bit in zip(surrogate.variables, bitstring, strict=True)
        }
        selected_count = sum(actual_builds.values()) + sum(
            value
            for name, value in surrogate.fixed_builds.items()
            if name not in surrogate.variable_names
        )
        toggle_decisions = {
            variable.name: variable.default_value ^ actual_builds[variable.name]
            for variable in surrogate.variables
        }
        objective = surrogate.surrogate_objective(toggle_decisions)
        feasible = selected_count <= surrogate.max_new_lines
        results.append(
            QuantumResult(
                bitstring=bitstring,
                sampling_prob=count / shots,
                objective=objective,
                feasible=feasible,
                certified_gap=None,
                actual_builds=actual_builds,
                selected_candidates=tuple(
                    sorted(name for name, selected in actual_builds.items() if selected)
                ),
            )
        )
    return results
