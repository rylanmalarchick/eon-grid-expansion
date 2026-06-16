from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Annotated

import typer

from eon.formulations.layer_b import LayerBSurrogate, LayerBVariable
from eon.instances.candidate_lines import CandidateLine
from eon.mps.protocol import run_mps_protocol

app = typer.Typer(add_completion=False, help="MPS/JuliQAOA smoke interface.")
_INSTANCE_PATTERN = re.compile(r"^(hard|easy_path)_n(8|12|20)$")


@app.command()
def main(
    instance: Annotated[str, typer.Option(help="Synthetic instance identifier.")] = "hard_n20",
    chi_limit: Annotated[int, typer.Option(help="Maximum MPS bond dimension.")] = 64,
    angle_iterations: Annotated[
        int,
        typer.Option(help="Per-restart derivative-free angle-search iterations."),
    ] = 10,
    sample_count: Annotated[int, typer.Option(help="Samples drawn from each MPS state.")] = 16,
) -> None:
    surrogate = _synthetic_surrogate(instance)
    result = run_mps_protocol(
        surrogate,
        instance_name=instance,
        chi_limit=chi_limit,
        angle_iterations=angle_iterations,
        sample_count=sample_count,
    )
    typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


def _synthetic_surrogate(instance: str) -> LayerBSurrogate:
    match = _INSTANCE_PATTERN.fullmatch(instance)
    if match is None:
        raise ValueError(
            "Synthetic instance must be one of 'hard_n8', 'hard_n12', 'hard_n20', "
            "'easy_path_n8', 'easy_path_n12', or 'easy_path_n20'."
        )
    family = match.group(1)
    node_count = int(match.group(2))
    candidates = [
        CandidateLine(
            name=f"{instance}_{idx}",
            from_bus=idx,
            to_bus=(idx + 5) % node_count,
            family="synthetic",
            length_km=1.0,
            r_ohm_per_km=0.4,
            x_ohm_per_km=0.3,
            max_i_ka=0.4,
            build_cost=100_000.0 + 1_000.0 * idx,
        )
        for idx in range(node_count)
    ]
    variables = tuple(
        LayerBVariable(
            name=candidate.name,
            candidate=candidate,
            default_value=0,
            stress_score=1.0,
        )
        for candidate in candidates
    )
    if family == "hard":
        linear = {variable.name: -10.0 + 0.2 * index for index, variable in enumerate(variables)}
        quadratic = {}
        for left in range(len(variables)):
            for right in range(left + 1, len(variables)):
                if (right - left) <= 3:
                    quadratic[(variables[left].name, variables[right].name)] = 0.75
    else:
        linear = {variable.name: -2.0 + 0.05 * index for index, variable in enumerate(variables)}
        quadratic = {
            (variables[left].name, variables[left + 1].name): 0.15
            for left in range(len(variables) - 1)
        }
    return LayerBSurrogate(
        variables=variables,
        offset=0.0,
        linear=linear,
        quadratic=quadratic,
        fixed_builds={},
        max_new_lines=max(1, node_count // 2),
        base_objective=0.0,
        base_selected_count=max(1, node_count // 2),
    )


if __name__ == "__main__":
    app()
