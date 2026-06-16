from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_phase1_scenarios, build_scenario_set

app = typer.Typer(add_completion=False, help="Solve the Layer A LinDistFlow expansion baseline.")


@app.command()
def main(
    case: Annotated[str, typer.Option(help="Distribution feeder identifier.")] = "ieee33",
    scenario_set: Annotated[str, typer.Option(help="Scenario-set identifier.")] = "phase1",
    candidate_family: Annotated[
        str,
        typer.Option(help="Candidate-line generator family."),
    ] = "distance_weighted",
    candidate_count: Annotated[
        int,
        typer.Option(help="Number of candidate lines to generate."),
    ] = 8,
    max_new_lines: Annotated[
        int,
        typer.Option(help="Maximum number of candidate lines that may be built."),
    ] = 3,
    seed: Annotated[int, typer.Option(help="Random seed for candidate generation.")] = 7,
    time_limit: Annotated[
        float,
        typer.Option(help="Gurobi time limit in seconds."),
    ] = 60.0,
    out_json: Annotated[
        Path | None,
        typer.Option(help="Optional JSON output path."),
    ] = None,
) -> None:
    net = load_distribution_feeder(case)
    scenarios = (
        build_phase1_scenarios()
        if scenario_set == "phase1"
        else build_scenario_set(scenario_set)
    )
    candidates = generate_candidate_lines(net, candidate_family, candidate_count, seed=seed)
    config = ExpansionProblemConfig(max_new_lines=max_new_lines, time_limit_s=time_limit)
    result = solve_lindistflow_expansion(net, scenarios, candidates, config)

    payload = result.to_dict()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
