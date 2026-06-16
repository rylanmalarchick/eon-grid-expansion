from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

import typer

from eon.formulations.layer_b import (
    build_layer_b_surrogate,
    solve_layer_b_surrogate,
    validate_layer_b_surrogate,
)
from eon.formulations.lindistflow import ExpansionProblemConfig, solve_lindistflow_expansion
from eon.formulations.qubo import compile_layer_b_qubo, solve_qubo_with_neal
from eon.instances.candidate_lines import generate_candidate_lines
from eon.instances.distribution_feeders import load_distribution_feeder
from eon.instances.scenarios import build_phase1_scenarios

app = typer.Typer(add_completion=False, help="Simulated annealing on the Layer B surrogate QUBO.")


@app.command()
def main(
    case: Annotated[str, typer.Option(help="Distribution feeder identifier.")] = "ieee33",
    candidate_family: Annotated[
        str,
        typer.Option(help="Candidate-line generator family."),
    ] = "distance_weighted",
    candidate_count: Annotated[
        int,
        typer.Option(help="Number of candidate lines to generate."),
    ] = 4,
    max_new_lines: Annotated[
        int,
        typer.Option(help="Maximum number of candidate lines that may be built."),
    ] = 1,
    neighborhood_size: Annotated[
        int,
        typer.Option(help="Layer B neighborhood size."),
    ] = 3,
    num_reads: Annotated[int, typer.Option(help="neal read count.")] = 256,
) -> None:
    net = load_distribution_feeder(case)
    scenarios = build_phase1_scenarios()
    candidates = generate_candidate_lines(net, candidate_family, candidate_count)
    config = ExpansionProblemConfig(max_new_lines=max_new_lines, time_limit_s=5.0)
    layer_a_result = solve_lindistflow_expansion(net, scenarios, candidates, config)
    surrogate = build_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        layer_a_result,
        config,
        neighborhood_size=neighborhood_size,
        evaluation_time_limit_s=3.0,
    )
    validation = validate_layer_b_surrogate(
        net,
        scenarios,
        candidates,
        surrogate,
        config,
        top_k=min(3, 2**len(surrogate.variables)),
        evaluation_time_limit_s=3.0,
    )
    gurobi_solution = solve_layer_b_surrogate(surrogate)
    compilation = compile_layer_b_qubo(surrogate)
    annealed_solution = solve_qubo_with_neal(surrogate, compilation, num_reads=num_reads)
    typer.echo(
        json.dumps(
            {
                "layer_a_objective": layer_a_result.objective_value,
                "layer_b_validation": asdict(validation),
                "layer_b_gurobi": asdict(gurobi_solution),
                "layer_b_neal": asdict(annealed_solution),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
