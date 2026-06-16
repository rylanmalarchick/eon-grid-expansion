# Formulations

- `lindistflow.py` — LinDistFlow expansion MILP (Layer A) solved with Gurobi:
  `solve_lindistflow_expansion`, `ExpansionProblemConfig`, `ExpansionResult`.
- `layer_b.py` — Layer B surrogate: `build_layer_b_surrogate`, `solve_layer_b_surrogate`,
  `validate_layer_b_surrogate`.
- `qubo.py` — Layer B surrogate -> QUBO compilation with the penalty method
  (`compile_layer_b_qubo`) and a `neal` simulated-annealing solve (`solve_qubo_with_neal`).

## Planned (not yet implemented)

- `blp_cuenca.py` — Binary linear program per Cuenca et al. 2024 (IEEE TPS).
- `steiner.py` — Capacitated Steiner tree per Duan and Yu 2003 (fallback).
- `dc_opf.py` — DC-OPF constraints and PTDF helpers.
