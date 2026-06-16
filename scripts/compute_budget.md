# Compute Budget

Phase I budget, aligned to `ATTACK_PLAN.md` §17.8 and the current codebase.

- Layer A Gurobi sweeps: 50-100 solver-hours on `theMachine` is enough for IEEE 33 / IEEE 123 candidate-family scans at 5-60 second time limits before scaling to larger feeders.
- Layer B surrogate calibration: a 6-variable neighborhood needs 1 base solve, 6 singleton solves, and 15 pair solves; at sub-5-second evaluation limits this stays under 2 wall-clock minutes per instance.
- QUBO / `neal` baselines: negligible compared to Gurobi. Thousands of reads per instance are practical on laptop CPU.
- QAOA simulator work: statevector/diagonal-cost constrained-mixer runs are practical up to roughly 10-12 variables on laptop CPU; beyond that, use decomposition or move to `theMachine`.
- MPS/JuliQAOA sweeps: 8-12 variable controls are laptop-feasible, but repeated 20-variable multi-ordering sweeps take minutes per instance because each record launches a Julia/ITensors job. Larger campaigns belong on `theMachine`.
- Storage: `instances/hard_instances.jsonl`, proposal figures, and experiment logs remain small; no special storage planning is needed in Phase I.

Open runtime risk:

- The backend is now real `JuliQAOA.jl`, but the classifier still launches one Julia process per instance. Batching or a persistent Julia worker is the next runtime win before large tensor-network campaigns.
