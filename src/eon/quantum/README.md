# Quantum solvers

- `cop_qaoa.py` — constrained QAOA for the Layer B surrogate: `build_p1_landscape`,
  `run_constrained_qaoa_subproblem`, and a Hamming-weight-preserving XX+YY mixer circuit.
- `decomposition.py` — `decompose_surrogate` splits a Layer B surrogate into smaller
  blocks via the interaction graph for subproblem QAOA.
- `postprocess.py` — `decode_counts` turns sampled bitstrings into `QuantumResult`
  records (objective, sampling probability, feasibility).

## Planned (not yet implemented)

- `qaoa.py` — QAOA with Clifford warm-start (arXiv:2602.14327) and LP warm-start.
- `vqe_ising.py` — VQE on the Ising Hamiltonian, PennyLane prototype.
- `qiskit_adapter.py` — Qiskit-compatible entry point for the EON submission.
- `hardware.py` — Qiskit Runtime / IQM Resonance job submission with dry-run cost estimation.
