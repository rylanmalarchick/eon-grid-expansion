# Quantum-enabled distribution-grid expansion planning

Code for the E.ON track of the 2026 Global Quantum + AI Challenge. Everything
reported in the Phase I proposal is produced by a script in this repository.

## Start here

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q tests/                      # full suite; no solver licence needed
python scripts/qiskit_validation.py   # runs the quantum path end to end
```

The package installs and imports **without** a Gurobi licence and without the
local provenance library, so the model can be inspected before either is
obtained. Validation still runs in that configuration — the numerical checks are
the same, only provenance recording degrades to a warning. Anything that needs
Gurobi says so and fails with a clear message rather than a stack trace.

## The two layers

**Layer A** (`eon.formulations.lindistflow`) is the planning model: a
multi-scenario LinDistFlow expansion MILP with joint network reconfiguration
under a radiality constraint. This is where the physics lives and where hardness
is established.

**Layer B** (`eon.formulations.layer_b`, `eon.formulations.qubo`) is a reduced
binary QUBO surrogate over the neighbourhood of a Layer A solution. It exists so
a NISQ-sized instance can be studied at all. Every quantum number is a statement
about Layer B, never about grid planning directly.

| Package | What is in it |
|---|---|
| `eon.formulations` | LinDistFlow MILP, Layer B surrogate, QUBO compilation |
| `eon.instances` | feeders, candidate generation, scenarios, external benchmarks |
| `eon.quantum` | QAOA engine, mixers, circuit export, decomposition + certificate |
| `eon.mps` | matrix-product-state protocol and the exact tensor-network control |
| `eon.classical` | Gurobi and simulated-annealing baselines |
| `eon.viz` | the proposal figures |
| `eon.bibliography` | reading list → BibTeX, with the do-not-cite list enforced |

## Reproducing the results

```bash
./scripts/reproduce_headline.sh quick   # wiring check, minutes
./scripts/reproduce_headline.sh full    # the reported numbers, hours
```

Eleven steps, from feeder hardness through to the figures. **Quick-mode numbers
are a wiring check and must never be quoted** — the time limits are far too
short to reproduce anything. Full mode uses the honest solve times.

## Validating in Qiskit

`docs/qiskit_validation.md` is the entry point for checking our results
independently. In short: `scripts/qiskit_validation.py` builds the gate-level
circuit, runs it on Aer, decodes the counts into line sets, and compares the
resulting expectation value against our exact simulation engine. Agreement
within sampling error establishes that the exported circuit is the algorithm we
benchmarked. The script exits non-zero if the two engines disagree, and writes
OpenQASM 3 alongside its result.

## What is proved and what is measured

Two results are machine-checked in Lean 4 (`lean/`, core Lean 4, no mathlib):

- `PosiformPlanting.lean` — the planted-optimum argument behind the synthetic
  benchmark generator.
- `DroppedCouplingBound.lean` — soundness of the decomposition certificate's
  lower bound.

Both carry an in-file axiom audit and contain no `sorry`. Everything else here
is empirical and is reported as measurement, never as guarantee.

Run them with `lean lean/DroppedCouplingBound.lean` (Lean 4.32).

## Conventions worth knowing before reading the code

- **Bound kinds are types.** `eon.quantum.bounds` separates
  `CertifiedLowerBound` from `HeuristicIncumbent`, and only the pair builds a
  `ClassicalDecompositionGap`. A proven bound and an achieved one are both a
  float, and using one where the other belongs makes a certificate look tighter
  than it is.
- **The certified gap is classical.** It measures the decomposition and
  integrality gap of the classical wrapper. It is not a quantum-versus-classical
  advantage measurement.
- **Toggle space vs build space.** Layer B variables are *toggles* relative to
  the Layer A incumbent; bit `i` of a basis index is build `i`. The two spaces
  are easy to confuse and several functions convert between them explicitly.
- **Penalty mode is explicit.** `build_energy_vector(..., penalty_mode=...)`
  selects `flat`, `quadratic`, or `none`. A flat penalty is a plateau and gives
  a variational method no gradient toward feasibility; the fair comparison uses
  `quadratic`.

## Development

```bash
pytest -q tests/ && ruff check . && mypy src
```

All three must be clean. Tests assert against real behaviour rather than mocks,
and several exist specifically as negative controls — they check that a
detector fires on the bug it was written for, because a check that cannot fail
is not a check.
