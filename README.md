# Quantum-enabled distribution-grid expansion planning

Code for the E.ON track of the 2026 Global Quantum + AI Challenge. A script in
this repository produces every number in the proposal.

## Install and run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q tests/                      # solver-dependent tests skip without a license
python scripts/qiskit_validation.py   # runs the quantum path end to end
```

Gurobi is the only optional dependency. Without a license the package still
installs, imports, and runs the full non-solver suite. The Layer A tests skip
and name the reason, so the summary shows what did not run. Code that needs
Gurobi reports a clear message instead of a stack trace.

No dependency comes from outside the public package indexes. The numerical
checks and the provenance records live in `eon._checks`. The Julia drivers
carry their own copies of the same checks.

## The two layers

**Layer A** (`eon.formulations.lindistflow`) is the planning model. It is a
multi-scenario LinDistFlow expansion MILP with joint network reconfiguration
under a radiality constraint. The physics lives here. Hardness is measured here.

**Layer B** (`eon.formulations.layer_b`, `eon.formulations.qubo`) is a reduced
binary QUBO surrogate over the neighborhood of a Layer A solution. It exists so
that a NISQ-sized instance can be studied. Every quantum number describes Layer
B, never grid planning directly.

| Package | Contents |
|---|---|
| `eon.formulations` | LinDistFlow MILP, Layer B surrogate, QUBO compilation |
| `eon.instances` | feeders, candidate generation, scenarios, external benchmarks |
| `eon.quantum` | QAOA engine, mixers, circuit export, decomposition, certificate |
| `eon.mps` | matrix-product-state protocol and the exact tensor-network control |
| `eon.classical` | Gurobi and simulated-annealing baselines |
| `eon.viz` | proposal figures |
| `eon.bibliography` | reading list to BibTeX, with the do-not-cite list enforced |

## Reproduce the results

```bash
./scripts/reproduce_headline.sh quick   # wiring check, minutes
./scripts/reproduce_headline.sh full    # the reported numbers, hours
```

The script runs eleven steps, from feeder hardness to the figures. Quick mode
uses short time limits and checks the wiring only. **Do not quote a quick-mode
number.** Full mode uses the real solve times and reproduces the proposal.

## Validate in Qiskit

`scripts/qiskit_validation.py` builds the gate-level circuit, runs it on Aer, and
decodes the counts into line sets. It then compares the resulting expectation
value against the exact simulation engine. `docs/qiskit_validation.md` documents
the layer. Agreement within sampling error shows that the exported circuit is the
algorithm we benchmarked. The script exits non-zero when the two engines
disagree. It also writes the OpenQASM 3 circuit next to its result.

## Proved and measured

Lean 4 checks two results. Both files use core Lean 4 without mathlib.

- `lean/PosiformPlanting.lean` proves the planted-optimum argument behind the
  synthetic benchmark generator.
- `lean/DroppedCouplingBound.lean` proves that the decomposition certificate
  lower bound is sound.

Each file ends with an axiom audit and contains no `sorry`. Run one with
`lean lean/DroppedCouplingBound.lean` (Lean 4.32).

Everything else in this repository is measured, not proved. Reports state it as
measurement, never as a guarantee.

## Conventions

- **Bound kinds are types.** `eon.quantum.bounds` separates
  `CertifiedLowerBound` from `HeuristicIncumbent`. Only the pair builds a
  `ClassicalDecompositionGap`. A proved bound and an achieved bound are both a
  float. Using one where the other belongs makes a certificate look tighter than
  it is.
- **The certified gap is classical.** It measures the decomposition and
  integrality gap of the classical wrapper. It does not measure quantum
  advantage.
- **Toggle space and build space differ.** Layer B variables are toggles
  relative to the Layer A incumbent. Bit `i` of a basis index is build `i`.
  Several functions convert between the two spaces explicitly.
- **Penalty mode is explicit.** `build_energy_vector(..., penalty_mode=...)`
  takes `flat`, `quadratic`, or `none`. A flat penalty is a plateau and gives a
  variational method no gradient toward feasibility. The fair comparison uses
  `quadratic`.

## Development

```bash
pytest -q tests/ && ruff check . && mypy src
```

All three must pass. The tests assert against real behavior and do not use
mocks. Several tests are negative controls. Each one checks that a detector
fires on the bug it was written for. A check that cannot fail is not a check.
