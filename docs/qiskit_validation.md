# Validating our results in Qiskit

The challenge asks that E.ON be able to execute our algorithm and validate the
reported results in Qiskit. This document is the entry point for that.

## What to run

From `workspace/`:

```
python scripts/qiskit_validation.py --shots 4096 --depth 2 \
  --out experiments/results/qiskit_validation/p2_x.json
```

No solver licence and no local-only package are required. The instance is
generated from a seed, so the run is self-contained.

## What it does

1. Generates a fused-planted QUBO benchmark instance. The optimum is *planted*,
   so the ground truth is known independently of any solver.
2. Builds the gate-level QAOA circuit — one RZ per local field, one RZZ per
   coupling, then the mixer. This is the exportable artifact, not the
   $2^n$ diagonal operator our fast simulation path uses internally.
3. Runs it on Qiskit Aer.
4. Decodes the measurement counts into line sets and objective values.
5. Compares the Aer expectation value against our exact numpy engine.

## What the output means

The three lines to read:

```
Aer   expectation: +2.920901e+01 (4096 shots)
exact expectation: +2.917941e+01 (numpy engine)
difference: 2.960e-02 = 0.06 sigma (0.1014% of |exact|) -> AGREE
```

`sigma` is the Monte-Carlo standard error of the shot-sampled mean. Agreement
means the circuit we hand over reproduces, on an independent simulator, the
expectation value our own engine reports — so the numbers in the proposal come
from the circuit in `--out`'s companion `.qasm` file, not from a lookalike.
The script exits non-zero on disagreement.

`tests/test_qiskit_validation.py` includes the negative control: it feeds the
cross-check a circuit built from perturbed angles and asserts the check
*rejects* it. A cross-check that cannot fail would validate nothing.

## Exported circuit

With `--out PATH`, the script also writes `PATH.qasm` — OpenQASM 3 for the
penalty ansatz, which Braket and most toolchains ingest directly.

## Known limitation

`--mixer xy_ring` (the constrained ansatz) runs on Aer but does **not** export
to OpenQASM 3, because its fixed-Hamming-weight initial state is prepared by
naive state-vector synthesis. The script reports this in the record's
`qasm3_exportable` field and skips the `.qasm` write rather than emitting a
file that does not round-trip.

This is an implementation gap, not a fundamental one: deterministic Dicke-state
preparation in $O(kn)$ gates and $O(n)$ depth with no ancillas is published
(Bärtschi & Eidenbenz 2019, arXiv:1904.07358), and adopting it is a Phase II
work item. The proposal's negative-results section quantifies what the current
implementation costs — roughly 3,400× the transpiled depth of the penalty
ansatz.
