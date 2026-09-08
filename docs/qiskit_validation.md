# Validate the results in Qiskit

The challenge asks E.ON to execute the algorithm and validate the reported
results in Qiskit. This document is the entry point.

## Run it

From `workspace/`:

```
python scripts/qiskit_validation.py --shots 4096 --depth 2 \
  --out experiments/results/qiskit_validation/p2_x.json
```

The run needs no solver license and no package outside the public indexes. A
seed generates the instance, so the run is self-contained.

## What the script does

1. Generates a fused-planted QUBO benchmark instance. The optimum is planted,
   so the ground truth is known without a solver.
2. Builds the gate-level QAOA circuit. One RZ per local field, one RZZ per
   coupling, then the mixer. This is the exportable circuit. The fast
   simulation path uses a 2^n diagonal operator instead, which no device
   accepts.
3. Runs the circuit on Qiskit Aer.
4. Decodes the measurement counts into line sets and objective values.
5. Compares the Aer expectation value against the exact numpy engine.

## Read the output

Three lines matter:

```
Aer   expectation: +2.920901e+01 (4096 shots)
exact expectation: +2.917941e+01 (numpy engine)
difference: 2.960e-02 = 0.06 sigma (0.1014% of |exact|) -> AGREE
```

`sigma` is the Monte-Carlo standard error of the shot-sampled mean. Agreement
means the circuit reproduces our own expectation value on an independent
simulator. The numbers in the proposal therefore come from the circuit in the
companion `.qasm` file. The script exits non-zero when the two engines disagree.

`tests/test_qiskit_validation.py` holds the negative control. It gives the
cross-check a circuit built from perturbed angles and asserts that the check
rejects it. A check that cannot fail validates nothing.

## Exported circuit

With `--out PATH` the script also writes `PATH.qasm`. This is OpenQASM 3 for the
penalty ansatz. Braket and most toolchains read it directly.

## Known limitation

`--mixer xy_ring` selects the constrained ansatz. It runs on Aer. It does not
export to OpenQASM 3, because naive state-vector synthesis prepares its
fixed-Hamming-weight initial state. The script sets `qasm3_exportable` to false
in the record and skips the `.qasm` write. It does not emit a file that fails to
round-trip.

This gap is an implementation limit, not a fundamental one. Baertschi and
Eidenbenz (2019, arXiv:1904.07358) give deterministic Dicke-state preparation in
O(kn) gates and O(n) depth with no ancillas. Adopting it is a Phase II work
item. The proposal measures the current cost at about 3,400 times the transpiled
depth of the penalty ansatz.
