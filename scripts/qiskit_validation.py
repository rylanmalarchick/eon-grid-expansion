"""D5: the Qiskit entry point E.ON runs to check our numbers independently.

The challenge requires that the sponsor be able to execute and validate results
in Qiskit. This script does that end to end with no solver license and no
local-only dependency:

  1. builds a benchmark instance (planted optimum, so ground truth is known),
  2. builds the gate-level QAOA circuit -- RZ/RZZ + mixer, the exportable
     artifact, not the 2^n DiagonalGate the fast path simulates,
  3. runs it on Qiskit Aer,
  4. decodes the counts into line sets and objective values,
  5. CROSS-CHECKS the Aer expectation against our exact numpy engine.

Step 5 is the point. Agreement within sampling error is evidence that the
circuit we hand over is the algorithm we benchmarked rather than a lookalike --
the same property tests/test_export.py pins up to global phase, re-established
here on a running simulator by a reader who need not trust our test suite.

Run from workspace/:
    python scripts/qiskit_validation.py [--shots 4096] [--depth 2] [--out PATH]

The constrained (xy_ring) ansatz is available via --mixer but is NOT
OpenQASM-3-exportable: its fixed-Hamming-weight initial state is prepared by
naive state-vector synthesis. That limitation is reported in the proposal, with
the published O(kn)-gate remedy cited (R47). The default mixer is the one that
actually exports.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit import qasm3, transpile
from qiskit_aer import AerSimulator

from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.cop_qaoa import _target_hamming_weight, _uniform_weight_state
from eon.quantum.energy import build_energy_vector
from eon.quantum.export import build_gate_level_qaoa_circuit
from eon.quantum.postprocess import decode_counts
from eon.quantum.simulator import (
    expected_energy_of_state,
    simulate_qaoa,
    uniform_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qiskit_validation")
# Qiskit's transpiler logs one INFO line per pass, which buries the three lines
# a reader actually came here for.
logging.getLogger("qiskit").setLevel(logging.WARNING)

# Fixed angles rather than an optimizer run: this script validates that two
# engines agree on the SAME circuit, which is a claim about the circuit, not
# about the angles. Optimized angles live in the P1/P3 experiment records.
DEFAULT_BETAS = (0.42, 0.19)
DEFAULT_GAMMAS = (0.31, 0.57)


def _initial_state(surrogate, mixer: str, n: int) -> np.ndarray:
    if mixer == "xy_ring":
        return _uniform_weight_state(n, _target_hamming_weight(surrogate))
    if mixer == "warm_start":
        raise ValueError(
            "warm_start needs the relaxation thetas; use --mixer x or xy_ring here"
        )
    return uniform_state(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=2, help="QAOA layers p")
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mixer", default="x", choices=["x", "xy_ring"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    betas = DEFAULT_BETAS[: args.depth]
    gammas = DEFAULT_GAMMAS[: args.depth]
    if len(betas) < args.depth:
        raise SystemExit(f"--depth {args.depth} exceeds the {len(DEFAULT_BETAS)} fixed angles")

    instance = generate_fused_planted(args.qubits, seed=args.seed, block_size=10, alpha=0.1)
    surrogate = build_external_surrogate(instance)
    n = len(surrogate.variables)
    logger.info("instance %s: %d variables, planted optimum known", instance.name, n)

    # --- 1. the exportable circuit, run on Aer -----------------------------
    circuit = build_gate_level_qaoa_circuit(
        surrogate, betas=betas, gammas=gammas, mixer=args.mixer, penalty_free=True
    )
    measured = circuit.copy()
    measured.measure_all()
    backend = AerSimulator()
    counts = backend.run(
        transpile(measured, backend), shots=args.shots, seed_simulator=args.seed
    ).result().get_counts()

    # measure_all appends a fresh classical register; the barrier-separated
    # creg label is dropped by get_counts, so keys are plain n-bit strings.
    decoded = decode_counts(surrogate, counts, total_shots=args.shots)
    best = min(decoded, key=lambda result: result.objective)
    aer_expectation = sum(result.sampling_prob * result.objective for result in decoded)

    # --- 2. the same circuit under the exact numpy engine ------------------
    energies = build_energy_vector(surrogate, penalty_mode="none")
    state = simulate_qaoa(
        _initial_state(surrogate, args.mixer, n),
        energies,
        betas,
        gammas,
        mixer=args.mixer,
    )
    exact_expectation = expected_energy_of_state(state, energies)

    # --- 3. do they agree? -------------------------------------------------
    # Monte-Carlo standard error of the mean over the sampled distribution.
    spread = math.sqrt(
        sum(
            result.sampling_prob * (result.objective - aer_expectation) ** 2
            for result in decoded
        )
    )
    standard_error = spread / math.sqrt(args.shots)
    difference = abs(aer_expectation - exact_expectation)
    sigmas = difference / standard_error if standard_error > 0 else 0.0
    # 5 sigma: with a few thousand shots this is a loose bound on agreement,
    # and a real disagreement between the two engines is orders of magnitude
    # out, not marginal.
    agrees = sigmas < 5.0

    scale = abs(exact_expectation) or 1.0
    logger.info("Aer   expectation: %+.6e (%d shots)", aer_expectation, args.shots)
    logger.info("exact expectation: %+.6e (numpy engine)", exact_expectation)
    logger.info(
        "difference: %.3e = %.2f sigma (%.4f%% of |exact|) -> %s",
        difference,
        sigmas,
        100.0 * difference / scale,
        "AGREE" if agrees else "DISAGREE",
    )
    logger.info(
        "best sampled: objective %+.6e, %d builds, feasible=%s",
        best.objective,
        sum(best.actual_builds.values()),
        best.feasible,
    )
    logger.info("selected candidates: %s", ", ".join(best.selected_candidates) or "(none)")

    record = {
        "instance_id": instance.name,
        "variable_count": n,
        "qaoa_depth_p": args.depth,
        "mixer": args.mixer,
        "shots": args.shots,
        "betas": list(betas),
        "gammas": list(gammas),
        "aer_expectation": aer_expectation,
        "exact_expectation": exact_expectation,
        "abs_difference": difference,
        "sampling_standard_error": standard_error,
        "sigmas": sigmas,
        "engines_agree": agrees,
        "best_sampled_objective": best.objective,
        "best_sampled_builds": best.selected_candidates,
        "planted_optimum": instance.planted_energy,
        "qasm3_exportable": args.mixer != "xy_ring",
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if record["qasm3_exportable"]:
            qasm_path = out.with_suffix(".qasm")
            qasm_path.write_text(qasm3.dumps(circuit))
            logger.info("wrote %s and %s", out, qasm_path)
        else:
            logger.info("wrote %s (no QASM: %s is not exportable)", out, args.mixer)

    if not agrees:
        raise SystemExit(
            f"engines disagree at {sigmas:.1f} sigma -- the exported circuit is NOT "
            "the benchmarked algorithm; do not trust downstream numbers"
        )


if __name__ == "__main__":
    main()
