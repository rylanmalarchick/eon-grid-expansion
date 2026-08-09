"""S4: logical performance per PHYSICAL gate (cop-QAOA vs vanilla).

A logically efficient ansatz is not automatically the cheaper one to run.
cop-QAOA searches only the feasible subspace, but pays for it physically:
fixed-Hamming-weight state preparation plus an XY ring whose two-qubit gates
must be routed onto a device topology. Vanilla penalty-QAOA has a product
initial state and a single-qubit mixer -- physically cheap, logically wasteful.
This script measures that trade explicitly: circuit depth and two-qubit gate
count after transpilation, across optimizer levels (the compiler-flag analogy).

HONESTY: no Braket/IQM SDK is installed here, so nothing below was executed on
a device or verified against a real calibration. The target is a documented
IQM-Garnet-LIKE model -- 20 qubits, {r, cz} native basis, square-lattice
nearest-neighbour coupling -- clearly an approximation of the published device
family, not a device-verified target. OpenQASM 3 is emitted alongside as the
portable artifact (Braket ingests OpenQASM 3).

Run from workspace/:
    python scripts/transpile_table.py [--depth 2] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.transpiler import CouplingMap

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.qubo import compile_external_qubo, compile_layer_b_qubo_hess
from eon.instances.external import build_external_surrogate, generate_fused_planted
from eon.quantum.export import build_gate_level_qaoa_circuit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("transpile_table")

# IQM-Garnet-LIKE: 20 qubits, square lattice, native {r (phased RX), cz}.
# APPROXIMATION of the published device family -- not device-verified.
GARNET_LIKE_BASIS = ["r", "cz"]
GARNET_LIKE_ROWS = 4
GARNET_LIKE_COLS = 5
TARGET_NOTE = (
    "IQM-Garnet-like: 20 qubits, {r, cz} basis, 4x5 square-lattice coupling. "
    "APPROXIMATION of the published device family; not device-verified, no SDK "
    "installed, nothing executed on hardware."
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _metrics(circuit: QuantumCircuit) -> dict[str, object]:
    counts = dict(circuit.count_ops())
    two_qubit = sum(
        count
        for name, count in counts.items()
        if name in {"cz", "cx", "ecr", "rzz", "rxx", "ryy", "xx_plus_yy"}
    )
    return {
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gates": two_qubit,
        "ops": counts,
    }


def _row(
    surrogate: LayerBSurrogate,
    label: str,
    mixer: str,
    *,
    depth: int,
    penalty_free: bool,
    coupling: CouplingMap,
    levels: tuple[int, ...],
) -> dict[str, object]:
    betas = tuple(0.3 for _ in range(depth))
    gammas = tuple(0.7 for _ in range(depth))
    logical = build_gate_level_qaoa_circuit(
        surrogate, betas=betas, gammas=gammas, mixer=mixer, penalty_free=penalty_free
    )
    row: dict[str, object] = {
        "algorithm": label,
        "mixer": mixer,
        "penalty_free": penalty_free,
        "cost_graph_edges": sum(
            1 for (a, b) in (
                compile_external_qubo(surrogate) if penalty_free
                else compile_layer_b_qubo_hess(surrogate)
            ).qubo if a != b
        ),
        "qaoa_depth_p": depth,
        "logical": _metrics(logical),
        "transpiled": {},
    }
    for level in levels:
        try:
            started = time.perf_counter()
            physical = transpile(
                logical,
                basis_gates=GARNET_LIKE_BASIS,
                coupling_map=coupling,
                optimization_level=level,
                seed_transpiler=7,
            )
            metrics = _metrics(physical)
            metrics["transpile_seconds"] = round(time.perf_counter() - started, 1)
            row["transpiled"][str(level)] = metrics  # type: ignore[index]
            logger.info(
                "  %s opt%d: depth=%s 2q=%s (%.0fs)",
                label, level, metrics["depth"], metrics["two_qubit_gates"],
                metrics["transpile_seconds"],
            )
        # Broad catch: record the per-level failure and keep the other levels.
        except Exception as exc:
            logger.exception("transpile failed for %s at level %d", label, level)
            row["transpiled"][str(level)] = {  # type: ignore[index]
                "error": f"{type(exc).__name__}: {exc}"
            }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else Path(f"experiments/results/s4_transpile_{stamp}/table.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qasm_dir = out_path.parent / "qasm3"
    qasm_dir.mkdir(exist_ok=True)

    coupling = CouplingMap.from_grid(GARNET_LIKE_ROWS, GARNET_LIKE_COLS)
    # A 20-variable planted instance: real, instantly reproducible, and needs
    # no Layer A solve. Its coupling density sets the gate counts.
    instance = generate_fused_planted(20, seed=7, block_size=10, alpha=0.1)
    surrogate = build_external_surrogate(instance)

    # Each arm is transpiled AS IT WOULD ACTUALLY RUN, which is the only
    # comparison that says anything about hardware cost.
    #
    # Penalty QAOA needs the cardinality penalty in its cost operator -- that is
    # what makes it "penalty" QAOA. The Hess (sum-K)^2 term is all-pairs, so its
    # cost graph is COMPLETE (190 edges at n=20), not the 129 of the raw
    # instance. The constrained mixer enforces cardinality by construction and
    # so runs penalty-free.
    #
    # Both arms were previously transpiled with penalty_free=True, i.e. the
    # penalty deleted from the penalty ansatz. That made the two cost layers
    # identical by construction and understated the arm the proposal went on to
    # recommend.
    rows = []
    for label, mixer, penalty_free in (
        ("vanilla", "x", False),
        ("cop", "xy_ring", True),
    ):
        rows.append(
            _row(surrogate, label, mixer, depth=args.depth, penalty_free=penalty_free,
                 coupling=coupling, levels=levels)
        )
        # Save after each algorithm: transpiling a densely-coupled QAOA circuit
        # onto a sparse lattice can take tens of minutes per level.
        out_path.write_text(json.dumps({"rows": rows, "partial": True}, indent=2) + "\n")

    for row in rows:
        circuit = build_gate_level_qaoa_circuit(
            surrogate,
            betas=tuple(0.3 for _ in range(args.depth)),
            gammas=tuple(0.7 for _ in range(args.depth)),
            mixer=str(row["mixer"]),
            penalty_free=bool(row["penalty_free"]),
        )
        path = qasm_dir / f"{row['algorithm']}_p{args.depth}_n20.qasm"
        try:
            path.write_text(qasm3.dumps(circuit))
            row["qasm3_path"] = str(path)
        # cop's fixed-weight state prep is an `initialize`, which QASM3 export
        # rejects; record that rather than pretending the artifact exists.
        except (qasm3.QASM3ExporterError, TypeError, ValueError) as exc:
            # Truncate: the exporter echoes the whole 2^20 statevector.
            row["qasm3_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
            logger.warning("OpenQASM 3 export failed for %s: %s", row["algorithm"], exc)

    payload = {
        "instance_id": instance.name,
        "variable_count": len(surrogate.variables),
        "coupling_count": len(instance.quadratic),
        "target": TARGET_NOTE,
        "rows": rows,
        "run": {
            "git_commit": _git_commit(),
            "timestamp_utc": stamp,
            "levels": list(levels),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    logger.info("=== logical performance per physical gate (p=%d, n=20) ===", args.depth)
    for row in rows:
        logical = row["logical"]
        logger.info(
            "%-8s logical: depth=%s 2q=%s", row["algorithm"],
            logical["depth"], logical["two_qubit_gates"],  # type: ignore[index]
        )
        for level in levels:
            m = row["transpiled"][str(level)]  # type: ignore[index]
            if "error" in m:
                logger.info("           opt%d: FAILED %s", level, m["error"][:60])
            else:
                logger.info(
                    "           opt%d: depth=%s 2q=%s total=%s",
                    level, m["depth"], m["two_qubit_gates"], m["size"],
                )
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
