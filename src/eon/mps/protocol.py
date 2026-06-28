from __future__ import annotations

import atexit
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from eon.formulations.layer_b import LayerBSurrogate
from eon.formulations.qubo import QuboCompilation, compile_layer_b_qubo_hess
from eon.validation import (
    agentbible_julia_path,
    provenance_path,
    validate_finite_array,
    validate_non_negative_array,
)

_TOGGLE_PATTERN = re.compile(r"^toggle\[(\d+)\]$")

# Coefficients with magnitude at or below this are treated as exactly zero when
# building the Ising model (drops numerical dust from the QUBO -> Ising transform).
_COEFF_ZERO_ATOL = 1e-12
_JULIQAOA_WORKER: _JuliQAOAWorker | None = None
_logger = logging.getLogger(__name__)


class JuliQAOATransportError(RuntimeError):
    pass


class JuliQAOABackendError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MpsOrderingResult:
    ordering: str
    chi_curve: dict[int, float]
    sampled_energy_curve: dict[int, float]
    entropy_curve: dict[int, float]
    entanglement_entropy_profiles: dict[int, list[float]]
    stopped_reason: str
    best_energy: float
    energy_variance: float


# The tropical contraction's largest intermediate tensor holds 2**(contraction
# width) Tropical{Float64} entries (8 bytes each), so sc=28 is ~2 GiB. Above this
# budget the GTN control reports the width but does not contract -- the instance is
# then a heuristic tree-TN-negative, not a solved-easy certificate.
_GTN_MEMORY_SC_BUDGET = 28.0


@dataclass(frozen=True, slots=True)
class TreeTensorControlResult:
    backend: str
    chi_max_reached: int
    chi_curve: dict[int, float]
    sampled_energy_curve: dict[int, float]
    sample_variance_curve: dict[int, float]
    stopped_reason: str
    best_energy: float
    reference_gap: float
    variable_order: tuple[str, ...]
    max_bag_size: int
    # TreeSA-optimized contraction width (log2 of the largest intermediate tensor =
    # log2 of the bond dimension an exact TN needs). within_budget is True when the
    # network was contracted exactly, so best_energy is the certified exact optimum.
    contraction_width: float
    within_budget: bool


@dataclass(frozen=True, slots=True)
class MpsProtocolResult:
    backend: str
    instance: str
    chi_max_reached: int
    ordering_results: list[MpsOrderingResult]
    ordering_sensitivity: float
    exact_ground_energy: float
    exact_qaoa_energy: float
    optimized_angles: tuple[float, ...]
    reference_energy: float
    tree_tn_result: TreeTensorControlResult | None


@dataclass(frozen=True, slots=True)
class _IsingModel:
    variable_names: tuple[str, ...]
    constant: float
    local_fields: dict[str, float]
    couplings: dict[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class _OrderingSpec:
    name: str
    variable_order: tuple[str, ...]


class _JuliQAOAWorker:
    def __init__(self, repo_root: Path) -> None:
        driver_path = repo_root / "src" / "eon" / "mps" / "juliqaoa_driver.jl"
        project_path = repo_root / "julia"
        env = dict(os.environ)
        env["EON_PROVENANCE_PATH"] = str(provenance_path())
        env["EON_AGENTBIBLE_JULIA_PATH"] = str(agentbible_julia_path())
        self._proc = subprocess.Popen(
            [
                "julia",
                f"--project={project_path}",
                "--startup-file=no",
                str(driver_path),
                "--server",
            ],
            cwd=repo_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None
        if self._proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(self._proc.stderr,),
                daemon=True,
            )
            self._stderr_thread.start()
        else:
            self._stderr_thread = None

    def _drain_stderr(self, stream: Any) -> None:
        for line in stream:
            with self._stderr_lock:
                self._stderr_lines.append(line.rstrip())
                self._stderr_lines = self._stderr_lines[-50:]

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines[-10:])

    def request(self, spec: dict[str, object]) -> dict[str, Any]:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise JuliQAOATransportError(
                "Persistent JuliQAOA worker was not created with pipes."
            )
        with self._lock:
            if self._proc.poll() is not None:
                raise JuliQAOATransportError(
                    "Persistent JuliQAOA worker exited unexpectedly.\n"
                    f"{self._stderr_tail()}"
                )
            self._proc.stdin.write(json.dumps(spec) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise JuliQAOATransportError(
                    "Persistent JuliQAOA worker returned no response.\n"
                    f"{self._stderr_tail()}"
                )
        payload = json.loads(line)
        if "error" in payload:
            raise JuliQAOABackendError(str(payload["error"]))
        return payload

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def run_mps_protocol(
    surrogate: LayerBSurrogate,
    *,
    instance_name: str = "layer_b",
    chi_values: tuple[int, ...] = (4, 8, 16, 32, 64),
    chi_limit: int = 64,
    plateau_tolerance: float = 1e-3,
    plateau_window: int = 2,
    qaoa_rounds: int = 1,
    angle_restarts: int = 1,
    angle_iterations: int = 10,
    sample_count: int = 16,
    cutoff: float = 1e-6,
    seed: int = 7,
    reference_energy: float | None = None,
) -> MpsProtocolResult:
    compilation = compile_layer_b_qubo_hess(surrogate)
    ising_model = _compile_qubo_to_ising(compilation, surrogate)
    orderings = _ordering_specs(ising_model)

    try:
        result = _run_juliqaoa_protocol(
            ising_model,
            orderings,
            instance_name=instance_name,
            chi_values=chi_values,
            chi_limit=chi_limit,
            plateau_tolerance=plateau_tolerance,
            plateau_window=plateau_window,
            qaoa_rounds=qaoa_rounds,
            angle_restarts=angle_restarts,
            angle_iterations=angle_iterations,
            sample_count=sample_count,
            cutoff=cutoff,
            seed=seed,
            reference_energy=reference_energy,
        )
    except (
        OSError,
        JuliQAOATransportError,
        JuliQAOABackendError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        result = _exact_fallback_protocol(
            compilation,
            ising_model,
            instance_name,
            chi_values,
            reference_energy=reference_energy,
        )
    # The tensor-network control is an optional backend: a Julia/GTN failure degrades
    # to no control (tree_tn_result=None), which the classifier reads as "tree-TN
    # negativity not established", never as a hardness signal.
    try:
        tree_result = _run_tree_tn_control(
            ising_model,
            seed=seed,
            reference_energy=reference_energy,
        )
    except (
        OSError,
        JuliQAOATransportError,
        JuliQAOABackendError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        _logger.warning(
            "GTN tensor-network control unavailable for %s (%s: %s); tree-TN "
            "negativity cannot be established for this instance.",
            instance_name,
            type(exc).__name__,
            exc,
        )
        tree_result = None
    return replace(result, tree_tn_result=tree_result)


def _run_juliqaoa_protocol(
    ising_model: _IsingModel,
    orderings: list[_OrderingSpec],
    *,
    instance_name: str,
    chi_values: tuple[int, ...],
    chi_limit: int,
    plateau_tolerance: float,
    plateau_window: int,
    qaoa_rounds: int,
    angle_restarts: int,
    angle_iterations: int,
    sample_count: int,
    cutoff: float,
    seed: int,
    reference_energy: float | None,
) -> MpsProtocolResult:
    ordering_results: list[MpsOrderingResult] = []
    exact_ground_energy = float("inf")
    exact_qaoa_energy = float("inf")
    optimized_angles: tuple[float, ...] = ()
    chi_max_reached = 0

    for ordering_index, ordering in enumerate(orderings):
        payload, stopped_reason = _run_ordering_with_adaptive_chi(
            ising_model,
            ordering,
            chi_values=chi_values,
            chi_limit=chi_limit,
            plateau_tolerance=plateau_tolerance,
            plateau_window=plateau_window,
            qaoa_rounds=qaoa_rounds,
            angle_restarts=angle_restarts,
            angle_iterations=angle_iterations,
            sample_count=sample_count,
            cutoff=cutoff,
            seed=seed + ordering_index,
            reference_energy=reference_energy,
        )
        ordering_results.append(_build_ordering_result(ordering.name, payload, stopped_reason))

        candidate_ground = payload.get("exact_ground_energy")
        if candidate_ground is not None and math.isfinite(float(candidate_ground)):
            exact_ground_energy = min(exact_ground_energy, float(candidate_ground))
        candidate_exact_qaoa = payload.get("exact_qaoa_energy")
        if (
            candidate_exact_qaoa is not None
            and math.isfinite(float(candidate_exact_qaoa))
            and float(candidate_exact_qaoa) < exact_qaoa_energy
        ):
            exact_qaoa_energy = float(candidate_exact_qaoa)
            optimized_angles = tuple(float(angle) for angle in payload["angles"])

        chi_results = payload["chi_results"]
        if chi_results:
            chi_max_reached = max(
                chi_max_reached,
                max(int(entry["chi"]) for entry in chi_results),
            )

    if not math.isfinite(exact_ground_energy):
        exact_ground_energy = float(
            reference_energy
            if reference_energy is not None
            else min(
                (result.best_energy for result in ordering_results),
                default=float("nan"),
            )
        )
    if not math.isfinite(exact_qaoa_energy):
        exact_qaoa_energy = float("nan")

    best_energies = [result.best_energy for result in ordering_results]
    ordering_sensitivity = max(best_energies) - min(best_energies) if best_energies else 0.0

    return MpsProtocolResult(
        backend="juliqaoa_mps",
        instance=instance_name,
        chi_max_reached=chi_max_reached,
        ordering_results=ordering_results,
        ordering_sensitivity=ordering_sensitivity,
        exact_ground_energy=exact_ground_energy,
        exact_qaoa_energy=exact_qaoa_energy,
        optimized_angles=optimized_angles,
        reference_energy=(
            float(reference_energy) if reference_energy is not None else exact_ground_energy
        ),
        tree_tn_result=None,
    )


def _run_ordering_with_adaptive_chi(
    ising_model: _IsingModel,
    ordering: _OrderingSpec,
    *,
    chi_values: tuple[int, ...],
    chi_limit: int,
    plateau_tolerance: float,
    plateau_window: int,
    qaoa_rounds: int,
    angle_restarts: int,
    angle_iterations: int,
    sample_count: int,
    cutoff: float,
    seed: int,
    reference_energy: float | None,
) -> tuple[dict[str, Any], str]:
    chi_schedule = _initial_chi_schedule(chi_values, chi_limit)
    if not chi_schedule:
        raise ValueError("chi_values must include at least one positive bond dimension.")

    cached_angles: tuple[float, ...] | None = None
    # Bounded loop (JPL Rule 2): chi at most doubles each step from >=1 toward
    # chi_limit, so the number of doublings cannot exceed chi_limit.bit_length() + 1.
    # The chi_limit / plateau returns below normally exit well before this bound.
    max_chi_doublings = chi_limit.bit_length() + 2
    for _doubling in range(max_chi_doublings):
        payload = _run_juliqaoa_ordering(
            ising_model,
            ordering,
            chi_schedule=chi_schedule,
            qaoa_rounds=qaoa_rounds,
            angle_restarts=angle_restarts,
            angle_iterations=angle_iterations,
            sample_count=sample_count,
            cutoff=cutoff,
            seed=seed,
            angles=cached_angles,
            reference_energy=reference_energy,
        )
        cached_angles = tuple(float(angle) for angle in payload["angles"])

        if _plateau_reached(
            payload["chi_results"],
            tolerance=plateau_tolerance,
            window=plateau_window,
        ):
            return payload, "adaptive_plateau"

        next_chi = min(chi_limit, chi_schedule[-1] * 2)
        if next_chi <= chi_schedule[-1]:
            return payload, "chi_limit"
        chi_schedule = [*chi_schedule, next_chi]

    raise AssertionError(
        f"adaptive-chi loop exceeded its bound of {max_chi_doublings} doublings "
        f"for chi_limit={chi_limit}; this should be unreachable."
    )


def _initial_chi_schedule(chi_values: tuple[int, ...], chi_limit: int) -> list[int]:
    schedule = sorted({int(chi) for chi in chi_values if 0 < int(chi) <= chi_limit})
    if schedule:
        return schedule
    if chi_limit <= 0:
        return []
    return [chi_limit]


def _run_juliqaoa_ordering(
    ising_model: _IsingModel,
    ordering: _OrderingSpec,
    *,
    chi_schedule: list[int],
    qaoa_rounds: int,
    angle_restarts: int,
    angle_iterations: int,
    sample_count: int,
    cutoff: float,
    seed: int,
    angles: tuple[float, ...] | None,
    reference_energy: float | None,
) -> dict[str, Any]:
    index = {name: offset + 1 for offset, name in enumerate(ordering.variable_order)}
    interactions: list[dict[str, object]] = []
    for name in ordering.variable_order:
        coefficient = ising_model.local_fields.get(name, 0.0)
        if abs(coefficient) > _COEFF_ZERO_ATOL:
            interactions.append({"qubits": [index[name]], "weight": coefficient})
    for left, right in sorted(ising_model.couplings):
        coefficient = ising_model.couplings[(left, right)]
        if abs(coefficient) > _COEFF_ZERO_ATOL:
            interactions.append(
                {"qubits": [index[left], index[right]], "weight": coefficient}
            )

    spec: dict[str, object] = {
        "angles": list(angles) if angles is not None else None,
        "angle_iterations": angle_iterations,
        "angle_restarts": angle_restarts,
        "chi_values": chi_schedule,
        "constant": ising_model.constant,
        "cutoff": cutoff,
        "interactions": interactions,
        "nqubits": len(ordering.variable_order),
        "qaoa_rounds": qaoa_rounds,
        "sample_count": sample_count,
        "seed": seed,
        "reference_energy": reference_energy,
        "max_exact_qubits": 24,
    }

    repo_root = Path(__file__).resolve().parents[3]
    try:
        return _get_juliqaoa_worker(repo_root).request(spec)
    except JuliQAOATransportError:
        return _run_juliqaoa_oneshot(repo_root, spec)


def _run_juliqaoa_oneshot(repo_root: Path, spec: dict[str, object]) -> dict[str, Any]:
    driver_path = Path(__file__).with_name("juliqaoa_driver.jl")
    project_path = repo_root / "julia"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        spec_path = Path(handle.name)

    try:
        proc = subprocess.run(
            [
                "julia",
                f"--project={project_path}",
                "--startup-file=no",
                str(driver_path),
                str(spec_path),
            ],
            cwd=repo_root,
            env={
                **os.environ,
                "EON_PROVENANCE_PATH": str(provenance_path()),
                "EON_AGENTBIBLE_JULIA_PATH": str(agentbible_julia_path()),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    finally:
        spec_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise JuliQAOABackendError(
            proc.stderr.strip() or proc.stdout.strip() or "Julia backend failed."
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise JuliQAOATransportError(
            proc.stderr.strip() or "Julia backend returned no JSON payload."
        )
    payload = json.loads(lines[-1])
    if "error" in payload:
        raise JuliQAOABackendError(str(payload["error"]))
    return payload


def _build_ordering_result(
    ordering: str,
    payload: dict[str, Any],
    stopped_reason: str,
) -> MpsOrderingResult:
    chi_curve: dict[int, float] = {}
    sampled_energy_curve: dict[int, float] = {}
    entropy_curve: dict[int, float] = {}
    entanglement_profiles: dict[int, list[float]] = {}
    best_energy = float("inf")
    best_variance = 0.0

    for entry in payload["chi_results"]:
        chi = int(entry["chi"])
        chi_curve[chi] = float(entry["expected_energy"])
        sampled_energy_curve[chi] = float(entry["best_sampled_energy"])
        entropy_curve[chi] = float(entry["max_entropy"])
        entanglement_profiles[chi] = [float(value) for value in entry.get("entropy_profile", [])]
        if sampled_energy_curve[chi] < best_energy:
            best_energy = sampled_energy_curve[chi]
            best_variance = float(entry["sample_variance"])

    validate_non_negative_array(
        np.asarray(sorted(chi_curve), dtype=float),
        name=f"{ordering}_chi_schedule",
    )
    validate_finite_array(
        np.asarray(list(chi_curve.values()), dtype=float),
        name=f"{ordering}_expected_energy_curve",
    )
    validate_finite_array(
        np.asarray(list(sampled_energy_curve.values()), dtype=float),
        name=f"{ordering}_sampled_energy_curve",
    )
    validate_non_negative_array(
        np.asarray(list(entropy_curve.values()), dtype=float),
        name=f"{ordering}_entropy_curve",
    )
    for chi, profile in entanglement_profiles.items():
        validate_non_negative_array(
            np.asarray(profile, dtype=float),
            name=f"{ordering}_entropy_profile_chi{chi}",
        )

    return MpsOrderingResult(
        ordering=ordering,
        chi_curve=chi_curve,
        sampled_energy_curve=sampled_energy_curve,
        entropy_curve=entropy_curve,
        entanglement_entropy_profiles=entanglement_profiles,
        stopped_reason=stopped_reason,
        best_energy=best_energy,
        energy_variance=best_variance,
    )


def _plateau_reached(
    chi_results: list[dict[str, Any]],
    *,
    tolerance: float,
    window: int,
) -> bool:
    if len(chi_results) <= window:
        return False

    sorted_results = sorted(chi_results, key=lambda result: int(result["chi"]))
    running_best = float("inf")
    cumulative_best: list[float] = []
    for result in sorted_results:
        running_best = min(running_best, float(result["best_sampled_energy"]))
        cumulative_best.append(running_best)
    improvements = [
        cumulative_best[index - 1] - cumulative_best[index]
        for index in range(1, len(cumulative_best))
    ]
    return all(improvement <= tolerance for improvement in improvements[-window:])


def _compile_qubo_to_ising(
    compilation: QuboCompilation,
    surrogate: LayerBSurrogate,
) -> _IsingModel:
    variable_names = surrogate.variable_names
    positions = {name: offset for offset, name in enumerate(variable_names)}
    local_fields = {name: 0.0 for name in variable_names}
    couplings: dict[tuple[str, str], float] = {}
    constant = float(compilation.offset)

    for (left_label, right_label), coefficient in compilation.qubo.items():
        if abs(coefficient) <= _COEFF_ZERO_ATOL:
            continue

        left_index = _toggle_index(left_label)
        right_index = _toggle_index(right_label)
        left_name = variable_names[left_index]
        right_name = variable_names[right_index]

        if left_index == right_index:
            constant += coefficient / 2.0
            local_fields[left_name] -= coefficient / 2.0
            continue

        constant += coefficient / 4.0
        local_fields[left_name] -= coefficient / 4.0
        local_fields[right_name] -= coefficient / 4.0

        key = (
            (left_name, right_name)
            if positions[left_name] <= positions[right_name]
            else (right_name, left_name)
        )
        couplings[key] = couplings.get(key, 0.0) + coefficient / 4.0

    local_fields = {
        name: value for name, value in local_fields.items() if abs(value) > _COEFF_ZERO_ATOL
    }
    couplings = {edge: value for edge, value in couplings.items() if abs(value) > _COEFF_ZERO_ATOL}
    return _IsingModel(
        variable_names=variable_names,
        constant=constant,
        local_fields=local_fields,
        couplings=couplings,
    )


def _ordering_specs(ising_model: _IsingModel) -> list[_OrderingSpec]:
    graph = nx.Graph()
    graph.add_nodes_from(ising_model.variable_names)
    for (left, right), weight in ising_model.couplings.items():
        graph.add_edge(left, right, weight=abs(weight))

    positions = {name: offset for offset, name in enumerate(ising_model.variable_names)}
    orderings: list[_OrderingSpec] = []
    seen: set[tuple[str, ...]] = set()

    def add_ordering(name: str, order: tuple[str, ...]) -> None:
        if order in seen:
            return
        seen.add(order)
        orderings.append(_OrderingSpec(name=name, variable_order=order))

    add_ordering("identity", tuple(ising_model.variable_names))
    if graph.number_of_edges() == 0:
        if len(ising_model.variable_names) > 1:
            add_ordering("reverse_identity", tuple(reversed(ising_model.variable_names)))
        return orderings

    root = min(
        ising_model.variable_names,
        key=lambda node: (-graph.degree(node), positions[node]),
    )
    bfs_order = tuple(_complete_order(graph, nx.bfs_tree(graph, root).nodes()))
    add_ordering("bfs", bfs_order)

    dfs_order = tuple(_complete_order(graph, nx.dfs_preorder_nodes(graph, root)))
    add_ordering("dfs", dfs_order)

    degree_order = tuple(
        sorted(
            ising_model.variable_names,
            key=lambda node: (-graph.degree(node), positions[node]),
        )
    )
    add_ordering("degree", degree_order)

    cuthill_mckee = tuple(_complete_order(graph, nx.utils.reverse_cuthill_mckee_ordering(graph)))
    add_ordering("cuthill_mckee", cuthill_mckee)
    if len(cuthill_mckee) > 1:
        add_ordering("reverse_cuthill_mckee", tuple(reversed(cuthill_mckee)))
    return orderings


def _run_tree_tn_control(
    ising_model: _IsingModel,
    *,
    seed: int,
    reference_energy: float | None,
    memory_sc_budget: float = _GTN_MEMORY_SC_BUDGET,
) -> TreeTensorControlResult | None:
    """Real tensor-network control via GenericTensorNetworks.jl.

    Contracts the Ising/QUBO as a tensor network. TreeSA finds a near-optimal
    contraction order whose space complexity (contraction width) is the log2 bond
    dimension an exact TN needs. When that width is within the memory budget the
    network is contracted exactly in the tropical semiring for the EXACT ground
    energy -- ground truth for scoring the bounded-chi MPS sweep, and a rigorous
    TN-easy certificate. A width above the chi budget is only a HEURISTIC hardness
    signal (TreeSA returns an upper bound on the optimal width).
    """
    if not ising_model.variable_names:
        return None

    repo_root = Path(__file__).resolve().parents[3]
    spec = _gtn_control_spec(ising_model, seed=seed, memory_sc_budget=memory_sc_budget)
    payload = _run_gtn_control_oneshot(repo_root, spec)

    contraction_width = float(payload["contraction_width"])
    within_budget = bool(payload["within_budget"])
    exact_ground = payload.get("exact_ground_energy")
    best_energy = float(exact_ground) if exact_ground is not None else float("inf")
    baseline = float(reference_energy) if reference_energy is not None else best_energy
    reference_gap = best_energy - baseline if math.isfinite(best_energy) else 0.0
    # chi_required (= 2**ceil(sc)) is reported only for contracted instances; it is
    # null for over-budget ones (and would overflow Float64 at absurd widths).
    chi_required = payload.get("chi_required")
    chi_max_reached = (
        int(chi_required) if chi_required is not None and math.isfinite(chi_required) else 0
    )

    return TreeTensorControlResult(
        backend=str(payload["backend"]),
        chi_max_reached=chi_max_reached,
        chi_curve={},
        sampled_energy_curve={},
        sample_variance_curve={},
        stopped_reason="contracted" if within_budget else "over_memory_budget",
        best_energy=best_energy,
        reference_gap=reference_gap,
        variable_order=tuple(ising_model.variable_names),
        max_bag_size=int(math.ceil(contraction_width)),
        contraction_width=contraction_width,
        within_budget=within_budget,
    )


def _gtn_control_spec(
    ising_model: _IsingModel, *, seed: int, memory_sc_budget: float
) -> dict[str, object]:
    index = {name: offset + 1 for offset, name in enumerate(ising_model.variable_names)}
    interactions: list[dict[str, object]] = []
    for name in ising_model.variable_names:
        coefficient = ising_model.local_fields.get(name, 0.0)
        if abs(coefficient) > _COEFF_ZERO_ATOL:
            interactions.append({"qubits": [index[name]], "weight": coefficient})
    for left, right in sorted(ising_model.couplings):
        coefficient = ising_model.couplings[(left, right)]
        if abs(coefficient) > _COEFF_ZERO_ATOL:
            interactions.append({"qubits": [index[left], index[right]], "weight": coefficient})
    return {
        "nqubits": len(ising_model.variable_names),
        "constant": ising_model.constant,
        "interactions": interactions,
        "memory_sc_budget": memory_sc_budget,
        "seed": seed,
    }


def _run_gtn_control_oneshot(repo_root: Path, spec: dict[str, object]) -> dict[str, Any]:
    driver_path = Path(__file__).with_name("gtn_control_driver.jl")
    project_path = repo_root / "julia"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(spec, handle)
        spec_path = Path(handle.name)

    try:
        proc = subprocess.run(
            [
                "julia",
                f"--project={project_path}",
                "--startup-file=no",
                str(driver_path),
                str(spec_path),
            ],
            cwd=repo_root,
            env={
                **os.environ,
                "EON_PROVENANCE_PATH": str(provenance_path()),
                "EON_AGENTBIBLE_JULIA_PATH": str(agentbible_julia_path()),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    finally:
        spec_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise JuliQAOABackendError(
            proc.stderr.strip() or proc.stdout.strip() or "GTN control failed."
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise JuliQAOATransportError(
            proc.stderr.strip() or "GTN control returned no JSON payload."
        )
    payload = json.loads(lines[-1])
    if "error" in payload:
        raise JuliQAOABackendError(str(payload["error"]))
    return payload


def _complete_order(graph: nx.Graph, ordering: Any) -> list[str]:
    ordered = list(ordering)
    missing = [node for node in graph.nodes if node not in ordered]
    return [*ordered, *missing]


def _toggle_index(label: str) -> int:
    match = _TOGGLE_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"Unexpected QUBO variable label: {label}")
    return int(match.group(1))


def _exact_fallback_protocol(
    compilation: QuboCompilation,
    ising_model: _IsingModel,
    instance_name: str,
    chi_values: tuple[int, ...],
    *,
    reference_energy: float | None,
) -> MpsProtocolResult:
    best_energy = _exact_qubo_ground_energy(compilation, len(ising_model.variable_names))
    if math.isnan(best_energy):
        best_energy = float(reference_energy) if reference_energy is not None else float("nan")
    ordering_results: list[MpsOrderingResult] = []
    for ordering in _ordering_specs(ising_model):
        chi_curve = {chi: best_energy for chi in chi_values}
        sampled_curve = {chi: best_energy for chi in chi_values}
        entropy_curve = {chi: 0.0 for chi in chi_values}
        profiles: dict[int, list[float]] = {chi: [] for chi in chi_values}
        ordering_results.append(
            MpsOrderingResult(
                ordering=ordering.name,
                chi_curve=chi_curve,
                sampled_energy_curve=sampled_curve,
                entropy_curve=entropy_curve,
                entanglement_entropy_profiles=profiles,
                stopped_reason="fallback_exact",
                best_energy=best_energy,
                energy_variance=0.0,
            )
        )

    return MpsProtocolResult(
        backend="exact_fallback",
        instance=instance_name,
        chi_max_reached=max(chi_values),
        ordering_results=ordering_results,
        ordering_sensitivity=0.0,
        exact_ground_energy=best_energy,
        exact_qaoa_energy=best_energy,
        optimized_angles=(),
        reference_energy=float(reference_energy) if reference_energy is not None else best_energy,
        tree_tn_result=None,
    )


def _exact_qubo_ground_energy(compilation: QuboCompilation, variable_count: int) -> float:
    if variable_count > 24:
        return float("nan")
    coefficients = [
        (_toggle_index(left), _toggle_index(right), coefficient)
        for (left, right), coefficient in compilation.qubo.items()
    ]
    best_energy = float("inf")
    for bits in product((0, 1), repeat=variable_count):
        energy = compilation.offset
        for left, right, coefficient in coefficients:
            energy += coefficient * bits[left] * bits[right]
        best_energy = min(best_energy, float(energy))
    return best_energy


def _get_juliqaoa_worker(repo_root: Path) -> _JuliQAOAWorker:
    global _JULIQAOA_WORKER
    if _JULIQAOA_WORKER is None or _JULIQAOA_WORKER._proc.poll() is not None:
        _JULIQAOA_WORKER = _JuliQAOAWorker(repo_root)
        atexit.register(_shutdown_juliqaoa_worker)
    return _JULIQAOA_WORKER


def _shutdown_juliqaoa_worker() -> None:
    global _JULIQAOA_WORKER
    if _JULIQAOA_WORKER is not None:
        _JULIQAOA_WORKER.close()
        _JULIQAOA_WORKER = None
