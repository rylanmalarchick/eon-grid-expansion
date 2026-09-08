import functools
import shutil
import subprocess
from pathlib import Path

import pytest

from eon.mps.juliqaoa_smoke import _synthetic_surrogate
from eon.mps.protocol import run_mps_protocol

_JULIA_PROJECT = Path(__file__).resolve().parents[1] / "julia"
_JULIA_PACKAGES = (
    "JSON3, JuliQAOA, ITensorMPS, ITensors, Optim, GenericTensorNetworks, Graphs"
)


@functools.cache
def _julia_backend_unavailable() -> str:
    """Return the reason the Julia backend cannot run, or "" when it can.

    The binary on PATH is not the dependency. The GitHub runner image ships
    julia without this project's packages, `using JSON3` fails inside the
    driver, and run_mps_protocol degrades to the exact brute force at small n.
    The test then asserted the fallback's backend name and failed for a missing
    package rather than a code defect. Probe the packages the drivers actually
    load.
    """
    if shutil.which("julia") is None:
        return "needs the julia toolchain (JuliQAOA/GenericTensorNetworks)"
    try:
        probe = subprocess.run(
            [
                "julia",
                f"--project={_JULIA_PROJECT}",
                "--startup-file=no",
                "-e",
                f"using {_JULIA_PACKAGES}",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return f"julia is installed but {_JULIA_PROJECT} did not load in 900 s"
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()
        return (
            f"julia is installed but the packages in {_JULIA_PROJECT} are not: "
            f"{detail[0] if detail else 'unknown load error'}"
        )
    return ""


needs_julia = pytest.mark.skipif(
    bool(_julia_backend_unavailable()),
    reason=_julia_backend_unavailable() or "julia toolchain present",
)


@needs_julia
@pytest.mark.requires_gurobi
def test_mps_protocol_smoke_runs() -> None:
    surrogate = _synthetic_surrogate("easy_path_n8")
    result = run_mps_protocol(
        surrogate,
        instance_name="easy_path_n8",
        chi_values=(4,),
        chi_limit=4,
        angle_iterations=2,
        sample_count=2,
    )
    assert result.backend == "juliqaoa_mps"
    assert result.chi_max_reached == 4
    assert len(result.ordering_results) >= 2
    assert result.tree_tn_result is not None
    assert result.tree_tn_result.backend == "generic_tn_tropical"
    assert result.tree_tn_result.within_budget is True
    # Two independent exact methods on the same compiled QUBO must agree: JuliQAOA's
    # brute-force enumeration (exact_ground_energy) and GTN's tropical contraction
    # (tree control best_energy). This cross-validates the GTN backend end to end.
    # Both are exact computations of the same QUBO ground energy, so they must agree
    # to ~machine precision, not the loose default rel=1e-6 (CLAUDE.md tolerance rule).
    assert result.tree_tn_result.best_energy == pytest.approx(
        result.exact_ground_energy, rel=1e-9, abs=1e-9
    )
    assert result.exact_ground_energy <= min(
        ordering.best_energy for ordering in result.ordering_results
    )
