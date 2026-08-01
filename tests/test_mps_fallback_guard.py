"""run_mps_protocol must never fabricate results above the exact-fallback limit.

The exact fallback exists for small instances where brute force is real; above
24 variables a backend failure previously substituted reference_energy into
every curve (excess-over-reference exactly 0.0 -- a phantom result that
poisoned 5 of 6 scale-sweep records on 2026-07-21)."""

from __future__ import annotations

import pytest

from eon.instances.external import build_external_surrogate, generate_longrange_spin_glass
from eon.mps import protocol
from eon.mps.protocol import JuliQAOABackendError, run_mps_protocol


def _failing_backend(*args, **kwargs):
    raise JuliQAOABackendError("simulated julia crash")


def test_backend_failure_above_exact_limit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = generate_longrange_spin_glass(30, seed=7, mean_degree=4.0)
    surrogate = build_external_surrogate(instance)
    monkeypatch.setattr(protocol, "_run_juliqaoa_protocol", _failing_backend)
    with pytest.raises(JuliQAOABackendError, match="no exact fallback"):
        run_mps_protocol(surrogate, penalty_free=True, reference_energy=-1.0)


def test_backend_failure_within_exact_limit_still_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = generate_longrange_spin_glass(8, seed=7, mean_degree=3.0)
    surrogate = build_external_surrogate(instance)
    monkeypatch.setattr(protocol, "_run_juliqaoa_protocol", _failing_backend)
    result = run_mps_protocol(surrogate, penalty_free=True, reference_energy=-1.0)
    assert result.backend == "exact_fallback"  # real brute force at n=8
