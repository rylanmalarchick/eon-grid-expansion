"""Numerical checks and provenance records.

These were originally a thin wrapper over a local package that could not be
installed from any index, which meant an external reviewer could not run the
validation at all. The implementation now lives here.

The checks RAISE. Nothing here is a no-op: a stand-in that quietly passed would
leave a green suite proving nothing, which is the failure this project has been
burned by before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_name: str
    passed: bool
    rtol: float = 0.0
    atol: float = 0.0
    norm_used: str = "n/a"
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class _Record:
    """Callers use .to_dict() to serialise a run's checks."""

    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def get_provenance_metadata(**_: Any) -> dict[str, Any]:
    """Run metadata. No hardware or package census is collected -- records say
    what was checked and whether it passed, not what machine it ran on."""
    return {"provenance": "inline"}


def build_provenance_record(**kwargs: Any) -> _Record:
    record = dict(kwargs)
    record["provenance_backend"] = "inline"
    checks = record.get("checks_passed") or []
    record["checks_passed"] = [
        {
            "check_name": check.check_name,
            "passed": check.passed,
            "rtol": check.rtol,
            "atol": check.atol,
            "norm_used": check.norm_used,
            "error_message": check.error_message,
        }
        if isinstance(check, CheckResult)
        else check
        for check in checks
    ]
    return _Record(record)


def _as_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name}: array is empty")
    return array


def check_finite_array(value: ArrayLike, *, name: str, strict: bool = True, **_: Any) -> None:
    array = _as_array(value, name)
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(f"{name}: {bad} non-finite value(s) (NaN or Inf)")


def check_non_negative_array(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    if np.any(array < -abs(atol)):
        raise ValueError(f"{name}: minimum {float(array.min())} is negative")


def check_normalized_l1(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    total = float(np.abs(array).sum())
    if not np.isclose(total, 1.0, atol=max(atol, 1e-9), rtol=0.0):
        raise ValueError(f"{name}: L1 norm {total} != 1")


def check_probability_array(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    check_finite_array(array, name=name)
    check_non_negative_array(array, name=name, atol=atol)
    if np.any(array > 1.0 + max(atol, 1e-9)):
        raise ValueError(f"{name}: maximum {float(array.max())} exceeds 1")


def check_symmetric(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name}: not a square matrix, shape {array.shape}")
    if not np.allclose(array, array.T, atol=max(atol, 1e-12), rtol=0.0):
        raise ValueError(f"{name}: matrix is not symmetric")


def check_hermitian(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name}: not a square matrix, shape {array.shape}")
    if not np.allclose(array, array.conj().T, atol=max(atol, 1e-12), rtol=0.0):
        raise ValueError(f"{name}: matrix is not Hermitian")


def check_unitary(
    value: ArrayLike,
    *,
    name: str,
    strict: bool = True,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    **_: Any,
) -> None:
    array = _as_array(value, name)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name}: not a square matrix, shape {array.shape}")
    identity = np.eye(array.shape[0], dtype=complex)
    if not np.allclose(array.conj().T @ array, identity, rtol=rtol, atol=max(atol, 1e-9)):
        raise ValueError(f"{name}: matrix is not unitary")


def check_positive_semidefinite(
    value: ArrayLike, *, name: str, strict: bool = True, atol: float = DEFAULT_ATOL, **_: Any
) -> None:
    array = _as_array(value, name)
    check_hermitian(array, name=name, atol=max(atol, 1e-9))
    eigenvalues = np.linalg.eigvalsh(array)
    if float(eigenvalues.min()) < -max(atol, 1e-9):
        raise ValueError(f"{name}: minimum eigenvalue {float(eigenvalues.min())} < 0")
