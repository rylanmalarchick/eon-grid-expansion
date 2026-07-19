from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentbible import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    check_finite_array,
    check_hermitian,
    check_non_negative_array,
    check_normalized_l1,
    check_positive_semidefinite,
    check_probability_array,
    check_symmetric,
    check_unitary,
)
from agentbible.provenance import (
    CheckResult,
    build_provenance_record,
    get_provenance_metadata,
)
from numpy.typing import ArrayLike

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "experiments" / "results"
_FINDINGS_ROOT = _REPO_ROOT / "experiments" / "logs" / "agentbible_findings"
_RUN_ID = os.environ.get("EON_RUN_ID") or datetime.now(UTC).strftime(
    "%Y%m%dT%H%M%SZ"
)
_PROVENANCE_PATH = Path(
    os.environ.get(
        "EON_PROVENANCE_PATH",
        str(_RESULTS_ROOT / _RUN_ID / "provenance.jsonl"),
    )
)
_AGENTBIBLE_JULIA_PATH = Path.home() / "dev" / "oss" / "agentbible" / "languages" / "julia"
_PROVENANCE_LOCK = threading.Lock()
_BASE_METADATA = get_provenance_metadata(
    description="EON numerical validation",
    include_pip_freeze=False,
    include_hardware=False,
)


@dataclass(frozen=True, slots=True)
class _CheckSpec:
    check_name: str
    norm_used: str
    rtol: float
    atol: float
    validator: Callable[..., Any]
    validator_kwargs: dict[str, Any]


def provenance_path() -> Path:
    _PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _PROVENANCE_PATH


def agentbible_julia_path() -> Path:
    return _AGENTBIBLE_JULIA_PATH


def write_agentbible_finding(name: str, body: str) -> Path:
    _FINDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    target = _FINDINGS_ROOT / name
    target.write_text(body.rstrip() + "\n", encoding="utf-8")
    return target


def validate_finite_array(value: ArrayLike, *, name: str) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="finite_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_finite_array,
                validator_kwargs={},
            )
        ],
        name=name,
    )


def validate_non_negative_array(
    value: ArrayLike,
    *,
    name: str,
    atol: float = 0.0,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="non_negative_array",
                norm_used="n/a",
                rtol=0.0,
                atol=atol,
                validator=check_non_negative_array,
                validator_kwargs={"atol": atol},
            )
        ],
        name=name,
    )


def validate_normalized_l1(
    value: ArrayLike,
    *,
    name: str,
    atol: float = 1e-10,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="normalized_l1",
                norm_used="l1",
                rtol=0.0,
                atol=atol,
                validator=check_normalized_l1,
                validator_kwargs={"atol": atol},
            )
        ],
        name=name,
    )


def validate_probability_array(value: ArrayLike, *, name: str) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="probability_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_probability_array,
                validator_kwargs={},
            ),
            _CheckSpec(
                check_name="normalized_l1",
                norm_used="l1",
                rtol=0.0,
                atol=1e-10,
                validator=check_normalized_l1,
                validator_kwargs={"atol": 1e-10},
            ),
        ],
        name=name,
    )


def validate_symmetric_matrix(
    value: ArrayLike,
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="finite_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_finite_array,
                validator_kwargs={},
            ),
            _CheckSpec(
                check_name="symmetric",
                norm_used="max_elementwise",
                rtol=0.0,
                atol=atol,
                validator=check_symmetric,
                validator_kwargs={"atol": atol},
            ),
        ],
        name=name,
    )


def validate_hermitian_matrix(
    value: ArrayLike,
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="finite_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_finite_array,
                validator_kwargs={},
            ),
            _CheckSpec(
                check_name="hermitian",
                norm_used="max_elementwise",
                rtol=0.0,
                atol=atol,
                validator=check_hermitian,
                validator_kwargs={"atol": atol},
            ),
        ],
        name=name,
    )


def validate_unitary_matrix(
    value: ArrayLike,
    *,
    name: str,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="finite_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_finite_array,
                validator_kwargs={},
            ),
            _CheckSpec(
                check_name="unitary",
                norm_used="frobenius",
                rtol=rtol,
                atol=atol,
                validator=check_unitary,
                validator_kwargs={"rtol": rtol, "atol": atol},
            ),
        ],
        name=name,
    )


def validate_positive_semidefinite_matrix(
    value: ArrayLike,
    *,
    name: str,
    atol: float = DEFAULT_ATOL,
) -> None:
    _run_checks(
        value,
        [
            _CheckSpec(
                check_name="finite_array",
                norm_used="n/a",
                rtol=0.0,
                atol=0.0,
                validator=check_finite_array,
                validator_kwargs={},
            ),
            _CheckSpec(
                check_name="positive_semidefinite",
                norm_used="n/a",
                rtol=0.0,
                atol=atol,
                validator=check_positive_semidefinite,
                validator_kwargs={"atol": atol},
            ),
        ],
        name=name,
    )


def _run_checks(value: ArrayLike, specs: list[_CheckSpec], *, name: str) -> None:
    check_results: list[CheckResult] = []
    for spec in specs:
        try:
            spec.validator(value, name=name, strict=True, **spec.validator_kwargs)
        except Exception as exc:
            # Deliberate broad catch: record a failed CheckResult for provenance
            # regardless of which validator failed, then re-raise. The root cause is
            # preserved by the bare `raise` below, so nothing is swallowed.
            check_results.append(
                CheckResult(
                    check_name=spec.check_name,
                    passed=False,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    norm_used=spec.norm_used,
                    error_message=str(exc),
                )
            )
            _append_record(check_results)
            raise
        check_results.append(
            CheckResult(
                check_name=spec.check_name,
                passed=True,
                rtol=spec.rtol,
                atol=spec.atol,
                norm_used=spec.norm_used,
                error_message=None,
            )
        )
    _append_record(check_results)


def _append_record(checks: list[CheckResult]) -> None:
    metadata = dict(_BASE_METADATA)
    metadata["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    record = build_provenance_record(
        language="python",
        checks_passed=checks,
        metadata=metadata,
    ).to_dict()
    payload = json.dumps(record, sort_keys=True)
    with _PROVENANCE_LOCK:
        target = provenance_path()
        with target.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
