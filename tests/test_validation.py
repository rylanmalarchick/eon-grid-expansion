"""Validation must raise, and must leave a record of what it checked.

Both halves matter. A checker that records a run but never rejects bad input is
decoration; a checker that rejects but leaves no trail cannot support a
reproducibility claim. These tests pin both, including that a FAILED check is
written out rather than swallowed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import eon.validation as validation

_REQUIRED_FIELDS = {"check_name", "passed", "rtol", "atol", "norm_used", "error_message"}


def _load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_wellformed(record: dict[str, object]) -> None:
    assert "checks_passed" in record, record
    checks = record["checks_passed"]
    assert isinstance(checks, list) and checks
    for check in checks:
        assert set(check) >= _REQUIRED_FIELDS, f"missing fields: {_REQUIRED_FIELDS - set(check)}"
        assert isinstance(check["passed"], bool)


def test_validation_emits_a_wellformed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "provenance.jsonl"
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", target)

    validation.validate_finite_array(np.asarray([1.0, 2.0, 3.0]), name="smoke_vector")

    records = _load_records(target)
    assert records
    _assert_wellformed(records[-1])
    assert records[-1]["checks_passed"][0]["check_name"] == "finite_array"
    assert records[-1]["checks_passed"][0]["passed"] is True


def test_a_failed_check_raises_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The failure path is the one that matters: a check that fails silently,
    or that fails without leaving a trace, is worse than no check at all."""
    target = tmp_path / "provenance.jsonl"
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", target)

    with pytest.raises(ValueError):
        validation.validate_probability_array(
            np.asarray([0.8, 0.3], dtype=float), name="bad_probabilities"
        )

    records = _load_records(target)
    assert records
    _assert_wellformed(records[-1])
    assert any(check["passed"] is False for check in records[-1]["checks_passed"])


@pytest.mark.parametrize(
    "validator,bad_value",
    [
        ("validate_finite_array", np.asarray([1.0, np.nan])),
        ("validate_non_negative_array", np.asarray([1.0, -2.0])),
        ("validate_probability_array", np.asarray([0.5, 1.7])),
        ("validate_normalized_l1", np.asarray([0.5, 0.2])),
    ],
)
def test_every_validator_rejects_its_own_bad_input(
    validator: str, bad_value: np.ndarray, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Coverage against the failure this project has been burned by: a suite
    that is green because the checks cannot fail."""
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", tmp_path / "provenance.jsonl")
    with pytest.raises(ValueError):
        getattr(validation, validator)(bad_value, name="deliberately_bad")


def test_validators_accept_good_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", tmp_path / "provenance.jsonl")
    validation.validate_finite_array(np.asarray([1.0, 2.0]), name="ok")
    validation.validate_non_negative_array(np.asarray([0.0, 2.0]), name="ok")
    validation.validate_probability_array(np.asarray([0.25, 0.75]), name="ok")
    validation.validate_normalized_l1(np.asarray([0.25, 0.75]), name="ok")
