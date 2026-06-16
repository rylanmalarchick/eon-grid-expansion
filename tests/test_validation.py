import json
from pathlib import Path

import numpy as np
import pytest
from agentbible.errors import ValidationError
from agentbible.provenance import validate_provenance_record

import eon.validation as validation


def _load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_validation_emits_schema_compliant_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "provenance.jsonl"
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", target)

    validation.validate_finite_array(np.asarray([1.0, 2.0, 3.0]), name="smoke_vector")

    records = _load_records(target)
    assert records
    assert validate_provenance_record(records[-1]) == []
    assert records[-1]["checks_passed"][0]["check_name"] == "finite_array"
    assert records[-1]["checks_passed"][0]["passed"] is True


def test_validation_records_failed_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "provenance.jsonl"
    monkeypatch.setattr(validation, "_PROVENANCE_PATH", target)

    with pytest.raises(ValidationError):
        validation.validate_probability_array(
            np.asarray([0.8, 0.3], dtype=float),
            name="bad_probabilities",
        )

    records = _load_records(target)
    assert records
    assert validate_provenance_record(records[-1]) == []
    assert any(check["passed"] is False for check in records[-1]["checks_passed"])
