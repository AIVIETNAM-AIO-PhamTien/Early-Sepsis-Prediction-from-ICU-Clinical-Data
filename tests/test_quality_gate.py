from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.validation.quality_gate import require_quality, validate_silver


def test_quality_gate_accepts_valid_patient_data(tmp_path, schema, valid_silver_frame) -> None:
    path = tmp_path / "patients.parquet"
    valid_silver_frame.to_parquet(path, index=False)

    report = validate_silver(path, schema)

    assert report["is_valid"] is True
    assert report["n_patients"] == 2
    assert report["sepsis_patient_rate"] == 0.5


def test_quality_gate_rejects_duplicate_patient_hour(tmp_path, schema, valid_silver_frame) -> None:
    duplicated = valid_silver_frame.iloc[[0]].copy()
    frame = pd.concat([valid_silver_frame, duplicated], ignore_index=True)
    path = tmp_path / "patients.parquet"
    frame.to_parquet(path, index=False)

    report = validate_silver(path, schema)

    assert report["is_valid"] is False
    with pytest.raises(ValueError, match="Data quality gate failed"):
        require_quality(report)


def test_quality_gate_rejects_non_binary_label(tmp_path, schema, valid_silver_frame) -> None:
    valid_silver_frame.loc[0, schema["label_column"]] = 2
    path = tmp_path / "patients.parquet"
    valid_silver_frame.to_parquet(path, index=False)

    report = validate_silver(path, schema)

    assert report["is_valid"] is False
    assert any("invalid values" in error for error in report["errors"])
