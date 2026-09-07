"""End-to-end prediction tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.inference.io import read_patient_file
from src.inference.loader import ModelLoader, ModelLoadError
from src.inference.predictor import run_prediction
from src.shared.paths import PROJECT_ROOT

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_patient.psv"


@pytest.fixture(scope="session")
def sample_patient_df() -> pd.DataFrame:
    df, _ = read_patient_file(FIXTURE_PATH)
    return df


def test_read_patient_file_extracts_patient_id():
    df, patient_id = read_patient_file(FIXTURE_PATH)
    assert patient_id == "sample_patient"
    assert "SepsisLabel" not in df.columns
    assert len(df) > 0


def test_e2e_prediction(sample_patient_df: pd.DataFrame):
    loader = ModelLoader()
    artifact = loader.get_artifact()
    results, validation, loaded = run_prediction(sample_patient_df, "sample_patient", loader)

    assert validation.is_valid
    assert loaded is not None
    assert not results.empty

    expected_cols = {"patient_id", "ICULOS", "risk_score", "prediction", "model_version"}
    assert expected_cols.issubset(results.columns)
    assert len(results) == len(sample_patient_df)
    assert (results["risk_score"] >= 0).all() and (results["risk_score"] <= 1).all()
    assert results["model_version"].iloc[0] == artifact.version


def test_artifact_missing_raises(tmp_path: Path):
    missing_manifest = tmp_path / "missing.json"
    loader = ModelLoader(manifest_path=missing_manifest)
    with pytest.raises(ModelLoadError):
        loader.get_artifact()
