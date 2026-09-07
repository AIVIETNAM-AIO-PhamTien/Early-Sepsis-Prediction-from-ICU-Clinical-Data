"""Validator unit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.inference.validator import validate_patient_df
from src.shared.schema import REQUIRED_COLUMNS


def _valid_row(iculos: int = 1) -> dict:
    row = {col: 1.0 for col in REQUIRED_COLUMNS}
    row["ICULOS"] = iculos
    return row


def test_valid_patient_passes(sample_patient_df: pd.DataFrame):
    result = validate_patient_df(sample_patient_df)
    assert result.is_valid
    assert not result.errors
    assert result.metadata["n_rows"] > 0


def test_missing_column_fails(sample_patient_df: pd.DataFrame):
    df = sample_patient_df.drop(columns=["HR"])
    result = validate_patient_df(df)
    assert not result.is_valid
    assert any("Missing required columns" in e for e in result.errors)


def test_empty_file_fails():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    result = validate_patient_df(df)
    assert not result.is_valid
    assert any("empty" in e.lower() for e in result.errors)


def test_non_monotonic_iculos_fails(sample_patient_df: pd.DataFrame):
    df = sample_patient_df.copy()
    df.loc[1, "ICULOS"] = 0
    result = validate_patient_df(df)
    assert not result.is_valid
    assert any("monotonic" in e.lower() for e in result.errors)


def test_duplicate_iculos_fails(sample_patient_df: pd.DataFrame):
    df = sample_patient_df.copy()
    df.loc[1, "ICULOS"] = df.loc[0, "ICULOS"]
    result = validate_patient_df(df)
    assert not result.is_valid
    assert any("duplicate" in e.lower() for e in result.errors)
