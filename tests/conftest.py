"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.io import read_patient_file

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_patient.psv"


@pytest.fixture(scope="session")
def sample_patient_df() -> pd.DataFrame:
    df, _ = read_patient_file(FIXTURE_PATH)
    return df
