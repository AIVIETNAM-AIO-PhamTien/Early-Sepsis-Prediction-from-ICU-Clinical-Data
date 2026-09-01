from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Airflow configures logging during import. Keeping AIRFLOW_HOME inside the
# ignored project runtime directory makes DAG import tests deterministic.
os.environ.setdefault("AIRFLOW_HOME", str(PROJECT_ROOT / ".airflow"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")


@pytest.fixture
def schema() -> dict:
    return json.loads((PROJECT_ROOT / "configs" / "data_schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def valid_silver_frame(schema: dict) -> pd.DataFrame:
    rows = []
    for patient_id, labels in (("p000001", [0, 0]), ("p000002", [0, 1])):
        for offset, label in enumerate(labels, start=1):
            row = {column: 1.0 for column in schema["clinical_columns"]}
            row.update(
                {
                    "patient_id": patient_id,
                    "source": "all",
                    "ICULOS": offset,
                    schema["label_column"]: label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)

