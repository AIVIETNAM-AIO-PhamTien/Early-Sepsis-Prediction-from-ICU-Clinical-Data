from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from src.pipelines.config import PROJECT_ROOT


@lru_cache(maxsize=1)
def _feature_module() -> ModuleType:
    module_path = PROJECT_ROOT / "data" / "feature-engineering.py"
    spec = importlib.util.spec_from_file_location("project_feature_engineering", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load feature helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engineer_patient_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    feature_config = config["features"]
    return _feature_module().add_all_feature_engineering(
        df,
        columns=feature_config["columns"],
        window_size=int(feature_config["moving_window_hours"]),
        delta_lag_hours=int(feature_config["delta_lag_hours"]),
        patient_col="patient_id",
        inplace=False,
    )


def model_feature_columns(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    excluded = {
        schema["patient_column"],
        schema["source_column"],
        schema["label_column"],
        "batch_id",
        "dataset_split",
    }
    return [column for column in df.columns if column not in excluded]


def add_lookback(
    feature_frame: pd.DataFrame,
    columns: list[str],
    patient_col: str = "patient_id",
    hours: int = 5,
    padding: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    output_parts: list[np.ndarray] = []
    ordered_names = [f"{column}__t-{lag}" for lag in range(hours) for column in columns]
    for _, patient in feature_frame.groupby(patient_col, sort=False):
        values = patient[columns].to_numpy(dtype=np.float32)
        matrix = np.full((len(values), len(columns) * hours), padding, dtype=np.float32)
        for lag in range(hours):
            if lag == 0:
                matrix[:, : len(columns)] = values
            elif lag < len(values):
                start = lag * len(columns)
                matrix[lag:, start : start + len(columns)] = values[:-lag]
        output_parts.append(matrix)
    if not output_parts:
        return np.empty((0, len(ordered_names)), dtype=np.float32), ordered_names
    return np.vstack(output_parts), ordered_names

