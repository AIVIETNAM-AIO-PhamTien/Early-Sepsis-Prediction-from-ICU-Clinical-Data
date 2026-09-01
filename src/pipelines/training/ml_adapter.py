from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pipelines.processing.feature_builder import add_lookback
from src.pipelines.processing.preprocessor import FittedPreprocessor, fit_preprocessor, transform_patients


def prepare_fold(
    train_patients: pd.DataFrame,
    valid_patients: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    frozen_log_columns: list[str] | None = None,
) -> dict[str, Any]:
    fitted = fit_preprocessor(train_patients, config, schema, frozen_log_columns=frozen_log_columns)
    train = transform_patients(train_patients, fitted, config)
    valid = transform_patients(valid_patients, fitted, config)
    feature_config = config["features"]
    hours = int(feature_config["lookback_hours"])
    padding = float(feature_config["lookback_padding"])
    x_train, lookback_columns = add_lookback(train, fitted.feature_columns, hours=hours, padding=padding)
    x_valid, valid_columns = add_lookback(valid, fitted.feature_columns, hours=hours, padding=padding)
    if lookback_columns != valid_columns:
        raise ValueError("Train and validation feature schemas differ")
    label = schema["label_column"]
    return {
        "X_train": x_train,
        "y_train": train[label].to_numpy(dtype=np.float32),
        "X_valid": x_valid,
        "y_valid": valid[label].to_numpy(dtype=np.float32),
        "valid_groups": valid["patient_id"].astype(str).to_numpy(),
        "valid_sources": valid["source"].astype(str).to_numpy(),
        "preprocessor": fitted,
        "feature_columns": lookback_columns,
    }


def group_patient_outputs(
    groups: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    label_groups: list[np.ndarray] = []
    probability_groups: list[np.ndarray] = []
    for patient_id in dict.fromkeys(groups.tolist()):
        mask = groups == patient_id
        label_groups.append(labels[mask])
        probability_groups.append(probabilities[mask])
    return label_groups, probability_groups

