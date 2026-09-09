from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.pipelines.processing import cleaning
from src.pipelines.processing.feature_builder import engineer_patient_features, model_feature_columns


@dataclass
class FittedPreprocessor:
    age_gender_values: dict[tuple[float, float], dict[str, float]]
    log_columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    fill_values: dict[str, float]
    feature_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["age_gender_values"] = {
            f"{age}|{gender}": values
            for (age, gender), values in self.age_gender_values.items()
        }
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedPreprocessor":
        age_gender_values: dict[tuple[float, float], dict[str, float]] = {}
        for key, values in payload.get("age_gender_values", {}).items():
            age, gender = key.split("|", maxsplit=1)
            age_gender_values[(float(age), float(gender))] = {
                column: float(value) for column, value in values.items()
            }
        return cls(
            age_gender_values=age_gender_values,
            log_columns=list(payload["log_columns"]),
            means={key: float(value) for key, value in payload["means"].items()},
            stds={key: float(value) for key, value in payload["stds"].items()},
            fill_values={key: float(value) for key, value in payload["fill_values"].items()},
            feature_columns=list(payload["feature_columns"]),
        )


def filter_absolute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return cleaning.filter_absolute_outliers(df, inplace=False)


def _clean_with_fitted_values(
    df: pd.DataFrame,
    fitted_values: dict[tuple[float, float], dict[str, float]],
    impute_columns: list[str],
) -> pd.DataFrame:
    return cleaning.forward_fill_impute(
        df,
        columns=impute_columns,
        patient_col="patient_id",
        fitted_age_gender_values=fitted_values,
        inplace=False,
    )


def _signed_log(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    transformed = frame.copy()
    for column in columns:
        if column in transformed:
            values = pd.to_numeric(transformed[column], errors="coerce")
            transformed[column] = np.sign(values) * np.log1p(np.abs(values))
    return transformed


def _select_log_columns(frame: pd.DataFrame, columns: list[str], threshold: float) -> list[str]:
    selected: list[str] = []
    for column in columns:
        skew = pd.to_numeric(frame[column], errors="coerce").skew()
        if pd.notna(skew) and abs(float(skew)) > threshold:
            selected.append(column)
    return selected


def fit_preprocessor(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    frozen_log_columns: list[str] | None = None,
) -> FittedPreprocessor:
    impute_columns = list(config["preprocessing"]["impute_columns"])
    fitted_age_gender = cleaning.fit_age_gender_normal_values(
        train_df,
        columns=impute_columns,
        patient_col="patient_id",
    )
    cleaned = _clean_with_fitted_values(train_df, fitted_age_gender, impute_columns)
    featured = engineer_patient_features(cleaned, config)
    columns = model_feature_columns(featured, schema)
    log_columns = frozen_log_columns or _select_log_columns(
        featured,
        columns,
        float(config["preprocessing"]["skew_threshold"]),
    )
    transformed = _signed_log(featured, log_columns)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    fills: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(transformed[column], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std(ddof=0)) if values.notna().any() else 1.0
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        means[column] = mean
        stds[column] = std
        fills[column] = mean if np.isfinite(mean) else 0.0
    return FittedPreprocessor(fitted_age_gender, log_columns, means, stds, fills, columns)


def transform_patients(
    df: pd.DataFrame,
    fitted: FittedPreprocessor,
    config: dict[str, Any],
) -> pd.DataFrame:
    cleaned = _clean_with_fitted_values(
        df,
        fitted.age_gender_values,
        list(config["preprocessing"]["impute_columns"]),
    )
    featured = engineer_patient_features(cleaned, config)
    transformed = _signed_log(featured, fitted.log_columns)
    for column in fitted.feature_columns:
        if column not in transformed:
            raise ValueError(f"Feature schema mismatch; missing column: {column}")
        values = pd.to_numeric(transformed[column], errors="coerce").fillna(fitted.fill_values[column])
        transformed[column] = (values - fitted.means[column]) / fitted.stds[column]
    return transformed
