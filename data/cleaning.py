"""Cleaning helpers for PhysioNet/CinC sepsis time-series data.

This module only defines reusable functions. It does not run any cleaning step
on import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PHYSIOLOGICAL_RANGES = {
    "TEM": (20.0, 50.0),
    "Temp": (20.0, 50.0),
    "SpO2": (21.0, 100.0),
    "O2Sat": (21.0, 100.0),
    "HR": (0.0, 300.0),
}


def filter_absolute_outliers(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Set values outside valid physiological ranges to NaN.

    Parameters
    ----------
    df:
        Input dataframe.
    ranges:
        Optional mapping of column name to inclusive (min, max) range. Defaults
        to Temp/TEM, O2Sat/SpO2, and HR ranges.
    inplace:
        If True, mutate ``df`` directly. Otherwise return a cleaned copy.
    """

    cleaned = df if inplace else df.copy()
    ranges = PHYSIOLOGICAL_RANGES if ranges is None else ranges

    for column, (lower, upper) in ranges.items():
        if column not in cleaned.columns:
            continue
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned.loc[~values.between(lower, upper), column] = np.nan

    return cleaned


def fit_age_gender_normal_values(
    reference_df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
    age_col: str = "Age",
    gender_col: str = "Gender",
    patient_col: str | None = None,
) -> dict[tuple[float, float], dict[str, float]]:
    """Fit median fallback values from a multi-patient reference dataframe.

    Raw PhysioNet files contain one patient per `.psv`, so build this reference
    from training patients only, then pass the result to ``forward_fill_impute``.
    """

    columns = list(PHYSIOLOGICAL_RANGES) if columns is None else list(columns)
    available_cols = [column for column in columns if column in reference_df.columns]
    if age_col not in reference_df or gender_col not in reference_df or not available_cols:
        return {}

    cohort = reference_df[[age_col, gender_col, *available_cols]].copy()
    cohort[age_col] = pd.to_numeric(cohort[age_col], errors="coerce")
    cohort[gender_col] = pd.to_numeric(cohort[gender_col], errors="coerce")
    for column in available_cols:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")

    if patient_col is not None and patient_col in reference_df.columns:
        cohort[patient_col] = reference_df[patient_col]
        cohort = cohort.groupby(patient_col, sort=False).agg(
            {age_col: "first", gender_col: "first", **{column: "median" for column in available_cols}}
        )

    medians = cohort.groupby([age_col, gender_col], dropna=False)[available_cols].median()
    return {
        (float(age), float(gender)): row.dropna().astype(float).to_dict()
        for (age, gender), row in medians.iterrows()
    }


def age_gender_normal_values(
    age: float | int | None,
    gender: float | int | None,
    fitted_values: dict[tuple[float, float], dict[str, float]] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Return fitted median normal values for the same age and gender.

    Use ``fit_age_gender_normal_values`` on a multi-patient training reference
    first. If no fitted match exists, conservative clinical defaults are used.
    """

    defaults = {"HR": 75.0, "Temp": 37.0, "TEM": 37.0, "O2Sat": 98.0, "SpO2": 98.0}
    columns = list(PHYSIOLOGICAL_RANGES) if columns is None else list(columns)
    values = {column: defaults.get(column, 0.0) for column in columns}

    if fitted_values is None:
        return values

    age_value = np.nan if age is None else float(age)
    gender_value = np.nan if gender is None else float(gender)
    matched = fitted_values.get((age_value, gender_value), {})
    values.update({column: float(value) for column, value in matched.items() if column in values})

    return values


def forward_fill_impute(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
    age_col: str = "Age",
    gender_col: str = "Gender",
    patient_col: str | None = None,
    normal_values: dict[str, float] | None = None,
    fitted_age_gender_values: dict[tuple[float, float], dict[str, float]] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Impute missing values by forward fill, then normal-value fallback.

    Forward filling is applied within each patient when ``patient_col`` is
    provided. Missing values at the beginning of each patient time series are
    filled using age- and gender-based normal values.
    """

    imputed = df if inplace else df.copy()
    columns = list(PHYSIOLOGICAL_RANGES) if columns is None else list(columns)
    columns = [column for column in columns if column in imputed.columns]

    if not columns:
        return imputed

    def _fill_values_for_group(group: pd.DataFrame) -> dict[str, float]:
        age = group[age_col].dropna().iloc[0] if age_col in group and group[age_col].notna().any() else None
        gender = (
            group[gender_col].dropna().iloc[0]
            if gender_col in group and group[gender_col].notna().any()
            else None
        )
        fill_values = age_gender_normal_values(
            age,
            gender,
            fitted_values=fitted_age_gender_values,
            columns=columns,
        )
        if normal_values is not None:
            fill_values.update(normal_values)
        return fill_values

    def _impute_group_at_index(group_index: pd.Index) -> None:
        group = imputed.loc[group_index]
        fill_values = _fill_values_for_group(group)
        for column in columns:
            imputed.loc[group_index, column] = group[column].ffill().fillna(fill_values.get(column, 0.0))

    if patient_col is not None and patient_col in imputed.columns:
        for _, group_index in imputed.groupby(patient_col, sort=False).groups.items():
            _impute_group_at_index(group_index)
        return imputed

    _impute_group_at_index(imputed.index)
    return imputed
