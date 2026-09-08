"""Feature engineering helpers for PhysioNet/CinC 2019 patient PSV files.

Each raw `.psv` file represents one patient, and each row is one ICU hour.
These functions only create callable helpers; nothing runs on import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_VITAL_COLUMNS = ("HR", "O2Sat", "Temp", "SBP", "Resp")


def _present_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _patient_groups(df: pd.DataFrame, patient_col: str | None):
    if patient_col is not None and patient_col in df.columns:
        return df.groupby(patient_col, group_keys=False, sort=False)
    return None


def _rolling_slope(window: pd.Series) -> float:
    values = pd.to_numeric(window, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    return float(np.polyfit(x, y, 1)[0])


def add_moving_features(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] = DEFAULT_VITAL_COLUMNS,
    window_size: int = 6,
    patient_col: str | None = None,
    min_periods: int = 1,
    include_mean: bool = True,
    include_var: bool = True,
    include_slope: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add moving mean, variance, and slope for selected columns.

    For raw patient `.psv` files, leave ``patient_col=None``. If many patients
    have already been concatenated, pass a patient id column to avoid leakage
    across patient boundaries.
    """

    featured = df if inplace else df.copy()
    columns = _present_columns(featured, columns)

    for column in columns:
        series = pd.to_numeric(featured[column], errors="coerce")

        if patient_col is not None and patient_col in featured.columns:
            grouped = series.groupby(featured[patient_col], sort=False)
            rolling = lambda x: x.rolling(window=window_size, min_periods=min_periods)
            if include_mean:
                featured[f"{column}_moving_mean_w{window_size}"] = grouped.transform(lambda x: rolling(x).mean())
            if include_var:
                featured[f"{column}_moving_var_w{window_size}"] = grouped.transform(lambda x: rolling(x).var())
            if include_slope:
                featured[f"{column}_moving_slope_w{window_size}"] = grouped.transform(
                    lambda x: rolling(x).apply(_rolling_slope, raw=False)
                )
            continue

        rolling = series.rolling(window=window_size, min_periods=min_periods)
        if include_mean:
            featured[f"{column}_moving_mean_w{window_size}"] = rolling.mean()
        if include_var:
            featured[f"{column}_moving_var_w{window_size}"] = rolling.var()
        if include_slope:
            featured[f"{column}_moving_slope_w{window_size}"] = rolling.apply(_rolling_slope, raw=False)

    return featured


def add_delta_features(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] = DEFAULT_VITAL_COLUMNS,
    lag_hours: int = 1,
    patient_col: str | None = None,
    fill_value: float | None = 0.0,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add delta features with default lag of 1 ICU hour."""

    featured = df if inplace else df.copy()
    columns = _present_columns(featured, columns)

    for column in columns:
        series = pd.to_numeric(featured[column], errors="coerce")
        if patient_col is not None and patient_col in featured.columns:
            delta = series.groupby(featured[patient_col], sort=False).diff(lag_hours)
        else:
            delta = series.diff(lag_hours)
        if fill_value is not None:
            delta = delta.fillna(fill_value)
        featured[f"{column}_delta_lag{lag_hours}h"] = delta

    return featured


def add_expanding_features(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] = DEFAULT_VITAL_COLUMNS,
    patient_col: str | None = None,
    min_periods: int = 1,
    include_mean: bool = True,
    include_min: bool = True,
    include_max: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add expanding mean, min, and max for selected columns."""

    featured = df if inplace else df.copy()
    columns = _present_columns(featured, columns)

    for column in columns:
        series = pd.to_numeric(featured[column], errors="coerce")

        if patient_col is not None and patient_col in featured.columns:
            grouped = series.groupby(featured[patient_col], sort=False)
            if include_mean:
                featured[f"{column}_expanding_mean"] = grouped.transform(
                    lambda x: x.expanding(min_periods=min_periods).mean()
                )
            if include_min:
                featured[f"{column}_expanding_min"] = grouped.transform(
                    lambda x: x.expanding(min_periods=min_periods).min()
                )
            if include_max:
                featured[f"{column}_expanding_max"] = grouped.transform(
                    lambda x: x.expanding(min_periods=min_periods).max()
                )
            continue

        expanding = series.expanding(min_periods=min_periods)
        if include_mean:
            featured[f"{column}_expanding_mean"] = expanding.mean()
        if include_min:
            featured[f"{column}_expanding_min"] = expanding.min()
        if include_max:
            featured[f"{column}_expanding_max"] = expanding.max()

    return featured


def add_partial_qsofa(
    df: pd.DataFrame,
    sbp_col: str = "SBP",
    resp_col: str = "Resp",
    output_col: str = "partial_qSOFA",
    inplace: bool = False,
) -> pd.DataFrame:
    """Add partial qSOFA score using available dataset columns.

    GCS is not available in PhysioNet 2019, so this score uses:
    SBP < 100 mmHg and Resp >= 20/min.
    """

    featured = df if inplace else df.copy()
    score = pd.Series(0, index=featured.index, dtype="int64")

    if sbp_col in featured.columns:
        sbp = pd.to_numeric(featured[sbp_col], errors="coerce")
        score += (sbp < 100).fillna(False).astype(int)
    if resp_col in featured.columns:
        resp = pd.to_numeric(featured[resp_col], errors="coerce")
        score += (resp >= 20).fillna(False).astype(int)

    featured[output_col] = score
    return featured


def _score_from_rules(values: pd.Series, rules: list[tuple[pd.Series, int]]) -> np.ndarray:
    conditions = [condition.fillna(False).to_numpy() for condition, _ in rules]
    scores = [score for _, score in rules]
    return np.select(conditions, scores, default=0).astype(int)


def add_news(
    df: pd.DataFrame,
    resp_col: str = "Resp",
    o2sat_col: str = "O2Sat",
    sbp_col: str = "SBP",
    hr_col: str = "HR",
    temp_col: str = "Temp",
    output_col: str = "NEWS",
    add_components: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add partial NEWS score from available PhysioNet 2019 observations.

    This implementation uses respiration rate, oxygen saturation, systolic
    blood pressure, heart rate, and temperature. Consciousness and supplemental
    oxygen are not present in the dataset overview, so they are omitted.
    """

    featured = df if inplace else df.copy()
    components: dict[str, np.ndarray] = {}

    if resp_col in featured.columns:
        resp = pd.to_numeric(featured[resp_col], errors="coerce")
        components["NEWS_resp"] = _score_from_rules(
            resp,
            [
                (resp <= 8, 3),
                (resp.between(9, 11), 1),
                (resp.between(12, 20), 0),
                (resp.between(21, 24), 2),
                (resp >= 25, 3),
            ],
        )
    if o2sat_col in featured.columns:
        o2sat = pd.to_numeric(featured[o2sat_col], errors="coerce")
        components["NEWS_o2sat"] = _score_from_rules(
            o2sat,
            [
                (o2sat <= 91, 3),
                (o2sat.between(92, 93), 2),
                (o2sat.between(94, 95), 1),
                (o2sat >= 96, 0),
            ],
        )
    if sbp_col in featured.columns:
        sbp = pd.to_numeric(featured[sbp_col], errors="coerce")
        components["NEWS_sbp"] = _score_from_rules(
            sbp,
            [
                (sbp <= 90, 3),
                (sbp.between(91, 100), 2),
                (sbp.between(101, 110), 1),
                (sbp.between(111, 219), 0),
                (sbp >= 220, 3),
            ],
        )
    if hr_col in featured.columns:
        hr = pd.to_numeric(featured[hr_col], errors="coerce")
        components["NEWS_hr"] = _score_from_rules(
            hr,
            [
                (hr <= 40, 3),
                (hr.between(41, 50), 1),
                (hr.between(51, 90), 0),
                (hr.between(91, 110), 1),
                (hr.between(111, 130), 2),
                (hr >= 131, 3),
            ],
        )
    if temp_col in featured.columns:
        temp = pd.to_numeric(featured[temp_col], errors="coerce")
        components["NEWS_temp"] = _score_from_rules(
            temp,
            [
                (temp <= 35.0, 3),
                (temp.between(35.1, 36.0), 1),
                (temp.between(36.1, 38.0), 0),
                (temp.between(38.1, 39.0), 1),
                (temp >= 39.1, 2),
            ],
        )

    if add_components:
        for column, values in components.items():
            featured[column] = values

    featured[output_col] = np.sum(list(components.values()), axis=0).astype(int) if components else 0
    return featured


def add_etco2_mask(
    df: pd.DataFrame,
    etco2_col: str = "EtCO2",
    output_col: str = "EtCO2_mask",
    inplace: bool = False,
) -> pd.DataFrame:
    """Add binary EtCO2 mask: 1 indicates present data, 0 indicates missing."""

    featured = df if inplace else df.copy()
    featured[output_col] = featured[etco2_col].notna().astype(int) if etco2_col in featured.columns else 0
    return featured


def add_all_feature_engineering(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...] = DEFAULT_VITAL_COLUMNS,
    window_size: int = 6,
    delta_lag_hours: int = 1,
    patient_col: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add all feature groups requested for a single patient or combined data."""

    featured = df if inplace else df.copy()
    featured = add_moving_features(featured, columns=columns, window_size=window_size, patient_col=patient_col)
    featured = add_delta_features(featured, columns=columns, lag_hours=delta_lag_hours, patient_col=patient_col)
    featured = add_partial_qsofa(featured)
    featured = add_news(featured)
    featured = add_expanding_features(featured, columns=columns, patient_col=patient_col)
    featured = add_etco2_mask(featured)
    return featured
