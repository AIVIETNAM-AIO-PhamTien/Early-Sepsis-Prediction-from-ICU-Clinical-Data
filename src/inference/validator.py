"""Input validation for patient data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.shared.schema import IMPORTANT_VITAL_COLUMNS, REQUIRED_COLUMNS


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _missing_pct(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)


def validate_patient_df(df: pd.DataFrame) -> ValidationResult:
    """Validate patient DataFrame before prediction."""
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_pct": _missing_pct(df),
    }

    if df.empty:
        errors.append("File is empty (0 rows).")
        return ValidationResult(False, errors, warnings, metadata)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing required columns ({len(missing_cols)}): "
            + ", ".join(missing_cols[:5])
            + ("..." if len(missing_cols) > 5 else "")
        )

    if "ICULOS" not in df.columns:
        errors.append("ICULOS column is missing.")
    else:
        iculos = pd.to_numeric(df["ICULOS"], errors="coerce")
        if iculos.isna().any():
            errors.append("ICULOS contains non-numeric or missing values.")
        else:
            iculos_vals = iculos.to_numpy()
            if np.any(np.diff(iculos_vals) < 0):
                errors.append("ICULOS is not monotonic non-decreasing.")
            if len(iculos_vals) != len(np.unique(iculos_vals)):
                errors.append("ICULOS contains duplicate values.")

    present_important = [c for c in IMPORTANT_VITAL_COLUMNS if c in df.columns]
    if present_important:
        high_missing = []
        for col in present_important:
            pct = float(df[col].isna().mean() * 100)
            if pct > 50:
                high_missing.append(f"{col} ({pct:.0f}%)")
        if high_missing:
            warnings.append(
                "High missing rate on important columns: " + ", ".join(high_missing)
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )
