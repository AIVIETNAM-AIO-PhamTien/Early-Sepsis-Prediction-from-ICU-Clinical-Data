from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_silver(
    silver_path: str | Path,
    schema: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_parquet(silver_path)
    required = [*schema["clinical_columns"], schema["label_column"], "patient_id", "source"]
    errors: list[str] = []
    missing = [column for column in required if column not in frame]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if errors:
        return {"is_valid": False, "errors": errors, "warnings": []}

    labels = set(pd.to_numeric(frame[schema["label_column"]], errors="coerce").dropna().unique())
    if not labels.issubset({0, 1}):
        errors.append(f"SepsisLabel contains invalid values: {sorted(labels)}")
    if frame.duplicated(["patient_id", "ICULOS"]).any():
        errors.append("Duplicate patient_id + ICULOS rows detected")
    non_monotonic = [
        patient_id
        for patient_id, patient in frame.groupby("patient_id", sort=False)
        if not patient["ICULOS"].is_monotonic_increasing
    ]
    if non_monotonic:
        errors.append(f"ICULOS is not monotonic for {len(non_monotonic)} patients")
    missing_rates = frame[schema["clinical_columns"]].isna().mean().to_dict()
    warnings = [f"High missingness for {column}: {rate:.3f}" for column, rate in missing_rates.items() if rate > 0.95]
    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_patients": int(frame["patient_id"].nunique()),
        "n_rows": int(len(frame)),
        "sepsis_patient_rate": float(frame.groupby("patient_id")[schema["label_column"]].max().mean()),
        "missing_rates": missing_rates,
        "source_counts": frame.groupby("source")["patient_id"].nunique().to_dict(),
    }


def require_quality(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("is_valid"):
        raise ValueError("Data quality gate failed: " + "; ".join(report.get("errors", [])))
    return report
