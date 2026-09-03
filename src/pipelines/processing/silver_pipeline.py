from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipelines.processing.preprocessor import filter_absolute_outliers


def classify_source(patient_id: str) -> str:
    """Return the single production cohort used by this project."""
    return "all"


def _read_patient(path: Path, schema: dict[str, Any]) -> pd.DataFrame:
    separator = schema.get("delimiter", "|") if path.suffix.lower() == ".psv" else ","
    frame = pd.read_csv(path, sep=separator)
    frame = frame.rename(columns=schema.get("aliases", {}))
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Alias canonicalization created duplicate columns in {path}: {duplicates}")
    if path.suffix.lower() == ".psv":
        frame["patient_id"] = path.stem
    elif "patient_id" not in frame:
        raise ValueError(f"Canonical CSV must contain patient_id: {path}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if "source" not in frame:
        frame["source"] = frame["patient_id"].map(classify_source)
    return frame


def load_raw_patients(
    raw_root: str | Path,
    schema: dict[str, Any],
) -> pd.DataFrame:
    root = Path(raw_root)
    paths = list(root.rglob("*.psv"))
    if not paths:
        raise FileNotFoundError(f"No raw patient files found under {root}")
    frames = [filter_absolute_outliers(_read_patient(path, schema)) for path in sorted(paths)]
    return pd.concat(frames, ignore_index=True).sort_values(["patient_id", "ICULOS"], kind="stable")


def build_silver(
    bronze_path: str | Path,
    silver_root: str | Path,
    batch_id: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    source = Path(bronze_path)
    files = sorted([*source.rglob("*.psv"), *source.rglob("*.csv")])
    if not files:
        raise ValueError(f"No patient files found in {source}")
    pattern = re.compile(schema["patient_id_pattern"])
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    outlier_counts = {"HR": 0, "O2Sat": 0, "Temp": 0}
    for path in files:
        if path.suffix.lower() == ".psv" and not pattern.fullmatch(path.stem):
            raise ValueError(f"Invalid patient filename: {path.name}")
        frame = _read_patient(path, schema)
        patient_ids = set(frame["patient_id"].astype(str))
        invalid_ids = sorted(patient_id for patient_id in patient_ids if not pattern.fullmatch(patient_id))
        if invalid_ids:
            raise ValueError(f"Invalid patient IDs in {path.name}: {invalid_ids[:5]}")
        duplicate_ids = patient_ids & seen
        if duplicate_ids:
            raise ValueError(f"Duplicate patient IDs across files: {sorted(duplicate_ids)[:5]}")
        seen.update(patient_ids)
        for patient_id, patient in frame.groupby("patient_id", sort=False):
            if not patient["ICULOS"].is_monotonic_increasing:
                raise ValueError(f"ICULOS is not monotonic for patient {patient_id}")
            if patient["ICULOS"].duplicated().any():
                raise ValueError(f"Duplicate ICULOS rows for patient {patient_id}")
        before = frame[[c for c in outlier_counts if c in frame]].notna()
        frame = filter_absolute_outliers(frame)
        for column in before:
            outlier_counts[column] += int((before[column] & frame[column].isna()).sum())
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["patient_id", "ICULOS"], kind="stable").reset_index(drop=True)
    destination = Path(silver_root) / batch_id
    destination.mkdir(parents=True, exist_ok=False)
    data_path = destination / "patients.parquet"
    combined.to_parquet(data_path, index=False)
    report = {
        "batch_id": batch_id,
        "n_patients": len(seen),
        "n_rows": len(combined),
        "outlier_counts": outlier_counts,
        "silver_path": str(data_path),
    }
    (destination / "silver_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
