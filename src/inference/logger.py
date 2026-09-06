"""Prediction logging to JSONL files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.inference.loader import ModelArtifact
from src.shared.paths import PREDICTION_LOG_DIR


def log_prediction(
    *,
    input_file: str,
    patient_id: str,
    artifact: ModelArtifact | None,
    results: pd.DataFrame | None,
    status: str,
    error: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """Append a prediction event to the daily JSONL log."""
    log_dir = log_dir or PREDICTION_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"{today}.jsonl"

    n_rows = int(len(results)) if results is not None and not results.empty else 0
    n_positive = (
        int((results["prediction"] == 1).sum())
        if results is not None and not results.empty and "prediction" in results.columns
        else 0
    )

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": artifact.version if artifact else None,
        "pipeline": artifact.pipeline if artifact else None,
        "input_file": input_file,
        "patient_id": patient_id,
        "status": status,
        "n_rows": n_rows,
        "n_positive": n_positive,
        "error": error,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return log_path
