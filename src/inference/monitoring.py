"""Aggregation utilities over the JSONL prediction log — post-deployment surveillance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.shared.paths import PREDICTION_LOG_DIR


def load_prediction_logs(log_dir: Path | None = None) -> pd.DataFrame:
    """Load all daily JSONL prediction logs into a single DataFrame."""
    log_dir = log_dir or PREDICTION_LOG_DIR
    if not log_dir.exists():
        return pd.DataFrame()

    records: list[dict] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


def daily_summary(logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predictions per day: run volume, elevated cases, failures."""
    if logs.empty:
        return logs
    grouped = logs.groupby("date").agg(
        total_runs=("patient_id", "count"),
        elevated_cases=("n_positive", lambda s: int((s > 0).sum())),
        failures=("status", lambda s: int((s != "success").sum())),
    )
    return grouped.reset_index()


def model_version_breakdown(logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predictions per model version, with elevated-case rate."""
    if logs.empty:
        return logs
    grouped = logs.groupby("model_version").agg(
        total_runs=("patient_id", "count"),
        elevated_cases=("n_positive", lambda s: int((s > 0).sum())),
    )
    grouped["elevated_rate"] = grouped["elevated_cases"] / grouped["total_runs"]
    return grouped.reset_index()
