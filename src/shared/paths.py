"""Project path configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = Path(os.environ.get("SEPSIS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
CURRENT_MODEL_MANIFEST = ARTIFACTS_DIR / "current_model.json"
PREDICTION_LOG_DIR = Path(
    os.environ.get("SEPSIS_PREDICTION_LOG_DIR", PROJECT_ROOT / "logs" / "predictions")
)
SAMPLE_DATA_DIR = Path(
    os.environ.get("SEPSIS_SAMPLE_DATA_DIR", PROJECT_ROOT / "app" / "sample_data")
)
