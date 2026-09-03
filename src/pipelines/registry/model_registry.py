from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = {
    "model.json",
    "model.pkl",
    "preprocessor.json",
    "preprocessor.pkl",
    "feature_config.json",
    "feature_schema.json",
    "threshold.json",
    "metrics.json",
    "cv_results.json",
    "utility_config.json",
    "metadata.json",
}


def _validate_artifacts(path: Path) -> None:
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Candidate model is missing artifacts: {missing}")
    json.loads((path / "feature_schema.json").read_text(encoding="utf-8"))
    json.loads((path / "threshold.json").read_text(encoding="utf-8"))


def register_model(candidate_dir: str | Path, model_root: str | Path, model_version: str) -> Path:
    candidate = Path(candidate_dir)
    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    _validate_artifacts(candidate)
    temporary = root / f"{model_version}.tmp"
    final = root / model_version
    if temporary.exists() or final.exists():
        raise FileExistsError(f"Model version already exists: {model_version}")
    shutil.copytree(candidate, temporary)
    _validate_artifacts(temporary)
    temporary.replace(final)
    return final


def promote_model(
    model_dir: str | Path,
    current_model_file: str | Path,
    dataset_version: str,
    threshold: float,
) -> Path:
    model_path = Path(model_dir)
    _validate_artifacts(model_path)
    pointer = Path(current_model_file)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": model_path.name,
        "dataset_version": dataset_version,
        "model_path": str(model_path / "model.pkl"),
        "preprocessor_path": str(model_path / "preprocessor.pkl"),
        "feature_config_path": str(model_path / "feature_config.json"),
        "threshold": float(threshold),
        "status": "current",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, pointer)
    return pointer
