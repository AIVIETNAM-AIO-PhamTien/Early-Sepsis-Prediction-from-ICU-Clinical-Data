from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "retraining.yaml"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "data_schema.json"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing retraining config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    schema_path = resolve_project_path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing data schema: {schema_path}")
    with schema_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_runtime_directories(config: dict[str, Any]) -> None:
    keys = (
        "raw_dir",
        "incoming_dir",
        "bronze_dir",
        "silver_dir",
        "gold_dir",
        "quarantine_dir",
        "registry_dir",
    )
    for key in keys:
        resolve_project_path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    resolve_project_path(config["model_registry"]["root_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_project_path(config["model_registry"]["runs_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_project_path(config["logging"]["history_file"]).parent.mkdir(parents=True, exist_ok=True)

