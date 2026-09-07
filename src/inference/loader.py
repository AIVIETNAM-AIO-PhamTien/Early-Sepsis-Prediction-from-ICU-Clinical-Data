"""Model artifact loader with manifest-based caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import xgboost as xgb

from src.shared.paths import ARTIFACTS_DIR, CURRENT_MODEL_MANIFEST, PROJECT_ROOT


class ModelLoadError(Exception):
    """Raised when model artifacts cannot be loaded."""


@dataclass
class ModelArtifact:
    version: str
    pipeline: str
    threshold: float
    model_format: str
    model: Any
    preprocessor: Any
    feature_config: dict
    loaded_at: datetime
    updated_at: str
    manifest_path: Path


class ModelLoader:
    def __init__(self, manifest_path: Path | None = None):
        self.manifest_path = Path(manifest_path or CURRENT_MODEL_MANIFEST)
        self._cache: ModelArtifact | None = None
        self._manifest_mtime: float | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _resolve_path(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            raise ModelLoadError(f"JSON file not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ModelLoadError(f"Invalid JSON in {path}: {exc}") from exc

    def _load_model(self, path: Path, model_format: str) -> xgb.Booster:
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")
        if model_format not in {"xgboost_json", "xgboost_ubj"}:
            raise ModelLoadError(f"Unsupported model format: {model_format}")
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            return booster
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    def _parse_manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise ModelLoadError(f"Manifest not found: {self.manifest_path}")
        manifest = self._load_json(self.manifest_path)
        required = [
            "model_version",
            "model_path",
            "preprocessor_path",
            "feature_config_path",
            "pipeline",
            "model_format",
            "threshold",
            "updated_at",
        ]
        missing = [k for k in required if k not in manifest]
        if missing:
            raise ModelLoadError(f"Manifest missing required fields: {missing}")
        return manifest

    def get_artifact(self, force_reload: bool = False) -> ModelArtifact:
        """Load or return cached artifact. Reload if manifest changes."""
        if not self.manifest_path.exists():
            self._last_error = f"Manifest not found: {self.manifest_path}"
            raise ModelLoadError(self._last_error)

        mtime = self.manifest_path.stat().st_mtime
        if (
            not force_reload
            and self._cache is not None
            and self._manifest_mtime == mtime
        ):
            return self._cache

        try:
            manifest = self._parse_manifest()
            model_path = self._resolve_path(manifest["model_path"])
            preprocessor_path = self._resolve_path(manifest["preprocessor_path"])
            feature_config_path = self._resolve_path(manifest["feature_config_path"])
            model_format = manifest["model_format"]

            artifact = ModelArtifact(
                version=manifest["model_version"],
                pipeline=manifest["pipeline"],
                threshold=float(manifest["threshold"]),
                model_format=model_format,
                model=self._load_model(model_path, model_format),
                preprocessor=self._load_json(preprocessor_path),
                feature_config=self._load_json(feature_config_path),
                loaded_at=datetime.now(),
                updated_at=manifest["updated_at"],
                manifest_path=self.manifest_path,
            )
            self._cache = artifact
            self._manifest_mtime = mtime
            self._last_error = None
            return artifact
        except ModelLoadError as exc:
            self._last_error = str(exc)
            raise

    def get_manifest_info(self) -> dict | None:
        """Return manifest fields for UI without loading the full model."""
        try:
            return self._parse_manifest()
        except ModelLoadError:
            return None
