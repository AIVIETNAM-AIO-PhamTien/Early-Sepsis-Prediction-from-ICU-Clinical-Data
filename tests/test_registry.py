from __future__ import annotations

import json

import pytest

from src.pipelines.registry.dataset_registry import (
    latest_dataset_metadata,
    next_dataset_version,
    register_dataset,
)
from src.pipelines.registry.model_registry import REQUIRED_ARTIFACTS, promote_model, register_model


def test_dataset_registry_versions_and_checksums_manifest(tmp_path) -> None:
    registry = tmp_path / "registry"
    manifest = tmp_path / "manifest.parquet"
    manifest.write_bytes(b"manifest")
    metadata = {
        "dataset_version": "dataset_v1",
        "manifest_path": str(manifest),
        "features_path": str(tmp_path / "features.parquet"),
    }

    registered_path = register_dataset(metadata, registry)
    registered = json.loads(registered_path.read_text(encoding="utf-8"))

    assert registered["status"] == "approved"
    assert registered["manifest_checksum"].startswith("sha256:")
    assert next_dataset_version(registry) == "dataset_v2"
    assert latest_dataset_metadata(registry)["dataset_version"] == "dataset_v1"
    with pytest.raises(FileExistsError):
        register_dataset(metadata, registry)


def test_model_registry_registers_and_promotes_complete_candidate(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in REQUIRED_ARTIFACTS:
        path = candidate / name
        if name.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_bytes(b"artifact")

    registered = register_model(candidate, tmp_path / "models", "model_v1")
    pointer = promote_model(registered, tmp_path / "models" / "current_model.json", "dataset_v1", 0.4)
    payload = json.loads(pointer.read_text(encoding="utf-8"))

    assert registered.name == "model_v1"
    assert payload["model_version"] == "model_v1"
    assert payload["dataset_version"] == "dataset_v1"
    assert payload["threshold"] == 0.4

