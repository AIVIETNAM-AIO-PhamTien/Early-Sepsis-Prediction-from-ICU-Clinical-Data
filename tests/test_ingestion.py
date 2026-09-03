from __future__ import annotations

import json
import zipfile

import pytest

from src.pipelines.ingestion.batch_detector import detect_batch, save_batch_metadata
from src.pipelines.ingestion.bronze_pipeline import create_bronze


def test_detect_batch_skips_empty_incoming_directory(tmp_path) -> None:
    result = detect_batch(tmp_path / "incoming", tmp_path / "registry")
    assert result["status"] == "no_data"
    assert result["files"] == []


def test_detect_batch_rejects_registered_checksum(tmp_path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "p000001.psv").write_text("HR|SepsisLabel\n80|0\n", encoding="utf-8")

    first = detect_batch(incoming, tmp_path / "registry")
    save_batch_metadata(first, tmp_path / "registry")
    duplicate = detect_batch(incoming, tmp_path / "registry")

    assert first["status"] == "detected"
    assert duplicate["status"] == "skipped_duplicate"
    assert duplicate["checksum"] == first["checksum"]


def test_bronze_copies_supported_patient_file(tmp_path) -> None:
    source = tmp_path / "p000001.psv"
    source.write_text("HR|SepsisLabel\n80|0\n", encoding="utf-8")
    batch = {"status": "detected", "batch_id": "batch_test", "files": [str(source)]}

    result = create_bronze(batch, tmp_path / "bronze")

    assert result["status"] == "bronze_created"
    assert result["n_input_files"] == 1
    assert (tmp_path / "bronze" / "batch_test" / source.name).read_text() == source.read_text()


def test_bronze_rejects_zip_path_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.psv", "HR|SepsisLabel\n80|0\n")
    batch = {"status": "detected", "batch_id": "batch_unsafe", "files": [str(archive)]}

    with pytest.raises(ValueError, match="Unsafe archive member"):
        create_bronze(batch, tmp_path / "bronze")

