from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def create_bronze(batch: dict[str, Any], bronze_root: str | Path) -> dict[str, Any]:
    if batch.get("status") != "detected":
        return batch
    destination = Path(bronze_root) / batch["batch_id"]
    if destination.exists():
        raise FileExistsError(f"Bronze batch already exists: {destination}")
    destination.mkdir(parents=True)

    for source_value in batch["files"]:
        source = Path(source_value)
        if source.suffix.lower() == ".zip":
            _safe_extract(source, destination)
        else:
            shutil.copy2(source, destination / source.name)

    psv_files = sorted(destination.rglob("*.psv"))
    csv_files = sorted(destination.rglob("*.csv"))
    if not psv_files and not csv_files:
        raise ValueError(f"Batch contains no usable PSV/CSV files: {destination}")
    return {
        **batch,
        "status": "bronze_created",
        "bronze_path": str(destination),
        "n_input_files": len(psv_files) + len(csv_files),
    }

