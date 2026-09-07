from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".psv", ".zip", ".csv"}


def _files_for_batch(incoming_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in incoming_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def calculate_batch_checksum(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _registered_checksums(registry_dir: Path) -> set[str]:
    checksums: set[str] = set()
    for metadata_path in registry_dir.glob("batch_*.json"):
        try:
            checksums.add(json.loads(metadata_path.read_text(encoding="utf-8"))["checksum"])
        except (KeyError, json.JSONDecodeError, OSError):
            continue
    return checksums


def detect_batch(incoming_dir: str | Path, registry_dir: str | Path) -> dict[str, Any]:
    incoming = Path(incoming_dir)
    registry = Path(registry_dir)
    incoming.mkdir(parents=True, exist_ok=True)
    registry.mkdir(parents=True, exist_ok=True)
    files = _files_for_batch(incoming)
    if not files:
        return {"status": "no_data", "source_path": str(incoming), "files": []}

    checksum = calculate_batch_checksum(files, incoming)
    if checksum in _registered_checksums(registry):
        return {
            "status": "skipped_duplicate",
            "source_path": str(incoming),
            "checksum": checksum,
            "files": [str(path) for path in files],
        }

    now = datetime.now(timezone.utc)
    batch_id = f"batch_{now:%Y%m%d_%H%M%S}_{checksum[:8]}"
    return {
        "batch_id": batch_id,
        "source_path": str(incoming),
        "checksum": checksum,
        "received_at": now.isoformat(),
        "status": "detected",
        "files": [str(path) for path in files],
    }


def save_batch_metadata(metadata: dict[str, Any], registry_dir: str | Path) -> Path:
    batch_id = metadata.get("batch_id")
    if not batch_id:
        raise ValueError("Cannot register batch metadata without batch_id")
    path = Path(registry_dir) / f"{batch_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path

