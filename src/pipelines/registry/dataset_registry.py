from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def next_dataset_version(registry_dir: str | Path) -> str:
    root = Path(registry_dir)
    versions: list[int] = []
    for path in root.glob("dataset_v*.json"):
        try:
            versions.append(int(path.stem.removeprefix("dataset_v")))
        except ValueError:
            continue
    return f"dataset_v{max(versions, default=0) + 1}"


def latest_dataset_metadata(registry_dir: str | Path) -> dict[str, Any] | None:
    root = Path(registry_dir)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("dataset_v*.json"):
        try:
            candidates.append((int(path.stem.removeprefix("dataset_v")), path))
        except ValueError:
            continue
    if not candidates:
        return None
    return json.loads(max(candidates, key=lambda item: item[0])[1].read_text(encoding="utf-8"))


def register_dataset(metadata: dict[str, Any], registry_dir: str | Path) -> Path:
    root = Path(registry_dir)
    root.mkdir(parents=True, exist_ok=True)
    version = metadata["dataset_version"]
    path = root / f"{version}.json"
    if path.exists():
        raise FileExistsError(f"Dataset version already registered: {version}")
    manifest_path = Path(metadata["manifest_path"])
    payload = {
        **metadata,
        "manifest_checksum": "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "status": "approved",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
