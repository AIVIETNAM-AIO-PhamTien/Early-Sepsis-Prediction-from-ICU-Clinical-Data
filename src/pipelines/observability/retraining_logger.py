from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_retraining_log(record: dict[str, Any], history_file: str | Path) -> Path:
    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path

