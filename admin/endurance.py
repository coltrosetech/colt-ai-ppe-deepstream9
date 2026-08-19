"""Read-only admin projection of the standalone endurance supervisor state."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATUS = Path("/workspace/validation/results/endurance/current/status.json")
MAX_STATUS_BYTES = 2 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_STATUS_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_endurance_status() -> dict[str, Any]:
    status_path = Path(os.getenv("DEEPSAFE_ENDURANCE_STATUS", str(DEFAULT_STATUS)))
    status = _read_json(status_path)
    if status is None:
        return {
            "available": False,
            "state": "not_started",
            "status_path": str(status_path),
        }
    result = dict(status)
    result["status_path"] = str(status_path)
    live = _read_json(status_path.parent / "live.json")
    if live is not None:
        timestamp = live.get("updated_at_utc")
        if isinstance(timestamp, str):
            try:
                updated = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
                live["heartbeat_age_seconds"] = round(age, 3)
                live["heartbeat_stale"] = (
                    live.get("state") == "running"
                    and age > float(os.getenv("DEEPSAFE_ENDURANCE_STALE_SECONDS", "90"))
                )
            except (ValueError, TypeError):
                live["heartbeat_timestamp_invalid"] = True
        result["live"] = live
    return result
