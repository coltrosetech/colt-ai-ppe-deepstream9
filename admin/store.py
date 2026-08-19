import json
import os
import threading
from copy import deepcopy
from pathlib import Path


DEFAULT = {
    "sources": [],
    "inference": {
        "person_profile": "yolo11s-640",
        "streammux_width": 1920,
        "streammux_height": 1080,
    },
    "analytics": {
        "person": {"enabled": True, "confidence": 0.35, "interval": 0},
        "pose": {"enabled": False, "confidence": 0.30, "interval": 0},
        "ppe": {"enabled": False, "confidence": 0.40, "interval": 1},
    },
}


def _normalise(value: dict) -> dict:
    """Add new settings without discarding an existing installation's state."""
    result = deepcopy(value)
    result.setdefault("sources", [])

    inference = result.setdefault("inference", {})
    if "person_profile" not in inference and "profile" not in inference:
        legacy_width = int(inference.get("width", 640))
        inference["person_profile"] = "yolo11s-960" if legacy_width == 960 else "yolo11s-640"
    inference.setdefault("streammux_width", DEFAULT["inference"]["streammux_width"])
    inference.setdefault("streammux_height", DEFAULT["inference"]["streammux_height"])

    analytics = result.setdefault("analytics", {})
    for name, defaults in DEFAULT["analytics"].items():
        settings = deepcopy(defaults)
        settings.update(analytics.get(name, {}))
        analytics[name] = settings
    return result


class JsonStore:
    def __init__(self, root: str | None = None):
        self.root = Path(root or os.getenv("DEEPSAFE_DATA", "/tmp/deepsafe"))
        self.path = self.root / "state.json"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(deepcopy(DEFAULT))

    def load(self):
        with self.lock:
            value = json.loads(self.path.read_text())
            normalised = _normalise(value)
            if value != normalised:
                self.save(normalised)
            return normalised

    def save(self, value):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, indent=2))
            tmp.replace(self.path)
