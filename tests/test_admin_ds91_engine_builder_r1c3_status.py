from __future__ import annotations

import json
from pathlib import Path

from admin import ds91_engine_builder_r1c3_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_engine_builder_card_replays_terminal_scope_without_overclaim() -> None:
    card = status.load_ds91_engine_builder_r1c3_status()
    assert card["state"] == "terminal_accepted_prepared_closed_identity_only"
    assert card["available"] is True
    assert card["terminal_accepted"] is True
    assert card["prepared_closed_image_identity_accepted"] is True
    assert card["image"]["id"] == status.IMAGE_ID
    assert card["image"]["config"] == status.CONFIG_ID
    assert card["tests"] == {"unit_passed": 30, "mutation_passed": 20, "gate_replays_passed": 6, "failed": 0, "p0": 0, "p1": 0, "p2": 0}
    for key in ("gpu_runtime_qualified", "tensorrt_engine_qualified", "deepstream_runtime_qualified", "inference_qualified", "production_ready"):
        assert card[key] is False
    assert card["execution_actions_available"] is False


def test_engine_builder_card_fails_closed_on_root_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    relative = "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json"
    pins[relative] = (pins[relative][0], "0" * 64)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_ds91_engine_builder_r1c3_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["terminal_accepted"] is False
    assert card["production_ready"] is False


def test_engine_builder_card_rejects_semantically_resigned_gpu_overclaim(monkeypatch) -> None:
    original = status._read_exact
    target = "validation/accepted-roots/ds91-engine-builder-r1c3-terminal/terminal-root-r1c3.json"

    def read(relative: str) -> bytes:
        raw = original(relative)
        if relative != target:
            return raw
        value = json.loads(raw)
        value["accepted_scope"]["gpu_runtime"] = True
        unsigned = {key: item for key, item in value.items() if key != "self_fingerprint"}
        import hashlib

        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        value["self_fingerprint"]["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_ds91_engine_builder_r1c3_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["gpu_runtime_qualified"] is False
    assert card["production_ready"] is False


def test_validation_payload_and_ui_show_narrow_engine_builder_acceptance() -> None:
    card = load_validation_status()["campaigns"]["deepstream91_engine_builder_r1c3"]
    assert card["terminal_accepted"] is True
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "deepstream91_engine_builder_r1c3" in page
    assert "campaign.tensorrt_engine_qualified" in page
    assert "fetch('/api/engine-builder/run')" not in page
