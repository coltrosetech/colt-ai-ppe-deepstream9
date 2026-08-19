from __future__ import annotations

from pathlib import Path

from admin import ds91_native_status as native
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_native_build_card_replays_exact_cpu_receipt() -> None:
    card = native.load_deepstream91_native_status()
    assert card["state"] == "passed_cpu_only"
    assert card["cpu_build_accepted"] is True
    assert card["gpu_runtime_qualified"] is False
    assert card["production_ready"] is False
    assert card["execution_actions_available"] is False
    assert card["runtime"]["deepstream"] == "9.1.0"
    assert card["runtime"]["cuda"] == "13.2.0.046"
    assert card["runtime"]["tensorrt"] == "10.16.0.72"
    assert card["image"]["id"] == native.IMAGE_ID
    assert card["image"]["live_cache_checked_by_admin"] is False
    assert card["tests"] == {"passed": 16, "failed": 0, "artifacts": 7}


def test_native_build_card_fails_closed_on_any_pin_drift(monkeypatch) -> None:
    pins = dict(native.PINS)
    pins["Dockerfile"] = (pins["Dockerfile"][0], "0" * 64)
    monkeypatch.setattr(native, "PINS", pins)
    card = native.load_deepstream91_native_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["available"] is False
    assert card["cpu_build_accepted"] is False
    assert card["production_ready"] is False


def test_validation_payload_and_ui_expose_read_only_native_card() -> None:
    payload = load_validation_status()
    card = payload["campaigns"]["deepstream91_native_build"]
    assert card["state"] == "passed_cpu_only"
    assert payload["read_only"] is True
    assert payload["execution_actions_available"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "deepstream91_native_build" in page
    assert "campaign.gpu_runtime_qualified" in page
    assert "fetch('/api/deepstream91/native/build')" not in page


def test_admin_image_copies_native_evidence_but_exposes_no_build_command() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    assert "COPY deepstream /app/deepstream" in dockerfile
    assert "deepstream/ds91-native/build.sh" not in dockerfile
