from __future__ import annotations

from pathlib import Path

from admin import ppe_safetyvision_status as safetyvision
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_safetyvision_card_replays_exact_static_receipt() -> None:
    card = safetyvision.load_safetyvision_challenger_status()
    assert card["state"] == "challenger_static_verified"
    assert card["available"] is True
    assert card["ready"] is False
    assert card["accepted_model"] is False
    assert card["production_ready"] is False
    assert card["execution_actions_available"] is False
    assert card["model"]["family"] == "Ultralytics YOLOv8s"
    assert card["model"]["runtime_class_mapping"] == {
        "helmet": 3,
        "no_helmet": 7,
        "hi_vis": 12,
        "no_hi_vis": 9,
    }
    assert card["artifacts"]["640"]["present"] is True
    assert card["artifacts"]["640"]["fixed_batch"] == 1
    assert card["artifacts"]["896"]["present"] is True
    assert card["artifacts"]["960"]["present"] is False
    assert card["reported_metrics"]["locally_reproduced"] is False
    assert card["reported_metrics"]["onnx_640_map50"] == 0.738
    assert card["readiness"]["dynamic_batch_12_present"] is False
    assert card["readiness"]["overhead_camera_qualified"] is False
    assert card["tests"] == {
        "onnx_checker_passed": 2,
        "unit_passed": 10,
        "failed": 0,
    }


def test_safetyvision_card_fails_closed_on_pin_drift(monkeypatch) -> None:
    pins = dict(safetyvision.PINS)
    pins["receipt"] = (pins["receipt"][0], "0" * 64)
    monkeypatch.setattr(safetyvision, "PINS", pins)
    card = safetyvision.load_safetyvision_challenger_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["available"] is False
    assert card["accepted_model"] is False
    assert card["production_ready"] is False


def test_validation_payload_and_ui_expose_read_only_challenger() -> None:
    payload = load_validation_status()
    card = payload["campaigns"]["ppe_safetyvision_challenger"]
    assert card["state"] == "challenger_static_verified"
    assert payload["read_only"] is True
    assert payload["execution_actions_available"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "ppe_safetyvision_challenger" in page
    assert "campaign.accepted_model" in page
    assert "fetch('/api/ppe/safetyvision/run')" not in page


def test_admin_mounts_challenger_evidence_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./validation/results:/workspace/validation-results:ro" in compose
    assert "./validation:/workspace/validation:ro" in compose
    assert "./data:/workspace/data:ro" in compose
