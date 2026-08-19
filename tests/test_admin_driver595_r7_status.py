from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admin import driver595_r7_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]
BOOT_ID = "e314924f-f2d5-4a86-9afa-7ee7a88068dc"


def test_driver_r7_card_replays_narrow_current_boot_acceptance(monkeypatch) -> None:
    monkeypatch.setattr(status, "_current_boot_id", lambda: BOOT_ID)
    card = status.load_driver595_r7_status()
    assert card["state"] == "current_boot_prerequisite_accepted"
    assert card["decision"] == "ACCEPT"
    assert card["current_boot_prerequisite_accepted"] is True
    assert card["current_boot_match"] is True
    assert card["driver"] == {"version": "595.71.05", "cuda_driver": "13.2"}
    assert card["kernel"] == "7.0.0-28-generic"
    assert card["gpu"]["name"] == "NVIDIA RTX A5000 Laptop GPU"
    assert card["gpu"]["compute_process_count"] == 0
    assert card["tests"] == {"author_passed": 35, "independent_passed": 45, "combined_passed": 80, "failed": 0}
    assert card["checks"] == {"passed": 8, "total": 8}
    for key in (
        "gpu_workload_authorized",
        "deepstream_runtime_authorized",
        "deepstream_runtime_validated",
        "production_ready",
        "download_authorized",
        "install_authorized",
        "remove_authorized",
        "update_authorized",
        "reboot_authorized",
        "execution_actions_available",
    ):
        assert card[key] is False


def test_driver_r7_card_fails_closed_on_review_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    size, _ = pins[status.REVIEW_PATH]
    pins[status.REVIEW_PATH] = (size, "0" * 64)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_driver595_r7_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["current_boot_prerequisite_accepted"] is False
    assert card["gpu_workload_authorized"] is False
    assert card["deepstream_runtime_authorized"] is False


def test_driver_r7_card_rejects_resigned_gpu_authority_overclaim(monkeypatch) -> None:
    original = status._read_exact

    def read(relative: str) -> bytes:
        raw = original(relative)
        if relative != status.REVIEW_PATH:
            return raw
        value = json.loads(raw)
        value["authority"]["gpu_workload_authorized"] = True
        unsigned = {key: item for key, item in value.items() if key != "review_fingerprint_sha256"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        value["review_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_driver595_r7_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["gpu_workload_authorized"] is False
    assert card["deepstream_runtime_authorized"] is False


def test_driver_r7_card_expires_after_boot_identity_change(monkeypatch) -> None:
    monkeypatch.setattr(status, "_current_boot_id", lambda: "00000000-0000-0000-0000-000000000000")
    card = status.load_driver595_r7_status()
    assert card["state"] == "stale_boot_identity"
    assert card["reason"] == "driver595_r7_boot_identity_changed"
    assert card["available"] is True
    assert card["decision"] == "ACCEPT"
    assert card["current_boot_match"] is False
    assert card["current_boot_prerequisite_accepted"] is False
    assert card["gpu_workload_authorized"] is False


def test_validation_payload_and_ui_show_r7_without_execution_actions(monkeypatch) -> None:
    monkeypatch.setattr(status, "_current_boot_id", lambda: BOOT_ID)
    card = load_validation_status()["campaigns"]["driver595_live_qualification_r7"]
    assert card["state"] == "current_boot_prerequisite_accepted"
    assert card["gpu_workload_authorized"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "driver595_live_qualification_r7" in page
    assert "campaign.gpu_workload_authorized" in page
    assert "fetch('/api/driver595/r7/run')" not in page


def test_admin_compose_mounts_only_required_r7_closure_members() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "./tests/test_driver595_live_qualification_r7.py:/workspace/tests/test_driver595_live_qualification_r7.py:ro" in compose
    assert "./tests/test_driver595_live_qualification_r7_independent_review.py:/workspace/tests/test_driver595_live_qualification_r7_independent_review.py:ro" in compose
    assert "./docs/deepstream91-driver595-live-qualification-r7.md:/workspace/docs/deepstream91-driver595-live-qualification-r7.md:ro" in compose
