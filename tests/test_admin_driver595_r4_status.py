from __future__ import annotations

import json
from pathlib import Path

from admin import driver595_r4_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_driver_r4_card_replays_reject_and_keeps_all_actions_closed() -> None:
    card = status.load_driver595_r4_status()
    assert card["state"] == "independent_reject_successor_required"
    assert card["decision"] == "REJECT"
    assert card["severity"] == {"p0": 0, "p1": 7, "p2": 3}
    assert card["cache"] == {"exact_present": 15, "required": 22, "missing": 7, "dedicated_present": False}
    assert card["tests"]["collected_subject_count"] == 39
    assert card["tests"]["claimed_subject_count"] == 44
    for key in ("subject_accepted", "operational_ready", "download_authorized", "install_authorized", "reboot_authorized", "gpu_workload_authorized", "production_ready"):
        assert card[key] is False
    assert card["execution_actions_available"] is False


def test_driver_r4_card_fails_closed_on_review_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    relative = "validation/results/driver595-maintenance/r4-independent-review/review.json"
    pins[relative] = (pins[relative][0], "0" * 64)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_driver595_r4_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["subject_accepted"] is False
    assert card["install_authorized"] is False
    assert card["reboot_authorized"] is False


def test_driver_r4_card_rejects_semantically_resigned_acceptance_overclaim(monkeypatch) -> None:
    original = status._read_exact
    target = "validation/results/driver595-maintenance/r4-independent-review/review.json"

    def read(relative: str) -> bytes:
        raw = original(relative)
        if relative != target:
            return raw
        value = json.loads(raw)
        value["decision"] = "ACCEPT"
        value["authority"]["install_authorized"] = True
        unsigned = {key: item for key, item in value.items() if key != "review_fingerprint_sha256"}
        import hashlib

        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        value["review_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_driver595_r4_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["subject_accepted"] is False
    assert card["install_authorized"] is False


def test_validation_payload_and_ui_show_driver_reject_not_authorization() -> None:
    card = load_validation_status()["campaigns"]["driver595_maintenance_r4"]
    assert card["decision"] == "REJECT"
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "driver595_maintenance_r4" in page
    assert "campaign.install_authorized" in page
    assert "fetch('/api/driver595/install')" not in page
