from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admin import ds91_preflight_r1_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_r1_card_replays_immutable_reject_without_authority() -> None:
    card = status.load_ds91_preflight_r1_status()
    assert card["state"] == "independent_reject_successor_in_progress"
    assert card["decision"] == "REJECT"
    assert card["severity"] == {"p0": 0, "p1": 4, "p2": 1}
    assert card["tests"] == {
        "subject_passed": 13,
        "independent_passed": 30,
        "failed": 0,
    }
    assert card["matrix"] == {
        "sources": 12,
        "distinct_video_types": 12,
        "profiles": [640, 960],
        "simulated_streams": 12,
        "measurement_seconds_per_profile": 300,
    }
    for key in (
        "subject_accepted",
        "execution_ready",
        "engine_build_authorized",
        "benchmark_authorized",
        "gpu_workload_authorized",
        "endurance_authorized",
        "production_ready",
    ):
        assert card[key] is False
    assert card["execution_actions_available"] is False


def test_preflight_r1_card_fails_closed_on_receipt_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    size, _, mode = pins[status.REVIEW_RELATIVE]
    pins[status.REVIEW_RELATIVE] = (size, "0" * 64, mode)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_ds91_preflight_r1_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["subject_accepted"] is False
    assert card["gpu_workload_authorized"] is False


def test_preflight_r1_card_rejects_resigned_authority_overclaim(monkeypatch) -> None:
    original = status._read_exact

    def read(relative: str) -> bytes:
        raw = original(relative)
        if relative != status.REVIEW_RELATIVE:
            return raw
        value = json.loads(raw)
        value["authority"]["gpu_workload_authorized"] = True
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "review_fingerprint_sha256"
        }
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        value["review_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_ds91_preflight_r1_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["execution_ready"] is False
    assert card["gpu_workload_authorized"] is False


def test_validation_payload_and_ui_show_reject_without_run_endpoint() -> None:
    card = load_validation_status()["campaigns"][
        "deepstream91_full_stack_preflight_r1"
    ]
    assert card["decision"] == "REJECT"
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "deepstream91_full_stack_preflight_r1" in page
    assert "campaign.benchmark_authorized" in page
    assert "fetch('/api/deepstream91/preflight/run')" not in page
