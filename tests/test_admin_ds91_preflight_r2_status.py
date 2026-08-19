from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admin import ds91_preflight_r2_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_r2_card_accepts_only_cpu_preflight_and_keeps_execution_closed() -> None:
    card = status.load_ds91_preflight_r2_status()
    assert card["state"] == "accepted_cpu_preflight_runtime_closed"
    assert card["decision"] == "ACCEPT"
    assert card["cpu_preflight_accepted"] is True
    assert card["severity"] == {"p0": 0, "p1": 0, "p2": 0}
    assert card["tests"] == {
        "author_passed": 39,
        "independent_passed": 44,
        "failed": 0,
    }
    assert card["matrix"] == {
        "sources": 12,
        "distinct_video_types": 12,
        "profiles": [640, 960],
        "simulated_streams": 12,
        "measurement_seconds_per_profile": 300,
    }
    assert card["r1_findings_fixed"] == 5
    assert card["remaining_blockers"] == 11
    for key in (
        "execution_ready",
        "gpu_workload_authorized",
        "engine_build_authorized",
        "benchmark_authorized",
        "endurance_authorized",
        "production_ready",
    ):
        assert card[key] is False
    assert card["execution_actions_available"] is False


def test_r2_card_fails_closed_on_subject_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    relative = "validation/deepstream91_full_stack_preflight_r2.py"
    size, _, mode = pins[relative]
    pins[relative] = (size, "0" * 64, mode)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_ds91_preflight_r2_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["cpu_preflight_accepted"] is False
    assert card["gpu_workload_authorized"] is False


def test_r2_card_rejects_resigned_runtime_overclaim(monkeypatch) -> None:
    original = status._read_exact

    def read(relative: str) -> bytes:
        raw = original(relative)
        if relative != status.REVIEW_RELATIVE:
            return raw
        value = json.loads(raw)
        value["accepted_scope"]["gpu_runtime_qualified"] = True
        value["permissions"]["gpu_workload"] = True
        unsigned = {
            key: item for key, item in value.items() if key != "self_fingerprint"
        }
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        value["self_fingerprint"]["canonical_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_ds91_preflight_r2_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["execution_ready"] is False
    assert card["gpu_workload_authorized"] is False


def test_validation_payload_and_ui_show_narrow_accept_without_run_endpoint() -> None:
    card = load_validation_status()["campaigns"][
        "deepstream91_full_stack_preflight_r2"
    ]
    assert card["cpu_preflight_accepted"] is True
    assert card["execution_ready"] is False
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "deepstream91_full_stack_preflight_r2" in page
    assert "campaign.engine_build_authorized" in page
    assert "fetch('/api/deepstream91/preflight/run')" not in page
