from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admin import model_gate_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_person_r14i_card_projects_narrow_independent_acceptance() -> None:
    card = status.load_person_r14i_status()
    assert card["state"] == "independent_static_controller_accepted_execution_closed"
    assert card["decision"] == "ACCEPT"
    assert card["independent_static_acceptance"] is True
    assert card["profiles"] == [640, 960]
    assert card["batch"] == {"min": 1, "opt": 12, "max": 12}
    assert card["tests"] == {
        "author_passed": 50,
        "independent_passed": 67,
        "combined_passed": 117,
        "failed": 0,
    }
    assert card["planned_outputs"] == 18
    assert card["outputs_present"] == 0
    for key in (
        "engine_build_authorized",
        "gpu_workload_authorized",
        "deepstream_runtime_authorized",
        "parity_authorized",
        "config_publication_authorized",
        "production_ready",
        "execution_actions_available",
    ):
        assert card[key] is False


def test_person_r14i_card_fails_closed_on_exact_pin_drift(monkeypatch) -> None:
    pins = dict(status.PERSON_PINS)
    size, _, mode = pins[status.PERSON_REVIEW]
    pins[status.PERSON_REVIEW] = (size, "0" * 64, mode)
    monkeypatch.setattr(status, "PERSON_PINS", pins)
    card = status.load_person_r14i_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["independent_static_acceptance"] is False
    assert card["engine_build_authorized"] is False
    assert card["gpu_workload_authorized"] is False


def test_person_r14i_card_rejects_resigned_engine_authority_overclaim(monkeypatch) -> None:
    original = status._read_exact

    def read(relative: str, pins: dict[str, tuple[int, str, str]]) -> bytes:
        raw = original(relative, pins)
        if relative != status.PERSON_REVIEW:
            return raw
        value = json.loads(raw)
        value["accepted_scope"]["engine_build_authorized"] = True
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "review_fingerprint_sha256"
        }
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        value["review_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_person_r14i_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["engine_build_authorized"] is False


def test_ppe_a32_card_projects_context_acceptance_without_build_authority() -> None:
    card = status.load_ppe_a32_status()
    assert card["state"] == "independent_phase_b_context_accepted_build_closed"
    assert card["decision"] == "ACCEPT"
    assert card["phase_b_context_accepted"] is True
    assert card["tests"] == {
        "author_runs": 2,
        "author_passed_each": 25,
        "independent_passed": 54,
        "failed": 0,
    }
    assert card["integrity"]["source_pins"] == 39
    assert card["integrity"]["strict_receipts"] == 6
    assert card["integrity"]["symlinks"] == 47
    assert card["integrity"]["writable_or_special_entries"] == 0
    for key in (
        "cpu_image_build_authorized",
        "context_copy_authorized",
        "model_or_onnx_load_authorized",
        "gpu_workload_authorized",
        "tensorrt_or_deepstream_authorized",
        "quality_validated",
        "production_ready",
        "execution_actions_available",
    ):
        assert card[key] is False


def test_ppe_a32_card_rejects_resigned_docker_authority_overclaim(monkeypatch) -> None:
    original = status._read_exact

    def read(relative: str, pins: dict[str, tuple[int, str, str]]) -> bytes:
        raw = original(relative, pins)
        if relative != status.PPE_REVIEW:
            return raw
        value = json.loads(raw)
        value["authority"]["docker_authorized"] = True
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "review_fingerprint_sha256"
        }
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        value["review_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    card = status.load_ppe_a32_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["phase_b_context_accepted"] is False
    assert card["cpu_image_build_authorized"] is False


def test_validation_payload_and_ui_include_both_read_only_model_gate_cards() -> None:
    campaigns = load_validation_status()["campaigns"]
    assert campaigns["person_rtdetrv4_tensorrt_r14i"]["decision"] == "ACCEPT"
    assert campaigns["ppe_safetyvision_phase_b_a32"]["decision"] == "ACCEPT"
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "person_rtdetrv4_tensorrt_r14i" in page
    assert "ppe_safetyvision_phase_b_a32" in page
    assert "fetch('/api/person/r14i/run')" not in page
    assert "fetch('/api/ppe/a32/build')" not in page


def test_admin_compose_mounts_exact_test_and_doc_closure_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    expected = (
        "tests/test_person_rtdetrv4_tensorrt_r14i.py",
        "tests/test_person_rtdetrv4_tensorrt_r14i_independent_review.py",
        "docs/person-rtdetrv4-tensorrt-r14i.md",
        "docs/person-rtdetrv4-tensorrt-r14i-independent-review.md",
        "tests/test_ppe_safetyvision_r5_phase_a32_postpublication.py",
        "tests/test_ppe_safetyvision_r5_phase_b_a32_independent_review.py",
        "docs/ppe-safetyvision-r5-phase-a32-postpublication.md",
    )
    for relative in expected:
        assert f"./{relative}:/workspace/{relative}:ro" in compose
