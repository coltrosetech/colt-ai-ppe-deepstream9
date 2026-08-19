from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admin import pose_r13i_status as status
from admin.validation import load_validation_status


ROOT = Path(__file__).resolve().parents[1]


def test_pose_r13i_card_projects_narrow_independent_static_acceptance() -> None:
    card = status.load_pose_r13i_status()
    assert card["state"] == "independent_static_controller_accepted_v5_host_gate_blocked"
    assert card["decision"] == "ACCEPT"
    assert card["independent_static_acceptance"] is True
    assert card["v5_host_realization_eligible"] is False
    assert card["profiles"] == [640, 960]
    assert card["batch"] == {"min": 1, "opt": 12, "max": 12}
    assert card["keypoints"] == {
        "layout": "COCO",
        "count": 17,
        "same_index_contract": True,
    }
    assert card["tests"] == {
        "author_passed": 83,
        "independent_passed": 85,
        "combined_passed": 168,
        "failed": 0,
    }
    assert card["planned_outputs"] == 20
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


def test_pose_r13i_card_fails_closed_on_exact_pin_drift(monkeypatch) -> None:
    pins = dict(status.PINS)
    size, _, mode = pins[status.REVIEW_PATH]
    pins[status.REVIEW_PATH] = (size, "0" * 64, mode)
    monkeypatch.setattr(status, "PINS", pins)
    card = status.load_pose_r13i_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["independent_static_acceptance"] is False
    assert card["engine_build_authorized"] is False
    assert card["gpu_workload_authorized"] is False


def test_pose_r13i_card_rejects_resigned_runtime_authority_overclaim(
    monkeypatch,
) -> None:
    original = status._read_exact
    rewritten_fingerprint: dict[str, str] = {}

    def read(relative: str, pins: dict[str, tuple[int, str, str]]) -> bytes:
        raw = original(relative, pins)
        if relative != status.REVIEW_PATH:
            return raw
        value = json.loads(raw)
        value["accepted_scope"]["runtime_execution_authorized"] = True
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "review_fingerprint_sha256"
        }
        canonical = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        fingerprint = hashlib.sha256(canonical).hexdigest()
        value["review_fingerprint_sha256"] = fingerprint
        rewritten_fingerprint["value"] = fingerprint
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()

    monkeypatch.setattr(status, "_read_exact", read)
    original_validate = status._validate_fingerprint

    def validate(review: dict, expected: str) -> None:
        original_validate(review, rewritten_fingerprint.get("value", expected))

    monkeypatch.setattr(status, "_validate_fingerprint", validate)
    card = status.load_pose_r13i_status()
    assert card["state"] == "unavailable_integrity_error"
    assert card["engine_build_authorized"] is False
    assert card["deepstream_runtime_authorized"] is False


def test_validation_payload_and_ui_include_pose_r13i_read_only_card() -> None:
    card = load_validation_status()["campaigns"][
        "pose_mmpose_yoloxpose_tensorrt_r13i"
    ]
    assert card["decision"] == "ACCEPT"
    assert card["read_only"] is True
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    assert "pose_mmpose_yoloxpose_tensorrt_r13i" in page
    assert "fetch('/api/pose/r13i/run')" not in page


def test_admin_compose_mounts_pose_exact_test_and_doc_closure_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    expected = (
        "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i.py",
        "tests/test_pose_mmpose_yoloxpose_tensorrt_ds91_r13i_independent_review.py",
        "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i.md",
        "docs/pose-mmpose-yoloxpose-tensorrt-ds91-r13i-independent-review.md",
    )
    for relative in expected:
        assert f"./{relative}:/workspace/{relative}:ro" in compose
