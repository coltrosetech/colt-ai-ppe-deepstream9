from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app


ROOT = Path(__file__).resolve().parents[1]


def _pin(path: str, raw: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _copy_pin(workspace: Path, pin: dict) -> None:
    source = ROOT / str(pin["path"])
    destination = workspace / str(pin["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture
def full_stack_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for pin in admin_validation.FULL_STACK_BENCHMARK_ADMIN_PINS.values():
        _copy_pin(workspace, pin)
    return workspace


def _reader(workspace: Path) -> admin_validation.ArtifactReader:
    return admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )


def _project(workspace: Path) -> dict:
    return admin_validation._full_stack_benchmark(_reader(workspace))


def test_checked_in_projection_separates_measured_baseline_from_estimate() -> None:
    campaign = _project(ROOT)

    assert campaign["available"] is True
    assert campaign["state"] == "blocked_missing_runtime_artifacts"
    assert campaign["execution_ready"] is False
    assert campaign["execution_actions_available"] is False
    assert campaign["final_claim_allowed"] is False
    assert campaign["scope"] == {
        "profiles": [640, 960],
        "separate_runs": True,
        "simulated_streams": 12,
        "distinct_sources": 12,
        "distinct_video_types": 12,
        "warmup_seconds_per_run": 15,
        "measurement_seconds_per_run": 300,
        "execution_mode": "foreground_only",
        "minimum_per_source_coverage": 0.95,
        "minimum_output_fps_per_source": 25.0,
    }
    assert campaign["blocker_count"] == 11
    assert len(campaign["blockers"]) == 11
    assert all(campaign["integrity"].values())

    expected = {
        "640": (464.733, 38.729, [90.0, 190.0], [7.5, 15.833]),
        "960": (305.799, 25.484, [55.0, 125.0], [4.583, 10.417]),
    }
    for profile, values in expected.items():
        projected = campaign["profiles"][profile]
        baseline = projected["person_only_baseline"]
        estimate = projected["full_stack_estimate"]
        assert baseline["classification"] == "person_only_measured_baseline"
        assert baseline["aggregate_mean_fps"] == values[0]
        assert baseline["per_stream_mean_fps"] == values[1]
        assert baseline["eligible_as_full_stack_result"] is False
        assert estimate["classification"] == "estimate_not_measured"
        assert estimate["aggregate_fps_range"] == values[2]
        assert estimate["per_stream_fps_range"] == values[3]
        assert estimate["eligible_as_result"] is False
        assert projected["measurement"] == {
            "executed": False,
            "result_available": False,
        }

    assert campaign["measurement"]["executed"] is False
    assert campaign["measurement"]["full_stack_result_available"] is False


def test_projection_redacts_paths_hashes_digests_and_private_identity() -> None:
    serialized = json.dumps(_project(ROOT), ensure_ascii=False)

    assert re.search(
        r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", serialized
    ) is None
    for fragment in (
        "/home/",
        "/workspace/",
        "validation/",
        "file:///",
        ".json",
        ".engine",
        "sha256:",
        "ced1b591",
        "2aa44",
        "GPU-",
    ):
        assert fragment not in serialized


def test_missing_or_tampered_plan_fails_closed(
    full_stack_workspace: Path,
) -> None:
    pin = admin_validation.FULL_STACK_BENCHMARK_ADMIN_PINS["plan"]
    path = full_stack_workspace / str(pin["path"])
    path.unlink()

    missing = _project(full_stack_workspace)

    assert missing["available"] is False
    assert missing["reason"] == "plan_missing"
    assert missing["execution_ready"] is False
    assert missing["integrity"]["plan_exact_pin_verified"] is False

    _copy_pin(full_stack_workspace, pin)
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b"tamper")

    tampered = _project(full_stack_workspace)

    assert tampered["available"] is False
    assert tampered["reason"] == "plan_pin_mismatch"
    assert tampered["execution_ready"] is False
    assert tampered["integrity"]["plan_exact_pin_verified"] is False


def test_resealed_ready_overclaim_fails_semantic_replay(
    full_stack_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.FULL_STACK_BENCHMARK_ADMIN_PINS)
    relative = str(pins["plan"]["path"])
    path = full_stack_workspace / relative
    path.chmod(0o640)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["state"] = "ready"
    plan["readiness"]["execution_ready"] = True
    plan["readiness"]["blockers"] = []
    plan["runtime"]["fusion"]["capability_status"] = "runtime_ready"
    for profile in plan["profiles"]:
        person = profile["modules"][0]
        profile["authorization_plan_pin"] = copy.deepcopy(
            person["infer_config_pin"]
        )
        for module in profile["modules"][1:]:
            module["status"] = "ready"
            module["engine_pin"] = copy.deepcopy(person["engine_pin"])
            module["infer_config_pin"] = copy.deepcopy(
                person["infer_config_pin"]
            )
    plan.pop("fingerprint_sha256")
    fingerprint = admin_validation._canonical_sha256(plan)
    plan["fingerprint_sha256"] = fingerprint
    raw = _write_json(path, plan)
    pins["plan"] = _pin(relative, raw)
    monkeypatch.setattr(
        admin_validation, "FULL_STACK_BENCHMARK_ADMIN_PINS", pins
    )
    monkeypatch.setattr(
        admin_validation,
        "FULL_STACK_BENCHMARK_PLAN_FINGERPRINT",
        fingerprint,
    )

    projected = _project(full_stack_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "exact_pin_schema_or_semantic_replay_invalid"
    )
    assert projected["integrity"]["plan_exact_pin_verified"] is True
    assert projected["integrity"]["plan_schema_replayed"] is True
    assert projected["integrity"]["plan_fingerprint_replayed"] is True
    assert projected["integrity"]["plan_semantics_replayed"] is False
    assert projected["execution_ready"] is False


def test_resealed_receipt_result_kind_drift_fails_contract_replay(
    full_stack_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.FULL_STACK_BENCHMARK_ADMIN_PINS)
    relative = str(pins["receipt_schema"]["path"])
    path = full_stack_workspace / relative
    path.chmod(0o640)
    schema = json.loads(path.read_text(encoding="utf-8"))
    schema["properties"]["result_kind"]["const"] = "estimate_not_measured"
    raw = _write_json(path, schema)
    pins["receipt_schema"] = _pin(relative, raw)
    monkeypatch.setattr(
        admin_validation, "FULL_STACK_BENCHMARK_ADMIN_PINS", pins
    )

    projected = _project(full_stack_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "exact_pin_schema_or_semantic_replay_invalid"
    )
    assert projected["integrity"][
        "receipt_schema_exact_pin_verified"
    ] is True
    assert projected["integrity"]["receipt_contract_replayed"] is False
    assert projected["execution_ready"] is False


def test_projection_does_not_mutate_product_readiness() -> None:
    before = admin_validation._product_readiness(_reader(ROOT))
    admin_validation._full_stack_benchmark(_reader(ROOT))
    after = admin_validation._product_readiness(_reader(ROOT))

    assert after == before


def test_validation_api_and_ui_show_read_only_dedicated_campaign(
    full_stack_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(full_stack_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    campaign = payload["campaigns"]["full_stack_benchmark"]
    assert payload["read_only"] is True
    assert payload["execution_actions_available"] is False
    assert campaign["execution_ready"] is False
    assert campaign["blocker_count"] == 11
    assert len(campaign["blockers"]) == 11
    assert campaign["profiles"]["640"]["person_only_baseline"][
        "classification"
    ] == "person_only_measured_baseline"
    assert campaign["profiles"]["640"]["full_stack_estimate"][
        "classification"
    ] == "estimate_not_measured"
    assert page.status_code == 200
    assert "Üç modül 12-kamera / 5-dakika benchmark" in page.text
    assert "Person-only measured baseline — full-stack sonucu değil" in page.text
    assert (
        "Full-stack estimate_not_measured — ölçülmüş sonuç değil"
        in page.text
    )
    assert "Açık blokajlar" in page.text


def test_dedicated_ui_section_has_no_execution_action() -> None:
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<section id="fullStackBenchmarkSection">(.*?)</section>',
        page,
        flags=re.DOTALL,
    )

    assert match is not None
    section = match.group(1)
    assert "<button" not in section
    assert "onclick=" not in section
    assert "/api/" not in section


def test_admin_image_packages_only_exact_pinned_small_control_bundle() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for key in ("validator", "plan", "plan_schema", "receipt_schema"):
        pin = admin_validation.FULL_STACK_BENCHMARK_ADMIN_PINS[key]
        path = str(pin["path"])
        raw = (ROOT / path).read_bytes()
        assert len(raw) == pin["bytes"]
        assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
        assert f"COPY {path} /workspace/{path}" in dockerfile
        if key in {"validator", "plan"}:
            assert f"!{path}" in dockerignore

    assert "COPY validation/results" not in dockerfile
