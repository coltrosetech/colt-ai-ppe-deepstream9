from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from admin import gpu_lease_v5_status


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
    source = ROOT / pin["path"]
    destination = workspace / pin["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture
def gpu_r2_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for descriptor in admin_validation.GPU_REENTRY_R2_ADMIN_PINS.values():
        _copy_pin(workspace, descriptor)

    receipt = json.loads(
        (
            ROOT
            / admin_validation.GPU_REENTRY_R2_ADMIN_PINS["receipt"]["path"]
        ).read_text(encoding="utf-8")
    )
    plan = json.loads(
        (
            ROOT / admin_validation.GPU_REENTRY_R2_ADMIN_PINS["plan"]["path"]
        ).read_text(encoding="utf-8")
    )
    lease = receipt["gpu_lease"]
    current_pins = [
        *receipt["artifacts"],
        lease["acquire_receipt"],
        lease["release_receipt"],
        *lease["renew_receipts"],
        lease["contract_projection"]["artifact"],
        receipt["legacy_v1_reentry"]["evidence"],
        receipt["minimal_cuda_smoke"]["raw_output"],
        receipt["training_execution_image"]["build_receipt"],
    ]
    prior = plan["prior_failed_live_smoke"]
    historical_pins = [
        prior["plan"]["artifact"],
        *prior["artifacts"].values(),
        prior["lease"]["acquire_receipt"],
        prior["lease"]["release_receipt"],
    ]
    copied: set[str] = {
        item["path"]
        for item in admin_validation.GPU_REENTRY_R2_ADMIN_PINS.values()
    }
    for pin in [*current_pins, *historical_pins]:
        if pin["path"] not in copied:
            _copy_pin(workspace, pin)
            copied.add(pin["path"])
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    projected = admin_validation._gpu_reentry_r2(reader)
    assert projected is not None
    return projected


def test_checked_in_r2_is_fail_closed_after_source_pin_drift() -> None:
    projected = _project(ROOT)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["reason"] == "current_receipt_schema_pin_mismatch"
    assert projected["progress"] == {
        "completed": 0,
        "total": 6,
        "remaining": 6,
        "fraction": 0.0,
    }
    assert projected["integrity"]["current_receipt_exact_pin_verified"] is True
    assert projected["integrity"]["current_plan_exact_pin_verified"] is True
    assert projected["integrity"]["current_receipt_schema_exact_pin_verified"] is False
    assert projected["integrity"]["current_plan_schema_exact_pin_verified"] is False
    assert projected["integrity"]["validator_exact_pin_verified"] is False
    assert projected["current_run"] == {}
    assert projected["historical_failed_run"] == {}


def test_projection_keeps_drifted_r2_claims_closed(
    gpu_r2_workspace: Path,
) -> None:
    projected = _project(gpu_r2_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "current_receipt_schema_pin_mismatch"
    assert projected["sustained_load_authorized"] is False
    assert projected["pending_gate_ids"] == [
        "deepstream9_engine_smoke",
        "seven_day_endurance",
    ]
    assert projected["current_run"] == {}
    assert projected["historical_failed_run"] == {}

    serialized = json.dumps(projected, ensure_ascii=False)
    for private_fragment in (
        "/home/",
        "validation/results/",
        "sha256:",
        "GPU-8cbaba",
        "432d75df",
        "67eb1577",
    ):
        assert private_fragment not in serialized


def test_missing_current_receipt_does_not_promote_legacy_fallback(
    gpu_r2_workspace: Path,
) -> None:
    relative = admin_validation.GPU_REENTRY_R2_ADMIN_PINS["receipt"]["path"]
    (gpu_r2_workspace / relative).unlink()

    projected = _project(gpu_r2_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "current_receipt_missing"
    assert projected["current_run"] == {}
    assert projected["historical_failed_run"] == {}


def test_historical_live_001_tamper_fails_current_card_closed(
    gpu_r2_workspace: Path,
) -> None:
    plan = json.loads(
        (
            gpu_r2_workspace
            / admin_validation.GPU_REENTRY_R2_ADMIN_PINS["plan"]["path"]
        ).read_text(encoding="utf-8")
    )
    relative = plan["prior_failed_live_smoke"]["artifacts"][
        "docker_stderr_log"
    ]["path"]
    path = gpu_r2_workspace / relative
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b"tamper")

    projected = _project(gpu_r2_workspace)

    assert projected["available"] is False
    assert projected["integrity"][
        "failed_live_001_artifact_pins_verified"
    ] is False
    assert projected["historical_failed_run"] == {}


def test_source_pin_drift_precedes_exact_repinned_gpu_overclaim(
    gpu_r2_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.GPU_REENTRY_R2_ADMIN_PINS)
    relative = pins["receipt"]["path"]
    path = gpu_r2_workspace / relative
    path.chmod(0o640)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["claims"]["deepstream_smoke_executed"] = True
    receipt.pop("fingerprint_sha256")
    receipt["fingerprint_sha256"] = admin_validation._canonical_sha256(receipt)
    raw = _write_json(path, receipt)
    pins["receipt"] = _pin(relative, raw)
    monkeypatch.setattr(admin_validation, "GPU_REENTRY_R2_ADMIN_PINS", pins)

    projected = _project(gpu_r2_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "current_receipt_schema_pin_mismatch"
    assert projected["integrity"]["current_receipt_exact_pin_verified"] is True
    assert projected["integrity"]["current_receipt_schema_exact_pin_verified"] is False
    assert projected["integrity"]["current_receipt_fingerprint_replayed"] is False


def test_validation_api_shows_closed_r2_and_accepted_gpu_lease_v5(
    gpu_r2_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in gpu_lease_v5_status.PINS:
        source = ROOT / relative
        destination = gpu_r2_workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    results = tmp_path / "results"
    results.mkdir()
    lease_results = results / "gpu-leases/v5"
    lease_results.mkdir(parents=True)
    lease_results.chmod(0o700)
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(gpu_r2_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    campaigns = response.json()["campaigns"]
    gpu = campaigns["gpu_reentry"]
    assert gpu["available"] is False
    assert gpu["reason"] == "current_receipt_schema_pin_mismatch"
    assert gpu["current_run"] == {}
    assert gpu["historical_failed_run"] == {}
    assert gpu["pending_gate_ids"] == [
        "deepstream9_engine_smoke",
        "seven_day_endurance",
    ]
    lease = campaigns["gpu_lease_v5"]
    assert lease["available"] is True
    assert lease["contract_verified"] is True
    assert lease["tests"]["independent_review"] == "pass"
    assert page.status_code == 200
    assert "GPU exact-pin / replay bütünlüğü" in page.text
    assert "GPU Lease V5" in page.text
