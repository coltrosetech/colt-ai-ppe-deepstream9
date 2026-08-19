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
    source = ROOT / pin["path"]
    destination = workspace / pin["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@pytest.fixture
def ppe_source_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for pin in admin_validation.PPE_LO_CPPED_SOURCE_ADMIN_PINS.values():
        _copy_pin(workspace, pin)
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    return admin_validation._ppe_lo_cpped_source_quarantine(reader)


def test_checked_in_lo_cpped_projection_is_metadata_only_and_blocked() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["state"] == "metadata_only_training_blocked"
    assert all(projected["integrity"].values())
    assert projected["sources"]["lo"]["images"] == 11000
    assert projected["sources"]["lo"]["bounding_boxes"] == 88725
    assert projected["sources"]["cpped"]["images"] == 2612
    assert projected["sources"]["cpped"]["bounding_boxes"] == 20172
    assert projected["history"]["r1_superseded"] is True
    assert projected["history"]["r1_authoritative"] is False
    assert projected["history"]["r2_authoritative"] is True
    assert projected["execution_boundary"] == {
        "metadata_only": True,
        "dataset_bytes_downloaded_or_persisted": False,
        "annotations_or_final_test_labels_opened": False,
        "training_or_inference_executed": False,
        "gpu_used": False,
    }
    assert not any(projected["gates"].values())
    assert not any(projected["eligibility"].values())


def test_projection_does_not_expose_paths_hashes_urls_or_remote_ids() -> None:
    serialized = json.dumps(_project(ROOT), ensure_ascii=False)

    assert re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", serialized) is None
    for fragment in (
        "/home/",
        "validation/results/",
        "data/manifests/",
        ".json",
        "https://",
        "1SBlAHoviLHT8uCF0Lhv0ED_Yrdbfwzpq",
        "1pd2_kzUA82M25s8HVL6rHCUUWCaTdVnv",
        "22de323a1ad4e27ed95cf586495399265a821ce7",
    ):
        assert fragment not in serialized


def test_missing_or_tampered_current_receipt_fails_closed(
    ppe_source_workspace: Path,
) -> None:
    relative = admin_validation.PPE_LO_CPPED_SOURCE_ADMIN_PINS[
        "current_receipt"
    ]["path"]
    path = ppe_source_workspace / relative
    path.unlink()

    missing = _project(ppe_source_workspace)

    assert missing["available"] is False
    assert missing["reason"] == "current_receipt_missing"
    assert not any(missing["gates"].values())

    _copy_pin(
        ppe_source_workspace,
        admin_validation.PPE_LO_CPPED_SOURCE_ADMIN_PINS["current_receipt"],
    )
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b"tamper")

    tampered = _project(ppe_source_workspace)

    assert tampered["available"] is False
    assert tampered["reason"] == "current_receipt_pin_mismatch"
    assert tampered["integrity"]["current_receipt_exact_pin_verified"] is False


def test_resealed_rights_overclaim_is_rejected_by_semantic_replay(
    ppe_source_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.PPE_LO_CPPED_SOURCE_ADMIN_PINS)
    self_hashes = copy.deepcopy(
        admin_validation.PPE_LO_CPPED_RECEIPT_SELF_SHA256
    )

    manifest_path = ppe_source_workspace / pins["manifest"]["path"]
    manifest_path.chmod(0o640)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["rights"][
        "dataset_specific_license_status"
    ] = "commercial_training_cleared"
    manifest["sources"][0]["rights"][
        "commercial_derivative_training_allowed"
    ] = True
    manifest_raw = _write_json(manifest_path, manifest)
    pins["manifest"] = _pin(pins["manifest"]["path"], manifest_raw)

    schema_path = ppe_source_workspace / pins["schema"]["path"]
    schema_path.chmod(0o640)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    summary = schema["properties"]["summary"]["properties"]
    summary["manifest_bytes"]["const"] = pins["manifest"]["bytes"]
    summary["manifest_sha256"]["const"] = pins["manifest"]["sha256"]
    schema_raw = _write_json(schema_path, schema)
    pins["schema"] = _pin(pins["schema"]["path"], schema_raw)

    for key, self_key in (
        ("current_receipt", "current"),
        ("historical_receipt", "historical"),
    ):
        receipt_path = ppe_source_workspace / pins[key]["path"]
        receipt_path.chmod(0o640)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inputs"]["manifest"] = pins["manifest"]
        receipt["inputs"]["schema"] = pins["schema"]
        receipt["summary"]["manifest_bytes"] = pins["manifest"]["bytes"]
        receipt["summary"]["manifest_sha256"] = pins["manifest"]["sha256"]
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = admin_validation._canonical_sha256(
            receipt
        )
        self_hashes[self_key] = receipt["receipt_sha256"]
        receipt_raw = _write_json(receipt_path, receipt)
        pins[key] = _pin(pins[key]["path"], receipt_raw)

    monkeypatch.setattr(
        admin_validation, "PPE_LO_CPPED_SOURCE_ADMIN_PINS", pins
    )
    monkeypatch.setattr(
        admin_validation,
        "PPE_LO_CPPED_RECEIPT_SELF_SHA256",
        self_hashes,
    )

    projected = _project(ppe_source_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "exact_pin_schema_or_semantic_replay_invalid"
    assert projected["integrity"]["manifest_exact_pin_verified"] is True
    assert projected["integrity"]["current_receipt_self_hash_replayed"] is True
    assert projected["integrity"]["manifest_semantics_replayed"] is False
    assert not any(projected["gates"].values())


def test_validation_api_and_ui_show_dedicated_lo_cpped_card(
    ppe_source_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(ppe_source_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    campaign = response.json()["campaigns"]["ppe_public_source_quarantine"]
    assert campaign["state"] == "metadata_only_training_blocked"
    assert campaign["eligibility"]["training_eligible"] is False
    serialized = json.dumps(campaign, ensure_ascii=False)
    assert re.search(r"[0-9a-f]{64}", serialized) is None
    assert "validation/results/" not in serialized
    assert page.status_code == 200
    assert "PPE Lo/CPPED açık kaynak karantinası" in page.text
    assert "Exact-pin / şema / replay bütünlüğü" in page.text
    assert "CPPED hedef eşleme" in page.text
