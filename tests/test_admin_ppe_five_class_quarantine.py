from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from validation import ppe_five_class_admin_projection as projection_validator


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


@pytest.fixture
def five_class_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    descriptors = [
        *admin_validation.PPE_FIVE_CLASS_ADMIN_PINS.values(),
        *admin_validation.PPE_FIVE_CLASS_NORMALIZATION_R2_PINS.values(),
        *admin_validation.PPE_FIVE_CLASS_SEMANTIC_R4_PINS.values(),
        *admin_validation.PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS.values(),
    ]
    for descriptor in descriptors:
        relative = descriptor["path"]
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    return admin_validation._ppe_five_class_readiness(reader)


def test_checked_in_five_class_chain_is_available_but_not_ready() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["state"] == "dry_run_group_split_complete_training_blocked"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert all(projected["integrity"].values())
    assert all(value is False for value in projected["gates"].values())


def test_projection_exposes_bounded_summary_and_keeps_eligibility_closed(
    five_class_workspace: Path,
) -> None:
    projected = _project(five_class_workspace)

    assert projected["dataset"] == {
        "source": "Mendeley PPE Detection Dataset (5-Class)",
        "repository_license": "CC-BY-4.0",
        "repository_metadata_verified": True,
        "embedded_third_party_rights_audit_complete": False,
        "published_independent_test_split_present": False,
        "archive_bytes": 208799718,
        "archive_exact_pin_verified_at_projection_generation": True,
        "archive_read_by_admin": False,
    }
    assert projected["source_receipt"] == {
        "bytes": 5456369,
        "stream_pin_verified": True,
        "canonical_self_hash_verified_at_projection_generation": True,
        "parsed_by_admin": False,
        "compact_projection_verified": True,
    }
    quarantine = projected["quarantine"]
    assert quarantine["structural_pass"] is True
    assert quarantine["accepted"] is True
    assert quarantine["training_eligible"] is False
    assert quarantine["structural_gates_passed"] == 24
    assert quarantine["images"] == 2586
    assert quarantine["bounding_boxes"] == 17827
    assert quarantine["exact_duplicate_groups"] == 31
    assert quarantine["cross_split_exact_duplicate_groups"] == 10
    assert projected["eligibility"] == {
        "embedded_rights_audit_complete": False,
        "camera_site_session_group_safe": False,
        "person_equipment_semantics_normalized": False,
        "published_independent_test_split_ready": False,
        "training_eligible": False,
        "final_validation_or_test_eligible": False,
    }
    normalization = projected["normalization_group_split"]
    assert normalization["final_group_count"] == 292
    assert normalization["roles"] == {
        "train": {"groups": 145, "images": 2068},
        "calibration": {"groups": 75, "images": 259},
        "test": {
            "groups": 72,
            "images": 259,
            "claim": "internal_heldout_audit_only",
        },
    }
    assert normalization["leakage"]["zero"] is True
    assert set(normalization["leakage"].values()) == {0, True}
    assert normalization["training_eligible"] is False
    assert normalization["final_validation_or_test_eligible"] is False
    semantic = projected["semantic_audit_r4"]
    assert semantic == {
        "status": "ai_semantic_audit_complete_human_adjudication_required",
        "exact_evidence_verified": True,
        "sample_images": 20,
        "source_groups": 18,
        "bbox_rows_checked": 488,
        "roles": {"train": 14, "calibration": 6},
        "decisions": {
            "accepted_with_guardrails": 2,
            "questionable_needs_adjudication": 15,
            "rejected_development_candidates": 3,
        },
        "issue_counts": {
            "vest_hi_vis_semantic_risk": 3,
            "helmet_semantic_ambiguity": 2,
            "no_vest_no_hi_vis_semantic_risk": 17,
        },
        "development_holdout_payload_files_opened": 0,
        "human_adjudication_required": True,
        "semantic_mapping_approved": False,
        "training_authorized_by_this_audit": False,
        "production_ready": False,
        "critical_findings": [
            "vest_to_hi_vis_harness_misclassification",
            "helmet_worn_vs_carried_ambiguous",
            "no_vest_to_no_hi_vis_unproven",
        ],
    }
    launch_gate = projected["semantic_launch_gate_r3"]
    assert launch_gate["historical_r2_plan_immutable"] is True
    assert launch_gate["image_build_preparation_allowed"] is True
    assert launch_gate["blocked_modes"] == [
        "smoke_train",
        "baseline_calibration",
        "full_train_150e",
        "resume",
        "evaluation",
        "export",
    ]
    assert launch_gate["new_authorization_receipt_present"] is False
    assert launch_gate["release_requirements_satisfied"] is False
    assert launch_gate["training_ready"] is False
    assert launch_gate["production_ready"] is False
    assert projected["quarantine_history"] == {
        "r1_stream_pin_verified": True,
        "r1_superseded": True,
        "r1_authoritative": False,
        "r1_training_eligible": False,
        "r2_stream_pin_verified": True,
        "r2_authoritative": True,
        "r2_structural_pass": True,
        "r2_accepted": True,
        "r2_training_eligible": False,
        "normalization_r2_current": True,
    }

    serialized = json.dumps(projected, ensure_ascii=False)
    for private_fragment in (
        "/workspace",
        "data/manifests/",
        "data/raw/",
        "validation/results/",
        "https://",
        "c06e7357",
        "2c845f04",
        "bf9af5ce",
        "4bc432e1",
        ".zip",
        ".json",
    ):
        assert private_fragment not in serialized


def test_admin_never_sends_large_receipt_to_json_parser(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = admin_validation._workspace_pin_json
    parsed_paths: list[str] = []

    def guarded(*args, **kwargs):
        expected_path = kwargs["expected_path"]
        parsed_paths.append(expected_path)
        assert expected_path != admin_validation.PPE_FIVE_CLASS_ADMIN_PINS[
            "authoritative_receipt"
        ]["path"]
        assert expected_path != admin_validation.PPE_FIVE_CLASS_ADMIN_PINS[
            "superseded_r1_receipt"
        ]["path"]
        return original(*args, **kwargs)

    monkeypatch.setattr(admin_validation, "_workspace_pin_json", guarded)

    projected = _project(five_class_workspace)

    assert projected["available"] is True
    assert projected["source_receipt"]["parsed_by_admin"] is False
    assert len(parsed_paths) == 11


def test_large_receipt_tamper_fails_card_closed(
    five_class_workspace: Path,
) -> None:
    relative = admin_validation.PPE_FIVE_CLASS_ADMIN_PINS[
        "authoritative_receipt"
    ]["path"]
    path = five_class_workspace / relative
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b" ")

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "authoritative_receipt_pin_mismatch"
    assert projected["integrity"][
        "authoritative_receipt_stream_pin_verified"
    ] is False
    assert projected["source_receipt"]["parsed_by_admin"] is False
    assert projected["quarantine"]["structural_pass"] is False
    assert all(value is False for value in projected["gates"].values())


def test_missing_compact_receipt_fails_card_closed(
    five_class_workspace: Path,
) -> None:
    relative = admin_validation.PPE_FIVE_CLASS_ADMIN_PINS[
        "projection_receipt"
    ]["path"]
    (five_class_workspace / relative).unlink()

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "compact_receipt_missing"
    assert projected["integrity"]["compact_receipt_exact_pin_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_source_receipt_explicit_size_bound_fails_closed(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        admin_validation,
        "PPE_FIVE_CLASS_SOURCE_RECEIPT_MAX_BYTES",
        5456368,
    )

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "authoritative_receipt_pin_mismatch"
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_compact_overclaim_is_rejected(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.PPE_FIVE_CLASS_ADMIN_PINS)
    relative = pins["projection_receipt"]["path"]
    path = five_class_workspace / relative
    path.chmod(0o640)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["gates"]["production_ready"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    raw = _write_json(path, receipt)
    pins["projection_receipt"] = {
        **_pin(relative, raw),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    monkeypatch.setattr(admin_validation, "PPE_FIVE_CLASS_ADMIN_PINS", pins)

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "five_class_projection_normalization_or_semantic_gate_invalid"
    )
    assert projected["integrity"]["compact_receipt_self_hash_verified"] is True
    assert projected["integrity"]["compact_schema_replay_verified"] is False
    assert projected["integrity"]["compact_semantics_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_superseded_r1_stream_history_is_required(
    five_class_workspace: Path,
) -> None:
    relative = admin_validation.PPE_FIVE_CLASS_ADMIN_PINS[
        "superseded_r1_receipt"
    ]["path"]
    (five_class_workspace / relative).unlink()

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "superseded_r1_receipt_missing"
    assert projected["integrity"][
        "superseded_r1_receipt_stream_pin_verified"
    ] is False
    assert projected["quarantine_history"]["r1_stream_pin_verified"] is False


def test_exact_repinned_normalization_training_overclaim_is_rejected(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(
        admin_validation.PPE_FIVE_CLASS_NORMALIZATION_R2_PINS
    )
    relative = pins["receipt"]["path"]
    path = five_class_workspace / relative
    path.chmod(0o640)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["readiness"]["normalized_dataset_training_eligible"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    raw = _write_json(path, receipt)
    pins["receipt"] = {
        **_pin(relative, raw),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    monkeypatch.setattr(
        admin_validation,
        "PPE_FIVE_CLASS_NORMALIZATION_R2_PINS",
        pins,
    )

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "five_class_projection_normalization_or_semantic_gate_invalid"
    )
    assert projected["integrity"][
        "normalization_receipt_exact_pin_verified"
    ] is True
    assert projected["integrity"][
        "normalization_receipt_self_hash_verified"
    ] is True
    assert projected["integrity"][
        "normalization_receipt_schema_replay_verified"
    ] is False
    assert projected["integrity"]["normalization_semantics_verified"] is False
    assert projected["normalization_group_split"]["training_eligible"] is False


def test_missing_semantic_r4_receipt_fails_card_closed(
    five_class_workspace: Path,
) -> None:
    relative = admin_validation.PPE_FIVE_CLASS_SEMANTIC_R4_PINS["receipt"][
        "path"
    ]
    (five_class_workspace / relative).unlink()

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "semantic_r4_receipt_missing"
    assert projected["integrity"]["semantic_r4_receipt_exact_pin_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_semantic_r4_overclaim_is_rejected(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.PPE_FIVE_CLASS_SEMANTIC_R4_PINS)
    relative = pins["receipt"]["path"]
    path = five_class_workspace / relative
    path.chmod(0o640)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["readiness"]["semantic_mapping_approved"] = True
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(receipt)
    raw = _write_json(path, receipt)
    pins["receipt"] = {
        **_pin(relative, raw),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    monkeypatch.setattr(
        admin_validation,
        "PPE_FIVE_CLASS_SEMANTIC_R4_PINS",
        pins,
    )

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "five_class_projection_normalization_or_semantic_gate_invalid"
    )
    assert projected["integrity"]["semantic_r4_receipt_exact_pin_verified"] is True
    assert projected["integrity"]["semantic_r4_receipt_self_hash_verified"] is True
    assert projected["integrity"]["semantic_r4_semantics_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_launch_gate_training_overclaim_is_rejected(
    five_class_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(
        admin_validation.PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS
    )
    relative = pins["gate"]["path"]
    path = five_class_workspace / relative
    path.chmod(0o640)
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate["launch_policy"]["full_train_150e"] = {
        "allowed": True,
        "scope": "container_preparation_only_no_dataset_or_model_execution",
    }
    gate.pop("fingerprint_sha256")
    gate["fingerprint_sha256"] = admin_validation._canonical_sha256(gate)
    raw = _write_json(path, gate)
    pins["gate"] = {
        **_pin(relative, raw),
        "fingerprint_sha256": gate["fingerprint_sha256"],
    }
    monkeypatch.setattr(
        admin_validation,
        "PPE_YOLO11S_SEMANTIC_LAUNCH_GATE_R3_PINS",
        pins,
    )

    projected = _project(five_class_workspace)

    assert projected["available"] is False
    assert projected["reason"] == (
        "five_class_projection_normalization_or_semantic_gate_invalid"
    )
    assert projected["integrity"]["semantic_launch_gate_r3_exact_pin_verified"] is True
    assert projected["integrity"]["semantic_launch_gate_r3_semantics_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_compact_receipt_schema_and_standalone_verifier() -> None:
    receipt_path = (
        ROOT
        / admin_validation.PPE_FIVE_CLASS_ADMIN_PINS["projection_receipt"]["path"]
    )
    schema_path = ROOT / admin_validation.PPE_FIVE_CLASS_ADMIN_PINS["schema"][
        "path"
    ]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(receipt, schema)

    verified = projection_validator.verify_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert verified["valid"] is True
    assert verified["structural_pass"] is True
    assert verified["accepted_to_quarantine"] is True
    assert verified["training_eligible"] is False
    assert verified["production_ready"] is False


def test_standalone_verifier_rejects_resealed_overclaim() -> None:
    receipt_path = (
        ROOT
        / admin_validation.PPE_FIVE_CLASS_ADMIN_PINS["projection_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["gates"]["training_eligible"] = True
    receipt.pop("receipt_sha256")
    receipt = projection_validator.seal_receipt(receipt)

    with pytest.raises(
        projection_validator.ProjectionError,
        match="overclaims",
    ):
        projection_validator.verify_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_separate_card_does_not_change_existing_ppe_seed_history() -> None:
    existing = admin_validation.ArtifactReader(
        ROOT / "validation/results",
        workspace_root=ROOT,
        schema_root=ROOT / "validation/schemas",
    )
    historical = admin_validation._ppe_seed_readiness(existing)
    separate = admin_validation._ppe_five_class_readiness(existing)

    assert historical["state"] == "acquired_quarantine_failed"
    assert historical["receipts"]["quarantine"]["accepted"] is False
    assert separate["state"] == "dry_run_group_split_complete_training_blocked"
    assert separate["quarantine"]["accepted"] is True
    assert separate["quarantine"]["training_eligible"] is False


def test_validation_api_and_ui_expose_separate_compact_card(
    five_class_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(five_class_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["ppe_five_class_quarantine"]
    assert projected["available"] is True
    assert projected["quarantine"]["images"] == 2586
    assert projected["quarantine"]["bounding_boxes"] == 17827
    assert projected["normalization_group_split"]["final_group_count"] == 292
    assert projected["normalization_group_split"]["roles"]["train"][
        "images"
    ] == 2068
    assert projected["normalization_group_split"]["leakage"]["zero"] is True
    assert projected["quarantine_history"]["r1_superseded"] is True
    assert projected["semantic_audit_r4"]["sample_images"] == 20
    assert projected["semantic_audit_r4"]["source_groups"] == 18
    assert projected["semantic_audit_r4"]["bbox_rows_checked"] == 488
    assert projected["semantic_audit_r4"][
        "development_holdout_payload_files_opened"
    ] == 0
    assert projected["semantic_launch_gate_r3"]["image_build_preparation_allowed"] is True
    assert "full_train_150e" in projected["semantic_launch_gate_r3"]["blocked_modes"]
    assert projected["semantic_launch_gate_r3"]["training_ready"] is False
    assert projected["gates"]["production_ready"] is False
    assert page.status_code == 200
    assert "PPE 5-Class R2 normalizasyon + karantina" in page.text
    assert "Büyük R2 receipt" in page.text
    assert "Exact duplicate grupları" in page.text
    assert "R2 grup-safe dry-run" in page.text
    assert "R4 semantik audit" in page.text
    assert "R3 semantik launch gate" in page.text
    assert "Yeni eğitim yetkisi" in page.text
    assert "Karantina tarihçesi" in page.text


def test_docs_define_non_parsing_admin_boundary_and_separate_history() -> None:
    dashboard = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )
    dataset = (ROOT / "docs/ppe-mendeley-five-class-quarantine.md").read_text(
        encoding="utf-8"
    )

    assert "PPE 5-Class R2 quarantine card is independent" in dashboard
    assert "`collect=false`" in dashboard
    assert "does not re-read the 208 MB archive" in dashboard
    assert "Admin paneli 5.456.369 byte authoritative receipt'i JSON olarak yüklemez" in dataset
    assert "mevcut `ppe_seed_readiness` tarihçesini değiştirmez" in dataset
