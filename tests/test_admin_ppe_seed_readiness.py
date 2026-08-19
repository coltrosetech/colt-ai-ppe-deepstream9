from __future__ import annotations

import hashlib
import json
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


@pytest.fixture
def ppe_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    relative_paths = [
        admin_validation.PPE_SEED_MANIFEST_PIN["path"],
        *(
            pin["path"]
            for pin in admin_validation.PPE_SEED_SCHEMA_PINS.values()
        ),
        *(
            pin["path"]
            for pin in admin_validation.PPE_SEED_RECEIPT_PINS.values()
        ),
        admin_validation.PPE_PROVENANCE_PLAN_PIN["path"],
        admin_validation.PPE_PROVENANCE_RECEIPT_PIN["path"],
        admin_validation.PPE_PROVENANCE_CODE_PIN["path"],
        *(
            pin["path"]
            for pin in admin_validation.PPE_PROVENANCE_SCHEMA_PINS.values()
        ),
        *(
            pin["path"]
            for pin in admin_validation.PPE_NORMALIZATION_SUPERSEDED_R1_PINS.values()
        ),
        admin_validation.PPE_NORMALIZATION_PLAN_PIN["path"],
        admin_validation.PPE_NORMALIZATION_ASSESSMENT_PIN["path"],
        *(
            pin["path"]
            for pin in admin_validation.PPE_NORMALIZATION_SCHEMA_PINS.values()
        ),
    ]
    for relative in relative_paths:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        destination.chmod(0o600)
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    return admin_validation._ppe_seed_readiness(reader)


def test_checked_in_ppe_seed_contract_matches_admin_trust_anchor() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["state"] == "acquired_quarantine_failed"
    assert all(projected["integrity"].values())
    assert projected["source_contract"]["source_count"] == 2
    assert projected["source_contract"]["fully_pinned_asset_count"] == 2
    assert projected["source_contract"]["fully_pinned_archive_count"] == 1
    assert projected["receipts"]["acquisition"]["accepted"] is True
    assert projected["receipts"]["quarantine"]["accepted"] is False
    assert projected["provenance_review"] == {
        "evidence_verified": True,
        "mechanical_audit_complete": True,
        "images": 2286,
        "train_images": 1829,
        "validation_images": 457,
        "exact_content_duplicate_groups": 0,
        "exact_original_key_duplicate_groups": 0,
        "filename_families": 6,
        "cross_split_filename_families": 6,
        "images_in_cross_split_filename_families": 2286,
        "strict_near_duplicate_candidate_pairs": 7,
        "strict_near_duplicate_validation_members": 4,
        "high_confidence_near_duplicate_candidate_pairs": 2,
        "item_level_source_mapping_complete": False,
        "embedded_rights_review_complete": False,
        "camera_site_session_metadata_present": False,
        "human_near_duplicate_review_complete": False,
        "training_eligible": False,
    }
    assert projected["quarantine_review"] == {
        "archive_entries": 4583,
        "decoded_images": 2286,
        "paired_images": 2286,
        "label_rows": 6038,
        "declared_image_width": 640,
        "declared_image_height": 640,
        "exact_declared_dimension_images": 472,
        "dimension_mismatch_images": 1814,
        "distinct_observed_dimensions": 905,
        "failed_structural_gate_count": 5,
        "failed_gate_ids": [
            "declared_image_dimensions_match",
            "valid_yolo_yaml",
            "yolo_split_paths_resolve_in_archive",
            "declared_classes_match_yaml",
            "valid_yolo_detection_labels",
        ],
        "absolute_windows_yaml_path_rejected": True,
        "independent_bbox_out_of_range_count": 52,
    }
    assert projected["normalization"] == {
        "evidence_verified": True,
        "provenance_review_evidence_present": True,
        "provenance_mechanical_audit_replayed": True,
        "provenance_review_approved": False,
        "embedded_rights_review_approved": False,
        "camera_group_split_approved": False,
        "normalization_ready": False,
        "source_training_eligible": False,
        "normalized_training_eligible": False,
        "independent_bbox_out_of_range_count": 52,
        "bbox_overflow_severity": {
            "classification": "small_but_blocking",
            "minimum": 0.0000031249999998816946,
            "median": 0.000007812500000037303,
            "p95": 0.000020833333333358794,
            "maximum": 0.00003597122302156919,
        },
    }


def test_projection_is_pathless_and_every_execution_gate_stays_closed(
    ppe_workspace: Path,
) -> None:
    projected = _project(ppe_workspace)

    assert projected["available"] is True
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["preparation"] == {
        "source_manifest_verified": True,
        "receipt_contracts_verified": True,
        "data_acquired": True,
        "quarantine_complete": False,
    }
    assert projected["receipts"] == {
        "acquisition": {
            "pin_declared": True,
            "verified": True,
            "accepted": True,
        },
        "quarantine": {
            "pin_declared": True,
            "verified": True,
            "accepted": False,
        },
    }
    assert projected["gates"]["source_contract_verified"] is True
    assert projected["gates"]["acquired"] is True
    assert all(
        value is False
        for key, value in projected["gates"].items()
        if key not in {"source_contract_verified", "acquired"}
    )

    serialized = json.dumps(projected, ensure_ascii=False)
    for private_fragment in (
        "/workspace",
        "data/manifests/",
        "ppe_dataset/",
        "data/raw/",
        "validation/results/",
        "https://",
        "10.17632/",
        "82cc91fb",
        "7a22e5cb",
        "file_downloaded",
    ):
        assert private_fragment not in serialized


def test_unpinned_archive_and_alternate_receipts_cannot_change_readiness(
    ppe_workspace: Path,
) -> None:
    raw_archive = (
        ppe_workspace
        / "data/raw/ppe/mendeley-ppe-v6/20250731-PPE2286y.zip"
    )
    raw_archive.parent.mkdir(parents=True, exist_ok=True)
    raw_archive.write_bytes(b"not-the-pinned-archive")
    receipt_root = ppe_workspace / "validation/results/ppe/seeds"
    receipt_root.mkdir(parents=True, exist_ok=True)
    (receipt_root / "forged-acquisition.json").write_text(
        json.dumps({"accepted": True, "training_eligible": False}),
        encoding="utf-8",
    )
    (receipt_root / "mendeley-ppe-v6-quarantine.json").write_text(
        json.dumps(
            {"accepted_to_quarantine": True, "training_eligible": False}
        ),
        encoding="utf-8",
    )

    projected = _project(ppe_workspace)

    assert projected["available"] is True
    assert projected["state"] == "acquired_quarantine_failed"
    assert projected["gates"]["acquired"] is True
    assert projected["gates"]["quarantined"] is False
    assert projected["receipts"]["acquisition"]["pin_declared"] is True
    assert projected["receipts"]["acquisition"]["accepted"] is True
    assert projected["receipts"]["quarantine"]["pin_declared"] is True
    assert projected["receipts"]["quarantine"]["accepted"] is False


def test_manifest_hash_drift_fails_the_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    manifest = ppe_workspace / admin_validation.PPE_SEED_MANIFEST_PIN["path"]
    manifest.write_bytes(manifest.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["preparation"]["source_manifest_verified"] is False
    assert all(value is False for value in projected["gates"].values())
    assert "pin_mismatch" in projected["reason"]


def test_manifest_symlink_is_rejected_even_when_target_bytes_match(
    ppe_workspace: Path,
) -> None:
    manifest = ppe_workspace / admin_validation.PPE_SEED_MANIFEST_PIN["path"]
    target = ppe_workspace / "same-ppe-manifest-at-another-name.json"
    target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(target)

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["integrity"]["source_manifest_fingerprint_verified"] is False
    assert all(value is False for value in projected["gates"].values())
    assert "unsafe_path" in projected["reason"]


def test_manifest_name_swap_during_descriptor_read_is_rejected(
    ppe_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ppe_workspace / admin_validation.PPE_SEED_MANIFEST_PIN["path"]
    original = manifest.read_bytes()
    displaced = ppe_workspace / "displaced-ppe-manifest.json"
    replacement = ppe_workspace / "replacement-ppe-manifest.json"
    replacement.write_bytes(original)
    original_read = admin_validation.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if not swapped and chunk and len(original) == len(chunk):
            swapped = True
            manifest.rename(displaced)
            replacement.rename(manifest)
        return chunk

    monkeypatch.setattr(admin_validation.os, "read", swapping_read)

    projected = _project(ppe_workspace)

    assert swapped is True
    assert projected["available"] is False
    assert projected["reason"] in {
        "source_manifest_changed",
        "source_manifest_pin_mismatch",
    }
    assert all(value is False for value in projected["gates"].values())


def test_receipt_schema_hash_drift_fails_every_gate_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_SEED_SCHEMA_PINS[
        "acquisition_receipt"
    ]["path"]
    schema = ppe_workspace / relative
    schema.write_bytes(schema.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert (
        projected["integrity"]["acquisition_receipt_schema_pin_verified"]
        is False
    )
    assert projected["preparation"]["receipt_contracts_verified"] is False
    assert all(value is False for value in projected["gates"].values())


@pytest.mark.parametrize("receipt_key", ["acquisition", "quarantine"])
def test_receipt_file_hash_drift_fails_every_gate_closed(
    ppe_workspace: Path,
    receipt_key: str,
) -> None:
    relative = admin_validation.PPE_SEED_RECEIPT_PINS[receipt_key]["path"]
    receipt = ppe_workspace / relative
    receipt.write_bytes(receipt.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert (
        projected["integrity"][f"{receipt_key}_receipt_file_pin_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_forged_embedded_receipt_hash_cannot_replace_external_trust_anchor(
    ppe_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = admin_validation.PPE_SEED_RECEIPT_PINS["acquisition"]["path"]
    receipt_path = ppe_workspace / relative
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["accepted"] = False
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(unsigned)
    raw = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)

    pins = {
        key: dict(value)
        for key, value in admin_validation.PPE_SEED_RECEIPT_PINS.items()
    }
    trusted_receipt_hash = pins["acquisition"]["receipt_sha256"]
    pins["acquisition"].update(_pin(relative, raw))
    assert pins["acquisition"]["receipt_sha256"] == trusted_receipt_hash
    monkeypatch.setattr(admin_validation, "PPE_SEED_RECEIPT_PINS", pins)

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "ppe_seed_receipt_contract_invalid"
    assert (
        projected["integrity"]["acquisition_receipt_file_pin_verified"]
        is True
    )
    assert (
        projected["integrity"]["acquisition_receipt_self_hash_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_missing_normalization_assessment_fails_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_NORMALIZATION_ASSESSMENT_PIN["path"]
    (ppe_workspace / relative).unlink()

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "normalization_assessment_missing"
    assert projected["normalization"]["evidence_verified"] is False
    assert projected["normalization"]["normalization_ready"] is False
    assert all(value is False for value in projected["gates"].values())


def test_missing_provenance_receipt_fails_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_PROVENANCE_RECEIPT_PIN["path"]
    (ppe_workspace / relative).unlink()

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "provenance_receipt_missing"
    assert projected["provenance_review"]["evidence_verified"] is False
    assert (
        projected["integrity"]["provenance_receipt_file_pin_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_tampered_provenance_receipt_fails_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_PROVENANCE_RECEIPT_PIN["path"]
    receipt = ppe_workspace / relative
    receipt.write_bytes(receipt.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "provenance_receipt_pin_mismatch"
    assert (
        projected["integrity"]["provenance_receipt_file_pin_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_exact_repinned_provenance_safety_overclaim_is_rejected(
    ppe_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = admin_validation.PPE_PROVENANCE_RECEIPT_PIN["path"]
    receipt_path = ppe_workspace / relative
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["observations"]["strict_cross_split_candidate_pairs"] = 0
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt_sha256 = admin_validation._canonical_sha256(unsigned)
    receipt["receipt_sha256"] = receipt_sha256
    raw = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(raw)
    pin = _pin(relative, raw)
    pin["receipt_sha256"] = receipt_sha256
    monkeypatch.setattr(
        admin_validation,
        "PPE_PROVENANCE_RECEIPT_PIN",
        pin,
    )

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "ppe_provenance_semantic_lineage_invalid"
    assert projected["integrity"]["provenance_receipt_self_hash_verified"]
    assert projected["integrity"]["provenance_semantic_lineage_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_tampered_normalization_assessment_fails_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_NORMALIZATION_ASSESSMENT_PIN["path"]
    assessment = ppe_workspace / relative
    assessment.write_bytes(assessment.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "normalization_assessment_pin_mismatch"
    assert (
        projected["integrity"][
            "normalization_assessment_file_pin_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_schema_invalid_exact_repinned_normalization_assessment_is_rejected(
    ppe_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = admin_validation.PPE_NORMALIZATION_ASSESSMENT_PIN["path"]
    assessment_path = ppe_workspace / relative
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    assessment["normalization_ready"] = True
    unsigned = dict(assessment)
    unsigned.pop("receipt_sha256")
    receipt_sha256 = admin_validation._canonical_sha256(unsigned)
    assessment["receipt_sha256"] = receipt_sha256
    raw = (
        json.dumps(assessment, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    assessment_path.write_bytes(raw)
    pin = _pin(relative, raw)
    pin["receipt_sha256"] = receipt_sha256
    monkeypatch.setattr(
        admin_validation,
        "PPE_NORMALIZATION_ASSESSMENT_PIN",
        pin,
    )

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "ppe_normalization_schema_contract_invalid"
    assert (
        projected["integrity"][
            "normalization_assessment_file_pin_verified"
        ]
        is True
    )
    assert (
        projected["integrity"][
            "normalization_assessment_schema_replay_verified"
        ]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_normalization_plan_or_schema_drift_fails_whole_card_closed(
    ppe_workspace: Path,
) -> None:
    relative = admin_validation.PPE_NORMALIZATION_SCHEMA_PINS["plan"][
        "path"
    ]
    schema = ppe_workspace / relative
    schema.write_bytes(schema.read_bytes() + b" ")

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "normalization_plan_schema_pin_mismatch"
    assert (
        projected["integrity"]["normalization_plan_schema_pin_verified"]
        is False
    )
    assert all(value is False for value in projected["gates"].values())


def test_manifest_hardlink_is_rejected_even_when_bytes_match(
    ppe_workspace: Path,
) -> None:
    manifest = ppe_workspace / admin_validation.PPE_SEED_MANIFEST_PIN["path"]
    second_name = ppe_workspace / "second-ppe-manifest-link.json"
    second_name.hardlink_to(manifest)

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["integrity"]["source_manifest_fingerprint_verified"] is False
    assert projected["reason"] == "source_manifest_unsafe_path"
    assert all(value is False for value in projected["gates"].values())


def test_exact_repin_of_schema_invalid_manifest_still_fails_semantic_replay(
    ppe_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = admin_validation.PPE_SEED_MANIFEST_PIN["path"]
    manifest_path = ppe_workspace / relative
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["eligibility"]["training"] = True
    raw = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(raw)
    monkeypatch.setattr(
        admin_validation,
        "PPE_SEED_MANIFEST_PIN",
        _pin(relative, raw),
    )

    projected = _project(ppe_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "ppe_seed_schema_contract_invalid"
    assert projected["integrity"]["source_manifest_fingerprint_verified"] is True
    assert projected["integrity"]["source_manifest_schema_replay_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_validation_api_and_ui_expose_compact_ppe_seed_card(
    ppe_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(ppe_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["ppe_seed_readiness"]
    assert projected["state"] == "acquired_quarantine_failed"
    assert projected["gates"]["acquired"] is True
    assert projected["gates"]["quarantined"] is False
    assert projected["gates"]["production_ready"] is False
    assert projected["normalization"]["evidence_verified"] is True
    assert projected["provenance_review"]["evidence_verified"] is True
    assert projected["provenance_review"][
        "cross_split_filename_families"
    ] == 6
    assert projected["provenance_review"][
        "strict_near_duplicate_candidate_pairs"
    ] == 7
    assert projected["normalization"][
        "provenance_mechanical_audit_replayed"
    ] is True
    assert projected["normalization"]["normalization_ready"] is False
    assert (
        projected["normalization"][
            "independent_bbox_out_of_range_count"
        ]
        == 52
    )
    assert page.status_code == 200
    assert "PPE veri tohumu hazırlığı" in page.text
    assert "edinildi; karantina reddedildi" in page.text
    assert "doğrulandı; edinim kabul edildi" in page.text
    assert "doğrulandı; karantina reddedildi" in page.text
    assert "640×640 boyut denetimi" in page.text
    assert "Normalizasyon assessment’i" in page.text
    assert "Provenance/split R2" in page.text
    assert "Yakın-kopya incelemesi" in page.text
    assert "R2 provenance → normalizasyon" in page.text
    assert "BBox taşma şiddeti" in page.text
    assert "küçük ama bloke edici" in page.text
    assert "Eğitim/export/DS9/kabul" in page.text


def test_compose_and_docs_keep_ppe_schema_trust_anchor_read_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docs = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )

    assert "./ppe_dataset:/workspace/ppe_dataset:ro" in compose
    assert "./ppe_dataset/schemas:/workspace/ppe_dataset/schemas:ro" not in compose
    assert "PPE seed-readiness card uses a separate immutable trust boundary" in docs
    assert "does not scan `data/raw`" in docs
    assert "`acquired=true`" in docs
    assert "`quarantined=false`" in docs
    assert "authoritative R2 quarantine receipt" in docs
    assert "52 `bbox_out_of_range` rows" in docs
    assert "`normalization_ready=false`" in docs
