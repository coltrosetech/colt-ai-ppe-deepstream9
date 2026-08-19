from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import stat
import subprocess
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_lo_cpped_source_quarantine as quarantine


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = quarantine.MANIFEST
RECEIPT = quarantine.DEFAULT_RECEIPT
HISTORICAL_R1_RECEIPT = (
    ROOT
    / "validation/results/ppe/source-quarantine/"
    "lo-cpped-metadata-acquisition-r1.json"
)
SCHEMA = quarantine.SCHEMA
PYTHON = ROOT / ".venv/bin/python"
MANIFEST_BYTES = 17332
MANIFEST_SHA256 = "abdfafcae701f32f502175f14e676db3e605d7646894b85be1131bbe0c4d34c9"
RECEIPT_FILE_SHA256 = "1bff485dba9261607236d7ca4f6ba8114d4059302025e9a07a6fbf1c2c0dd1c1"
RECEIPT_SELF_SHA256 = "93760142da146e139d2732f6598a7e804b6c43c3ce2b86739468c4e351009e5d"
HISTORICAL_R1_FILE_SHA256 = "a8fb15c5839b3f1ddbf585dcdb309357594575efa9323a8683b60b8eceb50308"
HISTORICAL_R1_SELF_SHA256 = "6b91c7e6e2b8f5b17944613bcc5d7d2c8f836516f96097a734c2f25ccc8dc48e"


def _manifest() -> dict:
    value, _ = quarantine._read_json(MANIFEST)
    return value


def _receipt() -> dict:
    value, _ = quarantine._read_json(RECEIPT)
    return value


def _source(value: dict, source_id: str) -> dict:
    return next(item for item in value["sources"] if item["id"] == source_id)


def test_checked_manifest_is_exact_and_all_authorizations_are_blocked() -> None:
    raw = MANIFEST.read_bytes()
    assert stat.S_IMODE(MANIFEST.stat().st_mode) == 0o440
    assert len(raw) == MANIFEST_BYTES
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256
    report = quarantine.validate_checked_manifest()
    assert report["valid"] is True
    assert report["source_ids"] == list(quarantine.SOURCE_IDS)
    assert report["dataset_artifact_count"] == 0
    assert report["blocker_count"] == 18
    assert report["all_eligibility_gates_blocked"] is True
    assert report["dataset_bytes_streamed_or_persisted"] is False
    assert report["annotations_or_final_test_labels_opened"] is False
    assert report["training_inference_or_gpu_used"] is False


def test_lo_declared_counts_and_current_drive_access_are_preserved() -> None:
    lo = _source(_manifest(), quarantine.SOURCE_IDS[0])
    declared = lo["declared_dataset"]
    assert declared["image_count"] == 11000
    assert declared["annotation_count"] == 88725
    assert declared["class_counts"] == quarantine.LO_CLASS_COUNTS
    assert sum(declared["class_counts"].values()) == declared["annotation_count"]
    drive = lo["official_dataset_asset"]
    assert drive["resource_id"] == "1SBlAHoviLHT8uCF0Lhv0ED_Yrdbfwzpq"
    assert drive["anonymous_access_result"] == "authentication_required"
    assert drive["file_listing_observed"] is False
    assert drive["file_name"] is None
    assert drive["bytes"] is None
    assert drive["sha256"] is None
    assert drive["dataset_archive_downloaded"] is False


def test_lo_article_license_is_not_promoted_to_dataset_rights() -> None:
    lo = _source(_manifest(), quarantine.SOURCE_IDS[0])
    rights = lo["rights"]
    assert rights["article_license"] == "CC-BY-4.0"
    assert rights["article_license_scope_proves_dataset_package_rights"] is False
    assert rights["dataset_specific_license_status"] == "missing_or_unknown"
    assert rights["commercial_derivative_training_allowed"] is False
    assert rights["embedded_web_media_provenance_cleared"] is False
    assert rights["camera_media_provenance_cleared"] is False
    assert rights["person_release_confirmed"] is False
    assert rights["location_release_confirmed"] is False


def test_lo_target_names_do_not_bypass_person_association_or_grouping() -> None:
    lo = _source(_manifest(), quarantine.SOURCE_IDS[0])
    grouping = lo["grouping_and_semantics"]
    assert grouping["target_label_mapping"] == {
        "helmet": "hard_hat",
        "no_helmet": "no_hard_hat",
        "hi_vis": "high_visibility_vest",
        "no_hi_vis": "no_high_visibility_vest",
    }
    assert grouping["target_label_names_match"] is True
    assert grouping["person_class_present"] is False
    assert grouping["person_to_equipment_association_verified"] is False
    assert grouping["source_asset_group_safe_split_possible_from_observed_metadata"] is False
    assert grouping["independent_test_identity_observed"] is False


def test_cpped_official_repository_snapshot_is_metadata_not_dataset() -> None:
    cpped = _source(_manifest(), quarantine.SOURCE_IDS[1])
    repository = cpped["official_repository_snapshot"]
    assert repository["commit"] == quarantine.CPPED_COMMIT
    assert repository["tree"] == quarantine.CPPED_TREE
    assert repository["commit_signature_verified"] is False
    assert repository["github_api_license_field"] is None
    assert repository["archive"]["bytes"] == quarantine.CPPED_REPO_ARCHIVE_BYTES
    assert repository["archive"]["sha256"] == quarantine.CPPED_REPO_ARCHIVE_SHA256
    assert repository["archive"]["persisted"] is False
    assert repository["file_count"] == len(repository["file_ledger"]) == 17
    assert repository["file_bytes"] == sum(item["bytes"] for item in repository["file_ledger"]) == 754101
    assert repository["opaque_spreadsheet_contents_interpreted"] is False
    assert repository["dataset_images_or_labels_present_in_repository_tree"] is False


def test_cpped_drive_rights_and_target_mapping_remain_blocked() -> None:
    cpped = _source(_manifest(), quarantine.SOURCE_IDS[1])
    drive = cpped["official_dataset_asset"]
    assert drive["resource_id"] == "1pd2_kzUA82M25s8HVL6rHCUUWCaTdVnv"
    assert drive["anonymous_access_result"] == "authentication_and_justified_access_request_required"
    assert drive["file_listing_observed"] is False
    assert drive["file_names"] == []
    assert drive["total_bytes"] is None
    assert drive["archive_sha256"] is None
    rights = cpped["rights"]
    assert rights["dataset_license_statement"] == "CC-BY-NC-4.0"
    assert rights["dataset_access_request_required"] is True
    assert rights["access_request_submitted_or_granted"] is False
    assert rights["commercial_derivative_training_allowed"] is False
    assert rights["ferdous_source_media_rights_cleared"] is False
    assert rights["internet_source_media_rights_cleared"] is False
    grouping = cpped["grouping_and_semantics"]
    assert grouping["target_label_mapping"]["no_helmet"] is None
    assert grouping["target_label_mapping"]["no_hi_vis"] is None
    assert grouping["head_class_assumed_to_mean_no_helmet"] is False
    assert grouping["complete_project_target_mapping_verified"] is False


def test_cpped_declared_class_sum_is_exact_but_does_not_unlock_training() -> None:
    cpped = _source(_manifest(), quarantine.SOURCE_IDS[1])
    declared = cpped["declared_dataset"]
    assert declared["image_count"] == 2612
    assert declared["category_count"] == 13
    assert declared["class_counts"] == quarantine.CPPED_CLASS_COUNTS
    assert sum(declared["class_counts"].values()) == declared["annotation_count"] == 20172
    assert set(cpped["eligibility"].values()) == {False}


def test_every_source_eligibility_and_aggregate_count_is_closed() -> None:
    value = _manifest()
    for source in value["sources"]:
        assert tuple(source["eligibility"]) == quarantine.ELIGIBILITY_KEYS
        assert all(result is False for result in source["eligibility"].values())
        assert source["blockers"]
    decision = value["decision"]
    assert decision["dataset_artifact_count"] == 0
    assert decision["download_authorized_source_count"] == 0
    assert decision["training_eligible_source_count"] == 0
    assert decision["commercial_product_training_eligible_source_count"] == 0
    assert decision["independent_validation_eligible_source_count"] == 0
    assert decision["final_test_eligible_source_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["scope"].__setitem__("annotations_opened", True),
            "scope overclaim",
        ),
        (
            lambda value: value["policy"].__setitem__("public_link_is_training_permission", True),
            "policy overclaim",
        ),
        (
            lambda value: _source(value, quarantine.SOURCE_IDS[0])["rights"].__setitem__(
                "dataset_specific_license_status", "verified_permissive"
            ),
            "Lo dataset license overclaim",
        ),
        (
            lambda value: _source(value, quarantine.SOURCE_IDS[0])["grouping_and_semantics"].__setitem__(
                "source_asset_group_safe_split_possible_from_observed_metadata", True
            ),
            "Lo grouping/semantics overclaim",
        ),
        (
            lambda value: _source(value, quarantine.SOURCE_IDS[1])["rights"].__setitem__(
                "commercial_derivative_training_allowed", True
            ),
            "CPPED rights overclaim",
        ),
        (
            lambda value: _source(value, quarantine.SOURCE_IDS[1])["grouping_and_semantics"].__setitem__(
                "head_class_assumed_to_mean_no_helmet", True
            ),
            "CPPED grouping/semantics overclaim",
        ),
        (
            lambda value: _source(value, quarantine.SOURCE_IDS[1])["eligibility"].__setitem__(
                "training_eligible", True
            ),
            "eligibility overclaims",
        ),
    ],
)
def test_manifest_overclaims_fail_closed(mutation, message: str) -> None:
    value = copy.deepcopy(_manifest())
    mutation(value)
    with pytest.raises(quarantine.SourceQuarantineError, match=message):
        quarantine.validate_manifest(value)


def test_dataset_artifact_identity_cannot_be_injected_without_acquisition() -> None:
    value = copy.deepcopy(_manifest())
    lo = _source(value, quarantine.SOURCE_IDS[0])["official_dataset_asset"]
    lo["file_name"] = "unverified.zip"
    lo["bytes"] = 123
    lo["sha256"] = "0" * 64
    lo["artifact_identity_pinned"] = True
    with pytest.raises(quarantine.SourceQuarantineError):
        quarantine.validate_manifest(value)


def test_repository_ledger_duplicate_path_and_bad_hash_fail_closed() -> None:
    value = copy.deepcopy(_manifest())
    ledger = _source(value, quarantine.SOURCE_IDS[1])["official_repository_snapshot"]["file_ledger"]
    ledger[1]["path"] = ledger[0]["path"]
    ledger[2]["sha256"] = "not-a-hash"
    with pytest.raises(quarantine.SourceQuarantineError):
        quarantine.validate_manifest(value)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(quarantine.SourceQuarantineError, match="duplicate JSON key"):
        quarantine._strict_json_bytes(b'{"a":1,"a":2}', source="fixture")
    with pytest.raises(quarantine.SourceQuarantineError, match="non-finite"):
        quarantine._strict_json_bytes(b'{"a":NaN}', source="fixture")


def test_strict_file_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"a":1}', encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target.name)
    with pytest.raises(quarantine.SourceQuarantineError, match="regular file"):
        quarantine._read_regular_single_link(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(quarantine.SourceQuarantineError, match="multiple hard links"):
        quarantine._read_regular_single_link(target)


def test_checked_receipt_is_exact_schema_valid_and_replayable() -> None:
    raw = RECEIPT.read_bytes()
    assert stat.S_IMODE(RECEIPT.stat().st_mode) == 0o440
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_FILE_SHA256
    value = _receipt()
    assert value["receipt_sha256"] == RECEIPT_SELF_SHA256
    assert quarantine.receipt_sha256(value) == RECEIPT_SELF_SHA256
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)
    replay = quarantine.verify_receipt(
        value,
        expected_receipt_sha256=RECEIPT_SELF_SHA256,
    )
    assert replay["valid"] is True
    assert replay["dataset_acquisition_authorized"] is False
    assert replay["training_eligible"] is False
    assert replay["independent_validation_eligible"] is False
    assert replay["production_ready"] is False


def test_historical_r1_receipt_is_immutable_and_not_authoritative() -> None:
    raw = HISTORICAL_R1_RECEIPT.read_bytes()
    assert stat.S_IMODE(HISTORICAL_R1_RECEIPT.stat().st_mode) == 0o440
    assert hashlib.sha256(raw).hexdigest() == HISTORICAL_R1_FILE_SHA256
    value, _ = quarantine._read_json(HISTORICAL_R1_RECEIPT)
    assert value["receipt_sha256"] == HISTORICAL_R1_SELF_SHA256
    assert quarantine.receipt_sha256(value) == HISTORICAL_R1_SELF_SHA256
    with pytest.raises(
        quarantine.SourceQuarantineError,
        match="input pin (byte count|hash) differs",
    ):
        quarantine.verify_receipt(
            value,
            expected_receipt_sha256=HISTORICAL_R1_SELF_SHA256,
        )


def test_receipt_external_pin_and_resealed_semantic_tamper_fail() -> None:
    value = _receipt()
    with pytest.raises(quarantine.SourceQuarantineError, match="external receipt pin"):
        quarantine.verify_receipt(value, expected_receipt_sha256="0" * 64)
    tampered = copy.deepcopy(value)
    tampered["gates"]["training_eligible"] = True
    tampered["receipt_sha256"] = quarantine.receipt_sha256(tampered)
    with pytest.raises(quarantine.SourceQuarantineError, match="receipt replay differs"):
        quarantine.verify_receipt(
            tampered,
            expected_receipt_sha256=tampered["receipt_sha256"],
        )


def test_schema_rejects_authorization_or_execution_overclaim() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for path in (
        ("gates", "dataset_acquisition_authorized"),
        ("gates", "training_eligible"),
        ("gates", "production_ready"),
        ("execution_boundary", "gpu_used"),
        ("execution_boundary", "declared_final_test_labels_opened"),
    ):
        value = copy.deepcopy(_receipt())
        value[path[0]][path[1]] = True
        assert list(validator.iter_errors(value)), path


def test_receipt_build_and_verify_are_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network use is forbidden during replay")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    value = quarantine.build_receipt(created_at=_receipt()["created_at"])
    assert value == _receipt()
    assert value["execution_boundary"]["network_replayed_during_receipt_build_or_verify"] is False


def test_atomic_writer_is_no_overwrite_and_mode_0440(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    value = quarantine.build_receipt(created_at="2026-07-18T01:20:00Z")
    quarantine.atomic_write_no_overwrite(output, value)
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    with pytest.raises(quarantine.SourceQuarantineError, match="refusing to overwrite"):
        quarantine.atomic_write_no_overwrite(output, value)


def test_cli_verify_reports_closed_gates_without_network() -> None:
    completed = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "validation.ppe_lo_cpped_source_quarantine",
            "verify",
            "--receipt",
            str(RECEIPT),
            "--expected-receipt-sha256",
            RECEIPT_SELF_SHA256,
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["valid"] is True
    assert result["dataset_acquisition_authorized"] is False
    assert result["training_eligible"] is False
    assert result["production_ready"] is False


def test_validator_has_no_downloader_trainer_gpu_or_container_entry_point() -> None:
    text = Path(quarantine.__file__).read_text(encoding="utf-8")
    forbidden_tokens = (
        "urllib.request",
        "requests.",
        "urlopen(",
        "torch.",
        "tensorflow",
        "cv2.",
        "nvidia-smi",
        "docker ",
        "subprocess.",
    )
    assert not [token for token in forbidden_tokens if token in text]
