from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ppe_dataset.acquisition import SeedContractError, write_receipt_no_overwrite
from ppe_dataset.five_class_normalization_r2 import (
    FiveClassNormalizationError,
    _verify_internal_receipt_semantics,
    build_dry_run,
    capture_key,
    verify_receipt_file,
    verify_receipt_hash,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "data/manifests/"
    "ppe-mendeley-five-class-v1-normalization-group-split-r2.plan.json"
)
PLAN_SCHEMA = (
    ROOT
    / "ppe_dataset/schemas/"
    "ppe-five-class-normalization-plan-v1.schema.json"
)
RECEIPT_SCHEMA = (
    ROOT
    / "validation/schemas/"
    "ppe-five-class-normalization-dry-run-receipt-v1.schema.json"
)
RECEIPT = (
    ROOT
    / "validation/results/ppe/normalization/"
    "mendeley-ppe-five-class-v1-normalization-group-split-dry-run-r2.json"
)
PLAN_SHA256 = "35f7fa2b03aa8fb32c2a349144628f3e242f7ee912fad5339c783498030e349d"
RECEIPT_BYTES = 607390
RECEIPT_FILE_SHA256 = "54c54364785b2625afbc109c360ebb715fc03df7462ea420a4ec930ee0cfed62"
RECEIPT_SHA256 = "2391fe3c47f881da190e4dbc8801d83cf5a9f2d586b4eada5b74e6118e9ccd23"
FIXED_TIME = "2026-07-18T00:00:00Z"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def dry_run() -> dict:
    return build_dry_run(
        PLAN,
        expected_plan_sha256=PLAN_SHA256,
        root=ROOT,
        receipt_schema_path=RECEIPT_SCHEMA,
        created_at=FIXED_TIME,
    )


def test_plan_is_closed_schema_valid_and_externally_pinned() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    schema = json.loads(PLAN_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(plan)
    assert _sha256(PLAN) == PLAN_SHA256
    assert plan["materialization"] == {
        "mode": "dry_run_before_materialization",
        "archive_extraction": False,
        "dataset_write": False,
        "overwrite": False,
    }
    assert plan["eligibility_policy"]["training_eligible"] is False
    assert plan["eligibility_policy"]["final_validation_or_test_eligible"] is False


def test_capture_key_contract_strips_roboflow_suffix_and_partitions_windows() -> None:
    prefix = "dataset/train/images/"
    assert capture_key(
        prefix + "ppe_0000_jpg.rf.0123456789abcdef0123456789abcdef.jpg",
        numeric_window_size=32,
    ) == "sequence:ppe:w32:0"
    assert capture_key(
        prefix + "ppe_0031_jpg.rf.0123456789abcdef0123456789abcdef.jpg",
        numeric_window_size=32,
    ) == "sequence:ppe:w32:0"
    assert capture_key(
        prefix + "ppe_0032_jpg.rf.0123456789abcdef0123456789abcdef.jpg",
        numeric_window_size=32,
    ) == "sequence:ppe:w32:1"
    assert capture_key(prefix + "10.jpg", numeric_window_size=32) != capture_key(
        prefix + "00010.jpg", numeric_window_size=32
    )


def test_actual_dry_run_has_exact_group_safe_roles_and_class_balance(
    dry_run: dict,
) -> None:
    verify_receipt_hash(
        dry_run, expected_receipt_sha256=dry_run["receipt_sha256"]
    )
    _verify_internal_receipt_semantics(dry_run)
    assert dry_run["source_observations"]["images"] == 2586
    assert dry_run["source_observations"]["label_rows"] == 17827
    assert dry_run["grouping"]["base_capture_key_count"] == 321
    assert dry_run["grouping"]["final_group_count"] == 292
    assert dry_run["grouping"]["maximum_group_image_count"] == 216
    assert dry_run["grouping"]["exact_duplicate_groups"] == 31
    assert dry_run["grouping"]["cross_upstream_split_exact_duplicate_groups"] == 10
    assert dry_run["grouping"]["duplicate_groups_with_annotation_row_variants"] == 31
    assert dry_run["grouping"]["duplicate_groups_with_class_histogram_conflicts"] == 2
    assert dry_run["grouping"]["duplicate_disposition"] == (
        "retain_all_members_grouped_pending_annotation_adjudication"
    )

    roles = dry_run["assignment"]["roles"]
    assert {role: values["image_count"] for role, values in roles.items()} == {
        "train": 2068,
        "calibration": 259,
        "test": 259,
    }
    assert {role: values["group_count"] for role, values in roles.items()} == {
        "train": 145,
        "calibration": 75,
        "test": 72,
    }
    assert roles["train"]["source_class_bbox_counts"] == {
        "helmet": 4032,
        "no_helmet": 820,
        "no_vest": 2496,
        "person": 4758,
        "vest": 2156,
    }
    assert roles["calibration"]["source_class_bbox_counts"] == {
        "helmet": 502,
        "no_helmet": 105,
        "no_vest": 310,
        "person": 608,
        "vest": 271,
    }
    assert roles["test"]["source_class_bbox_counts"] == {
        "helmet": 502,
        "no_helmet": 101,
        "no_vest": 310,
        "person": 589,
        "vest": 267,
    }
    assert dry_run["assignment"]["max_abs_feature_share_error_ppm"] == 2339
    assert dry_run["assignment"]["within_balance_tolerance"] is True
    assert dry_run["leakage_audit"] == {
        "unique_image_path_coverage": True,
        "image_path_role_leakage_count": 0,
        "capture_key_role_leakage_count": 0,
        "exact_duplicate_role_leakage_count": 0,
        "group_role_leakage_count": 0,
        "leakage_zero": True,
    }


def test_mapping_is_canonical_order_but_remains_semantically_blocked(
    dry_run: dict,
) -> None:
    contract = dry_run["canonical_contract"]
    assert contract["decision_class_order"] == [
        "helmet",
        "no_helmet",
        "hi_vis",
        "no_hi_vis",
    ]
    assert [item["source_name"] for item in contract["mapping"]] == [
        "helmet",
        "no_helmet",
        "no_vest",
        "person",
        "vest",
    ]
    assert [item["canonical_id"] for item in contract["mapping"]] == [
        0,
        1,
        3,
        None,
        2,
    ]
    assert contract["mapping_is_training_ready"] is False
    assert contract["person_is_association_anchor_not_decision_class"] is True
    assert contract["absence_is_not_inferred_from_missing_detection"] is True


def test_dry_run_never_overclaims_upstream_valid_rights_or_training(
    dry_run: dict,
) -> None:
    assert dry_run["source_observations"]["upstream_validation_policy"] == (
        "legacy_source_membership_only_not_independent_test"
    )
    assert dry_run["assignment"]["roles"]["test"]["claim"] == (
        "internal_heldout_audit_only_not_independent_final_test"
    )
    assert dry_run["execution"] == {
        "mode": "dry_run_only",
        "archive_extracted": False,
        "dataset_materialized": False,
        "materialized_dataset_files": 0,
        "archive_redownloaded": False,
        "network_executed": False,
        "gpu_executed": False,
        "docker_executed": False,
        "training_executed": False,
        "admin_service_restarted": False,
    }
    assert dry_run["readiness"]["rights_approved"] is False
    assert dry_run["readiness"]["camera_site_session_grouping_verified"] is False
    assert dry_run["readiness"]["canonical_person_equipment_semantics_approved"] is False
    assert dry_run["readiness"]["normalized_dataset_training_eligible"] is False
    assert dry_run["readiness"]["final_validation_or_test_eligible"] is False
    for blocker in (
        "provenance_review_required",
        "embedded_third_party_rights_review_required",
        "camera_group_split_review_required",
        "exact_duplicate_image_review_required",
        "exact_duplicate_annotation_adjudication_required",
        "vest_to_hi_vis_semantic_review_required",
        "published_test_split_missing",
        "independent_final_test_missing",
    ):
        assert blocker in dry_run["eligibility_blockers"]


def test_build_is_deterministic_when_timestamp_is_fixed(dry_run: dict) -> None:
    replay = build_dry_run(
        PLAN,
        expected_plan_sha256=PLAN_SHA256,
        root=ROOT,
        receipt_schema_path=RECEIPT_SCHEMA,
        created_at=FIXED_TIME,
    )
    assert replay == dry_run


def test_wrong_external_plan_pin_fails_before_publication() -> None:
    with pytest.raises(FiveClassNormalizationError, match="external plan"):
        build_dry_run(
            PLAN,
            expected_plan_sha256="0" * 64,
            root=ROOT,
            receipt_schema_path=RECEIPT_SCHEMA,
        )


def test_atomic_receipt_publication_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    value = {"schema_version": "fixture.no-overwrite/v1", "ok": True}
    write_receipt_no_overwrite(destination, value)
    original = destination.read_bytes()
    with pytest.raises(SeedContractError, match="zaten var"):
        write_receipt_no_overwrite(destination, {**value, "ok": False})
    assert destination.read_bytes() == original


def test_published_dry_run_receipt_verifies_closed_schema_and_semantics(
    tmp_path: Path, dry_run: dict
) -> None:
    destination = tmp_path / "dry-run-receipt.json"
    write_receipt_no_overwrite(destination, dry_run)
    verified = verify_receipt_file(
        destination,
        expected_receipt_sha256=dry_run["receipt_sha256"],
        schema_path=RECEIPT_SCHEMA,
    )
    assert verified["valid"] is True
    assert verified["json_schema_validated"] is True
    assert verified["semantic_fail_closed_checks_verified"] is True


def test_immutable_published_receipt_has_exact_external_pins_and_replays() -> None:
    info = RECEIPT.stat()
    assert info.st_size == RECEIPT_BYTES
    assert info.st_mode & 0o777 == 0o440
    assert info.st_nlink == 1
    assert _sha256(RECEIPT) == RECEIPT_FILE_SHA256
    published = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert published["receipt_sha256"] == RECEIPT_SHA256
    verified = verify_receipt_file(
        RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA256,
        schema_path=RECEIPT_SCHEMA,
    )
    assert verified["file_sha256"] == RECEIPT_FILE_SHA256
    replay = build_dry_run(
        PLAN,
        expected_plan_sha256=PLAN_SHA256,
        root=ROOT,
        receipt_schema_path=RECEIPT_SCHEMA,
        created_at=published["created_at"],
    )
    assert replay == published
