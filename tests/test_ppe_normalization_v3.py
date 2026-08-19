from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from ppe_dataset.normalization_v2 import (
    NormalizationContractError,
    _canonical_receipt_sha256,
)
from ppe_dataset.normalization_v3 import (
    _compare_provenance_replay,
    build_assessment_r2,
    verify_assessment_receipt_r2,
)


PLAN_PATH = Path("data/manifests/ppe-normalization-mendeley-v6-r2.plan.json")
PLAN_SHA256 = "a14e31b5c0aa08827ff935fff3a0bf2476495207ef2877ed6f00b13993450d5f"
RECEIPT_PATH = Path(
    "validation/results/ppe/normalization/mendeley-ppe-v6-normalization-assessment-r2.json"
)
RECEIPT_FILE_SHA256 = "033817d5782fc0ca3259eefc88fed5c4c32ba31e59bda5cda0a3ac4d72fb8327"
RECEIPT_SHA256 = "04af74793d48e9af477c85d8800170c71fc0d8b45100e79c19d94b3bd6acb704"
RECEIPT_SCHEMA_SHA256 = "d7922922611709f6a4d85d2faf38c82bf6be0cd5737464eca1da1a497f74afbd"
R1_PLAN_SHA256 = "41c9943dc0487a75f859a825f77da0e281b8699cb95b40604ac8c293e71df463"
R1_RECEIPT_FILE_SHA256 = "c182d78789d2cf927974803db23b233257b2d21c500c8ad3519f27f23f0361cb"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r2_plan_receipt_and_schema_are_exact_pinned() -> None:
    assert hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == PLAN_SHA256
    assert hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest() == RECEIPT_FILE_SHA256
    verification = verify_assessment_receipt_r2(
        RECEIPT_PATH,
        expected_receipt_sha256=RECEIPT_SHA256,
        expected_schema_sha256=RECEIPT_SCHEMA_SHA256,
    )
    assert verification["valid"] is True
    assert verification["external_pin_verified"] is True
    assert verification["schema_external_pin_verified"] is True
    assert verification["provenance_review_evidence_present"] is True
    assert verification["provenance_mechanical_audit_replayed"] is True
    assert verification["provenance_review_approved"] is False
    assert verification["training_eligible"] is False
    assert verification["normalization_ready"] is False


def test_r1_history_is_unchanged_and_read_only() -> None:
    r1_plan = Path("data/manifests/ppe-normalization-mendeley-v6.plan.json")
    r1_receipt = Path(
        "validation/results/ppe/normalization/mendeley-ppe-v6-normalization-assessment-r1.json"
    )
    assert hashlib.sha256(r1_plan.read_bytes()).hexdigest() == R1_PLAN_SHA256
    assert hashlib.sha256(r1_receipt.read_bytes()).hexdigest() == R1_RECEIPT_FILE_SHA256
    assert os.stat(r1_receipt).st_mode & 0o777 == 0o440
    assert os.stat(RECEIPT_PATH).st_mode & 0o777 == 0o440


def test_r2_exposes_exact_provenance_replay_without_promoting_approval() -> None:
    receipt = _load(RECEIPT_PATH)
    authoritative = receipt["observations"]["provenance_authoritative"]
    replay = receipt["observations"]["provenance_replay"]
    assert authoritative == replay
    assert replay["images"] == 2286
    assert replay["exact_content_duplicate_groups"] == 0
    assert replay["exact_original_key_duplicate_groups"] == 0
    assert replay["filename_families"] == 6
    assert replay["cross_split_filename_families"] == 6
    assert replay["images_in_cross_split_filename_families"] == 2286
    assert replay["strict_cross_split_candidate_pairs"] == 7
    assert replay["strict_cross_split_validation_members"] == 4
    assert replay["high_confidence_automated_candidate_pairs"] == 2
    assert replay["human_review_complete"] is False
    assert receipt["provenance_review_evidence_present"] is True
    assert receipt["provenance_review_approved"] is False
    assert receipt["embedded_third_party_rights_review_approved"] is False
    assert receipt["camera_site_session_group_split_approved"] is False


def test_r2_preserves_bbox_and_dimension_findings_without_repair() -> None:
    receipt = _load(RECEIPT_PATH)
    labels = receipt["observations"]["label_audit_replay"]
    assert labels["image_count"] == labels["label_file_count"] == 2286
    assert labels["label_row_count"] == 6038
    assert labels["class_id_histogram"] == {
        "0": 2421,
        "1": 1174,
        "2": 1058,
        "3": 1385,
    }
    assert labels["issue_counts"] == {"bbox_out_of_range": 52}
    assert labels["automatic_repairs_applied"] == []
    assert labels["silent_bbox_clipping_applied"] is False
    assert labels["canonical_labels_emitted"] is False
    assert receipt["observations"]["dimension_audit"] == {
        "declared": {"height": 640, "width": 640},
        "decoded_count": 2286,
        "gate_passed": False,
        "mismatch_count": 1814,
        "policy": "preserve_decoded_per_image_dimensions_no_resize_claim",
    }


def test_receipt_content_tamper_fails_self_hash(tmp_path: Path) -> None:
    receipt = _load(RECEIPT_PATH)
    receipt["failure"]["message"] += " tampered"
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(NormalizationContractError) as error:
        verify_assessment_receipt_r2(
            target,
            expected_receipt_sha256=RECEIPT_SHA256,
            expected_schema_sha256=RECEIPT_SCHEMA_SHA256,
        )
    assert error.value.code == "receipt_self_hash_mismatch"


def test_resealed_tamper_still_fails_external_receipt_pin(tmp_path: Path) -> None:
    receipt = _load(RECEIPT_PATH)
    receipt["failure"]["message"] += " resealed-tamper"
    receipt["receipt_sha256"] = _canonical_receipt_sha256(receipt)
    target = tmp_path / "resealed.json"
    target.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(NormalizationContractError) as error:
        verify_assessment_receipt_r2(
            target,
            expected_receipt_sha256=RECEIPT_SHA256,
            expected_schema_sha256=RECEIPT_SCHEMA_SHA256,
        )
    assert error.value.code == "receipt_external_pin_mismatch"


def test_wrong_schema_external_pin_is_rejected() -> None:
    with pytest.raises(NormalizationContractError) as error:
        verify_assessment_receipt_r2(
            RECEIPT_PATH,
            expected_receipt_sha256=RECEIPT_SHA256,
            expected_schema_sha256="0" * 64,
        )
    assert error.value.code == "schema_external_pin_mismatch"


def test_wrong_plan_external_pin_fails_before_expensive_replay() -> None:
    with pytest.raises(NormalizationContractError) as error:
        build_assessment_r2(PLAN_PATH, expected_plan_sha256="0" * 64)
    assert error.value.code == "plan_external_pin_mismatch"


def test_mechanical_replay_comparator_detects_observation_tamper() -> None:
    provenance_receipt = _load(
        Path(
            "validation/results/ppe/provenance/mendeley-ppe-v6-provenance-review-r2.json"
        )
    )
    tampered = copy.deepcopy(provenance_receipt)
    tampered["observations"]["images"] = 2285
    expected = _load(PLAN_PATH)["expected_observations"]["provenance"]
    passed, details = _compare_provenance_replay(
        provenance_receipt, tampered, expected
    )
    assert passed is False
    assert details["authoritative"]["images"] == 2286
    assert details["replay"]["images"] == 2285
