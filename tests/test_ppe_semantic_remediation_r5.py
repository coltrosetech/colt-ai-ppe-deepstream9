from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from validation.ppe_semantic_remediation_r5 import (
    CANDIDATE_CLASSES,
    DEFAULT_CONTRACT_SCHEMA,
    DEFAULT_LEDGER_SCHEMA,
    DEFAULT_QUARANTINE_SCHEMA,
    DEFAULT_RECEIPT_SCHEMA,
    _load_canonical_jsonl,
    _transform_label,
    verify,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "models/ppe/training-lanes/"
    "yolo11s-mendeley-four-class-remediated-r5/transform-contract-r5.json"
)
RESULT = (
    ROOT
    / "validation/results/ppe/semantic-remediation/"
    "mendeley-ppe-four-class-remediated-r5"
)
RECEIPT = RESULT / "receipt.json"
DATASET = ROOT / "data/derived/ppe/mendeley-ppe-four-class-remediated-r5"
CONTRACT_SHA256 = "2ad22266d82c56996f298d44d33c385d9c15c67e9f1b290377b4999668c47c9e"
RECEIPT_SHA256 = "880c54a687ca61dc111d8c634647b7d033c337ac718fa155b7e637b31eb0b887"
RECEIPT_FILE_SHA256 = "8a2397315f070d3a6a945b778ca89e973f5af441e435eeb812c788e184a8058b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_and_all_r5_schemas_are_exact_and_strict() -> None:
    assert _sha256(CONTRACT) == CONTRACT_SHA256
    assert stat.S_IMODE(CONTRACT.stat().st_mode) == 0o440
    assert CONTRACT.stat().st_nlink == 1
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path in (
        DEFAULT_CONTRACT_SCHEMA,
        DEFAULT_RECEIPT_SCHEMA,
        DEFAULT_LEDGER_SCHEMA,
        DEFAULT_QUARANTINE_SCHEMA,
    ):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        json.loads(DEFAULT_CONTRACT_SCHEMA.read_text(encoding="utf-8"))
    ).validate(contract)
    assert contract["status"] == "semantic_remediation_prepared_not_training_authorized"
    assert contract["readiness"] == {
        "semantic_remediation_prepared": True,
        "human_qa_required": True,
        "training_authorized": False,
        "production_ready": False,
    }
    assert all(value is False for value in contract["execution_constraints"].values())


def test_full_receipt_replays_every_authorized_payload_and_exact_transform() -> None:
    result = verify(
        RECEIPT,
        expected_receipt_sha256=RECEIPT_SHA256,
        expected_contract_sha256=CONTRACT_SHA256,
        root=ROOT,
    )
    assert result["valid"] is True
    assert result["receipt_file_sha256"] == RECEIPT_FILE_SHA256
    assert result["images"] == 2327
    assert result["retained_bbox_rows"] == 13059
    assert result["quarantined_bbox_rows"] == 2999
    assert result["development_holdout_payload_files_opened"] == 0
    assert result["human_qa_required"] is True
    assert result["training_authorized"] is False
    assert result["production_ready"] is False


def test_exact_candidate_and_quarantine_counts() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["candidate"]["retained_class_counts"] == {
        "helmet_worn_candidate": 4365,
        "hi_vis_worn_candidate": 2403,
        "no_helmet_explicit": 925,
        "person": 5366,
    }
    assert receipt["candidate"]["quarantine_reason_counts"] == {
        "helmet_center_below_associated_person_top_35_percent": 90,
        "helmet_no_person_center_association": 79,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 24,
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 2806,
    }
    assert receipt["candidate"]["roles"] == {
        "train": {
            "images": 2068,
            "groups": 145,
            "retained_bbox_rows": 11612,
            "retained_class_counts": {
                "helmet_worn_candidate": 3878,
                "hi_vis_worn_candidate": 2156,
                "no_helmet_explicit": 820,
                "person": 4758,
            },
        },
        "calibration": {
            "images": 259,
            "groups": 75,
            "retained_bbox_rows": 1447,
            "retained_class_counts": {
                "helmet_worn_candidate": 487,
                "hi_vis_worn_candidate": 247,
                "no_helmet_explicit": 105,
                "person": 608,
            },
        },
    }


def test_no_runtime_no_hi_vis_class_and_all_candidate_ids_are_zero_to_three() -> None:
    policy = json.loads((DATASET / "metadata/semantic-policy.json").read_text())
    assert policy["no_hi_vis_runtime_detector_class"] is False
    assert policy["no_hi_vis_absence_policy"] == (
        "calibrated_person_association_policy_or_unknown_only"
    )
    observed: Counter[int] = Counter()
    empty_labels = 0
    for path in (DATASET / "labels").rglob("*.txt"):
        rows = path.read_text(encoding="utf-8").splitlines()
        empty_labels += not rows
        for row in rows:
            class_id = int(row.split()[0])
            assert 0 <= class_id <= 3
            observed[class_id] += 1
    assert observed == {0: 5366, 1: 4365, 2: 925, 3: 2403}
    assert empty_labels == 18
    assert list(CANDIDATE_CLASSES) == [
        "person",
        "helmet_worn_candidate",
        "no_helmet_explicit",
        "hi_vis_worn_candidate",
    ]


def test_quarantine_ledger_is_complete_and_semantically_scoped() -> None:
    rows = _load_canonical_jsonl(
        (DATASET / "metadata/quarantine.jsonl").read_bytes(),
        artifact="test_quarantine",
    )
    assert len(rows) == 2999
    assert Counter(row["reason"] for row in rows) == {
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 2806,
        "helmet_no_person_center_association": 79,
        "helmet_center_below_associated_person_top_35_percent": 90,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 24,
    }
    below = [
        row
        for row in rows
        if row["reason"]
        == "helmet_center_below_associated_person_top_35_percent"
    ]
    assert all(row["associated_person_vertical_fraction"] > 0.35 for row in below)
    unassociated = [
        row for row in rows if row["reason"] == "helmet_no_person_center_association"
    ]
    assert all(row["associated_person_source_line"] is None for row in unassociated)
    assert all(row["human_qa_required"] is True for row in rows)
    assert all(row["training_authorization_effect"] == "none" for row in rows)


def test_synthetic_head_zone_and_group_quarantine_rule() -> None:
    group = "grp_" + "a" * 64
    row = {
        "role": "train",
        "group_id": group,
        "materialized_image": {"path": f"images/train/{'b' * 64}.jpg"},
        "materialized_label": {"path": f"labels/train/{'b' * 64}.txt"},
    }
    payload = (
        b"0 0.5 0.5 0.4 0.8\n"
        b"1 0.5 0.2 0.1 0.1\n"
        b"1 0.5 0.7 0.1 0.1\n"
        b"1 0.9 0.2 0.1 0.1\n"
        b"2 0.5 0.2 0.1 0.1\n"
        b"3 0.5 0.5 0.2 0.2\n"
        b"4 0.5 0.5 0.2 0.2\n"
    )
    output, summary, quarantine = _transform_label(
        row=row, label_payload=payload, quarantined_groups={group}
    )
    assert output.decode().splitlines() == [
        "0 0.5 0.5 0.4 0.8",
        "1 0.5 0.2 0.1 0.1",
        "2 0.5 0.2 0.1 0.1",
    ]
    assert summary["retained_class_counts"] == {
        "helmet_worn_candidate": 1,
        "no_helmet_explicit": 1,
        "person": 1,
    }
    assert Counter(row["reason"] for row in quarantine) == {
        "helmet_center_below_associated_person_top_35_percent": 1,
        "helmet_no_person_center_association": 1,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 1,
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 1,
    }


def test_payload_access_and_human_qa_request_do_not_overclaim() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["payload_access"] == {
        "train_image_payload_files_opened": 2068,
        "train_label_payload_files_opened": 2068,
        "calibration_image_payload_files_opened": 259,
        "calibration_label_payload_files_opened": 259,
        "development_holdout_image_payload_files_opened": 0,
        "development_holdout_label_payload_files_opened": 0,
    }
    request = json.loads((RESULT / "human-qa-request.json").read_text())
    assert request["status"] == (
        "awaiting_independent_human_qa_and_new_training_authorization"
    )
    assert request["development_holdout_payload_access_allowed"] is False
    assert request["training_authorized_by_request"] is False
    assert request["authorization_effect"] == "none_request_only"


def test_final_candidate_and_receipt_trees_are_single_link_read_only() -> None:
    for root in (DATASET, RESULT):
        assert stat.S_IMODE(root.stat().st_mode) == 0o550
        for path in root.rglob("*"):
            info = path.lstat()
            if path.is_dir():
                assert stat.S_IMODE(info.st_mode) == 0o550
            else:
                assert stat.S_ISREG(info.st_mode)
                assert info.st_nlink == 1
                assert stat.S_IMODE(info.st_mode) == 0o440
