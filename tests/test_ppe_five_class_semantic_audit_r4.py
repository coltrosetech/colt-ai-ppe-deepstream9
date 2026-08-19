from __future__ import annotations

import copy
import hashlib
import json
import stat
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ppe_dataset.five_class_semantic_audit_r4 import (
    DEFAULT_RECEIPT_SCHEMA,
    DEFAULT_REVIEW_SCHEMA,
    DEFAULT_SELECTION_SCHEMA,
    FiveClassSemanticAuditError,
    _aggregate_reviews,
    _assert_authorized_payload,
    _parse_label,
    _validate_jsonl,
    verify_audit,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/manifests/ppe-mendeley-five-class-v1-semantic-audit-r4.plan.json"
PLAN_SCHEMA = ROOT / "ppe_dataset/schemas/ppe-five-class-semantic-audit-plan-v1.schema.json"
RESULT = ROOT / "validation/results/ppe/semantic-audit/mendeley-ppe-five-class-v1-r4"
RECEIPT = RESULT / "receipt.json"
SELECTION = RESULT / "selection.jsonl"
REVIEWS = RESULT / "manual-reviews.jsonl"
ACCESS = RESULT / "payload-access.jsonl"
PLAN_SHA256 = "ce9839151104c954c7414f7aae8d409a40ce90b79f0ea3627fc1350a84fe3b74"
RECEIPT_SHA256 = "42c1e5ef444c598cf8b80cefcb447c3f60e1ddd1d0467ee610dafaf9d5a038ff"
RECEIPT_FILE_SHA256 = "298dc32bea3c101cfefd76513692202baa75d3cf796b3a3938f7409e7bcd5694"
SELECTION_SHA256 = "016261a2d26e061577f1c048e4b912443be851801ce3f1a5f766cb82dc90f18e"
REVIEW_SHA256 = "53323b45a775be77d31253db2476815ed3e3a42931a7fb8c12e19f6a44544460"
ACCESS_SHA256 = "02a39fd5edc9bae25843966a3062992ee505a78786fe21066cf0ff4720eea7ad"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def selection() -> list[dict]:
    return _validate_jsonl(SELECTION, DEFAULT_SELECTION_SCHEMA)


@pytest.fixture(scope="module")
def reviews() -> list[dict]:
    return _validate_jsonl(REVIEWS, DEFAULT_REVIEW_SCHEMA)


def test_plan_and_all_schemas_are_strict_and_exact_pinned() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    schema = json.loads(PLAN_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(plan)
    for path in (DEFAULT_SELECTION_SCHEMA, DEFAULT_REVIEW_SCHEMA, DEFAULT_RECEIPT_SCHEMA):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
    assert _sha256(PLAN) == PLAN_SHA256
    assert plan["sampling"] == {
        "seed": "deepsafe-ppe-semantic-r4-20260718",
        "authorized_roles": ["train", "calibration"],
        "excluded_role": "development_holdout",
        "selected_images": 20,
        "role_image_counts": {"train": 14, "calibration": 6},
        "minimum_groups": 12,
        "maximum_images_per_group": 2,
        "minimum_bbox_per_class": 12,
        "minimum_bbox_per_class_size": 2,
        "size_bin_definition": {
            "metric": "sqrt_bbox_area_pixels_at_native_640",
            "small": "value_lt_32",
            "medium": "32_lte_value_lt_96",
            "large": "value_gte_96",
        },
        "selection_method": "deterministic_greedy_role_group_class_size_deficit_v1",
    }
    assert plan["execution_constraints"] == {
        "cpu_only": True,
        "network_allowed": False,
        "gpu_query_allowed": False,
        "docker_allowed": False,
        "training_allowed": False,
        "export_allowed": False,
        "admin_rebuild_allowed": False,
    }


def test_receipt_replay_hashes_and_claim_guardrails_are_exact() -> None:
    result = verify_audit(RECEIPT, expected_receipt_sha256=RECEIPT_SHA256, root=ROOT)
    assert result == {
        "valid": True,
        "path": str(RECEIPT),
        "file_sha256": RECEIPT_FILE_SHA256,
        "receipt_sha256": RECEIPT_SHA256,
        "images": 20,
        "groups": 18,
        "review_records": 20,
        "development_holdout_payload_files_opened": 0,
        "human_adjudication_required": True,
        "production_ready": False,
    }
    assert _sha256(RECEIPT) == RECEIPT_FILE_SHA256
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["readiness"] == {
        "development_semantic_audit_complete": True,
        "ai_review_complete": True,
        "human_adjudication_required": True,
        "semantic_mapping_approved": False,
        "dataset_rights_cleared": False,
        "training_authorized_by_this_audit": False,
        "development_holdout_opened": False,
        "independent_final_test_available": False,
        "production_ready": False,
    }


def test_selection_is_role_group_class_and_size_stratified(selection: list[dict]) -> None:
    assert _sha256(SELECTION) == SELECTION_SHA256
    assert len(selection) == 20
    assert Counter(row["role"] for row in selection) == {"train": 14, "calibration": 6}
    groups = Counter(row["group_id"] for row in selection)
    assert len(groups) == 18
    assert max(groups.values()) == 2
    classes: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    for row in selection:
        classes.update(row["canonical_class_bbox_counts"])
        strata.update(row["class_size_bbox_counts"])
        assert row["source_to_canonical_geometry_exact"] is True
        assert "development_holdout" not in json.dumps(row).casefold()
    assert classes == {
        "person": 163,
        "helmet": 109,
        "no_helmet": 58,
        "hi_vis": 37,
        "no_hi_vis": 121,
    }
    assert set(strata) == {
        f"{class_name}:{size_name}"
        for class_name in ("person", "helmet", "no_helmet", "hi_vis", "no_hi_vis")
        for size_name in ("small", "medium", "large")
    }
    assert min(strata.values()) >= 2


def test_payload_access_ledger_proves_holdout_payload_count_zero() -> None:
    assert _sha256(ACCESS) == ACCESS_SHA256
    rows = [json.loads(line) for line in ACCESS.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2367
    assert Counter(row["kind"] for row in rows) == {
        "materialized_label_for_stratification": 2327,
        "selected_materialized_image": 20,
        "selected_source_label_zip_member": 20,
    }
    assert set(row["role"] for row in rows) == {"train", "calibration"}
    assert all("development_holdout" not in row["path"].casefold() for row in rows)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["payload_access"]["development_holdout_payload_files_opened"] == 0


def test_ai_review_findings_are_complete_and_do_not_overclaim(
    selection: list[dict], reviews: list[dict]
) -> None:
    assert _sha256(REVIEWS) == REVIEW_SHA256
    summary = _aggregate_reviews(reviews, selection)
    assert summary["overall_decisions"] == {
        "accept_for_development_with_guardrails": 2,
        "questionable_needs_adjudication": 15,
        "reject_from_development_candidate": 3,
    }
    assert summary["mapping_decisions"] == {
        "person": {"acceptable": 20},
        "helmet": {"acceptable": 14, "questionable": 2},
        "no_helmet": {"acceptable": 12},
        "hi_vis": {"acceptable": 4, "incorrect": 1, "questionable": 2},
        "no_hi_vis": {"questionable": 17},
    }
    assert summary["issue_code_counts"] == {
        "association_ambiguity": 5,
        "helmet_semantic_ambiguity": 2,
        "no_vest_no_hi_vis_semantic_risk": 17,
        "occlusion_limits_review": 4,
        "person_box_issue": 1,
        "vest_hi_vis_semantic_risk": 3,
    }
    assert summary["duplicate_annotation_decisions"] == {"none_seen": 20}
    assert all(row["review"]["status"] == "ai_reviewed_needs_human_qa" for row in reviews)


def test_held_helmet_and_harness_failures_are_explicit(reviews: list[dict]) -> None:
    by_id = {row["sample_id"]: row for row in reviews}
    harness = by_id["ppe-sem-r4-004"]
    assert harness["observation"]["overall_decision"] == "reject_from_development_candidate"
    hi_vis = next(
        item for item in harness["observation"]["mapping"]
        if item["canonical_class"] == "hi_vis"
    )
    assert hi_vis["decision"] == "incorrect"
    assert hi_vis["risk_codes"] == ["vest_not_confirmably_high_visibility"]
    for sample_id, held_ids in {
        "ppe-sem-r4-016": {"a024", "a029", "a030"},
        "ppe-sem-r4-019": {"a023", "a028", "a032"},
    }.items():
        review = by_id[sample_id]
        assert review["observation"]["association"]["decision"] == "issue"
        helmet = next(
            item for item in review["observation"]["mapping"]
            if item["canonical_class"] == "helmet"
        )
        assert helmet["decision"] == "questionable"
        referenced = {
            annotation_id
            for issue in review["observation"]["issues"]
            for annotation_id in issue["annotation_ids"]
        }
        assert held_ids <= referenced


def test_role_gate_and_source_class_remap_fail_closed() -> None:
    _assert_authorized_payload("train", "data/x/images/train/a.jpg", "image")
    _assert_authorized_payload(
        "calibration", "data/x/labels/calibration/a.txt", "label"
    )
    for role, path in (
        ("development_holdout", "data/x/images/development_holdout/a.jpg"),
        ("train", "data/x/images/development_holdout/a.jpg"),
        ("train", "data/x/images/calibration/a.jpg"),
    ):
        with pytest.raises(FiveClassSemanticAuditError):
            _assert_authorized_payload(role, path, "fixture")
    source = _parse_label(
        b"0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n"
        b"2 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n"
        b"4 0.5 0.5 0.1 0.1\n",
        source=True,
    )
    assert [row["canonical_class_id"] for row in source] == [1, 2, 4, 0, 3]


def test_review_count_evidence_and_annotation_reference_tamper_fail_closed(
    selection: list[dict], reviews: list[dict]
) -> None:
    bad = copy.deepcopy(reviews)
    bad[0]["observation"]["mapping"][0]["annotation_count"] += 1
    with pytest.raises(FiveClassSemanticAuditError, match="annotation count"):
        _aggregate_reviews(bad, selection)
    bad = copy.deepcopy(reviews)
    bad[0]["evidence"]["contact_sheet_tile_index"] = 3
    with pytest.raises(FiveClassSemanticAuditError, match="evidence"):
        _aggregate_reviews(bad, selection)
    bad = copy.deepcopy(reviews)
    bad[0]["observation"]["issues"][0]["annotation_ids"] = ["a999"]
    with pytest.raises(FiveClassSemanticAuditError, match="unknown annotation"):
        _aggregate_reviews(bad, selection)


def test_final_evidence_tree_is_single_link_and_read_only() -> None:
    assert stat.S_IMODE(RESULT.stat().st_mode) == 0o550
    for path in RESULT.rglob("*"):
        info = path.lstat()
        if path.is_dir():
            assert stat.S_IMODE(info.st_mode) == 0o550
        else:
            assert stat.S_ISREG(info.st_mode)
            assert stat.S_IMODE(info.st_mode) == 0o440
            assert info.st_nlink == 1
