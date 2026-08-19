from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from validation.ppe_human_qa_r6 import (
    DEFAULT_ADJUDICATION_SCHEMA,
    DEFAULT_PLAN,
    DEFAULT_PLAN_SCHEMA,
    DEFAULT_RECEIPT_SCHEMA,
    DEFAULT_SAMPLE_SCHEMA,
    PpeHumanQaR6Error,
    _build_universe,
    _load_plan,
    _select_samples,
    verify_packet,
    verify_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "7a122696d2b1adca5006cb556689fed2154d0e24df33ef73defa8e8355d091ae"
PACKET = ROOT / "validation/results/ppe/human-qa/mendeley-ppe-four-class-r6"
PACKET_RECEIPT = PACKET / "receipt.json"
PACKET_RECEIPT_SHA256 = "87c62ea4ba3515ad549d30a32c9a628d68909f37ca3b372b06fddd44fa071325"
PACKET_RECEIPT_FILE_SHA256 = "f34218294ada32965206326c508dbab7613a1233fac66d2be63605a0c3eafc55"


@pytest.fixture(scope="module")
def plan() -> dict:
    return _load_plan(DEFAULT_PLAN, PLAN_SHA256, root=ROOT)[0]


@pytest.fixture(scope="module")
def universe(plan: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    return _build_universe(plan, root=ROOT)


@pytest.fixture(scope="module")
def samples(plan: dict, universe: tuple[list[dict], list[dict], list[dict], list[dict]]) -> list[dict]:
    retained, quarantine, zeros, _ = universe
    return _select_samples(plan, retained, quarantine, zeros)


def test_all_r6_schemas_are_strict_and_plan_is_exact_pinned() -> None:
    for path in (
        DEFAULT_PLAN_SCHEMA,
        DEFAULT_SAMPLE_SCHEMA,
        DEFAULT_RECEIPT_SCHEMA,
        DEFAULT_ADJUDICATION_SCHEMA,
    ):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
    result = verify_plan(DEFAULT_PLAN, PLAN_SHA256, root=ROOT)
    assert result["expected_samples"] == 718
    assert result["development_holdout_payload_allowed"] is False
    assert result["gpu_allowed"] is False
    assert result["training_authorized"] is False
    assert result["production_ready"] is False


def test_r5_replay_universe_is_exact_and_holdout_free(
    universe: tuple[list[dict], list[dict], list[dict], list[dict]],
) -> None:
    retained, quarantine, zeros, accesses = universe
    assert Counter(row["semantic_class"] for row in retained) == {
        "helmet_worn_candidate": 4365,
        "hi_vis_worn_candidate": 2403,
        "no_helmet_explicit": 925,
    }
    assert len(quarantine) == 2999
    assert Counter(row["quarantine_reason"] for row in quarantine) == {
        "helmet_center_below_associated_person_top_35_percent": 90,
        "helmet_no_person_center_association": 79,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 24,
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 2806,
    }
    assert len(zeros) == 18
    assert Counter(row["kind"] for row in accesses) == {
        "source_label_payload": 2327,
        "candidate_label_payload": 2327,
    }
    assert set(row["role"] for row in accesses) == {"train", "calibration"}
    assert all("development_holdout" not in row["path"].casefold() for row in accesses)


def test_selection_meets_every_r5_human_qa_minimum_and_extra_negative_strata(
    samples: list[dict],
) -> None:
    assert len(samples) == 718
    assert len({row["sample_id"] for row in samples}) == 718
    assert len({row["record_id"] for row in samples}) == 718
    assert Counter(row["category"] for row in samples) == {
        "helmet_head_zone_boundary_retained": 50,
        "helmet_head_zone_boundary_quarantined_below": 50,
        "helmet_worn_candidate_random": 200,
        "hi_vis_worn_candidate_random": 200,
        "quarantine_reason_stratified": 100,
        "no_helmet_explicit_random": 100,
        "candidate_zero_label_context_all": 18,
    }
    assert set(row["role"] for row in samples) == {"train", "calibration"}
    assert len({row["group_id"] for row in samples}) >= 100
    retained_boundary = [
        row for row in samples
        if row["category"] == "helmet_head_zone_boundary_retained"
    ]
    dropped_boundary = [
        row for row in samples
        if row["category"] == "helmet_head_zone_boundary_quarantined_below"
    ]
    assert all(row["association"]["vertical_fraction"] <= 0.35 for row in retained_boundary)
    assert all(row["association"]["vertical_fraction"] > 0.35 for row in dropped_boundary)
    quarantine = [row for row in samples if row["category"] == "quarantine_reason_stratified"]
    assert Counter(row["quarantine_reason"] for row in quarantine) == {
        "helmet_center_below_associated_person_top_35_percent": 25,
        "helmet_no_person_center_association": 25,
        "hi_vis_group_semantic_quarantine_r4_harness_or_uncertain": 24,
        "no_vest_proxy_removed_no_runtime_no_hi_vis_class": 26,
    }
    zeros = [row for row in samples if row["category"] == "candidate_zero_label_context_all"]
    assert all(row["candidate_state"] == "zero_label_unknown_context" for row in zeros)
    assert all(row["xywh"] is None for row in zeros)
    assert all(
        "development_holdout" not in json.dumps(row, sort_keys=True).casefold()
        for row in samples
    )


def test_selection_is_deterministic(
    plan: dict,
    universe: tuple[list[dict], list[dict], list[dict], list[dict]],
    samples: list[dict],
) -> None:
    retained, quarantine, zeros, _ = universe
    replay = _select_samples(plan, retained, quarantine, zeros)
    assert [row["record_id"] for row in replay] == [row["record_id"] for row in samples]


def test_external_plan_pin_tamper_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "plan.json"
    value = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
    value["sampling"]["expected_total_samples"] = 717
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PpeHumanQaR6Error, match="external SHA-256 mismatch"):
        verify_plan(tampered, PLAN_SHA256, root=ROOT)


def test_adjudication_schema_cannot_authorize_training() -> None:
    schema = json.loads(DEFAULT_ADJUDICATION_SCHEMA.read_text(encoding="utf-8"))
    minimal = {
        "schema_version": "deepsafe.ppe-human-qa-adjudication-r6/v1",
        "review_id": "human-review-r6",
        "state": "in_progress",
        "reviewer_identity": "reviewer",
        "packet": {
            "path": "validation/results/ppe/human-qa/x/receipt.json",
            "bytes": 1,
            "sha256": "0" * 64,
            "receipt_sha256": "1" * 64,
        },
        "policy_decisions": {
            name: {"decision": "pending", "notes": "", "decided_at": None}
            for name in (
                "head_zone_helmet_worn_candidate_policy",
                "retained_helmet_false_worn_risk",
                "hi_vis_quarantine_and_retained_semantics",
                "runtime_no_hi_vis_class_absent",
                "unresolved_person_ppe_absence_unknown",
                "dataset_rights_and_ultralytics_license_basis",
            )
        },
        "sample_decisions": [],
        "completion": {
            "expected_samples": 718,
            "decided_samples": 0,
            "all_samples_decided": False,
            "all_policies_decided": False,
            "completed_at": None,
        },
        "training_authorized": False,
        "production_ready": False,
        "authorization_effect": "none_human_qa_only_new_exact_training_authorization_required",
    }
    Draft202012Validator(schema).validate(minimal)
    bad = copy.deepcopy(minimal)
    bad["training_authorized"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(bad)


def test_final_packet_replays_all_visual_hashes_and_no_authorization() -> None:
    result = verify_packet(
        PACKET_RECEIPT,
        expected_receipt_sha256=PACKET_RECEIPT_SHA256,
        root=ROOT,
        verify_visuals=True,
    )
    assert result == {
        "valid": True,
        "path": str(PACKET_RECEIPT),
        "file_sha256": PACKET_RECEIPT_FILE_SHA256,
        "receipt_sha256": PACKET_RECEIPT_SHA256,
        "samples": 718,
        "contact_sheets": 45,
        "roles": {"train": 386, "calibration": 332},
        "groups": 213,
        "development_holdout_payload_files_opened": 0,
        "human_qa_complete": False,
        "training_authorized": False,
        "production_ready": False,
    }


def test_final_packet_tree_is_single_link_and_read_only() -> None:
    assert PACKET.stat().st_mode & 0o777 == 0o550
    for path in PACKET.rglob("*"):
        info = path.lstat()
        if path.is_dir():
            assert info.st_mode & 0o777 == 0o550
        else:
            assert info.st_mode & 0o777 == 0o440
            assert info.st_nlink == 1
