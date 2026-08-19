from __future__ import annotations

import copy
import json

import pytest

from ppe_dataset.cli import main
from ppe_dataset.contract_v2 import validate_dataset_v2


def valid_v2_dataset() -> dict:
    return {
        "schema_version": "ppe-person-equipment-decisions-v2.0",
        "dataset": {
            "id": "fixture-v2",
            "title": "PPE decision ROI fixture",
            "modality": "static_images",
            "annotation_status": "approved",
            "license_review": "fixture approved",
            "privacy_review": "fixture approved",
        },
        "label_policy": {
            "policy_id": "fixture-policy-v1",
            "decision_bbox_semantics": "visible_head_or_torso_region_used_for_both_present_and_absent_class_decisions",
            "unknown_emits_training_label": False,
            "absence_minimum_visible_fraction": 0.5,
            "hi_vis_definition": "Visible certified high-visibility garment with project-approved reflective pattern.",
            "carried_helmet_compliance": "absent",
        },
        "sources": [
            {
                "id": "source-1",
                "source_url": "https://example.invalid/source",
                "local_path": "fixture.jpg",
                "sha256": "a" * 64,
                "license": {
                    "spdx": "CC0-1.0",
                    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "attribution": "fixture",
                    "commercial_use": True,
                    "review_status": "approved",
                },
                "provenance_group_id": "capture-1",
            }
        ],
        "videos": [],
        "images": [
            {
                "id": 1,
                "source_id": "source-1",
                "file_name": "fixture.jpg",
                "video_id": None,
                "frame_index": None,
                "timestamp_ms": None,
                "width": 640,
                "height": 480,
                "split": "train",
                "sha256": "b" * 64,
            }
        ],
        "persons": [
            {
                "id": 1,
                "image_id": 1,
                "person_bbox": [100, 50, 120, 300],
                "person_area": 36000,
                "iscrowd": 0,
                "ignore": False,
                "track_id": None,
                "attributes": {
                    "helmet": {
                        "state": "absent",
                        "decision_class": "no_helmet",
                        "decision_bbox": [110, 50, 100, 90],
                        "decision_area": 9000,
                        "item_bbox": None,
                        "item_area": None,
                        "visibility": "visible",
                        "visible_fraction": 0.9,
                        "occlusion": "none",
                        "truncated": False,
                        "blur": "none",
                        "unknown_reason": None,
                        "worn_correctly": False,
                        "review_status": "double_review",
                    },
                    "hi_vis": {
                        "state": "present",
                        "decision_class": "hi_vis",
                        "decision_bbox": [105, 120, 110, 140],
                        "decision_area": 15400,
                        "item_bbox": [110, 125, 100, 130],
                        "item_area": 13000,
                        "visibility": "partial",
                        "visible_fraction": 0.75,
                        "occlusion": "partial",
                        "truncated": False,
                        "blur": "mild",
                        "unknown_reason": None,
                        "worn_correctly": True,
                        "review_status": "adjudicated",
                    },
                    "person_occlusion": "partial",
                    "truncated": False,
                    "distance_m": 22.5,
                    "distance_source": "calibrated",
                    "distance_evaluation_eligible": True,
                },
            }
        ],
        "decision_classes": [
            {"id": 0, "name": "helmet"},
            {"id": 1, "name": "no_helmet"},
            {"id": 2, "name": "hi_vis"},
            {"id": 3, "name": "no_hi_vis"},
        ],
    }


def test_valid_v2_semantics_and_schema() -> None:
    document = valid_v2_dataset()
    result = validate_dataset_v2(document)
    assert result.valid, result.as_dict()
    assert result.stats["training_labels"] == 2
    assert result.stats["class_counts"] == {
        "helmet": 0,
        "no_helmet": 1,
        "hi_vis": 1,
        "no_hi_vis": 0,
    }
    assert result.stats["verified_persons_at_or_below_25m"] == 1

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        open(
            "ppe_dataset/schemas/person-equipment-decisions-v2.schema.json",
            encoding="utf-8",
        ).read()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(document)


def test_unknown_cannot_emit_a_yolo_label_and_requires_reason() -> None:
    document = valid_v2_dataset()
    helmet = document["persons"][0]["attributes"]["helmet"]
    helmet["state"] = "unknown"
    helmet["unknown_reason"] = None
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "unknown_emits_training_label" in codes
    assert "unknown_reason_required" in codes


def test_absent_requires_visible_decision_region() -> None:
    document = valid_v2_dataset()
    helmet = document["persons"][0]["attributes"]["helmet"]
    helmet["visible_fraction"] = 0.2
    helmet["visibility"] = "too_small"
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "absence_without_visibility" in codes
    assert "known_region_not_observable" in codes


def test_decision_region_must_be_inside_person_and_match_class() -> None:
    document = valid_v2_dataset()
    helmet = document["persons"][0]["attributes"]["helmet"]
    helmet["decision_bbox"] = [1, 1, 20, 20]
    helmet["decision_area"] = 400
    helmet["decision_class"] = "helmet"
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "decision_bbox_outside_person" in codes
    assert "decision_class_mismatch" in codes


def test_carried_or_incorrect_helmet_cannot_be_present() -> None:
    document = valid_v2_dataset()
    helmet = document["persons"][0]["attributes"]["helmet"]
    helmet["state"] = "present"
    helmet["decision_class"] = "helmet"
    helmet["worn_correctly"] = False
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "present_must_be_worn" in codes


def test_approved_video_ground_truth_requires_track_ids() -> None:
    document = valid_v2_dataset()
    document["dataset"]["modality"] = "video_frames"
    document["videos"] = [
        {
            "id": "video-1",
            "source_id": "source-1",
            "camera_id": "cam-1",
            "camera_group_id": "physical-cam-1",
            "session_id": "session-1",
            "provenance_group_id": "capture-1",
            "width": 640,
            "height": 480,
            "fps": 25.0,
            "duration_ms": 1000.0,
        }
    ]
    image = document["images"][0]
    image["video_id"] = "video-1"
    image["frame_index"] = 0
    image["timestamp_ms"] = 0
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "approved_video_requires_track" in codes


def test_static_provenance_group_cannot_leak_across_splits() -> None:
    document = valid_v2_dataset()
    document["images"].append(
        {
            **copy.deepcopy(document["images"][0]),
            "id": 2,
            "file_name": "fixture-2.jpg",
            "split": "test",
            "sha256": "c" * 64,
        }
    )
    codes = {issue.code for issue in validate_dataset_v2(document).issues}
    assert "provenance_split_leakage" in codes


def test_cli_qa_v2_writes_machine_readable_report(tmp_path) -> None:
    pytest.importorskip("jsonschema")
    input_path = tmp_path / "dataset.json"
    output_path = tmp_path / "qa.json"
    input_path.write_text(json.dumps(valid_v2_dataset()), encoding="utf-8")

    assert main(["qa-v2", "--input", str(input_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["semantic"]["stats"]["training_labels"] == 2


def test_semantic_validator_reports_non_object_records_without_crashing() -> None:
    document = valid_v2_dataset()
    document["sources"].append("not-an-object")
    document["videos"].append(42)
    document["images"].append(None)
    document["persons"].append([])

    result = validate_dataset_v2(document)
    assert not result.valid
    assert sum(issue.code == "expected_object" for issue in result.issues) == 4
