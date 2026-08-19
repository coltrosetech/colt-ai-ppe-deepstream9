import copy
import json

import pytest

from ppe_dataset.cli import main
from ppe_dataset.contract import validate_dataset
from ppe_dataset.sampling import FrameMetric, select_diverse_frames
from ppe_dataset.scoring import score_events
from ppe_dataset.splitting import assign_splits


def valid_dataset():
    return {
        "schema_version": "ppe-person-attributes-v1.0",
        "dataset": {
            "id": "fixture",
            "title": "PPE fixture",
            "annotation_status": "qa_pending",
            "privacy_review": "test fixture",
        },
        "sources": [
            {
                "id": "source-1",
                "source_url": "https://example.invalid/source",
                "download_url": None,
                "local_path": "fixture.mp4",
                "sha256": "a" * 64,
                "license": {
                    "spdx": "CC0-1.0",
                    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "attribution": "fixture",
                    "share_alike": False,
                },
            }
        ],
        "videos": [
            {
                "id": "video-1",
                "source_id": "source-1",
                "camera_id": "camera-1",
                "camera_group_id": "physical-camera-1",
                "provenance_group_id": "capture-1",
                "width": 640,
                "height": 480,
                "fps": 25.0,
                "duration_ms": 1000.0,
            }
        ],
        "images": [
            {
                "id": 1,
                "file_name": "frame.jpg",
                "video_id": "video-1",
                "frame_index": 0,
                "timestamp_ms": 0,
                "width": 640,
                "height": 480,
                "split": "train",
                "sha256": "b" * 64,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 50, 120, 300],
                "area": 36000,
                "iscrowd": 0,
                "ignore": False,
                "attributes": {
                    "helmet": {"state": "absent", "visibility": "visible", "visible_fraction": 0.9},
                    "hi_vis": {"state": "present", "visibility": "partial", "visible_fraction": 0.7},
                    "person_occlusion": "partial",
                    "truncated": False,
                    "distance_m": 12.0,
                    "distance_source": "measured",
                    "distance_bin": "10-15",
                    "distance_evaluation_eligible": True,
                },
            }
        ],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }


def test_valid_contract_passes_schema_and_semantic_rules():
    jsonschema = pytest.importorskip("jsonschema")
    document = valid_dataset()
    schema = json.loads(open("ppe_dataset/schemas/person-ppe-attributes-v1.schema.json", encoding="utf-8").read())
    jsonschema.Draft202012Validator(schema).validate(document)
    result = validate_dataset(document)
    assert result.valid
    assert result.stats["verified_persons_at_or_below_25m"] == 1
    assert result.stats["ppe_states"]["helmet"]["absent"] == 1


def test_absent_requires_observable_body_region():
    document = valid_dataset()
    document["annotations"][0]["attributes"]["helmet"] = {
        "state": "absent",
        "visibility": "too_small",
        "visible_fraction": 0.1,
    }
    codes = {issue.code for issue in validate_dataset(document).issues}
    assert "unobservable_must_be_unknown" in codes
    assert "absence_without_visibility" in codes


def test_visible_unknown_is_review_warning_not_error():
    document = valid_dataset()
    document["annotations"][0]["attributes"]["hi_vis"] = {
        "state": "unknown",
        "visibility": "visible",
        "visible_fraction": 0.95,
    }
    result = validate_dataset(document)
    assert result.valid
    assert "reviewable_unknown" in {issue.code for issue in result.issues}


def test_estimated_distance_cannot_enter_verified_25m_kpi():
    document = valid_dataset()
    attributes = document["annotations"][0]["attributes"]
    attributes["distance_source"] = "estimated"
    result = validate_dataset(document)
    assert not result.valid
    assert "unverified_distance_kpi" in {issue.code for issue in result.issues}


def test_camera_and_provenance_leakage_is_rejected():
    document = valid_dataset()
    document["videos"].append(
        {
            **document["videos"][0],
            "id": "video-2",
            "camera_id": "camera-1-segment-2",
            "provenance_group_id": "capture-2",
        }
    )
    document["images"].append(
        {
            **document["images"][0],
            "id": 2,
            "video_id": "video-2",
            "frame_index": 1,
            "split": "test",
            "sha256": "c" * 64,
        }
    )
    result = validate_dataset(document)
    assert "camera_group_split_leakage" in {issue.code for issue in result.issues}


def test_split_uses_transitive_camera_and_provenance_components():
    document = valid_dataset()
    document["annotations"] = []
    document["videos"] = [
        {**document["videos"][0], "id": "v1", "camera_group_id": "cam-a", "provenance_group_id": "p1"},
        {**document["videos"][0], "id": "v2", "camera_group_id": "cam-b", "provenance_group_id": "p1"},
        {**document["videos"][0], "id": "v3", "camera_group_id": "cam-b", "provenance_group_id": "p2"},
        {**document["videos"][0], "id": "v4", "camera_group_id": "cam-c", "provenance_group_id": "p3"},
    ]
    document["images"] = [
        {**valid_dataset()["images"][0], "id": index, "video_id": video_id, "frame_index": 0, "split": "unassigned", "sha256": str(index) * 64}
        for index, video_id in enumerate(("v1", "v2", "v3", "v4"), start=1)
    ]
    first = assign_splits(copy.deepcopy(document), train=0.5, val=0.25, test=0.25, seed="stable")
    second = assign_splits(copy.deepcopy(document), train=0.5, val=0.25, test=0.25, seed="stable")
    assert [image["split"] for image in first["images"]] == [image["split"] for image in second["images"]]
    assert len({image["split"] for image in first["images"][:3]}) == 1
    assert validate_dataset(first).valid


def test_sampler_rejects_blur_and_near_duplicates():
    candidates = [
        FrameMetric(0, 0.0, 10.0, 0, (1.0, 0.0)),
        FrameMetric(1, 40.0, 100.0, 0, (1.0, 0.0)),
        FrameMetric(2, 80.0, 100.0, 1, (0.98, 0.02)),
        FrameMetric(3, 120.0, 100.0, (1 << 63) - 1, (0.0, 1.0)),
    ]
    decisions = select_diverse_frames(candidates, min_blur=80, duplicate_hamming=6, duplicate_histogram_distance=0.08)
    assert [decision.reason for decision in decisions] == ["blur", "accepted", "near_duplicate", "accepted"]


def test_event_scoring_and_verified_25m_hook():
    truth = [
        {"id": "g1", "camera_id": "c1", "type": "no_helmet", "start_ms": 1000, "end_ms": 3000, "distance_m": 20, "distance_source": "calibrated"},
        {"id": "g2", "camera_id": "c1", "type": "no_hi_vis", "start_ms": 5000, "end_ms": 7000},
    ]
    predictions = [
        {"id": "p1", "camera_id": "c1", "type": "no_helmet", "start_ms": 1200, "end_ms": 2800, "confidence": 0.9},
        {"id": "p2", "camera_id": "c1", "type": "no_helmet", "start_ms": 8000, "end_ms": 9000, "confidence": 0.8},
    ]
    report = score_events(truth, predictions, confidence=0.25, temporal_iou_threshold=0.5)
    assert report["overall"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert report["verified_at_or_below_25m"]["recall"] == 1.0


def test_cli_qa_writes_machine_readable_report(tmp_path):
    pytest.importorskip("jsonschema")
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "qa.json"
    dataset_path.write_text(json.dumps(valid_dataset()), encoding="utf-8")
    assert main(["qa", "--input", str(dataset_path), "--output", str(report_path)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True
