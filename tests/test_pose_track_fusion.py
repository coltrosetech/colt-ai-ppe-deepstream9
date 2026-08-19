import copy
import json

import pytest

from content.pose_track_fusion import (
    PERSON_SCHEMA_VERSION,
    POSE_SCHEMA_VERSION,
    PoseTrackFusionConfig,
    PoseTrackFusionError,
    fuse_jsonl,
    fuse_records,
    load_person_detection_records,
)


def _person_detection(
    track_id: int | None,
    bbox: list[float],
) -> dict:
    detection = {
        "class_id": 0,
        "class_name": "person",
        "detector_class_name": "person",
        "confidence": 0.91,
        "bbox_norm_xywh": bbox,
        "tracker_confidence": 0.88,
    }
    if track_id is not None:
        detection["track_id"] = track_id
    return detection


def _person_frame(
    frame_index: int,
    detections: list[dict],
) -> dict:
    return {
        "schema_version": PERSON_SCHEMA_VERSION,
        "sequence_id": "camera-01",
        "frame_index": frame_index,
        "image_width": 960,
        "image_height": 640,
        "timestamp_ns": frame_index * 40_000_000,
        "source_uri": "file:///fixture.mp4",
        "model_id": "person-engine",
        "detections": detections,
    }


def _pose(
    bbox: list[float],
    *,
    marker_x: float,
) -> dict:
    return {
        "score": 0.94,
        "bbox_norm_xywh": bbox,
        "keypoints": [
            {
                "name": "nose",
                "x_norm": marker_x,
                "y_norm": 0.20,
                "confidence": 0.92,
                "visible": True,
            },
            {
                "name": "left_ankle",
                "x_norm": marker_x,
                "y_norm": 0.70,
                "confidence": 0.87,
                "visible": True,
            },
        ],
    }


def _pose_frame(
    frame_index: int,
    poses: list[dict],
) -> dict:
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "camera_label": "CAM-01",
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 25.0,
        "image_width": 960,
        "image_height": 640,
        "poses": poses,
    }


def _pose_by_track(result) -> dict[int, dict]:
    return {
        int(detection["track_id"]): detection["pose"]
        for detection in result.records[0]["detections"]
    }


def test_auto_contiguous_offset_fuses_and_preserves_person_contract():
    people = [
        _person_frame(
            frame_index,
            [_person_detection(7, [0.10, 0.10, 0.25, 0.60])],
        )
        for frame_index in (0, 1)
    ]
    poses = [
        _pose_frame(
            frame_index,
            [_pose([0.10, 0.10, 0.25, 0.60], marker_x=0.20)],
        )
        for frame_index in (100, 101)
    ]
    original_people = copy.deepcopy(people)
    original_poses = copy.deepcopy(poses)

    result = fuse_records(people, poses)

    assert result.pose_frame_offset == 100
    assert result.matched_people == 2
    assert result.unknown_people == 0
    assert result.unmatched_poses == 0
    assert people == original_people
    assert poses == original_poses
    first = result.records[0]
    assert first["schema_version"] == PERSON_SCHEMA_VERSION
    assert first["sequence_id"] == "camera-01"
    assert first["model_id"] == "person-engine"
    assert first["detections"][0]["track_id"] == 7
    assert first["detections"][0]["detector_class_name"] == "person"
    assert first["detections"][0]["tracker_confidence"] == 0.88
    assert first["detections"][0]["pose"]["association_status"] == "matched"
    assert first["detections"][0]["pose"]["association_iou"] == 1.0
    assert result.receipt()["schema_version"] == (
        "colt-ai.pose-track-fusion/v1"
    )


def test_explicit_offset_uses_pose_index_equals_person_index_plus_offset():
    people = [_person_frame(50, [_person_detection(9, [0.2, 0.1, 0.2, 0.7])])]
    poses = [
        _pose_frame(
            12,
            [_pose([0.2, 0.1, 0.2, 0.7], marker_x=0.3)],
        )
    ]

    result = fuse_records(
        people,
        poses,
        config=PoseTrackFusionConfig(pose_frame_offset=-38),
    )

    assert result.pose_frame_offset == -38
    assert result.matched_people == 1


def test_highest_iou_greedy_is_one_to_one_and_input_order_independent():
    detections = [
        _person_detection(20, [0.60, 0.10, 0.20, 0.70]),
        _person_detection(10, [0.10, 0.10, 0.20, 0.70]),
    ]
    poses = [
        _pose([0.10, 0.10, 0.20, 0.70], marker_x=0.15),
        _pose([0.60, 0.10, 0.20, 0.70], marker_x=0.65),
    ]

    normal = fuse_records(
        [_person_frame(0, detections)],
        [_pose_frame(0, poses)],
    )
    reversed_inputs = fuse_records(
        [_person_frame(0, list(reversed(detections)))],
        [_pose_frame(0, list(reversed(poses)))],
    )

    normal_map = _pose_by_track(normal)
    reversed_map = _pose_by_track(reversed_inputs)
    assert {
        track_id: pose["keypoints"][0]["x_norm"]
        for track_id, pose in normal_map.items()
    } == {
        10: 0.15,
        20: 0.65,
    }
    assert {
        track_id: pose["keypoints"][0]["x_norm"]
        for track_id, pose in reversed_map.items()
    } == {
        10: 0.15,
        20: 0.65,
    }
    assert normal.matched_people == reversed_inputs.matched_people == 2
    assert normal.unmatched_poses == reversed_inputs.unmatched_poses == 0


def test_close_person_candidates_make_pose_association_unknown():
    people = [
        _person_frame(
            0,
            [_person_detection(3, [0.10, 0.10, 0.40, 0.40])],
        )
    ]
    poses = [
        _pose_frame(
            0,
            [
                _pose([0.10, 0.10, 0.40, 0.40], marker_x=0.20),
                _pose([0.11, 0.10, 0.40, 0.40], marker_x=0.21),
            ],
        )
    ]

    result = fuse_records(
        people,
        poses,
        config=PoseTrackFusionConfig(ambiguity_margin=0.05),
    )

    attached = result.records[0]["detections"][0]["pose"]
    assert attached == {
        "association_status": "unknown",
        "association_reason": "ambiguous_pose_association",
    }
    assert result.matched_people == 0
    assert result.unknown_people == 1
    assert result.unmatched_poses == 2


def test_pose_shared_by_close_tracks_is_ambiguous_for_both_people():
    people = [
        _person_frame(
            0,
            [
                _person_detection(1, [0.10, 0.10, 0.40, 0.40]),
                _person_detection(2, [0.11, 0.10, 0.40, 0.40]),
            ],
        )
    ]
    poses = [
        _pose_frame(
            0,
            [_pose([0.10, 0.10, 0.40, 0.40], marker_x=0.20)],
        )
    ]

    result = fuse_records(people, poses)

    assert {
        detection["pose"]["association_reason"]
        for detection in result.records[0]["detections"]
    } == {"ambiguous_pose_association"}
    assert result.matched_people == 0
    assert result.unknown_people == 2
    assert result.unmatched_poses == 1


def test_below_minimum_iou_and_missing_track_are_explicit_unknowns():
    people = [
        _person_frame(
            0,
            [
                _person_detection(1, [0.05, 0.05, 0.15, 0.30]),
                _person_detection(None, [0.60, 0.10, 0.20, 0.60]),
            ],
        )
    ]
    poses = [
        _pose_frame(
            0,
            [_pose([0.75, 0.10, 0.20, 0.60], marker_x=0.80)],
        )
    ]

    result = fuse_records(people, poses)

    assert [
        detection["pose"]["association_reason"]
        for detection in result.records[0]["detections"]
    ] == ["no_pose_above_minimum_iou", "missing_track_id"]
    assert result.matched_people == 0
    assert result.unknown_people == 2
    assert result.unmatched_poses == 1


def test_auto_offset_rejects_non_contiguous_streams():
    people = [
        _person_frame(index, [_person_detection(1, [0.1, 0.1, 0.2, 0.7])])
        for index in (0, 2)
    ]
    poses = [
        _pose_frame(
            index,
            [_pose([0.1, 0.1, 0.2, 0.7], marker_x=0.2)],
        )
        for index in (10, 11)
    ]

    with pytest.raises(
        PoseTrackFusionError,
        match="requires contiguous frame indices",
    ):
        fuse_records(people, poses)


def test_fuse_jsonl_publishes_reloadable_augmented_person_stream(tmp_path):
    person_path = tmp_path / "people.jsonl"
    pose_path = tmp_path / "poses.jsonl"
    output_path = tmp_path / "fused.jsonl"
    person_record = _person_frame(
        0,
        [_person_detection(4, [0.2, 0.1, 0.2, 0.7])],
    )
    pose_record = _pose_frame(
        0,
        [_pose([0.2, 0.1, 0.2, 0.7], marker_x=0.3)],
    )
    person_path.write_text(json.dumps(person_record) + "\n", encoding="utf-8")
    pose_path.write_text(json.dumps(pose_record) + "\n", encoding="utf-8")

    result = fuse_jsonl(
        person_path,
        pose_path,
        output_path=output_path,
    )
    loaded = load_person_detection_records(output_path)

    assert result.matched_people == 1
    assert loaded == list(result.records)
    with pytest.raises(PoseTrackFusionError, match="output already exists"):
        fuse_jsonl(person_path, pose_path, output_path=output_path)
