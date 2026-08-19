import json

import numpy as np
import pytest

from content.person_ppe_fusion import (
    PersonPpeFusion,
    PersonPpeFusionConfig,
    PersonPpeFusionError,
    build_parser,
    fuse_prediction_streams,
)
from content.ppe_video import draw_ppe_frame


def detection(
    canonical_class: str,
    box: list[float],
    confidence: float = 0.9,
    *,
    track_id: int | None = None,
) -> dict:
    value = {
        "canonical_class": canonical_class,
        "bbox_norm_xywh": box,
        "confidence": confidence,
    }
    if track_id is not None:
        value["track_id"] = track_id
    return value


PERSON = [0.1, 0.1, 0.2, 0.7]
HELMET = [0.15, 0.12, 0.1, 0.1]
VEST = [0.13, 0.31, 0.14, 0.25]


def test_missing_visible_helmet_opens_one_person_alarm_after_hysteresis():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=3,
            inferred_absence_enter_frames=3,
            alarm_clear_frames=2,
            infer_no_helmet_from_missing=True,
            infer_no_hi_vis_from_missing=False,
        )
    )
    first = fusion.process_frame(
        frame_index=0,
        detections=[detection("person", PERSON, track_id=41)],
    )
    second = fusion.process_frame(
        frame_index=1,
        detections=[detection("person", PERSON, track_id=41)],
    )
    third = fusion.process_frame(
        frame_index=2,
        detections=[detection("person", PERSON, track_id=41)],
    )

    assert first["persons"][0]["helmet"]["state"] == "unknown"
    assert second["persons"][0]["alarm"] is False
    person = third["persons"][0]
    assert person["track_id"] == 41
    assert person["person_id"] == "KISI-041"
    assert person["helmet"]["evidence"] == "absent"
    assert person["helmet"]["link_status"] == "inferred_visible_missing"
    assert (
        person["helmet"]["evidence_source"]
        == "visible_zone_no_positive_observation"
    )
    assert person["helmet"]["state"] == "absent"
    assert person["alarms"] == ["no_helmet"]
    assert third["alarm_events"] == [
        {
            "kind": "started",
            "type": "no_helmet",
            "track_id": 41,
            "frame_index": 2,
        }
    ]


def test_present_helmet_clears_same_track_alarm_only_after_clear_streak():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=2,
            inferred_absence_enter_frames=2,
            alarm_clear_frames=2,
            infer_no_helmet_from_missing=True,
            infer_no_hi_vis_from_missing=False,
        )
    )
    for frame_index in range(2):
        result = fusion.process_frame(
            frame_index=frame_index,
            detections=[detection("person", PERSON, track_id=7)],
        )
    assert result["persons"][0]["helmet"]["state"] == "absent"

    one_present = fusion.process_frame(
        frame_index=2,
        detections=[
            detection("person", PERSON, track_id=7),
            detection("helmet", HELMET),
        ],
    )
    cleared = fusion.process_frame(
        frame_index=3,
        detections=[
            detection("person", PERSON, track_id=7),
            detection("helmet", HELMET),
        ],
    )
    assert one_present["persons"][0]["helmet"]["state"] == "absent"
    assert one_present["alarm_events"] == []
    assert cleared["persons"][0]["helmet"]["state"] == "present"
    assert cleared["persons"][0]["alarm"] is False
    assert cleared["alarm_events"][0]["kind"] == "ended"
    assert cleared["alarm_events"][0]["type"] == "no_helmet"


def test_long_unknown_hold_prevents_label_flicker_but_explicit_absence_is_fast():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=3,
            alarm_clear_frames=1,
            unknown_after_frames=90,
            infer_no_helmet_from_missing=False,
            infer_no_hi_vis_from_missing=False,
        )
    )
    qualified = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=12),
            detection("helmet", HELMET),
        ],
    )
    assert qualified["persons"][0]["helmet"]["evidence"] == "present"

    for frame_index in range(1, 90):
        held = fusion.process_frame(
            frame_index=frame_index,
            detections=[detection("person", PERSON, track_id=12)],
        )
    assert held["persons"][0]["helmet"]["state"] == "present"
    assert held["persons"][0]["helmet"]["evidence"] == "unknown"
    assert (
        held["persons"][0]["helmet"]["link_status"]
        == "unknown_no_observation"
    )
    assert held["persons"][0]["helmet"]["observations"] == []
    unknown = fusion.process_frame(
        frame_index=90,
        detections=[detection("person", PERSON, track_id=12)],
    )
    assert unknown["persons"][0]["helmet"]["evidence"] == "unknown"

    for frame_index in range(91, 94):
        explicit_absence = fusion.process_frame(
            frame_index=frame_index,
            detections=[
                detection("person", PERSON, track_id=12),
                detection("no_helmet", HELMET),
            ],
        )
    assert explicit_absence["persons"][0]["helmet"]["evidence"] == "absent"
    assert explicit_absence["persons"][0]["alarms"] == ["no_helmet"]
    assert explicit_absence["alarm_events"] == [
        {
            "kind": "started",
            "type": "no_helmet",
            "track_id": 12,
            "frame_index": 93,
        }
    ]


def test_selector_policy_style_missing_helmet_needs_eight_reliable_frames():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            inferred_absence_enter_frames=8,
            infer_no_helmet_from_missing=True,
            infer_no_hi_vis_from_missing=False,
        )
    )
    for frame_index in range(7):
        result = fusion.process_frame(
            frame_index=frame_index,
            detections=[detection("person", PERSON, track_id=31)],
        )
        assert result["persons"][0]["helmet"]["state"] == "unknown"
        assert result["alarm_events"] == []

    qualified = fusion.process_frame(
        frame_index=7,
        detections=[detection("person", PERSON, track_id=31)],
    )
    helmet = qualified["persons"][0]["helmet"]
    assert helmet["state"] == "absent"
    assert helmet["evidence"] == "absent"
    assert helmet["evidence_source"] == "visible_zone_no_positive_observation"
    assert helmet["link_status"] == "inferred_visible_missing"
    assert qualified["alarm_events"] == [
        {
            "kind": "started",
            "type": "no_helmet",
            "track_id": 31,
            "frame_index": 7,
        }
    ]


def test_verified_present_helmet_survives_p01_style_79_frame_detector_gap():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_clear_frames=1,
            inferred_absence_enter_frames=8,
            verified_present_missing_grace_frames=90,
            unknown_after_frames=8,
            infer_no_helmet_from_missing=True,
            infer_no_hi_vis_from_missing=False,
        )
    )
    verified = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=22),
            detection("helmet", HELMET),
        ],
    )
    assert verified["persons"][0]["helmet"]["state"] == "present"

    for frame_index in range(1, 80):
        held = fusion.process_frame(
            frame_index=frame_index,
            detections=[detection("person", PERSON, track_id=22)],
        )
        helmet = held["persons"][0]["helmet"]
        assert helmet["state"] == "present"
        assert helmet["evidence"] == "unknown"
        assert (
            helmet["evidence_source"]
            == "verified_present_grace_no_observation"
        )
        assert helmet["link_status"] == "unknown_no_observation"
        assert held["alarm_events"] == []

    recovered = fusion.process_frame(
        frame_index=80,
        detections=[
            detection("person", PERSON, track_id=22),
            detection("helmet", HELMET),
        ],
    )
    assert recovered["persons"][0]["helmet"]["state"] == "present"
    assert recovered["persons"][0]["helmet"]["evidence"] == "present"
    assert recovered["persons"][0]["helmet"]["link_status"] == "matched"
    assert recovered["alarm_events"] == []


def test_verified_present_grace_then_inferred_threshold_opens_alarm():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_clear_frames=1,
            verified_present_missing_grace_frames=3,
            inferred_absence_enter_frames=2,
            unknown_after_frames=1,
            infer_no_helmet_from_missing=True,
        )
    )
    fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=23),
            detection("helmet", HELMET),
        ],
    )
    for frame_index in range(1, 5):
        result = fusion.process_frame(
            frame_index=frame_index,
            detections=[detection("person", PERSON, track_id=23)],
        )
    assert result["persons"][0]["helmet"]["state"] == "unknown"
    assert result["persons"][0]["helmet"]["evidence"] == "absent"
    assert result["alarm_events"] == []

    alarm = fusion.process_frame(
        frame_index=5,
        detections=[detection("person", PERSON, track_id=23)],
    )
    assert alarm["persons"][0]["helmet"]["state"] == "absent"
    assert alarm["persons"][0]["helmet"]["link_status"] == (
        "inferred_visible_missing"
    )
    assert alarm["alarm_events"] == [
        {
            "kind": "started",
            "type": "no_helmet",
            "track_id": 23,
            "frame_index": 5,
        }
    ]


def test_missing_helmet_inference_requires_reliable_uncropped_head_area():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            inferred_absence_enter_frames=1,
            infer_no_helmet_from_missing=True,
        )
    )
    too_narrow_for_head_inference = [0.1, 0.1, 0.021, 0.07]
    top_cropped = [0.4, 0.0, 0.2, 0.7]
    result = fusion.process_frame(
        frame_index=0,
        detections=[
            detection(
                "person",
                too_narrow_for_head_inference,
                track_id=1,
            ),
            detection("person", top_cropped, track_id=2),
        ],
    )
    assert [person["helmet"]["state"] for person in result["persons"]] == [
        "unknown",
        "unknown",
    ]
    assert result["alarm_events"] == []


def test_absent_to_unknown_emits_one_ended_transition():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=1,
            unknown_after_frames=2,
            infer_no_helmet_from_missing=False,
        )
    )
    started = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=9),
            detection("no_helmet", HELMET),
        ],
    )
    held = fusion.process_frame(
        frame_index=1,
        detections=[detection("person", PERSON, track_id=9)],
    )
    ended = fusion.process_frame(
        frame_index=2,
        detections=[detection("person", PERSON, track_id=9)],
    )

    assert started["alarm_events"][0]["kind"] == "started"
    assert held["persons"][0]["helmet"]["state"] == "absent"
    assert held["persons"][0]["helmet"]["evidence"] == "unknown"
    assert held["alarm_events"] == []
    assert ended["persons"][0]["helmet"]["state"] == "unknown"
    assert ended["alarm_events"] == [
        {
            "kind": "ended",
            "type": "no_helmet",
            "track_id": 9,
            "frame_index": 2,
        }
    ]


def test_expired_alarm_track_emits_one_ended_transition():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=1,
            track_ttl_frames=1,
            infer_no_helmet_from_missing=False,
        )
    )
    fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=14),
            detection("no_helmet", HELMET),
        ],
    )
    missing = fusion.process_frame(frame_index=1, detections=[])
    expired = fusion.process_frame(frame_index=2, detections=[])
    after_expiry = fusion.process_frame(frame_index=3, detections=[])

    assert missing["alarm_events"] == []
    assert expired["alarm_events"] == [
        {
            "kind": "ended",
            "type": "no_helmet",
            "track_id": 14,
            "frame_index": 2,
        }
    ]
    assert after_expiry["alarm_events"] == []


def test_head_and_torso_observations_are_attached_to_correct_person_boxes():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(alarm_enter_frames=1, alarm_clear_frames=1)
    )
    left = [0.05, 0.1, 0.25, 0.75]
    right = [0.60, 0.1, 0.25, 0.75]
    result = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", left, track_id=10),
            detection("person", right, track_id=20),
            detection("helmet", [0.12, 0.12, 0.1, 0.1]),
            detection("hi_vis", [0.09, 0.32, 0.18, 0.28]),
            detection("no_helmet", [0.67, 0.12, 0.1, 0.1]),
            detection("no_hi_vis", [0.64, 0.32, 0.18, 0.28]),
        ],
    )
    by_id = {person["track_id"]: person for person in result["persons"]}
    assert by_id[10]["overall_state"] == "compliant"
    assert by_id[10]["helmet"]["observations"][0]["canonical_class"] == "helmet"
    assert by_id[20]["alarms"] == ["no_helmet", "no_hi_vis"]
    assert by_id[20]["helmet"]["observations"][0]["canonical_class"] == "no_helmet"


def test_iou_tracker_keeps_ephemeral_id_without_external_tracker_metadata():
    fusion = PersonPpeFusion()
    first = fusion.process_frame(
        frame_index=0, detections=[detection("person", PERSON)]
    )
    moved = [0.108, 0.103, 0.2, 0.7]
    second = fusion.process_frame(
        frame_index=1, detections=[detection("person", moved)]
    )
    assert first["persons"][0]["track_id"] == second["persons"][0]["track_id"]
    assert fusion.active_track_count == 1


def test_low_confidence_and_tiny_person_fragments_never_create_tracks():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            minimum_person_confidence=0.25,
            minimum_person_width=0.02,
            minimum_person_height=0.05,
        )
    )
    result = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, 0.91),
            detection("person", [0.91, 0.20, 0.05, 0.40], 0.249),
            detection("person", [0.95, 0.20, 0.019, 0.40], 0.95),
            detection("person", [0.80, 0.20, 0.10, 0.049], 0.95),
        ],
    )
    assert len(result["persons"]) == 1
    assert result["persons"][0]["confidence"] == 0.91
    assert result["filtered_person_detection_count"] == 3
    assert fusion.active_track_count == 1


def test_close_conflicting_present_and_absent_observations_fail_to_unknown():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(
            alarm_enter_frames=1,
            alarm_clear_frames=1,
            infer_no_hi_vis_from_missing=False,
        )
    )
    result = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=1),
            detection("helmet", HELMET, 0.80),
            detection("no_helmet", HELMET, 0.75),
        ],
    )
    helmet = result["persons"][0]["helmet"]
    assert helmet["state"] == "unknown"
    assert helmet["evidence_source"] == "conflicting_observations"
    assert result["persons"][0]["alarm"] is False


def test_fusion_person_schema_is_consumed_directly_by_person_overlay():
    fusion = PersonPpeFusion(
        PersonPpeFusionConfig(alarm_enter_frames=1, alarm_clear_frames=1)
    )
    fused = fusion.process_frame(
        frame_index=0,
        detections=[
            detection("person", PERSON, track_id=55),
            detection("no_helmet", HELMET),
            detection("hi_vis", VEST),
        ],
    )
    rendered, counts = draw_ppe_frame(
        np.zeros((360, 640, 3), dtype=np.uint8),
        detections=[],
        persons=fused["persons"],
        camera_label="CAM-PPE",
    )
    assert rendered.any()
    assert counts == {"person": 1, "no_helmet": 1, "hi_vis": 1}


def test_fuse_separate_person_and_ppe_streams_writes_person_schema(tmp_path):
    people_path = tmp_path / "people.jsonl"
    ppe_path = tmp_path / "ppe.jsonl"
    output = tmp_path / "fused.jsonl"
    people_rows = []
    ppe_rows = []
    for frame_index in range(3):
        people_rows.append(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "fixture",
                "frame_index": frame_index,
                "image_width": 640,
                "image_height": 360,
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox_norm_xywh": PERSON,
                    },
                    {
                        "class_id": 7,
                        "class_name": "forklift_candidate",
                        "confidence": 0.8,
                        "track_id": 90,
                        "bbox_norm_xywh": [0.02, 0.02, 0.8, 0.9],
                    }
                ],
            }
        )
        ppe_rows.append(
            {
                "schema_version": "colt-ai.ppe-detections/v1",
                "sequence_id": "fixture",
                "frame_index": frame_index,
                "image_width": 640,
                "image_height": 360,
                "detections": [detection("helmet", HELMET)],
            }
        )
    people_path.write_text(
        "".join(json.dumps(row) + "\n" for row in people_rows),
        encoding="utf-8",
    )
    ppe_path.write_text(
        "".join(json.dumps(row) + "\n" for row in ppe_rows),
        encoding="utf-8",
    )
    receipt = fuse_prediction_streams(
        ppe_predictions=ppe_path,
        person_predictions=people_path,
        output=output,
        config=PersonPpeFusionConfig(
            alarm_enter_frames=2,
            alarm_clear_frames=2,
            infer_no_hi_vis_from_missing=False,
        ),
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert receipt["frames"] == 3
    assert receipt["person_observations"] == 3
    assert rows[-1]["persons"][0]["helmet"]["state"] == "present"
    assert rows[-1]["persons"][0]["person_id"] == "KISI-001"


def test_cli_exposes_unknown_hold_for_content_rendering():
    args = build_parser().parse_args(
        [
            "--ppe-predictions",
            "ppe.jsonl",
            "--output",
            "fused.jsonl",
            "--unknown-after-frames",
            "90",
        ]
    )
    assert args.unknown_after_frames == 90


def test_rejects_nonmonotonic_frames_and_invalid_boxes():
    fusion = PersonPpeFusion()
    fusion.process_frame(
        frame_index=0, detections=[detection("person", PERSON)]
    )
    with pytest.raises(PersonPpeFusionError, match="strictly increasing"):
        fusion.process_frame(frame_index=0, detections=[])
    fresh = PersonPpeFusion()
    with pytest.raises(PersonPpeFusionError, match="outside"):
        fresh.process_frame(
            frame_index=0,
            detections=[detection("person", [0.9, 0.1, 0.2, 0.2])],
        )
