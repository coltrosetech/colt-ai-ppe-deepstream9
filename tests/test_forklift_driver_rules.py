from __future__ import annotations

import copy
import json

import pytest

from content.forklift_driver_rules import (
    ForkliftDriverConfig,
    ForkliftDriverRuleEngine,
    ForkliftDriverRuleError,
    associate_people_to_vehicles,
    augment_frame_record,
    augment_jsonl_streams,
    augment_person_ppe_with_tracked_vehicles,
    merge_raw_vehicle_detector_fallback,
    person_center_inside_vehicle,
    person_intersection_over_area,
)


def person(track_id: int, bbox: list[float]) -> dict[str, object]:
    return {
        "track_id": track_id,
        "bbox_xyxy": bbox,
        "person_id": f"KISI-{track_id:03d}",
    }


def vehicle(
    track_id: int,
    bbox: list[float],
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "bbox_xyxy": bbox,
        "confidence": confidence,
        "class_name": "forklift_candidate",
    }


def immediate_config(**overrides: object) -> ForkliftDriverConfig:
    values: dict[str, object] = {
        "vehicle_confidence": 0.35,
        "person_ioa": 0.55,
        "require_center_inside": True,
        "enter_frames": 1,
        "exit_frames": 1,
        "ttl_frames": 30,
        "suppress_ppe": True,
        "suppress_walkway": True,
    }
    values.update(overrides)
    return ForkliftDriverConfig(**values)


def context(result: dict, track_id: int) -> dict:
    return next(
        row["vehicle_context"]
        for row in result["persons"]
        if row["track_id"] == track_id
    )


def test_default_policy_matches_runtime_contract() -> None:
    config = ForkliftDriverConfig()
    assert config.vehicle_confidence == pytest.approx(0.35)
    assert config.person_ioa == pytest.approx(0.55)
    assert config.require_center_inside is True
    assert config.enter_frames == 4
    assert config.exit_frames == 8
    assert config.ttl_frames == 30
    assert config.suppress_ppe is True
    assert config.suppress_walkway is True


def test_center_and_person_ioa_gate_activate_without_removing_bbox() -> None:
    engine = ForkliftDriverRuleEngine(immediate_config())
    source_person = person(10, [40, 30, 60, 70])
    result = engine.process_frame(
        frame_index=0,
        persons=[source_person],
        vehicles=[vehicle(100, [20, 10, 90, 90])],
    )

    assert person_intersection_over_area(
        (40, 30, 60, 70),
        (20, 10, 90, 90),
    ) == pytest.approx(1.0)
    assert person_center_inside_vehicle(
        (40, 30, 60, 70),
        (20, 10, 90, 90),
    )
    assert result["persons"][0]["bbox_xyxy"] == [40, 30, 60, 70]
    assert source_person.get("vehicle_context") is None
    assert context(result, 10) == {
        "raw_match": True,
        "suppression_active": True,
        "role": "forklift_operator",
        "vehicle_track_id": 100,
        "person_ioa": 1.0,
        "ppe_alert_eligible": False,
        "walkway_alert_eligible": False,
        "suppression_scopes": ["ppe", "walkway"],
        "enter_streak": 1,
        "exit_streak": 0,
    }
    assert result["transition_events"][0]["kind"] == "started"
    assert result["transition_events"][0]["reason"] == "enter_hysteresis_met"


def test_pedestrian_beside_forklift_is_not_suppressed() -> None:
    engine = ForkliftDriverRuleEngine(immediate_config())
    result = engine.process_frame(
        frame_index=0,
        persons=[person(1, [80, 20, 120, 90])],
        vehicles=[vehicle(7, [0, 0, 100, 100])],
    )

    # Only half the person area intersects and the center lies on the vehicle
    # boundary. The IoA threshold prevents a beside-vehicle false association.
    value = context(result, 1)
    assert value["raw_match"] is False
    assert value["suppression_active"] is False
    assert value["role"] is None
    assert value["ppe_alert_eligible"] is True
    assert value["walkway_alert_eligible"] is True


def test_one_vehicle_can_have_only_one_occupant() -> None:
    config = immediate_config(person_ioa=0.4)
    matches = associate_people_to_vehicles(
        [
            person(20, [20, 20, 70, 90]),
            person(10, [50, 20, 110, 90]),
        ],
        [vehicle(500, [0, 0, 100, 100])],
        config,
    )

    assert list(matches) == [20]
    assert matches[20]["person_ioa"] == pytest.approx(1.0)


def test_enter_exit_hysteresis_and_missing_person_is_unknown() -> None:
    engine = ForkliftDriverRuleEngine(
        immediate_config(enter_frames=4, exit_frames=3, ttl_frames=10)
    )
    tracked_person = person(5, [20, 20, 50, 80])
    tracked_vehicle = vehicle(50, [0, 0, 100, 100])

    for frame_index in range(3):
        result = engine.process_frame(
            frame_index=frame_index,
            persons=[tracked_person],
            vehicles=[tracked_vehicle],
        )
        assert context(result, 5)["suppression_active"] is False
        assert result["transition_events"] == []

    entered = engine.process_frame(
        frame_index=3,
        persons=[tracked_person],
        vehicles=[tracked_vehicle],
    )
    assert context(entered, 5)["suppression_active"] is True
    assert entered["transition_events"][0]["kind"] == "started"

    # An absent person is unknown, not a visible unmatched observation.
    missing_person = engine.process_frame(
        frame_index=4,
        persons=[],
        vehicles=[],
    )
    assert missing_person["transition_events"] == []
    assert missing_person["active_operator_track_ids"] == [5]

    first_miss = engine.process_frame(
        frame_index=5,
        persons=[tracked_person],
        vehicles=[],
    )
    second_miss = engine.process_frame(
        frame_index=6,
        persons=[tracked_person],
        vehicles=[],
    )
    assert context(first_miss, 5)["suppression_active"] is True
    assert context(second_miss, 5)["suppression_active"] is True
    assert context(second_miss, 5)["vehicle_track_id"] == 50

    ended = engine.process_frame(
        frame_index=7,
        persons=[tracked_person],
        vehicles=[],
    )
    assert context(ended, 5)["suppression_active"] is False
    assert context(ended, 5)["vehicle_track_id"] is None
    assert ended["transition_events"][0]["kind"] == "ended"
    assert ended["transition_events"][0]["reason"] == "exit_hysteresis_met"


def test_low_confidence_vehicle_is_not_a_raw_match() -> None:
    engine = ForkliftDriverRuleEngine(immediate_config())
    result = engine.process_frame(
        frame_index=0,
        persons=[person(1, [20, 20, 50, 80])],
        vehicles=[vehicle(2, [0, 0, 100, 100], confidence=0.349)],
    )
    assert context(result, 1)["raw_match"] is False
    assert result["raw_associations"] == []


def test_greedy_tie_break_is_deterministic_across_input_order() -> None:
    config = immediate_config()
    forward = associate_people_to_vehicles(
        [
            person(9, [20, 20, 50, 80]),
            person(3, [20, 20, 50, 80]),
        ],
        [vehicle(99, [0, 0, 100, 100])],
        config,
    )
    reverse = associate_people_to_vehicles(
        [
            person(3, [20, 20, 50, 80]),
            person(9, [20, 20, 50, 80]),
        ],
        [vehicle(99, [0, 0, 100, 100])],
        config,
    )
    assert forward == reverse
    assert list(forward) == [3]


def test_suppression_scopes_are_independently_configurable() -> None:
    engine = ForkliftDriverRuleEngine(
        immediate_config(suppress_ppe=False, suppress_walkway=True)
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[person(1, [20, 20, 50, 80])],
        vehicles=[vehicle(2, [0, 0, 100, 100])],
    )
    value = context(result, 1)
    assert value["suppression_active"] is True
    assert value["ppe_alert_eligible"] is True
    assert value["walkway_alert_eligible"] is False
    assert value["suppression_scopes"] == ["walkway"]
    assert result["transition_events"][0]["suppression_scopes"] == ["walkway"]


def test_active_track_expires_after_ttl_without_raising_on_empty_frames() -> None:
    engine = ForkliftDriverRuleEngine(immediate_config(ttl_frames=2))
    engine.process_frame(
        frame_index=0,
        persons=[person(1, [20, 20, 50, 80])],
        vehicles=[vehicle(2, [0, 0, 100, 100])],
    )
    engine.process_frame(frame_index=1, persons=[], vehicles=[])
    engine.process_frame(frame_index=2, persons=[], vehicles=[])
    expired = engine.process_frame(frame_index=3, persons=[], vehicles=[])
    assert expired["active_operator_track_ids"] == []
    assert expired["transition_events"][0]["reason"] == "track_expired"


def test_frame_helper_preserves_unrelated_fields_and_inputs() -> None:
    people = {
        "frame_index": 4,
        "sequence_id": "S04",
        "persons": [person(1, [20, 20, 50, 80])],
    }
    vehicles = {
        "frame_index": 4,
        "vehicles": [vehicle(2, [0, 0, 100, 100])],
    }
    before_people = copy.deepcopy(people)
    before_vehicles = copy.deepcopy(vehicles)
    output = augment_frame_record(
        people,
        vehicles,
        ForkliftDriverRuleEngine(immediate_config()),
    )

    assert people == before_people
    assert vehicles == before_vehicles
    assert output["sequence_id"] == "S04"
    assert output["persons"][0]["vehicle_context"]["suppression_active"] is True
    assert output["vehicle_context"]["active_operator_track_ids"] == [1]


def test_jsonl_helper_is_frame_aligned_and_atomic(tmp_path) -> None:
    person_path = tmp_path / "persons.jsonl"
    vehicle_path = tmp_path / "vehicles.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    person_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "persons": [person(1, [20, 20, 50, 80])],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vehicle_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "vehicles": [vehicle(2, [0, 0, 100, 100])],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = augment_jsonl_streams(
        person_path,
        vehicle_path,
        output_path,
        config=immediate_config(),
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["frames"] == 1
    assert result["transitions"] == 1
    assert row["persons"][0]["vehicle_context"]["role"] == (
        "forklift_operator"
    )


def test_invalid_bbox_and_duplicate_track_ids_fail_closed() -> None:
    engine = ForkliftDriverRuleEngine(immediate_config())
    with pytest.raises(ForkliftDriverRuleError, match="positive width"):
        engine.process_frame(
            frame_index=0,
            persons=[person(1, [10, 10, 10, 20])],
            vehicles=[],
        )
    with pytest.raises(ForkliftDriverRuleError, match="duplicate person"):
        ForkliftDriverRuleEngine(immediate_config()).process_frame(
            frame_index=0,
            persons=[
                person(1, [10, 10, 20, 20]),
                person(1, [30, 30, 40, 40]),
            ],
            vehicles=[],
        )


def test_normalized_xywh_runtime_boxes_are_supported() -> None:
    matches = associate_people_to_vehicles(
        [
            {
                "track_id": 3,
                "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
            }
        ],
        [
            {
                "track_id": 9,
                "bbox_norm_xywh": [0.1, 0.1, 0.7, 0.8],
                "confidence": 0.8,
            }
        ],
    )
    assert matches[3]["vehicle_track_id"] == 9
    assert matches[3]["person_ioa"] == pytest.approx(1.0)


def test_production_stream_helper_suppresses_effective_ppe_and_marks_zone(
    tmp_path,
) -> None:
    person_path = tmp_path / "person-ppe.jsonl"
    tracked_path = tmp_path / "tracked.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    person_rows = []
    tracked_rows = []
    for frame_index in range(2):
        person_rows.append(
            {
                "frame_index": frame_index,
                "image_width": 100,
                "image_height": 100,
                "persons": [
                    {
                        "track_id": 1,
                        "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
                        "alarms": ["no_helmet"],
                        "alarm": True,
                        "overall_state": "noncompliant",
                    }
                ],
                "alarm_events": [],
            }
        )
        tracked_rows.append(
            {
                "frame_index": frame_index,
                "image_width": 100,
                "image_height": 100,
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "track_id": 1,
                        "confidence": 0.9,
                        "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
                    },
                    {
                        "class_id": 7,
                        "class_name": "forklift_candidate",
                        "track_id": 9,
                        "confidence": 0.8,
                        "bbox_norm_xywh": [0.1, 0.1, 0.7, 0.8],
                    },
                ],
            }
        )
    person_path.write_text(
        "".join(json.dumps(row) + "\n" for row in person_rows),
        encoding="utf-8",
    )
    tracked_path.write_text(
        "".join(json.dumps(row) + "\n" for row in tracked_rows),
        encoding="utf-8",
    )

    receipt = augment_person_ppe_with_tracked_vehicles(
        person_path,
        tracked_path,
        output_path,
        config=immediate_config(),
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert receipt["forklift_candidate_observations"] == 2
    assert receipt["suppressed_person_frames"] == 2
    assert rows[0]["persons"][0]["alarms"] == []
    assert rows[0]["persons"][0]["raw_alarms"] == ["no_helmet"]
    assert rows[0]["persons"][0]["zone_alert_eligible"] is False
    assert rows[0]["persons"][0]["overall_state"] == (
        "suppressed_forklift_operator"
    )
    assert rows[0]["alarm_events"] == []
    assert rows[0]["vehicle_context"]["active_operator_track_ids"] == [1]


def test_production_stream_helper_keeps_alarm_open_across_missing_track(
    tmp_path,
) -> None:
    person_path = tmp_path / "person-ppe.jsonl"
    tracked_path = tmp_path / "tracked.jsonl"
    output_path = tmp_path / "augmented.jsonl"
    person_rows = [
        {
            "frame_index": 0,
            "persons": [
                {
                    "track_id": 1,
                    "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
                    "alarms": ["no_helmet"],
                    "alarm": True,
                    "overall_state": "noncompliant",
                }
            ],
            "alarm_events": [
                {
                    "kind": "started",
                    "type": "no_helmet",
                    "track_id": 1,
                    "frame_index": 0,
                }
            ],
        },
        {
            "frame_index": 1,
            "persons": [],
            "alarm_events": [],
        },
        {
            "frame_index": 2,
            "persons": [
                {
                    "track_id": 1,
                    "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
                    "alarms": ["no_helmet"],
                    "alarm": True,
                    "overall_state": "noncompliant",
                }
            ],
            "alarm_events": [],
        },
        {
            "frame_index": 3,
            "persons": [],
            "alarm_events": [
                {
                    "kind": "ended",
                    "type": "no_helmet",
                    "track_id": 1,
                    "frame_index": 3,
                }
            ],
        },
    ]
    tracked_rows = [
        {
            "frame_index": frame_index,
            "detections": [],
        }
        for frame_index in range(4)
    ]
    person_path.write_text(
        "".join(json.dumps(row) + "\n" for row in person_rows),
        encoding="utf-8",
    )
    tracked_path.write_text(
        "".join(json.dumps(row) + "\n" for row in tracked_rows),
        encoding="utf-8",
    )

    receipt = augment_person_ppe_with_tracked_vehicles(
        person_path,
        tracked_path,
        output_path,
        config=immediate_config(),
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [row["alarm_events"] for row in rows] == [
        [
            {
                "kind": "started",
                "type": "no_helmet",
                "track_id": 1,
                "frame_index": 0,
                "reason": "effective_alarm_state_changed",
            }
        ],
        [],
        [],
        [
            {
                "kind": "ended",
                "type": "no_helmet",
                "track_id": 1,
                "frame_index": 3,
                "reason": "effective_alarm_state_changed",
            }
        ],
    ]
    assert receipt["effective_open_alarms_at_end"] == []


def test_engine_retains_qualified_vehicle_across_tracker_score_dip() -> None:
    engine = ForkliftDriverRuleEngine(
        ForkliftDriverConfig(
            vehicle_confidence=0.35,
            person_ioa=0.5,
            enter_frames=1,
            exit_frames=1,
            vehicle_confidence_ttl_frames=2,
        )
    )
    person = {
        "track_id": 1,
        "bbox_norm_xywh": [0.3, 0.2, 0.2, 0.5],
    }

    rows = []
    for frame_index, confidence in enumerate((0.8, 0.1, 0.1, 0.1)):
        rows.append(
            engine.process_frame(
                frame_index=frame_index,
                persons=[person],
                vehicles=[
                    {
                        "track_id": 9,
                        "bbox_norm_xywh": [0.1, 0.1, 0.7, 0.8],
                        "confidence": confidence,
                    }
                ],
            )
        )

    assert [
        row["persons"][0]["vehicle_context"]["suppression_active"]
        for row in rows
    ] == [True, True, True, False]


def test_engine_allows_only_one_active_operator_per_vehicle() -> None:
    engine = ForkliftDriverRuleEngine(
        ForkliftDriverConfig(
            vehicle_confidence=0.35,
            person_ioa=0.3,
            enter_frames=1,
            exit_frames=8,
        )
    )
    vehicle = {
        "track_id": 9,
        "bbox_norm_xywh": [0.1, 0.1, 0.7, 0.8],
        "confidence": 0.8,
    }
    first = {
        "track_id": 1,
        "bbox_norm_xywh": [0.2, 0.2, 0.2, 0.5],
    }
    second = {
        "track_id": 2,
        "bbox_norm_xywh": [0.5, 0.2, 0.2, 0.5],
    }

    opened = engine.process_frame(
        frame_index=0,
        persons=[first],
        vehicles=[vehicle],
    )
    handed_off = engine.process_frame(
        frame_index=1,
        persons=[first, second],
        vehicles=[vehicle],
    )

    assert opened["active_operator_track_ids"] == [1]
    assert handed_off["active_operator_track_ids"] in ([1], [2])
    assert len(handed_off["active_operator_track_ids"]) == 1


def test_raw_detector_fallback_keeps_native_vehicle_id_across_tracker_gap(
    tmp_path,
) -> None:
    tracked = tmp_path / "tracked.jsonl"
    raw_kitti = tmp_path / "kitti"
    output = tmp_path / "merged.jsonl"
    raw_kitti.mkdir()
    tracked.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "schema_version": "deepsafe.person-detections/v1",
                    "sequence_id": "clip",
                    "frame_index": 0,
                    "image_width": 200,
                    "image_height": 100,
                    "model_id": "person-vehicle",
                    "detections": [
                        {
                            "class_id": 7,
                            "class_name": "forklift_candidate",
                            "detector_class_name": "truck",
                            "confidence": 0.2,
                            "track_id": 9,
                            "bbox_norm_xywh": [0.2, 0.2, 0.4, 0.6],
                        }
                    ],
                },
                {
                    "schema_version": "deepsafe.person-detections/v1",
                    "sequence_id": "clip",
                    "frame_index": 1,
                    "image_width": 200,
                    "image_height": 100,
                    "model_id": "person-vehicle",
                    "detections": [],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for frame_index, confidence in ((0, 0.8), (1, 0.6)):
        (raw_kitti / f"00_000_{frame_index:06d}.txt").write_text(
            (
                "truck 0.0 0 0.0 40 20 120 80 "
                f"0 0 0 0 0 0 0 {confidence}\n"
            ),
            encoding="utf-8",
        )

    receipt = merge_raw_vehicle_detector_fallback(
        tracked,
        raw_kitti,
        output,
        coordinate_width=200,
        coordinate_height=100,
    )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert receipt["raw_detector_observations"] == 2
    assert receipt["native_enriched_observations"] == 1
    assert receipt["fallback_observations"] == 1
    assert rows[0]["detections"][0]["track_id"] == 9
    assert rows[0]["detections"][0]["confidence"] == 0.8
    assert rows[1]["detections"][0]["track_id"] == 9
    assert rows[1]["detections"][0]["confidence"] == 0.6
