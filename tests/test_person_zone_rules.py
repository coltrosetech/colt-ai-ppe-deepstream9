from __future__ import annotations

import pytest

from content.person_zone_rules import (
    COCO17_KEYPOINT_NAMES,
    FENCE_EVENT_SCHEMA_VERSION,
    FenceBoundaryRuleConfig,
    PersonZoneRuleEngine,
    PersonZoneRuleError,
    PoseKeypointRuleConfig,
    ZoneArea,
    ZoneRuleConfig,
    bbox_bottom_center,
    point_in_polygon,
)


LEFT_RESTRICTED = ZoneArea(
    area_id="restricted-left",
    area_type="restricted_zone",
    name="Sol Kısıtlı Alan",
    points=((0.0, 0.5), (0.3, 0.5), (0.3, 1.0), (0.0, 1.0)),
)
RIGHT_RESTRICTED = ZoneArea(
    area_id="restricted-right",
    area_type="restricted_zone",
    name="Sağ Kısıtlı Alan",
    points=((0.7, 0.5), (1.0, 0.5), (1.0, 1.0), (0.7, 1.0)),
)
LEFT_WALKWAY = ZoneArea(
    area_id="walkway-left",
    area_type="safe_walkway",
    name="Sol Yürüyüş Yolu",
    points=((0.0, 0.5), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0)),
)
RIGHT_WALKWAY = ZoneArea(
    area_id="walkway-right",
    area_type="safe_walkway",
    name="Sağ Yürüyüş Yolu",
    points=((0.6, 0.5), (1.0, 0.5), (1.0, 1.0), (0.6, 1.0)),
)
POSE_BODY_4 = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


def person(
    track_id: int,
    *,
    footpoint: tuple[float, float] | None = None,
    bbox: list[float] | None = None,
    with_ppe: bool = False,
    pose_data: dict[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {"track_id": track_id}
    if footpoint is not None:
        value["footpoint_norm_xy"] = list(footpoint)
    else:
        value["bbox_norm_xywh"] = bbox or [0.4, 0.2, 0.2, 0.6]
    if with_ppe:
        value["helmet"] = {"state": "present"}
        value["hi_vis"] = {"state": "absent"}
        value["person_id"] = f"KISI-{track_id:03d}"
    if pose_data is not None:
        value["pose"] = pose_data
    return value


def pose_payload(
    points: list[tuple[str, float, float, float, bool]],
    *,
    score: float = 0.9,
    association_status: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "score": score,
        "bbox_norm_xywh": [0.1, 0.1, 0.8, 0.8],
        "keypoints": [
            {
                "name": name,
                "x_norm": x,
                "y_norm": y,
                "confidence": confidence,
                "visible": visible,
            }
            for name, x, y, confidence, visible in points
        ],
    }
    if association_status is not None:
        value["association_status"] = association_status
    return value


def body_pose(
    *,
    inside_names: set[str],
    confidence: float = 0.9,
) -> dict[str, object]:
    return pose_payload(
        [
            (
                name,
                0.2 if name in inside_names else 0.5,
                0.8,
                confidence,
                True,
            )
            for name in POSE_BODY_4
        ]
    )


def pose_rule(
    *,
    names: tuple[str, ...] = POSE_BODY_4,
    minimum_confidence: float = 0.3,
    minimum_visible: int = 2,
    minimum_inside_ratio: float = 0.5,
) -> PoseKeypointRuleConfig:
    return PoseKeypointRuleConfig(
        keypoint_names=names,
        minimum_confidence=minimum_confidence,
        minimum_visible=minimum_visible,
        minimum_inside_ratio=minimum_inside_ratio,
    )


def immediate_config(
    *,
    ttl: int = 2,
    restricted_pose_rule: PoseKeypointRuleConfig | None = None,
) -> ZoneRuleConfig:
    return ZoneRuleConfig(
        restricted_enter_frames=1,
        restricted_exit_frames=1,
        walkway_enter_frames=1,
        walkway_exit_frames=1,
        track_ttl_frames=ttl,
        restricted_pose_rule=restricted_pose_rule,
    )


def strict_fence_rule(
    *,
    breach_enter_frames: int = 3,
    breach_exit_frames: int = 2,
    climb_enter_frames: int = 2,
    climb_exit_frames: int = 2,
) -> FenceBoundaryRuleConfig:
    return FenceBoundaryRuleConfig(
        boundary_start=(0.5, 0.1),
        boundary_end=(0.5, 0.95),
        forbidden_side="right",
        contact_band=0.025,
        minimum_confidence=0.3,
        minimum_core_visible=2,
        breach_enter_frames=breach_enter_frames,
        breach_exit_frames=breach_exit_frames,
        approach_minimum_count=1,
        wrist_contact_required=1,
        hip_rise_ratio=0.08,
        raised_knee_ratio=0.10,
        climb_enter_frames=climb_enter_frames,
        climb_exit_frames=climb_exit_frames,
        history_window_frames=20,
    )


def strict_fence_config(
    **kwargs: int,
) -> ZoneRuleConfig:
    return ZoneRuleConfig(
        restricted_enter_frames=1,
        restricted_exit_frames=1,
        walkway_enter_frames=1,
        walkway_exit_frames=1,
        track_ttl_frames=3,
        restricted_fence_rule=strict_fence_rule(**kwargs),
    )


def fence_pose(
    *,
    hip_x: float,
    hip_y: float = 0.62,
    wrist_x: float = 0.40,
    wrist_y: float = 0.42,
    knee_y: float = 0.80,
    include_hips: bool = True,
    include_wrists: bool = True,
    include_knees: bool = True,
    association_status: str | None = None,
) -> dict[str, object]:
    points: list[tuple[str, float, float, float, bool]] = [
        ("left_shoulder", hip_x, max(0.0, hip_y - 0.20), 0.95, True),
        ("right_shoulder", hip_x, max(0.0, hip_y - 0.20), 0.95, True),
    ]
    if include_hips:
        points.extend(
            (
                ("left_hip", hip_x - 0.01, hip_y, 0.95, True),
                ("right_hip", hip_x + 0.01, hip_y, 0.95, True),
            )
        )
    if include_wrists:
        points.extend(
            (
                ("left_wrist", wrist_x, wrist_y, 0.95, True),
                ("right_wrist", wrist_x, wrist_y + 0.02, 0.95, True),
            )
        )
    if include_knees:
        points.extend(
            (
                ("left_knee", hip_x - 0.01, knee_y, 0.95, True),
                ("right_knee", hip_x + 0.01, knee_y + 0.01, 0.95, True),
            )
        )
    return pose_payload(points, association_status=association_status)


def safety(result: dict, track_id: int) -> dict:
    return next(
        item["zone_safety"]
        for item in result["persons"]
        if item["track_id"] == track_id
    )


def test_bbox_bottom_center_and_polygon_boundaries_are_inclusive() -> None:
    assert bbox_bottom_center([0.2, 0.1, 0.4, 0.7]) == pytest.approx(
        (0.4, 0.8)
    )
    square = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
    assert point_in_polygon((0.2, 0.5), square) is True
    assert point_in_polygon((0.5, 0.5), square) is True
    assert point_in_polygon((0.19, 0.5), square) is False


def test_restricted_polygons_are_one_any_inside_union() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED, RIGHT_RESTRICTED],
        immediate_config(),
    )
    first = engine.process_frame(
        frame_index=0,
        persons=[person(7, footpoint=(0.8, 0.8))],
    )
    first_safety = safety(first, 7)

    assert first_safety["violation_active"] is True
    assert first_safety["contributing_area_ids"] == ["restricted-right"]
    assert first["safety_events"] == [
        {
            "schema_version": "colt-ai.person-safety-transition/v1",
            "kind": "started",
            "type": "restricted_area_intrusion",
            "track_id": 7,
            "frame_index": 0,
            "footpoint_norm_xy": [0.8, 0.8],
            "contributing_area_ids": ["restricted-right"],
            "reason": "debounce_qualified",
        }
    ]

    # Moving directly between two restricted polygons retains one aggregate
    # violation and does not emit another start transition.
    second = engine.process_frame(
        frame_index=1,
        persons=[person(7, footpoint=(0.2, 0.8))],
    )
    assert safety(second, 7)["contributing_area_ids"] == ["restricted-left"]
    assert second["safety_events"] == []


def test_walkway_polygons_are_allowed_union_and_boundary_is_safe() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_WALKWAY, RIGHT_WALKWAY],
        immediate_config(),
    )
    left_boundary = engine.process_frame(
        frame_index=0,
        persons=[person(1, footpoint=(0.4, 0.7))],
    )
    right_inside = engine.process_frame(
        frame_index=1,
        persons=[person(1, footpoint=(0.8, 0.7))],
    )
    outside_all = engine.process_frame(
        frame_index=2,
        persons=[person(1, footpoint=(0.5, 0.7))],
    )

    assert safety(left_boundary, 1)["violation_active"] is False
    assert safety(right_inside, 1)["violation_active"] is False
    outside = safety(outside_all, 1)
    assert outside["violation_active"] is True
    assert outside["contributing_area_ids"] == [
        "walkway-left",
        "walkway-right",
    ]
    assert outside_all["safety_events"][0]["type"] == "walkway_violation"


def test_restricted_boundary_is_an_intrusion() -> None:
    engine = PersonZoneRuleEngine([LEFT_RESTRICTED], immediate_config())
    result = engine.process_frame(
        frame_index=0,
        persons=[person(5, footpoint=(0.3, 0.7))],
    )
    assert safety(result, 5)["violation_active"] is True
    assert safety(result, 5)["contributing_area_ids"] == ["restricted-left"]


def test_two_people_have_independent_temporal_state() -> None:
    engine = PersonZoneRuleEngine([LEFT_RESTRICTED], immediate_config())
    first = engine.process_frame(
        frame_index=0,
        persons=[
            person(10, footpoint=(0.2, 0.8)),
            person(20, footpoint=(0.5, 0.8)),
        ],
    )
    second = engine.process_frame(
        frame_index=1,
        persons=[
            person(10, footpoint=(0.5, 0.8)),
            person(20, footpoint=(0.2, 0.8)),
        ],
    )

    assert safety(first, 10)["violation_active"] is True
    assert safety(first, 20)["violation_active"] is False
    assert [(event["kind"], event["track_id"]) for event in first["safety_events"]] == [
        ("started", 10)
    ]
    assert safety(second, 10)["violation_active"] is False
    assert safety(second, 20)["violation_active"] is True
    assert [(event["kind"], event["track_id"]) for event in second["safety_events"]] == [
        ("ended", 10),
        ("started", 20),
    ]


def test_debounce_suppresses_boundary_jitter_and_per_frame_spam() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        ZoneRuleConfig(
            restricted_enter_frames=2,
            restricted_exit_frames=3,
            walkway_enter_frames=2,
            walkway_exit_frames=2,
            track_ttl_frames=3,
        ),
    )
    positions = [
        (0.31, 0.8),  # safe
        (0.29, 0.8),  # one raw intrusion
        (0.31, 0.8),  # jitter resets entry
        (0.29, 0.8),
        (0.28, 0.8),  # qualified start
        (0.27, 0.8),  # held, no repeated event
        (0.31, 0.8),
        (0.29, 0.8),  # jitter resets exit
        (0.31, 0.8),
        (0.32, 0.8),
        (0.33, 0.8),  # qualified end
    ]
    results = [
        engine.process_frame(
            frame_index=index,
            persons=[person(3, footpoint=position)],
        )
        for index, position in enumerate(positions)
    ]
    transitions = [
        (index, event["kind"])
        for index, result in enumerate(results)
        for event in result["safety_events"]
    ]
    assert transitions == [(4, "started"), (10, "ended")]
    assert safety(results[8], 3)["violation_active"] is True
    assert safety(results[10], 3)["violation_active"] is False


def test_track_expiry_closes_open_violation_exactly_once() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        immediate_config(ttl=1),
    )
    started = engine.process_frame(
        frame_index=0,
        persons=[person(14, footpoint=(0.2, 0.8))],
    )
    missing = engine.process_frame(frame_index=1, persons=[])
    expired = engine.process_frame(frame_index=2, persons=[])
    after_expiry = engine.process_frame(frame_index=3, persons=[])

    assert started["safety_events"][0]["kind"] == "started"
    assert missing["safety_events"] == []
    assert expired["safety_events"] == [
        {
            "schema_version": "colt-ai.person-safety-transition/v1",
            "kind": "ended",
            "type": "restricted_area_intrusion",
            "track_id": 14,
            "frame_index": 2,
            "footpoint_norm_xy": [0.2, 0.8],
            "contributing_area_ids": ["restricted-left"],
            "reason": "track_expired",
        }
    ]
    assert after_expiry["safety_events"] == []
    assert engine.active_track_count == 0


def test_ppe_people_remain_full_frame_while_walkway_state_is_augmented() -> None:
    engine = PersonZoneRuleEngine([LEFT_WALKWAY], immediate_config())
    inside = person(31, footpoint=(0.2, 0.8), with_ppe=True)
    outside = person(32, footpoint=(0.8, 0.8), with_ppe=True)
    result = engine.process_frame(
        frame_index=0,
        persons=[inside, outside],
    )

    assert result["person_visibility_policy"] == "full_frame_no_zone_filter"
    assert [item["track_id"] for item in result["persons"]] == [31, 32]
    by_id = {item["track_id"]: item for item in result["persons"]}
    assert by_id[31]["helmet"] == {"state": "present"}
    assert by_id[32]["hi_vis"] == {"state": "absent"}
    assert by_id[31]["zone_safety"]["violation_active"] is False
    assert by_id[32]["zone_safety"]["violation_active"] is True
    # The input mappings are not mutated.
    assert "zone_safety" not in inside
    assert "zone_safety" not in outside


def test_simultaneous_restricted_and_walkway_violations_are_separate() -> None:
    restricted_middle = ZoneArea(
        area_id="machine-zone",
        area_type="restricted_zone",
        points=((0.4, 0.5), (0.6, 0.5), (0.6, 1.0), (0.4, 1.0)),
    )
    engine = PersonZoneRuleEngine(
        [restricted_middle, LEFT_WALKWAY],
        immediate_config(),
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[person(9, footpoint=(0.5, 0.8))],
    )

    zone_safety = safety(result, 9)
    assert zone_safety["active_violation_types"] == [
        "restricted_area_intrusion",
        "walkway_violation",
    ]
    assert [event["type"] for event in result["safety_events"]] == [
        "restricted_area_intrusion",
        "walkway_violation",
    ]


def test_mapping_form_and_invalid_identity_are_validated() -> None:
    engine = PersonZoneRuleEngine(
        [
            {
                "area_id": "walkway",
                "type": "safe_walkway",
                "points": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
            }
        ],
        immediate_config(),
    )
    assert engine.process_frame(
        frame_index=0,
        persons=[person(1, bbox=[0.2, 0.1, 0.2, 0.7])],
    )["persons"][0]["zone_safety"]["footpoint_norm_xy"] == [0.3, 0.8]

    with pytest.raises(PersonZoneRuleError, match="track_id"):
        engine.process_frame(
            frame_index=1,
            persons=[{"track_id": "1", "footpoint_norm_xy": [0.3, 0.8]}],
        )


def test_queue_roi_type_mapping_is_accepted() -> None:
    engine = PersonZoneRuleEngine(
        [
            {
                "roi_id": "safe-route",
                "name": "Güvenli Yol",
                "roi_type": "safe_walkway",
                "points": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.9, "y": 0.1},
                    {"x": 0.9, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
            }
        ],
        immediate_config(),
    )

    result = engine.process_frame(
        frame_index=0,
        persons=[person(4, footpoint=(0.5, 0.5))],
    )

    assert safety(result, 4)["violation_active"] is False


def test_pose_rule_config_accepts_only_unique_coco17_names_and_valid_gates() -> None:
    config = PoseKeypointRuleConfig(
        keypoint_names=["left_shoulder", "right_shoulder"],  # type: ignore[arg-type]
        minimum_confidence=0.25,
        minimum_visible=2,
        minimum_inside_ratio=0.5,
    )

    assert config.keypoint_names == ("left_shoulder", "right_shoulder")
    assert set(config.keypoint_names) <= set(COCO17_KEYPOINT_NAMES)
    with pytest.raises(PersonZoneRuleError, match="unique"):
        pose_rule(names=("left_shoulder", "left_shoulder"))
    with pytest.raises(PersonZoneRuleError, match="unsupported COCO17"):
        pose_rule(names=("left_shoulder", "pelvis"))
    with pytest.raises(PersonZoneRuleError, match="minimum_visible"):
        pose_rule(names=("left_shoulder",), minimum_visible=2)
    with pytest.raises(PersonZoneRuleError, match="minimum_confidence"):
        pose_rule(minimum_confidence=float("nan"))
    with pytest.raises(PersonZoneRuleError, match="minimum_inside_ratio"):
        pose_rule(minimum_inside_ratio=0.0)
    with pytest.raises(PersonZoneRuleError, match="restricted_pose_rule"):
        PersonZoneRuleEngine(
            [LEFT_RESTRICTED],
            ZoneRuleConfig(restricted_pose_rule={"invalid": True}),  # type: ignore[arg-type]
        )


def test_pose_ratio_exact_threshold_enters_and_below_threshold_exits() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        immediate_config(
            restricted_pose_rule=pose_rule(minimum_visible=4),
        ),
    )
    exact = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                71,
                pose_data=body_pose(
                    inside_names={"left_shoulder", "right_shoulder"}
                ),
            )
        ],
    )
    below = engine.process_frame(
        frame_index=1,
        persons=[
            person(
                71,
                pose_data=body_pose(inside_names={"left_shoulder"}),
            )
        ],
    )

    exact_safety = safety(exact, 71)
    evidence = exact_safety["anchor_evidence"]
    assert exact_safety["footpoint_norm_xy"] == [0.5, 0.8]
    assert exact_safety["violation_active"] is True
    assert evidence["observation_valid"] is True
    assert evidence["selected_count"] == 4
    assert evidence["visible_count"] == 4
    assert evidence["inside_count"] == 2
    assert evidence["inside_ratio"] == 0.5
    assert evidence["ratio_denominator"] == "configured_keypoints"
    assert exact["safety_events"][0]["contributing_area_ids"] == [
        "restricted-left"
    ]

    below_safety = safety(below, 71)
    assert below_safety["violation_active"] is False
    assert below_safety["anchor_evidence"]["inside_ratio"] == 0.25
    assert below_safety["anchor_evidence"]["contributing_area_ids"] == [
        "restricted-left"
    ]
    assert below_safety["rules"]["restricted_zone"][
        "raw_contributing_area_ids"
    ] == []
    assert [(event["kind"], event["type"]) for event in below["safety_events"]] == [
        ("ended", "restricted_area_intrusion")
    ]


def test_pose_boundary_is_inside_and_confidence_is_inclusive() -> None:
    names = ("left_shoulder", "right_shoulder")
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        immediate_config(
            restricted_pose_rule=pose_rule(
                names=names,
                minimum_confidence=0.5,
                minimum_visible=1,
                minimum_inside_ratio=0.5,
            )
        ),
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                72,
                pose_data=pose_payload(
                    [
                        ("left_shoulder", 0.3, 0.7, 0.5, True),
                        ("right_shoulder", 0.2, 0.7, 0.49, True),
                    ]
                ),
            )
        ],
    )

    evidence = safety(result, 72)["anchor_evidence"]
    assert evidence["visible_count"] == 1
    assert evidence["inside_count"] == 1
    assert evidence["inside_ratio"] == 0.5
    assert evidence["inside_keypoint_names"] == ["left_shoulder"]
    assert safety(result, 72)["violation_active"] is True


def test_insufficient_visible_pose_is_unknown_not_safe() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        immediate_config(
            restricted_pose_rule=pose_rule(
                names=("left_shoulder", "right_shoulder"),
                minimum_confidence=0.5,
                minimum_visible=2,
                minimum_inside_ratio=0.5,
            )
        ),
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                73,
                pose_data=pose_payload(
                    [
                        ("left_shoulder", 0.2, 0.7, 0.9, True),
                        ("right_shoulder", 0.2, 0.7, 0.49, True),
                    ]
                ),
            )
        ],
    )

    pose_safety = safety(result, 73)
    assert pose_safety["violation_active"] is False
    assert pose_safety["anchor_evidence"]["observation_valid"] is False
    assert (
        pose_safety["anchor_evidence"]["reason"]
        == "insufficient_visible_keypoints"
    )
    assert pose_safety["anchor_evidence"]["inside_ratio"] == 0.5
    assert pose_safety["rules"]["restricted_zone"]["raw_violation"] is None
    assert pose_safety["rules"]["restricted_zone"]["streak"] == 0
    assert result["safety_events"] == []


def test_unknown_pose_pauses_entry_and_exit_streaks_without_clearing_active() -> None:
    rule = pose_rule(
        names=("left_shoulder", "right_shoulder"),
        minimum_visible=2,
        minimum_inside_ratio=0.5,
    )
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        ZoneRuleConfig(
            restricted_enter_frames=2,
            restricted_exit_frames=2,
            walkway_enter_frames=1,
            walkway_exit_frames=1,
            track_ttl_frames=3,
            restricted_pose_rule=rule,
        ),
    )
    inside = pose_payload(
        [
            ("left_shoulder", 0.2, 0.8, 0.9, True),
            ("right_shoulder", 0.5, 0.8, 0.9, True),
        ]
    )
    outside = pose_payload(
        [
            ("left_shoulder", 0.5, 0.8, 0.9, True),
            ("right_shoulder", 0.5, 0.8, 0.9, True),
        ]
    )
    ambiguous = pose_payload(
        [
            ("left_shoulder", 0.2, 0.8, 0.9, True),
            ("right_shoulder", 0.5, 0.8, 0.9, True),
        ],
        association_status="ambiguous",
    )

    first_inside = engine.process_frame(
        frame_index=0,
        persons=[person(74, pose_data=inside)],
    )
    missing_pose = engine.process_frame(
        frame_index=1,
        persons=[person(74)],
    )
    entered = engine.process_frame(
        frame_index=2,
        persons=[person(74, pose_data=inside)],
    )
    first_outside = engine.process_frame(
        frame_index=3,
        persons=[person(74, pose_data=outside)],
    )
    ambiguous_pose = engine.process_frame(
        frame_index=4,
        persons=[person(74, pose_data=ambiguous)],
    )
    exited = engine.process_frame(
        frame_index=5,
        persons=[person(74, pose_data=outside)],
    )

    assert safety(first_inside, 74)["rules"]["restricted_zone"]["streak"] == 1
    assert safety(missing_pose, 74)["rules"]["restricted_zone"]["streak"] == 1
    assert safety(missing_pose, 74)["rules"]["restricted_zone"][
        "raw_violation"
    ] is None
    assert entered["safety_events"][0]["kind"] == "started"
    assert safety(first_outside, 74)["violation_active"] is True
    assert safety(first_outside, 74)["rules"]["restricted_zone"]["streak"] == 1
    assert safety(ambiguous_pose, 74)["violation_active"] is True
    assert safety(ambiguous_pose, 74)["rules"]["restricted_zone"]["streak"] == 1
    assert (
        safety(ambiguous_pose, 74)["anchor_evidence"]["reason"]
        == "pose_missing_or_ambiguous"
    )
    assert exited["safety_events"][0]["kind"] == "ended"
    assert safety(exited, 74)["violation_active"] is False


def test_pose_keypoints_use_multi_restricted_roi_union_once_per_joint() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED, RIGHT_RESTRICTED],
        immediate_config(
            restricted_pose_rule=pose_rule(minimum_visible=4),
        ),
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                75,
                pose_data=pose_payload(
                    [
                        ("left_shoulder", 0.2, 0.8, 0.9, True),
                        ("right_shoulder", 0.8, 0.8, 0.9, True),
                        ("left_hip", 0.5, 0.8, 0.9, True),
                        ("right_hip", 0.5, 0.8, 0.9, True),
                    ]
                ),
            )
        ],
    )

    evidence = safety(result, 75)["anchor_evidence"]
    assert evidence["inside_count"] == 2
    assert evidence["inside_ratio"] == 0.5
    assert evidence["contributing_area_ids"] == [
        "restricted-left",
        "restricted-right",
    ]
    assert result["safety_events"][0]["contributing_area_ids"] == [
        "restricted-left",
        "restricted-right",
    ]


def test_pose_rule_does_not_change_walkway_footpoint_semantics() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED, LEFT_WALKWAY],
        immediate_config(
            restricted_pose_rule=pose_rule(
                names=("left_shoulder", "right_shoulder"),
                minimum_visible=2,
                minimum_inside_ratio=1.0,
            )
        ),
    )
    result = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                76,
                footpoint=(0.8, 0.8),
                pose_data=pose_payload(
                    [
                        ("left_shoulder", 0.5, 0.8, 0.9, True),
                        ("right_shoulder", 0.5, 0.8, 0.9, True),
                    ]
                ),
            )
        ],
    )

    pose_safety = safety(result, 76)
    assert pose_safety["rules"]["restricted_zone"]["raw_violation"] is False
    assert pose_safety["rules"]["safe_walkway"]["raw_violation"] is True
    assert pose_safety["active_violation_types"] == ["walkway_violation"]
    assert result["safety_events"][0]["type"] == "walkway_violation"


def test_pose_violation_is_closed_once_by_track_ttl() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_RESTRICTED],
        immediate_config(
            ttl=1,
            restricted_pose_rule=pose_rule(
                names=("left_shoulder", "right_shoulder"),
                minimum_visible=2,
                minimum_inside_ratio=0.5,
            ),
        ),
    )
    started = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                77,
                pose_data=pose_payload(
                    [
                        ("left_shoulder", 0.2, 0.8, 0.9, True),
                        ("right_shoulder", 0.5, 0.8, 0.9, True),
                    ]
                ),
            )
        ],
    )
    missing = engine.process_frame(frame_index=1, persons=[])
    expired = engine.process_frame(frame_index=2, persons=[])
    after_expiry = engine.process_frame(frame_index=3, persons=[])

    assert started["safety_events"][0]["kind"] == "started"
    assert missing["safety_events"] == []
    assert expired["safety_events"] == [
        {
            "schema_version": "colt-ai.person-safety-transition/v1",
            "kind": "ended",
            "type": "restricted_area_intrusion",
            "track_id": 77,
            "frame_index": 2,
            "footpoint_norm_xy": [0.5, 0.8],
            "contributing_area_ids": ["restricted-left"],
            "reason": "track_expired",
        }
    ]
    assert after_expiry["safety_events"] == []


def test_external_alert_ineligibility_keeps_person_and_clears_walkway_alarm() -> None:
    engine = PersonZoneRuleEngine(
        [LEFT_WALKWAY],
        ZoneRuleConfig(
            walkway_enter_frames=1,
            walkway_exit_frames=2,
            track_ttl_frames=5,
        ),
    )
    violating = {
        "track_id": 8,
        "bbox_norm_xywh": [0.7, 0.1, 0.1, 0.5],
    }
    opened = engine.process_frame(frame_index=0, persons=[violating])
    assert safety(opened, 8)["violation_active"] is True

    suppressed = {
        **violating,
        "zone_alert_eligible": False,
        "vehicle_context": {"role": "forklift_operator"},
    }
    held = engine.process_frame(frame_index=1, persons=[suppressed])
    cleared = engine.process_frame(frame_index=2, persons=[suppressed])

    assert held["persons"][0]["vehicle_context"]["role"] == "forklift_operator"
    assert safety(held, 8)["violation_active"] is False
    assert safety(cleared, 8)["violation_active"] is False
    assert any(
        event["kind"] == "ended"
        and event["reason"] == "external_alert_suppressed"
        for event in held["safety_events"]
    )
    assert cleared["safety_events"] == []
    assert safety(cleared, 8)["rules"]["safe_walkway"]["alert_eligible"] is False


def test_external_alert_eligibility_requires_boolean() -> None:
    engine = PersonZoneRuleEngine([LEFT_RESTRICTED])
    with pytest.raises(PersonZoneRuleError, match="zone_alert_eligible"):
        engine.process_frame(
            frame_index=0,
            persons=[
                {
                    "track_id": 1,
                    "bbox_norm_xywh": [0.1, 0.1, 0.2, 0.2],
                    "zone_alert_eligible": "false",
                }
            ],
        )


def test_strict_fence_config_normalizes_mapping_and_validates_geometry() -> None:
    config = FenceBoundaryRuleConfig.from_mapping(
        {
            "boundary": [[0.5, 0.1], {"x": 0.5, "y": 0.9}],
            "forbidden_side": "right",
            "contact_band": 0.0,
            "minimum_core_visible": 2,
        }
    )

    assert config.boundary_points == ((0.5, 0.1), (0.5, 0.9))
    assert config.contact_band == 0.0
    with pytest.raises(PersonZoneRuleError, match="distinct"):
        FenceBoundaryRuleConfig(
            boundary_start=(0.5, 0.5),
            boundary_end=(0.5, 0.5),
            forbidden_side="right",
        )
    with pytest.raises(PersonZoneRuleError, match="left or right"):
        FenceBoundaryRuleConfig(
            boundary_start=(0.2, 0.2),
            boundary_end=(0.8, 0.8),
            forbidden_side="top",  # type: ignore[arg-type]
        )
    with pytest.raises(PersonZoneRuleError, match="restricted_fence_rule"):
        PersonZoneRuleEngine(
            [LEFT_RESTRICTED],
            ZoneRuleConfig(
                restricted_fence_rule={"boundary": []},  # type: ignore[arg-type]
            ),
        )


def test_strict_fence_partial_pose_overlap_is_only_noncritical_approach() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(breach_enter_frames=2),
    )
    # Both wrists and shoulders touch/cross the rendered fence band, while the
    # hip core remains firmly on the allowed side.
    result = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                101,
                pose_data=fence_pose(
                    hip_x=0.40,
                    wrist_x=0.505,
                ),
            )
        ],
    )
    zone = safety(result, 101)

    assert zone["fence_state"] == "approach"
    assert zone["violation_active"] is False
    assert zone["rules"]["restricted_zone"]["raw_violation"] is False
    assert zone["fence_evidence"]["contact_count"] >= 1
    assert zone["fence_evidence"]["core_on_forbidden_side"] is False
    assert result["safety_events"] == []
    assert result["fence_events"] == [
        {
            "schema_version": FENCE_EVENT_SCHEMA_VERSION,
            "kind": "started",
            "type": "fence_approach",
            "track_id": 101,
            "frame_index": 0,
            "from_state": "clear",
            "to_state": "approach",
            "reason": "fence_evidence_updated",
        }
    ]


def test_strict_fence_walking_parallel_never_becomes_breach_or_climb() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(),
    )
    results = [
        engine.process_frame(
            frame_index=index,
            persons=[
                person(
                    102,
                    pose_data=fence_pose(
                        hip_x=0.40,
                        hip_y=0.58 + index * 0.005,
                        wrist_x=0.40,
                        wrist_y=0.35 + index * 0.01,
                        knee_y=0.78 + index * 0.005,
                    ),
                )
            ],
        )
        for index in range(6)
    ]

    assert all(safety(result, 102)["fence_state"] == "clear" for result in results)
    assert all(result["safety_events"] == [] for result in results)
    assert all(result["fence_events"] == [] for result in results)


def test_strict_fence_reaching_without_body_lift_stays_approach() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(climb_enter_frames=1),
    )
    neutral = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                103,
                pose_data=fence_pose(
                    hip_x=0.40,
                    wrist_x=0.40,
                ),
            )
        ],
    )
    reaching = engine.process_frame(
        frame_index=1,
        persons=[
            person(
                103,
                pose_data=fence_pose(
                    hip_x=0.40,
                    wrist_x=0.50,
                ),
            )
        ],
    )
    evidence = safety(reaching, 103)["fence_evidence"]

    assert safety(neutral, 103)["fence_state"] == "clear"
    assert safety(reaching, 103)["fence_state"] == "approach"
    assert evidence["wrist_contact_count"] == 2
    assert evidence["hip_upward_displacement_ratio"] == 0.0
    assert evidence["climb_candidate"] is False
    assert reaching["safety_events"] == []


def test_strict_fence_true_climb_is_noncritical_until_core_crosses() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(
            breach_enter_frames=2,
            climb_enter_frames=2,
        ),
    )
    neutral = engine.process_frame(
        frame_index=0,
        persons=[
            person(
                104,
                pose_data=fence_pose(
                    hip_x=0.40,
                    hip_y=0.65,
                    wrist_x=0.40,
                    knee_y=0.84,
                ),
            )
        ],
    )
    first_lift = engine.process_frame(
        frame_index=1,
        persons=[
            person(
                104,
                pose_data=fence_pose(
                    hip_x=0.41,
                    hip_y=0.55,
                    wrist_x=0.50,
                    knee_y=0.68,
                ),
            )
        ],
    )
    qualified_climb = engine.process_frame(
        frame_index=2,
        persons=[
            person(
                104,
                pose_data=fence_pose(
                    hip_x=0.42,
                    hip_y=0.54,
                    wrist_x=0.50,
                    knee_y=0.66,
                ),
            )
        ],
    )
    evidence = safety(qualified_climb, 104)["fence_evidence"]

    assert safety(neutral, 104)["fence_state"] == "clear"
    assert safety(first_lift, 104)["fence_state"] == "approach"
    assert safety(qualified_climb, 104)["fence_state"] == "climbing"
    assert evidence["hip_upward_displacement_ratio"] > 0.08
    assert evidence["knee_upward_displacement_ratio"] > 0.10
    assert evidence["raised_knee"] is True
    assert evidence["climb_candidate"] is True
    assert evidence["climb_active"] is True
    assert neutral["safety_events"] == []
    assert first_lift["safety_events"] == []
    assert qualified_climb["safety_events"] == []
    assert [
        (event["kind"], event["type"])
        for event in qualified_climb["fence_events"]
    ] == [
        ("ended", "fence_approach"),
        ("started", "fence_climb_attempt"),
    ]


def test_strict_fence_history_is_scoped_to_same_nvdscf_track_id() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(climb_enter_frames=1),
    )
    engine.process_frame(
        frame_index=0,
        persons=[
            person(
                201,
                pose_data=fence_pose(
                    hip_x=0.40,
                    hip_y=0.65,
                    wrist_x=0.40,
                    knee_y=0.84,
                ),
            )
        ],
    )
    different_track = engine.process_frame(
        frame_index=1,
        persons=[
            person(
                202,
                pose_data=fence_pose(
                    hip_x=0.40,
                    hip_y=0.54,
                    wrist_x=0.50,
                    knee_y=0.66,
                ),
            )
        ],
    )

    evidence = safety(different_track, 202)["fence_evidence"]
    assert evidence["hip_baseline_y"] is None
    assert evidence["hip_upward_displacement_ratio"] == 0.0
    assert evidence["climb_candidate"] is False
    assert safety(different_track, 202)["fence_state"] == "approach"


def test_strict_fence_actual_core_crossing_is_inclusive_and_debounced() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(
            breach_enter_frames=3,
            breach_exit_frames=2,
            climb_enter_frames=1,
        ),
    )
    poses = [
        # Establish this NvDCF track's standing baseline.
        fence_pose(
            hip_x=0.45,
            hip_y=0.65,
            wrist_x=0.40,
            knee_y=0.84,
        ),
        # Qualify climbing while the core is still on the allowed side.
        fence_pose(
            hip_x=0.45,
            hip_y=0.54,
            wrist_x=0.50,
            knee_y=0.66,
        ),
        # The boundary itself is forbidden inclusively.  Three consecutive
        # climb-qualified core observations open the critical breach.
        fence_pose(
            hip_x=0.50,
            hip_y=0.53,
            wrist_x=0.50,
            knee_y=0.65,
        ),
        fence_pose(
            hip_x=0.54,
            hip_y=0.52,
            wrist_x=0.50,
            knee_y=0.64,
        ),
        fence_pose(
            hip_x=0.58,
            hip_y=0.51,
            wrist_x=0.50,
            knee_y=0.63,
        ),
        fence_pose(
            hip_x=0.45,
            hip_y=0.65,
            wrist_x=0.40,
            knee_y=0.84,
        ),
        fence_pose(
            hip_x=0.44,
            hip_y=0.65,
            wrist_x=0.40,
            knee_y=0.84,
        ),
    ]
    results = [
        engine.process_frame(
            frame_index=index,
            persons=[
                person(
                    105,
                    pose_data=pose_data,
                )
            ],
        )
        for index, pose_data in enumerate(poses)
    ]

    assert safety(results[2], 105)["fence_evidence"]["core_side"] == "boundary"
    assert safety(results[2], 105)["fence_evidence"][
        "core_on_forbidden_side"
    ] is True
    assert safety(results[2], 105)["fence_evidence"][
        "breach_entry_eligible"
    ] is True
    assert safety(results[3], 105)["violation_active"] is False
    assert safety(results[4], 105)["fence_state"] == "breach"
    assert [(event["kind"], event["reason"]) for event in results[4]["safety_events"]] == [
        ("started", "core_crossing_qualified")
    ]
    assert safety(results[5], 105)["fence_state"] == "breach"
    assert safety(results[6], 105)["fence_state"] == "clear"
    assert [(event["kind"], event["reason"]) for event in results[6]["safety_events"]] == [
        ("ended", "core_crossing_cleared")
    ]


def test_strict_fence_core_projection_without_climb_never_opens_breach() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(
            breach_enter_frames=2,
            climb_enter_frames=1,
        ),
    )
    results = [
        engine.process_frame(
            frame_index=index,
            persons=[
                person(
                    107,
                    pose_data=fence_pose(
                        hip_x=0.58,
                        hip_y=0.62,
                        wrist_x=0.40,
                        knee_y=0.80,
                    ),
                )
            ],
        )
        for index in range(5)
    ]

    assert all(
        safety(result, 107)["fence_evidence"]["core_on_forbidden_side"]
        is True
        for result in results
    )
    assert all(
        safety(result, 107)["fence_evidence"]["breach_entry_eligible"]
        is False
        for result in results
    )
    assert all(
        safety(result, 107)["rules"]["restricted_zone"]["raw_violation"]
        is False
        for result in results
    )
    assert all(
        safety(result, 107)["violation_active"] is False
        for result in results
    )
    assert all(result["safety_events"] == [] for result in results)


def test_strict_fence_pose_occlusion_pauses_breach_entry_and_exit() -> None:
    engine = PersonZoneRuleEngine(
        [RIGHT_RESTRICTED],
        strict_fence_config(
            breach_enter_frames=2,
            breach_exit_frames=2,
            climb_enter_frames=1,
        ),
    )
    engine.process_frame(
        frame_index=0,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.42,
                    hip_y=0.65,
                    wrist_x=0.40,
                    knee_y=0.84,
                ),
            )
        ],
    )
    engine.process_frame(
        frame_index=1,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.42,
                    hip_y=0.54,
                    wrist_x=0.50,
                    knee_y=0.66,
                ),
            )
        ],
    )
    first_forbidden = engine.process_frame(
        frame_index=2,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.58,
                    hip_y=0.53,
                    wrist_x=0.50,
                    knee_y=0.65,
                ),
            )
        ],
    )
    missing = engine.process_frame(
        frame_index=3,
        persons=[person(106)],
    )
    entered = engine.process_frame(
        frame_index=4,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.58,
                    hip_y=0.52,
                    wrist_x=0.50,
                    knee_y=0.64,
                ),
            )
        ],
    )
    first_allowed = engine.process_frame(
        frame_index=5,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.42,
                    hip_y=0.65,
                    wrist_x=0.40,
                    knee_y=0.84,
                ),
            )
        ],
    )
    ambiguous = engine.process_frame(
        frame_index=6,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.42,
                    association_status="ambiguous",
                ),
            )
        ],
    )
    exited = engine.process_frame(
        frame_index=7,
        persons=[
            person(
                106,
                pose_data=fence_pose(
                    hip_x=0.42,
                    hip_y=0.65,
                    wrist_x=0.40,
                    knee_y=0.84,
                ),
            )
        ],
    )

    assert safety(first_forbidden, 106)["fence_evidence"]["breach_streak"] == 1
    assert safety(missing, 106)["fence_evidence"]["observation_valid"] is False
    assert safety(missing, 106)["fence_evidence"]["breach_streak"] == 1
    assert missing["safety_events"] == []
    assert entered["safety_events"][0]["kind"] == "started"
    assert safety(first_allowed, 106)["fence_state"] == "breach"
    assert safety(ambiguous, 106)["fence_state"] == "breach"
    assert safety(ambiguous, 106)["fence_evidence"]["breach_streak"] == 1
    assert ambiguous["safety_events"] == []
    assert exited["safety_events"][0]["kind"] == "ended"
    assert safety(exited, 106)["fence_state"] == "clear"
