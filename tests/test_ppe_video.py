from __future__ import annotations

import numpy as np
import pytest

from content import ppe_video
from content.ppe_video import (
    PERSON_EVENT_SCHEMA_VERSION,
    TRANSITION_EVENT_SCHEMA_VERSION,
    PpeVideoError,
    WALKWAY_ALERT_LABEL,
    build_parser,
    draw_ppe_frame,
    person_ppe_events,
    transition_event_contract,
)
from content.person_zone_rules import (
    PersonZoneRuleEngine,
    ZoneArea,
    ZoneRuleConfig,
)
from content.theme import BRAND_NAME, THEME_ID


def detection(
    canonical_class: str,
    compliance: str,
    *,
    box: list[float] | None = None,
) -> dict[str, object]:
    return {
        "canonical_class": canonical_class,
        "compliance": compliance,
        "confidence": 0.83,
        "bbox_norm_xywh": box or [0.2, 0.2, 0.3, 0.4],
    }


def roi(
    *,
    left: float = 0.05,
    right: float = 0.5,
) -> list[dict[str, object]]:
    return [
        {
            "roi_id": "roi-1",
            "name": "PPE Alanı",
            "points": [
                {"x": left, "y": 0.15},
                {"x": right, "y": 0.15},
                {"x": right, "y": 0.9},
                {"x": left, "y": 0.9},
            ],
        }
    ]


def safe_walkway() -> list[dict[str, object]]:
    return [
        {
            "roi_id": "walkway-1",
            "name": "Ana Yürüyüş Yolu",
            "roi_type": "safe_walkway",
            "points": [
                {"x": 0.05, "y": 0.15},
                {"x": 0.5, "y": 0.15},
                {"x": 0.5, "y": 0.95},
                {"x": 0.05, "y": 0.95},
            ],
        }
    ]


def test_opencv_overlay_text_transliterates_turkish_without_question_marks() -> None:
    assert WALKWAY_ALERT_LABEL == "YURUYUS YOLU DISI"
    assert WALKWAY_ALERT_LABEL.isascii()
    assert (
        ppe_video._opencv_text("Güvenli Yürüyüş Yolu · Çıkış")
        == "Guvenli Yuruyus Yolu Cikis"
    )


def person(
    *,
    track_id: int = 17,
    helmet: str = "present",
    hi_vis: str = "present",
    helmet_state: str | None = None,
    hi_vis_state: str | None = None,
    box: list[float] | None = None,
) -> dict[str, object]:
    def equipment(
        evidence: str,
        state: str | None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence": evidence,
            "confidence": 0.91 if evidence != "unknown" else 0.0,
            "link_status": (
                "matched"
                if evidence in {"present", "absent"}
                else "unknown_no_observation"
            ),
        }
        if state is not None:
            result["state"] = state
        return result

    return {
        "person_id": f"person-{track_id}",
        "track_id": track_id,
        "bbox_norm_xywh": box or [0.2, 0.18, 0.35, 0.65],
        "confidence": 0.94,
        "occluded": False,
        "helmet": equipment(helmet, helmet_state),
        "hi_vis": equipment(hi_vis, hi_vis_state),
    }


def test_draw_ppe_frame_uses_shared_visual_contract() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered, counts = draw_ppe_frame(
        frame,
        detections=[
            detection("helmet", "compliant"),
            detection(
                "no_hi_vis",
                "noncompliant",
                box=[0.55, 0.25, 0.2, 0.5],
            ),
        ],
        camera_label="CAM-P04",
    )

    assert rendered.shape == (360, 640, 3)
    assert counts == {"helmet": 1, "no_hi_vis": 1}
    assert rendered.any()
    assert BRAND_NAME == "COLT AI - COLLBRAI"
    assert THEME_ID == "colt-collbrai-navy-v1"


def test_person_mode_draws_one_human_bbox_with_per_person_ppe_status() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered, counts = draw_ppe_frame(
        frame,
        # Person mode deliberately ignores legacy object boxes.
        detections=[
            {
                "canonical_class": "unsupported-in-person-mode",
            }
        ],
        persons=[person(helmet="present", hi_vis="absent")],
        camera_label="CAM-P04",
    )

    assert counts == {
        "person": 1,
        "helmet": 1,
        "no_hi_vis": 1,
    }
    assert rendered.any()
    # Human bbox edge at x=128 is painted while there is no independent PPE
    # object-box contract in person mode.
    assert rendered[180, 128].any()


def test_absent_helmet_draws_critical_alert_zone_and_emits_person_event() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered, counts = draw_ppe_frame(
        frame,
        detections=[],
        persons=[person(track_id=42, helmet="absent", hi_vis="present")],
        camera_label="CAM-P08",
    )
    events = person_ppe_events(
        frame_index=11,
        timestamp_ns=440_000_000,
        persons=[person(track_id=42, helmet="absent", hi_vis="present")],
        camera_label="CAM-P08",
    )

    assert counts == {"person": 1, "no_helmet": 1, "hi_vis": 1}
    # The full-width lower strip is the dedicated helmet alert panel.
    assert rendered[-20:, :].mean() > rendered[260:280, :].mean()
    assert len(events) == 1
    assert events[0]["schema_version"] == PERSON_EVENT_SCHEMA_VERSION
    assert events[0]["severity"] == "critical"
    assert events[0]["track_id"] == 42
    assert events[0]["violations"] == ["helmet_missing"]


def test_forklift_operator_stays_visible_without_ppe_or_walkway_alert() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    operator = person(track_id=42, helmet="absent", hi_vis="absent")
    operator["vehicle_context"] = {
        "raw_match": True,
        "suppression_active": True,
        "role": "forklift_operator",
        "vehicle_track_id": 9,
        "person_ioa": 0.9,
        "ppe_alert_eligible": False,
        "walkway_alert_eligible": False,
        "suppression_scopes": ["ppe", "walkway"],
        "enter_streak": 4,
        "exit_streak": 0,
    }
    operator["zone_alert_eligible"] = False
    operator["zone_safety"] = {
        "footpoint_norm_xy": [0.375, 0.83],
        "violation_active": True,
        "active_violation_types": ["walkway_violation"],
        "contributing_area_ids": ["walkway-1"],
        "rules": {},
    }
    vehicle = {
        "class_id": 7,
        "class_name": "forklift_candidate",
        "track_id": 9,
        "confidence": 0.82,
        "bbox_norm_xywh": [0.1, 0.1, 0.7, 0.8],
    }

    rendered, counts = draw_ppe_frame(
        frame,
        detections=[],
        persons=[operator],
        vehicles=[vehicle],
        camera_label="CAM-S04",
    )
    events = person_ppe_events(
        frame_index=11,
        persons=[operator],
        camera_label="CAM-S04",
    )

    assert rendered.any()
    assert counts["person"] == 1
    assert counts["forklift_candidate"] == 1
    assert counts["forklift_operator"] == 1
    assert counts["suppressed_no_helmet"] == 1
    assert counts["suppressed_no_hi_vis"] == 1
    assert "no_helmet" not in counts
    assert "walkway_violation" not in counts
    assert events == []


def test_unknown_equipment_is_visible_but_never_becomes_a_violation() -> None:
    _, counts = draw_ppe_frame(
        np.zeros((240, 400, 3), dtype=np.uint8),
        detections=[],
        persons=[person(helmet="unknown", hi_vis="unknown")],
        camera_label="CAM-P13",
    )
    events = person_ppe_events(
        frame_index=0,
        persons=[person(helmet="unknown", hi_vis="unknown")],
        camera_label="CAM-P13",
    )

    assert counts == {
        "person": 1,
        "helmet_unknown": 1,
        "hi_vis_unknown": 1,
    }
    assert events == []


def test_temporal_absent_state_keeps_alert_during_unknown_evidence_frame() -> None:
    subject = person(
        track_id=73,
        helmet="unknown",
        helmet_state="absent",
        hi_vis="unknown",
        hi_vis_state="present",
    )
    rendered, counts = draw_ppe_frame(
        np.zeros((360, 640, 3), dtype=np.uint8),
        detections=[],
        persons=[subject],
        camera_label="CAM-P08",
    )
    events = person_ppe_events(
        frame_index=19,
        persons=[subject],
        camera_label="CAM-P08",
    )

    assert counts == {"person": 1, "no_helmet": 1, "hi_vis": 1}
    assert rendered[-20:, :].mean() > rendered[260:280, :].mean()
    assert len(events) == 1
    assert events[0]["violations"] == ["helmet_missing"]
    assert events[0]["equipment"]["helmet"] == {
        "state": "absent",
        "state_source": "temporal_state",
        "evidence": "unknown",
        "confidence": 0.0,
        "link_status": "unknown_no_observation",
    }


def test_temporal_present_state_suppresses_single_absent_evidence_outlier() -> None:
    subject = person(
        helmet="absent",
        helmet_state="present",
        hi_vis="present",
        hi_vis_state="present",
    )
    _, counts = draw_ppe_frame(
        np.zeros((240, 400, 3), dtype=np.uint8),
        detections=[],
        persons=[subject],
        camera_label="CAM-P13",
    )

    assert counts == {"person": 1, "helmet": 1, "hi_vis": 1}
    assert (
        person_ppe_events(
            frame_index=0,
            persons=[subject],
            camera_label="CAM-P13",
        )
        == []
    )


def test_person_mode_requires_unique_identity_and_matched_absence() -> None:
    duplicate = [person(track_id=8), person(track_id=8)]
    with pytest.raises(PpeVideoError, match="duplicate track_id"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=duplicate,
            camera_label="CAM-PPE",
        )

    invalid = person(helmet="absent")
    invalid["helmet"]["link_status"] = "unknown_ambiguous"
    with pytest.raises(PpeVideoError, match="contract is inconsistent"):
        person_ppe_events(
            frame_index=0,
            persons=[invalid],
            camera_label="CAM-PPE",
        )

    invalid_state = person()
    invalid_state["helmet"]["state"] = "stale"
    with pytest.raises(PpeVideoError, match=r"helmet\.state"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=[invalid_state],
            camera_label="CAM-PPE",
        )


def test_inferred_missing_evidence_uses_explicit_nonmatched_link() -> None:
    subject = person(helmet="absent", hi_vis="unknown")
    subject["helmet"]["evidence_source"] = (
        "visible_zone_no_positive_observation"
    )
    subject["helmet"]["link_status"] = "inferred_visible_missing"
    subject["hi_vis"]["evidence_source"] = (
        "insufficient_visibility_or_no_observation"
    )

    _, counts = draw_ppe_frame(
        np.zeros((240, 400, 3), dtype=np.uint8),
        detections=[],
        persons=[subject],
        camera_label="CAM-PPE",
    )
    assert counts["no_helmet"] == 1

    subject["helmet"]["link_status"] = "matched"
    with pytest.raises(PpeVideoError, match="contract is inconsistent"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=[subject],
            camera_label="CAM-PPE",
        )


def test_matched_detector_contract_rejects_explicit_empty_observations() -> None:
    subject = person(helmet="present", hi_vis="present")
    subject["helmet"]["evidence_source"] = "detector_observation"
    subject["helmet"]["observations"] = []
    with pytest.raises(PpeVideoError, match="needs an observation"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=[subject],
            camera_label="CAM-PPE",
        )


def test_fusion_transition_contract_counts_state_changes_only_once() -> None:
    rows = [
        {"frame_index": 0, "persons": [], "alarm_events": []},
        {
            "frame_index": 1,
            "persons": [],
            "alarm_events": [
                {
                    "kind": "started",
                    "type": "no_helmet",
                    "track_id": 7,
                    "frame_index": 1,
                }
            ],
        },
        {"frame_index": 2, "persons": [], "alarm_events": []},
        {"frame_index": 3, "persons": [], "alarm_events": []},
        {
            "frame_index": 4,
            "persons": [],
            "alarm_events": [
                {
                    "kind": "ended",
                    "type": "no_helmet",
                    "track_id": 7,
                    "frame_index": 4,
                }
            ],
        },
    ]

    contract = transition_event_contract(rows, person_mode=True)

    assert contract["schema_version"] == TRANSITION_EVENT_SCHEMA_VERSION
    assert contract["mode"] == "fusion_state_transitions"
    assert contract["per_frame_violation_events_emitted"] is False
    assert contract["observations"] == {
        "ended": 1,
        "ended:no_helmet": 1,
        "started": 1,
        "started:no_helmet": 1,
        "total": 2,
    }
    assert contract["open_alarms_at_end"] == []


def test_transition_contract_rejects_duplicate_start_and_marks_legacy() -> None:
    duplicate = [
        {
            "frame_index": 0,
            "persons": [],
            "alarm_events": [
                {
                    "kind": "started",
                    "type": "no_hi_vis",
                    "track_id": 3,
                    "frame_index": 0,
                }
            ],
        },
        {
            "frame_index": 1,
            "persons": [],
            "alarm_events": [
                {
                    "kind": "started",
                    "type": "no_hi_vis",
                    "track_id": 3,
                    "frame_index": 1,
                }
            ],
        },
    ]
    with pytest.raises(PpeVideoError, match="started twice"):
        transition_event_contract(duplicate, person_mode=True)

    legacy = transition_event_contract(
        [{"frame_index": 0, "persons": []}],
        person_mode=True,
    )
    assert legacy["mode"] == "legacy_person_rows_without_transitions"
    assert legacy["observations"] == {}


def test_roi_filters_boxes_by_center_and_draws_navy_polygon() -> None:
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    rendered, counts = draw_ppe_frame(
        frame,
        detections=[
            detection("helmet", "compliant", box=[0.1, 0.35, 0.2, 0.3]),
            detection(
                "no_hi_vis",
                "noncompliant",
                box=[0.7, 0.35, 0.2, 0.3],
            ),
        ],
        camera_label="CAM-P04",
        rois=roi(),
    )

    assert counts == {"helmet": 1}
    # ROI boundary is visible while the excluded right-hand box is not drawn.
    assert rendered[120, 20].any()
    assert not rendered[120, 320].any()


def test_roi_boundary_is_inside_and_omitted_roi_keeps_full_frame() -> None:
    detections = [
        detection("helmet", "compliant", box=[0.4, 0.35, 0.2, 0.2]),
        detection(
            "no_helmet",
            "noncompliant",
            box=[0.7, 0.35, 0.2, 0.2],
        ),
    ]
    _, filtered = draw_ppe_frame(
        np.zeros((240, 400, 3), dtype=np.uint8),
        detections=detections,
        camera_label="CAM-P04",
        rois=roi(),
    )
    _, full_frame = draw_ppe_frame(
        np.zeros((240, 400, 3), dtype=np.uint8),
        detections=detections,
        camera_label="CAM-P04",
    )

    assert filtered == {"helmet": 1}
    assert full_frame == {"helmet": 1, "no_helmet": 1}


def test_empty_or_invalid_roi_is_rejected() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(PpeVideoError, match="non-empty"):
        draw_ppe_frame(
            frame,
            detections=[],
            camera_label="CAM-PPE",
            rois=[],
        )
    invalid = roi()
    invalid[0]["points"][0] = {"x": -0.1, "y": 0.1}
    with pytest.raises(PpeVideoError, match="normalized"):
        draw_ppe_frame(
            frame,
            detections=[],
            camera_label="CAM-PPE",
            rois=invalid,
        )


def test_safe_walkway_keeps_every_person_and_marks_outside_track() -> None:
    walkway = safe_walkway()[0]
    engine = PersonZoneRuleEngine(
        [
            ZoneArea(
                area_id=str(walkway["roi_id"]),
                area_type="safe_walkway",
                name=str(walkway["name"]),
                points=tuple(
                    (float(point["x"]), float(point["y"]))
                    for point in walkway["points"]
                ),
            )
        ],
        ZoneRuleConfig(
            restricted_enter_frames=1,
            restricted_exit_frames=1,
            walkway_enter_frames=1,
            walkway_exit_frames=1,
            track_ttl_frames=2,
        ),
    )
    qualified = engine.process_frame(
        frame_index=0,
        persons=[
            person(track_id=1, box=[0.1, 0.2, 0.2, 0.6]),
            person(track_id=2, box=[0.7, 0.2, 0.2, 0.6]),
        ],
    )

    rendered, counts = draw_ppe_frame(
        np.zeros((360, 640, 3), dtype=np.uint8),
        detections=[],
        persons=qualified["persons"],
        camera_label="CAM-S04",
        safe_walkways=safe_walkway(),
    )

    assert counts == {
        "person": 2,
        "helmet": 2,
        "hi_vis": 2,
        "walkway_violation": 1,
    }
    assert [item["track_id"] for item in qualified["persons"]] == [1, 2]
    assert WALKWAY_ALERT_LABEL == "YURUYUS YOLU DISI"
    # Both the cyan walkway boundary and the separate lower warning row exist.
    assert rendered[180, 32].any()
    assert rendered[-20:, :].mean() > rendered[250:270, :].mean()


def test_walkway_violation_and_helmet_warning_share_distinct_panel_rows() -> None:
    subject = person(
        track_id=44,
        helmet="absent",
        hi_vis="present",
        box=[0.7, 0.2, 0.2, 0.6],
    )
    subject["zone_safety"] = {
        "footpoint_norm_xy": [0.8, 0.8],
        "violation_active": True,
        "active_violation_types": ["walkway_violation"],
        "contributing_area_ids": ["walkway-1"],
        "rules": {},
    }
    rendered, counts = draw_ppe_frame(
        np.zeros((480, 800, 3), dtype=np.uint8),
        detections=[],
        persons=[subject],
        camera_label="CAM-S04",
        safe_walkways=safe_walkway(),
    )

    assert counts["no_helmet"] == 1
    assert counts["walkway_violation"] == 1
    # Two alert rows make the lower warning region taller than a helmet-only
    # panel while preserving one person bbox.
    assert counts["person"] == 1
    assert rendered[-100:, :].mean() > rendered[300:340, :].mean()


def test_safe_walkway_rejects_restricted_type_and_legacy_filter_mix() -> None:
    restricted = safe_walkway()
    restricted[0]["roi_type"] = "restricted_zone"
    with pytest.raises(PpeVideoError, match="safe_walkway"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=[person()],
            camera_label="CAM-S04",
            safe_walkways=restricted,
        )
    with pytest.raises(PpeVideoError, match="cannot be combined"):
        draw_ppe_frame(
            np.zeros((240, 400, 3), dtype=np.uint8),
            detections=[],
            persons=[person()],
            camera_label="CAM-S04",
            rois=roi(),
            safe_walkways=safe_walkway(),
        )


def test_cli_without_roi_file_preserves_full_frame_default() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "input.mp4",
            "--predictions-jsonl",
            "predictions.jsonl",
            "--output",
            "output.mp4",
            "--camera-label",
            "CAM-P04",
        ]
    )

    assert args.rois_json is None
    assert args.safe_walkways_json is None


@pytest.mark.parametrize(
    ("canonical_class", "compliance", "box"),
    [
        ("gloves", "compliant", [0.1, 0.1, 0.2, 0.2]),
        ("helmet", "unknown", [0.1, 0.1, 0.2, 0.2]),
        ("helmet", "compliant", [0.9, 0.1, 0.2, 0.2]),
    ],
)
def test_draw_ppe_frame_rejects_invalid_detection_contract(
    canonical_class: str,
    compliance: str,
    box: list[float],
) -> None:
    with pytest.raises(PpeVideoError):
        draw_ppe_frame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[detection(canonical_class, compliance, box=box)],
            camera_label="CAM-PPE",
        )
