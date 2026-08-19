import json
from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient

from video_selector.app import create_app


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "video_selector/catalog.json"
MEDIA = ROOT / "content/video-selector"
PPE_CATALOG = ROOT / "video_selector/ppe-catalog.json"
PPE_MEDIA = ROOT / "content/ppe-video-selector"


def _valid_selection() -> dict:
    return {
        "videos": [
            {
                "video_id": "01",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 300,
                "rois": [
                    {
                        "roi_id": "roi-1",
                        "name": "Kapı alanı",
                        "roi_type": "restricted_zone",
                        "points": [
                            {"x": 0.1, "y": 0.2},
                            {"x": 0.5, "y": 0.2},
                            {"x": 0.55, "y": 0.8},
                            {"x": 0.08, "y": 0.8},
                        ],
                    }
                ],
            },
            {
                "video_id": "08",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 5.08,
                "rois": [
                    {
                        "roi_id": "roi-1",
                        "name": "Koridor",
                        "roi_type": "restricted_zone",
                        "points": [
                            {"x": 0.25, "y": 0.45},
                            {"x": 0.75, "y": 0.45},
                            {"x": 0.9, "y": 0.95},
                            {"x": 0.1, "y": 0.95},
                        ],
                    }
                ],
            },
        ]
    }


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            catalog_path=CATALOG,
            media_root=MEDIA,
            ppe_catalog_path=PPE_CATALOG,
            ppe_media_root=PPE_MEDIA,
            state_root=tmp_path / "state",
        )
    )


def test_catalog_exposes_all_office_and_ppe_candidates(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/videos")
    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog_revision"] == (
        "office-20260728-r3+ppe-20260728-r3"
    )
    assert payload["catalog_revisions"] == {
        "person_office": "office-20260728-r3",
        "ppe_safety": "ppe-20260728-r3",
    }
    assert len(payload["videos"]) == 48
    assert {video["video_id"] for video in payload["videos"]} == {
        f"{index:02d}" for index in range(1, 21)
    } | {f"F{index:02d}" for index in range(1, 5)} | {
        f"N{index:02d}" for index in range(1, 21)
    } | {f"S{index:02d}" for index in range(1, 5)}
    office = next(video for video in payload["videos"] if video["video_id"] == "01")
    ppe = next(video for video in payload["videos"] if video["video_id"] == "N01")
    assert office["category"] == "person_office"
    assert office["scenario"] == "fence_security"
    assert office["default_roi_type"] == "restricted_zone"
    assert office["roi_required"] is True
    assert office["pipeline"] == "person_roi"
    assert office["supported_modules"] == ["person_roi", "pose"]
    assert office["default_requested_modules"] == ["person_roi"]
    assert ppe["category"] == "ppe_safety"
    assert ppe["scenario"] == "ppe_safety"
    assert ppe["default_roi_type"] == "safe_walkway"
    assert ppe["roi_required"] is False
    assert ppe["pipeline"] == "ppe"
    assert ppe["supported_modules"] == ["ppe", "forklift"]
    assert ppe["default_requested_modules"] == ["ppe"]
    categories = {item["category"]: item for item in payload["categories"]}
    assert categories["person_office"]["scenario"] == "fence_security"
    assert categories["person_office"]["default_roi_type"] == "restricted_zone"
    assert categories["person_office"]["roi_required"] is True
    assert categories["ppe_safety"]["scenario"] == "ppe_safety"
    assert categories["ppe_safety"]["default_roi_type"] == "safe_walkway"
    assert categories["ppe_safety"]["roi_required"] is False
    assert all(video["media_url"].startswith("/api/videos/") for video in payload["videos"])
    assert all("processing_source_path" not in video for video in payload["videos"])
    assert all("_media_root" not in video for video in payload["videos"])


def test_media_supports_browser_byte_ranges(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        responses = [
            client.get(
                f"/api/videos/{video_id}/media",
                headers={"Range": "bytes=0-1023"},
            )
            for video_id in ("04", "F01", "N01", "S04")
        ]
    for response in responses:
        assert response.status_code == 206
        assert response.headers["content-range"].startswith("bytes 0-1023/")
        assert len(response.content) == 1024


def test_save_creates_manual_start_queue_without_execution(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=_valid_selection())
        latest = client.get("/api/selection/latest")
        queue = client.get("/api/queue/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "awaiting_manual_start"
    assert payload["schema_version"] == "colt-ai.video-roi-selection/v2"
    assert payload["processing_started"] is False
    assert "işleme henüz başlamadı" in payload["message"]
    assert "analysis_options" not in payload["videos"][0]
    assert latest.json()["selection_id"] == payload["selection_id"]

    queue_payload = queue.json()
    assert queue_payload["schema_version"] == "colt-ai.person-processing-queue/v2"
    assert queue_payload["selection_id"] == payload["selection_id"]
    assert queue_payload["state"] == "awaiting_manual_start"
    assert queue_payload["execution"] == {
        "requested": False,
        "started": False,
        "gpu_or_model_execution": False,
    }
    assert len(queue_payload["items"]) == 2
    assert queue_payload["items"][0]["source_path"].endswith("10-meva.avi")
    assert queue_payload["items"][0]["pipeline"] == "person_roi"
    assert queue_payload["items"][0]["scenario"] == "fence_security"
    assert queue_payload["items"][0]["requested_modules"] == ["person_roi"]
    assert queue_payload["items"][0]["rois"][0]["roi_type"] == "restricted_zone"
    assert queue_payload["items"][0]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
        "ppe_scope": "disabled",
    }
    assert queue_payload["items"][1]["clip"]["end_seconds"] == 5.08
    assert queue_payload["requested_modules"] == ["person_roi"]

    state = tmp_path / "state"
    assert (state / "latest.json").is_file()
    assert (state / "latest-queue.json").is_file()
    persisted = json.loads((state / "latest-queue.json").read_text(encoding="utf-8"))
    assert persisted == queue_payload


def test_unknown_video_and_invalid_roi_are_rejected(tmp_path: Path) -> None:
    unknown = _valid_selection()
    unknown["videos"][0]["video_id"] = "99"
    invalid = _valid_selection()
    invalid["videos"][0]["rois"][0]["points"] = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.1, "y": 0.9},
        {"x": 0.9, "y": 0.1},
    ]

    with _client(tmp_path) as client:
        unknown_response = client.put("/api/selection", json=unknown)
        invalid_response = client.put("/api/selection", json=invalid)

    assert unknown_response.status_code == 422
    assert invalid_response.status_code == 422


def test_fence_requires_restricted_zone_and_matching_scenario(
    tmp_path: Path,
) -> None:
    zero_roi = {
        "videos": [
            {
                "video_id": "F01",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [],
            }
        ]
    }
    wrong_type = deepcopy(zero_roi)
    wrong_type["videos"][0]["rois"] = [
        {
            "roi_id": "roi-1",
            "name": "Yanlış yürüyüş yolu",
            "roi_type": "safe_walkway",
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.1},
                {"x": 0.9, "y": 0.9},
                {"x": 0.1, "y": 0.9},
            ],
        }
    ]
    wrong_scenario = deepcopy(wrong_type)
    wrong_scenario["videos"][0]["scenario"] = "ppe_safety"

    with _client(tmp_path) as client:
        zero_response = client.put("/api/selection", json=zero_roi)
        wrong_type_response = client.put("/api/selection", json=wrong_type)
        wrong_scenario_response = client.put(
            "/api/selection",
            json=wrong_scenario,
        )

    assert zero_response.status_code == 422
    assert wrong_type_response.status_code == 422
    assert wrong_scenario_response.status_code == 422


def test_ppe_accepts_no_walkway_and_rejects_restricted_zone(
    tmp_path: Path,
) -> None:
    no_walkway = {
        "videos": [
            {
                "video_id": "S04",
                "scenario": "ppe_safety",
                "start_seconds": 0,
                "end_seconds": 10,
                "rois": [],
            }
        ]
    }
    wrong_type = deepcopy(no_walkway)
    wrong_type["videos"][0]["rois"] = [
        {
            "roi_id": "roi-1",
            "name": "Yasak alan",
            "roi_type": "restricted_zone",
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.1},
                {"x": 0.9, "y": 0.9},
                {"x": 0.1, "y": 0.9},
            ],
        }
    ]

    with _client(tmp_path) as client:
        accepted = client.put("/api/selection", json=no_walkway)
        queue = client.get("/api/queue/latest").json()
        rejected = client.put("/api/selection", json=wrong_type)

    assert accepted.status_code == 200
    assert accepted.json()["videos"][0]["scenario"] == "ppe_safety"
    assert accepted.json()["videos"][0]["rois"] == []
    assert queue["items"][0]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "disabled",
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "ppe_scope": "all_tracked_persons",
        "ppe_requirements": {"helmet": True, "hi_vis": True},
    }
    assert rejected.status_code == 422


def test_multiple_typed_zones_are_preserved_with_exact_alert_policies(
    tmp_path: Path,
) -> None:
    def zone(roi_id: str, roi_type: str, x_offset: float) -> dict:
        return {
            "roi_id": roi_id,
            "name": f"Alan {roi_id}",
            "roi_type": roi_type,
            "points": [
                {"x": 0.05 + x_offset, "y": 0.1},
                {"x": 0.35 + x_offset, "y": 0.1},
                {"x": 0.35 + x_offset, "y": 0.9},
                {"x": 0.05 + x_offset, "y": 0.9},
            ],
        }

    selection = {
        "videos": [
            {
                "video_id": "F01",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [
                    zone("roi-1", "restricted_zone", 0.0),
                    zone("roi-2", "restricted_zone", 0.5),
                ],
            },
            {
                "video_id": "S04",
                "scenario": "ppe_safety",
                "start_seconds": 0,
                "end_seconds": 10,
                "rois": [
                    zone("roi-1", "safe_walkway", 0.0),
                    zone("roi-2", "safe_walkway", 0.5),
                ],
            },
        ]
    }

    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=selection)
        queue = client.get("/api/queue/latest").json()

    assert response.status_code == 200
    assert [
        [roi["roi_type"] for roi in item["rois"]]
        for item in queue["items"]
    ] == [
        ["restricted_zone", "restricted_zone"],
        ["safe_walkway", "safe_walkway"],
    ]
    assert queue["items"][0]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
        "ppe_scope": "disabled",
    }
    assert queue["items"][1]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "outside_all_safe_walkways",
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "ppe_scope": "all_tracked_persons",
        "ppe_requirements": {"helmet": True, "hi_vis": True},
    }


def test_mixed_selection_records_pipeline_per_queue_item(tmp_path: Path) -> None:
    selection = deepcopy(_valid_selection())
    selection["videos"] = [selection["videos"][0]]
    selection["videos"].append(
        {
            "video_id": "N01",
            "scenario": "ppe_safety",
            "start_seconds": 1,
            "end_seconds": 8,
            "rois": [
                {
                    "roi_id": "roi-1",
                    "name": "Güvenli yürüyüş yolu",
                    "roi_type": "safe_walkway",
                    "points": [
                        {"x": 0.05, "y": 0.05},
                        {"x": 0.95, "y": 0.05},
                        {"x": 0.95, "y": 0.95},
                        {"x": 0.05, "y": 0.95},
                    ],
                }
            ],
        }
    )

    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=selection)
        queue = client.get("/api/queue/latest").json()

    assert response.status_code == 200
    assert response.json()["catalog_revision"] == (
        "office-20260728-r3+ppe-20260728-r3"
    )
    assert response.json()["catalog_revisions"] == {
        "person_office": "office-20260728-r3",
        "ppe_safety": "ppe-20260728-r3",
    }
    assert queue["requested_modules"] == ["person_roi", "ppe"]
    assert [
        (
            item["video_id"],
            item["category"],
            item["pipeline"],
            item["requested_modules"],
        )
        for item in queue["items"]
    ] == [
        ("01", "person_office", "person_roi", ["person_roi"]),
        ("N01", "ppe_safety", "ppe", ["ppe"]),
    ]
    assert queue["items"][1]["scenario"] == "ppe_safety"
    assert queue["items"][1]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "outside_all_safe_walkways",
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "ppe_scope": "all_tracked_persons",
        "ppe_requirements": {"helmet": True, "hi_vis": True},
    }


def test_analysis_options_derive_pose_and_forklift_queue_contracts(
    tmp_path: Path,
) -> None:
    fence_zone = {
        "roi_id": "roi-1",
        "name": "Çit hattı",
        "roi_type": "restricted_zone",
        "points": [
            {"x": 0.1, "y": 0.2},
            {"x": 0.9, "y": 0.2},
            {"x": 0.9, "y": 0.8},
            {"x": 0.1, "y": 0.8},
        ],
    }
    selection = {
        "videos": [
            {
                "video_id": "F01",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [fence_zone],
                "analysis_options": {
                    "fence_pose_roi": {
                        "enabled": True,
                        "selected_keypoints": [
                            "left_wrist",
                            "right_wrist",
                            "left_ankle",
                            "right_ankle",
                        ],
                        "inside_ratio_threshold": 0.75,
                        "keypoint_confidence_threshold": 0.30,
                        "minimum_visible_keypoints": 2,
                    }
                },
            },
            {
                "video_id": "S01",
                "scenario": "ppe_safety",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [],
                "analysis_options": {
                    "forklift_driver_suppression": {
                        "enabled": True,
                        "suppressed_alerts": [
                            "ppe_violation",
                            "safe_walkway_violation",
                        ],
                        "minimum_forklift_confidence": 0.40,
                        "minimum_person_ioa": 0.60,
                        "enter_debounce_frames": 5,
                        "exit_debounce_frames": 9,
                    }
                },
            },
        ]
    }

    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=selection)
        queue = client.get("/api/queue/latest").json()

    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["videos"][0]["analysis_options"] == selection["videos"][0][
        "analysis_options"
    ]
    assert snapshot["videos"][1]["analysis_options"] == selection["videos"][1][
        "analysis_options"
    ]
    assert queue["requested_modules"] == [
        "person_roi",
        "pose",
        "ppe",
        "forklift",
    ]
    fence = queue["items"][0]
    assert fence["requested_modules"] == ["person_roi", "pose"]
    assert fence["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "pose_keypoint_ratio",
        "zone_rule": (
            "selected_pose_keypoint_ratio_inside_restricted_zone"
        ),
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
        "ppe_scope": "disabled",
        "pose_zone_rule": {
            "keypoint_layout": "coco17",
            "selected_keypoints": [
                "left_wrist",
                "right_wrist",
                "left_ankle",
                "right_ankle",
            ],
            "inside_ratio_threshold": 0.75,
            "keypoint_confidence_threshold": 0.30,
            "minimum_visible_keypoints": 2,
            "ratio_denominator": "selected_keypoints",
            "roi_aggregation": "union_any",
            "polygon_boundary": "inclusive",
            "person_pose_association": "highest_iou_to_nvdcf_track",
            "insufficient_pose_policy": "no_alert",
        },
    }
    ppe = queue["items"][1]
    assert ppe["requested_modules"] == ["ppe", "forklift"]
    assert ppe["alert_policy"]["forklift_driver_suppression"] == {
        "enabled": True,
        "forklift_class": "forklift_candidate",
        "detector_evidence": "coco_truck_class_7",
        "classification_scope": "industrial_forklift_candidate",
        "tracking_identity": "nvdcf_track_id",
        "association_rule": "temporal_person_forklift_ioa",
        "minimum_forklift_confidence": 0.40,
        "minimum_person_ioa": 0.60,
        "enter_debounce_frames": 5,
        "exit_debounce_frames": 9,
        "maximum_occupants_per_forklift": 1,
        "suppressed_alerts": [
            "ppe_violation",
            "safe_walkway_violation",
        ],
        "render_state": "forklift_driver",
        "missing_forklift_evidence": "do_not_suppress",
    }


def test_fence_boundary_creates_strict_staged_crossing_contract(
    tmp_path: Path,
) -> None:
    crossing = {
        "enabled": True,
        "boundary_start": {"x": 0.12, "y": 0.58},
        "boundary_end": {"x": 0.91, "y": 0.43},
        "forbidden_side": "right",
    }
    selection = {
        "videos": [
            {
                "video_id": "F01",
                "scenario": "fence_security",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [
                    {
                        "roi_id": "roi-1",
                        "name": "Çit yaklaşma bandı",
                        "roi_type": "restricted_zone",
                        "points": [
                            {"x": 0.1, "y": 0.2},
                            {"x": 0.9, "y": 0.2},
                            {"x": 0.9, "y": 0.8},
                            {"x": 0.1, "y": 0.8},
                        ],
                    }
                ],
                "analysis_options": {
                    "fence_pose_roi": {
                        "enabled": True,
                        "selected_keypoints": [
                            "left_wrist",
                            "right_wrist",
                            "left_hip",
                            "right_hip",
                        ],
                        "inside_ratio_threshold": 0.5,
                        "keypoint_confidence_threshold": 0.25,
                        "minimum_visible_keypoints": 2,
                    },
                    "fence_crossing_rule": crossing,
                },
            }
        ]
    }

    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=selection)
        queue = client.get("/api/queue/latest").json()

    assert response.status_code == 200
    saved_rule = response.json()["videos"][0]["analysis_options"][
        "fence_crossing_rule"
    ]
    expected_rule = {
        **crossing,
        "contact_band": 0.03,
        "minimum_confidence": 0.30,
        "minimum_core_visible": 1,
        "breach_enter_frames": 4,
        "breach_exit_frames": 4,
        "approach_keypoint_names": [
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        ],
        "approach_minimum_count": 1,
        "wrist_contact_required": 1,
        "hip_rise_ratio": 0.08,
        "raised_knee_ratio": 0.10,
        "climb_enter_frames": 2,
        "climb_exit_frames": 2,
        "history_window_frames": 30,
    }
    assert saved_rule == expected_rule
    item = queue["items"][0]
    assert item["requested_modules"] == ["person_roi", "pose"]
    assert item["alert_policy"]["person_anchor"] == "mid_hip"
    assert item["alert_policy"]["zone_rule"] == (
        "staged_fence_boundary_crossing"
    )
    assert item["alert_policy"]["pose_zone_rule"][
        "selected_keypoints"
    ] == [
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]
    assert item["alert_policy"]["fence_crossing_rule"] == expected_rule


def test_fence_boundary_rejects_ambiguous_or_wrong_scenario_rules(
    tmp_path: Path,
) -> None:
    base = deepcopy(_valid_selection())
    base["videos"] = [base["videos"][0]]
    same_endpoints = deepcopy(base)
    same_endpoints["videos"][0]["analysis_options"] = {
        "fence_crossing_rule": {
            "boundary_start": {"x": 0.4, "y": 0.4},
            "boundary_end": {"x": 0.4, "y": 0.4},
            "forbidden_side": "left",
        }
    }
    unknown_field = deepcopy(base)
    unknown_field["videos"][0]["analysis_options"] = {
        "fence_crossing_rule": {
            "boundary_start": {"x": 0.1, "y": 0.5},
            "boundary_end": {"x": 0.9, "y": 0.5},
            "forbidden_side": "left",
            "guess_forbidden_side": True,
        }
    }
    pose_disabled = deepcopy(base)
    pose_disabled["videos"][0]["analysis_options"] = {
        "fence_pose_roi": {"enabled": False},
        "fence_crossing_rule": {
            "boundary_start": {"x": 0.1, "y": 0.5},
            "boundary_end": {"x": 0.9, "y": 0.5},
            "forbidden_side": "right",
        },
    }
    ppe = {
        "videos": [
            {
                "video_id": "S01",
                "scenario": "ppe_safety",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [],
                "analysis_options": {
                    "fence_crossing_rule": {
                        "boundary_start": {"x": 0.1, "y": 0.5},
                        "boundary_end": {"x": 0.9, "y": 0.5},
                        "forbidden_side": "left",
                    }
                },
            }
        ]
    }

    with _client(tmp_path) as client:
        responses = [
            client.put("/api/selection", json=payload)
            for payload in (
                same_endpoints,
                unknown_field,
                pose_disabled,
                ppe,
            )
        ]

    assert [response.status_code for response in responses] == [422] * 4


def test_selector_exposes_two_point_fence_line_controls(
    tmp_path: Path,
) -> None:
    with _client(tmp_path) as client:
        html = client.get("/").text
        javascript = client.get("/static/app.js").text

    assert 'id="drawFenceLineButton"' in html
    assert 'name="fenceForbiddenSide"' in html
    assert 'value="left"' in html
    assert 'value="right"' in html
    assert "fence_crossing_rule" in javascript
    assert "forbiddenHalfPlanePolygon" in javascript


def test_disabled_analysis_options_keep_legacy_queue_behavior(
    tmp_path: Path,
) -> None:
    selection = deepcopy(_valid_selection())
    selection["videos"] = [selection["videos"][0]]
    selection["videos"][0]["analysis_options"] = {
        "fence_pose_roi": {"enabled": False},
        "fence_crossing_rule": {"enabled": False},
    }

    with _client(tmp_path) as client:
        response = client.put("/api/selection", json=selection)
        queue = client.get("/api/queue/latest").json()

    assert response.status_code == 200
    assert queue["requested_modules"] == ["person_roi"]
    assert queue["items"][0]["requested_modules"] == ["person_roi"]
    assert queue["items"][0]["alert_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
        "ppe_scope": "disabled",
    }


def test_analysis_options_reject_wrong_scenario_and_invalid_values(
    tmp_path: Path,
) -> None:
    fence = deepcopy(_valid_selection())
    fence["videos"] = [fence["videos"][0]]
    wrong_scenario = deepcopy(fence)
    wrong_scenario["videos"][0]["analysis_options"] = {
        "forklift_driver_suppression": {}
    }
    duplicate_keypoints = deepcopy(fence)
    duplicate_keypoints["videos"][0]["analysis_options"] = {
        "fence_pose_roi": {
            "selected_keypoints": ["left_wrist", "left_wrist"],
            "minimum_visible_keypoints": 1,
        }
    }
    too_few_visible = deepcopy(fence)
    too_few_visible["videos"][0]["analysis_options"] = {
        "fence_pose_roi": {
            "selected_keypoints": ["left_wrist"],
            "minimum_visible_keypoints": 2,
        }
    }
    ppe = {
        "videos": [
            {
                "video_id": "S01",
                "scenario": "ppe_safety",
                "start_seconds": 0,
                "end_seconds": 5,
                "rois": [],
                "analysis_options": {
                    "forklift_driver_suppression": {
                        "suppressed_alerts": [],
                    }
                },
            }
        ]
    }

    with _client(tmp_path) as client:
        responses = [
            client.put("/api/selection", json=payload)
            for payload in (
                wrong_scenario,
                duplicate_keypoints,
                too_few_visible,
                ppe,
            )
        ]

    assert [response.status_code for response in responses] == [422] * 4


def test_explicit_single_catalog_contract_remains_available(
    tmp_path: Path,
) -> None:
    app = create_app(
        catalog_path=CATALOG,
        media_root=MEDIA,
        state_root=tmp_path / "state",
    )
    with TestClient(app) as client:
        payload = client.get("/api/videos").json()
    assert payload["catalog_revision"] == "office-20260728-r3"
    assert len(payload["videos"]) == 24


def test_no_admin_token_or_execution_route_exists(tmp_path: Path) -> None:
    app = create_app(
        catalog_path=CATALOG,
        media_root=MEDIA,
        ppe_catalog_path=PPE_CATALOG,
        ppe_media_root=PPE_MEDIA,
        state_root=tmp_path / "state",
    )
    paths = {route.path for route in app.routes}
    assert not any("execute" in path for path in paths)
    assert not any("token" in path for path in paths)
    assert "/api/selection" in paths
