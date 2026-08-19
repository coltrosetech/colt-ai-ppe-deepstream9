import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from content import roi_demo
from content.theme import BRAND_NAME, THEME, THEME_ID, hex_to_bgr


def _base_config() -> dict:
    return {
        "schema_version": "deepsafe.content-roi-demo/v1",
        "demo_id": "fixture-roi-demo",
        "title": "İNSAN TESPİTİ",
        "camera_label": "CAM-TEST",
        "disclosure": {
            "mode": "baseline_preview",
            "label": "TEST BASELINE",
        },
        "source": {
            "asset_id": "fixture",
            "video_path": "data/fixture.avi",
            "source_url": "https://example.test/source",
            "license_id": "CC0-1.0",
            "license_url": "https://example.test/license",
            "attribution": "Fixture",
            "modification_notice": "Synthetic test video with overlay.",
        },
        "detections": {
            "kind": "predictions_jsonl",
            "path": "data/predictions.jsonl",
            "sequence_id": "fixture",
            "confidence_threshold": 0.25,
            "expected_model_id": "fixture-model",
        },
        "clip": {
            "start_seconds": 0.0,
            "end_seconds": 2.0,
        },
        "roi": {
            "id": "fixture-zone",
            "label": "KISITLI ALAN",
            "polygon_norm": [
                [0.5, 0.5],
                [0.95, 0.5],
                [0.95, 0.95],
                [0.5, 0.95],
            ],
        },
        "event_policy": {
            "rule": "bbox_bottom_center_inside",
            "enter_debounce_frames": 2,
            "exit_debounce_frames": 2,
        },
        "output": {
            "directory": "results/fixture-roi-demo",
            "width": 640,
            "height": 360,
            "codec": "libx264",
            "crf": 24,
            "preset": "ultrafast",
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _make_fixture(root: Path) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    video = data / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (100, 100),
    )
    assert writer.isOpened()
    try:
        for index in range(10):
            frame = np.full((100, 100, 3), 28 + index * 3, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    rows = []
    for frame_index in range(10):
        if 2 <= frame_index <= 5:
            box = [0.60, 0.55, 0.20, 0.25]
        else:
            box = [0.05, 0.10, 0.20, 0.25]
        rows.append(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "fixture",
                "frame_index": frame_index,
                "image_width": 100,
                "image_height": 100,
                "timestamp_ns": frame_index * 200_000_000,
                "source_uri": "file:///fixture.avi",
                "model_id": "fixture-model",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox_norm_xywh": box,
                    }
                ],
            }
        )
    predictions = data / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = root / "content/config.json"
    _write_json(config, _base_config())
    return config


def _make_v2_fixture(root: Path) -> Path:
    config_path = _make_fixture(root)
    predictions = root / "data/predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        detection = row["detections"][0]
        detection["track_id"] = 7
        if row["frame_index"] in {2, 3}:
            detection["bbox_norm_xywh"] = [0.60, 0.55, 0.20, 0.25]
        elif row["frame_index"] in {4, 5}:
            detection["bbox_norm_xywh"] = [0.10, 0.55, 0.20, 0.25]
        else:
            detection["bbox_norm_xywh"] = [0.40, 0.10, 0.20, 0.25]
    predictions.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = _base_config()
    config["schema_version"] = "deepsafe.content-roi-demo/v2"
    config["title"] = "ÇİT GÜVENLİĞİ"
    config["disclosure"] = {
        "mode": "production_inference",
        "label": "AKTİF ANALİZ",
    }
    config.pop("roi")
    config["rois"] = [
        {
            "id": "restricted-left",
            "label": "SOL YASAK ALAN",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.05, 0.5],
                [0.35, 0.5],
                [0.35, 0.95],
                [0.05, 0.95],
            ],
        },
        {
            "id": "restricted-right",
            "label": "SAĞ YASAK ALAN",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.55, 0.5],
                [0.95, 0.5],
                [0.95, 0.95],
                [0.55, 0.95],
            ],
        },
    ]
    config["event_policy"] = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 2,
        "exit_debounce_frames": 2,
    }
    _write_json(config_path, config)
    return config_path


def _pose_payload(*, violating: bool) -> dict:
    if violating:
        coordinates = {
            "nose": (0.62, 0.18),
            "left_shoulder": (0.55, 0.30),
            "right_shoulder": (0.70, 0.30),
            "left_elbow": (0.52, 0.44),
            "right_elbow": (0.74, 0.44),
            "left_wrist": (0.50, 0.58),
            "right_wrist": (0.20, 0.58),
            "left_hip": (0.56, 0.60),
            "right_hip": (0.69, 0.60),
            "left_knee": (0.55, 0.73),
            "right_knee": (0.70, 0.73),
            "left_ankle": (0.54, 0.86),
            "right_ankle": (0.71, 0.86),
        }
    else:
        coordinates = {
            "nose": (0.26, 0.18),
            "left_shoulder": (0.20, 0.30),
            "right_shoulder": (0.32, 0.30),
            "left_elbow": (0.18, 0.44),
            "right_elbow": (0.34, 0.44),
            "left_wrist": (0.16, 0.58),
            "right_wrist": (0.36, 0.58),
            "left_hip": (0.22, 0.60),
            "right_hip": (0.31, 0.60),
            "left_knee": (0.22, 0.73),
            "right_knee": (0.31, 0.73),
            "left_ankle": (0.21, 0.86),
            "right_ankle": (0.32, 0.86),
        }
    return {
        "score": 0.94,
        "bbox_norm_xywh": [0.10, 0.10, 0.75, 0.80],
        "keypoints": [
            {
                "name": name,
                "x_norm": x,
                "y_norm": y,
                "confidence": 0.92,
                "visible": True,
            }
            for name, (x, y) in coordinates.items()
        ],
        "association_status": "matched",
        "association_iou": 0.78,
        "association_method": "highest_iou_greedy",
    }


def _pose_zone_policy() -> dict:
    return {
        "keypoint_layout": "coco17",
        "selected_keypoints": [
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
        ],
        "inside_ratio_threshold": 0.75,
        "keypoint_confidence_threshold": 0.30,
        "minimum_visible_keypoints": 4,
        "ratio_denominator": "selected_keypoints",
        "roi_aggregation": "union_any",
        "polygon_boundary": "inclusive",
        "person_pose_association": "highest_iou_to_nvdcf_track",
        "insufficient_pose_policy": "no_alert",
    }


def _make_v2_pose_fixture(root: Path) -> Path:
    config_path = _make_fixture(root)
    predictions = root / "data/predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        detection = row["detections"][0]
        detection["track_id"] = 7
        detection["bbox_norm_xywh"] = [0.10, 0.10, 0.75, 0.80]
        detection["pose"] = _pose_payload(
            violating=row["frame_index"] in {2, 3}
        )
    predictions.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    config = _base_config()
    config["schema_version"] = "deepsafe.content-roi-demo/v2"
    config["title"] = "ÇİT GÜVENLİĞİ • POSE"
    config["disclosure"] = {
        "mode": "production_inference",
        "label": "AKTİF ANALİZ",
    }
    config.pop("roi")
    config["rois"] = [
        {
            "id": "restricted-upper",
            "label": "ÜST YASAK ALAN",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.45, 0.15],
                [0.80, 0.15],
                [0.80, 0.65],
                [0.45, 0.65],
            ],
        }
    ]
    config["event_policy"] = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "pose_keypoint_ratio",
        "zone_rule": (
            "selected_pose_keypoint_ratio_inside_restricted_zone"
        ),
        "enter_debounce_frames": 2,
        "exit_debounce_frames": 2,
        "pose_zone_rule": _pose_zone_policy(),
    }
    _write_json(config_path, config)
    return config_path


def _fence_pose_payload(
    *,
    hip_x: float,
    hip_y: float = 0.62,
    wrist_x: float = 0.40,
    wrist_y: float = 0.42,
    knee_y: float = 0.80,
) -> dict:
    coordinates = {
        "left_shoulder": (hip_x - 0.04, hip_y - 0.20),
        "right_shoulder": (hip_x + 0.04, hip_y - 0.20),
        "left_elbow": (wrist_x - 0.02, wrist_y - 0.06),
        "right_elbow": (wrist_x + 0.02, wrist_y - 0.04),
        "left_wrist": (wrist_x, wrist_y),
        "right_wrist": (wrist_x, wrist_y + 0.02),
        "left_hip": (hip_x - 0.01, hip_y),
        "right_hip": (hip_x + 0.01, hip_y),
        "left_knee": (hip_x - 0.01, knee_y),
        "right_knee": (hip_x + 0.01, knee_y + 0.01),
        "left_ankle": (hip_x - 0.01, min(0.98, knee_y + 0.13)),
        "right_ankle": (hip_x + 0.01, min(0.98, knee_y + 0.14)),
    }
    return {
        "score": 0.95,
        "bbox_norm_xywh": [0.10, 0.10, 0.75, 0.80],
        "keypoints": [
            {
                "name": name,
                "x_norm": x,
                "y_norm": y,
                "confidence": 0.95,
                "visible": True,
            }
            for name, (x, y) in coordinates.items()
        ],
        "association_status": "matched",
        "association_iou": 0.82,
        "association_method": "highest_iou_greedy",
    }


def _fence_crossing_policy() -> dict:
    return {
        "enabled": True,
        "boundary_start": {"x": 0.50, "y": 0.10},
        "boundary_end": {"x": 0.50, "y": 0.95},
        "forbidden_side": "right",
        "contact_band": 0.03,
        "minimum_confidence": 0.30,
        "minimum_core_visible": 2,
        "breach_enter_frames": 2,
        "breach_exit_frames": 2,
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
        "climb_enter_frames": 1,
        "climb_exit_frames": 2,
        "history_window_frames": 30,
    }


def _make_v2_staged_fence_fixture(root: Path) -> Path:
    config_path = _make_fixture(root)
    predictions = root / "data/predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    poses = [
        _fence_pose_payload(hip_x=0.40, wrist_x=0.40),
        _fence_pose_payload(hip_x=0.40, wrist_x=0.50),
        _fence_pose_payload(
            hip_x=0.41,
            hip_y=0.55,
            wrist_x=0.50,
            knee_y=0.68,
        ),
        _fence_pose_payload(
            hip_x=0.42,
            hip_y=0.54,
            wrist_x=0.50,
            knee_y=0.66,
        ),
        _fence_pose_payload(
            hip_x=0.55,
            hip_y=0.54,
            wrist_x=0.50,
            knee_y=0.66,
        ),
        _fence_pose_payload(
            hip_x=0.56,
            hip_y=0.54,
            wrist_x=0.50,
            knee_y=0.66,
        ),
        _fence_pose_payload(hip_x=0.40, wrist_x=0.40),
        _fence_pose_payload(hip_x=0.40, wrist_x=0.40),
        _fence_pose_payload(hip_x=0.40, wrist_x=0.40),
        _fence_pose_payload(hip_x=0.40, wrist_x=0.40),
    ]
    for row, pose in zip(rows, poses):
        detection = row["detections"][0]
        detection["track_id"] = 17
        # Its footpoint remains inside the old polygon throughout. Staged
        # alarms must therefore come only from the pose/core fence rule.
        detection["bbox_norm_xywh"] = [0.10, 0.10, 0.75, 0.80]
        detection["pose"] = pose
    predictions.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    config = _base_config()
    config["schema_version"] = "deepsafe.content-roi-demo/v2"
    config["title"] = "ÇİT GÜVENLİĞİ • AŞAMALI"
    config["disclosure"] = {
        "mode": "production_inference",
        "label": "AKTİF ANALİZ",
    }
    config.pop("roi")
    config["rois"] = [
        {
            "id": "legacy-fence-band",
            "label": "ÇİT İZLEME ALANI",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.05, 0.05],
                [0.95, 0.05],
                [0.95, 0.95],
                [0.05, 0.95],
            ],
        }
    ]
    config["event_policy"] = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "mid_hip",
        "zone_rule": "staged_fence_boundary_crossing",
        "enter_debounce_frames": 2,
        "exit_debounce_frames": 2,
        "pose_zone_rule": _pose_zone_policy(),
        "fence_crossing_rule": _fence_crossing_policy(),
    }
    _write_json(config_path, config)
    return config_path


def test_theme_is_the_admin_visual_contract():
    assert THEME.theme_id == THEME_ID == "colt-collbrai-navy-v1"
    assert THEME.brand_name == BRAND_NAME == "COLT AI - COLLBRAI"
    assert THEME.background == "#06152D"
    assert THEME.panel == "#0D2B52"
    assert THEME.safe == "#46C7FF"
    assert THEME.warning == "#789DFF"
    assert THEME.violation == "#FF637D"
    assert hex_to_bgr("#46C7FF") == (255, 199, 70)
    with pytest.raises(KeyError):
        THEME.bgr("brand_name")


def test_overlay_uses_brand_without_source_or_model_text(monkeypatch):
    captured_specs = []

    def capture_text_specs(frame, specs):
        captured_specs.extend(specs)
        return frame

    monkeypatch.setattr(roi_demo, "_apply_text", capture_text_specs)
    config = _base_config()
    config["disclosure"]["label"] = "fixture-model"
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    geometry = roi_demo._geometry(100, 100, 640, 360)
    frame_data = roi_demo.FrameDetections(
        width=100,
        height=100,
        detections=(),
        model_id="fixture-model",
    )

    roi_demo.render_overlay(
        frame,
        frame_data,
        config,
        geometry,
        source_seconds=0.0,
        alert_active=False,
    )

    texts = [spec.text for spec in captured_specs]
    assert texts.count(BRAND_NAME) == 1
    assert "ALAN GÜVENLİĞİ" in texts
    assert config["title"] in texts
    assert all("Kaynak:" not in text for text in texts)
    assert config["source"]["attribution"] not in texts
    assert config["detections"]["expected_model_id"] not in texts
    brand_spec = next(spec for spec in captured_specs if spec.text == BRAND_NAME)
    assert brand_spec.x > geometry.output_width // 2
    assert brand_spec.y > geometry.output_height // 2


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.5, 0.5), True),
        ((0.75, 0.75), True),
        ((0.95, 0.75), True),
        ((0.49, 0.75), False),
        ((0.75, 0.96), False),
    ],
)
def test_point_in_polygon_includes_boundaries(point, expected):
    polygon = [[0.5, 0.5], [0.95, 0.5], [0.95, 0.95], [0.5, 0.95]]
    assert roi_demo.point_in_polygon(point, polygon) is expected


def test_occupancy_debounce_emits_one_enter_and_one_clear():
    state = roi_demo.OccupancyDebouncer(enter_frames=2, exit_frames=3)
    observations = [
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    events = [state.update(value) for value in observations]
    assert [event for event in events if event] == ["zone_enter", "zone_clear"]
    assert state.active is False


def test_config_rejects_unknown_fields(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["unexpected"] = True
    _write_json(config_path, config)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)
    with pytest.raises(roi_demo.DemoError, match="Additional properties"):
        roi_demo.load_config(config_path)


def test_config_rejects_degenerate_polygon(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["roi"]["polygon_norm"] = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    _write_json(config_path, config)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)
    with pytest.raises(roi_demo.DemoError, match="non-zero area"):
        roi_demo.load_config(config_path)


def test_synthetic_video_renders_events_and_manifest(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)
    result = roi_demo.render(config_path)

    assert result["status"] == "rendered"
    assert result["events"] == 2
    assert result["rendered_frames"] == 10
    output = tmp_path / result["output_directory"]
    assert (output / "demo.mp4").stat().st_size > 1000
    assert (output / "preview-safe.png").stat().st_size > 100
    assert (output / "preview-alert.png").stat().st_size > 100

    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "zone_enter",
        "zone_clear",
    ]
    assert events[0]["source_frame"] == 3
    assert events[1]["source_frame"] == 7

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["theme"] == THEME.as_dict()
    assert manifest["statistics"]["rendered_frames"] == 10
    assert manifest["statistics"]["maximum_roi_person_count"] == 1
    assert manifest["gpu_or_model_execution"] is False
    assert manifest["provenance"]["non_commercial_project"] is True


def test_v2_multi_zone_render_uses_tracked_union_transitions(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_fixture(tmp_path)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)

    plan = roi_demo.build_plan(config_path)
    result = roi_demo.render(config_path)

    assert plan["schema_version"] == "deepsafe.content-roi-demo-plan/v2"
    assert plan["multiple_roi_policy"] == "union_any"
    assert plan["tracking_identity"] == "nvdcf_track_id"
    assert len(plan["rois"]) == 2
    assert result["status"] == "rendered"
    assert result["events"] == 2
    output = tmp_path / result["output_directory"]
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        (
            event["transition"],
            event["event_type"],
            event["track_id"],
            event["source_frame"],
        )
        for event in events
    ] == [
        ("started", "restricted_area_intrusion", 7, 3),
        ("ended", "restricted_area_intrusion", 7, 7),
    ]
    assert events[0]["contributing_area_ids"] == ["restricted-right"]
    assert events[1]["contributing_area_ids"] == ["restricted-left"]
    assert all(
        event["schema_version"]
        == "colt-ai.person-safety-transition/v1"
        for event in events
    )
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == (
        "deepsafe.content-roi-demo-result/v2"
    )
    assert manifest["person_zone_contract"] == {
        "engine": "content.person_zone_rules.PersonZoneRuleEngine",
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "multiple_roi_policy": "union_any",
        "event_scope": "per_track_state_transitions",
        "model_name_visible": False,
    }
    assert manifest["statistics"]["event_types"] == [
        "started:restricted_area_intrusion",
        "ended:restricted_area_intrusion",
    ]


def test_v2_pose_rule_propagates_from_predictions_to_events_and_manifest(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_pose_fixture(tmp_path)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)

    config = roi_demo.load_config(config_path)
    frames = roi_demo.load_frame_detections(config)
    detection = frames[2].detections[0]
    assert detection.pose is not None
    assert detection.pose["association_status"] == "matched"
    assert roi_demo.point_in_polygon(
        detection.footpoint,
        config["rois"][0]["polygon_norm"],
    ) is False
    tracked = roi_demo._tracked_persons(frames[2])
    assert tracked[0]["track_id"] == 7
    assert tracked[0]["pose"] == detection.pose

    plan = roi_demo.build_plan(config_path)
    result = roi_demo.render(config_path)

    assert plan["event_policy"]["pose_zone_rule"] == _pose_zone_policy()
    assert plan["multiple_roi_policy"] == "union_any"
    assert result["status"] == "rendered"
    assert result["events"] == 2
    output = tmp_path / result["output_directory"]
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        (
            event["transition"],
            event["event_type"],
            event["source_frame"],
        )
        for event in events
    ] == [
        ("started", "restricted_area_intrusion", 3),
        ("ended", "restricted_area_intrusion", 5),
    ]
    assert events[0]["contributing_area_ids"] == ["restricted-upper"]
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["person_zone_contract"]["person_anchor"] == (
        "pose_keypoint_ratio"
    )
    assert manifest["person_zone_contract"]["pose_zone_rule"] == (
        _pose_zone_policy()
    )
    assert manifest["statistics"]["raw_roi_occupied_frames"] == 2
    assert manifest["statistics"]["maximum_roi_person_count"] == 1


def test_v2_pose_overlay_draws_skeleton_and_marks_inside_keypoints_red(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_pose_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    geometry = roi_demo._geometry(100, 100, 640, 360)
    detection = roi_demo.Detection(
        bbox_norm_xywh=(0.10, 0.10, 0.75, 0.80),
        confidence=0.93,
        object_id="7",
        pose=_pose_payload(violating=True),
    )
    frame_data = roi_demo.FrameDetections(
        width=100,
        height=100,
        detections=(detection,),
        model_id="fixture-model",
    )
    zone_frame = {
        "persons": [
            {
                "track_id": 7,
                "zone_safety": {
                    "violation_active": True,
                    "rules": {
                        "restricted_zone": {"raw_violation": True}
                    },
                    "anchor_evidence": {
                        "mode": "pose_keypoint_ratio",
                        "inside_ratio": 0.75,
                    },
                },
            }
        ]
    }
    circle_colors = []
    line_colors = []
    text_specs = []
    original_circle = cv2.circle
    original_line = cv2.line

    def capture_circle(*args, **kwargs):
        circle_colors.append(args[3])
        return original_circle(*args, **kwargs)

    def capture_line(*args, **kwargs):
        line_colors.append(args[3])
        return original_line(*args, **kwargs)

    def capture_text(frame_value, specs):
        text_specs.extend(specs)
        return frame_value

    monkeypatch.setattr(roi_demo.cv2, "circle", capture_circle)
    monkeypatch.setattr(roi_demo.cv2, "line", capture_line)
    monkeypatch.setattr(roi_demo, "_apply_text", capture_text)

    _, inside_count = roi_demo.render_overlay(
        frame,
        frame_data,
        config,
        geometry,
        source_seconds=0.0,
        alert_active=True,
        zone_frame=zone_frame,
        active_track_count=1,
    )

    assert inside_count == 1
    assert THEME.bgr("violation") in circle_colors
    assert THEME.bgr("safe") in circle_colors
    assert THEME.bgr("violation") in line_colors
    assert THEME.bgr("safe") in line_colors
    assert any("ROI %75" in spec.text for spec in text_specs)


def test_staged_fence_overlay_has_distinct_sides_and_never_bbox_falls_back(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_staged_fence_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    geometry = roi_demo._geometry(100, 100, 640, 360)
    detection = roi_demo.Detection(
        # Footpoint (0.475, 0.90) is inside the legacy ROI and very close to
        # the forbidden half-plane. Missing raw rules must still fail closed.
        bbox_norm_xywh=(0.10, 0.10, 0.75, 0.80),
        confidence=0.93,
        object_id="17",
        pose=_fence_pose_payload(hip_x=0.40, wrist_x=0.50),
    )
    frame_data = roi_demo.FrameDetections(
        width=100,
        height=100,
        detections=(detection,),
        model_id="fixture-model",
    )
    zone_frame = {
        "persons": [
            {
                "track_id": 17,
                "zone_safety": {
                    "violation_active": False,
                    "fence_state": "approach",
                    "fence_evidence": {
                        "contact_keypoint_names": [
                            "left_wrist",
                            "right_wrist",
                        ],
                        "core_point_norm_xy": [0.40, 0.62],
                    },
                    # Deliberately omit rules to exercise the no-fallback path.
                },
            }
        ]
    }
    blend_colors = []
    arrow_colors = []
    text_specs = []
    original_blend = roi_demo._blend_polygon
    original_arrow = roi_demo.cv2.arrowedLine

    def capture_blend(canvas, points, color, *, alpha):
        blend_colors.append((color, alpha))
        original_blend(canvas, points, color, alpha=alpha)

    def capture_arrow(*args, **kwargs):
        arrow_colors.append(args[3])
        return original_arrow(*args, **kwargs)

    def capture_text(frame_value, specs):
        text_specs.extend(specs)
        return frame_value

    monkeypatch.setattr(roi_demo, "_blend_polygon", capture_blend)
    monkeypatch.setattr(roi_demo.cv2, "arrowedLine", capture_arrow)
    monkeypatch.setattr(roi_demo, "_apply_text", capture_text)

    _, inside_count = roi_demo.render_overlay(
        frame,
        frame_data,
        config,
        geometry,
        source_seconds=0.0,
        alert_active=False,
        zone_frame=zone_frame,
        active_track_count=0,
    )

    assert inside_count == 0
    assert any(color == THEME.bgr("safe") for color, _ in blend_colors)
    assert any(
        color == THEME.bgr("violation") for color, _ in blend_colors
    )
    assert arrow_colors == [THEME.bgr("warning")]
    texts = [spec.text for spec in text_specs]
    assert "GÜVENLİ TARAF" in texts
    assert "YASAK TARAF" in texts
    assert "ÇİT SINIRI" in texts
    assert any("YAKLAŞMA" in text for text in texts)
    assert "ÇİT SINIRI İHLALİ" not in texts


def test_staged_fence_render_logs_noncritical_stages_and_breach_only_alarm(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_staged_fence_fixture(tmp_path)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)

    config = roi_demo.load_config(config_path)
    fence_rule = roi_demo._fence_rule_config(config)
    assert fence_rule is not None
    assert fence_rule.boundary_points == ((0.5, 0.1), (0.5, 0.95))
    result = roi_demo.render(config_path)

    assert result["status"] == "rendered"
    output = tmp_path / result["output_directory"]
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        (
            event["transition"],
            event["event_type"],
            event["source_frame"],
        )
        for event in events
    ] == [
        ("started", "fence_approach", 1),
        ("ended", "fence_approach", 2),
        ("started", "fence_climb_attempt", 2),
        ("ended", "fence_climb_attempt", 5),
        ("started", "restricted_area_intrusion", 5),
        ("ended", "restricted_area_intrusion", 7),
    ]
    assert all(
        event["event_type"] != "restricted_area_intrusion"
        for event in events[:4]
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    statistics = manifest["statistics"]
    assert statistics["fence_noncritical_event_count"] == 4
    assert statistics["fence_breach_event_count"] == 2
    assert statistics["fence_state_frame_counts"] == {
        "clear": 4,
        "approach": 1,
        "climbing": 3,
        "breach": 2,
    }
    assert statistics["fence_state_person_frames"] == {
        "clear": 4,
        "approach": 1,
        "climbing": 3,
        "breach": 2,
    }
    assert statistics["alert_active_frames"] == 2
    contract = manifest["person_zone_contract"]
    assert contract["fence_crossing_rule"] == _fence_crossing_policy()
    assert contract["fence_state_contract"] == {
        "states": ["clear", "approach", "climbing", "breach"],
        "critical_state": "breach",
        "critical_event_type": "restricted_area_intrusion",
        "noncritical_event_types": [
            "fence_approach",
            "fence_climb_attempt",
        ],
        "unknown_pose_policy": "retain_track_state_no_transition",
        "bbox_or_polygon_fallback": False,
    }
    assert (output / "preview-safe.png").stat().st_size > 100
    assert (output / "preview-alert.png").stat().st_size > 100


@pytest.mark.parametrize(
    ("state", "alert_active", "expected_status"),
    [
        ("clear", False, "DURUM  GÜVENLİ"),
        ("approach", False, "ÇİT HATTINA YAKLAŞMA"),
        ("climbing", False, "TIRMANMA GİRİŞİMİ"),
        ("breach", True, "ÇİT SINIRI İHLALİ"),
    ],
)
def test_staged_fence_overlay_exposes_all_four_states(
    tmp_path,
    monkeypatch,
    state,
    alert_active,
    expected_status,
):
    config_path = _make_v2_staged_fence_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    geometry = roi_demo._geometry(100, 100, 640, 360)
    frame_data = roi_demo.FrameDetections(
        width=100,
        height=100,
        detections=(
            roi_demo.Detection(
                bbox_norm_xywh=(0.10, 0.10, 0.75, 0.80),
                confidence=0.91,
                object_id="17",
                pose=_fence_pose_payload(
                    hip_x=0.56 if state == "breach" else 0.40,
                    wrist_x=0.50,
                ),
            ),
        ),
        model_id="fixture-model",
    )
    zone_frame = {
        "persons": [
            {
                "track_id": 17,
                "zone_safety": {
                    "violation_active": alert_active,
                    "fence_state": state,
                    "fence_evidence": {
                        "contact_keypoint_names": ["left_wrist"],
                        "core_point_norm_xy": [
                            0.56 if state == "breach" else 0.40,
                            0.62,
                        ],
                    },
                    "rules": {
                        "restricted_zone": {
                            "raw_violation": state == "breach"
                        }
                    },
                },
            }
        ]
    }
    text_specs = []

    def capture_text(frame_value, specs):
        text_specs.extend(specs)
        return frame_value

    monkeypatch.setattr(roi_demo, "_apply_text", capture_text)
    roi_demo.render_overlay(
        frame,
        frame_data,
        config,
        geometry,
        source_seconds=0.0,
        alert_active=alert_active,
        zone_frame=zone_frame,
        active_track_count=1 if alert_active else 0,
    )

    assert any(
        expected_status in spec.text
        for spec in text_specs
    )


def test_v2_overlay_draws_every_restricted_polygon_in_violation_color(
    tmp_path,
    monkeypatch,
):
    config_path = _make_v2_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    captured_colors = []
    original_blend = roi_demo._blend_polygon

    def capture_blend(frame, points, color, *, alpha):
        captured_colors.append(color)
        original_blend(frame, points, color, alpha=alpha)

    monkeypatch.setattr(roi_demo, "_blend_polygon", capture_blend)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    geometry = roi_demo._geometry(100, 100, 640, 360)
    frame_data = roi_demo.FrameDetections(
        width=100,
        height=100,
        detections=(
            roi_demo.Detection(
                bbox_norm_xywh=(0.6, 0.55, 0.2, 0.25),
                confidence=0.9,
                object_id="7",
            ),
        ),
        model_id="fixture-model",
    )
    zone_frame = {
        "persons": [
            {
                "track_id": 7,
                "zone_safety": {"violation_active": True},
            }
        ]
    }

    roi_demo.render_overlay(
        frame,
        frame_data,
        config,
        geometry,
        source_seconds=0.0,
        alert_active=True,
        zone_frame=zone_frame,
        active_track_count=1,
    )

    assert captured_colors == [THEME.bgr("violation")] * 2


def test_interactive_clip_can_render_without_alert_preview(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["roi"]["polygon_norm"] = [
        [0.75, 0.05],
        [0.95, 0.05],
        [0.95, 0.25],
        [0.75, 0.25],
    ]
    _write_json(config_path, config)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)

    with pytest.raises(roi_demo.DemoError, match="never entered the alert state"):
        roi_demo.render(config_path)

    result = roi_demo.render(
        config_path,
        allow_missing_preview_states=True,
    )

    assert result["status"] == "rendered"
    assert result["preview_states"] == ["safe"]
    output = tmp_path / result["output_directory"]
    assert (output / "demo.mp4").stat().st_size > 1000
    assert (output / "preview-safe.png").stat().st_size > 100
    assert not (output / "preview-alert.png").exists()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["preview_states"] == ["safe"]
    assert "preview-safe.png" in manifest["artifacts"]
    assert "preview-alert.png" not in manifest["artifacts"]


def test_plan_fails_when_prediction_dimensions_differ(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    predictions = tmp_path / "data/predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[4]["image_width"] = 101
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)
    with pytest.raises(roi_demo.DemoError, match="dimensions differ"):
        roi_demo.build_plan(config_path)


def test_ground_truth_mode_must_use_ground_truth_source(tmp_path, monkeypatch):
    config_path = _make_fixture(tmp_path)
    config = copy.deepcopy(_base_config())
    config["disclosure"] = {
        "mode": "ground_truth_validation",
        "label": "GT",
    }
    _write_json(config_path, config)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)
    with pytest.raises(
        roi_demo.DemoError,
        match="requires an explicit ground-truth source",
    ):
        roi_demo.load_config(config_path)


def test_meva_geometry_ground_truth_materializes_trimmed_empty_frames(
    tmp_path, monkeypatch
):
    config_path = _make_fixture(tmp_path)
    geometry = tmp_path / "data/fixture.geom.yml"
    types = tmp_path / "data/fixture.types.yml"
    geometry.write_text(
        "\n".join(
            [
                '- { meta: "fixture geometry" }',
                (
                    "- { geom: { id1: 7, id0: 0, ts0: 102, ts1: 20.4, "
                    "g0: 60 55 80 80, src: truth, } }"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    types.write_text(
        "\n".join(
            [
                '- { meta: "fixture types" }',
                "- { types: { id1: 7, cset3: { Person: 1.0 } } }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = _base_config()
    config["disclosure"] = {
        "mode": "ground_truth_validation",
        "label": "MEVA GT",
    }
    config["detections"] = {
        "kind": "meva_geometry_ground_truth",
        "path": "data/fixture.geom.yml",
        "types_path": "data/fixture.types.yml",
        "sequence_id": "fixture",
        "image_width": 100,
        "image_height": 100,
        "source_frame_offset": 100,
        "output_frame_count": 10,
    }
    _write_json(config_path, config)
    monkeypatch.setattr(roi_demo, "REPO_ROOT", tmp_path)

    loaded = roi_demo.load_config(config_path)
    frames = roi_demo.load_frame_detections(loaded)

    assert len(frames) == 10
    assert frames[0].detections == ()
    assert frames[2].detections[0].object_id == "7"
    assert frames[2].detections[0].confidence is None
    assert frames[3].detections == ()
