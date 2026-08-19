import hashlib
import json
from pathlib import Path

import pytest

from content import roi_demo
from video_selector.process_selection import (
    CONTAINER_IMAGE,
    EXECUTION_MODE,
    MODEL_INPUT,
    TRACKER_NAME,
    SelectionProcessingError,
    SelectionProcessor,
    WorkerLock,
    build_execution_plan,
    convert_tracked_kitti_directory,
)


SELECTION_ID = "d58c660f38d947b9b4db3583c5981889"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(
    root: Path,
    *,
    rois: list[dict] | None = None,
    frame_count: int = 2,
) -> tuple[Path, Path]:
    source = root / "content/raw/office.avi"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture-video")
    model = root / "models/person/960"
    model.mkdir(parents=True)
    (model / "labels.txt").write_text("person\n", encoding="utf-8")
    (model / "yolo11s_b12_gpu0_fp16.engine").write_bytes(b"engine")
    roi = {
        "roi_id": "roi-1",
        "name": "Alan 1",
        "points": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.8, "y": 0.1},
            {"x": 0.8, "y": 0.8},
            {"x": 0.1, "y": 0.8},
        ],
    }
    catalog = {
        "schema_version": "colt-ai.video-catalog/v1",
        "catalog_revision": "office-test-r1",
        "videos": [
            {
                "video_id": "01",
                "title": "Test Ofisi",
                "duration_seconds": frame_count / 25,
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "frame_count": frame_count,
                "processing_source_path": "content/raw/office.avi",
                "ground_truth_path": None,
                "source_url": "https://www.epfl.ch/labs/mmspg/downloads/pevid-hd/",
                "license": "PEViD-HD · araştırma amaçlı",
            }
        ],
    }
    queue = {
        "schema_version": "colt-ai.person-processing-queue/v1",
        "selection_id": SELECTION_ID,
        "catalog_revision": "office-test-r1",
        "catalog_revisions": {"person_office": "office-test-r1"},
        "state": "awaiting_manual_start",
        "execution": {
            "requested": False,
            "started": False,
            "gpu_or_model_execution": False,
        },
        "items": [
            {
                "video_id": "01",
                "title": "Test Ofisi",
                "category": "person_office",
                "pipeline": "person_roi",
                "requested_modules": ["person_roi", "pose"],
                "catalog_revision": "office-test-r1",
                "source_path": "content/raw/office.avi",
                "source_url": "https://www.epfl.ch/labs/mmspg/downloads/pevid-hd/",
                "license": "PEViD-HD · araştırma amaçlı",
                "ground_truth_path": None,
                "clip": {
                    "start_seconds": 0.0,
                    "end_seconds": frame_count / 25,
                },
                "rois": rois if rois is not None else [roi],
                "source_video": {
                    "width": 1920,
                    "height": 1080,
                    "fps": 25.0,
                    "duration_seconds": frame_count / 25,
                    "frame_count": frame_count,
                },
            }
        ],
    }
    catalog_path = root / "video_selector/catalog.json"
    queue_path = (
        root
        / "content/video-selector/state/queues"
        / f"{SELECTION_ID}.json"
    )
    _write_json(catalog_path, catalog)
    _write_json(queue_path, queue)
    return queue_path, catalog_path


def _v2_fixture(
    root: Path,
    *,
    rois: list[dict] | None = None,
    frame_count: int = 2,
) -> tuple[Path, Path]:
    queue_path, catalog_path = _fixture(
        root,
        rois=rois,
        frame_count=frame_count,
    )
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    queue["schema_version"] = "colt-ai.person-processing-queue/v2"
    queue["catalog_revision"] = "office-test-r2"
    queue["catalog_revisions"] = {"person_office": "office-test-r2"}
    item = queue["items"][0]
    item["video_id"] = "F01"
    item["scenario"] = "fence_security"
    item["requested_modules"] = ["person_roi"]
    item["catalog_revision"] = "office-test-r2"
    item["source_url"] = "urn:colt-ai:user-video:F01"
    item["license"] = "Kullanıcı tarafından sağlandı"
    item["alert_policy"] = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
        "ppe_scope": "disabled",
    }
    if rois is None:
        item["rois"][0]["roi_type"] = "restricted_zone"
    catalog["catalog_revision"] = "office-test-r2"
    catalog_item = catalog["videos"][0]
    catalog_item["video_id"] = "F01"
    catalog_item["source_url"] = "urn:colt-ai:user-video:F01"
    catalog_item["license"] = "Kullanıcı tarafından sağlandı"
    _write_json(queue_path, queue)
    _write_json(catalog_path, catalog)
    return queue_path, catalog_path


def _staged_fence_policy() -> dict:
    return {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "mid_hip",
        "zone_rule": "staged_fence_boundary_crossing",
        "enter_debounce_frames": 4,
        "exit_debounce_frames": 4,
        "ppe_scope": "disabled",
        "pose_zone_rule": {
            "keypoint_layout": "coco17",
            "selected_keypoints": [
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
            ],
            "inside_ratio_threshold": 0.5,
            "keypoint_confidence_threshold": 0.25,
            "minimum_visible_keypoints": 2,
            "ratio_denominator": "selected_keypoints",
            "roi_aggregation": "union_any",
            "polygon_boundary": "inclusive",
            "person_pose_association": "highest_iou_to_nvdcf_track",
            "insufficient_pose_policy": "no_alert",
        },
        "fence_crossing_rule": {
            "enabled": True,
            "boundary_start": {"x": 0.1, "y": 0.5},
            "boundary_end": {"x": 0.9, "y": 0.5},
            "forbidden_side": "right",
            "contact_band": 0.03,
            "minimum_confidence": 0.25,
            "minimum_core_visible": 1,
            "breach_enter_frames": 4,
            "breach_exit_frames": 4,
            "approach_keypoint_names": [
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
            ],
            "approach_minimum_count": 1,
            "wrist_contact_required": 1,
            "hip_rise_ratio": 0.08,
            "raised_knee_ratio": 0.1,
            "climb_enter_frames": 2,
            "climb_exit_frames": 2,
            "history_window_frames": 30,
        },
    }


def test_plan_maps_queue_to_direct_ds9_and_roi_render_config(tmp_path: Path) -> None:
    queue, catalog = _fixture(tmp_path)

    plan = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )

    assert plan["execution_mode"] == EXECUTION_MODE
    assert plan["execution_policy"]["parallelism"] == 1
    assert plan["execution_policy"]["legacy_gpu_guard_or_reentry"] is False
    assert plan["tracker"]["backend"] == TRACKER_NAME == "NvDCF-perf"
    assert plan["tracker"]["native_object_id_output"] is True
    job = plan["jobs"][0]
    assert job["container_image"] == CONTAINER_IMAGE
    assert job["docker_command"][:6] == [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        f"colt-video-selector-{SELECTION_ID[:8]}-01-960",
    ]
    assert CONTAINER_IMAGE in job["docker_command"]
    assert not any(
        "guard" in value or "reentry" in value
        for value in job["docker_command"]
    )
    assert job["render_command"][-2:] == [
        "--render",
        "--allow-missing-preview-states",
    ]
    config = job["content_config"]
    assert config["clip"] == {"start_seconds": 0.0, "end_seconds": 0.08}
    assert config["roi"] == {
        "id": "roi-1",
        "label": "ALAN 1",
        "polygon_norm": [
            [0.1, 0.1],
            [0.8, 0.1],
            [0.8, 0.8],
            [0.1, 0.8],
        ],
    }
    assert config["detections"]["expected_model_id"] == "yolo11s-960-fp16"
    assert config["output"]["width"] == 1920
    assert config["output"]["height"] == 1080
    assert config["output"]["crf"] == 14
    assert job["model_input"] == MODEL_INPUT == 960
    assert job["tracker"] == plan["tracker"]
    assert job["paths"]["tracker_kitti"].endswith("/tracker-kitti")
    assert job["paths"]["deliverable"].endswith(
        "COLT-AI-COLLBRAI-CAM-01-d58c660f.mp4"
    )


def test_multiple_rois_are_rejected_before_execution(tmp_path: Path) -> None:
    roi = {
        "roi_id": "roi-1",
        "name": "Alan",
        "points": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.8, "y": 0.1},
            {"x": 0.8, "y": 0.8},
        ],
    }
    queue, catalog = _fixture(
        tmp_path,
        rois=[roi, {**roi, "roi_id": "roi-2"}],
    )

    with pytest.raises(
        SelectionProcessingError,
        match="exactly one ROI is currently supported",
    ):
        build_execution_plan(queue, catalog, repository_root=tmp_path)


def test_v2_fence_plan_preserves_multi_zone_union_and_queue_debounce(
    tmp_path: Path,
) -> None:
    first = {
        "roi_id": "roi-1",
        "name": "Sol Yasak Alan",
        "roi_type": "restricted_zone",
        "points": [
            {"x": 0.05, "y": 0.1},
            {"x": 0.35, "y": 0.1},
            {"x": 0.35, "y": 0.9},
            {"x": 0.05, "y": 0.9},
        ],
    }
    second = {
        "roi_id": "roi-2",
        "name": "Sağ Yasak Alan",
        "roi_type": "restricted_zone",
        "points": [
            {"x": 0.65, "y": 0.1},
            {"x": 0.95, "y": 0.1},
            {"x": 0.95, "y": 0.9},
            {"x": 0.65, "y": 0.9},
        ],
    }
    queue, catalog = _v2_fixture(tmp_path, rois=[first, second])

    plan = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )

    assert plan["execution_policy"]["multiple_roi_policy"] == "union_any"
    job = plan["jobs"][0]
    assert job["scenario"] == "fence_security"
    assert job["rois"] == [first, second]
    assert job["alert_policy"]["tracking_identity"] == "nvdcf_track_id"
    assert job["alert_policy"]["enter_debounce_frames"] == 3
    assert job["alert_policy"]["exit_debounce_frames"] == 6
    config = job["content_config"]
    assert config["schema_version"] == "deepsafe.content-roi-demo/v2"
    assert config["demo_id"].endswith("-f01")
    assert config["title"] == "ÇİT GÜVENLİĞİ"
    assert config["detections"]["sequence_id"].startswith("fence-")
    assert config["event_policy"] == {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "inside_any_restricted_zone",
        "enter_debounce_frames": 3,
        "exit_debounce_frames": 6,
    }
    assert config["rois"] == [
        {
            "id": "roi-1",
            "label": "SOL YASAK ALAN",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.05, 0.1],
                [0.35, 0.1],
                [0.35, 0.9],
                [0.05, 0.9],
            ],
        },
        {
            "id": "roi-2",
            "label": "SAĞ YASAK ALAN",
            "roi_type": "restricted_zone",
            "polygon_norm": [
                [0.65, 0.1],
                [0.95, 0.1],
                [0.95, 0.9],
                [0.65, 0.9],
            ],
        },
    ]
    assert config["source"]["source_url"] == "urn:colt-ai:user-video:F01"
    assert config["source"]["license_id"] == "USER-PROVIDED"
    assert config["source"]["license_url"] == "urn:colt-ai:user-video:F01"
    assert config["source"]["attribution"] == (
        "Kullanıcı tarafından sağlanan video"
    )


def test_v2_pose_policy_routes_render_through_track_fused_keypoints(
    tmp_path: Path,
) -> None:
    queue, catalog = _v2_fixture(tmp_path)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    item = payload["items"][0]
    item["requested_modules"] = ["person_roi", "pose"]
    item["alert_policy"] = {
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
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "inside_ratio_threshold": 0.5,
            "keypoint_confidence_threshold": 0.25,
            "minimum_visible_keypoints": 4,
            "ratio_denominator": "selected_keypoints",
            "roi_aggregation": "union_any",
            "polygon_boundary": "inclusive",
            "person_pose_association": "highest_iou_to_nvdcf_track",
            "insufficient_pose_policy": "no_alert",
        },
    }
    _write_json(queue, payload)

    job = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )["jobs"][0]

    assert job["pose_enabled"] is True
    assert job["requested_modules"] == ["person_roi", "pose"]
    assert job["paths"]["render_predictions"] == (
        job["paths"]["pose_predictions"]
    )
    assert job["content_config"]["detections"]["path"] == (
        job["paths"]["pose_predictions"]
    )
    assert job["pose"] == {
        "model_family": "YOLOX-Pose-S",
        "keypoint_layout": "COCO17",
        "device": "cuda:0",
        "person_association": "highest_iou_to_nvdcf_track",
        "minimum_iou": 0.15,
        "ambiguity_margin": 0.05,
        "pose_frame_offset": 0,
    }
    assert (
        job["content_config"]["event_policy"]["pose_zone_rule"]
        == item["alert_policy"]["pose_zone_rule"]
    )


def test_v2_staged_fence_policy_is_strict_and_reaches_renderer(
    tmp_path: Path,
) -> None:
    queue, catalog = _v2_fixture(tmp_path)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    item = payload["items"][0]
    item["requested_modules"] = ["person_roi", "pose"]
    item["alert_policy"] = _staged_fence_policy()
    _write_json(queue, payload)

    job = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )["jobs"][0]

    assert job["pose_enabled"] is True
    assert job["alert_policy"] == item["alert_policy"]
    assert (
        job["content_config"]["event_policy"]["fence_crossing_rule"]
        == item["alert_policy"]["fence_crossing_rule"]
    )
    assert (
        job["content_config"]["event_policy"]["zone_rule"]
        == "staged_fence_boundary_crossing"
    )

    payload["items"][0]["alert_policy"] = _staged_fence_policy()
    payload["items"][0]["alert_policy"]["enter_debounce_frames"] = 3
    _write_json(queue, payload)
    with pytest.raises(
        SelectionProcessingError,
        match="must match breach debounce",
    ):
        build_execution_plan(queue, catalog, repository_root=tmp_path)


def test_v2_fence_rejects_wrong_roi_type_or_alert_policy(
    tmp_path: Path,
) -> None:
    wrong_type = {
        "roi_id": "roi-1",
        "name": "Yürüyüş Yolu",
        "roi_type": "safe_walkway",
        "points": [
            {"x": 0.1, "y": 0.1},
            {"x": 0.8, "y": 0.1},
            {"x": 0.8, "y": 0.8},
        ],
    }
    queue, catalog = _v2_fixture(
        tmp_path / "wrong-type",
        rois=[wrong_type],
    )
    with pytest.raises(
        SelectionProcessingError,
        match="must be restricted_zone",
    ):
        build_execution_plan(
            queue,
            catalog,
            repository_root=tmp_path / "wrong-type",
        )

    queue, catalog = _v2_fixture(tmp_path / "wrong-policy")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["items"][0]["alert_policy"]["tracking_identity"] = "frame_id"
    _write_json(queue, payload)
    with pytest.raises(
        SelectionProcessingError,
        match="tracking_identity is invalid",
    ):
        build_execution_plan(
            queue,
            catalog,
            repository_root=tmp_path / "wrong-policy",
        )


def test_complete_kitti_repairs_abort_manifest_and_resume_skips_container(
    tmp_path: Path,
) -> None:
    queue, catalog = _fixture(tmp_path)
    plan = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )
    job = plan["jobs"][0]
    kitti = tmp_path / job["paths"]["tracker_kitti"]
    kitti.mkdir(parents=True)
    (kitti / "00_000_000000.txt").write_text("", encoding="utf-8")
    (kitti / "00_000_000001.txt").write_text("", encoding="utf-8")
    manifest = tmp_path / job["paths"]["inference_manifest"]
    _write_json(manifest, {"status": "safety_abort", "error": "legacy guard"})
    calls: list[list[str]] = []

    def forbidden_command(command, _log, _cwd):
        calls.append(list(command))
        raise AssertionError("complete KITTI must not start a container")

    processor = SelectionProcessor(
        plan,
        repository_root=tmp_path,
        command_runner=forbidden_command,
        video_probe=lambda _path: {
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "frames": 2,
        },
        converter=convert_tracked_kitti_directory,
    )

    assert (
        processor._ensure_inference(job)
        == "recovered_complete_tracker_kitti"
    )
    repaired = json.loads(manifest.read_text(encoding="utf-8"))
    assert repaired["status"] == "complete"
    assert repaired["execution_mode"] == "operator_authorized_direct"
    assert (
        repaired["deepstream_action"]
        == "reused_complete_tracker_kitti"
    )
    assert repaired["recovered_previous_manifest"]["status"] == "safety_abort"
    predictions = tmp_path / job["paths"]["predictions"]
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
    assert processor._ensure_inference(job) == "resumed_complete"
    assert calls == []


def test_completed_render_is_resumed_and_only_deliverable_is_published(
    tmp_path: Path,
) -> None:
    queue, catalog = _fixture(tmp_path)
    plan = build_execution_plan(
        queue,
        catalog,
        repository_root=tmp_path,
    )
    job = plan["jobs"][0]
    output = tmp_path / job["paths"]["delivery_directory"]
    output.mkdir(parents=True)
    demo = output / "demo.mp4"
    demo.write_bytes(b"rendered-video")
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "deepsafe.content-roi-demo-result/v1",
            "status": "rendered",
            "demo_id": job["content_config"]["demo_id"],
            "gpu_or_model_execution": False,
            "plan": {
                "config_path": job["paths"]["content_config"],
                "roi": job["content_config"]["roi"],
                "event_policy": job["content_config"]["event_policy"],
                "detections": {"path": job["paths"]["predictions"]},
            },
            "artifacts": {
                "demo.mp4": {
                    "bytes": demo.stat().st_size,
                    "sha256": hashlib.sha256(demo.read_bytes()).hexdigest(),
                }
            },
        },
    )

    def forbidden_command(*_args):
        raise AssertionError("completed render must be skipped")

    processor = SelectionProcessor(
        plan,
        repository_root=tmp_path,
        command_runner=forbidden_command,
    )
    action, deliverable = processor._ensure_render(job)

    assert action == "resumed_complete"
    assert deliverable["path"] == job["paths"]["deliverable"]
    assert (tmp_path / job["paths"]["deliverable"]).read_bytes() == b"rendered-video"


def test_worker_lock_rejects_a_second_worker(tmp_path: Path) -> None:
    path = tmp_path / "worker.lock"
    with WorkerLock(path):
        with pytest.raises(SelectionProcessingError, match="another"):
            with WorkerLock(path):
                pass
