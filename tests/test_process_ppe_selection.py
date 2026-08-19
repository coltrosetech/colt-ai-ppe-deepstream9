from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_selector.process_ppe_selection import (
    PLAN_SCHEMA_VERSION,
    PpeSelectionError,
    build_parser,
    build_plan,
    persist_plan,
)


SELECTION_ID = "a" * 32


def _roi() -> list[dict[str, object]]:
    return [
        {
            "roi_id": "roi-1",
            "name": "Çalışma Alanı",
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.9, "y": 0.1},
                {"x": 0.9, "y": 0.9},
                {"x": 0.1, "y": 0.9},
            ],
        }
    ]


def _safe_walkways() -> list[dict[str, object]]:
    return [
        {
            "roi_id": "walkway-1",
            "name": "Güvenli Yürüyüş Yolu",
            "roi_type": "safe_walkway",
            "points": [
                {"x": 0.1, "y": 0.4},
                {"x": 0.45, "y": 0.4},
                {"x": 0.45, "y": 0.95},
                {"x": 0.1, "y": 0.95},
            ],
        },
        {
            "roi_id": "walkway-2",
            "name": "Güvenli Geçiş",
            "roi_type": "safe_walkway",
            "points": [
                {"x": 0.55, "y": 0.4},
                {"x": 0.9, "y": 0.4},
                {"x": 0.9, "y": 0.95},
                {"x": 0.55, "y": 0.95},
            ],
        },
    ]


def _v2_alert_policy(*, enabled: bool) -> dict[str, object]:
    return {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": (
            "outside_all_safe_walkways" if enabled else "disabled"
        ),
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "ppe_scope": "all_tracked_persons",
        "ppe_requirements": {"helmet": True, "hi_vis": True},
    }


def _ppe_item(video_id: str = "P04") -> dict[str, object]:
    return {
        "video_id": video_id,
        "title": "Şantiye Denetimi",
        "category": "ppe_safety",
        "pipeline": "ppe",
        "requested_modules": ["ppe"],
        "supported_modules": ["ppe"],
        "catalog_revision": "ppe-20260725-r1",
        "source_path": "content/raw/ppe.mp4",
        "source_url": "https://www.pexels.com/video/example/",
        "license": "Pexels",
        "ground_truth_path": None,
        "clip": {"start_seconds": 2.0, "end_seconds": 7.0},
        "rois": _roi(),
        "source_video": {
            "width": 3840,
            "height": 2160,
            "fps": 30.0,
            "duration_seconds": 12.0,
            "frame_count": 360,
        },
    }


def _ppe_item_v2(
    video_id: str = "S04",
    *,
    walkways: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    item = _ppe_item(video_id)
    item["scenario"] = "ppe_safety"
    item["rois"] = walkways or []
    item["alert_policy"] = _v2_alert_policy(enabled=bool(walkways))
    return item


def _ppe_item_v2_with_forklift() -> dict[str, object]:
    item = _ppe_item_v2("S04", walkways=_safe_walkways())
    item["requested_modules"] = ["ppe", "forklift"]
    item["supported_modules"] = ["ppe", "forklift"]
    item["alert_policy"]["forklift_driver_suppression"] = {
        "enabled": True,
        "forklift_class": "forklift_candidate",
        "detector_evidence": "coco_truck_class_7",
        "classification_scope": "industrial_forklift_candidate",
        "tracking_identity": "nvdcf_track_id",
        "association_rule": "temporal_person_forklift_ioa",
        "minimum_forklift_confidence": 0.35,
        "minimum_person_ioa": 0.55,
        "enter_debounce_frames": 4,
        "exit_debounce_frames": 8,
        "maximum_occupants_per_forklift": 1,
        "suppressed_alerts": [
            "ppe_violation",
            "safe_walkway_violation",
        ],
        "render_state": "forklift_driver",
        "missing_forklift_evidence": "do_not_suppress",
    }
    return item


def _office_item() -> dict[str, object]:
    return {
        "video_id": "01",
        "title": "Ofis",
        "category": "person_office",
        "pipeline": "person_roi",
        "requested_modules": ["person_roi", "pose"],
        "source_path": "content/raw/office.mp4",
        "clip": {"start_seconds": 0.0, "end_seconds": 2.0},
        "rois": _roi(),
        "source_video": {
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "duration_seconds": 2.0,
            "frame_count": 50,
        },
    }


def _fixture(
    root: Path,
    *,
    items: list[dict[str, object]] | None = None,
    queue_schema_version: str = "colt-ai.person-processing-queue/v1",
) -> None:
    for name in ("ppe.mp4", "office.mp4"):
        path = root / "content/raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{name}".encode())
    queue = {
        "schema_version": queue_schema_version,
        "selection_id": SELECTION_ID,
        "catalog_revision": "office-r1+ppe-20260725-r1",
        "catalog_revisions": {
            "person_office": "office-r1",
            "ppe_safety": "ppe-20260725-r1",
        },
        "requested_modules": ["person_roi", "pose", "ppe"],
        "created_at_utc": "2026-07-25T18:00:00Z",
        "state": "awaiting_manual_start",
        "execution": {
            "requested": False,
            "started": False,
            "gpu_or_model_execution": False,
        },
        "items": items or [_office_item(), _ppe_item()],
    }
    path = (
        root
        / "content/video-selector/state/queues"
        / f"{SELECTION_ID}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue), encoding="utf-8")


def test_mixed_queue_plans_only_ppe_with_clip_roi_and_profiles(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    plan = build_plan(
        SELECTION_ID,
        repository_root=tmp_path,
        profiles=(640, 960),
    )

    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["runtime"]["profiles"] == [640, 960]
    assert plan["execution_policy"] == {
        "plan_only_by_default": True,
        "manual_cli_execute_flag": "--execute",
        "http_execute_route": False,
    }
    assert [job["video_id"] for job in plan["jobs"]] == ["P04"]
    job = plan["jobs"][0]
    assert job["source"]["path"] == "content/raw/ppe.mp4"
    assert job["source"]["content_license"] == "Pexels"
    assert job["clip"]["start_seconds"] == 2.0
    assert job["clip"]["end_seconds"] == 7.0
    assert job["clip"]["requested_start_seconds"] == 2.0
    assert job["clip"]["requested_end_seconds"] == 7.0
    assert job["clip"]["execution_start_seconds"] == 2.0
    assert job["clip"]["execution_end_seconds"] == 7.0
    assert job["clip"]["execution_duration_seconds"] == 5.0
    assert job["clip"]["estimated_frame_count"] == 150
    assert job["rois"] == _roi()
    assert job["roi_policy"] == {
        "selection_preserved": True,
        "inference_scope": "full_frame",
        "render_filter": "object_bbox_center_inside",
        "combination": "any_roi",
    }
    assert job["profile_selection"] == [640, 960]
    assert [item["profile"] for item in job["paths"]["profiles"]] == [
        640,
        960,
    ]
    assert job["paths"]["deepstream"]["source_clip"].endswith(
        "/source-smoke.mp4"
    )
    assert job["paths"]["person_deepstream"]["run_root"].endswith(
        "/person-deepstream-640-960"
    )
    assert all(
        item["person_predictions"].endswith("/predictions.jsonl")
        and item["person_deepstream_manifest"].endswith("/manifest.json")
        for item in job["paths"]["profiles"]
    )
    assert (
        plan["runtime"]["person_centric"]["person_source"]
        == "canonical_yolo11s_person_pgie_deepstream9"
    )
    assert plan["runtime"]["person_centric"]["ppe_person_class_used"] is False
    person_policy = plan["runtime"]["person_centric"]
    assert person_policy["infer_no_helmet_from_visible_missing"] is True
    assert person_policy["inferred_absence_enter_frames"] == 8
    assert person_policy["verified_present_missing_grace_frames"] == 90
    assert person_policy["unknown_after_frames"] == 8
    assert person_policy["infer_no_hi_vis_from_visible_missing"] is False


def test_full_clip_execution_is_clamped_to_exact_source_frame_span(
    tmp_path: Path,
) -> None:
    item = _ppe_item(video_id="N01")
    item["clip"] = {
        "start_seconds": 0.0,
        "end_seconds": 22.271666,
    }
    item["source_video"] = {
        "width": 1920,
        "height": 1080,
        "fps": 29.97002997002997,
        "duration_seconds": 22.271667,
        "frame_count": 667,
    }
    _fixture(tmp_path, items=[item])

    clip = build_plan(
        SELECTION_ID, repository_root=tmp_path
    )["jobs"][0]["clip"]

    assert clip["requested_end_seconds"] == 22.271666
    assert clip["end_seconds"] == 22.271666
    assert clip["estimated_start_frame"] == 0
    assert clip["estimated_end_frame_exclusive"] == 667
    assert clip["estimated_frame_count"] == 667
    assert clip["execution_start_seconds"] == 0.0
    assert clip["execution_end_seconds"] == pytest.approx(
        667 / 29.97002997002997
    )
    assert clip["execution_duration_seconds"] == pytest.approx(
        667 / 29.97002997002997
    )
    assert (
        clip["execution_start_seconds"]
        + clip["execution_duration_seconds"]
        <= clip["source_frame_span_seconds"] + 1e-12
    )


def test_nonzero_clip_uses_deterministic_frame_boundaries_at_source_end(
    tmp_path: Path,
) -> None:
    item = _ppe_item(video_id="N01")
    item["clip"] = {
        "start_seconds": 20.01,
        "end_seconds": 22.271666,
    }
    item["source_video"] = {
        "width": 1920,
        "height": 1080,
        "fps": 29.97002997002997,
        "duration_seconds": 22.271667,
        "frame_count": 667,
    }
    _fixture(tmp_path, items=[item])

    clip = build_plan(
        SELECTION_ID, repository_root=tmp_path
    )["jobs"][0]["clip"]

    assert clip["requested_start_seconds"] == 20.01
    assert clip["estimated_start_frame"] == 600
    assert clip["estimated_end_frame_exclusive"] == 667
    assert clip["estimated_frame_count"] == 67
    assert clip["execution_start_seconds"] == pytest.approx(
        600 / 29.97002997002997
    )
    assert clip["execution_duration_seconds"] == pytest.approx(
        67 / 29.97002997002997
    )
    assert (
        clip["execution_start_seconds"]
        + clip["execution_duration_seconds"]
        <= clip["source_frame_span_seconds"] + 1e-12
    )


def test_plan_is_honest_and_does_not_require_runtime_artifacts(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, items=[_ppe_item()])

    plan = build_plan(SELECTION_ID, repository_root=tmp_path)

    # The fixture deliberately has no DeepStream runner, renderer, model, or
    # engine. Planning must still succeed and defer those checks to execution.
    assert all(
        dependency["planning_requires_file"] is False
        for dependency in plan["dependencies"]
    )
    dependency_paths = {
        dependency["path"] for dependency in plan["dependencies"]
    }
    assert "validation/run_person_deepstream_direct.py" in dependency_paths
    assert "content/person_zone_rules.py" in dependency_paths
    assert (
        "models/person/640/yolo11s_b12_gpu0_fp16.engine"
        in dependency_paths
    )
    assert (
        "models/person/960/yolo11s_b12_gpu0_fp16.engine"
        in dependency_paths
    )
    assert plan["model"]["license_id"] == "AGPL-3.0"
    assert plan["model"]["commercially_cleared"] is False
    assert plan["model"]["accepted_model"] is False
    assert plan["model"]["production_ready"] is False
    assert (
        plan["model"]["acceptance_effect"]
        == "diagnostic_content_evidence_only"
    )
    assert len(plan["contract_sha256"]) == 64


def test_persisted_plan_records_the_same_immutable_contract(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, items=[_ppe_item()])
    plan = build_plan(
        SELECTION_ID,
        repository_root=tmp_path,
        profiles=(960,),
    )

    path = persist_plan(plan, repository_root=tmp_path)
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert persisted == plan
    assert path.name == "selection-plan-960-person-nvdcf-hq-v3.json"
    assert persist_plan(plan, repository_root=tmp_path) == path


def test_filter_rejects_non_ppe_id_and_inconsistent_ppe_contract(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    with pytest.raises(PpeSelectionError, match="not PPE items"):
        build_plan(
            SELECTION_ID,
            repository_root=tmp_path,
            video_ids={"01"},
        )

    broken = _ppe_item()
    broken["requested_modules"] = ["pose"]
    _fixture(tmp_path, items=[broken])
    with pytest.raises(PpeSelectionError, match="inconsistent"):
        build_plan(SELECTION_ID, repository_root=tmp_path)


def test_invalid_profile_or_roi_is_rejected(tmp_path: Path) -> None:
    _fixture(tmp_path, items=[_ppe_item()])
    with pytest.raises(PpeSelectionError, match="profiles"):
        build_plan(
            SELECTION_ID,
            repository_root=tmp_path,
            profiles=(896,),
        )

    broken = _ppe_item()
    broken["rois"] = _roi()
    broken["rois"][0]["points"][0] = {"x": -0.1, "y": 0.1}
    _fixture(tmp_path, items=[broken])
    with pytest.raises(PpeSelectionError, match="normalized"):
        build_plan(SELECTION_ID, repository_root=tmp_path)


def test_v2_ppe_only_job_has_full_frame_scope_without_walkway(
    tmp_path: Path,
) -> None:
    _fixture(
        tmp_path,
        items=[_ppe_item_v2(walkways=[])],
        queue_schema_version="colt-ai.person-processing-queue/v2",
    )

    plan = build_plan(SELECTION_ID, repository_root=tmp_path)
    job = plan["jobs"][0]

    assert plan["queue"]["schema_version"] == (
        "colt-ai.person-processing-queue/v2"
    )
    assert job["scenario"] == "ppe_safety"
    assert job["rois"] == []
    assert job["safe_walkways"] == []
    assert job["alert_policy"]["zone_rule"] == "disabled"
    assert job["alert_policy"]["enter_debounce_frames"] == 6
    assert job["alert_policy"]["exit_debounce_frames"] == 4
    assert job["roi_policy"] == {
        "selection_preserved": True,
        "inference_scope": "full_frame",
        "render_filter": "none",
        "person_scope": "all_tracked_persons",
        "rule": "ppe_only_full_frame",
        "combination": None,
    }


def test_v2_safe_walkways_are_allowed_union_not_render_filter(
    tmp_path: Path,
) -> None:
    _fixture(
        tmp_path,
        items=[_ppe_item_v2(walkways=_safe_walkways())],
        queue_schema_version="colt-ai.person-processing-queue/v2",
    )

    plan = build_plan(SELECTION_ID, repository_root=tmp_path)
    job = plan["jobs"][0]

    assert job["rois"] == []
    assert [area["area_id"] for area in job["safe_walkways"]] == [
        "walkway-1",
        "walkway-2",
    ]
    assert {
        area["area_type"] for area in job["safe_walkways"]
    } == {"safe_walkway"}
    assert job["alert_policy"]["zone_rule"] == (
        "outside_all_safe_walkways"
    )
    assert job["alert_policy"]["event_mode"] == (
        "track_state_transitions"
    )
    assert job["alert_policy"]["person_visibility_policy"] == (
        "full_frame_no_zone_filter"
    )
    assert job["roi_policy"]["render_filter"] == "none"
    assert job["roi_policy"]["combination"] == "allowed_union"


def test_v2_rejects_restricted_roi_and_tampered_debounce(
    tmp_path: Path,
) -> None:
    wrong_type = _safe_walkways()
    wrong_type[0]["roi_type"] = "restricted_zone"
    _fixture(
        tmp_path,
        items=[_ppe_item_v2(walkways=wrong_type)],
        queue_schema_version="colt-ai.person-processing-queue/v2",
    )
    with pytest.raises(PpeSelectionError, match="safe_walkway"):
        build_plan(SELECTION_ID, repository_root=tmp_path)

    tampered = _ppe_item_v2(walkways=_safe_walkways())
    tampered["alert_policy"]["enter_debounce_frames"] = 1
    _fixture(
        tmp_path,
        items=[tampered],
        queue_schema_version="colt-ai.person-processing-queue/v2",
    )
    with pytest.raises(PpeSelectionError, match="enter_debounce_frames"):
        build_plan(SELECTION_ID, repository_root=tmp_path)


def test_cli_is_plan_only_by_default() -> None:
    args = build_parser().parse_args(["--selection-id", SELECTION_ID])

    assert args.execute is False
    assert args.profiles == [640, 960]


def test_forklift_policy_plans_same_pgie_vehicle_evidence_and_effective_stream(
    tmp_path: Path,
) -> None:
    _fixture(
        tmp_path,
        items=[_ppe_item_v2_with_forklift()],
        queue_schema_version="colt-ai.person-processing-queue/v2",
    )

    plan = build_plan(
        SELECTION_ID,
        repository_root=tmp_path,
        profiles=(960,),
    )
    job = plan["jobs"][0]
    profile = job["paths"]["profiles"][0]

    assert job["requested_modules"] == ["ppe", "forklift"]
    assert profile["person_ppe_effective_predictions"].endswith(
        "/person-ppe-forklift-960.jsonl"
    )
    assert profile["person_predictions"].endswith(
        "/960/person/predictions.jsonl"
    )
    assert profile["person_vehicle_predictions"].endswith(
        "/960/person-vehicle/predictions.jsonl"
    )
    assert profile["vehicle_evidence_predictions"].endswith(
        "/960/person-vehicle/predictions-detector-fallback.jsonl"
    )
    assert (
        profile["person_run_root"]
        != profile["person_vehicle_run_root"]
    )
    assert plan["runtime"]["forklift_driver_suppression"] == {
        "enabled_for_any_job": True,
        "vehicle_evidence_source": (
            "same_yolo11s_pgie_coco_truck_class_7_raw_plus_nvdcf"
        ),
        "semantic_vehicle_class": "forklift_candidate",
        "second_detector_required": False,
        "association_engine": (
            "content.forklift_driver_rules.ForkliftDriverRuleEngine"
        ),
        "person_bbox_preserved": True,
        "events": "started_ended_transitions_only",
    }
    dependency_paths = {
        dependency["path"] for dependency in plan["dependencies"]
    }
    assert "content/forklift_driver_rules.py" in dependency_paths
