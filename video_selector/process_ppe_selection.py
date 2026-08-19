#!/usr/bin/env python3
"""Plan or explicitly execute PPE delivery for a saved video selection.

Only queue items that explicitly request the ``ppe`` pipeline are considered.
Planning is intentionally independent of the DeepStream/model artifacts: it
records every dependency that must exist at execution time without requiring
those dependencies to be ready while an operator is still selecting videos.

The command is plan-only unless ``--execute`` is supplied.  No HTTP execution
route or token mechanism is part of this module.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCHEMA_VERSION = "colt-ai.person-processing-queue/v2"
LEGACY_QUEUE_SCHEMA_VERSION = "colt-ai.person-processing-queue/v1"
SUPPORTED_QUEUE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_QUEUE_SCHEMA_VERSION, QUEUE_SCHEMA_VERSION}
)
PLAN_SCHEMA_VERSION = "colt-ai.ppe-selection-plan/v1"
RUN_SCHEMA_VERSION = "colt-ai.ppe-selection-run/v1"
DELIVERY_SCHEMA_VERSION = "colt-ai.ppe-selection-delivery/v1"
DEEPSTREAM_RUNNER = Path("validation/run_ppe_deepstream.py")
PERSON_DEEPSTREAM_RUNNER = Path(
    "validation/run_person_deepstream_direct.py"
)
VIDEO_RENDERER = Path("content/ppe_video.py")
PERSON_PPE_FUSION = Path("content/person_ppe_fusion.py")
PERSON_ZONE_RULES = Path("content/person_zone_rules.py")
FORKLIFT_DRIVER_RULES = Path("content/forklift_driver_rules.py")
CONTAINER_IMAGE = "deepsafe-deepstream:9.0-control-refresh-20260725"
MODEL_ID = "safetyvision-yolov8s-v2"
MODEL_SOURCE_REPOSITORY = "ayushgupta7777/safetyvision-yolov8"
MODEL_SOURCE_COMMIT = "56a71758b55f0e9f2b4b2d6b51a779a1f882da10"
SUPPORTED_PROFILES = (640, 960)
PROFILE_ONNX = {
    profile: Path(
        "models/ppe/safetyvision-yolov8s-v2"
        f"/{profile}/safetyvision-yolov8s-v2-{profile}-ds9-raw6.onnx"
    )
    for profile in SUPPORTED_PROFILES
}
SOURCE_PROFILE_ONNX = {
    profile: Path(
        "validation/results/ppe/models/"
        "safetyvision-yolov8s-v2-cpu-export-r3/"
        "safetyvision-v2-cpu-export-r3-001/artifacts/"
        f"safetyvision-yolov8s-v2-{profile}-bdynamic-opset18.onnx"
    )
    for profile in SUPPORTED_PROFILES
}
LABELS = Path("models/ppe/safetyvision-yolov8s-v2/labels.txt")
PERSON_PROFILE_ENGINE = {
    profile: Path(
        f"models/person/{profile}/yolo11s_b12_gpu0_fp16.engine"
    )
    for profile in SUPPORTED_PROFILES
}
PERSON_PROFILE_ONNX = {
    profile: Path(f"models/person/{profile}/yolo11s.onnx")
    for profile in SUPPORTED_PROFILES
}
PERSON_PROFILE_LABELS = {
    profile: Path(f"models/person/{profile}/labels.txt")
    for profile in SUPPORTED_PROFILES
}
_SELECTION_ID = re.compile(r"^[0-9a-f]{32}$")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$")
_ROI_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


class PpeSelectionError(RuntimeError):
    """The saved PPE selection or delivery state is invalid."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_hash(plan: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(plan))
    body.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_name)
    try:
        os.fchmod(descriptor, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _repository_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    root = root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PpeSelectionError(f"{label} escapes repository root") from exc
    if must_exist and not candidate.is_file():
        raise PpeSelectionError(f"{label} is not a file: {value}")
    return candidate


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root.resolve()).as_posix()


def _load_queue(root: Path, selection_id: str) -> tuple[Path, dict[str, Any]]:
    if not _SELECTION_ID.fullmatch(selection_id):
        raise PpeSelectionError("selection_id must contain 32 lowercase hex digits")
    queue_path = (
        root
        / "content/video-selector/state/queues"
        / f"{selection_id}.json"
    )
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PpeSelectionError(f"cannot load selection queue: {exc}") from exc
    if not isinstance(queue, dict):
        raise PpeSelectionError("selection queue must be an object")
    if queue.get("schema_version") not in SUPPORTED_QUEUE_SCHEMA_VERSIONS:
        raise PpeSelectionError("selection queue schema is unsupported")
    if queue.get("selection_id") != selection_id:
        raise PpeSelectionError("selection_id differs inside queue")
    items = queue.get("items")
    if not isinstance(items, list) or not items:
        raise PpeSelectionError("selection queue has no items")
    return queue_path, queue


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PpeSelectionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PpeSelectionError(f"{label} must be a finite number")
    return result


def _nearest_frame(seconds: float, fps: float) -> int:
    """Map a timestamp to a frame boundary without Python's ties-to-even."""

    return int(
        (Decimal(str(seconds)) * Decimal(str(fps))).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def _polygon_area(points: Sequence[Mapping[str, float]]) -> float:
    return abs(
        sum(
            points[index]["x"] * points[(index + 1) % len(points)]["y"]
            - points[(index + 1) % len(points)]["x"] * points[index]["y"]
            for index in range(len(points))
        )
        / 2.0
    )


def _validate_rois(value: object, *, video_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise PpeSelectionError(f"{video_id}: one to eight ROIs are required")
    result: list[dict[str, Any]] = []
    roi_ids: set[str] = set()
    for raw_roi in value:
        if not isinstance(raw_roi, Mapping):
            raise PpeSelectionError(f"{video_id}: ROI must be an object")
        roi_id = raw_roi.get("roi_id")
        name = raw_roi.get("name")
        raw_points = raw_roi.get("points")
        if not isinstance(roi_id, str) or not _ROI_ID.fullmatch(roi_id):
            raise PpeSelectionError(f"{video_id}: ROI id is invalid")
        if roi_id in roi_ids:
            raise PpeSelectionError(f"{video_id}: ROI ids must be unique")
        roi_ids.add(roi_id)
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            raise PpeSelectionError(f"{video_id}: ROI name is invalid")
        if not isinstance(raw_points, list) or not 3 <= len(raw_points) <= 16:
            raise PpeSelectionError(f"{video_id}: ROI needs 3..16 points")
        points: list[dict[str, float]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, Mapping):
                raise PpeSelectionError(f"{video_id}: ROI point is invalid")
            x = _finite_number(raw_point.get("x"), label=f"{video_id}: ROI x")
            y = _finite_number(raw_point.get("y"), label=f"{video_id}: ROI y")
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise PpeSelectionError(
                    f"{video_id}: ROI points must be normalized"
                )
            points.append({"x": x, "y": y})
        identities = {
            (round(point["x"], 9), round(point["y"], 9)) for point in points
        }
        if len(identities) != len(points):
            raise PpeSelectionError(f"{video_id}: ROI has duplicate points")
        if _polygon_area(points) < 1e-5:
            raise PpeSelectionError(f"{video_id}: ROI area is too small")
        result.append(
            {"roi_id": roi_id, "name": name.strip(), "points": points}
        )
    return result


def _validate_safe_walkways(
    value: object,
    *,
    video_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 8:
        raise PpeSelectionError(
            f"{video_id}: zero to eight safe walkways are required"
        )
    if not value:
        return []
    normalized = _validate_rois(
        [
            {
                "roi_id": raw.get("roi_id") if isinstance(raw, Mapping) else None,
                "name": raw.get("name") if isinstance(raw, Mapping) else None,
                "points": raw.get("points") if isinstance(raw, Mapping) else None,
            }
            for raw in value
        ],
        video_id=video_id,
    )
    for index, raw in enumerate(value):
        if (
            not isinstance(raw, Mapping)
            or raw.get("roi_type") != "safe_walkway"
        ):
            raise PpeSelectionError(
                f"{video_id}: ROI {index + 1} must be safe_walkway"
            )
    return [
        {
            "area_id": roi["roi_id"],
            "roi_id": roi["roi_id"],
            "area_type": "safe_walkway",
            "name": roi["name"],
            "points": roi["points"],
        }
        for roi in normalized
    ]


def _validate_v2_alert_policy(
    value: object,
    *,
    video_id: str,
    walkway_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PpeSelectionError(f"{video_id}: alert_policy is missing")
    expected_rule = (
        "outside_all_safe_walkways" if walkway_enabled else "disabled"
    )
    expected = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": expected_rule,
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "ppe_scope": "all_tracked_persons",
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise PpeSelectionError(
                f"{video_id}: alert_policy.{field} must be {expected_value!r}"
            )
    requirements = value.get("ppe_requirements")
    if requirements != {"helmet": True, "hi_vis": True}:
        raise PpeSelectionError(
            f"{video_id}: PPE requirements must enable helmet and hi_vis"
        )
    suppression = value.get("forklift_driver_suppression")
    if suppression is not None:
        if not isinstance(suppression, Mapping):
            raise PpeSelectionError(
                f"{video_id}: forklift driver suppression must be an object"
            )
        expected_suppression = {
            "enabled": True,
            "forklift_class": "forklift_candidate",
            "detector_evidence": "coco_truck_class_7",
            "classification_scope": "industrial_forklift_candidate",
            "tracking_identity": "nvdcf_track_id",
            "association_rule": "temporal_person_forklift_ioa",
            "maximum_occupants_per_forklift": 1,
            "render_state": "forklift_driver",
            "missing_forklift_evidence": "do_not_suppress",
        }
        for field, expected_value in expected_suppression.items():
            if suppression.get(field) != expected_value:
                raise PpeSelectionError(
                    f"{video_id}: forklift suppression {field} is invalid"
                )
        for field in (
            "minimum_forklift_confidence",
            "minimum_person_ioa",
        ):
            number = suppression.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or not 0.0 < float(number) <= 1.0
            ):
                raise PpeSelectionError(
                    f"{video_id}: forklift suppression {field} is invalid"
                )
        for field, maximum in (
            ("enter_debounce_frames", 30),
            ("exit_debounce_frames", 60),
        ):
            number = suppression.get(field)
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or not 1 <= number <= maximum
            ):
                raise PpeSelectionError(
                    f"{video_id}: forklift suppression {field} is invalid"
                )
        suppressed_alerts = suppression.get("suppressed_alerts")
        if (
            not isinstance(suppressed_alerts, list)
            or not suppressed_alerts
            or len(suppressed_alerts) != len(set(suppressed_alerts))
            or any(
                alert
                not in {"ppe_violation", "safe_walkway_violation"}
                for alert in suppressed_alerts
            )
        ):
            raise PpeSelectionError(
                f"{video_id}: forklift suppressed_alerts is invalid"
            )
    return {
        **dict(value),
        "track_ttl_frames": 15,
        "event_mode": (
            "track_state_transitions" if walkway_enabled else "disabled"
        ),
        "person_visibility_policy": "full_frame_no_zone_filter",
    }


def _legacy_alert_policy() -> dict[str, Any]:
    return {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
        "zone_rule": "disabled",
        "enter_debounce_frames": 6,
        "exit_debounce_frames": 4,
        "track_ttl_frames": 15,
        "ppe_scope": "legacy_roi_visual_filter",
        "event_mode": "disabled",
        "person_visibility_policy": "legacy_roi_visual_filter",
    }


def _normalize_profiles(profiles: Iterable[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(profile) for profile in profiles))
    if not result or any(profile not in SUPPORTED_PROFILES for profile in result):
        raise PpeSelectionError("profiles must contain one or both of 640 and 960")
    return result


def _dependency(path: Path, *, purpose: str) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "purpose": purpose,
        "required_at_execution": True,
        "planning_requires_file": False,
    }


def _is_ppe_item(item: Mapping[str, Any]) -> bool:
    modules = item.get("requested_modules")
    return (
        item.get("pipeline") == "ppe"
        and isinstance(modules, list)
        and "ppe" in modules
    )


def build_plan(
    selection_id: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    video_ids: set[str] | None = None,
    profiles: Iterable[int] = SUPPORTED_PROFILES,
    gpu: int = 0,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """Build a deterministic PPE plan without requiring runtime artifacts."""

    root = repository_root.resolve()
    requested_profiles = _normalize_profiles(profiles)
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise PpeSelectionError("gpu must be a non-negative integer")
    threshold = _finite_number(threshold, label="threshold")
    if not 0.0 < threshold < 1.0:
        raise PpeSelectionError("threshold must be inside (0, 1)")
    if video_ids is not None:
        if not video_ids or any(
            not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id)
            for video_id in video_ids
        ):
            raise PpeSelectionError("video filter contains an invalid ID")

    queue_path, queue = _load_queue(root, selection_id)
    queue_schema_version = str(queue["schema_version"])
    queue_v2 = queue_schema_version == QUEUE_SCHEMA_VERSION
    items = queue["items"]
    all_queue_ids: set[str] = set()
    ppe_items: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise PpeSelectionError("selection queue item must be an object")
        video_id = item.get("video_id")
        if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
            raise PpeSelectionError("selection queue contains an invalid video ID")
        if video_id in all_queue_ids:
            raise PpeSelectionError(f"duplicate queue video ID: {video_id}")
        all_queue_ids.add(video_id)
        declares_ppe = item.get("pipeline") == "ppe" or (
            isinstance(item.get("requested_modules"), list)
            and "ppe" in item["requested_modules"]
        )
        if declares_ppe and not _is_ppe_item(item):
            raise PpeSelectionError(
                f"{video_id}: PPE pipeline/module declaration is inconsistent"
            )
        if _is_ppe_item(item):
            if queue_v2 and item.get("scenario") != "ppe_safety":
                raise PpeSelectionError(
                    f"{video_id}: v2 PPE item scenario must be ppe_safety"
                )
            ppe_items.append(item)

    available_ppe_ids = {str(item["video_id"]) for item in ppe_items}
    if video_ids is not None:
        missing = sorted(video_ids - available_ppe_ids)
        if missing:
            raise PpeSelectionError(
                f"video IDs are not PPE items in this selection: {missing}"
            )
        ppe_items = [
            item for item in ppe_items if str(item["video_id"]) in video_ids
        ]
    if not ppe_items:
        raise PpeSelectionError("selection has no PPE jobs")

    short_id = selection_id[:8]
    profile_slug = "-".join(str(profile) for profile in requested_profiles)
    delivery_root = Path(
        "validation/results/content-deliveries"
        f"/video-selector-{short_id}/ppe"
    )
    jobs: list[dict[str, Any]] = []
    for item in ppe_items:
        video_id = str(item["video_id"])
        source_value = item.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            raise PpeSelectionError(f"{video_id}: source_path is missing")
        source = _repository_path(
            root,
            source_value,
            label=f"{video_id}: source video",
            must_exist=True,
        )
        if source.stat().st_size <= 0:
            raise PpeSelectionError(f"{video_id}: source video is empty")
        source_video = item.get("source_video")
        if not isinstance(source_video, Mapping):
            raise PpeSelectionError(f"{video_id}: source_video is missing")
        width = int(source_video.get("width", 0))
        height = int(source_video.get("height", 0))
        frame_count = int(source_video.get("frame_count", 0))
        fps = _finite_number(
            source_video.get("fps"), label=f"{video_id}: source fps"
        )
        duration = _finite_number(
            source_video.get("duration_seconds"),
            label=f"{video_id}: source duration",
        )
        if (
            width <= 0
            or height <= 0
            or frame_count <= 0
            or fps <= 0
            or duration <= 0
        ):
            raise PpeSelectionError(f"{video_id}: source metadata is invalid")
        clip = item.get("clip")
        if not isinstance(clip, Mapping):
            raise PpeSelectionError(f"{video_id}: clip is missing")
        start = _finite_number(
            clip.get("start_seconds"), label=f"{video_id}: clip start"
        )
        end = _finite_number(
            clip.get("end_seconds"), label=f"{video_id}: clip end"
        )
        requested_start = start
        requested_end = end
        tolerance = max(0.05, 1.0 / fps)
        if start < 0 or end <= start or end > duration + tolerance:
            raise PpeSelectionError(f"{video_id}: clip range is invalid")
        end = min(end, duration)
        if queue_v2:
            safe_walkways = _validate_safe_walkways(
                item.get("rois"),
                video_id=video_id,
            )
            rois: list[dict[str, Any]] = []
            alert_policy = _validate_v2_alert_policy(
                item.get("alert_policy"),
                video_id=video_id,
                walkway_enabled=bool(safe_walkways),
            )
            roi_policy = {
                "selection_preserved": True,
                "inference_scope": "full_frame",
                "render_filter": "none",
                "person_scope": "all_tracked_persons",
                "rule": (
                    "bbox_bottom_center_outside_all_safe_walkways"
                    if safe_walkways
                    else "ppe_only_full_frame"
                ),
                "combination": (
                    "allowed_union" if safe_walkways else None
                ),
            }
        else:
            rois = _validate_rois(item.get("rois"), video_id=video_id)
            safe_walkways = []
            alert_policy = _legacy_alert_policy()
            roi_policy = {
                "selection_preserved": True,
                "inference_scope": "full_frame",
                "render_filter": "object_bbox_center_inside",
                "combination": "any_roi",
            }
        job_root = delivery_root / video_id
        deepstream_root = job_root / f"deepstream-{profile_slug}"
        person_deepstream_root = (
            job_root / f"person-deepstream-{profile_slug}"
        )
        forklift_suppression = alert_policy.get(
            "forklift_driver_suppression"
        )
        forklift_enabled = bool(
            isinstance(forklift_suppression, Mapping)
            and forklift_suppression.get("enabled") is True
        )
        profile_outputs = []
        for profile in requested_profiles:
            person_profile_root = person_deepstream_root / str(profile)
            canonical_person_root = (
                person_profile_root / "person"
                if forklift_enabled
                else person_profile_root
            )
            profile_output = {
                    "profile": profile,
                    "predictions": (
                        deepstream_root / str(profile) / "predictions.jsonl"
                    ).as_posix(),
                    "conversion": (
                        deepstream_root / str(profile) / "conversion.json"
                    ).as_posix(),
                    "person_predictions": (
                        canonical_person_root / "predictions.jsonl"
                    ).as_posix(),
                    "person_conversion": (
                        canonical_person_root / "conversion.json"
                    ).as_posix(),
                    "person_deepstream_manifest": (
                        canonical_person_root / "manifest.json"
                    ).as_posix(),
                    "person_run_root": canonical_person_root.as_posix(),
                    "person_ppe_predictions": (
                        job_root / f"person-ppe-{profile}.jsonl"
                    ).as_posix(),
                    "annotated_video": (
                        job_root
                        / (
                            f"COLT-AI-COLLBRAI-CAM-{video_id}-"
                            f"PPE-{profile}-{short_id}.mp4"
                        )
                    ).as_posix(),
                }
            if forklift_enabled:
                person_vehicle_root = person_profile_root / "person-vehicle"
                profile_output.update(
                    {
                        "person_vehicle_run_root": (
                            person_vehicle_root.as_posix()
                        ),
                        "person_vehicle_predictions": (
                            person_vehicle_root / "predictions.jsonl"
                        ).as_posix(),
                        "vehicle_evidence_predictions": (
                            person_vehicle_root
                            / "predictions-detector-fallback.jsonl"
                        ).as_posix(),
                        "person_vehicle_conversion": (
                            person_vehicle_root / "conversion.json"
                        ).as_posix(),
                        "person_vehicle_deepstream_manifest": (
                            person_vehicle_root / "manifest.json"
                        ).as_posix(),
                    }
                )
                profile_output["person_ppe_effective_predictions"] = (
                    job_root
                    / f"person-ppe-forklift-{profile}.jsonl"
                ).as_posix()
            profile_outputs.append(profile_output)
        # Queue timestamps describe the operator's requested interval and stay
        # untouched for audit.  Runtime timestamps are snapped to source frame
        # boundaries.  Some containers report a duration slightly longer than
        # frame_count / fps (for example N01: 22.271666 vs 22.255567 seconds);
        # passing the container duration would make the DeepStream runner
        # correctly reject the interval as extending beyond the last frame.
        estimated_start_frame = min(
            frame_count - 1, _nearest_frame(start, fps)
        )
        estimated_end_frame = max(
            estimated_start_frame + 1,
            min(frame_count, _nearest_frame(end, fps)),
        )
        estimated_frame_count = (
            estimated_end_frame - estimated_start_frame
        )
        execution_start = estimated_start_frame / fps
        execution_end = estimated_end_frame / fps
        execution_duration = estimated_frame_count / fps
        jobs.append(
            {
                "video_id": video_id,
                "title": str(item.get("title", video_id)),
                "camera_label": f"CAM-{video_id}",
                "pipeline": "ppe",
                "requested_modules": (
                    list(item.get("requested_modules", ["ppe"]))
                    if queue_v2
                    else ["ppe"]
                ),
                "queue_schema_version": queue_schema_version,
                "scenario": (
                    str(item.get("scenario"))
                    if queue_v2
                    else "legacy_ppe_roi_filter"
                ),
                "catalog_revision": item.get("catalog_revision"),
                "source": {
                    "path": _relative(root, source),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                    "source_url": item.get("source_url"),
                    "content_license": item.get("license"),
                    "ground_truth_path": item.get("ground_truth_path"),
                    "metadata": {
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "duration_seconds": duration,
                        "frame_count": frame_count,
                    },
                },
                "clip": {
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": round(end - start, 6),
                    "requested_start_seconds": requested_start,
                    "requested_end_seconds": requested_end,
                    "execution_start_seconds": execution_start,
                    "execution_end_seconds": execution_end,
                    "execution_duration_seconds": execution_duration,
                    "source_frame_span_seconds": frame_count / fps,
                    "estimated_start_frame": estimated_start_frame,
                    "estimated_end_frame_exclusive": estimated_end_frame,
                    "estimated_frame_count": estimated_frame_count,
                    "frame_count_basis": (
                        "queue_metadata_frame_boundaries_half_up"
                    ),
                },
                "rois": rois,
                "safe_walkways": safe_walkways,
                "alert_policy": alert_policy,
                "roi_policy": roi_policy,
                "profile_selection": list(requested_profiles),
                "paths": {
                    "directory": job_root.as_posix(),
                    "deepstream": {
                        "run_root": deepstream_root.as_posix(),
                        "source_clip": (
                            deepstream_root / "source-smoke.mp4"
                        ).as_posix(),
                        "manifest": (
                            deepstream_root / "manifest.json"
                        ).as_posix(),
                    },
                    "person_deepstream": {
                        "run_root": person_deepstream_root.as_posix(),
                    },
                    "profiles": profile_outputs,
                    "delivery_manifest": (
                        job_root / f"ppe-manifest-{profile_slug}.json"
                    ).as_posix(),
                },
            }
        )

    # Keep superseded plans as immutable evidence. HQ v3 also pins the
    # OpenCV-safe Turkish overlay renderer so published walking-path labels
    # cannot degrade into question marks.
    plan_path = (
        delivery_root
        / f"selection-plan-{profile_slug}-person-nvdcf-hq-v3.json"
    )
    forklift_enabled_any = any(
        isinstance(
            job["alert_policy"].get("forklift_driver_suppression"),
            Mapping,
        )
        for job in jobs
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "selection_id": selection_id,
        "queue": {
            "path": _relative(root, queue_path),
            "bytes": queue_path.stat().st_size,
            "sha256": _sha256(queue_path),
            "schema_version": queue_schema_version,
            "created_at_utc": queue.get("created_at_utc"),
            "catalog_revision": queue.get("catalog_revision"),
            "catalog_revisions": queue.get("catalog_revisions"),
        },
        "selection_scope": {
            "pipeline": "ppe",
            "required_module": "ppe",
            "mixed_queue_non_ppe_items_ignored": True,
            "queue_v1_legacy_roi_filter_supported": True,
            "queue_v2_safe_walkway_supported": True,
        },
        "execution_policy": {
            "plan_only_by_default": True,
            "manual_cli_execute_flag": "--execute",
            "http_execute_route": False,
        },
        "model": {
            "id": MODEL_ID,
            "family": "YOLOv8s",
            "source_repository": MODEL_SOURCE_REPOSITORY,
            "source_commit": MODEL_SOURCE_COMMIT,
            "license_id": "AGPL-3.0",
            "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
            "commercially_cleared": False,
            "accepted_model": False,
            "production_ready": False,
            "acceptance_effect": "diagnostic_content_evidence_only",
            "known_limitations": [
                "NO-Safety Vest is the weakest documented target",
                "overhead-camera qualification is not complete",
                "small distant workers may be missed",
            ],
        },
        "runtime": {
            "deepstream_major": 9,
            "container_image": CONTAINER_IMAGE,
            "gpu_index": gpu,
            "precision": "fp16",
            "profiles": list(requested_profiles),
            "display_threshold": threshold,
            "person_centric": {
                "enabled": True,
                "person_source": (
                    "canonical_yolo11s_person_pgie_deepstream9"
                ),
                "ppe_person_class_used": False,
                "person_inference_scope": "same_source_clip_per_profile",
                "tracking_and_alarm_source": "person_ppe_fusion",
                "head_torso_association": True,
                "minimum_person_confidence": 0.25,
                "minimum_person_width": 0.015,
                "minimum_person_height": 0.04,
                "minimum_absence_zone_width": 0.02,
                "minimum_absence_zone_height": 0.025,
                "minimum_zone_visible_fraction": 0.65,
                "alarm_enter_frames": 3,
                "alarm_clear_frames": 3,
                "unknown_after_frames": 8,
                "infer_no_helmet_from_visible_missing": True,
                "inferred_absence_enter_frames": 8,
                "verified_present_missing_grace_frames": 90,
                "infer_no_hi_vis_from_visible_missing": False,
            },
            "safe_walkway": {
                "engine": "content.person_zone_rules.PersonZoneRuleEngine",
                "tracking_identity": "nvdcf_track_id",
                "person_anchor": "bbox_bottom_center",
                "rule": "outside_all_safe_walkways",
                "enter_debounce_frames": 6,
                "exit_debounce_frames": 4,
                "track_ttl_frames": 15,
                "person_scope": "full_frame_no_zone_filter",
                "events": "started_ended_transitions_only",
            },
            "forklift_driver_suppression": {
                "enabled_for_any_job": forklift_enabled_any,
                "vehicle_evidence_source": (
                    "same_yolo11s_pgie_coco_truck_class_7_raw_plus_nvdcf"
                ),
                "semantic_vehicle_class": "forklift_candidate",
                "second_detector_required": False,
                "association_engine": (
                    "content.forklift_driver_rules."
                    "ForkliftDriverRuleEngine"
                ),
                "person_bbox_preserved": True,
                "events": "started_ended_transitions_only",
            },
        },
        "dependencies": [
            _dependency(DEEPSTREAM_RUNNER, purpose="DeepStream 9 PPE runner"),
            _dependency(
                PERSON_DEEPSTREAM_RUNNER,
                purpose="direct DeepStream 9 YOLO11s person runner",
            ),
            _dependency(VIDEO_RENDERER, purpose="COLT AI PPE video renderer"),
            _dependency(
                PERSON_PPE_FUSION,
                purpose="person tracking, PPE association and alarm smoothing",
            ),
            _dependency(
                PERSON_ZONE_RULES,
                purpose="tracked-person safe walkway transition engine",
            ),
            *(
                [
                    _dependency(
                        FORKLIFT_DRIVER_RULES,
                        purpose=(
                            "tracked forklift-driver association and "
                            "effective alert suppression"
                        ),
                    )
                ]
                if forklift_enabled_any
                else []
            ),
            _dependency(LABELS, purpose="PPE class labels"),
            *[
                _dependency(
                    SOURCE_PROFILE_ONNX[profile],
                    purpose=f"{profile} source ONNX",
                )
                for profile in requested_profiles
            ],
            *[
                _dependency(
                    PROFILE_ONNX[profile],
                    purpose=f"{profile} DeepStream adapter ONNX",
                )
                for profile in requested_profiles
            ],
            *[
                _dependency(
                    PERSON_PROFILE_ENGINE[profile],
                    purpose=f"{profile} YOLO11s person TensorRT engine",
                )
                for profile in requested_profiles
            ],
            *[
                _dependency(
                    PERSON_PROFILE_ONNX[profile],
                    purpose=f"{profile} YOLO11s person ONNX",
                )
                for profile in requested_profiles
            ],
            *[
                _dependency(
                    PERSON_PROFILE_LABELS[profile],
                    purpose=f"{profile} YOLO11s person labels",
                )
                for profile in requested_profiles
            ],
        ],
        "visual_contract": {
            "brand_name": "COLT AI - COLLBRAI",
            "theme_id": "colt-collbrai-navy-v1",
            "model_name_visible": False,
        },
        "paths": {"plan": plan_path.as_posix()},
        "jobs": jobs,
    }
    plan["contract_sha256"] = _plan_hash(plan)
    return plan


def persist_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Persist a deterministic plan and reject a conflicting plan in place."""

    if plan.get("contract_sha256") != _plan_hash(plan):
        raise PpeSelectionError("plan contract hash differs")
    raw_path = plan.get("paths", {}).get("plan")
    if not isinstance(raw_path, str):
        raise PpeSelectionError("plan output path is missing")
    path = _repository_path(
        repository_root.resolve(),
        raw_path,
        label="plan output",
        must_exist=False,
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PpeSelectionError(f"cannot read existing plan: {exc}") from exc
        if existing != dict(plan):
            raise PpeSelectionError("plan path already contains a different plan")
        return path
    _atomic_json(path, plan)
    return path


def _pin_output(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PpeSelectionError(f"execution artifact is absent or empty: {path}")
    return {
        "path": _relative(root, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _jsonl_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def execute_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Execute DeepStream and render both requested profile deliveries."""

    root = repository_root.resolve()
    if root != REPOSITORY_ROOT.resolve():
        raise PpeSelectionError(
            "execution is supported only from the active repository root"
        )
    if plan.get("contract_sha256") != _plan_hash(plan):
        raise PpeSelectionError("plan contract hash differs")
    missing = [
        dependency["path"]
        for dependency in plan["dependencies"]
        if not (root / dependency["path"]).is_file()
    ]
    if missing:
        raise PpeSelectionError(
            f"execution dependencies are not ready: {sorted(missing)}"
        )

    # Imports remain execution-local so plan-only operation has no model/runtime
    # dependency beyond Python's standard library.
    from content import (
        forklift_driver_rules,
        person_ppe_fusion,
        ppe_video,
    )
    from validation import run_person_deepstream_direct as person_deepstream
    from validation import run_ppe_deepstream as deepstream

    persist_plan(plan, repository_root=root)
    results: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        manifest_path = root / job["paths"]["delivery_manifest"]
        if manifest_path.is_file():
            try:
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PpeSelectionError(
                    f"{job['video_id']}: cannot read delivery manifest: {exc}"
                ) from exc
            if (
                existing_manifest.get("status") == "complete"
                and existing_manifest.get("plan_contract_sha256")
                == plan["contract_sha256"]
            ):
                results.append(
                    {
                        "video_id": job["video_id"],
                        "status": "resumed",
                        "manifest": _relative(root, manifest_path),
                        "profiles": existing_manifest.get("profiles", []),
                    }
                )
                continue
            raise PpeSelectionError(
                f"{job['video_id']}: delivery manifest conflicts with plan"
            )
        source = root / job["source"]["path"]
        deepstream_root = root / job["paths"]["deepstream"]["run_root"]
        try:
            ds_plan = deepstream.build_plan(
                video=source,
                run_root=deepstream_root,
                profiles=plan["runtime"]["profiles"],
                gpu=int(plan["runtime"]["gpu_index"]),
                start_seconds=float(
                    job["clip"]["execution_start_seconds"]
                ),
                duration_seconds=float(
                    job["clip"]["execution_duration_seconds"]
                ),
                threshold=float(plan["runtime"]["display_threshold"]),
            )
            terminal = deepstream.execute_plan(ds_plan)
        except (
            deepstream.PpeDeepStreamError,
            subprocess.SubprocessError,
        ) as exc:
            raise PpeSelectionError(
                f"{job['video_id']}: DeepStream execution failed: {exc}"
            ) from exc
        if terminal.get("status") != "complete":
            raise PpeSelectionError(
                f"{job['video_id']}: DeepStream terminal is not complete"
            )
        source_clip = root / job["paths"]["deepstream"]["source_clip"]
        profile_results: list[dict[str, Any]] = []
        for profile_output in job["paths"]["profiles"]:
            profile = int(profile_output["profile"])
            predictions = root / profile_output["predictions"]
            person_predictions = root / profile_output["person_predictions"]
            person_vehicle_predictions = (
                root
                / profile_output.get(
                    "person_vehicle_predictions",
                    profile_output["person_predictions"],
                )
            )
            vehicle_evidence_predictions = (
                root
                / profile_output.get(
                    "vehicle_evidence_predictions",
                    profile_output["person_predictions"],
                )
            )
            person_ppe_predictions = (
                root / profile_output["person_ppe_predictions"]
            )
            suppression_policy = job["alert_policy"].get(
                "forklift_driver_suppression"
            )
            forklift_enabled = bool(
                isinstance(suppression_policy, Mapping)
                and suppression_policy.get("enabled") is True
            )
            effective_predictions = (
                root
                / profile_output.get(
                    "person_ppe_effective_predictions",
                    profile_output["person_ppe_predictions"],
                )
            )
            output = root / profile_output["annotated_video"]
            if output.exists():
                raise PpeSelectionError(
                    f"{job['video_id']} profile {profile}: "
                    f"annotated output already exists: {output}"
                )
            try:
                person_plan = person_deepstream.build_plan(
                    video=source_clip,
                    run_root=root / profile_output["person_run_root"],
                    profile=profile,
                    gpu=int(plan["runtime"]["gpu_index"]),
                    sequence_id=(
                        f"ppe-{job['video_id']}-person-{profile}-"
                        f"{plan['selection_id'][:8]}"
                    ),
                    include_forklift_candidates=False,
                )
                person_terminal = person_deepstream.execute_plan(person_plan)
                if person_terminal.get("status") != "complete":
                    raise PpeSelectionError(
                        f"{job['video_id']} profile {profile}: "
                        "person DeepStream terminal is not complete"
                    )
                if forklift_enabled:
                    person_vehicle_plan = person_deepstream.build_plan(
                        video=source_clip,
                        run_root=(
                            root
                            / profile_output["person_vehicle_run_root"]
                        ),
                        profile=profile,
                        gpu=int(plan["runtime"]["gpu_index"]),
                        sequence_id=(
                            f"ppe-{job['video_id']}-person-vehicle-"
                            f"{profile}-{plan['selection_id'][:8]}"
                        ),
                        include_forklift_candidates=True,
                    )
                    person_vehicle_terminal = (
                        person_deepstream.execute_plan(
                            person_vehicle_plan
                        )
                    )
                    if (
                        person_vehicle_terminal.get("status")
                        != "complete"
                    ):
                        raise PpeSelectionError(
                            f"{job['video_id']} profile {profile}: "
                            "person/vehicle DeepStream terminal is not "
                            "complete"
                        )
                    vehicle_fallback_receipt = (
                        forklift_driver_rules.
                        merge_raw_vehicle_detector_fallback(
                            person_vehicle_predictions,
                            (
                                root
                                / profile_output[
                                    "person_vehicle_run_root"
                                ]
                                / "kitti"
                            ),
                            vehicle_evidence_predictions,
                            coordinate_width=int(
                                person_vehicle_plan["streammux"]["width"]
                            ),
                            coordinate_height=int(
                                person_vehicle_plan["streammux"]["height"]
                            ),
                        )
                    )
                else:
                    vehicle_fallback_receipt = None
                fusion_policy = plan["runtime"]["person_centric"]
                fused = person_ppe_fusion.fuse_prediction_streams(
                    ppe_predictions=predictions,
                    person_predictions=person_predictions,
                    output=person_ppe_predictions,
                    config=person_ppe_fusion.PersonPpeFusionConfig(
                        alarm_enter_frames=int(
                            fusion_policy["alarm_enter_frames"]
                        ),
                        alarm_clear_frames=int(
                            fusion_policy["alarm_clear_frames"]
                        ),
                        unknown_after_frames=int(
                            fusion_policy["unknown_after_frames"]
                        ),
                        inferred_absence_enter_frames=int(
                            fusion_policy["inferred_absence_enter_frames"]
                        ),
                        verified_present_missing_grace_frames=int(
                            fusion_policy[
                                "verified_present_missing_grace_frames"
                            ]
                        ),
                        minimum_person_confidence=float(
                            fusion_policy["minimum_person_confidence"]
                        ),
                        minimum_person_width=float(
                            fusion_policy["minimum_person_width"]
                        ),
                        minimum_person_height=float(
                            fusion_policy["minimum_person_height"]
                        ),
                        minimum_absence_zone_width=float(
                            fusion_policy["minimum_absence_zone_width"]
                        ),
                        minimum_absence_zone_height=float(
                            fusion_policy["minimum_absence_zone_height"]
                        ),
                        minimum_zone_visible_fraction=float(
                            fusion_policy["minimum_zone_visible_fraction"]
                        ),
                        infer_no_helmet_from_missing=bool(
                            fusion_policy[
                                "infer_no_helmet_from_visible_missing"
                            ]
                        ),
                        infer_no_hi_vis_from_missing=bool(
                            fusion_policy[
                                "infer_no_hi_vis_from_visible_missing"
                            ]
                        ),
                    ),
                )
                forklift_receipt: dict[str, Any] | None = None
                render_predictions = person_ppe_predictions
                if forklift_enabled:
                    if not isinstance(suppression_policy, Mapping):
                        raise PpeSelectionError(
                            "forklift suppression policy disappeared"
                        )
                    suppressed_alerts = set(
                        suppression_policy["suppressed_alerts"]
                    )
                    forklift_receipt = (
                        forklift_driver_rules.
                        augment_person_ppe_with_tracked_vehicles(
                            person_ppe_predictions,
                            vehicle_evidence_predictions,
                            effective_predictions,
                            config=(
                                forklift_driver_rules.
                                ForkliftDriverConfig(
                                    vehicle_confidence=float(
                                        suppression_policy[
                                            "minimum_forklift_confidence"
                                        ]
                                    ),
                                    person_ioa=float(
                                        suppression_policy[
                                            "minimum_person_ioa"
                                        ]
                                    ),
                                    enter_frames=int(
                                        suppression_policy[
                                            "enter_debounce_frames"
                                        ]
                                    ),
                                    exit_frames=int(
                                        suppression_policy[
                                            "exit_debounce_frames"
                                        ]
                                    ),
                                    suppress_ppe=(
                                        "ppe_violation"
                                        in suppressed_alerts
                                    ),
                                    suppress_walkway=(
                                        "safe_walkway_violation"
                                        in suppressed_alerts
                                    ),
                                )
                            ),
                        )
                    )
                    render_predictions = effective_predictions
                rendered = ppe_video.render(
                    source=source_clip,
                    predictions=render_predictions,
                    output=output,
                    camera_label=job["camera_label"],
                    rois=job["rois"] or None,
                    safe_walkways=job["safe_walkways"],
                    alert_policy=job["alert_policy"],
                )
            except (
                person_deepstream.PersonDeepStreamError,
                person_ppe_fusion.PersonPpeFusionError,
                forklift_driver_rules.ForkliftDriverRuleError,
                ppe_video.PpeVideoError,
                subprocess.SubprocessError,
            ) as exc:
                raise PpeSelectionError(
                    f"{job['video_id']} profile {profile}: "
                    f"rendering failed: {exc}"
                ) from exc
            profile_result = {
                    "profile": profile,
                    "status": "complete",
                    "predictions": {
                        **_pin_output(root, predictions),
                        "frames": _jsonl_count(predictions),
                    },
                    "person_predictions": {
                        **_pin_output(root, person_predictions),
                        "frames": _jsonl_count(person_predictions),
                        "runner": "yolo11s_deepstream9_direct_no_guard",
                    },
                    "person_deepstream_manifest": _pin_output(
                        root,
                        root
                        / profile_output[
                            "person_deepstream_manifest"
                        ],
                    ),
                    "person_conversion": _pin_output(
                        root, root / profile_output["person_conversion"]
                    ),
                    "person_ppe_predictions": {
                        **_pin_output(root, person_ppe_predictions),
                        "frames": _jsonl_count(person_ppe_predictions),
                        "fusion": fused,
                    },
                    "annotated_video": {
                        **_pin_output(root, output),
                        **rendered["video"],
                    },
                    "class_observations": rendered["class_observations"],
                    "event_contract": rendered["event_contract"],
                    "roi_filter": rendered["roi_filter"],
                    "analysis_scope": rendered["analysis_scope"],
                    "safe_walkways": rendered["safe_walkways"],
                    "safety_event_contract": rendered[
                        "safety_event_contract"
                    ],
                    "safety_events": rendered["safety_events"],
                }
            if forklift_receipt is not None:
                profile_result["person_ppe_effective_predictions"] = {
                    **_pin_output(root, effective_predictions),
                    "frames": _jsonl_count(effective_predictions),
                    "forklift_driver_fusion": forklift_receipt,
                }
                profile_result["forklift_driver_suppression"] = {
                    "enabled": True,
                    "detector_evidence": "coco_truck_class_7",
                    "semantic_class": "forklift_candidate",
                    "person_bbox_preserved": True,
                    "policy": dict(suppression_policy),
                }
                profile_result["person_vehicle_predictions"] = {
                    **_pin_output(root, person_vehicle_predictions),
                    "frames": _jsonl_count(
                        person_vehicle_predictions
                    ),
                    "runner": (
                        "yolo11s_deepstream9_person_vehicle_evidence"
                    ),
                }
                profile_result["person_vehicle_deepstream_manifest"] = (
                    _pin_output(
                        root,
                        root
                        / profile_output[
                            "person_vehicle_deepstream_manifest"
                        ],
                    )
                )
                profile_result["person_vehicle_conversion"] = _pin_output(
                    root,
                    root
                    / profile_output["person_vehicle_conversion"],
                )
                profile_result["vehicle_evidence_predictions"] = {
                    **_pin_output(root, vehicle_evidence_predictions),
                    "frames": _jsonl_count(
                        vehicle_evidence_predictions
                    ),
                    "raw_detector_fallback": (
                        vehicle_fallback_receipt
                    ),
                }
            profile_results.append(profile_result)

        manifest = {
            "schema_version": DELIVERY_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "selection_id": plan["selection_id"],
            "plan_contract_sha256": plan["contract_sha256"],
            "video_id": job["video_id"],
            "model": plan["model"],
            "runtime": plan["runtime"],
            "visual_contract": plan["visual_contract"],
            "source": job["source"],
            "clip": job["clip"],
            "analysis_scope": (
                "full_frame_person_ppe_with_safe_walkway_rules"
                if job["safe_walkways"]
                else (
                    "legacy_roi_visual_filter"
                    if job["rois"]
                    else "full_frame_person_ppe"
                )
            ),
            "rois": job["rois"],
            "safe_walkways": job["safe_walkways"],
            "alert_policy": job["alert_policy"],
            "roi_policy": job["roi_policy"],
            "deepstream_manifest": _pin_output(
                root, root / job["paths"]["deepstream"]["manifest"]
            ),
            "profiles": profile_results,
        }
        _atomic_json(manifest_path, manifest)
        results.append(
            {
                "video_id": job["video_id"],
                "status": "complete",
                "manifest": _relative(root, manifest_path),
                "profiles": profile_results,
            }
        )
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "complete",
        "selection_id": plan["selection_id"],
        "model_acceptance": {
            "accepted_model": False,
            "production_ready": False,
            "effect": "diagnostic_content_evidence_only",
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Plan only this PPE video ID; may be repeated.",
    )
    parser.add_argument(
        "--profiles",
        type=int,
        nargs="+",
        choices=SUPPORTED_PROFILES,
        default=list(SUPPORTED_PROFILES),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            args.selection_id,
            video_ids=set(args.video_ids) if args.video_ids else None,
            profiles=args.profiles,
            gpu=args.gpu,
            threshold=args.threshold,
        )
        persist_plan(plan)
        result = execute_plan(plan) if args.execute else plan
    except (OSError, ValueError, KeyError, PpeSelectionError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
