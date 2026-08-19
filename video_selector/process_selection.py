#!/usr/bin/env python3
"""Process one immutable video-selector queue with direct DeepStream 9.

The queue itself never grants execution.  The CLI is validation-only unless
``--execute`` is supplied.  Execution is deliberately simple:

1. run one local DeepStream container at a time;
2. retain NvDCF's native object ids while converting its KITTI dump;
3. render the selected clip and ROI with :mod:`content.roi_demo`; and
4. publish one deterministic MP4 name per selected video.

This runner does not import or invoke the legacy GPU guard/re-entry workflow.
It does, however, require a complete frame-for-frame NvDCF KITTI dump before
an inference stage can be receipted.  Person identity is therefore supplied by
DeepStream tracking rather than being regenerated independently on every
frame.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from validation.kitti_to_jsonl import KITTI_NAME
from validation.run_caviar import (
    attest_engine_load,
    calculate_streammux_dimensions,
    probe_video,
    render_infer_config,
)
from validation.run_person_deepstream_direct import (
    TRACKER_CONFIG,
    TRACKER_DIMENSIONS,
    TRACKER_LIBRARY,
    TRACKER_NAME,
    UNTRACKED_OBJECT_ID,
    _convert_tracked_kitti_directory as convert_tracked_kitti_directory,
    _render_tracked_app_config as render_tracked_app_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_QUEUE_SCHEMA_VERSION = "colt-ai.person-processing-queue/v1"
QUEUE_SCHEMA_VERSION = "colt-ai.person-processing-queue/v2"
SUPPORTED_QUEUE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_QUEUE_SCHEMA_VERSION, QUEUE_SCHEMA_VERSION}
)
CATALOG_SCHEMA_VERSION = "colt-ai.video-catalog/v1"
PLAN_SCHEMA_VERSION = "colt-ai.video-selector-execution-manifest/v1"
STATUS_SCHEMA_VERSION = "colt-ai.video-selector-execution-status/v1"
INFERENCE_SCHEMA_VERSION = "colt-ai.video-selector-direct-inference/v1"
EXECUTION_MODE = "operator_authorized_direct"
CONTAINER_IMAGE = "deepsafe-deepstream:9.0-control-refresh-20260725"
MODEL_INPUT = 960
MODEL_ID = "yolo11s-960-fp16"
EXPORT_THRESHOLD = 0.001
DISPLAY_THRESHOLD = 0.25
PERSON_CATEGORY = "person_office"
PERSON_PIPELINE = "person_roi"
POSE_MODULE = "pose"
_SELECTION_ID = re.compile(r"^[0-9a-f]{32}$")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$")
_CONTENT_ROI_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,39}$")
_EPSILON = 1e-9
_COCO17_KEYPOINTS = frozenset(
    {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
)


class SelectionProcessingError(ValueError):
    """A queue, artifact, resume, or execution contract error."""


CommandRunner = Callable[[Sequence[str], Path, Path], None]
VideoProbe = Callable[[Path], dict[str, object]]
KittiConverter = Callable[..., dict[str, object]]


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path: Path, repository_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionProcessingError(f"artifact is not a file: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise SelectionProcessingError(f"artifact is empty: {path}")
    return {
        "path": path.relative_to(repository_root).as_posix(),
        "bytes": size,
        "sha256": _sha256_file(path),
    }


def _atomic_write(path: Path, content: bytes, *, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_name)
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object, *, mode: int = 0o640) -> None:
    _atomic_write(path, _canonical_json(value), mode=mode)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionProcessingError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionProcessingError(f"{label} must be a JSON object")
    return value


def _inside_repository(
    value: str | Path,
    repository_root: Path,
    *,
    must_exist: bool,
    regular_file: bool = False,
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(repository_root)
    except ValueError as exc:
        raise SelectionProcessingError(
            f"path escapes repository root: {candidate}"
        ) from exc
    cursor = repository_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise SelectionProcessingError(
                f"symlink path components are not allowed: {cursor}"
            )
        if not cursor.exists():
            break
    if must_exist and not candidate.exists():
        raise SelectionProcessingError(f"path does not exist: {candidate}")
    if regular_file and not candidate.is_file():
        raise SelectionProcessingError(f"path is not a regular file: {candidate}")
    return candidate


def _relative(path: Path, repository_root: Path) -> str:
    return path.relative_to(repository_root).as_posix()


def _plan_hash(plan: dict[str, Any]) -> str:
    body = copy.deepcopy(plan)
    body.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _polygon_area(points: list[dict[str, float]]) -> float:
    return abs(
        sum(
            points[index]["x"] * points[(index + 1) % len(points)]["y"]
            - points[(index + 1) % len(points)]["x"] * points[index]["y"]
            for index in range(len(points))
        )
        / 2.0
    )


def _orientation(
    start: dict[str, float],
    end: dict[str, float],
    point: dict[str, float],
) -> float:
    return (end["x"] - start["x"]) * (point["y"] - start["y"]) - (
        end["y"] - start["y"]
    ) * (point["x"] - start["x"])


def _on_segment(
    point: dict[str, float],
    start: dict[str, float],
    end: dict[str, float],
) -> bool:
    return (
        abs(_orientation(start, end, point)) <= _EPSILON
        and min(start["x"], end["x"]) - _EPSILON
        <= point["x"]
        <= max(start["x"], end["x"]) + _EPSILON
        and min(start["y"], end["y"]) - _EPSILON
        <= point["y"]
        <= max(start["y"], end["y"]) + _EPSILON
    )


def _segments_intersect(
    a: dict[str, float],
    b: dict[str, float],
    c: dict[str, float],
    d: dict[str, float],
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    if (
        (ab_c > _EPSILON and ab_d < -_EPSILON)
        or (ab_c < -_EPSILON and ab_d > _EPSILON)
    ) and (
        (cd_a > _EPSILON and cd_b < -_EPSILON)
        or (cd_a < -_EPSILON and cd_b > _EPSILON)
    ):
        return True
    return (
        _on_segment(c, a, b)
        or _on_segment(d, a, b)
        or _on_segment(a, c, d)
        or _on_segment(b, c, d)
    )


def _validate_roi(
    roi: object,
    *,
    video_id: str,
    require_restricted_type: bool,
) -> dict[str, Any]:
    if not isinstance(roi, dict):
        raise SelectionProcessingError(f"{video_id}: ROI must be an object")
    expected_fields = {"roi_id", "name", "points"}
    if require_restricted_type:
        expected_fields.add("roi_type")
    if set(roi) != expected_fields:
        raise SelectionProcessingError(f"{video_id}: ROI fields are invalid")
    roi_id = roi.get("roi_id")
    name = roi.get("name")
    points_value = roi.get("points")
    roi_type = roi.get("roi_type")
    if require_restricted_type and roi_type != "restricted_zone":
        raise SelectionProcessingError(
            f"{video_id}: fence ROI type must be restricted_zone"
        )
    if not isinstance(roi_id, str) or not _CONTENT_ROI_ID.fullmatch(roi_id):
        raise SelectionProcessingError(
            f"{video_id}: ROI id is not compatible with content renderer"
        )
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 40:
        raise SelectionProcessingError(
            f"{video_id}: ROI name must contain 1..40 characters"
        )
    if (
        not isinstance(points_value, list)
        or not 3 <= len(points_value) <= 16
    ):
        raise SelectionProcessingError(f"{video_id}: ROI needs 3..16 points")
    points: list[dict[str, float]] = []
    for point in points_value:
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise SelectionProcessingError(f"{video_id}: ROI point is invalid")
        x = point.get("x")
        y = point.get("y")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or not 0.0 <= float(x) <= 1.0
            or not 0.0 <= float(y) <= 1.0
        ):
            raise SelectionProcessingError(
                f"{video_id}: ROI points must be finite and normalized"
            )
        points.append({"x": float(x), "y": float(y)})
    identities = {(round(p["x"], 9), round(p["y"], 9)) for p in points}
    if len(identities) != len(points):
        raise SelectionProcessingError(f"{video_id}: ROI has duplicate points")
    if _polygon_area(points) < 1e-5:
        raise SelectionProcessingError(f"{video_id}: ROI area is too small")
    count = len(points)
    for first in range(count):
        first_end = (first + 1) % count
        for second in range(first + 1, count):
            second_end = (second + 1) % count
            if (
                first == second
                or first_end == second
                or second_end == first
            ):
                continue
            if _segments_intersect(
                points[first],
                points[first_end],
                points[second],
                points[second_end],
            ):
                raise SelectionProcessingError(
                    f"{video_id}: ROI edges intersect"
                )
    result = {"roi_id": roi_id, "name": name.strip(), "points": points}
    if require_restricted_type:
        result["roi_type"] = "restricted_zone"
    return result


def _validate_fence_alert_policy(
    value: object,
    *,
    video_id: str,
) -> dict[str, Any]:
    base_fields = {
        "tracking_identity",
        "person_anchor",
        "zone_rule",
        "enter_debounce_frames",
        "exit_debounce_frames",
        "ppe_scope",
    }
    if not isinstance(value, dict):
        raise SelectionProcessingError(
            f"{video_id}: fence alert_policy fields are invalid"
        )
    if frozenset(value) not in {
        frozenset(base_fields),
        frozenset(base_fields | {"pose_zone_rule"}),
        frozenset(
            base_fields | {"pose_zone_rule", "fence_crossing_rule"}
        ),
    }:
        raise SelectionProcessingError(
            f"{video_id}: fence alert_policy fields are invalid"
        )
    pose_enabled = "pose_zone_rule" in value
    staged_fence = "fence_crossing_rule" in value
    if staged_fence and not pose_enabled:
        raise SelectionProcessingError(
            f"{video_id}: staged fence policy requires pose_zone_rule"
        )
    expected = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": (
            "mid_hip"
            if staged_fence
            else "pose_keypoint_ratio"
            if pose_enabled
            else "bbox_bottom_center"
        ),
        "zone_rule": (
            "staged_fence_boundary_crossing"
            if staged_fence
            else "selected_pose_keypoint_ratio_inside_restricted_zone"
            if pose_enabled
            else "inside_any_restricted_zone"
        ),
        "ppe_scope": "disabled",
    }
    for key, required in expected.items():
        if value.get(key) != required:
            raise SelectionProcessingError(
                f"{video_id}: fence alert_policy {key} is invalid"
            )
    enter = value.get("enter_debounce_frames")
    exit_frames = value.get("exit_debounce_frames")
    if (
        not isinstance(enter, int)
        or isinstance(enter, bool)
        or not 1 <= enter <= 30
        or not isinstance(exit_frames, int)
        or isinstance(exit_frames, bool)
        or not 1 <= exit_frames <= 60
    ):
        raise SelectionProcessingError(
            f"{video_id}: fence debounce values are invalid"
        )
    if pose_enabled:
        pose = value.get("pose_zone_rule")
        expected_pose = {
            "keypoint_layout": "coco17",
            "ratio_denominator": "selected_keypoints",
            "roi_aggregation": "union_any",
            "polygon_boundary": "inclusive",
            "person_pose_association": "highest_iou_to_nvdcf_track",
            "insufficient_pose_policy": "no_alert",
        }
        if not isinstance(pose, Mapping):
            raise SelectionProcessingError(
                f"{video_id}: pose_zone_rule must be an object"
            )
        for key, required in expected_pose.items():
            if pose.get(key) != required:
                raise SelectionProcessingError(
                    f"{video_id}: pose_zone_rule {key} is invalid"
                )
        selected = pose.get("selected_keypoints")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) != len(set(selected))
            or any(keypoint not in _COCO17_KEYPOINTS for keypoint in selected)
        ):
            raise SelectionProcessingError(
                f"{video_id}: selected pose keypoints are invalid"
            )
        for field, allow_zero in (
            ("inside_ratio_threshold", False),
            ("keypoint_confidence_threshold", True),
        ):
            number = pose.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) > 1.0
                or (float(number) < 0.0 if allow_zero else float(number) <= 0.0)
            ):
                raise SelectionProcessingError(
                    f"{video_id}: pose_zone_rule {field} is invalid"
                )
        minimum_visible = pose.get("minimum_visible_keypoints")
        if (
            not isinstance(minimum_visible, int)
            or isinstance(minimum_visible, bool)
            or not 1 <= minimum_visible <= len(selected)
        ):
            raise SelectionProcessingError(
                f"{video_id}: pose minimum_visible_keypoints is invalid"
            )
    if staged_fence:
        fence = value.get("fence_crossing_rule")
        fence_fields = {
            "enabled",
            "boundary_start",
            "boundary_end",
            "forbidden_side",
            "contact_band",
            "minimum_confidence",
            "minimum_core_visible",
            "breach_enter_frames",
            "breach_exit_frames",
            "approach_keypoint_names",
            "approach_minimum_count",
            "wrist_contact_required",
            "hip_rise_ratio",
            "raised_knee_ratio",
            "climb_enter_frames",
            "climb_exit_frames",
            "history_window_frames",
        }
        if not isinstance(fence, dict) or set(fence) != fence_fields:
            raise SelectionProcessingError(
                f"{video_id}: fence_crossing_rule fields are invalid"
            )
        if fence.get("enabled") is not True:
            raise SelectionProcessingError(
                f"{video_id}: fence_crossing_rule must be enabled"
            )

        def normalized_point(field: str) -> tuple[float, float]:
            point = fence.get(field)
            if not isinstance(point, dict) or set(point) != {"x", "y"}:
                raise SelectionProcessingError(
                    f"{video_id}: fence {field} is invalid"
                )
            coordinates: list[float] = []
            for axis in ("x", "y"):
                number = point.get(axis)
                if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(float(number))
                    or not 0.0 <= float(number) <= 1.0
                ):
                    raise SelectionProcessingError(
                        f"{video_id}: fence {field}.{axis} is invalid"
                    )
                coordinates.append(float(number))
            return coordinates[0], coordinates[1]

        boundary_start = normalized_point("boundary_start")
        boundary_end = normalized_point("boundary_end")
        if math.dist(boundary_start, boundary_end) < 0.01:
            raise SelectionProcessingError(
                f"{video_id}: fence boundary is too short"
            )
        if fence.get("forbidden_side") not in {"left", "right"}:
            raise SelectionProcessingError(
                f"{video_id}: fence forbidden_side is invalid"
            )
        for field, lower, upper, lower_inclusive in (
            ("contact_band", 0.0, 0.25, False),
            ("minimum_confidence", 0.0, 1.0, True),
            ("hip_rise_ratio", 0.0, 1.0, True),
            ("raised_knee_ratio", 0.0, 1.0, True),
        ):
            number = fence.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                raise SelectionProcessingError(
                    f"{video_id}: fence {field} is invalid"
                )
            numeric = float(number)
            below_lower = (
                numeric < lower
                if lower_inclusive
                else numeric <= lower
            )
            if below_lower or numeric > upper:
                raise SelectionProcessingError(
                    f"{video_id}: fence {field} is invalid"
                )
        integer_ranges = {
            "minimum_core_visible": (1, 2),
            "breach_enter_frames": (1, 30),
            "breach_exit_frames": (1, 60),
            "approach_minimum_count": (1, 17),
            "wrist_contact_required": (1, 2),
            "climb_enter_frames": (1, 30),
            "climb_exit_frames": (1, 60),
            "history_window_frames": (2, 300),
        }
        for field, (lower, upper) in integer_ranges.items():
            number = fence.get(field)
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or not lower <= number <= upper
            ):
                raise SelectionProcessingError(
                    f"{video_id}: fence {field} is invalid"
                )
        approach_names = fence.get("approach_keypoint_names")
        if (
            not isinstance(approach_names, list)
            or not approach_names
            or len(approach_names) > len(_COCO17_KEYPOINTS)
            or len(approach_names) != len(set(approach_names))
            or any(
                name not in _COCO17_KEYPOINTS for name in approach_names
            )
        ):
            raise SelectionProcessingError(
                f"{video_id}: fence approach keypoints are invalid"
            )
        if fence["approach_minimum_count"] > len(approach_names):
            raise SelectionProcessingError(
                f"{video_id}: fence approach minimum is invalid"
            )
        if fence["history_window_frames"] < max(
            fence["breach_enter_frames"],
            fence["climb_enter_frames"],
        ):
            raise SelectionProcessingError(
                f"{video_id}: fence history window is too short"
            )
        if (
            enter != fence["breach_enter_frames"]
            or exit_frames != fence["breach_exit_frames"]
        ):
            raise SelectionProcessingError(
                f"{video_id}: fence debounce values must match breach debounce"
            )
    return copy.deepcopy(value)


def _license_fields(catalog_item: dict[str, Any]) -> dict[str, str]:
    license_label = str(catalog_item["license"])
    source_url = str(catalog_item["source_url"])
    if license_label.startswith("MEVA"):
        return {
            "license_id": "CC-BY-4.0",
            "license_url": "https://mevadata.org/resources/MEVA-data-license.txt",
            "attribution": "MEVA / Kitware Inc. & IARPA / CC BY 4.0",
        }
    if license_label.startswith("PEViD-HD"):
        return {
            "license_id": "PEVID-HD-RESEARCH-ONLY",
            "license_url": source_url,
            "attribution": "PEViD-HD / EPFL MMSPG / research use",
        }
    if license_label == "Pexels":
        return {
            "license_id": "PEXELS-LICENSE",
            "license_url": "https://www.pexels.com/license/",
            "attribution": "Pexels source video",
        }
    if license_label == "Kullanıcı tarafından sağlandı":
        return {
            "license_id": "USER-PROVIDED",
            "license_url": source_url,
            "attribution": "Kullanıcı tarafından sağlanan video",
        }
    raise SelectionProcessingError(
        f"unsupported catalog license mapping: {license_label}"
    )


def _asset_id(
    *,
    selection_short: str,
    video_id: str,
    source_path: str,
    source_url: str,
    license_label: str,
) -> str:
    if license_label.startswith("MEVA"):
        match = re.search(r"\.G(?P<camera>[0-9]+)\.", source_url)
        if match is not None:
            return (
                f"meva-g{match['camera'].lower()}-office-{selection_short}"
            )
    if license_label.startswith("PEViD-HD"):
        action = Path(source_path).stem.split("_", 1)[0]
        action = re.sub(r"(?<!^)(?=[A-Z])", "-", action).lower()
        return f"pevid-{action}-office-{selection_short}"
    return f"video-selector-{selection_short}-{video_id}"


def _docker_command(
    *,
    repository_root: Path,
    run_root: Path,
    app_config: Path,
    container_name: str,
    gpu: int,
) -> list[str]:
    run_relative = _relative(run_root, repository_root)
    config_relative = _relative(app_config, repository_root)
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--gpus",
        f"device={gpu}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
        "-v",
        f"{repository_root}:/workspace:ro",
        "-v",
        f"{run_root}:/workspace/{run_relative}:rw",
        "-w",
        "/workspace",
        CONTAINER_IMAGE,
        "deepstream-app",
        "-c",
        f"/workspace/{config_relative}",
    ]


def _content_config(
    *,
    selection_short: str,
    item: dict[str, Any],
    catalog_item: dict[str, Any],
    predictions_path: str,
    delivery_directory: str,
    legacy_queue: bool,
) -> dict[str, Any]:
    licensing = _license_fields(catalog_item)
    video_id = item["video_id"]
    sequence_prefix = "office" if legacy_queue else "fence"
    asset_id = _asset_id(
        selection_short=selection_short,
        video_id=video_id,
        source_path=item["source_path"],
        source_url=item["source_url"],
        license_label=item["license"],
    )
    config: dict[str, Any] = {
        "schema_version": (
            "deepsafe.content-roi-demo/v1"
            if legacy_queue
            else "deepsafe.content-roi-demo/v2"
        ),
        # Content schema identifiers are intentionally lowercase even when the
        # operator-facing catalog uses IDs such as F01.
        "demo_id": f"video-selector-{selection_short}-{video_id}".lower(),
        "title": "İNSAN TESPİTİ" if legacy_queue else "ÇİT GÜVENLİĞİ",
        "camera_label": f"CAM-{video_id}",
        "disclosure": {
            "mode": "production_inference",
            "label": "AKTİF ANALİZ",
        },
        "source": {
            "asset_id": asset_id,
            "video_path": item["source_path"],
            "source_url": item["source_url"],
            **licensing,
            "modification_notice": (
                (
                    "İnsan algılama, kullanıcı ROI alanı ve "
                    "COLT AI - COLLBRAI görsel katmanları eklendi."
                )
                if legacy_queue
                else (
                    "İnsan algılama, takip kimliği, kullanıcı yasak alanları "
                    "ve COLT AI - COLLBRAI görsel katmanları eklendi."
                )
            ),
        },
        "detections": {
            "kind": "predictions_jsonl",
            "path": predictions_path,
            "sequence_id": f"{sequence_prefix}-{selection_short}-{video_id}",
            "confidence_threshold": DISPLAY_THRESHOLD,
            "expected_model_id": MODEL_ID,
        },
        "clip": copy.deepcopy(item["clip"]),
        "output": {
            "directory": delivery_directory,
            # Preserve the selected source's native presentation resolution.
            # The detector still runs with the fixed 960 profile.
            "width": int(item["source_video"]["width"]),
            "height": int(item["source_video"]["height"]),
            "codec": "libx264",
            "crf": 14,
            "preset": "slow",
        },
    }
    if legacy_queue:
        roi = item["rois"][0]
        config["roi"] = {
            "id": roi["roi_id"],
            "label": roi["name"].upper(),
            "polygon_norm": [
                [point["x"], point["y"]] for point in roi["points"]
            ],
        }
        config["event_policy"] = {
            "rule": "bbox_bottom_center_inside",
            "enter_debounce_frames": 3,
            "exit_debounce_frames": 15,
        }
    else:
        config["rois"] = [
            {
                "id": roi["roi_id"],
                "label": roi["name"].upper(),
                "roi_type": roi["roi_type"],
                "polygon_norm": [
                    [point["x"], point["y"]] for point in roi["points"]
                ],
            }
            for roi in item["rois"]
        ]
        alert_policy = item["alert_policy"]
        config["event_policy"] = {
            "tracking_identity": alert_policy["tracking_identity"],
            "person_anchor": alert_policy["person_anchor"],
            "zone_rule": alert_policy["zone_rule"],
            "enter_debounce_frames": alert_policy[
                "enter_debounce_frames"
            ],
            "exit_debounce_frames": alert_policy[
                "exit_debounce_frames"
            ],
        }
        if "pose_zone_rule" in alert_policy:
            config["event_policy"]["pose_zone_rule"] = copy.deepcopy(
                alert_policy["pose_zone_rule"]
            )
        if "fence_crossing_rule" in alert_policy:
            config["event_policy"]["fence_crossing_rule"] = copy.deepcopy(
                alert_policy["fence_crossing_rule"]
            )
    return config


def build_execution_plan(
    queue_path: Path,
    catalog_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    gpu: int = 0,
) -> dict[str, Any]:
    """Validate and map one immutable queue into a deterministic execution plan."""

    repository_root = repository_root.resolve()
    queue_path = _inside_repository(
        queue_path, repository_root, must_exist=True, regular_file=True
    )
    catalog_path = _inside_repository(
        catalog_path, repository_root, must_exist=True, regular_file=True
    )
    queue = _load_json(queue_path, label="processing queue")
    catalog = _load_json(catalog_path, label="video catalog")
    queue_schema_version = queue.get("schema_version")
    if queue_schema_version not in SUPPORTED_QUEUE_SCHEMA_VERSIONS:
        raise SelectionProcessingError("unsupported processing queue schema")
    legacy_queue = queue_schema_version == LEGACY_QUEUE_SCHEMA_VERSION
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise SelectionProcessingError("unsupported video catalog schema")
    selection_id = queue.get("selection_id")
    if not isinstance(selection_id, str) or not _SELECTION_ID.fullmatch(selection_id):
        raise SelectionProcessingError("selection_id must be 32 lowercase hex characters")
    if queue_path.name != f"{selection_id}.json":
        raise SelectionProcessingError(
            "only immutable state/queues/<selection_id>.json may be processed"
        )
    catalog_revision = catalog.get("catalog_revision")
    queue_catalog_revisions = queue.get("catalog_revisions")
    if queue_catalog_revisions is not None:
        if not isinstance(queue_catalog_revisions, dict):
            raise SelectionProcessingError("queue catalog_revisions must be an object")
        queue_person_revision = queue_catalog_revisions.get(PERSON_CATEGORY)
    else:
        # Compatibility with selections saved before the combined selector.
        queue_person_revision = queue.get("catalog_revision")
    if queue_person_revision != catalog_revision:
        raise SelectionProcessingError(
            "queue person_office and office catalog revisions differ"
        )
    if queue.get("state") != "awaiting_manual_start":
        raise SelectionProcessingError("queue is not awaiting manual start")
    if queue.get("execution") != {
        "requested": False,
        "started": False,
        "gpu_or_model_execution": False,
    }:
        raise SelectionProcessingError("queue execution flags are not closed")
    catalog_rows = catalog.get("videos")
    items = queue.get("items")
    if not isinstance(catalog_rows, list) or not isinstance(items, list) or not items:
        raise SelectionProcessingError("queue or catalog has no video items")
    if any(not isinstance(item, dict) for item in items):
        raise SelectionProcessingError("queue item must be an object")
    if legacy_queue:
        person_items = [
            item
            for item in items
            if item.get("category") == PERSON_CATEGORY
            and item.get("pipeline") == PERSON_PIPELINE
            and (
                not isinstance(item.get("requested_modules"), list)
                or PERSON_PIPELINE in item["requested_modules"]
            )
        ]
    else:
        person_items = [
            item
            for item in items
            if item.get("category") == PERSON_CATEGORY
            and item.get("scenario") == "fence_security"
            and item.get("pipeline") == PERSON_PIPELINE
            and isinstance(item.get("requested_modules"), list)
            and PERSON_PIPELINE in item["requested_modules"]
        ]
    if not person_items:
        raise SelectionProcessingError(
            "queue has no requested fence_security/person_roi items"
        )
    catalog_by_id = {
        str(item.get("video_id")): item
        for item in catalog_rows
        if isinstance(item, dict)
    }
    if len(catalog_by_id) != len(catalog_rows):
        raise SelectionProcessingError("catalog video ids are duplicated or invalid")

    selection_short = selection_id[:8]
    for required_model_artifact in (
        "models/person/960/yolo11s_b12_gpu0_fp16.engine",
        "models/person/960/labels.txt",
    ):
        _inside_repository(
            required_model_artifact,
            repository_root,
            must_exist=True,
            regular_file=True,
        )
    control_root = (
        repository_root
        / "validation/results/content-processing"
        / f"video-selector-{selection_short}"
    )
    inference_base = (
        repository_root
        / "validation/results/content-inference"
        / f"video-selector-{selection_short}"
    )
    delivery_base = (
        repository_root
        / "validation/results/content-deliveries"
        / f"video-selector-{selection_short}"
    )
    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_item in person_items:
        video_id = raw_item.get("video_id")
        if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
            raise SelectionProcessingError("queue video_id is invalid")
        if video_id in seen_ids:
            raise SelectionProcessingError(f"duplicate queue video: {video_id}")
        seen_ids.add(video_id)
        catalog_item = catalog_by_id.get(video_id)
        if catalog_item is None:
            raise SelectionProcessingError(f"video absent from catalog: {video_id}")
        if (
            raw_item.get("catalog_revision") is not None
            and raw_item.get("catalog_revision") != catalog_revision
        ):
            raise SelectionProcessingError(
                f"{video_id}: queue item catalog revision differs from office catalog"
            )
        expected_bindings = {
            "title": catalog_item.get("title"),
            "source_path": catalog_item.get("processing_source_path"),
            "source_url": catalog_item.get("source_url"),
            "license": catalog_item.get("license"),
            "ground_truth_path": catalog_item.get("ground_truth_path"),
        }
        for key, expected in expected_bindings.items():
            if raw_item.get(key) != expected:
                raise SelectionProcessingError(
                    f"{video_id}: queue {key} differs from catalog"
                )
        expected_video = {
            "width": catalog_item.get("width"),
            "height": catalog_item.get("height"),
            "fps": catalog_item.get("fps"),
            "duration_seconds": catalog_item.get("duration_seconds"),
            "frame_count": catalog_item.get("frame_count"),
        }
        if raw_item.get("source_video") != expected_video:
            raise SelectionProcessingError(
                f"{video_id}: queue source metadata differs from catalog"
            )
        source = _inside_repository(
            str(raw_item["source_path"]),
            repository_root,
            must_exist=True,
            regular_file=True,
        )
        rois = raw_item.get("rois")
        if not isinstance(rois, list):
            raise SelectionProcessingError(f"{video_id}: rois must be a list")
        if legacy_queue:
            if len(rois) != 1:
                raise SelectionProcessingError(
                    f"{video_id}: exactly one ROI is currently supported; "
                    f"got {len(rois)}"
                )
        elif not 1 <= len(rois) <= 8:
            raise SelectionProcessingError(
                f"{video_id}: fence runtime requires one to eight ROIs"
            )
        normalized_rois = [
            _validate_roi(
                roi,
                video_id=video_id,
                require_restricted_type=not legacy_queue,
            )
            for roi in rois
        ]
        alert_policy = (
            None
            if legacy_queue
            else _validate_fence_alert_policy(
                raw_item.get("alert_policy"),
                video_id=video_id,
            )
        )
        requested_modules = raw_item.get("requested_modules")
        if (
            not isinstance(requested_modules, list)
            or any(
                not isinstance(module, str) or not module
                for module in requested_modules
            )
            or len(set(requested_modules)) != len(requested_modules)
        ):
            raise SelectionProcessingError(
                f"{video_id}: requested_modules is invalid"
            )
        pose_enabled = bool(
            not legacy_queue
            and alert_policy is not None
            and "pose_zone_rule" in alert_policy
        )
        if not legacy_queue and (
            (POSE_MODULE in requested_modules) != pose_enabled
        ):
            raise SelectionProcessingError(
                f"{video_id}: pose module and pose_zone_rule must agree"
            )
        clip = raw_item.get("clip")
        if not isinstance(clip, dict) or set(clip) != {
            "start_seconds",
            "end_seconds",
        }:
            raise SelectionProcessingError(f"{video_id}: clip is invalid")
        start = clip.get("start_seconds")
        end = clip.get("end_seconds")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise SelectionProcessingError(f"{video_id}: clip interval is invalid")
        fps = float(expected_video["fps"])
        start_frame = int(round(float(start) * fps))
        end_frame = int(round(float(end) * fps))
        frame_count = int(expected_video["frame_count"])
        if start_frame < 0 or end_frame > frame_count or end_frame <= start_frame:
            raise SelectionProcessingError(
                f"{video_id}: clip frame interval exceeds the source"
            )
        item = copy.deepcopy(raw_item)
        item["clip"] = {
            "start_seconds": float(start),
            "end_seconds": float(end),
        }
        item["rois"] = normalized_rois
        if alert_policy is not None:
            item["alert_policy"] = alert_policy
        width = int(expected_video["width"])
        height = int(expected_video["height"])
        mux_width, mux_height, mux_scale = calculate_streammux_dimensions(
            width,
            height,
            MODEL_INPUT,
            policy="model-active-area",
        )
        run_root = inference_base / video_id / str(MODEL_INPUT)
        generated = run_root / "generated"
        app_config = generated / "deepstream-app.txt"
        infer_config = generated / "config-infer-primary.txt"
        predictions = run_root / "predictions.jsonl"
        delivery_directory = delivery_base / video_id
        pose_directory = delivery_base / "pose" / video_id
        pose_keypoints = pose_directory / "keypoints.jsonl"
        pose_predictions = run_root / "pose-track-predictions.jsonl"
        render_predictions = (
            pose_predictions if pose_enabled else predictions
        )
        content_config_path = (
            repository_root
            / "content/configs"
            / f"video-selector-{selection_short}-{video_id}.json"
        )
        deliverable_name = (
            f"COLT-AI-COLLBRAI-CAM-{video_id}-{selection_short}.mp4"
        )
        content_config = _content_config(
            selection_short=selection_short,
            item=item,
            catalog_item=catalog_item,
            predictions_path=_relative(
                render_predictions,
                repository_root,
            ),
            delivery_directory=_relative(delivery_directory, repository_root),
            legacy_queue=legacy_queue,
        )
        job = {
            "video_id": video_id,
            "title": raw_item["title"],
            "sequence_id": content_config["detections"]["sequence_id"],
            "source_path": _relative(source, repository_root),
            "source_video": copy.deepcopy(expected_video),
            "clip": copy.deepcopy(item["clip"]),
            "requested_modules": list(requested_modules),
            "pose_enabled": pose_enabled,
            "model_input": MODEL_INPUT,
            "model_id": MODEL_ID,
            "container_image": CONTAINER_IMAGE,
            "streammux": {
                "width": mux_width,
                "height": mux_height,
                "source_to_mux_scale": mux_scale,
                "policy": "model-active-area",
            },
            "tracker": {
                "backend": TRACKER_NAME,
                "library": TRACKER_LIBRARY,
                "config": TRACKER_CONFIG,
                "width": TRACKER_DIMENSIONS[MODEL_INPUT][0],
                "height": TRACKER_DIMENSIONS[MODEL_INPUT][1],
                "native_object_id_output": True,
            },
            "paths": {
                "run_root": _relative(run_root, repository_root),
                "kitti": _relative(run_root / "kitti", repository_root),
                "tracker_kitti": _relative(
                    run_root / "tracker-kitti", repository_root
                ),
                "predictions": _relative(predictions, repository_root),
                "render_predictions": _relative(
                    render_predictions,
                    repository_root,
                ),
                "conversion": _relative(run_root / "conversion.json", repository_root),
                "inference_manifest": _relative(
                    run_root / "run-manifest.json", repository_root
                ),
                "deepstream_log": _relative(
                    run_root / "deepstream.log", repository_root
                ),
                "deepstream_config": _relative(app_config, repository_root),
                "infer_config": _relative(infer_config, repository_root),
                "content_config": _relative(
                    content_config_path, repository_root
                ),
                "delivery_directory": _relative(
                    delivery_directory, repository_root
                ),
                "deliverable": _relative(
                    delivery_base / "deliverables" / deliverable_name,
                    repository_root,
                ),
            },
            "content_config": content_config,
        }
        if pose_enabled:
            job["paths"].update(
                {
                    "pose_directory": _relative(
                        pose_directory,
                        repository_root,
                    ),
                    "pose_keypoints": _relative(
                        pose_keypoints,
                        repository_root,
                    ),
                    "pose_predictions": _relative(
                        pose_predictions,
                        repository_root,
                    ),
                    "pose_fusion_receipt": _relative(
                        run_root / "pose-track-fusion.json",
                        repository_root,
                    ),
                }
            )
            job["pose"] = {
                "model_family": "YOLOX-Pose-S",
                "keypoint_layout": "COCO17",
                "device": f"cuda:{gpu}",
                "person_association": "highest_iou_to_nvdcf_track",
                "minimum_iou": 0.15,
                "ambiguity_margin": 0.05,
                # Both streams use source-absolute frame indices.
                "pose_frame_offset": 0,
            }
        if legacy_queue:
            job["roi"] = copy.deepcopy(content_config["roi"])
        else:
            job["scenario"] = "fence_security"
            job["rois"] = copy.deepcopy(item["rois"])
            job["alert_policy"] = copy.deepcopy(alert_policy)
        job["docker_command"] = _docker_command(
            repository_root=repository_root,
            run_root=run_root,
            app_config=app_config,
            container_name=(
                f"colt-video-selector-{selection_short}-{video_id}-960"
            ),
            gpu=gpu,
        )
        job["render_command"] = [
            sys.executable,
            "-m",
            "content.roi_demo",
            "--config",
            job["paths"]["content_config"],
            "--render",
            "--allow-missing-preview-states",
        ]
        jobs.append(job)

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "selection_id": selection_id,
        "catalog_revision": catalog_revision,
        "queue_catalog_revision": queue["catalog_revision"],
        "queue_catalog_revisions": copy.deepcopy(queue_catalog_revisions),
        "execution_mode": EXECUTION_MODE,
        "execution_policy": {
            "requires_explicit_execute": True,
            "parallelism": 1,
            "resume_completed_stages": True,
            "legacy_gpu_guard_or_reentry": False,
            "multiple_roi_policy": (
                "reject" if legacy_queue else "union_any"
            ),
        },
        "queue": _pin(queue_path, repository_root),
        "catalog": _pin(catalog_path, repository_root),
        "control_root": _relative(control_root, repository_root),
        "container_image": CONTAINER_IMAGE,
        "model": {
            "input": MODEL_INPUT,
            "model_id": MODEL_ID,
            "engine": "models/person/960/yolo11s_b12_gpu0_fp16.engine",
            "labels": "models/person/960/labels.txt",
            "export_threshold": EXPORT_THRESHOLD,
            "display_threshold": DISPLAY_THRESHOLD,
        },
        "pose_requested": any(job["pose_enabled"] for job in jobs),
        "tracker": {
            "backend": TRACKER_NAME,
            "library": TRACKER_LIBRARY,
            "config": TRACKER_CONFIG,
            "width": TRACKER_DIMENSIONS[MODEL_INPUT][0],
            "height": TRACKER_DIMENSIONS[MODEL_INPUT][1],
            "native_object_id_output": True,
        },
        "jobs": jobs,
    }
    plan["contract_sha256"] = _plan_hash(plan)
    return plan


def _run_logged(command: Sequence[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode:
        raise SelectionProcessingError(
            f"command failed with exit code {completed.returncode}: {command[0]}"
        )


class WorkerLock:
    """One non-blocking flock owner for a selection execution root."""

    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "WorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.descriptor = os.open(
            self.path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o640,
        )
        try:
            fcntl.flock(
                self.descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            os.close(self.descriptor)
            self.descriptor = None
            raise SelectionProcessingError(
                "another video-selector worker owns the execution lock"
            ) from exc
        os.ftruncate(self.descriptor, 0)
        os.write(
            self.descriptor,
            f"pid={os.getpid()} acquired={_utc_now()}\n".encode("utf-8"),
        )
        os.fsync(self.descriptor)
        return self

    def __exit__(self, *_args: object) -> None:
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


class SelectionProcessor:
    def __init__(
        self,
        plan: dict[str, Any],
        *,
        repository_root: Path = REPOSITORY_ROOT,
        command_runner: CommandRunner = _run_logged,
        video_probe: VideoProbe = probe_video,
        converter: KittiConverter = convert_tracked_kitti_directory,
        require_engine_attestation: bool = True,
    ):
        self.plan = copy.deepcopy(plan)
        self.repository_root = repository_root.resolve()
        self.command_runner = command_runner
        self.video_probe = video_probe
        self.converter = converter
        self.require_engine_attestation = require_engine_attestation
        self.control_root = _inside_repository(
            plan["control_root"],
            self.repository_root,
            must_exist=False,
        )
        self.status_path = self.control_root / "execution-status.json"
        self.manifest_path = self.control_root / "execution-manifest.json"

    def _path(self, relative: str, *, must_exist: bool = False) -> Path:
        return _inside_repository(
            relative,
            self.repository_root,
            must_exist=must_exist,
        )

    def _write_or_verify_plan(self) -> None:
        if self.manifest_path.exists():
            existing = _load_json(
                self.manifest_path, label="execution manifest"
            )
            if existing != self.plan:
                raise SelectionProcessingError(
                    "existing execution manifest differs from immutable plan"
                )
            return
        _atomic_json(self.manifest_path, self.plan, mode=0o440)

    def _new_status(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "selection_id": self.plan["selection_id"],
            "contract_sha256": self.plan["contract_sha256"],
            "execution_mode": EXECUTION_MODE,
            "state": "running",
            "started_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "finished_at_utc": None,
            "jobs": [
                {
                    "video_id": job["video_id"],
                    "state": "pending",
                    "inference": "pending",
                    "pose": (
                        "pending"
                        if job.get("pose_enabled") is True
                        else "not_requested"
                    ),
                    "render": "pending",
                    "deliverable": None,
                    "message": None,
                }
                for job in self.plan["jobs"]
            ],
        }

    def _write_status(self, value: dict[str, Any]) -> None:
        value["updated_at_utc"] = _utc_now()
        _atomic_json(self.status_path, value)

    def _probe_matches(self, job: dict[str, Any]) -> None:
        metadata = self.video_probe(
            self._path(job["source_path"], must_exist=True)
        )
        expected = job["source_video"]
        actual_frames = metadata.get("frames", metadata.get("frame_count"))
        if (
            int(metadata["width"]) != int(expected["width"])
            or int(metadata["height"]) != int(expected["height"])
            or not math.isclose(
                float(metadata["fps"]),
                float(expected["fps"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or actual_frames is None
            or int(actual_frames) != int(expected["frame_count"])
        ):
            raise SelectionProcessingError(
                f"{job['video_id']}: live video metadata differs from queue"
            )

    def _kitti_complete(self, job: dict[str, Any]) -> bool:
        # Raw GIE KITTI has detections but no durable object identity.  Only
        # NvDCF's frame-complete tracker stream is eligible for conversion or
        # resume.
        directory = self._path(job["paths"]["tracker_kitti"])
        if not directory.is_dir():
            return False
        expected_count = int(job["source_video"]["frame_count"])
        indices: set[int] = set()
        for path in directory.glob("*.txt"):
            match = KITTI_NAME.fullmatch(path.name)
            if match is None:
                continue
            if int(match["app_index"]) != 0 or int(match["source_id"]) != 0:
                continue
            index = int(match["frame_index"])
            if index in indices:
                return False
            indices.add(index)
        return indices == set(range(expected_count))

    def _predictions_complete(self, job: dict[str, Any]) -> bool:
        path = self._path(job["paths"]["predictions"])
        if not path.is_file():
            return False
        expected_count = int(job["source_video"]["frame_count"])
        count = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        return False
                    row = json.loads(raw)
                    if (
                        not isinstance(row, dict)
                        or row.get("schema_version")
                        != "deepsafe.person-detections/v1"
                        or row.get("sequence_id") != job["sequence_id"]
                        or row.get("frame_index") != count
                        or row.get("image_width")
                        != int(job["source_video"]["width"])
                        or row.get("image_height")
                        != int(job["source_video"]["height"])
                        or row.get("model_id") != MODEL_ID
                        or not isinstance(row.get("detections"), list)
                    ):
                        return False
                    track_ids: set[int] = set()
                    for detection in row["detections"]:
                        if not isinstance(detection, dict):
                            return False
                        track_id = detection.get("track_id")
                        confidence = detection.get("confidence")
                        bbox = detection.get("bbox_norm_xywh")
                        if (
                            detection.get("class_name") != "person"
                            or not isinstance(track_id, int)
                            or isinstance(track_id, bool)
                            or track_id < 0
                            or track_id == UNTRACKED_OBJECT_ID
                            or track_id in track_ids
                            or isinstance(confidence, bool)
                            or not isinstance(confidence, (int, float))
                            or not math.isfinite(float(confidence))
                            or not 0.0 <= float(confidence) <= 1.0
                            or not isinstance(bbox, list)
                            or len(bbox) != 4
                            or any(
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not math.isfinite(float(value))
                                for value in bbox
                            )
                        ):
                            return False
                        track_ids.add(track_id)
                    count += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return count == expected_count

    def _direct_manifest_complete(self, job: dict[str, Any]) -> bool:
        path = self._path(job["paths"]["inference_manifest"])
        if not path.is_file():
            return False
        try:
            value = _load_json(path, label="inference manifest")
        except SelectionProcessingError:
            return False
        return (
            value.get("schema_version") == INFERENCE_SCHEMA_VERSION
            and value.get("selection_id") == self.plan["selection_id"]
            and value.get("video_id") == job["video_id"]
            and value.get("sequence_id") == job["sequence_id"]
            and value.get("execution_mode") == EXECUTION_MODE
            and value.get("status") == "complete"
            and value.get("model_id") == MODEL_ID
            and value.get("source_path") == job["source_path"]
            and value.get("tracker") == job["tracker"]
            and value.get("conversion", {}).get("native_track_ids") is True
            and value.get("conversion", {}).get("tracking_backend")
            == TRACKER_NAME
        )

    def _write_generated_configs(self, job: dict[str, Any]) -> None:
        infer_path = self._path(job["paths"]["infer_config"])
        app_path = self._path(job["paths"]["deepstream_config"])
        infer_text = render_infer_config(
            MODEL_INPUT,
            EXPORT_THRESHOLD,
            parser="cuda",
        )
        app_text = render_tracked_app_config(
            video_container_path=f"/workspace/{job['source_path']}",
            kitti_container_path=f"/workspace/{job['paths']['kitti']}",
            tracker_kitti_container_path=(
                f"/workspace/{job['paths']['tracker_kitti']}"
            ),
            infer_config_container_path=(
                f"/workspace/{job['paths']['infer_config']}"
            ),
            width=int(job["streammux"]["width"]),
            height=int(job["streammux"]["height"]),
            profile=MODEL_INPUT,
        )
        _atomic_write(infer_path, infer_text.encode("utf-8"))
        _atomic_write(app_path, app_text.encode("utf-8"))

    def _convert_kitti(self, job: dict[str, Any]) -> dict[str, object]:
        conversion = self.converter(
            self._path(job["paths"]["tracker_kitti"], must_exist=True),
            self._path(job["paths"]["predictions"]),
            sequence_id=job["sequence_id"],
            image_width=int(job["source_video"]["width"]),
            image_height=int(job["source_video"]["height"]),
            coordinate_width=int(job["streammux"]["width"]),
            coordinate_height=int(job["streammux"]["height"]),
            expected_frames=int(job["source_video"]["frame_count"]),
            fps=float(job["source_video"]["fps"]),
            source_uri=f"file:///workspace/{job['source_path']}",
            model_id=MODEL_ID,
        )
        _atomic_json(self._path(job["paths"]["conversion"]), conversion)
        if not self._predictions_complete(job):
            raise SelectionProcessingError(
                f"{job['video_id']}: converted predictions are incomplete"
            )
        return conversion

    def _write_inference_manifest(
        self,
        job: dict[str, Any],
        *,
        conversion: dict[str, object],
        deepstream_action: str,
        previous_manifest: dict[str, Any] | None,
        engine_attestation: dict[str, Any] | None,
    ) -> None:
        path = self._path(job["paths"]["inference_manifest"])
        previous = None
        if previous_manifest is not None:
            previous = {
                "status": previous_manifest.get("status"),
                "execution_mode": previous_manifest.get("execution_mode"),
                "sha256": (
                    _sha256_file(path) if path.is_file() else None
                ),
            }
        manifest = {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "selection_id": self.plan["selection_id"],
            "video_id": job["video_id"],
            "sequence_id": job["sequence_id"],
            "execution_mode": EXECUTION_MODE,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "deepstream_action": deepstream_action,
            "container_image": CONTAINER_IMAGE,
            "docker_command": copy.deepcopy(job["docker_command"]),
            "source_path": job["source_path"],
            "source_video": copy.deepcopy(job["source_video"]),
            "model_input": MODEL_INPUT,
            "model_id": MODEL_ID,
            "bbox_parser": "cuda",
            "streammux": copy.deepcopy(job["streammux"]),
            "tracker": copy.deepcopy(job["tracker"]),
            "conversion": conversion,
            "engine_load_attestation": engine_attestation,
            "recovered_previous_manifest": previous,
            "artifacts": {
                "predictions": _pin(
                    self._path(job["paths"]["predictions"], must_exist=True),
                    self.repository_root,
                ),
                "conversion": _pin(
                    self._path(job["paths"]["conversion"], must_exist=True),
                    self.repository_root,
                ),
                "deepstream_config": _pin(
                    self._path(
                        job["paths"]["deepstream_config"], must_exist=True
                    ),
                    self.repository_root,
                ),
                "infer_config": _pin(
                    self._path(job["paths"]["infer_config"], must_exist=True),
                    self.repository_root,
                ),
            },
        }
        log_path = self._path(job["paths"]["deepstream_log"])
        if log_path.is_file() and log_path.stat().st_size > 0:
            manifest["artifacts"]["deepstream_log"] = _pin(
                log_path, self.repository_root
            )
        _atomic_json(path, manifest)

    def _ensure_inference(self, job: dict[str, Any]) -> str:
        self._probe_matches(job)
        self._write_generated_configs(job)
        if self._predictions_complete(job) and self._direct_manifest_complete(job):
            return "resumed_complete"

        manifest_path = self._path(job["paths"]["inference_manifest"])
        previous_manifest = (
            _load_json(manifest_path, label="previous inference manifest")
            if manifest_path.is_file()
            else None
        )
        if self._kitti_complete(job):
            conversion = self._convert_kitti(job)
            self._write_inference_manifest(
                job,
                conversion=conversion,
                deepstream_action="reused_complete_tracker_kitti",
                previous_manifest=previous_manifest,
                engine_attestation=None,
            )
            return "recovered_complete_tracker_kitti"

        for directory_key in ("kitti", "tracker_kitti"):
            directory = self._path(job["paths"][directory_key])
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
        for key in ("predictions", "conversion"):
            self._path(job["paths"][key]).unlink(missing_ok=True)
        log_path = self._path(job["paths"]["deepstream_log"])
        self.command_runner(
            job["docker_command"],
            log_path,
            self.repository_root,
        )
        engine_attestation = None
        if self.require_engine_attestation:
            try:
                engine_attestation = attest_engine_load(log_path, MODEL_INPUT)
            except (OSError, ValueError) as exc:
                raise SelectionProcessingError(
                    f"{job['video_id']}: TensorRT engine attestation failed: {exc}"
                ) from exc
        conversion = self._convert_kitti(job)
        self._write_inference_manifest(
            job,
            conversion=conversion,
            deepstream_action="container_executed",
            previous_manifest=previous_manifest,
            engine_attestation=engine_attestation,
        )
        return "container_executed"

    def _pose_fusion_complete(self, job: dict[str, Any]) -> bool:
        if job.get("pose_enabled") is not True:
            return True
        output = self._path(job["paths"]["pose_predictions"])
        receipt_path = self._path(job["paths"]["pose_fusion_receipt"])
        person_path = self._path(job["paths"]["predictions"])
        keypoints_path = self._path(job["paths"]["pose_keypoints"])
        if not all(
            path.is_file()
            for path in (
                output,
                receipt_path,
                person_path,
                keypoints_path,
            )
        ):
            return False
        try:
            from content.pose_track_fusion import (
                FUSION_SCHEMA_VERSION,
                load_person_detection_records,
            )

            receipt = _load_json(
                receipt_path,
                label="pose-track fusion receipt",
            )
            records = load_person_detection_records(
                output,
                allow_pose=True,
            )
        except (OSError, ValueError, SelectionProcessingError):
            return False
        expected_frames = int(job["source_video"]["frame_count"])
        if len(records) != expected_frames:
            return False
        for frame_index, record in enumerate(records):
            if (
                record.get("sequence_id") != job["sequence_id"]
                or record.get("frame_index") != frame_index
                or any(
                    str(detection.get("class_name", "")).casefold()
                    == "person"
                    and "pose" not in detection
                    for detection in record["detections"]
                )
            ):
                return False
        return (
            receipt.get("schema_version") == FUSION_SCHEMA_VERSION
            and receipt.get("selection_id") == self.plan["selection_id"]
            and receipt.get("video_id") == job["video_id"]
            and receipt.get("pose_frame_offset") == 0
            and receipt.get("frames") == expected_frames
            and receipt.get("inputs", {}).get("person_predictions_sha256")
            == _sha256_file(person_path)
            and receipt.get("inputs", {}).get("pose_keypoints_sha256")
            == _sha256_file(keypoints_path)
            and receipt.get("output", {}).get("sha256")
            == _sha256_file(output)
        )

    def _ensure_pose(self, job: dict[str, Any]) -> str:
        if job.get("pose_enabled") is not True:
            return "not_requested"
        from content.pose_track_fusion import (
            PoseTrackFusionConfig,
            fuse_jsonl,
        )
        from video_selector.process_pose_selection import (
            PoseSelectionError,
            build_plan as build_pose_plan,
            execute_plan as execute_pose_plan,
        )

        try:
            pose_plan = build_pose_plan(
                self.plan["selection_id"],
                repository_root=self.repository_root,
                video_ids={job["video_id"]},
            )
            pose_job = pose_plan["jobs"][0]
            if pose_job["paths"]["keypoints"] != job["paths"]["pose_keypoints"]:
                raise SelectionProcessingError(
                    f"{job['video_id']}: pose keypoint path contract differs"
                )
            pose_result = execute_pose_plan(
                pose_plan,
                repository_root=self.repository_root,
            )
        except PoseSelectionError as exc:
            raise SelectionProcessingError(
                f"{job['video_id']}: pose inference failed: {exc}"
            ) from exc
        result_rows = pose_result.get("results", [])
        if (
            not isinstance(result_rows, list)
            or len(result_rows) != 1
            or result_rows[0].get("video_id") != job["video_id"]
        ):
            raise SelectionProcessingError(
                f"{job['video_id']}: pose execution receipt is invalid"
            )

        if self._pose_fusion_complete(job):
            return (
                "resumed_complete"
                if result_rows[0].get("status") == "resumed"
                else "inference_complete_fusion_resumed"
            )

        output = self._path(job["paths"]["pose_predictions"])
        receipt_path = self._path(job["paths"]["pose_fusion_receipt"])
        output.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        person_path = self._path(
            job["paths"]["predictions"],
            must_exist=True,
        )
        keypoints_path = self._path(
            job["paths"]["pose_keypoints"],
            must_exist=True,
        )
        fusion = fuse_jsonl(
            person_path,
            keypoints_path,
            output_path=output,
            config=PoseTrackFusionConfig(
                minimum_iou=float(job["pose"]["minimum_iou"]),
                ambiguity_margin=float(
                    job["pose"]["ambiguity_margin"]
                ),
                pose_frame_offset=int(
                    job["pose"]["pose_frame_offset"]
                ),
                auto_contiguous_offset=False,
            ),
        )
        receipt = {
            **fusion.receipt(),
            "selection_id": self.plan["selection_id"],
            "video_id": job["video_id"],
            "completed_at_utc": _utc_now(),
            "association": copy.deepcopy(job["pose"]),
            "inputs": {
                "person_predictions_sha256": _sha256_file(person_path),
                "pose_keypoints_sha256": _sha256_file(keypoints_path),
            },
            "output": _pin(output, self.repository_root),
        }
        _atomic_json(receipt_path, receipt)
        if not self._pose_fusion_complete(job):
            raise SelectionProcessingError(
                f"{job['video_id']}: pose-track fusion is incomplete"
            )
        return (
            "inference_resumed_fused"
            if result_rows[0].get("status") == "resumed"
            else "inference_executed_fused"
        )

    def _write_content_config(self, job: dict[str, Any]) -> Path:
        path = self._path(job["paths"]["content_config"])
        _atomic_json(path, job["content_config"])
        return path

    def _render_complete(self, job: dict[str, Any]) -> bool:
        output = self._path(job["paths"]["delivery_directory"])
        manifest_path = output / "manifest.json"
        demo_path = output / "demo.mp4"
        if not manifest_path.is_file() or not demo_path.is_file():
            return False
        try:
            manifest = _load_json(manifest_path, label="content manifest")
        except SelectionProcessingError:
            return False
        artifact = manifest.get("artifacts", {}).get("demo.mp4")
        plan = manifest.get("plan")
        if not isinstance(artifact, dict) or not isinstance(plan, dict):
            return False
        content_v2 = (
            job["content_config"].get("schema_version")
            == "deepsafe.content-roi-demo/v2"
        )
        zone_contract_matches = (
            plan.get("rois") == job["content_config"]["rois"]
            if content_v2
            else plan.get("roi") == job["content_config"]["roi"]
        )
        return (
            manifest.get("schema_version")
            == (
                "deepsafe.content-roi-demo-result/v2"
                if content_v2
                else "deepsafe.content-roi-demo-result/v1"
            )
            and manifest.get("status") == "rendered"
            and manifest.get("demo_id")
            == job["content_config"]["demo_id"]
            and manifest.get("gpu_or_model_execution") is False
            and plan.get("config_path") == job["paths"]["content_config"]
            and zone_contract_matches
            and plan.get("event_policy")
            == job["content_config"]["event_policy"]
            and plan.get("detections", {}).get("path")
            == job["paths"]["render_predictions"]
            and artifact.get("bytes") == demo_path.stat().st_size
            and artifact.get("sha256") == _sha256_file(demo_path)
        )

    def _publish_deliverable(self, job: dict[str, Any]) -> dict[str, Any]:
        source = (
            self._path(job["paths"]["delivery_directory"], must_exist=True)
            / "demo.mp4"
        )
        destination = self._path(job["paths"]["deliverable"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.is_file()
            and destination.stat().st_size == source.stat().st_size
            and _sha256_file(destination) == _sha256_file(source)
        ):
            return _pin(destination, self.repository_root)
        temporary = destination.parent / (
            f".{destination.name}.{os.getpid()}.tmp"
        )
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return _pin(destination, self.repository_root)

    def _ensure_render(self, job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        self._write_content_config(job)
        if self._render_complete(job):
            return "resumed_complete", self._publish_deliverable(job)
        output = self._path(job["paths"]["delivery_directory"])
        if output.exists():
            shutil.rmtree(output)
        log_path = self.control_root / "logs" / f"{job['video_id']}-render.log"
        self.command_runner(
            job["render_command"],
            log_path,
            self.repository_root,
        )
        if not self._render_complete(job):
            raise SelectionProcessingError(
                f"{job['video_id']}: renderer completed without a valid manifest"
            )
        return "rendered", self._publish_deliverable(job)

    def execute(self) -> dict[str, Any]:
        """Run/resume every job sequentially under one process lock."""

        self.control_root.mkdir(parents=True, exist_ok=True)
        with WorkerLock(self.control_root / "worker.lock"):
            self._write_or_verify_plan()
            status = self._new_status()
            self._write_status(status)
            try:
                for index, job in enumerate(self.plan["jobs"]):
                    row = status["jobs"][index]
                    row["state"] = "running"
                    row["message"] = "Direct DeepStream inference doğrulanıyor."
                    self._write_status(status)
                    inference_action = self._ensure_inference(job)
                    row["inference"] = "complete"
                    row["inference_action"] = inference_action
                    if job.get("pose_enabled") is True:
                        row["message"] = (
                            "Pose keypointleri kişi trackleriyle "
                            "eşleştiriliyor."
                        )
                        self._write_status(status)
                        pose_action = self._ensure_pose(job)
                        row["pose"] = "complete"
                        row["pose_action"] = pose_action
                    row["message"] = "ROI videosu hazırlanıyor."
                    self._write_status(status)
                    render_action, deliverable = self._ensure_render(job)
                    row["render"] = "complete"
                    row["render_action"] = render_action
                    row["deliverable"] = deliverable
                    row["state"] = "complete"
                    row["message"] = "İşlenmiş video hazır."
                    self._write_status(status)
                status["state"] = "complete"
                status["finished_at_utc"] = _utc_now()
                self._write_status(status)
                return status
            except BaseException as exc:
                status["state"] = "failed"
                status["finished_at_utc"] = _utc_now()
                status["error"] = f"{type(exc).__name__}: {exc}"
                for row in status["jobs"]:
                    if row["state"] == "running":
                        row["state"] = "failed"
                        row["message"] = str(exc)
                        break
                self._write_status(status)
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-id",
        required=True,
        help="32-character immutable selection id",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("video_selector/catalog.json"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run/resume direct DeepStream inference and ROI rendering",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not _SELECTION_ID.fullmatch(args.selection_id):
            raise SelectionProcessingError("invalid --selection-id")
        queue = Path(
            "content/video-selector/state/queues"
        ) / f"{args.selection_id}.json"
        plan = build_execution_plan(
            queue,
            args.catalog,
            repository_root=REPOSITORY_ROOT,
            gpu=args.gpu,
        )
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "validated_not_started",
                        "selection_id": plan["selection_id"],
                        "contract_sha256": plan["contract_sha256"],
                        "execution_mode": plan["execution_mode"],
                        "jobs": [
                            {
                                "video_id": job["video_id"],
                                "source_path": job["source_path"],
                                "deliverable": job["paths"]["deliverable"],
                            }
                            for job in plan["jobs"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = SelectionProcessor(plan).execute()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (
        OSError,
        SelectionProcessingError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
