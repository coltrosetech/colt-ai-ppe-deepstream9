from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator


SELECTION_SCHEMA_VERSION = "colt-ai.video-roi-selection/v2"
QUEUE_SCHEMA_VERSION = "colt-ai.person-processing-queue/v2"
_ROI_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}$")
_MIN_POLYGON_AREA = 1e-5
_EPSILON = 1e-9
PERSON_CATEGORY = "person_office"
PPE_CATEGORY = "ppe_safety"
Scenario = Literal["fence_security", "ppe_safety"]
RoiType = Literal["restricted_zone", "safe_walkway"]
PoseKeypoint = Literal[
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
]
SuppressedAlert = Literal["ppe_violation", "safe_walkway_violation"]
FenceSide = Literal["left", "right"]
DEFAULT_FENCE_KEYPOINTS: tuple[PoseKeypoint, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class CatalogSource:
    catalog_path: Path
    media_root: Path
    category: str
    category_label: str
    pipeline: str
    pipeline_label: str
    supported_modules: tuple[str, ...]
    default_requested_modules: tuple[str, ...]
    scenario: Scenario
    default_roi_type: RoiType
    roi_required: bool


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    model_config = ConfigDict(extra="forbid")


class RoiSelection(BaseModel):
    roi_id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=80)
    roi_type: RoiType
    points: list[NormalizedPoint] = Field(min_length=3, max_length=16)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_roi(self) -> "RoiSelection":
        self.roi_id = self.roi_id.strip()
        self.name = self.name.strip()
        if not _ROI_ID_PATTERN.fullmatch(self.roi_id):
            raise ValueError("ROI kimliği geçersiz")
        if not self.name:
            raise ValueError("ROI adı boş olamaz")
        _validate_polygon([(point.x, point.y) for point in self.points])
        return self


class FencePoseRoiOptions(BaseModel):
    enabled: bool = True
    selected_keypoints: list[PoseKeypoint] = Field(
        default_factory=lambda: list(DEFAULT_FENCE_KEYPOINTS),
        min_length=1,
        max_length=17,
    )
    inside_ratio_threshold: float = Field(
        default=0.50,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    keypoint_confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    minimum_visible_keypoints: int = Field(default=4, ge=1, le=17)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_keypoints(self) -> "FencePoseRoiOptions":
        if len(self.selected_keypoints) != len(set(self.selected_keypoints)):
            raise ValueError("Pose keypoint seçimleri benzersiz olmalı")
        if self.minimum_visible_keypoints > len(self.selected_keypoints):
            raise ValueError(
                "Minimum görünür keypoint sayısı seçim sayısını aşamaz"
            )
        return self


class FenceCrossingRuleOptions(BaseModel):
    """Two-point fence boundary and deterministic staged alarm thresholds."""

    enabled: bool = True
    boundary_start: NormalizedPoint | None = None
    boundary_end: NormalizedPoint | None = None
    forbidden_side: FenceSide = "left"
    contact_band: float = Field(
        default=0.03,
        gt=0.0,
        le=0.25,
        allow_inf_nan=False,
    )
    minimum_confidence: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    minimum_core_visible: int = Field(default=1, ge=1, le=2)
    breach_enter_frames: int = Field(default=4, ge=1, le=30)
    breach_exit_frames: int = Field(default=4, ge=1, le=60)
    approach_keypoint_names: list[PoseKeypoint] = Field(
        default_factory=lambda: [
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        ],
        min_length=1,
        max_length=17,
    )
    approach_minimum_count: int = Field(default=1, ge=1, le=17)
    wrist_contact_required: int = Field(default=1, ge=1, le=2)
    hip_rise_ratio: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    raised_knee_ratio: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    climb_enter_frames: int = Field(default=2, ge=1, le=30)
    climb_exit_frames: int = Field(default=2, ge=1, le=60)
    history_window_frames: int = Field(default=30, ge=2, le=300)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_boundary(self) -> "FenceCrossingRuleOptions":
        if self.enabled and (
            self.boundary_start is None or self.boundary_end is None
        ):
            raise ValueError(
                "Etkin aşamalı çit kuralı için iki uçlu çit hattı gerekli"
            )
        if (self.boundary_start is None) != (self.boundary_end is None):
            raise ValueError("Çit hattının iki ucu birlikte tanımlanmalı")
        if self.boundary_start is not None and self.boundary_end is not None:
            distance = math.hypot(
                self.boundary_end.x - self.boundary_start.x,
                self.boundary_end.y - self.boundary_start.y,
            )
            if distance < 0.01:
                raise ValueError(
                    "Çit hattının iki ucu birbirinden farklı olmalı"
                )
        if len(self.approach_keypoint_names) != len(
            set(self.approach_keypoint_names)
        ):
            raise ValueError("Yaklaşma keypoint seçimleri benzersiz olmalı")
        if self.approach_minimum_count > len(self.approach_keypoint_names):
            raise ValueError(
                "Minimum yaklaşma keypoint sayısı seçim sayısını aşamaz"
            )
        if self.history_window_frames < max(
            self.breach_enter_frames,
            self.climb_enter_frames,
        ):
            raise ValueError(
                "Geçmiş penceresi alarm doğrulama süresinden kısa olamaz"
            )
        return self


class ForkliftDriverSuppressionOptions(BaseModel):
    enabled: bool = True
    suppressed_alerts: list[SuppressedAlert] = Field(
        default_factory=lambda: [
            "ppe_violation",
            "safe_walkway_violation",
        ],
        min_length=1,
        max_length=2,
    )
    minimum_forklift_confidence: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    minimum_person_ioa: float = Field(
        default=0.55,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    enter_debounce_frames: int = Field(default=4, ge=1, le=30)
    exit_debounce_frames: int = Field(default=8, ge=1, le=60)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_alerts(self) -> "ForkliftDriverSuppressionOptions":
        if len(self.suppressed_alerts) != len(set(self.suppressed_alerts)):
            raise ValueError("Bastırılacak alarm türleri benzersiz olmalı")
        return self


class AnalysisOptions(BaseModel):
    fence_pose_roi: FencePoseRoiOptions | None = None
    fence_crossing_rule: FenceCrossingRuleOptions | None = None
    forklift_driver_suppression: (
        ForkliftDriverSuppressionOptions | None
    ) = None

    model_config = ConfigDict(extra="forbid")


class VideoSelection(BaseModel):
    video_id: str = Field(min_length=1, max_length=24)
    scenario: Scenario
    start_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    rois: list[RoiSelection] = Field(min_length=0, max_length=8)
    analysis_options: AnalysisOptions | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_video_selection(self) -> "VideoSelection":
        self.video_id = self.video_id.strip()
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Bitiş zamanı başlangıçtan büyük olmalı")
        roi_ids = [roi.roi_id for roi in self.rois]
        roi_names = [roi.name.casefold() for roi in self.rois]
        if len(roi_ids) != len(set(roi_ids)):
            raise ValueError("ROI kimlikleri benzersiz olmalı")
        if len(roi_names) != len(set(roi_names)):
            raise ValueError("ROI adları benzersiz olmalı")
        if self.analysis_options is not None:
            if (
                self.scenario == "fence_security"
                and self.analysis_options.forklift_driver_suppression
                is not None
            ):
                raise ValueError(
                    "Çit güvenliğinde forklift sürücüsü bastırma ayarı kullanılamaz"
                )
            if (
                self.scenario == "ppe_safety"
                and (
                    self.analysis_options.fence_pose_roi is not None
                    or self.analysis_options.fence_crossing_rule is not None
                )
            ):
                raise ValueError(
                    "İSG / PPE senaryosunda çit alarm ayarı kullanılamaz"
                )
            crossing = self.analysis_options.fence_crossing_rule
            pose = self.analysis_options.fence_pose_roi
            if (
                self.scenario == "fence_security"
                and crossing is not None
                and crossing.enabled
                and pose is not None
                and not pose.enabled
            ):
                raise ValueError(
                    "Aşamalı çit kuralı etkin olduğunda pose kapatılamaz"
                )
        return self


class SelectionInput(BaseModel):
    videos: list[VideoSelection] = Field(min_length=1, max_length=80)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_unique_videos(self) -> "SelectionInput":
        video_ids = [video.video_id for video in self.videos]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("Video kimlikleri benzersiz olmalı")
        return self


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _orientation(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])


def _on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    return (
        abs(_orientation(start, end, point)) <= _EPSILON
        and min(start[0], end[0]) - _EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _EPSILON
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    a = _orientation(first_start, first_end, second_start)
    b = _orientation(first_start, first_end, second_end)
    c = _orientation(second_start, second_end, first_start)
    d = _orientation(second_start, second_end, first_end)
    if ((a > _EPSILON and b < -_EPSILON) or (a < -_EPSILON and b > _EPSILON)) and (
        (c > _EPSILON and d < -_EPSILON) or (c < -_EPSILON and d > _EPSILON)
    ):
        return True
    return (
        _on_segment(second_start, first_start, first_end)
        or _on_segment(second_end, first_start, first_end)
        or _on_segment(first_start, second_start, second_end)
        or _on_segment(first_end, second_start, second_end)
    )


def _validate_polygon(points: list[tuple[float, float]]) -> None:
    rounded = {(round(x, 9), round(y, 9)) for x, y in points}
    if len(rounded) != len(points):
        raise ValueError("ROI köşeleri birbirinden farklı olmalı")
    if _polygon_area(points) < _MIN_POLYGON_AREA:
        raise ValueError("ROI alanı çok küçük")
    count = len(points)
    for first_index in range(count):
        first_start = points[first_index]
        first_end = points[(first_index + 1) % count]
        for second_index in range(first_index + 1, count):
            if second_index in {
                first_index,
                (first_index + 1) % count,
                (first_index - 1) % count,
            }:
                continue
            if first_index == 0 and second_index == count - 1:
                continue
            second_start = points[second_index]
            second_end = points[(second_index + 1) % count]
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                raise ValueError("ROI kenarları kesişemez")


class VideoCatalog:
    def __init__(self, sources: list[CatalogSource]):
        if not sources:
            raise RuntimeError("at least one video catalog is required")
        self.videos: dict[str, dict[str, Any]] = {}
        self.revisions: dict[str, str] = {}
        self.categories: dict[str, dict[str, Any]] = {}
        for source in sources:
            self._add_source(source)

    def _add_source(self, source: CatalogSource) -> None:
        payload = json.loads(source.catalog_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "colt-ai.video-catalog/v1":
            raise RuntimeError("video catalog schema is invalid")
        videos = payload.get("videos")
        if not isinstance(videos, list) or not videos:
            raise RuntimeError("video catalog is empty")
        revision = str(payload["catalog_revision"])
        if source.category in self.revisions:
            raise RuntimeError(f"duplicate video category: {source.category}")

        category_videos: list[dict[str, Any]] = []
        for raw_video in videos:
            video = dict(raw_video)
            video_id = str(video["video_id"])
            if video_id in self.videos:
                raise RuntimeError(f"video catalog has duplicate id: {video_id}")
            video.update(
                {
                    "category": source.category,
                    "category_label": source.category_label,
                    "pipeline": source.pipeline,
                    "pipeline_label": source.pipeline_label,
                    "supported_modules": list(source.supported_modules),
                    "default_requested_modules": list(
                        source.default_requested_modules
                    ),
                    "scenario": source.scenario,
                    "default_roi_type": source.default_roi_type,
                    "roi_required": source.roi_required,
                    "_catalog_revision": revision,
                    "_media_root": source.media_root.resolve(),
                }
            )
            self.videos[video_id] = video
            category_videos.append(video)

        self.revisions[source.category] = revision
        self.categories[source.category] = {
            "category": source.category,
            "label": source.category_label,
            "pipeline": source.pipeline,
            "pipeline_label": source.pipeline_label,
            "supported_modules": list(source.supported_modules),
            "default_requested_modules": list(source.default_requested_modules),
            "scenario": source.scenario,
            "default_roi_type": source.default_roi_type,
            "roi_required": source.roi_required,
            "video_count": len(category_videos),
        }

    @property
    def revision(self) -> str:
        return "+".join(self.revisions.values())

    def revision_for(self, video_ids: list[str]) -> str:
        revisions = list(
            dict.fromkeys(
                str(self.videos[video_id]["_catalog_revision"])
                for video_id in video_ids
            )
        )
        return "+".join(revisions)

    def revisions_for(self, video_ids: list[str]) -> dict[str, str]:
        categories = {
            str(self.videos[video_id]["category"]) for video_id in video_ids
        }
        return {
            category: revision
            for category, revision in self.revisions.items()
            if category in categories
        }

    def media_path(self, video: dict[str, Any], kind: str) -> Path:
        if kind not in {"media", "posters"}:
            raise RuntimeError("unsupported media kind")
        field = "media_filename" if kind == "media" else "poster_filename"
        filename = str(video[field])
        if Path(filename).name != filename:
            raise RuntimeError("catalog media filename must be a basename")
        directory = (Path(video["_media_root"]) / kind).resolve()
        path = (directory / filename).resolve()
        if path.parent != directory:
            raise RuntimeError("catalog media path escapes its root")
        return path

    def public_payload(self) -> dict[str, Any]:
        public_videos = []
        private_fields = {
            "media_filename",
            "poster_filename",
            "processing_source_path",
            "ground_truth_path",
            "source_url",
            "license",
        }
        for video in self.videos.values():
            item = {
                key: value
                for key, value in video.items()
                if key not in private_fields and not key.startswith("_")
            }
            item["media_url"] = f"/api/videos/{video['video_id']}/media"
            item["poster_url"] = f"/api/videos/{video['video_id']}/poster"
            public_videos.append(item)
        return {
            "schema_version": "colt-ai.video-catalog-public/v1",
            "catalog_revision": self.revision,
            "catalog_revisions": dict(self.revisions),
            "categories": list(self.categories.values()),
            "videos": public_videos,
        }


def _requested_modules(
    *,
    video: dict[str, Any],
    selected: VideoSelection,
) -> list[str]:
    modules = list(video["default_requested_modules"])
    options = selected.analysis_options
    if (
        selected.scenario == "fence_security"
        and options is not None
        and (
            (
                options.fence_pose_roi is not None
                and options.fence_pose_roi.enabled
            )
            or (
                options.fence_crossing_rule is not None
                and options.fence_crossing_rule.enabled
            )
        )
    ):
        modules.append("pose")
    if (
        selected.scenario == "ppe_safety"
        and options is not None
        and options.forklift_driver_suppression is not None
        and options.forklift_driver_suppression.enabled
    ):
        modules.append("forklift")
    return list(dict.fromkeys(modules))


def _alert_policy(
    *,
    scenario: str,
    has_rois: bool,
    analysis_options: AnalysisOptions | None,
) -> dict[str, Any]:
    common = {
        "tracking_identity": "nvdcf_track_id",
        "person_anchor": "bbox_bottom_center",
    }
    if scenario == "fence_security":
        crossing = (
            analysis_options.fence_crossing_rule
            if analysis_options is not None
            else None
        )
        pose = (
            analysis_options.fence_pose_roi
            if analysis_options is not None
            else None
        )
        if crossing is not None and crossing.enabled:
            active_pose = (
                pose
                if pose is not None and pose.enabled
                else FencePoseRoiOptions()
            )
            return {
                "tracking_identity": "nvdcf_track_id",
                "person_anchor": "mid_hip",
                "zone_rule": "staged_fence_boundary_crossing",
                "enter_debounce_frames": crossing.breach_enter_frames,
                "exit_debounce_frames": crossing.breach_exit_frames,
                "ppe_scope": "disabled",
                "pose_zone_rule": {
                    "keypoint_layout": "coco17",
                    "selected_keypoints": list(
                        active_pose.selected_keypoints
                    ),
                    "inside_ratio_threshold": (
                        active_pose.inside_ratio_threshold
                    ),
                    "keypoint_confidence_threshold": (
                        active_pose.keypoint_confidence_threshold
                    ),
                    "minimum_visible_keypoints": (
                        active_pose.minimum_visible_keypoints
                    ),
                    "ratio_denominator": "selected_keypoints",
                    "roi_aggregation": "union_any",
                    "polygon_boundary": "inclusive",
                    "person_pose_association": (
                        "highest_iou_to_nvdcf_track"
                    ),
                    "insufficient_pose_policy": "no_alert",
                },
                "fence_crossing_rule": crossing.model_dump(mode="json"),
            }
        if pose is not None and pose.enabled:
            return {
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
                    "selected_keypoints": list(pose.selected_keypoints),
                    "inside_ratio_threshold": (
                        pose.inside_ratio_threshold
                    ),
                    "keypoint_confidence_threshold": (
                        pose.keypoint_confidence_threshold
                    ),
                    "minimum_visible_keypoints": (
                        pose.minimum_visible_keypoints
                    ),
                    "ratio_denominator": "selected_keypoints",
                    "roi_aggregation": "union_any",
                    "polygon_boundary": "inclusive",
                    "person_pose_association": (
                        "highest_iou_to_nvdcf_track"
                    ),
                    "insufficient_pose_policy": "no_alert",
                },
            }
        return {
            **common,
            "zone_rule": "inside_any_restricted_zone",
            "enter_debounce_frames": 3,
            "exit_debounce_frames": 6,
            "ppe_scope": "disabled",
        }
    if scenario == "ppe_safety":
        policy = {
            **common,
            "zone_rule": (
                "outside_all_safe_walkways" if has_rois else "disabled"
            ),
            "enter_debounce_frames": 6,
            "exit_debounce_frames": 4,
            "ppe_scope": "all_tracked_persons",
            "ppe_requirements": {
                "helmet": True,
                "hi_vis": True,
            },
        }
        suppression = (
            analysis_options.forklift_driver_suppression
            if analysis_options is not None
            else None
        )
        if suppression is not None and suppression.enabled:
            policy["forklift_driver_suppression"] = {
                "enabled": True,
                "forklift_class": "forklift_candidate",
                "detector_evidence": "coco_truck_class_7",
                "classification_scope": (
                    "industrial_forklift_candidate"
                ),
                "tracking_identity": "nvdcf_track_id",
                "association_rule": "temporal_person_forklift_ioa",
                "minimum_forklift_confidence": (
                    suppression.minimum_forklift_confidence
                ),
                "minimum_person_ioa": suppression.minimum_person_ioa,
                "enter_debounce_frames": (
                    suppression.enter_debounce_frames
                ),
                "exit_debounce_frames": (
                    suppression.exit_debounce_frames
                ),
                "maximum_occupants_per_forklift": 1,
                "suppressed_alerts": list(
                    suppression.suppressed_alerts
                ),
                "render_state": "forklift_driver",
                "missing_forklift_evidence": "do_not_suppress",
            }
        return policy
    raise RuntimeError(f"unsupported video scenario: {scenario}")


class SelectionStore:
    def __init__(self, state_root: Path, catalog: VideoCatalog):
        self.state_root = state_root
        self.catalog = catalog
        self.lock = threading.Lock()

    @property
    def latest_path(self) -> Path:
        return self.state_root / "latest.json"

    def latest(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.latest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def save(self, value: SelectionInput) -> dict[str, Any]:
        selections = []
        queue_items = []
        selected_video_ids: list[str] = []
        for selected in value.videos:
            video = self.catalog.videos.get(selected.video_id)
            if video is None:
                raise HTTPException(422, f"Bilinmeyen video: {selected.video_id}")
            if selected.scenario != video["scenario"]:
                raise HTTPException(
                    422,
                    (
                        f"{selected.video_id} için senaryo katalogla eşleşmiyor: "
                        f"{selected.scenario}"
                    ),
                )
            if video["roi_required"] and not selected.rois:
                raise HTTPException(
                    422,
                    f"{selected.video_id} için en az bir alan gerekli",
                )
            expected_roi_type = str(video["default_roi_type"])
            mismatched_roi_types = [
                roi.roi_type
                for roi in selected.rois
                if roi.roi_type != expected_roi_type
            ]
            if mismatched_roi_types:
                raise HTTPException(
                    422,
                    (
                        f"{selected.video_id} alan türü {expected_roi_type} "
                        "olmalı"
                    ),
                )
            tolerance = max(0.05, 1.0 / float(video["fps"]))
            if selected.end_seconds > float(video["duration_seconds"]) + tolerance:
                raise HTTPException(
                    422,
                    f"{selected.video_id} için bitiş zamanı video süresini aşıyor",
                )
            selected_payload = selected.model_dump(
                mode="json",
                exclude_none=True,
            )
            selected_payload["end_seconds"] = min(
                float(selected_payload["end_seconds"]),
                float(video["duration_seconds"]),
            )
            selected_payload["category"] = video["category"]
            selected_payload["pipeline"] = video["pipeline"]
            requested_modules = _requested_modules(
                video=video,
                selected=selected,
            )
            selected_payload["requested_modules"] = requested_modules
            selections.append(selected_payload)
            selected_video_ids.append(selected.video_id)
            queue_items.append(
                {
                    "video_id": selected.video_id,
                    "title": video["title"],
                    "category": video["category"],
                    "scenario": video["scenario"],
                    "pipeline": video["pipeline"],
                    "requested_modules": requested_modules,
                    "supported_modules": list(video["supported_modules"]),
                    "catalog_revision": video["_catalog_revision"],
                    "source_path": video["processing_source_path"],
                    "source_url": video["source_url"],
                    "license": video["license"],
                    "ground_truth_path": video["ground_truth_path"],
                    "clip": {
                        "start_seconds": selected_payload["start_seconds"],
                        "end_seconds": selected_payload["end_seconds"],
                    },
                    "rois": selected_payload["rois"],
                    "alert_policy": _alert_policy(
                        scenario=video["scenario"],
                        has_rois=bool(selected_payload["rois"]),
                        analysis_options=selected.analysis_options,
                    ),
                    "source_video": {
                        "width": video["width"],
                        "height": video["height"],
                        "fps": video["fps"],
                        "duration_seconds": video["duration_seconds"],
                        "frame_count": video["frame_count"],
                    },
                }
            )

        selection_catalog_revision = self.catalog.revision_for(
            selected_video_ids
        )
        selection_catalog_revisions = self.catalog.revisions_for(
            selected_video_ids
        )
        selection_id = uuid.uuid4().hex
        created_at = (
            dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        snapshot = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_id": selection_id,
            "catalog_revision": selection_catalog_revision,
            "catalog_revisions": selection_catalog_revisions,
            "status": "awaiting_manual_start",
            "processing_started": False,
            "created_at_utc": created_at,
            "videos": selections,
        }
        queue = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "selection_id": selection_id,
            "catalog_revision": selection_catalog_revision,
            "catalog_revisions": selection_catalog_revisions,
            "requested_modules": list(
                dict.fromkeys(
                    module
                    for item in queue_items
                    for module in item["requested_modules"]
                )
            ),
            "state": "awaiting_manual_start",
            "execution": {
                "requested": False,
                "started": False,
                "gpu_or_model_execution": False,
            },
            "created_at_utc": created_at,
            "items": queue_items,
        }

        with self.lock:
            queue_path = self.state_root / "queues" / f"{selection_id}.json"
            selection_path = (
                self.state_root / "selections" / f"{selection_id}.json"
            )
            _atomic_json_write(queue_path, queue)
            _atomic_json_write(selection_path, snapshot)
            _atomic_json_write(self.latest_path, snapshot)
            _atomic_json_write(self.state_root / "latest-queue.json", queue)
        return snapshot

    def queue(self, selection_id: str | None = None) -> dict[str, Any] | None:
        path = (
            self.state_root / "latest-queue.json"
            if selection_id is None
            else self.state_root / "queues" / f"{selection_id}.json"
        )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            os.chmod(temporary.name, 0o600)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        # The selector runs in a container, while the explicitly started
        # processing worker runs as the workstation user.  Keep selections
        # immutable-by-convention but readable by that downstream worker.
        os.chmod(path, 0o644)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def create_app(
    *,
    catalog_path: Path | None = None,
    media_root: Path | None = None,
    ppe_catalog_path: Path | None = None,
    ppe_media_root: Path | None = None,
    state_root: Path | None = None,
) -> FastAPI:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parent
    resolved_catalog = catalog_path or Path(
        os.getenv("VIDEO_SELECTOR_CATALOG", package_root / "catalog.json")
    )
    resolved_media = media_root or Path(
        os.getenv(
            "VIDEO_SELECTOR_MEDIA_ROOT",
            repository_root / "content/video-selector",
        )
    )
    resolved_ppe_catalog: Path | None = ppe_catalog_path
    resolved_ppe_media: Path | None = ppe_media_root
    # Explicit catalog_path keeps the original single-catalog test/embedding
    # contract unless a PPE path is explicitly supplied. The normal app entry
    # point loads both built-in catalogs.
    if catalog_path is None:
        resolved_ppe_catalog = resolved_ppe_catalog or Path(
            os.getenv(
                "VIDEO_SELECTOR_PPE_CATALOG",
                package_root / "ppe-catalog.json",
            )
        )
        resolved_ppe_media = resolved_ppe_media or Path(
            os.getenv(
                "VIDEO_SELECTOR_PPE_MEDIA_ROOT",
                repository_root / "content/ppe-video-selector",
            )
        )
    elif resolved_ppe_catalog is not None and resolved_ppe_media is None:
        resolved_ppe_media = repository_root / "content/ppe-video-selector"
    resolved_state = state_root or Path(
        os.getenv(
            "VIDEO_SELECTOR_STATE_ROOT",
            repository_root / "content/video-selector/state",
        )
    )
    sources = [
        CatalogSource(
            catalog_path=resolved_catalog.resolve(),
            media_root=resolved_media.resolve(),
            category=PERSON_CATEGORY,
            category_label="Çit Güvenliği",
            pipeline="person_roi",
            pipeline_label="İnsan + Alan Güvenliği",
            supported_modules=("person_roi", "pose"),
            default_requested_modules=("person_roi",),
            scenario="fence_security",
            default_roi_type="restricted_zone",
            roi_required=True,
        )
    ]
    if resolved_ppe_catalog is not None:
        if resolved_ppe_media is None:
            raise RuntimeError("PPE media root is required with a PPE catalog")
        sources.append(
            CatalogSource(
                catalog_path=resolved_ppe_catalog.resolve(),
                media_root=resolved_ppe_media.resolve(),
                category=PPE_CATEGORY,
                category_label="İSG / PPE",
                pipeline="ppe",
                pipeline_label="PPE",
                supported_modules=("ppe", "forklift"),
                default_requested_modules=("ppe",),
                scenario="ppe_safety",
                default_roi_type="safe_walkway",
                roi_required=False,
            )
        )
    catalog = VideoCatalog(sources)
    store = SelectionStore(resolved_state.resolve(), catalog)

    application = FastAPI(
        title="COLT AI - COLLBRAI Video Seçimi",
        version="1.0.0",
    )
    application.state.catalog = catalog
    application.state.selection_store = store
    # Keep the original state hook for local integrations and expose the
    # category-aware roots alongside it.
    application.state.media_root = resolved_media.resolve()
    application.state.media_roots = {
        source.category: source.media_root.resolve() for source in sources
    }
    application.mount(
        "/static",
        StaticFiles(directory=package_root / "static"),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(package_root / "static/index.html")

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "catalog_revision": catalog.revision,
            "video_count": len(catalog.videos),
            "processing_started": False,
        }

    @application.get("/api/videos")
    def videos() -> dict[str, Any]:
        return catalog.public_payload()

    def catalog_video(video_id: str) -> dict[str, Any]:
        video = catalog.videos.get(video_id)
        if video is None:
            raise HTTPException(404, "Video bulunamadı")
        return video

    @application.get("/api/videos/{video_id}/media")
    def media(video_id: str) -> FileResponse:
        video = catalog_video(video_id)
        path = catalog.media_path(video, "media")
        if not path.is_file():
            raise HTTPException(404, "Video dosyası hazır değil")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=None,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @application.get("/api/videos/{video_id}/poster")
    def poster(video_id: str) -> FileResponse:
        video = catalog_video(video_id)
        path = catalog.media_path(video, "posters")
        if not path.is_file():
            raise HTTPException(404, "Poster hazır değil")
        return FileResponse(
            path,
            media_type="image/jpeg",
            filename=None,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @application.get("/api/selection/latest")
    def latest_selection() -> dict[str, Any]:
        payload = store.latest()
        if payload is None:
            raise HTTPException(404, "Henüz kaydedilmiş seçim yok")
        return payload

    @application.put("/api/selection")
    def save_selection(value: SelectionInput) -> dict[str, Any]:
        snapshot = store.save(value)
        return {
            **snapshot,
            "message": "Seçimler kaydedildi; işleme henüz başlamadı.",
        }

    @application.get("/api/queue/latest")
    def latest_queue() -> dict[str, Any]:
        payload = store.queue()
        if payload is None:
            raise HTTPException(404, "Henüz hazır işlem kuyruğu yok")
        return payload

    @application.get("/api/selections/{selection_id}/queue")
    def selection_queue(selection_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{32}", selection_id):
            raise HTTPException(404, "Seçim bulunamadı")
        payload = store.queue(selection_id)
        if payload is None:
            raise HTTPException(404, "Seçim bulunamadı")
        return payload

    return application


app = create_app()
