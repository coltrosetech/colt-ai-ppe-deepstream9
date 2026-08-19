"""Safe API primitives for the operator-driven ROI and clip editor.

The editor deliberately has no execution hook.  It can only expose the
explicitly approved clean preview files and persist an immutable processing
plan.  Rendering/inference remains a separate, explicit workflow.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import Response


PLAN_SCHEMA_VERSION = "colt-ai.roi-processing-plan/v1"
MAX_PLAN_BYTES = 256 * 1024
_PLAN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_POLYGON_EPSILON = 1e-9
_MIN_POLYGON_AREA = 1e-5


@dataclass(frozen=True)
class ApprovedVideo:
    video_id: str
    display_name: str
    cache_filename: str
    source_path: str
    detections_path: str
    sequence_id: str
    expected_model_id: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int


# This is an intentional, closed registry.  Request data is never converted
# into a filesystem path.
APPROVED_VIDEOS: tuple[ApprovedVideo, ...] = (
    ApprovedVideo(
        "01",
        "CAM-01 · G326 Giriş",
        "01.mp4",
        "data/samples/open/meva/2018-03-07.16-50-01.16-55-01.admin.G326.r13.avi",
        "validation/results/content-inference/colt-candidate-01/960/predictions.jsonl",
        "colt-candidate-01",
        "yolo11s-960-fp16",
        300.03333333333336,
        1920,
        1072,
        30.0,
        9001,
    ),
    ApprovedVideo(
        "02",
        "CAM-02 · G329 Merdiven",
        "02.mp4",
        "data/samples/open/meva/2018-03-07.16-50-00.16-55-00.admin.G329.r13.avi",
        "validation/results/content-inference/colt-candidate-02/960/predictions.jsonl",
        "colt-candidate-02",
        "yolo11s-960-fp16",
        300.06666666666666,
        1920,
        1072,
        30.0,
        9002,
    ),
    ApprovedVideo(
        "03",
        "CAM-03 · G331 Bekleme Alanı",
        "03.mp4",
        "data/samples/open/meva/2018-03-07.16-50-00.16-55-00.bus.G331.r13.avi",
        "validation/results/content-inference/colt-candidate-03/960/predictions.jsonl",
        "colt-candidate-03",
        "yolo11s-960-fp16",
        300.1666666666667,
        1920,
        1080,
        30.0,
        9005,
    ),
    ApprovedVideo(
        "04",
        "CAM-04 · G421 Lobi",
        "04.mp4",
        "data/samples/open/meva/2018-03-11.11-30-01.11-35-01.school.G421.r13.avi",
        "validation/results/content-inference/colt-candidate-04/960/predictions.jsonl",
        "colt-candidate-04",
        "yolo11s-960-fp16",
        300.23333333333335,
        1920,
        1072,
        30.0,
        9007,
    ),
    ApprovedVideo(
        "06",
        "CAM-06 · G420 Merdiven Girişi",
        "06.mp4",
        "data/samples/open/meva/2018-03-07.16-50-01.16-55-01.school.G420.r13.avi",
        "validation/results/content-inference/colt-candidate-06/960/predictions.jsonl",
        "colt-candidate-06",
        "yolo11s-960-fp16",
        300.0,
        1920,
        1072,
        30.0,
        9000,
    ),
    ApprovedVideo(
        "07",
        "CAM-07 · G506 Giriş ve Bank",
        "07.mp4",
        "data/samples/open/meva/2018-03-15.15-35-01.15-40-01.bus.G506.r13.avi",
        "validation/results/content-inference/colt-candidate-07/960/predictions.jsonl",
        "colt-candidate-07",
        "yolo11s-960-fp16",
        300.03333333333336,
        1920,
        1072,
        30.0,
        9001,
    ),
    ApprovedVideo(
        "08",
        "CAM-08 · G301 Yükleme Rampası",
        "08.mp4",
        "data/samples/open/meva/2018-03-07.16-50-07.16-55-07.hospital.G301.r13.avi",
        "validation/results/content-inference/colt-candidate-08/960/predictions.jsonl",
        "colt-candidate-08",
        "yolo11s-960-fp16",
        300.0,
        1920,
        1080,
        30.0,
        9000,
    ),
    ApprovedVideo(
        "14",
        "CAM-14 · G638 Okul Meydanı",
        "14.mp4",
        "data/samples/open/meva/2018-03-12.10-45-00.10-50-00.school.G638.r13.avi",
        "validation/results/content-inference/colt-candidate-14/960/predictions.jsonl",
        "colt-candidate-14",
        "yolo11s-960-fp16",
        300.03333333333336,
        1920,
        1072,
        30.0,
        9001,
    ),
    ApprovedVideo(
        "18",
        "CAM-18 · Metro Ermita Koridoru",
        "18.mp4",
        "data/derived/open-h264/metro-ermita-corridor-cc-by-sa-4.0.mp4",
        "validation/results/open-video-review/mx_metro_ermita_corridor/960/predictions.jsonl",
        "mx_metro_ermita_corridor",
        "yolo11s-960-fp16",
        12.245566666666667,
        1920,
        1080,
        30000.0 / 1001.0,
        367,
    ),
)


class RoiEditorError(ValueError):
    def __init__(self, state: str, *, status_code: int = 422):
        super().__init__(state)
        self.state = state
        self.status_code = status_code


class NormalizedPointIn(BaseModel):
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    model_config = ConfigDict(extra="forbid")


class RoiIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    points: list[NormalizedPointIn] = Field(min_length=3, max_length=16)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_polygon(self) -> "RoiIn":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("ROI adı boş olamaz")
        points = [(point.x, point.y) for point in self.points]
        _validate_polygon(points)
        return self


class ProcessingPlanIn(BaseModel):
    video_id: str = Field(min_length=1, max_length=16)
    start_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    rois: list[RoiIn] = Field(min_length=1, max_length=8)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_interval_and_names(self) -> "ProcessingPlanIn":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Bitiş zamanı başlangıç zamanından büyük olmalı")
        names = [roi.name.casefold() for roi in self.rois]
        if len(names) != len(set(names)):
            raise ValueError("ROI adları benzersiz olmalı")
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
        abs(_orientation(start, end, point)) <= _POLYGON_EPSILON
        and min(start[0], end[0]) - _POLYGON_EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _POLYGON_EPSILON
        and min(start[1], end[1]) - _POLYGON_EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _POLYGON_EPSILON
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_a = _orientation(first_start, first_end, second_start)
    first_b = _orientation(first_start, first_end, second_end)
    second_a = _orientation(second_start, second_end, first_start)
    second_b = _orientation(second_start, second_end, first_end)
    if (
        (first_a > _POLYGON_EPSILON and first_b < -_POLYGON_EPSILON)
        or (first_a < -_POLYGON_EPSILON and first_b > _POLYGON_EPSILON)
    ) and (
        (second_a > _POLYGON_EPSILON and second_b < -_POLYGON_EPSILON)
        or (second_a < -_POLYGON_EPSILON and second_b > _POLYGON_EPSILON)
    ):
        return True
    return (
        _on_segment(second_start, first_start, first_end)
        or _on_segment(second_end, first_start, first_end)
        or _on_segment(first_start, second_start, second_end)
        or _on_segment(first_end, second_start, second_end)
    )


def _validate_polygon(points: list[tuple[float, float]]) -> None:
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("ROI noktaları sonlu sayı olmalı")
    for index, point in enumerate(points):
        if any(
            math.dist(point, other) <= _POLYGON_EPSILON
            for other in points[index + 1 :]
        ):
            raise ValueError("ROI aynı noktayı birden fazla kez içeremez")
    if _polygon_area(points) < _MIN_POLYGON_AREA:
        raise ValueError("ROI alanı çok küçük veya sıfır")
    edge_count = len(points)
    for first in range(edge_count):
        first_end = (first + 1) % edge_count
        for second in range(first + 1, edge_count):
            second_end = (second + 1) % edge_count
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
                raise ValueError("ROI kenarları birbiriyle kesişemez")


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


class RoiEditorService:
    def __init__(
        self,
        *,
        validation_root: Path | None = None,
        data_root: Path | None = None,
        approved_videos: tuple[ApprovedVideo, ...] = APPROVED_VIDEOS,
    ):
        self._validation_root_override = validation_root
        self._data_root_override = data_root
        self._videos = {video.video_id: video for video in approved_videos}
        if len(self._videos) != len(approved_videos):
            raise ValueError("approved video IDs must be unique")
        self._lock = threading.RLock()

    def _validation_root(self) -> Path:
        value = self._validation_root_override or Path(
            os.getenv(
                "DEEPSAFE_VALIDATION_ROOT",
                Path(__file__).resolve().parents[1] / "validation/results",
            )
        )
        return value.expanduser().resolve()

    def _plan_root(self) -> Path:
        data_root = self._data_root_override or Path(
            os.getenv("DEEPSAFE_DATA", "/tmp/deepsafe")
        )
        return (data_root.expanduser().resolve() / "roi-editor/plans")

    def _clean_video_path(
        self,
        video: ApprovedVideo,
        *,
        require_file: bool,
    ) -> Path | None:
        cache_root = (
            self._validation_root() / "content-editor/clean-sources"
        ).resolve()
        candidate = cache_root / video.cache_filename
        try:
            resolved = candidate.resolve(strict=require_file)
        except (FileNotFoundError, OSError):
            return None
        try:
            resolved.relative_to(cache_root)
        except ValueError:
            return None
        if resolved.suffix.lower() != ".mp4":
            return None
        if require_file and (
            candidate.is_symlink() or not resolved.is_file()
        ):
            return None
        return resolved

    def list_videos(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for video in self._videos.values():
            playable = self._clean_video_path(video, require_file=True) is not None
            item: dict[str, Any] = {
                "video_id": video.video_id,
                "display_name": video.display_name,
                "stream_url": f"/api/roi-editor/videos/{video.video_id}/stream",
                "duration_seconds": video.duration_seconds,
                "width": video.width,
                "height": video.height,
                "fps": video.fps,
                "frame_count": video.frame_count,
                "playable": playable,
            }
            if not playable:
                item["unavailable_reason"] = "Temiz MP4 önizlemesi henüz hazır değil"
            items.append(item)
        return {
            "schema_version": "colt-ai.roi-editor-video-list/v1",
            "videos": items,
        }

    def stream_path(self, video_id: str) -> Path:
        video = self._videos.get(video_id)
        if video is None:
            raise RoiEditorError("video_not_found", status_code=404)
        path = self._clean_video_path(video, require_file=True)
        if path is None:
            raise RoiEditorError("clean_preview_unavailable", status_code=409)
        return path

    def create_plan(self, value: ProcessingPlanIn) -> dict[str, Any]:
        video = self._videos.get(value.video_id)
        if video is None:
            raise RoiEditorError("video_not_found", status_code=404)
        if value.end_seconds > video.duration_seconds:
            raise RoiEditorError("clip_exceeds_video")
        start_frame = int(round(value.start_seconds * video.fps))
        end_frame = int(round(value.end_seconds * video.fps))
        if (
            start_frame < 0
            or end_frame > video.frame_count
            or end_frame <= start_frame
        ):
            raise RoiEditorError("clip_has_no_complete_frame")

        plan_id = uuid.uuid4().hex
        created_at = (
            dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            "status": "ready",
            "created_at_utc": created_at,
            "video": {
                "video_id": video.video_id,
                "display_name": video.display_name,
                "source_path": video.source_path,
                "duration_seconds": video.duration_seconds,
                "width": video.width,
                "height": video.height,
                "fps": video.fps,
                "frame_count": video.frame_count,
            },
            "clip": {
                "semantics": "half_open",
                "start_seconds": value.start_seconds,
                "end_seconds": value.end_seconds,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "output_frame_count": end_frame - start_frame,
            },
            "rois": [
                {
                    "roi_id": f"roi-{index + 1}",
                    "name": roi.name,
                    "coordinate_space": "source_video_normalized",
                    "points": [
                        {"x": point.x, "y": point.y}
                        for point in roi.points
                    ],
                    "polygon_norm": [
                        [point.x, point.y]
                        for point in roi.points
                    ],
                }
                for index, roi in enumerate(value.rois)
            ],
            "analytics": {
                "module": "person_detection",
                "detections_path": video.detections_path,
                "sequence_id": video.sequence_id,
                "expected_model_id": video.expected_model_id,
                "occupancy_rule": "bbox_bottom_center_inside",
            },
            "execution": {
                "requested": False,
                "gpu_or_model_execution": False,
                "note": "Plan kaydı herhangi bir işleme süreci başlatmaz.",
            },
        }
        self._write_plan(plan_id, plan)
        return {
            "plan_id": plan_id,
            "status": "ready",
            "plan_url": f"/api/roi-editor/plans/{plan_id}",
            "gpu_or_model_execution": False,
        }

    def _write_plan(self, plan_id: str, plan: dict[str, Any]) -> None:
        content = _canonical_json(plan)
        if len(content) > MAX_PLAN_BYTES:
            raise RoiEditorError("plan_too_large", status_code=413)
        with self._lock:
            root = self._plan_root()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            root = root.resolve()
            destination = root / f"{plan_id}.json"
            temporary = root / f".{plan_id}.{uuid.uuid4().hex}.tmp"
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def load_plan(self, plan_id: str) -> tuple[bytes, str]:
        if not _PLAN_ID_PATTERN.fullmatch(plan_id):
            raise RoiEditorError("plan_not_found", status_code=404)
        with self._lock:
            root = self._plan_root()
            if not root.is_dir():
                raise RoiEditorError("plan_not_found", status_code=404)
            root = root.resolve()
            candidate = root / f"{plan_id}.json"
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise RoiEditorError("plan_not_found", status_code=404) from exc
            if candidate.is_symlink() or not resolved.is_file():
                raise RoiEditorError("plan_not_found", status_code=404)
            if resolved.stat().st_size > MAX_PLAN_BYTES:
                raise RoiEditorError("plan_unavailable", status_code=409)
            content = resolved.read_bytes()
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RoiEditorError("plan_unavailable", status_code=409) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != PLAN_SCHEMA_VERSION
            or payload.get("plan_id") != plan_id
        ):
            raise RoiEditorError("plan_unavailable", status_code=409)
        canonical = _canonical_json(payload)
        etag = '"' + hashlib.sha256(canonical).hexdigest() + '"'
        return canonical, etag


def _http_error(exc: RoiEditorError) -> HTTPException:
    labels = {
        "video_not_found": "Video onaylı listede bulunamadı",
        "clean_preview_unavailable": "Temiz MP4 önizlemesi henüz hazır değil",
        "clip_exceeds_video": "Seçilen zaman aralığı video süresini aşıyor",
        "clip_has_no_complete_frame": "Seçilen aralıkta işlenecek tam kare yok",
        "plan_too_large": "İşleme planı izin verilen boyutu aşıyor",
        "plan_not_found": "İşleme planı bulunamadı",
        "plan_unavailable": "İşleme planı güvenli biçimde okunamıyor",
    }
    return HTTPException(
        exc.status_code,
        labels.get(exc.state, "ROI düzenleyici isteği kullanılamıyor"),
    )


def create_roi_editor_router(
    service: RoiEditorService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    router = APIRouter(prefix="/api/roi-editor", tags=["roi-editor"])

    @router.get("/videos")
    def list_videos(response: Response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return service.list_videos()

    @router.get("/videos/{video_id}/stream")
    def stream_video(video_id: str):
        try:
            path = service.stream_path(video_id)
        except RoiEditorError as exc:
            raise _http_error(exc) from exc
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"colt-ai-clean-source-{video_id}.mp4",
            content_disposition_type="inline",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.post(
        "/plans",
        status_code=201,
        dependencies=[Depends(auth_dependency)],
    )
    def create_plan(value: ProcessingPlanIn):
        try:
            return service.create_plan(value)
        except RoiEditorError as exc:
            raise _http_error(exc) from exc

    @router.get("/plans/{plan_id}")
    def get_plan(plan_id: str):
        try:
            content, etag = service.load_plan(plan_id)
        except RoiEditorError as exc:
            raise _http_error(exc) from exc
        return Response(
            content,
            media_type="application/json",
            headers={
                "ETag": etag,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
