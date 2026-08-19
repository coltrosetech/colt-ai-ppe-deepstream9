"""CPU-only render jobs for saved ROI editor plans.

The worker consumes the already-produced, allow-listed prediction JSONL files.
It never starts DeepStream, CUDA, a model, or inference.  A saved plan is
revalidated against the closed video registry before a subprocess is queued.
"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import Response

from .roi_editor import (
    APPROVED_VIDEOS,
    PLAN_SCHEMA_VERSION,
    ApprovedVideo,
    ProcessingPlanIn,
    RoiEditorError,
    RoiEditorService,
)


JOB_SCHEMA_VERSION = "colt-ai.roi-render-job/v1"
MAX_JOB_BYTES = 64 * 1024
_JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_PLAN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TERMINATE_GRACE_SECONDS = 3.0


class RoiRenderJobError(ValueError):
    def __init__(self, state: str, *, status_code: int = 409):
        super().__init__(state)
        self.state = state
        self.status_code = status_code


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


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class RoiRenderJobService:
    """A restart-aware, single-concurrency CPU render queue."""

    def __init__(
        self,
        plan_service: RoiEditorService,
        *,
        approved_videos: tuple[ApprovedVideo, ...] = APPROVED_VIDEOS,
        repository_root: Path | None = None,
        data_root: Path | None = None,
        render_root: Path | None = None,
        process_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    ):
        self._plan_service = plan_service
        self._videos = {video.video_id: video for video in approved_videos}
        if len(self._videos) != len(approved_videos):
            raise ValueError("approved video IDs must be unique")
        self._repository_root_override = repository_root
        self._data_root_override = data_root
        self._render_root_override = render_root
        self._process_factory = process_factory
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: set[concurrent.futures.Future[None]] = set()
        self._active_process: subprocess.Popen[Any] | None = None
        self._session_job_ids: set[str] = set()

    def _repository_root(self) -> Path:
        value = self._repository_root_override or Path(
            os.getenv(
                "DEEPSAFE_REPOSITORY_ROOT",
                Path(__file__).resolve().parents[1],
            )
        )
        return value.expanduser().resolve()

    def _job_root(self) -> Path:
        data_root = self._data_root_override or Path(
            os.getenv("DEEPSAFE_DATA", "/tmp/deepsafe")
        )
        return data_root.expanduser().resolve() / "roi-editor/jobs"

    def _render_root(self) -> Path:
        repository_root = self._repository_root()
        value = self._render_root_override or Path(
            os.getenv(
                "DEEPSAFE_ROI_RENDER_ROOT",
                repository_root / "runtime-results/roi-editor/jobs",
            )
        )
        resolved = value.expanduser().resolve()
        try:
            resolved.relative_to(repository_root)
        except ValueError as exc:
            raise RoiRenderJobError(
                "render_root_unavailable",
                status_code=503,
            ) from exc
        return resolved

    def start(self) -> None:
        """Start a fresh executor and fail stale in-flight states closed."""

        with self._lock:
            if self._executor is not None:
                return
            self._stop_event.clear()
            self._session_job_ids.clear()
            self._recover_interrupted_jobs()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="roi-cpu-render",
            )

    def shutdown(self) -> None:
        """Cancel queued work and stop the active process group."""

        with self._lock:
            executor = self._executor
            if executor is None:
                return
            self._executor = None
            self._stop_event.set()
            process = self._active_process
            session_jobs = tuple(self._session_job_ids)

        if process is not None:
            self._terminate_process_group(process)
        executor.shutdown(wait=True, cancel_futures=True)

        for job_id in session_jobs:
            try:
                state = self._load_job(job_id)
            except RoiRenderJobError:
                continue
            if state["status"] in {"queued", "running"}:
                self._update_job(
                    job_id,
                    status="error",
                    progress=int(state.get("progress", 0)),
                    message="Yönetim servisi kapatıldığı için iş durduruldu.",
                    finished_at_utc=_utc_now(),
                    output_url=None,
                )

        with self._lock:
            self._active_process = None
            self._futures.clear()
            self._session_job_ids.clear()

    def _recover_interrupted_jobs(self) -> None:
        root = self._job_root()
        if not root.is_dir():
            return
        for path in root.glob("*.json"):
            if not _JOB_ID_PATTERN.fullmatch(path.stem):
                continue
            try:
                state = self._load_job(path.stem)
            except RoiRenderJobError:
                continue
            if state["status"] in {"queued", "running"}:
                self._update_job(
                    path.stem,
                    status="error",
                    progress=int(state.get("progress", 0)),
                    message=(
                        "Yönetim servisi yeniden başlatıldığı için önceki iş "
                        "durduruldu."
                    ),
                    finished_at_utc=_utc_now(),
                    output_url=None,
                )

    def enqueue(self, plan_id: str) -> dict[str, Any]:
        if not _PLAN_ID_PATTERN.fullmatch(plan_id):
            raise RoiRenderJobError("plan_not_found", status_code=404)
        with self._lock:
            executor = self._executor
            if executor is None or self._stop_event.is_set():
                raise RoiRenderJobError(
                    "render_service_unavailable",
                    status_code=503,
                )

        plan, video = self._load_and_revalidate_plan(plan_id)
        job_id = uuid.uuid4().hex
        config = self._build_config(job_id, plan, video)
        render_job_dir = self._render_job_dir(job_id, create=True)
        config_path = render_job_dir / "request.json"
        self._atomic_write(config_path, config, max_bytes=MAX_JOB_BYTES)

        now = _utc_now()
        state = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "plan_id": plan_id,
            "video_id": video.video_id,
            "status": "queued",
            "progress": 0,
            "message": "CPU video işleme sırasına alındı.",
            "created_at_utc": now,
            "updated_at_utc": now,
            "started_at_utc": None,
            "finished_at_utc": None,
            "output_url": None,
            "gpu_or_model_execution": False,
        }
        self._write_job(state)

        with self._lock:
            executor = self._executor
            if executor is None or self._stop_event.is_set():
                self._update_job(
                    job_id,
                    status="error",
                    message="Yönetim servisi kapatıldığı için iş başlatılamadı.",
                    finished_at_utc=_utc_now(),
                )
                raise RoiRenderJobError(
                    "render_service_unavailable",
                    status_code=503,
                )
            self._session_job_ids.add(job_id)
            future = executor.submit(self._run_job, job_id, config_path)
            self._futures.add(future)
            future.add_done_callback(self._discard_future)

        return {
            "job_id": job_id,
            "status": "queued",
            "job_url": f"/api/roi-editor/jobs/{job_id}",
            "gpu_or_model_execution": False,
        }

    def _discard_future(
        self,
        future: concurrent.futures.Future[None],
    ) -> None:
        with self._lock:
            self._futures.discard(future)

    def _load_and_revalidate_plan(
        self,
        plan_id: str,
    ) -> tuple[dict[str, Any], ApprovedVideo]:
        try:
            content, _etag = self._plan_service.load_plan(plan_id)
        except RoiEditorError as exc:
            status_code = 404 if exc.state == "plan_not_found" else 409
            raise RoiRenderJobError(exc.state, status_code=status_code) from exc
        try:
            plan = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RoiRenderJobError("plan_unavailable") from exc
        if not isinstance(plan, dict):
            raise RoiRenderJobError("plan_unavailable")

        expected_top = {
            "schema_version",
            "plan_id",
            "status",
            "created_at_utc",
            "video",
            "clip",
            "rois",
            "analytics",
            "execution",
        }
        if (
            set(plan) != expected_top
            or plan.get("schema_version") != PLAN_SCHEMA_VERSION
            or plan.get("plan_id") != plan_id
            or plan.get("status") != "ready"
            or not isinstance(plan.get("created_at_utc"), str)
        ):
            raise RoiRenderJobError("plan_unavailable")

        video_value = plan.get("video")
        if not isinstance(video_value, dict):
            raise RoiRenderJobError("plan_unavailable")
        video = self._videos.get(video_value.get("video_id"))
        if video is None:
            raise RoiRenderJobError("video_not_approved")
        expected_video = {
            "video_id": video.video_id,
            "display_name": video.display_name,
            "source_path": video.source_path,
            "duration_seconds": video.duration_seconds,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frame_count": video.frame_count,
        }
        if video_value != expected_video:
            raise RoiRenderJobError("plan_video_binding_invalid")

        analytics = plan.get("analytics")
        if analytics != {
            "module": "person_detection",
            "detections_path": video.detections_path,
            "sequence_id": video.sequence_id,
            "expected_model_id": video.expected_model_id,
            "occupancy_rule": "bbox_bottom_center_inside",
        }:
            raise RoiRenderJobError("plan_analytics_binding_invalid")
        if plan.get("execution") != {
            "requested": False,
            "gpu_or_model_execution": False,
            "note": "Plan kaydı herhangi bir işleme süreci başlatmaz.",
        }:
            raise RoiRenderJobError("plan_execution_binding_invalid")

        clip = plan.get("clip")
        rois = plan.get("rois")
        if (
            not isinstance(clip, dict)
            or not isinstance(rois, list)
            or len(rois) != 1
            or not isinstance(rois[0], dict)
        ):
            raise RoiRenderJobError("single_roi_required")
        roi = rois[0]
        if (
            set(roi)
            != {
                "roi_id",
                "name",
                "coordinate_space",
                "points",
                "polygon_norm",
            }
            or roi.get("roi_id") != "roi-1"
            or roi.get("coordinate_space") != "source_video_normalized"
            or not isinstance(roi.get("name"), str)
            or not isinstance(roi.get("points"), list)
            or not isinstance(roi.get("polygon_norm"), list)
        ):
            raise RoiRenderJobError("plan_roi_invalid")
        if set(clip) != {
            "semantics",
            "start_seconds",
            "end_seconds",
            "start_frame",
            "end_frame_exclusive",
            "output_frame_count",
        } or clip.get("semantics") != "half_open":
            raise RoiRenderJobError("plan_clip_invalid")
        if not _is_number(clip.get("start_seconds")) or not _is_number(
            clip.get("end_seconds")
        ):
            raise RoiRenderJobError("plan_clip_invalid")

        try:
            validated = ProcessingPlanIn.model_validate(
                {
                    "video_id": video.video_id,
                    "start_seconds": clip["start_seconds"],
                    "end_seconds": clip["end_seconds"],
                    "rois": [
                        {
                            "name": roi["name"],
                            "points": roi["points"],
                        }
                    ],
                }
            )
        except (TypeError, ValueError) as exc:
            raise RoiRenderJobError("plan_geometry_invalid") from exc
        if validated.end_seconds > video.duration_seconds:
            raise RoiRenderJobError("plan_clip_invalid")
        start_frame = int(round(validated.start_seconds * video.fps))
        end_frame = int(round(validated.end_seconds * video.fps))
        if (
            start_frame < 0
            or end_frame > video.frame_count
            or end_frame <= start_frame
        ):
            raise RoiRenderJobError("plan_clip_invalid")
        expected_clip = {
            "semantics": "half_open",
            "start_seconds": validated.start_seconds,
            "end_seconds": validated.end_seconds,
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "output_frame_count": end_frame - start_frame,
        }
        expected_points = [
            {"x": point.x, "y": point.y}
            for point in validated.rois[0].points
        ]
        expected_roi = {
            "roi_id": "roi-1",
            "name": validated.rois[0].name,
            "coordinate_space": "source_video_normalized",
            "points": expected_points,
            "polygon_norm": [
                [point["x"], point["y"]]
                for point in expected_points
            ],
        }
        if clip != expected_clip or roi != expected_roi:
            raise RoiRenderJobError("plan_unavailable")

        self._approved_regular_file(
            video.source_path,
            state="approved_source_unavailable",
        )
        self._approved_regular_file(
            video.detections_path,
            state="approved_detections_unavailable",
        )
        return plan, video

    def _approved_regular_file(self, relative: str, *, state: str) -> Path:
        repository_root = self._repository_root()
        lexical = repository_root / relative
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(repository_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RoiRenderJobError(state) from exc
        if lexical.is_symlink() or not resolved.is_file():
            raise RoiRenderJobError(state)
        return resolved

    def _load_template(
        self,
        video: ApprovedVideo,
    ) -> dict[str, Any]:
        relative = Path(
            f"content/configs/colt-collbrai-person-full-{video.video_id}.json"
        )
        path = self._approved_regular_file(
            relative.as_posix(),
            state="render_template_unavailable",
        )
        try:
            template = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RoiRenderJobError("render_template_unavailable") from exc
        if not isinstance(template, dict):
            raise RoiRenderJobError("render_template_unavailable")
        source = template.get("source")
        detections = template.get("detections")
        disclosure = template.get("disclosure")
        if (
            not isinstance(source, dict)
            or source.get("video_path") != video.source_path
            or not isinstance(disclosure, dict)
            or not isinstance(template.get("title"), str)
            or not isinstance(detections, dict)
            or detections.get("kind") != "predictions_jsonl"
            or detections.get("path") != video.detections_path
            or detections.get("sequence_id") != video.sequence_id
            or detections.get("expected_model_id") != video.expected_model_id
        ):
            raise RoiRenderJobError("render_template_binding_invalid")
        return template

    def _build_config(
        self,
        job_id: str,
        plan: dict[str, Any],
        video: ApprovedVideo,
    ) -> dict[str, Any]:
        template = self._load_template(video)
        roi = plan["rois"][0]
        output_directory = (
            self._render_job_dir(job_id, create=False) / "output"
        ).relative_to(self._repository_root())
        return {
            "schema_version": "deepsafe.content-roi-demo/v1",
            "demo_id": f"roi-job-{job_id}",
            "title": str(template.get("title", "İNSAN TESPİTİ"))[:80],
            "camera_label": f"CAM-{video.video_id}",
            "disclosure": copy.deepcopy(template["disclosure"]),
            "source": copy.deepcopy(template["source"]),
            "detections": copy.deepcopy(template["detections"]),
            "clip": {
                "start_seconds": plan["clip"]["start_seconds"],
                "end_seconds": plan["clip"]["end_seconds"],
            },
            "roi": {
                "id": "operator-roi",
                "label": roi["name"][:40],
                "polygon_norm": copy.deepcopy(roi["polygon_norm"]),
            },
            "event_policy": {
                "rule": "bbox_bottom_center_inside",
                "enter_debounce_frames": 3,
                "exit_debounce_frames": 15,
            },
            "output": {
                "directory": output_directory.as_posix(),
                "width": 1920,
                "height": 1080,
                "codec": "libx264",
                "crf": 16,
                "preset": "fast",
            },
        }

    def _run_job(self, job_id: str, config_path: Path) -> None:
        if self._stop_event.is_set():
            self._update_job(
                job_id,
                status="error",
                message="Yönetim servisi kapatıldığı için iş başlatılmadı.",
                finished_at_utc=_utc_now(),
            )
            return
        self._update_job(
            job_id,
            status="running",
            progress=5,
            message=(
                "Kayıtlı insan tespitleriyle CPU üzerinde video hazırlanıyor."
            ),
            started_at_utc=_utc_now(),
        )
        command = [
            sys.executable,
            "-m",
            "content.roi_demo",
            "--config",
            str(config_path),
            "--render",
            "--allow-missing-preview-states",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
        log_path = config_path.parent / "renderer.log"

        process: subprocess.Popen[Any] | None = None
        try:
            with log_path.open("wb") as log_handle:
                process = self._process_factory(
                    command,
                    cwd=str(self._repository_root()),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=True,
                )
                with self._lock:
                    self._active_process = process
                while process.poll() is None:
                    if self._stop_event.wait(0.25):
                        self._terminate_process_group(process)
                        raise RoiRenderJobError("render_stopped")
                return_code = process.wait()
            if return_code != 0:
                raise RoiRenderJobError("renderer_failed")
            output = self._verified_output_path(job_id)
            if output.stat().st_size <= 0:
                raise RoiRenderJobError("renderer_output_unavailable")
            self._verify_manifest(job_id)
            self._update_job(
                job_id,
                status="complete",
                progress=100,
                message="Video işleme tamamlandı.",
                finished_at_utc=_utc_now(),
                output_url=f"/api/roi-editor/jobs/{job_id}/output",
            )
        except BaseException as exc:
            if process is not None and process.poll() is None:
                self._terminate_process_group(process)
            state = (
                exc.state
                if isinstance(exc, RoiRenderJobError)
                else "renderer_failed"
            )
            labels = {
                "render_stopped": "Video işleme servis kapanışıyla durduruldu.",
                "renderer_output_unavailable": "İşlem tamamlandı ancak MP4 üretilemedi.",
                "renderer_manifest_invalid": "İşlem çıktısı güvenlik doğrulamasını geçemedi.",
                "renderer_failed": "CPU video işleme tamamlanamadı.",
            }
            try:
                current = self._load_job(job_id)
                progress = int(current.get("progress", 5))
                if current["status"] != "complete":
                    self._update_job(
                        job_id,
                        status="error",
                        progress=progress,
                        message=labels.get(
                            state,
                            "CPU video işleme tamamlanamadı.",
                        ),
                        finished_at_utc=_utc_now(),
                        output_url=None,
                    )
            except RoiRenderJobError:
                pass
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None

    def _verify_manifest(self, job_id: str) -> None:
        path = self._render_job_dir(job_id, create=False) / "output/manifest.json"
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._render_root())
            payload = json.loads(resolved.read_bytes())
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise RoiRenderJobError("renderer_manifest_invalid") from exc
        if (
            path.is_symlink()
            or not resolved.is_file()
            or not isinstance(payload, dict)
            or payload.get("status") != "rendered"
            or payload.get("demo_id") != f"roi-job-{job_id}"
            or payload.get("gpu_or_model_execution") is not False
        ):
            raise RoiRenderJobError("renderer_manifest_invalid")

    def _terminate_process_group(
        self,
        process: subprocess.Popen[Any],
    ) -> None:
        if process.poll() is not None:
            return
        pid = getattr(process, "pid", None)
        try:
            if isinstance(pid, int) and pid > 0:
                os.killpg(pid, signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                if isinstance(pid, int) and pid > 0:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, TypeError):
            pass

    def get_job(self, job_id: str) -> dict[str, Any]:
        state = self._load_job(job_id)
        result = {
            key: state[key]
            for key in (
                "job_id",
                "plan_id",
                "video_id",
                "status",
                "progress",
                "message",
                "created_at_utc",
                "updated_at_utc",
                "started_at_utc",
                "finished_at_utc",
                "output_url",
                "gpu_or_model_execution",
            )
        }
        result["progress_percent"] = result["progress"]
        return result

    def output_path(self, job_id: str) -> tuple[Path, str]:
        state = self._load_job(job_id)
        if state["status"] != "complete":
            raise RoiRenderJobError("job_not_complete")
        return (
            self._verified_output_path(job_id),
            (
                f"COLT-AI-COLLBRAI-CAM-{state['video_id']}-"
                f"{job_id[:8]}.mp4"
            ),
        )

    def _verified_output_path(self, job_id: str) -> Path:
        path = self._render_job_dir(job_id, create=False) / "output/demo.mp4"
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._render_root())
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RoiRenderJobError("renderer_output_unavailable") from exc
        if path.is_symlink() or not resolved.is_file():
            raise RoiRenderJobError("renderer_output_unavailable")
        return resolved

    def _render_job_dir(self, job_id: str, *, create: bool) -> Path:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise RoiRenderJobError("job_not_found", status_code=404)
        root = self._render_root()
        if create:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = root.resolve()
        path = root / job_id
        if create:
            try:
                path.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise RoiRenderJobError("job_conflict") from exc
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise RoiRenderJobError("render_root_unavailable", status_code=503) from exc
        if path.is_symlink():
            raise RoiRenderJobError("render_root_unavailable", status_code=503)
        return path

    def _job_path(self, job_id: str) -> Path:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise RoiRenderJobError("job_not_found", status_code=404)
        return self._job_root() / f"{job_id}.json"

    def _write_job(self, value: dict[str, Any]) -> None:
        self._atomic_write(
            self._job_path(value["job_id"]),
            value,
            max_bytes=MAX_JOB_BYTES,
        )

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            value = self._load_job(job_id)
            value.update(changes)
            value["updated_at_utc"] = _utc_now()
            self._write_job(value)

    def _load_job(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        with self._lock:
            root = self._job_root()
            if not root.is_dir():
                raise RoiRenderJobError("job_not_found", status_code=404)
            root = root.resolve()
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise RoiRenderJobError("job_not_found", status_code=404) from exc
            if path.is_symlink() or not resolved.is_file():
                raise RoiRenderJobError("job_not_found", status_code=404)
            if resolved.stat().st_size > MAX_JOB_BYTES:
                raise RoiRenderJobError("job_unavailable")
            try:
                value = json.loads(resolved.read_bytes())
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise RoiRenderJobError("job_unavailable") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != JOB_SCHEMA_VERSION
            or value.get("job_id") != job_id
            or value.get("status")
            not in {"queued", "running", "complete", "error"}
            or value.get("gpu_or_model_execution") is not False
        ):
            raise RoiRenderJobError("job_unavailable")
        return value

    def _atomic_write(
        self,
        path: Path,
        value: dict[str, Any],
        *,
        max_bytes: int,
    ) -> None:
        content = _canonical_json(value)
        if len(content) > max_bytes:
            raise RoiRenderJobError("job_unavailable")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            root = path.parent.resolve()
            try:
                path.resolve().relative_to(root)
            except ValueError as exc:
                raise RoiRenderJobError("job_unavailable") from exc
            temporary = root / f".{path.name}.{uuid.uuid4().hex}.tmp"
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
                os.replace(temporary, path)
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


def _http_error(exc: RoiRenderJobError) -> HTTPException:
    labels = {
        "plan_not_found": "İşleme planı bulunamadı",
        "plan_unavailable": "İşleme planı güvenli biçimde doğrulanamadı",
        "video_not_approved": "Plan videosu onaylı listede değil",
        "plan_video_binding_invalid": "Planın video bağlantısı değişmiş",
        "plan_analytics_binding_invalid": "Planın algılama bağlantısı değişmiş",
        "plan_execution_binding_invalid": "Planın yürütme bağlantısı değişmiş",
        "plan_clip_invalid": "Planın zaman aralığı geçersiz",
        "plan_roi_invalid": "Planın ROI verisi geçersiz",
        "plan_geometry_invalid": "Planın ROI geometrisi geçersiz",
        "single_roi_required": "Bu işlem için planda tam olarak bir ROI olmalı",
        "approved_source_unavailable": "Onaylı kaynak video kullanılamıyor",
        "approved_detections_unavailable": "Kayıtlı insan tespitleri kullanılamıyor",
        "render_template_unavailable": "Onaylı video şablonu kullanılamıyor",
        "render_template_binding_invalid": "Video şablonu onaylı kaynakla eşleşmiyor",
        "render_root_unavailable": "CPU video çıktı alanı kullanılamıyor",
        "render_service_unavailable": "CPU video işleme servisi hazır değil",
        "job_not_found": "Video işleme işi bulunamadı",
        "job_unavailable": "Video işleme işi güvenli biçimde okunamıyor",
        "job_not_complete": "Video çıktısı henüz hazır değil",
        "renderer_output_unavailable": "İşlenmiş MP4 kullanılamıyor",
    }
    return HTTPException(
        exc.status_code,
        labels.get(exc.state, "CPU video işleme isteği kullanılamıyor"),
    )


def create_roi_render_jobs_router(
    service: RoiRenderJobService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    router = APIRouter(prefix="/api/roi-editor", tags=["roi-editor"])

    @router.post(
        "/plans/{plan_id}/execute",
        status_code=202,
        dependencies=[Depends(auth_dependency)],
    )
    def execute_plan(plan_id: str):
        try:
            return service.enqueue(plan_id)
        except RoiRenderJobError as exc:
            raise _http_error(exc) from exc

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str, response: Response):
        try:
            value = service.get_job(job_id)
        except RoiRenderJobError as exc:
            raise _http_error(exc) from exc
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return value

    @router.get("/jobs/{job_id}/output")
    def get_job_output(job_id: str):
        try:
            path, filename = service.output_path(job_id)
        except RoiRenderJobError as exc:
            raise _http_error(exc) from exc
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=filename,
            content_disposition_type="attachment",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
