#!/usr/bin/env python3
"""Plan or execute GPU pose rendering for a saved video selection.

The queue remains immutable.  Pose artifacts are written under a separate
delivery root, so the person/ROI pipeline and its state are never modified.
Execution uses the local Apache-2.0 YOLOX-Pose-S checkpoint on CUDA and then
normalizes the OpenCV intermediate to browser-compatible H.264 with NVENC.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPOSITORY_ROOT / "content/video-selector/state/queues"
IMAGE = "deepsafe-mmpose-yoloxpose-export:child-v8-symlink-aware"
WORKER = "content/pose_video.py"
CONFIG = (
    "third_party/mmpose/configs/body_2d_keypoint/yoloxpose/coco/"
    "yoloxpose_s_8xb32-300e_coco-640.py"
)
CHECKPOINT = (
    "models/pose/candidates/mmpose-yoloxpose-s/"
    "yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth"
)


class PoseSelectionError(RuntimeError):
    """Saved selection or execution state is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file(root: Path, value: str, label: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PoseSelectionError(f"{label} escapes repository root") from exc
    if not path.is_file():
        raise PoseSelectionError(f"{label} is not a file: {value}")
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_plan(
    selection_id: str,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    video_ids: set[str] | None = None,
    max_frames: int = 0,
) -> dict[str, Any]:
    queue_path = (
        repository_root
        / "content/video-selector/state/queues"
        / f"{selection_id}.json"
    )
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoseSelectionError(f"cannot load selection queue: {exc}") from exc
    if queue.get("selection_id") != selection_id:
        raise PoseSelectionError("selection_id differs inside queue")
    items = queue.get("items")
    if not isinstance(items, list) or not items:
        raise PoseSelectionError("selection queue has no items")
    if max_frames < 0:
        raise PoseSelectionError("max_frames cannot be negative")

    _repo_file(repository_root, WORKER, "pose worker")
    _repo_file(repository_root, CONFIG, "pose config")
    checkpoint = _repo_file(repository_root, CHECKPOINT, "pose checkpoint")
    jobs: list[dict[str, Any]] = []
    selected: set[str] = set()
    short_id = selection_id[:8]
    for item in items:
        video_id = str(item.get("video_id", ""))
        if video_ids is not None and video_id not in video_ids:
            continue
        selected.add(video_id)
        source_value = str(item.get("source_path", ""))
        source = _repo_file(repository_root, source_value, "source video")
        clip = item.get("clip")
        if not isinstance(clip, Mapping):
            raise PoseSelectionError(f"{video_id}: clip is missing")
        start = float(clip.get("start_seconds", -1))
        end = float(clip.get("end_seconds", -1))
        if start < 0 or end <= start:
            raise PoseSelectionError(f"{video_id}: clip range is invalid")
        source_video = item.get("source_video")
        if not isinstance(source_video, Mapping):
            raise PoseSelectionError(f"{video_id}: source_video is missing")

        relative_output = Path(
            f"validation/results/content-deliveries/"
            f"video-selector-{short_id}/pose/{video_id}"
        )
        output = repository_root / relative_output
        final_name = (
            f"COLT-AI-COLLBRAI-CAM-{video_id}-POSE-{short_id}.mp4"
        )
        jobs.append(
            {
                "video_id": video_id,
                "camera_label": f"CAM-{video_id}",
                "source": source_value,
                "source_sha256": _sha256(source),
                "clip": {"start_seconds": start, "end_seconds": end},
                "expected": {
                    "fps": float(source_video["fps"]),
                    "frames": (
                        min(
                            max_frames,
                            int(round((end - start) * float(source_video["fps"]))),
                        )
                        if max_frames
                        else int(round((end - start) * float(source_video["fps"])))
                    ),
                    "width": int(source_video["width"]),
                    "height": int(source_video["height"]),
                },
                "paths": {
                    "directory": str(relative_output),
                    "intermediate": str(relative_output / "pose-intermediate.mp4"),
                    "keypoints": str(relative_output / "keypoints.jsonl"),
                    "final": str(relative_output / final_name),
                    "manifest": str(relative_output / "pose-manifest.json"),
                    "log": str(relative_output / "pose-worker.log"),
                },
            }
        )
    if video_ids is not None and selected != video_ids:
        missing = sorted(video_ids - selected)
        raise PoseSelectionError(f"video IDs are not in selection: {missing}")
    if not jobs:
        raise PoseSelectionError("no jobs selected")
    return {
        "schema_version": "colt-ai.pose-selection-plan/v1",
        "selection_id": selection_id,
        "queue_path": str(queue_path.relative_to(repository_root)),
        "runtime": {
            "model_family": "YOLOX-Pose-S",
            "checkpoint": CHECKPOINT,
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_license": "Apache-2.0",
            "keypoints": "COCO17",
            "device": "cuda:0",
            "container_image": IMAGE,
            "deepstream_profile": 640,
        },
        "visual_contract": {
            "brand_name": "COLT AI - COLLBRAI",
            "theme_id": "colt-collbrai-navy-v1",
            "model_name_visible": False,
        },
        "max_frames": max_frames,
        "jobs": jobs,
    }


def _probe_video(path: Path, *, cwd: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def _jsonl_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _complete(root: Path, job: Mapping[str, Any]) -> bool:
    final = root / job["paths"]["final"]
    keypoints = root / job["paths"]["keypoints"]
    if not final.is_file() or not keypoints.is_file():
        return False
    try:
        probe = _probe_video(final, cwd=root)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False
    expected = job["expected"]
    return (
        probe["codec"] == "h264"
        and probe["width"] == expected["width"]
        and probe["height"] == expected["height"]
        and probe["frames"] == expected["frames"]
        and _jsonl_count(keypoints) == expected["frames"]
    )


def execute_plan(
    plan: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    keep_intermediate: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        output = repository_root / job["paths"]["directory"]
        output.mkdir(parents=True, exist_ok=True)
        if _complete(repository_root, job):
            final = repository_root / job["paths"]["final"]
            keypoints = repository_root / job["paths"]["keypoints"]
            probe = _probe_video(final, cwd=repository_root)
            manifest_path = repository_root / job["paths"]["manifest"]
            if not manifest_path.exists():
                _atomic_json(
                    manifest_path,
                    {
                        "schema_version": "colt-ai.pose-delivery/v1",
                        "status": "complete",
                        "selection_id": plan["selection_id"],
                        "video_id": job["video_id"],
                        "completed_at_utc": _utc_now(),
                        "runtime": plan["runtime"],
                        "visual_contract": plan["visual_contract"],
                        "source": {
                            "path": job["source"],
                            "sha256": job["source_sha256"],
                            "clip": job["clip"],
                        },
                        "artifacts": {
                            "video": {
                                "path": job["paths"]["final"],
                                "bytes": final.stat().st_size,
                                "sha256": _sha256(final),
                                **probe,
                            },
                            "keypoints": {
                                "path": job["paths"]["keypoints"],
                                "frames": _jsonl_count(keypoints),
                                "bytes": keypoints.stat().st_size,
                                "sha256": _sha256(keypoints),
                            },
                        },
                    },
                )
            results.append(
                {
                    "video_id": job["video_id"],
                    "status": "resumed",
                    "video": job["paths"]["final"],
                    "keypoints": job["paths"]["keypoints"],
                }
            )
            continue
        intermediate = repository_root / job["paths"]["intermediate"]
        keypoints = repository_root / job["paths"]["keypoints"]
        final = repository_root / job["paths"]["final"]
        for candidate in (intermediate, keypoints, final):
            if candidate.exists():
                raise PoseSelectionError(
                    f"{job['video_id']}: partial output exists: {candidate}"
                )

        source = repository_root / job["source"]
        docker = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
            "--ipc=host",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,size=2g",
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-e",
            "HOME=/tmp/home",
            "-e",
            "MPLCONFIGDIR=/tmp/mpl",
            "-e",
            "XDG_CACHE_HOME=/tmp/cache",
            "-e",
            "PYTHONNOUSERSITE=1",
            "-v",
            f"{repository_root}:/workspace:ro",
            "-v",
            f"{output}:/output:rw",
            "--entrypoint",
            "/opt/deepsafe-export/bin/python3",
            IMAGE,
            f"/workspace/{WORKER}",
            "--input",
            f"/workspace/{job['source']}",
            "--output",
            "/output/pose-intermediate.mp4",
            "--predictions-jsonl",
            "/output/keypoints.jsonl",
            "--config",
            f"/workspace/{CONFIG}",
            "--checkpoint",
            f"/workspace/{CHECKPOINT}",
            "--start-seconds",
            str(job["clip"]["start_seconds"]),
            "--end-seconds",
            str(job["clip"]["end_seconds"]),
            "--camera-label",
            job["camera_label"],
            "--device",
            "cuda:0",
            "--keypoint-threshold",
            "0.25",
        ]
        if plan["max_frames"]:
            docker.extend(["--max-frames", str(plan["max_frames"])])
        log = repository_root / job["paths"]["log"]
        with log.open("w", encoding="utf-8") as stream:
            completed = command_runner(
                docker,
                cwd=repository_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise PoseSelectionError(
                f"{job['video_id']}: pose worker failed; see {log}"
            )
        if _jsonl_count(keypoints) != job["expected"]["frames"]:
            raise PoseSelectionError(
                f"{job['video_id']}: keypoint frame count differs"
            )

        transcode = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(final),
        ]
        transcoded = command_runner(
            transcode,
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if transcoded.returncode != 0:
            raise PoseSelectionError(
                f"{job['video_id']}: NVENC transcode failed: "
                f"{transcoded.stderr[-500:]}"
            )
        probe = _probe_video(final, cwd=repository_root)
        if (
            probe["codec"] != "h264"
            or probe["frames"] != job["expected"]["frames"]
            or probe["width"] != job["expected"]["width"]
            or probe["height"] != job["expected"]["height"]
        ):
            raise PoseSelectionError(
                f"{job['video_id']}: final video contract differs: {probe}"
            )
        if not keep_intermediate:
            intermediate.unlink()
        manifest = {
            "schema_version": "colt-ai.pose-delivery/v1",
            "status": "complete",
            "selection_id": plan["selection_id"],
            "video_id": job["video_id"],
            "completed_at_utc": _utc_now(),
            "runtime": plan["runtime"],
            "visual_contract": plan["visual_contract"],
            "source": {
                "path": job["source"],
                "sha256": job["source_sha256"],
                "clip": job["clip"],
            },
            "artifacts": {
                "video": {
                    "path": job["paths"]["final"],
                    "bytes": final.stat().st_size,
                    "sha256": _sha256(final),
                    **probe,
                },
                "keypoints": {
                    "path": job["paths"]["keypoints"],
                    "frames": _jsonl_count(keypoints),
                    "bytes": keypoints.stat().st_size,
                    "sha256": _sha256(keypoints),
                },
            },
        }
        _atomic_json(repository_root / job["paths"]["manifest"], manifest)
        results.append(
            {
                "video_id": job["video_id"],
                "status": "complete",
                "video": job["paths"]["final"],
                "keypoints": job["paths"]["keypoints"],
            }
        )
    return {
        "schema_version": "colt-ai.pose-selection-run/v1",
        "status": "complete",
        "selection_id": plan["selection_id"],
        "results": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--selection-id", required=True)
    result.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Process only this ID; may be repeated.",
    )
    result.add_argument("--max-frames", type=int, default=0)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--keep-intermediate", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build_plan(
            args.selection_id,
            video_ids=set(args.video_ids) if args.video_ids else None,
            max_frames=args.max_frames,
        )
        result = (
            execute_plan(plan, keep_intermediate=args.keep_intermediate)
            if args.execute
            else plan
        )
    except (OSError, ValueError, KeyError, PoseSelectionError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
