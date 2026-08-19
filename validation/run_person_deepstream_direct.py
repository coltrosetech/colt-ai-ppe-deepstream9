#!/usr/bin/env python3
"""Run the dedicated YOLO11s person detector through DeepStream 9.

This module is the small, direct person-inference companion to
``run_ppe_deepstream``.  It deliberately has no token, approval artifact,
GPU-guard, or HTTP boundary: an explicit Python/CLI execute call runs one
already-prepared clip and writes a complete frame-aligned person JSONL stream.

The primary detector feeds DeepStream's NvDCF tracker.  The frame-aligned
output retains each ``NvDsObjectMeta.object_id`` as ``track_id``, so
``content.person_ppe_fusion`` can associate helmet/hi-vis observations with
canonical person tracks even when the PPE checkpoint's own optional
``Person`` class produces no detections.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from validation.run_caviar import (
    attest_engine_load,
    calculate_streammux_dimensions,
    probe_video,
    render_deepstream_config_paths,
    render_infer_config,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "colt-ai.person-deepstream-direct/v2"
CONTAINER_IMAGE = "deepsafe-deepstream:9.0-control-refresh-20260725"
SUPPORTED_PROFILES = (640, 960)
EXPORT_THRESHOLD = 0.05
MODEL_ID_TEMPLATE = "yolo11s-person-{profile}"
PERSON_CLASS_ID = 0
FORKLIFT_CANDIDATE_CLASS_ID = 7
TRACKED_CLASS_NAMES = {
    PERSON_CLASS_ID: "person",
    FORKLIFT_CANDIDATE_CLASS_ID: "forklift_candidate",
}
TRACKER_NAME = "NvDCF-perf"
TRACKER_LIBRARY = (
    "/opt/nvidia/deepstream/deepstream/lib/"
    "libnvds_nvmultiobjecttracker.so"
)
TRACKER_CONFIG = (
    "/opt/nvidia/deepstream/deepstream-9.0/samples/configs/"
    "deepstream-app/config_tracker_NvDCF_perf.yml"
)
TRACKER_DIMENSIONS = {
    640: (640, 384),
    960: (960, 544),
}
UNTRACKED_OBJECT_ID = (1 << 64) - 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_KITTI_NAME = re.compile(
    r"^(?P<app_index>\d{2})_(?P<source_id>\d{3})_"
    r"(?P<frame_index>\d{6,})\.txt$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESUME_ARTIFACT_PATHS = {
    "predictions": "predictions",
    "conversion": "conversion",
    "deepstream_log": "deepstream_log",
    "deepstream_config": "deepstream_config",
    "infer_config": "infer_config",
}


class PersonDeepStreamError(RuntimeError):
    """Raised when direct person inference cannot be verified."""


CommandRunner = Callable[[Sequence[str], Path], None]
VideoProbe = Callable[[Path], Mapping[str, object]]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_name)
    try:
        os.fchmod(descriptor, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
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


def _atomic_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_json(value))


def _repository_path(
    value: str | Path, *, label: str, must_exist: bool
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PersonDeepStreamError(f"{label} escapes repository root") from exc
    if must_exist and not candidate.is_file():
        raise PersonDeepStreamError(f"{label} is not a file: {candidate}")
    return candidate


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PersonDeepStreamError(f"artifact is absent or empty: {path}")
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, list(command))


def _render_tracked_app_config(
    *,
    video_container_path: str,
    kitti_container_path: str,
    tracker_kitti_container_path: str,
    infer_config_container_path: str,
    width: int,
    height: int,
    profile: int,
) -> str:
    """Render the reference app with NvDCF and native tracker KITTI output."""

    base = render_deepstream_config_paths(
        video_container_path=video_container_path,
        kitti_container_path=kitti_container_path,
        infer_config_container_path=infer_config_container_path,
        width=width,
        height=height,
    )
    base = base.replace(
        f"gie-kitti-output-dir={kitti_container_path}\n",
        (
            f"gie-kitti-output-dir={kitti_container_path}\n"
            f"kitti-track-output-dir={tracker_kitti_container_path}\n"
        ),
        1,
    )
    tracker_width, tracker_height = TRACKER_DIMENSIONS[profile]
    tracker_section = f"""[tracker]
enable=1
tracker-width={tracker_width}
tracker-height={tracker_height}
ll-lib-file={TRACKER_LIBRARY}
ll-config-file={TRACKER_CONFIG}
gpu-id=0
display-tracking-id=0

"""
    return base.replace("[tests]\n", tracker_section + "[tests]\n", 1)


def _tracked_kitti_frames(directory: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for path in sorted(directory.glob("*.txt")):
        match = _KITTI_NAME.fullmatch(path.name)
        if match is None:
            continue
        if int(match["app_index"]) != 0 or int(match["source_id"]) != 0:
            continue
        frame_index = int(match["frame_index"])
        if frame_index in frames:
            raise ValueError(f"{directory}: duplicate tracker frame {frame_index}")
        frames[frame_index] = path
    if not frames:
        raise ValueError(f"{directory}: no NvDCF tracker KITTI files")
    return frames


def _convert_tracked_kitti_directory(
    directory: Path,
    output: Path,
    *,
    sequence_id: str,
    image_width: int,
    image_height: int,
    coordinate_width: int,
    coordinate_height: int,
    expected_frames: int,
    fps: float,
    source_uri: str,
    model_id: str,
    included_class_ids: Sequence[int] = (PERSON_CLASS_ID,),
) -> dict[str, object]:
    """Convert DeepStream's tracker KITTI format, retaining ``object_id``."""

    frames = _tracked_kitti_frames(directory)
    expected = set(range(expected_frames))
    actual = set(frames)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "tracker KITTI frame sequence mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    class_ids = tuple(included_class_ids)
    if (
        not class_ids
        or len(class_ids) != len(set(class_ids))
        or any(class_id not in TRACKED_CLASS_NAMES for class_id in class_ids)
        or PERSON_CLASS_ID not in class_ids
    ):
        raise ValueError("tracked class IDs must contain person and supported IDs")
    source_labels = {
        "person": PERSON_CLASS_ID,
        # The shared COCO engine labels class 7 as truck.  It is deliberately
        # exported as a candidate rather than overclaiming exact forklift
        # classification.
        "truck": FORKLIFT_CANDIDATE_CLASS_ID,
    }
    detections_total = 0
    detections_by_class = {
        TRACKED_CLASS_NAMES[class_id]: 0 for class_id in class_ids
    }
    clipped_total = 0
    dropped_total = 0
    unique_ids: set[int] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for frame_index in range(expected_frames):
                detections: list[dict[str, object]] = []
                source = frames[frame_index]
                for line_number, raw in enumerate(
                    source.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if not raw.strip():
                        continue
                    fields = raw.split()
                    # DeepStream tracker KITTI is:
                    # label, object_id, then the standard 15 numeric fields.
                    if len(fields) < 17:
                        raise ValueError(
                            f"{source}:{line_number}: incomplete tracker KITTI row"
                        )
                    source_class_id = source_labels.get(fields[0].casefold())
                    if (
                        source_class_id is None
                        or source_class_id not in class_ids
                    ):
                        continue
                    try:
                        track_id = int(fields[1])
                        values = [float(value) for value in fields[2:17]]
                    except ValueError as exc:
                        raise ValueError(
                            f"{source}:{line_number}: invalid tracker KITTI field"
                        ) from exc
                    if (
                        track_id < 0
                        or track_id == UNTRACKED_OBJECT_ID
                        or not all(math.isfinite(value) for value in values)
                    ):
                        raise ValueError(
                            f"{source}:{line_number}: invalid native track metadata"
                        )
                    left, top, right, bottom = values[3:7]
                    raw_tracker_confidence = values[14]
                    if raw_tracker_confidence < 0.0:
                        raise ValueError(
                            f"{source}:{line_number}: negative tracker confidence"
                        )
                    # NvDCF's correlation response can exceed 1.0 by a small
                    # amount (observed up to 1.016471 on the DS9 P01 run).
                    # Keep the raw diagnostic value while satisfying the
                    # person-detection confidence contract.
                    tracker_confidence = min(1.0, raw_tracker_confidence)
                    clipped = (
                        max(0.0, min(float(coordinate_width), left)),
                        max(0.0, min(float(coordinate_height), top)),
                        max(0.0, min(float(coordinate_width), right)),
                        max(0.0, min(float(coordinate_height), bottom)),
                    )
                    if clipped != (left, top, right, bottom):
                        clipped_total += 1
                    left, top, right, bottom = clipped
                    if right <= left or bottom <= top:
                        dropped_total += 1
                        continue
                    detections.append(
                        {
                            "class_id": source_class_id,
                            "class_name": TRACKED_CLASS_NAMES[source_class_id],
                            "detector_class_name": fields[0].casefold(),
                            "confidence": tracker_confidence,
                            "tracker_confidence": tracker_confidence,
                            "tracker_confidence_raw": raw_tracker_confidence,
                            "track_id": track_id,
                            "bbox_norm_xywh": [
                                round(left / coordinate_width, 10),
                                round(top / coordinate_height, 10),
                                round((right - left) / coordinate_width, 10),
                                round((bottom - top) / coordinate_height, 10),
                            ],
                        }
                    )
                    detections_total += 1
                    detections_by_class[
                        TRACKED_CLASS_NAMES[source_class_id]
                    ] += 1
                    unique_ids.add(track_id)
                record = {
                    "schema_version": "deepsafe.person-detections/v1",
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "image_width": image_width,
                    "image_height": image_height,
                    "timestamp_ns": round(frame_index * 1_000_000_000 / fps),
                    "source_uri": source_uri,
                    "model_id": model_id,
                    "detections": detections,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": sequence_id,
        "decoded_frame_files": expected_frames,
        "exported_frame_records": expected_frames,
        "json_image_dimensions": [image_width, image_height],
        "kitti_coordinate_dimensions": [coordinate_width, coordinate_height],
        "detections_total": detections_total,
        "person_detections": detections_by_class["person"],
        "detections_by_class": detections_by_class,
        "clipped_person_boxes": clipped_total,
        "dropped_degenerate_person_boxes": dropped_total,
        "tracking_backend": TRACKER_NAME,
        "native_track_ids": True,
        "unique_track_ids": sorted(unique_ids),
    }


def _docker_command(
    *,
    app_config: Path,
    run_root: Path,
    gpu: int,
    sequence_id: str,
    profile: int,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        f"colt-person-{sequence_id}-{profile}",
        "--gpus",
        f"device={gpu}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
        "-v",
        f"{ROOT.resolve()}:/workspace:ro",
        "-v",
        f"{run_root.resolve()}:/workspace/{_relative(run_root)}:rw",
        "-w",
        "/workspace",
        CONTAINER_IMAGE,
        "deepstream-app",
        "-c",
        f"/workspace/{_relative(app_config)}",
    ]


def build_plan(
    *,
    video: Path,
    run_root: Path,
    profile: int,
    gpu: int,
    sequence_id: str,
    video_probe: VideoProbe = probe_video,
    export_threshold: float = EXPORT_THRESHOLD,
    include_forklift_candidates: bool = False,
) -> dict[str, Any]:
    """Build one direct, no-guard DeepStream person-inference plan."""

    source = _repository_path(video, label="source clip", must_exist=True)
    output_root = _repository_path(
        run_root, label="person run root", must_exist=False
    )
    if profile not in SUPPORTED_PROFILES:
        raise PersonDeepStreamError("profile must be 640 or 960")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise PersonDeepStreamError("GPU index must be a non-negative integer")
    if not isinstance(sequence_id, str) or not _SAFE_ID.fullmatch(sequence_id):
        raise PersonDeepStreamError("sequence_id is not path/container safe")
    if not 0.0 < float(export_threshold) < 1.0:
        raise PersonDeepStreamError("export threshold must be inside (0, 1)")
    if not isinstance(include_forklift_candidates, bool):
        raise PersonDeepStreamError(
            "include_forklift_candidates must be a boolean"
        )
    included_class_ids = (
        (PERSON_CLASS_ID, FORKLIFT_CANDIDATE_CLASS_ID)
        if include_forklift_candidates
        else (PERSON_CLASS_ID,)
    )

    metadata = dict(video_probe(source))
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    fps = float(metadata.get("fps", 0.0))
    raw_frames = metadata.get("frames", metadata.get("frame_count"))
    frames = int(raw_frames) if raw_frames is not None else 0
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise PersonDeepStreamError(
            "source clip must expose width, height, fps and frame count"
        )
    mux_width, mux_height, mux_scale = calculate_streammux_dimensions(
        width,
        height,
        profile,
        policy="model-active-area",
    )

    generated = output_root / "generated"
    app_config = generated / "deepstream-app.txt"
    infer_config = generated / "config-infer-primary.txt"
    kitti = output_root / "kitti"
    tracker_kitti = output_root / "tracker-kitti"
    predictions = output_root / "predictions.jsonl"
    conversion = output_root / "conversion.json"
    log = output_root / "deepstream.log"
    manifest = output_root / "manifest.json"
    model_root = Path("models/person") / str(profile)
    engine = ROOT / model_root / "yolo11s_b12_gpu0_fp16.engine"
    onnx = ROOT / model_root / "yolo11s.onnx"
    labels = ROOT / model_root / "labels.txt"
    for required, label in (
        (engine, "person TensorRT engine"),
        (onnx, "person ONNX"),
        (labels, "person labels"),
    ):
        if not required.is_file():
            raise PersonDeepStreamError(f"{label} is absent: {required}")

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "execution_mode": "direct_no_guard",
        "sequence_id": sequence_id,
        "profile": profile,
        "gpu_index": gpu,
        "container_image": CONTAINER_IMAGE,
        "source": {
            **_pin(source),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frames,
        },
        "model": {
            "id": MODEL_ID_TEMPLATE.format(profile=profile),
            "input": profile,
            "export_threshold": float(export_threshold),
            "engine": _pin(engine),
            "onnx": _pin(onnx),
            "labels": _pin(labels),
            "included_class_ids": list(included_class_ids),
            "forklift_evidence": (
                {
                    "enabled": True,
                    "semantic_class": "forklift_candidate",
                    "detector_class_id": FORKLIFT_CANDIDATE_CLASS_ID,
                    "detector_class_name": "truck",
                    "precision_claim": "candidate_vehicle_evidence_only",
                }
                if include_forklift_candidates
                else {"enabled": False}
            ),
        },
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
            "width": TRACKER_DIMENSIONS[profile][0],
            "height": TRACKER_DIMENSIONS[profile][1],
            "native_object_id_output": True,
        },
        "paths": {
            "run_root": _relative(output_root),
            "kitti": _relative(kitti),
            "tracker_kitti": _relative(tracker_kitti),
            "predictions": _relative(predictions),
            "conversion": _relative(conversion),
            "deepstream_log": _relative(log),
            "deepstream_config": _relative(app_config),
            "infer_config": _relative(infer_config),
            "manifest": _relative(manifest),
        },
    }
    plan["docker_command"] = _docker_command(
        app_config=app_config,
        run_root=output_root,
        gpu=gpu,
        sequence_id=sequence_id,
        profile=profile,
    )
    plan["contract_sha256"] = hashlib.sha256(_canonical_json(plan)).hexdigest()
    return plan


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_person_detection(detection: object) -> bool:
    if not isinstance(detection, Mapping):
        return False
    track_id = detection.get("track_id")
    confidence = detection.get("confidence")
    tracker_confidence = detection.get("tracker_confidence")
    raw_tracker_confidence = detection.get("tracker_confidence_raw")
    bbox = detection.get("bbox_norm_xywh")
    class_id = detection.get("class_id")
    class_name = detection.get("class_name")
    if (
        not isinstance(track_id, int)
        or isinstance(track_id, bool)
        or track_id < 0
        or track_id == UNTRACKED_OBJECT_ID
        or not _finite_number(confidence)
        or not 0.0 <= float(confidence) <= 1.0
        or not _finite_number(tracker_confidence)
        or not 0.0 <= float(tracker_confidence) <= 1.0
        or not _finite_number(raw_tracker_confidence)
        or float(raw_tracker_confidence) < 0.0
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(_finite_number(value) for value in bbox)
        or class_id not in TRACKED_CLASS_NAMES
        or class_name != TRACKED_CLASS_NAMES[class_id]
    ):
        return False
    left, top, width, height = (float(value) for value in bbox)
    return (
        left >= 0.0
        and top >= 0.0
        and width > 0.0
        and height > 0.0
        and left + width <= 1.000001
        and top + height <= 1.000001
    )


def _predictions_complete(plan: Mapping[str, Any], path: Path) -> bool:
    expected = int(plan["source"]["frame_count"])
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(raw) for raw in stream if raw.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        len(rows) == expected
        and all(
            isinstance(row, Mapping)
            and row.get("schema_version") == "deepsafe.person-detections/v1"
            and isinstance(row.get("frame_index"), int)
            and not isinstance(row.get("frame_index"), bool)
            and row["frame_index"] == index
            and row.get("image_width") == plan["source"]["width"]
            and row.get("image_height") == plan["source"]["height"]
            and row.get("sequence_id") == plan["sequence_id"]
            and row.get("source_uri") == plan["source"]["path"]
            and row.get("model_id") == plan["model"]["id"]
            and isinstance(row.get("timestamp_ns"), int)
            and not isinstance(row.get("timestamp_ns"), bool)
            and row["timestamp_ns"] >= 0
            and isinstance(row.get("detections"), list)
            and all(
                _valid_person_detection(detection)
                for detection in row["detections"]
            )
            for index, row in enumerate(rows)
        )
    )


def _verify_resume_artifacts(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PersonDeepStreamError(
            "existing person manifest has no artifact integrity pins"
        )
    expected_keys = set(_RESUME_ARTIFACT_PATHS)
    if set(artifacts) != expected_keys:
        raise PersonDeepStreamError(
            "existing person manifest artifact pin set differs: "
            f"expected={sorted(expected_keys)} actual={sorted(artifacts)}"
        )
    for artifact_name, plan_path_key in _RESUME_ARTIFACT_PATHS.items():
        pin = artifacts.get(artifact_name)
        if not isinstance(pin, Mapping):
            raise PersonDeepStreamError(
                f"existing person manifest {artifact_name} pin is invalid"
            )
        expected_path = plan["paths"][plan_path_key]
        expected_bytes = pin.get("bytes")
        expected_sha256 = pin.get("sha256")
        if pin.get("path") != expected_path:
            raise PersonDeepStreamError(
                f"existing person manifest {artifact_name} path differs"
            )
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
            or not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise PersonDeepStreamError(
                f"existing person manifest {artifact_name} pin is invalid"
            )
        path = ROOT / expected_path
        if not path.is_file():
            raise PersonDeepStreamError(
                f"existing person resume artifact is absent: {artifact_name}"
            )
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise PersonDeepStreamError(
                "existing person resume artifact integrity differs: "
                f"{artifact_name}"
            )


def execute_plan(
    plan: Mapping[str, Any],
    *,
    command_runner: CommandRunner = _run_logged,
) -> dict[str, Any]:
    """Execute or resume a verified direct person inference plan."""

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise PersonDeepStreamError("person plan schema is unsupported")
    body = dict(plan)
    supplied_hash = body.pop("contract_sha256", None)
    if supplied_hash != hashlib.sha256(_canonical_json(body)).hexdigest():
        raise PersonDeepStreamError("person plan contract hash differs")

    paths = {
        key: ROOT / value
        for key, value in plan["paths"].items()
        if key != "run_root"
    }
    manifest_path = paths["manifest"]
    predictions_path = paths["predictions"]
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PersonDeepStreamError(
                f"cannot read existing person manifest: {exc}"
            ) from exc
        if not isinstance(existing, Mapping):
            raise PersonDeepStreamError(
                "existing person manifest root must be an object"
            )
        if (
            existing.get("status") == "complete"
            and existing.get("plan_contract_sha256") == supplied_hash
        ):
            _verify_resume_artifacts(plan, existing)
            if not _predictions_complete(plan, predictions_path):
                raise PersonDeepStreamError(
                    "existing person prediction stream is structurally invalid"
                )
            return existing
        raise PersonDeepStreamError("existing person manifest conflicts with plan")

    profile = int(plan["profile"])
    infer_text = render_infer_config(
        profile,
        float(plan["model"]["export_threshold"]),
        parser="cuda",
        included_class_ids=tuple(plan["model"]["included_class_ids"]),
    )
    app_text = _render_tracked_app_config(
        video_container_path=f"/workspace/{plan['source']['path']}",
        kitti_container_path=f"/workspace/{plan['paths']['kitti']}",
        tracker_kitti_container_path=(
            f"/workspace/{plan['paths']['tracker_kitti']}"
        ),
        infer_config_container_path=f"/workspace/{plan['paths']['infer_config']}",
        width=int(plan["streammux"]["width"]),
        height=int(plan["streammux"]["height"]),
        profile=profile,
    )
    _atomic_write(paths["infer_config"], infer_text.encode("utf-8"))
    _atomic_write(paths["deepstream_config"], app_text.encode("utf-8"))

    for directory_key in ("kitti", "tracker_kitti"):
        directory = paths[directory_key]
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    for key in ("predictions", "conversion", "deepstream_log"):
        paths[key].unlink(missing_ok=True)
    try:
        command_runner(plan["docker_command"], paths["deepstream_log"])
        engine_attestation = attest_engine_load(
            paths["deepstream_log"], profile
        )
        conversion = _convert_tracked_kitti_directory(
            paths["tracker_kitti"],
            predictions_path,
            sequence_id=str(plan["sequence_id"]),
            image_width=int(plan["source"]["width"]),
            image_height=int(plan["source"]["height"]),
            coordinate_width=int(plan["streammux"]["width"]),
            coordinate_height=int(plan["streammux"]["height"]),
            expected_frames=int(plan["source"]["frame_count"]),
            fps=float(plan["source"]["fps"]),
            source_uri=str(plan["source"]["path"]),
            model_id=str(plan["model"]["id"]),
            included_class_ids=tuple(plan["model"]["included_class_ids"]),
        )
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        raise PersonDeepStreamError(f"direct person inference failed: {exc}") from exc
    if not _predictions_complete(plan, predictions_path):
        raise PersonDeepStreamError("person prediction stream is incomplete")
    _atomic_json(paths["conversion"], conversion)
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "plan_contract_sha256": supplied_hash,
        "execution_mode": "direct_no_guard",
        "sequence_id": plan["sequence_id"],
        "profile": profile,
        "gpu_index": plan["gpu_index"],
        "container_image": plan["container_image"],
        "source": plan["source"],
        "model": plan["model"],
        "streammux": plan["streammux"],
        "tracker": plan["tracker"],
        "conversion": conversion,
        "engine_load_attestation": engine_attestation,
        "artifacts": {
            "predictions": _pin(predictions_path),
            "conversion": _pin(paths["conversion"]),
            "deepstream_log": _pin(paths["deepstream_log"]),
            "deepstream_config": _pin(paths["deepstream_config"]),
            "infer_config": _pin(paths["infer_config"]),
        },
    }
    _atomic_json(manifest_path, terminal)
    return terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", type=int, choices=SUPPORTED_PROFILES, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--export-threshold", type=float, default=EXPORT_THRESHOLD)
    parser.add_argument(
        "--include-forklift-candidates",
        action="store_true",
        help=(
            "also export tracked COCO truck detections as "
            "forklift_candidate evidence"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            video=args.video,
            run_root=args.run_root,
            profile=args.profile,
            gpu=args.gpu,
            sequence_id=args.sequence_id,
            export_threshold=args.export_threshold,
            include_forklift_candidates=args.include_forklift_candidates,
        )
        result = execute_plan(plan) if args.execute else plan
    except PersonDeepStreamError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
