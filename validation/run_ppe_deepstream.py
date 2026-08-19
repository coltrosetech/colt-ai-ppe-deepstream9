#!/usr/bin/env python3
"""Run the SafetyVision YOLOv8s v2 PPE challenger through DeepStream 9.

This is a deliberately modular diagnostic/content lane.  It uses the same
DeepStream-Yolo CUDA parser as the accepted person lane, but it does not
promote the SafetyVision checkpoint to a production-accepted model.  Every
plan and terminal manifest records the AGPL-3.0 license and the model's
diagnostic-only acceptance state.

The CLI is plan-only unless ``--execute`` is supplied.  A short H.264 clip is
remuxed from the requested source, then each selected profile is processed in
its own DeepStream container.  TensorRT engines are persisted outside the run
directory so later videos can reuse them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from validation.run_caviar import calculate_streammux_dimensions, probe_video


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "colt-ai.ppe-deepstream-diagnostic/v1"
PLAN_SCHEMA_VERSION = "colt-ai.ppe-deepstream-plan/v1"
PREDICTION_SCHEMA_VERSION = "colt-ai.ppe-detections/v1"
CONTAINER_IMAGE = "deepsafe-deepstream:9.0-control-refresh-20260725"
DEFAULT_RUN_ROOT = Path(
    "validation/results/ppe/deepstream-diagnostic/"
    "safetyvision-yolov8s-v2/ppe-construction-smoke-r3"
)
MODEL_ROOT = Path("models/ppe/safetyvision-yolov8s-v2")
EXPORT_ROOT = Path(
    "validation/results/ppe/models/"
    "safetyvision-yolov8s-v2-cpu-export-r3/"
    "safetyvision-v2-cpu-export-r3-001/artifacts"
)
PROFILE_ONNX = {
    640: EXPORT_ROOT / "safetyvision-yolov8s-v2-640-bdynamic-opset18.onnx",
    960: EXPORT_ROOT / "safetyvision-yolov8s-v2-960-bdynamic-opset18.onnx",
}
PROFILE_DS_ONNX = {
    profile: (
        MODEL_ROOT
        / str(profile)
        / f"safetyvision-yolov8s-v2-{profile}-ds9-raw6.onnx"
    )
    for profile in PROFILE_ONNX
}
PARITY_RECEIPT = (
    MODEL_ROOT / "ds9-raw6-real-frame-parity.json"
)
CLASS_NAMES = (
    "Fall-Detected",
    "Gloves",
    "Goggles",
    "Hardhat",
    "Mask",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "No_Harness",
    "Person",
    "Safety Vest",
)
PPE_CLASS_MAPPING = {
    "Hardhat": {"canonical_class": "helmet", "compliance": "compliant"},
    "NO-Hardhat": {
        "canonical_class": "no_helmet",
        "compliance": "noncompliant",
    },
    "NO-Safety Vest": {
        "canonical_class": "no_hi_vis",
        "compliance": "noncompliant",
    },
    "Person": {
        "canonical_class": "person",
        "compliance": "neutral",
    },
    "Safety Vest": {"canonical_class": "hi_vis", "compliance": "compliant"},
}
# Person is kept in the same DeepStream output so the content/fusion lane can
# draw one canonical person box and attach head/torso PPE state to it.  The
# production architecture still uses the dedicated person GIE + nvtracker as
# the canonical track source; this diagnostic model's Person class is an
# immediately runnable bridge.
SELECTED_CLASS_IDS = frozenset({3, 7, 9, 11, 12})
KITTI_NAME = re.compile(
    r"^(?P<app_index>\d{2})_(?P<source_id>\d{3})_"
    r"(?P<frame_index>\d{6,})\.txt$"
)
FPS_PATTERN = re.compile(
    r"\*\*PERF:\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"\(([0-9]+(?:\.[0-9]+)?)\)"
)


class PpeDeepStreamError(RuntimeError):
    """Raised when the diagnostic lane cannot produce verified output."""


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


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    absolute = path.resolve(strict=True)
    if ROOT.resolve() != absolute and ROOT.resolve() not in absolute.parents:
        raise PpeDeepStreamError(f"artifact escapes repository: {absolute}")
    size = absolute.stat().st_size
    if size <= 0:
        raise PpeDeepStreamError(f"artifact is empty: {absolute}")
    return {
        "path": absolute.relative_to(ROOT.resolve()).as_posix(),
        "bytes": size,
        "sha256": _sha256_file(absolute),
    }


def _repository_path(value: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PpeDeepStreamError(
            f"path escapes repository root: {candidate}"
        ) from exc
    if must_exist and not candidate.is_file():
        raise PpeDeepStreamError(f"required file is absent: {candidate}")
    return candidate


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PpeDeepStreamError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PpeDeepStreamError(f"JSON artifact must be an object: {path}")
    return value


def _validate_adapter_receipt(
    receipt_path: Path,
    *,
    source_onnx: Path,
    adapted_onnx: Path,
) -> None:
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version")
        != "colt-ai.ppe-ds9-onnx-adapter/v1"
        or receipt.get("status") != "adapted_static_verified"
        or receipt.get("production_accepted") is not False
        or receipt.get("license_id") != "AGPL-3.0"
        or receipt.get("source") != _pin(source_onnx)
        or receipt.get("artifact") != _pin(adapted_onnx)
        or receipt.get("transform", {}).get("nms_embedded") is not False
    ):
        raise PpeDeepStreamError(
            f"adapter receipt does not match live artifacts: {receipt_path}"
        )


def _validate_parity_receipt(path: Path, profiles: Sequence[int]) -> None:
    receipt = _load_json(path)
    rows = receipt.get("profiles")
    if (
        receipt.get("schema_version")
        != "colt-ai.ppe-ds9-onnx-adapter-parity/v1"
        or receipt.get("status") != "pass"
        or receipt.get("production_accepted") is not False
        or receipt.get("commercially_cleared") is not False
        or receipt.get("license_id") != "AGPL-3.0"
        or not isinstance(rows, list)
    ):
        raise PpeDeepStreamError("PPE adapter parity receipt is invalid")
    by_profile = {
        row.get("profile"): row for row in rows if isinstance(row, dict)
    }
    for profile in profiles:
        row = by_profile.get(profile)
        if (
            not isinstance(row, dict)
            or row.get("status") != "pass"
            or row.get("maximum_absolute_difference") != 0.0
            or not isinstance(row.get("selected_candidate_count"), int)
            or row["selected_candidate_count"] <= 0
        ):
            raise PpeDeepStreamError(
                f"profile {profile} lacks exact real-frame adapter parity"
            )


def engine_path(profile: int, gpu: int) -> Path:
    if profile not in PROFILE_ONNX:
        raise PpeDeepStreamError(f"unsupported PPE profile: {profile}")
    if gpu < 0:
        raise PpeDeepStreamError("GPU index must be non-negative")
    return (
        ROOT
        / MODEL_ROOT
        / str(profile)
        / f"safetyvision_yolov8s_v2_ds9raw6_b12_gpu{gpu}_fp16.engine"
    )


def render_infer_config(
    profile: int,
    *,
    gpu: int = 0,
    threshold: float = 0.10,
) -> str:
    """Render a CUDA-parser config for person, helmet and hi-vis classes."""

    if profile not in PROFILE_ONNX:
        raise PpeDeepStreamError(f"unsupported PPE profile: {profile}")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise PpeDeepStreamError("threshold must be finite and inside (0, 1)")
    filtered = ";".join(
        str(class_id)
        for class_id in range(len(CLASS_NAMES))
        if class_id not in SELECTED_CLASS_IDS
    )
    engine = "/" + (
        Path("workspace")
        / MODEL_ROOT
        / str(profile)
        / engine_path(profile, gpu).name
    ).as_posix()
    labels = "/" + (Path("workspace") / MODEL_ROOT / "labels.txt").as_posix()
    return f"""[property]
gpu-id={gpu}
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file={engine}
labelfile-path={labels}
batch-size=1
network-mode=2
num-detected-classes=13
filter-out-class-ids={filtered}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYoloCuda
custom-lib-path=/opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold={threshold:.8g}
topk=300
"""


def render_deepstream_config(
    *,
    video: Path,
    kitti_dir: Path,
    infer_config: Path,
    width: int,
    height: int,
    gpu: int = 0,
) -> str:
    if width <= 0 or height <= 0:
        raise PpeDeepStreamError("streammux dimensions must be positive")
    video_path = "/workspace/" + _relative(video)
    kitti_path = "/workspace/" + _relative(kitti_dir)
    infer_path = "/workspace/" + _relative(infer_config)
    return f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
gie-kitti-output-dir={kitti_path}

[tiled-display]
enable=0

[source0]
enable=1
type=2
uri=file://{video_path}
gpu-id={gpu}
cudadec-memtype=0

[sink0]
enable=1
type=1
sync=0
qos=0

[osd]
enable=0

[streammux]
gpu-id={gpu}
live-source=0
batch-size=1
batched-push-timeout=40000
width={width}
height={height}
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id={gpu}
batch-size=1
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file={infer_path}

[tests]
file-loop=0
"""


def build_docker_command(
    *,
    app_config: Path,
    run_root: Path,
    profile: int,
    gpu: int,
    container_name: str,
    image: str = CONTAINER_IMAGE,
) -> list[str]:
    model_root = (ROOT / MODEL_ROOT).resolve()
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
        f"{ROOT.resolve()}:/workspace:ro",
        "-v",
        f"{run_root.resolve()}:/workspace/{_relative(run_root)}:rw",
        "-v",
        f"{model_root}:/workspace/{MODEL_ROOT.as_posix()}:rw",
        "-w",
        f"/workspace/{MODEL_ROOT.as_posix()}/{profile}",
        image,
        "deepstream-app",
        "-c",
        f"/workspace/{_relative(app_config)}",
    ]


def build_engine_command(
    *,
    profile: int,
    gpu: int,
    image: str = CONTAINER_IMAGE,
) -> list[str]:
    """Build a reusable dynamic B1..B12 FP16 engine in the DS9 image."""

    onnx = "/workspace/" + PROFILE_DS_ONNX[profile].as_posix()
    engine = "/workspace/" + _relative(engine_path(profile, gpu))
    dimensions = f"3x{profile}x{profile}"
    model_root = (ROOT / MODEL_ROOT).resolve()
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        f"colt-ppe-engine-{profile}-gpu{gpu}",
        "--gpus",
        f"device={gpu}",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-v",
        f"{ROOT.resolve()}:/workspace:ro",
        "-v",
        f"{model_root}:/workspace/{MODEL_ROOT.as_posix()}:rw",
        "-w",
        f"/workspace/{MODEL_ROOT.as_posix()}/{profile}",
        image,
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        "--noTF32",
        f"--minShapes=images:1x{dimensions}",
        f"--optShapes=images:12x{dimensions}",
        f"--maxShapes=images:12x{dimensions}",
        "--memPoolSize=workspace:4096",
        "--builderOptimizationLevel=3",
        "--skipInference",
    ]


def parse_fps(log_path: Path) -> dict[str, Any]:
    samples = [
        {"current": float(match.group(1)), "average": float(match.group(2))}
        for match in FPS_PATTERN.finditer(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    ]
    averages = [item["average"] for item in samples]
    return {
        "sample_count": len(samples),
        "samples": samples,
        "last_reported_average": averages[-1] if averages else None,
        "mean_reported_average": (
            round(sum(averages) / len(averages), 3) if averages else None
        ),
        "peak_current": (
            max(item["current"] for item in samples) if samples else None
        ),
    }


def attest_engine_load(log_path: Path, model_engine: Path) -> dict[str, Any]:
    """Prove that nvinfer deserialized the prepared engine without fallback."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    expected = "/workspace/" + _relative(model_engine)
    deserialized = re.findall(
        r"deserialized trt engine from\s*:(\S+)", text, flags=re.IGNORECASE
    )
    selected = re.findall(
        r"Use deserialized engine model:\s*(\S+)",
        text,
        flags=re.IGNORECASE,
    )
    rebuild_markers = len(
        re.findall(
            r"Trying to create engine from model files|Building the TensorRT Engine",
            text,
            flags=re.IGNORECASE,
        )
    )
    observed = deserialized + selected
    if expected not in observed:
        raise PpeDeepStreamError(
            f"DeepStream log does not attest engine load: {expected}"
        )
    if rebuild_markers:
        raise PpeDeepStreamError(
            "DeepStream attempted an engine fallback during smoke inference"
        )
    return {
        "status": "pass",
        "expected_engine": expected,
        "deserialize_records": deserialized,
        "selected_records": selected,
        "fallback_markers": rebuild_markers,
    }


def _parse_kitti_row(
    raw: str, source: Path, line_number: int
) -> tuple[str, list[float]]:
    fields = raw.split()
    if len(fields) < 16:
        raise PpeDeepStreamError(
            f"{source}:{line_number}: malformed KITTI record"
        )
    label = " ".join(fields[:-15])
    try:
        values = [float(value) for value in fields[-15:]]
    except ValueError as exc:
        raise PpeDeepStreamError(
            f"{source}:{line_number}: non-numeric KITTI value"
        ) from exc
    if not label or not all(math.isfinite(value) for value in values):
        raise PpeDeepStreamError(
            f"{source}:{line_number}: invalid KITTI record"
        )
    return label, values


def _kitti_frames(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.glob("*.txt"):
        match = KITTI_NAME.fullmatch(path.name)
        if match is None:
            continue
        if int(match["app_index"]) != 0 or int(match["source_id"]) != 0:
            continue
        index = int(match["frame_index"])
        if index in result:
            raise PpeDeepStreamError(f"duplicate KITTI frame {index}")
        result[index] = path
    return result


def convert_kitti(
    directory: Path,
    output: Path,
    *,
    sequence_id: str,
    source_width: int,
    source_height: int,
    coordinate_width: int,
    coordinate_height: int,
    expected_frames: int,
    threshold: float,
) -> dict[str, Any]:
    """Convert a complete PPE KITTI dump into canonical frame JSONL."""

    frames = _kitti_frames(directory)
    expected = set(range(expected_frames))
    if set(frames) != expected:
        missing = sorted(expected - set(frames))
        extra = sorted(set(frames) - expected)
        raise PpeDeepStreamError(
            "KITTI frame sequence mismatch: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    class_ids = {name.casefold(): index for index, name in enumerate(CLASS_NAMES)}
    counts = {mapping["canonical_class"]: 0 for mapping in PPE_CLASS_MAPPING.values()}
    detection_count = 0
    dropped_below_threshold = 0
    clipped_boxes = 0
    rows: list[bytes] = []
    scale_x = source_width / coordinate_width
    scale_y = source_height / coordinate_height
    for frame_index in range(expected_frames):
        detections: list[dict[str, Any]] = []
        for line_number, raw in enumerate(
            frames[frame_index].read_text(
                encoding="utf-8", errors="strict"
            ).splitlines(),
            1,
        ):
            if not raw.strip():
                continue
            label, values = _parse_kitti_row(
                raw, frames[frame_index], line_number
            )
            mapping = PPE_CLASS_MAPPING.get(label)
            if mapping is None:
                raise PpeDeepStreamError(
                    f"{frames[frame_index]}:{line_number}: "
                    f"unexpected unfiltered class {label!r}"
                )
            confidence = values[-1]
            if not 0 <= confidence <= 1:
                raise PpeDeepStreamError(
                    f"{frames[frame_index]}:{line_number}: invalid confidence"
                )
            if confidence < threshold:
                dropped_below_threshold += 1
                continue
            original = tuple(values[3:7])
            left = max(0.0, min(float(coordinate_width), original[0]))
            top = max(0.0, min(float(coordinate_height), original[1]))
            right = max(0.0, min(float(coordinate_width), original[2]))
            bottom = max(0.0, min(float(coordinate_height), original[3]))
            if (left, top, right, bottom) != original:
                clipped_boxes += 1
            if right <= left or bottom <= top:
                continue
            x = left * scale_x
            y = top * scale_y
            width = (right - left) * scale_x
            height = (bottom - top) * scale_y
            detection = {
                "class_id": class_ids[label.casefold()],
                "class_name": label,
                "canonical_class": mapping["canonical_class"],
                "compliance": mapping["compliance"],
                "confidence": confidence,
                "bbox_xywh": [
                    round(x, 4),
                    round(y, 4),
                    round(width, 4),
                    round(height, 4),
                ],
                "bbox_norm_xywh": [
                    round(x / source_width, 8),
                    round(y / source_height, 8),
                    round(width / source_width, 8),
                    round(height / source_height, 8),
                ],
            }
            detections.append(detection)
            detection_count += 1
            counts[mapping["canonical_class"]] += 1
        rows.append(
            _canonical_json(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "image_width": source_width,
                    "image_height": source_height,
                    "model_id": "safetyvision-yolov8s-v2-diagnostic",
                    "production_accepted": False,
                    "detections": detections,
                }
            )
        )
    _atomic_write(output, b"".join(rows))
    ppe_detection_count = sum(
        counts[class_name]
        for class_name in ("helmet", "no_helmet", "hi_vis", "no_hi_vis")
    )
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "decoded_frame_files": expected_frames,
        "exported_frame_records": expected_frames,
        "detections": detection_count,
        "ppe_detections": ppe_detection_count,
        "person_detections": counts["person"],
        "class_counts": counts,
        "dropped_below_threshold": dropped_below_threshold,
        "clipped_boxes": clipped_boxes,
        "source_dimensions": [source_width, source_height],
        "kitti_coordinate_dimensions": [coordinate_width, coordinate_height],
    }


def _require_selected_ppe(
    conversion: Mapping[str, Any],
    *,
    profile: int,
) -> None:
    count = conversion.get("ppe_detections")
    if count is None and isinstance(conversion.get("class_counts"), Mapping):
        class_counts = conversion["class_counts"]
        values = [
            class_counts.get(class_name)
            for class_name in ("helmet", "no_helmet", "hi_vis", "no_hi_vis")
        ]
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in values
        ):
            count = sum(values)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
    ):
        raise PpeDeepStreamError(
            f"profile {profile} produced no selected PPE detections"
        )


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _remove_known_outputs(run_root: Path) -> None:
    for path in (
        run_root / "deepstream.log",
        run_root / "predictions.jsonl",
        run_root / "conversion.json",
    ):
        path.unlink(missing_ok=True)
    kitti = run_root / "kitti"
    if kitti.is_dir():
        for path in kitti.glob("*.txt"):
            if KITTI_NAME.fullmatch(path.name):
                path.unlink()


def _remux_smoke_clip(
    source: Path,
    destination: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0:v:0",
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
        "-an",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    os.replace(temporary, destination)


def _gpu_snapshot(gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(fields) != 4:
        raise PpeDeepStreamError("unexpected nvidia-smi output")
    return {
        "index": gpu,
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": int(fields[3]),
    }


def build_plan(
    *,
    video: Path,
    run_root: Path,
    profiles: Iterable[int],
    gpu: int,
    start_seconds: float,
    duration_seconds: float,
    threshold: float,
) -> dict[str, Any]:
    source = _repository_path(video)
    run_root = _repository_path(run_root, must_exist=False)
    requested = tuple(dict.fromkeys(int(profile) for profile in profiles))
    if not requested or any(profile not in PROFILE_ONNX for profile in requested):
        raise PpeDeepStreamError("profiles must be one or more of 640, 960")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise PpeDeepStreamError("start time must be non-negative and finite")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise PpeDeepStreamError("duration must be positive and finite")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise PpeDeepStreamError("threshold must be inside (0, 1)")
    labels = ROOT / MODEL_ROOT / "labels.txt"
    if labels.read_text(encoding="utf-8").splitlines() != list(CLASS_NAMES):
        raise PpeDeepStreamError("PPE label order differs from model contract")
    source_metadata = probe_video(source)
    source_frames = source_metadata.get("frames")
    if source_frames is not None:
        source_duration = int(source_frames) / float(source_metadata["fps"])
        if start_seconds + duration_seconds > source_duration + 1e-6:
            raise PpeDeepStreamError(
                "requested smoke interval exceeds source duration"
            )
    models = []
    for profile in requested:
        source_onnx = ROOT / PROFILE_ONNX[profile]
        onnx = ROOT / PROFILE_DS_ONNX[profile]
        adapter_receipt = onnx.parent / "ds9-raw6-receipt.json"
        if not source_onnx.is_file():
            raise PpeDeepStreamError(
                f"missing source ONNX profile: {source_onnx}"
            )
        if not onnx.is_file() or not adapter_receipt.is_file():
            raise PpeDeepStreamError(
                f"missing DeepStream raw6 adapter artifacts: {onnx}"
            )
        _validate_adapter_receipt(
            adapter_receipt,
            source_onnx=source_onnx,
            adapted_onnx=onnx,
        )
        models.append(
            {
                "profile": profile,
                "source_onnx": _pin(source_onnx),
                "onnx": _pin(onnx),
                "adapter_receipt": _pin(adapter_receipt),
                "engine_path": _relative(engine_path(profile, gpu)),
            }
        )
    parity_receipt = ROOT / PARITY_RECEIPT
    if not parity_receipt.is_file():
        raise PpeDeepStreamError(
            f"missing real-frame adapter parity: {parity_receipt}"
        )
    _validate_parity_receipt(parity_receipt, requested)
    definition: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "purpose": "diagnostic_content_evaluation",
        "production_accepted": False,
        "commercially_cleared": False,
        "model": {
            "id": "safetyvision-yolov8s-v2",
            "family": "YOLOv8s",
            "accepted_model": False,
            "source_repository": "ayushgupta7777/safetyvision-yolov8",
            "source_commit": "56a71758b55f0e9f2b4b2d6b51a779a1f882da10",
            "license_id": "AGPL-3.0",
            "class_mapping": {
                "helmet": 3,
                "no_helmet": 7,
                "no_hi_vis": 9,
                "person": 11,
                "hi_vis": 12,
            },
            "known_limitations": [
                "NO-Safety Vest is the weakest documented target",
                "overhead-camera qualification is not complete",
                "small distant workers may be missed",
            ],
        },
        "runtime": {
            "container_image": CONTAINER_IMAGE,
            "deepstream_major": 9,
            "parser": "NvDsInferParseYoloCuda",
            "precision": "fp16",
            "gpu_index": gpu,
        },
        "source": _pin(source),
        "start_seconds_requested": start_seconds,
        "duration_seconds_requested": duration_seconds,
        "display_threshold": threshold,
        "labels": _pin(labels),
        "adapter_parity": _pin(parity_receipt),
        "profiles": models,
        "run_root": _relative(run_root),
    }
    definition["contract_sha256"] = hashlib.sha256(
        _canonical_json(definition)
    ).hexdigest()
    return definition


def execute_plan(
    plan: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    run_root = ROOT / plan["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise PpeDeepStreamError(
                "run root already contains a different immutable plan"
            )
    else:
        _atomic_json(plan_path, plan, mode=0o440)
    terminal_path = run_root / "manifest.json"
    if terminal_path.exists() and not force:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("status") == "complete":
            profiles = terminal.get("profiles")
            if not isinstance(profiles, list) or not profiles:
                raise PpeDeepStreamError(
                    "complete manifest has no profile evidence"
                )
            for profile_item in profiles:
                if not isinstance(profile_item, Mapping):
                    raise PpeDeepStreamError(
                        "complete manifest profile evidence is invalid"
                    )
                conversion = profile_item.get("conversion")
                statistics = (
                    conversion.get("statistics")
                    if isinstance(conversion, Mapping)
                    else None
                )
                if not isinstance(statistics, Mapping):
                    raise PpeDeepStreamError(
                        "complete manifest lacks conversion statistics"
                    )
                _require_selected_ppe(
                    statistics,
                    profile=int(profile_item.get("profile", -1)),
                )
            return terminal
        raise PpeDeepStreamError(
            "incomplete manifest exists; use --force to retry known outputs"
        )
    source = ROOT / plan["source"]["path"]
    clip = run_root / "source-smoke.mp4"
    _remux_smoke_clip(
        source,
        clip,
        start_seconds=float(plan["start_seconds_requested"]),
        duration_seconds=float(plan["duration_seconds_requested"]),
    )
    metadata = probe_video(clip)
    frames_value = metadata.get("frames")
    if frames_value is None:
        raise PpeDeepStreamError("smoke clip has no exact frame count")
    gpu = int(plan["runtime"]["gpu_index"])
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "purpose": plan["purpose"],
        "production_accepted": False,
        "commercially_cleared": False,
        "license_id": "AGPL-3.0",
        "model_acceptance": {
            "accepted_model": False,
            "production_ready": False,
            "effect": "diagnostic_evidence_only",
        },
        "plan": _pin(plan_path),
        "source_clip": {
            **_pin(clip),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "fps": float(metadata["fps"]),
            "frames": int(frames_value),
            "duration_seconds": round(
                int(frames_value) / float(metadata["fps"]), 6
            ),
        },
        "gpu": _gpu_snapshot(gpu),
        "profiles": [],
    }
    _atomic_json(terminal_path, manifest)
    try:
        for profile_item in plan["profiles"]:
            profile = int(profile_item["profile"])
            profile_root = run_root / str(profile)
            generated = profile_root / "generated"
            kitti = profile_root / "kitti"
            generated.mkdir(parents=True, exist_ok=True)
            kitti.mkdir(parents=True, exist_ok=True)
            if force:
                _remove_known_outputs(profile_root)
            elif any(kitti.iterdir()):
                raise PpeDeepStreamError(
                    f"profile {profile} already has KITTI output; use --force"
                )
            model_engine = engine_path(profile, gpu)
            model_engine.parent.mkdir(parents=True, exist_ok=True)
            infer_config = generated / "config-infer-primary.txt"
            app_config = generated / "deepstream-app.txt"
            mux_width, mux_height, mux_scale = calculate_streammux_dimensions(
                int(metadata["width"]),
                int(metadata["height"]),
                profile,
                policy="model-active-area",
            )
            _atomic_write(
                infer_config,
                render_infer_config(
                    profile,
                    gpu=gpu,
                    threshold=float(plan["display_threshold"]),
                ).encode("utf-8"),
            )
            _atomic_write(
                app_config,
                render_deepstream_config(
                    video=clip,
                    kitti_dir=kitti,
                    infer_config=infer_config,
                    width=mux_width,
                    height=mux_height,
                    gpu=gpu,
                ).encode("utf-8"),
            )
            engine_existed = model_engine.is_file()
            engine_build_log = profile_root / "engine-build.log"
            engine_build_wall_seconds = 0.0
            if not engine_existed:
                engine_started = time.monotonic()
                _run_logged(
                    build_engine_command(profile=profile, gpu=gpu),
                    engine_build_log,
                )
                engine_build_wall_seconds = round(
                    time.monotonic() - engine_started, 6
                )
                if (
                    not model_engine.is_file()
                    or model_engine.stat().st_size <= 0
                ):
                    raise PpeDeepStreamError(
                        f"profile {profile} did not persist a TensorRT engine"
                    )
            command = build_docker_command(
                app_config=app_config,
                run_root=run_root,
                profile=profile,
                gpu=gpu,
                container_name=f"colt-ppe-smoke-{profile}-gpu{gpu}",
            )
            log = profile_root / "deepstream.log"
            started = time.monotonic()
            _run_logged(command, log)
            wall_seconds = round(time.monotonic() - started, 6)
            if not model_engine.is_file() or model_engine.stat().st_size <= 0:
                raise PpeDeepStreamError(
                    f"profile {profile} did not persist a TensorRT engine"
                )
            engine_attestation = attest_engine_load(log, model_engine)
            predictions = profile_root / "predictions.jsonl"
            conversion = convert_kitti(
                kitti,
                predictions,
                sequence_id=f"ppe-construction-smoke-{profile}",
                source_width=int(metadata["width"]),
                source_height=int(metadata["height"]),
                coordinate_width=mux_width,
                coordinate_height=mux_height,
                expected_frames=int(frames_value),
                threshold=float(plan["display_threshold"]),
            )
            _require_selected_ppe(conversion, profile=profile)
            conversion_path = profile_root / "conversion.json"
            _atomic_json(conversion_path, conversion)
            profile_manifest = {
                "profile": profile,
                "status": "complete",
                "streammux": {
                    "width": mux_width,
                    "height": mux_height,
                    "source_to_mux_scale": mux_scale,
                    "policy": "model-active-area",
                },
                "engine_built_this_run": not engine_existed,
                "engine_build_wall_seconds": engine_build_wall_seconds,
                "engine_load_attestation": engine_attestation,
                "wall_seconds": wall_seconds,
                "deepstream_fps": parse_fps(log),
                "onnx": profile_item["onnx"],
                "engine": _pin(model_engine),
                "engine_build_log": (
                    _pin(engine_build_log)
                    if engine_build_log.is_file()
                    else None
                ),
                "deepstream_log": _pin(log),
                "deepstream_config": _pin(app_config),
                "infer_config": _pin(infer_config),
                "predictions": _pin(predictions),
                "conversion": {
                    **_pin(conversion_path),
                    "statistics": conversion,
                },
            }
            manifest["profiles"].append(profile_manifest)
            _atomic_json(terminal_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(terminal_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _utc_now()
    _atomic_json(terminal_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the diagnostic DeepStream 9 PPE lane"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="PPE smoke video path inside the repository",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--profiles",
        type=int,
        nargs="+",
        choices=sorted(PROFILE_ONNX),
        default=sorted(PROFILE_ONNX),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start-seconds", type=float, default=80.0)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            video=args.video,
            run_root=args.run_root,
            profiles=args.profiles,
            gpu=args.gpu,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            threshold=args.threshold,
        )
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        manifest = execute_plan(plan, force=args.force)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        PpeDeepStreamError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
