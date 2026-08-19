#!/usr/bin/env python3
"""Persist real-frame ONNX Runtime parity for the PPE DeepStream adapter."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from validation.ppe_ds9_onnx_adapter import expected_ds9_output


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "colt-ai.ppe-ds9-onnx-adapter-parity/v1"
DEFAULT_VIDEO = Path("data/samples/ppe-construction-2025-h264.mp4")
DEFAULT_OUTPUT = Path(
    "models/ppe/safetyvision-yolov8s-v2/ds9-raw6-real-frame-parity.json"
)
DEFAULT_IMAGE = Path(
    "validation/results/ppe/deepstream-diagnostic/"
    "safetyvision-yolov8s-v2/adapter-parity/frame-80.jpg"
)
RAW_ONNX = {
    profile: Path(
        "validation/results/ppe/models/"
        "safetyvision-yolov8s-v2-cpu-export-r3/"
        "safetyvision-v2-cpu-export-r3-001/artifacts/"
        f"safetyvision-yolov8s-v2-{profile}-bdynamic-opset18.onnx"
    )
    for profile in (640, 960)
}
ADAPTED_ONNX = {
    profile: Path(
        f"models/ppe/safetyvision-yolov8s-v2/{profile}/"
        f"safetyvision-yolov8s-v2-{profile}-ds9-raw6.onnx"
    )
    for profile in (640, 960)
}
SELECTED_CLASS_IDS = np.asarray([3, 7, 9, 12], dtype=np.int64)


class PpeAdapterParityError(RuntimeError):
    """Raised when real-frame parity or candidate evidence is absent."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": path.relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_frame(video: Path, timestamp_seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise PpeAdapterParityError(f"cannot open video: {video}")
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise PpeAdapterParityError(
                f"cannot decode frame at {timestamp_seconds}s"
            )
        return frame
    finally:
        capture.release()


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized
    return (
        canvas[:, :, ::-1]
        .transpose(2, 0, 1)[None]
        .astype(np.float32)
        / np.float32(255.0)
    )


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_parity(
    *,
    video: Path,
    image: Path | None,
    timestamp_seconds: float,
    profiles: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    video = (ROOT / video).resolve() if not video.is_absolute() else video.resolve()
    if image is None:
        frame = _read_frame(video, timestamp_seconds)
        image_pin = None
    else:
        image = (
            (ROOT / image).resolve()
            if not image.is_absolute()
            else image.resolve()
        )
        frame = cv2.imread(str(image))
        if frame is None or frame.size == 0:
            raise PpeAdapterParityError(f"cannot decode image: {image}")
        image_pin = _pin(image)
    results = []
    for profile in profiles:
        if profile not in RAW_ONNX:
            raise PpeAdapterParityError(f"unsupported profile: {profile}")
        raw_path = ROOT / RAW_ONNX[profile]
        adapted_path = ROOT / ADAPTED_ONNX[profile]
        tensor = _letterbox(frame, profile)
        raw_session = _session(raw_path)
        adapted_session = _session(adapted_path)
        raw = raw_session.run(
            None, {raw_session.get_inputs()[0].name: tensor}
        )[0]
        adapted = adapted_session.run(
            None, {adapted_session.get_inputs()[0].name: tensor}
        )[0]
        expected = expected_ds9_output(raw)
        maximum_absolute_difference = float(
            np.max(np.abs(adapted - expected))
        )
        if maximum_absolute_difference > 1e-5:
            raise PpeAdapterParityError(
                f"profile {profile} parity drift: "
                f"{maximum_absolute_difference}"
            )
        class_ids = adapted[0, :, 5].astype(np.int64)
        selected = np.isin(class_ids, SELECTED_CLASS_IDS) & (
            adapted[0, :, 4] >= threshold
        )
        selected_rows = adapted[0, selected]
        if selected_rows.size == 0:
            raise PpeAdapterParityError(
                f"profile {profile} has no selected PPE candidates"
            )
        selected_rows = selected_rows[
            np.argsort(-selected_rows[:, 4])
        ][:10]
        results.append(
            {
                "profile": profile,
                "status": "pass",
                "input_tensor": {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": _sha256_bytes(tensor.tobytes()),
                },
                "raw_onnx": _pin(raw_path),
                "adapted_onnx": _pin(adapted_path),
                "raw_output": {
                    "shape": list(raw.shape),
                    "sha256": _sha256_bytes(raw.tobytes()),
                },
                "adapted_output": {
                    "shape": list(adapted.shape),
                    "sha256": _sha256_bytes(adapted.tobytes()),
                },
                "reference_output_sha256": _sha256_bytes(
                    expected.tobytes()
                ),
                "maximum_absolute_difference": (
                    maximum_absolute_difference
                ),
                "selected_candidate_threshold": threshold,
                "selected_candidate_count": int(selected.sum()),
                "top_selected_raw6": [
                    [round(float(value), 6) for value in row]
                    for row in selected_rows
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "observed_at_utc": _utc_now(),
        "purpose": "deepstream_parser_adapter_diagnostic_parity",
        "production_accepted": False,
        "commercially_cleared": False,
        "license_id": "AGPL-3.0",
        "source_video": _pin(video),
        "source_image": image_pin,
        "timestamp_seconds": timestamp_seconds,
        "decoded_frame": {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "sha256": _sha256_bytes(frame.tobytes()),
        },
        "runtime": {
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "profiles": results,
    }


def _atomic_json(path: Path, value: object) -> None:
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
    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and persist real-frame PPE adapter parity"
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--timestamp-seconds", type=float, default=80.0)
    parser.add_argument(
        "--profiles",
        type=int,
        nargs="+",
        choices=sorted(RAW_ONNX),
        default=sorted(RAW_ONNX),
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_parity(
        video=args.video,
        image=args.image,
        timestamp_seconds=args.timestamp_seconds,
        profiles=args.profiles,
        threshold=args.threshold,
    )
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
