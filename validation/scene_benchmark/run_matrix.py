#!/usr/bin/env python3
"""Run a resumable 12-stream DeepStream person-throughput scene matrix.

This tool deliberately measures throughput only. Looped frames are never sent to
the person-accuracy evaluator; unique-frame accuracy runs live in the separate
``evaluation``/``validation.run_caviar`` path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve()
SUMMARIZER_PATH = PROJECT_ROOT / "benchmark/summarize.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.summarize import PERF_PAIR, parse_gpu_metrics, parse_perf  # noqa: E402
from validation.ds9_runtime_compatibility import (  # noqa: E402
    DEFAULT_RECEIPT as DEFAULT_DS9_COMPATIBILITY_RECEIPT,
    prevalidate_runtime_compatibility,
    require_runtime_compatibility,
    write_pending_report as write_ds9_pending_report,
)


SCHEMA_VERSION = "deepsafe.scene-benchmark-run/v2"
RUNTIME_CONTAINER_ATTESTATION_SCHEMA = (
    "deepsafe.scene-runtime-container-attestation/v1"
)
DEFAULT_MANIFEST = Path(__file__).with_name("scenes.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "validation/results/scene-benchmark"
DEFAULT_SMOKE_OUTPUT = PROJECT_ROOT / "validation/results/scene-benchmark-smoke"
DEFAULT_REENTRY_EVIDENCE = (
    PROJECT_ROOT / "validation/results/gpu-reentry/current/evidence.json"
)
EXPECTED_PERSON_FILTER = list(range(1, 80))
EXPECTED_EXIT_CODES = {0, 130, 137, 143}
MAX_CONSECUTIVE_GPU_QUERY_ERRORS = 3
GPU_IDENTITY_CHECK_INTERVAL_SAMPLES = 30
DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W = 5.0
DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES = 2
DEFAULT_GPU_OPERATING_POLICY_MODE = "workstation_managed"
LEGACY_STRICT_GPU_OPERATING_POLICY_MODE = "legacy_strict"
GPU_OPERATING_POLICY_MODES = (
    DEFAULT_GPU_OPERATING_POLICY_MODE,
    LEGACY_STRICT_GPU_OPERATING_POLICY_MODE,
)
PREFLIGHT_GPU_SAMPLES = 2
PREFLIGHT_GPU_SAMPLE_INTERVAL_SECONDS = 1.0
LEGACY_GPU_CSV_HEADER = [
    "timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_draw_w",
    "sm_clock_mhz",
    "memory_clock_mhz",
]
GPU_CSV_HEADER = [
    *LEGACY_GPU_CSV_HEADER,
    "power_requested_limit_w",
    "power_current_limit_w",
    "power_default_limit_w",
    "pstate",
    "clock_event_reasons_active_mask",
    "clock_event_sw_power_cap",
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
]
GPU_QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "power.limit",
    "enforced.power.limit",
    "power.default_limit",
    "pstate",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
]
FATAL_LOG_PATTERNS = {
    "gstreamer_error": re.compile(r"ERROR from element", re.IGNORECASE),
    "pipeline_failed": re.compile(r"Failed to (?:set pipeline|create pipeline)", re.IGNORECASE),
    "cuda_error": re.compile(r"CUDA error", re.IGNORECASE),
    "engine_deserialize_error": re.compile(
        r"(?:deserialize.*(?:failed|error)|(?:failed|error).*deserialize)", re.IGNORECASE
    ),
    "out_of_memory": re.compile(
        r"(?:out of memory|cudaErrorMemoryAllocation)", re.IGNORECASE
    ),
}
REQUIRED_SLOWDOWN_TELEMETRY_FIELDS = (
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
)
REQUIRED_GPU_IDENTITY_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "memory.total",
    "pci.bus_id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def gpu_lock(gpu_index: int):
    """Share the same advisory GPU lock used by the CAVIAR campaign."""
    lock_path = Path(f"/tmp/deepsafe-caviar-gpu{gpu_index}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"GPU {gpu_index} is locked by another DeepSafe campaign: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"campaign=scene-benchmark pid={os.getpid()} started={utc_now()}\n"
        )
        handle.flush()
        yield {"path": str(lock_path), "pid": os.getpid()}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as exc:
        raise ValueError(f"Path must stay inside project root: {path}") from exc


def container_path(path: Path) -> str:
    return f"/workspace/{project_relative(path)}"


@functools.lru_cache(maxsize=64)
def sha256_file(path: Path, size_bytes: int, mtime_ns: int) -> str:
    """Hash a stable file revision; stat values make the in-process cache safe."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    current = path.stat()
    if current.st_size != size_bytes or current.st_mtime_ns != mtime_ns:
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def command(
    args: list[str], *, timeout: float = 30, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=check,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "deepsafe.scene-benchmark/v1":
        raise ValueError(f"Unsupported scene manifest schema: {payload.get('schema_version')}")
    campaign = payload.get("campaign", {})
    if campaign.get("streams") != 12:
        raise ValueError("Scene benchmark manifest must simulate exactly 12 streams")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < 10:
        raise ValueError("Scene benchmark requires at least 10 scenes")
    ids = [scene.get("id") for scene in scenes]
    types = [scene.get("benchmark_type") for scene in scenes]
    if len(ids) != len(set(ids)) or not all(isinstance(item, str) and item for item in ids):
        raise ValueError("Scene ids must be non-empty and unique")
    if len(types) != len(set(types)) or not all(
        isinstance(item, str) and item for item in types
    ):
        raise ValueError("Every scene must define a unique benchmark_type")

    source_catalog_path = resolve_project_path(payload["source_catalog"])
    source_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
    source_assets = {asset["id"]: asset for asset in source_catalog.get("assets", [])}
    for scene in scenes:
        video = resolve_project_path(scene["video_path"])
        if not video.is_file():
            raise FileNotFoundError(f"Scene video missing: {video}")
        if video.suffix.lower() != ".mp4":
            raise ValueError(f"Performance scene must be normalized H.264 MP4: {video}")
        source_id = scene.get("source_manifest_id")
        if source_id not in source_assets:
            raise ValueError(f"Scene {scene['id']} has no source-catalog record: {source_id}")
        license_data = source_assets[source_id].get("license", {})
        if not license_data.get("spdx") or not license_data.get("attribution"):
            raise ValueError(f"Scene {scene['id']} source record lacks license/attribution")
        scene["source_metadata"] = {
            "title": source_assets[source_id].get("title"),
            "asset_page_url": source_assets[source_id].get("asset_page_url"),
            "license": license_data,
            "camera": source_assets[source_id].get("camera"),
            "ground_truth": source_assets[source_id].get("ground_truth"),
        }
    payload["manifest_path"] = project_relative(path)
    payload["source_catalog_path"] = project_relative(source_catalog_path)
    return payload


def parse_fraction(value: str) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    result = command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
    )
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise ValueError(f"No video stream found: {path}")
    stream = payload["streams"][0]
    if stream.get("codec_name") != "h264":
        raise ValueError(f"Expected H.264, got {stream.get('codec_name')}: {path}")
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
    stat = path.stat()
    return {
        "path": project_relative(path),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_fraction": fps_text,
        "fps": round(parse_fraction(fps_text) or 0.0, 6),
        "duration_seconds": round(float(payload["format"]["duration"]), 6),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path, stat.st_size, stat.st_mtime_ns),
    }


def infer_config_path(size: int) -> Path:
    return PROJECT_ROOT / f"models/person/{size}/config_infer_primary.txt"


def validate_person_profile(size: int) -> dict[str, Any]:
    if size not in (640, 960):
        raise ValueError(f"Only person model inputs 640 and 960 are supported: {size}")
    infer = infer_config_path(size)
    text = infer.read_text(encoding="utf-8")
    filtered_line = next(
        (line for line in text.splitlines() if line.startswith("filter-out-class-ids=")),
        None,
    )
    if filtered_line is None:
        raise ValueError(f"Person-only filter missing: {infer}")
    filtered = [int(item) for item in filtered_line.partition("=")[2].split(";")]
    if filtered != EXPECTED_PERSON_FILTER:
        raise ValueError(f"Person-only filter is not exactly COCO classes 1..79: {infer}")
    engine_line = next(
        (line for line in text.splitlines() if line.startswith("model-engine-file=")),
        None,
    )
    if engine_line is None:
        raise ValueError(f"model-engine-file missing: {infer}")
    engine_container_path = engine_line.partition("=")[2]
    if not engine_container_path.startswith("/models/"):
        raise ValueError(f"Engine path must be rooted at /models: {engine_container_path}")
    engine = PROJECT_ROOT / "models" / engine_container_path.removeprefix("/models/")
    if not engine.is_file() or engine.stat().st_size == 0:
        raise FileNotFoundError(f"Prebuilt TensorRT engine missing: {engine}")
    infer_stat = infer.stat()
    engine_stat = engine.stat()
    return {
        "size": size,
        "infer_config": project_relative(infer),
        "engine": project_relative(engine),
        "infer_config_sha256": sha256_file(
            infer, infer_stat.st_size, infer_stat.st_mtime_ns
        ),
        "engine_size_bytes": engine_stat.st_size,
        "engine_sha256": sha256_file(
            engine, engine_stat.st_size, engine_stat.st_mtime_ns
        ),
        "person_only_classes": [0],
    }


def active_area_dimensions(source_width: int, source_height: int, size: int) -> tuple[int, int]:
    """Fit the decoded frame into the square model's active area without distortion."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    if size not in (640, 960):
        raise ValueError(f"Only person model inputs 640 and 960 are supported: {size}")
    scale = size / max(source_width, source_height)

    def even_dimension(value: float) -> int:
        return max(2, min(size, int(round(value / 2.0)) * 2))

    return even_dimension(source_width * scale), even_dimension(source_height * scale)


def render_deepstream_config(
    destination: Path,
    *,
    video_path: Path,
    video_metadata: dict[str, Any],
    size: int,
    streams: int,
    perf_interval_seconds: int,
    write: bool = True,
) -> str:
    if destination.suffix != ".txt":
        raise ValueError(f"DeepStream config must use .txt extension: {destination}")
    if streams != 12:
        raise ValueError("The validated person engine and campaign require exactly 12 streams")
    width, height = active_area_dimensions(
        int(video_metadata["width"]), int(video_metadata["height"]), size
    )
    text = f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec={perf_interval_seconds}

[tiled-display]
enable=0

[source0]
enable=1
type=3
uri=file://{container_path(video_path)}
num-sources={streams}
gpu-id=0
cudadec-memtype=0

[sink0]
enable=1
type=1
sync=0
qos=0

[osd]
enable=0

[streammux]
gpu-id=0
live-source=0
batch-size={streams}
batched-push-timeout=40000
width={width}
height={height}
enable-padding=0
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
batch-size={streams}
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=/models/person/{size}/config_infer_primary.txt

[tests]
file-loop=1
"""
    if write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return text


def sha256_json(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def build_fingerprint(
    *,
    scene: dict[str, Any],
    video_metadata: dict[str, Any],
    model_profile: dict[str, Any],
    config_text: str,
    duration: int,
    warmup: int,
    perf_interval: int,
    startup_timeout: int,
    streams: int,
    image: str,
    image_id: str,
    gpu_index: int,
    gpu_identity: dict[str, Any],
    power_profile: dict[str, Any],
    max_temperature_c: float,
    power_safety_policy: dict[str, Any],
    preflight_power_limits_w: dict[str, Any],
    platform_thermal_sources: dict[str, Any],
    ds9_runtime_compatibility: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    runner_stat = RUNNER_PATH.stat()
    summarizer_stat = SUMMARIZER_PATH.stat()
    payload = {
        "run_schema_version": SCHEMA_VERSION,
        "benchmark_code": {
            "runner_sha256": sha256_file(
                RUNNER_PATH, runner_stat.st_size, runner_stat.st_mtime_ns
            ),
            "summarizer_sha256": sha256_file(
                SUMMARIZER_PATH, summarizer_stat.st_size, summarizer_stat.st_mtime_ns
            ),
        },
        "scene_id": scene["id"],
        "video_path": video_metadata["path"],
        "video_size_bytes": video_metadata["size_bytes"],
        "video_mtime_ns": video_metadata["mtime_ns"],
        "video_sha256": video_metadata["sha256"],
        "model_profile": model_profile,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "duration_seconds": duration,
        "warmup_seconds": warmup,
        "perf_interval_seconds": perf_interval,
        "startup_timeout_seconds": startup_timeout,
        "streams": streams,
        "image": image,
        "image_id": image_id,
        "gpu_index": gpu_index,
        "gpu_identity": gpu_identity,
        "power_profile": power_profile,
        "max_temperature_c": max_temperature_c,
        "power_safety_policy": power_safety_policy,
        "preflight_power_limits_w": preflight_power_limits_w,
        "platform_thermal_sources": platform_thermal_fingerprint(
            platform_thermal_sources
        ),
        "ds9_runtime_compatibility": ds9_runtime_compatibility,
    }
    return sha256_json(payload), payload


def query_gpu_row(gpu_index: int) -> list[str]:
    result = command(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed")
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_CSV_HEADER):
        raise RuntimeError(f"Unexpected nvidia-smi row: {result.stdout!r}")
    return [item.strip() for item in rows[0]]


def gpu_row_snapshot(row: list[str]) -> dict[str, str]:
    if len(row) != len(GPU_CSV_HEADER):
        raise ValueError(
            f"GPU row has {len(row)} fields; expected {len(GPU_CSV_HEADER)}"
        )
    return dict(zip(GPU_CSV_HEADER, row))


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clock_event_active(value: Any) -> bool:
    return str(value).strip().casefold() in {"active", "yes", "true", "1"}


def required_slowdown_telemetry_valid(snapshot: dict[str, Any]) -> bool:
    accepted = {"active", "not active", "yes", "no", "true", "false", "1", "0"}
    return all(
        str(snapshot.get(field, "")).strip().casefold() in accepted
        for field in REQUIRED_SLOWDOWN_TELEMETRY_FIELDS
    )


def assess_gpu_safety(
    snapshot: dict[str, Any], *, power_limit_drop_tolerance_w: float
) -> dict[str, Any]:
    """Return a machine-readable safety assessment for one NVML sample."""
    requested = optional_float(snapshot.get("power_requested_limit_w"))
    current = optional_float(snapshot.get("power_current_limit_w"))
    default = optional_float(snapshot.get("power_default_limit_w"))
    comparable_limits = {
        name: value
        for name, value in {
            "power_requested_limit_w": requested,
            "power_current_limit_w": current,
        }.items()
        if value is not None and math.isfinite(value) and value > 0
    }
    drops = {
        name: round(default - value, 3)
        for name, value in comparable_limits.items()
        if default is not None and default - value >= power_limit_drop_tolerance_w
    }
    slowdown_flags = {
        name: clock_event_active(snapshot.get(name))
        for name in (
            "clock_event_sw_thermal_slowdown",
            "clock_event_hw_slowdown",
            "clock_event_hw_thermal_slowdown",
            "clock_event_hw_power_brake_slowdown",
        )
    }
    return {
        "timestamp": snapshot.get("timestamp"),
        "power_limits_w": {
            "requested": requested,
            "current": current,
            "default": default,
        },
        "power_limit_telemetry_complete": (
            current is not None
            and default is not None
            and math.isfinite(current)
            and math.isfinite(default)
            and current > 0
            and default > 0
        ),
        "power_limit_drop_tolerance_w": power_limit_drop_tolerance_w,
        "power_limit_drop_detected": bool(drops),
        "power_limit_drop_w_by_field": drops,
        "pstate": snapshot.get("pstate"),
        "clock_event_reasons_active_mask": snapshot.get(
            "clock_event_reasons_active_mask"
        ),
        "clock_event_sw_power_cap": clock_event_active(
            snapshot.get("clock_event_sw_power_cap")
        ),
        "dangerous_slowdown_flags": slowdown_flags,
        "dangerous_slowdown_active": any(slowdown_flags.values()),
    }


def safety_event(
    code: str,
    reason: str,
    *,
    sample_number: int,
    snapshot: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "deepsafe.gpu-safety-event/v1",
        "detected_at_utc": utc_now(),
        "code": code,
        "reason": reason,
        "sample_number": sample_number,
        "gpu_csv_line_number": sample_number + 1,
        "snapshot": snapshot,
        "assessment": assessment,
    }


def validate_gpu_operating_policy_mode(value: str) -> str:
    if value not in GPU_OPERATING_POLICY_MODES:
        raise ValueError(
            "GPU operating-policy mode must be one of: "
            + ", ".join(GPU_OPERATING_POLICY_MODES)
        )
    return value


def classify_static_signal_event(
    event: dict[str, Any], *, operating_policy_mode: str
) -> dict[str, Any]:
    """Bind a temperature/power/slowdown observation to its run disposition."""
    mode = validate_gpu_operating_policy_mode(operating_policy_mode)
    return {
        **event,
        "operating_policy_mode": mode,
        "measurement_quality_signal": True,
        "disposition": (
            "safety_abort"
            if mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
            else "record_only_workstation_hardware_managed"
        ),
    }


def discover_platform_thermal_sources(
    hwmon_root: Path = Path("/sys/class/hwmon"),
    thermal_root: Path = Path("/sys/class/thermal"),
) -> dict[str, Any]:
    """Discover Dell fan/temperature and selected ACPI thermal sources.

    Platform telemetry is diagnostic and best-effort: missing or unreadable sysfs
    entries never make a benchmark fail.
    """
    sources: list[dict[str, Any]] = []
    errors: list[str] = []
    dell_path: Path | None = None
    try:
        for candidate in sorted(hwmon_root.glob("hwmon*")):
            name_path = candidate / "name"
            try:
                if name_path.read_text(encoding="utf-8").strip() == "dell_smm":
                    dell_path = candidate
                    break
            except OSError:
                continue
    except OSError as exc:
        errors.append(f"hwmon discovery: {exc}")

    if dell_path is not None:
        for fan_index in (1, 2):
            path = dell_path / f"fan{fan_index}_input"
            if path.is_file() and os.access(path, os.R_OK):
                sources.append(
                    {
                        "column": f"dell_fan{fan_index}_rpm",
                        "source_path": str(path),
                        "resolved_source_path": str(path.resolve()),
                        "kind": "dell_smm_fan",
                        "unit": "rpm",
                        "scale_divisor": 1.0,
                    }
                )
        for temp_index in range(1, 9):
            path = dell_path / f"temp{temp_index}_input"
            if path.is_file() and os.access(path, os.R_OK):
                sources.append(
                    {
                        "column": f"dell_temp{temp_index}_c",
                        "source_path": str(path),
                        "resolved_source_path": str(path.resolve()),
                        "kind": "dell_smm_temperature",
                        "unit": "celsius",
                        "scale_divisor": 1000.0,
                    }
                )

    requested_thermal_types = ("TVGA", "TCPU", "TSKN")
    found_thermal_types: set[str] = set()
    try:
        thermal_zones = sorted(thermal_root.glob("thermal_zone*"))
    except OSError as exc:
        thermal_zones = []
        errors.append(f"thermal-zone discovery: {exc}")
    for zone in thermal_zones:
        try:
            thermal_type = (zone / "type").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if thermal_type not in requested_thermal_types or thermal_type in found_thermal_types:
            continue
        path = zone / "temp"
        if not path.is_file() or not os.access(path, os.R_OK):
            continue
        try:
            initial_raw = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        # Linux thermal-zone drivers normally expose millidegrees Celsius, but
        # Dell's TVGA source on this host exposes whole degrees. Preserve the
        # inferred divisor in the source manifest so the conversion is auditable.
        divisor = 1000.0 if abs(initial_raw) >= 1000 else 1.0
        sources.append(
            {
                "column": f"thermal_{thermal_type.casefold()}_c",
                "source_path": str(path),
                "resolved_source_path": str(path.resolve()),
                "kind": "acpi_thermal_zone",
                "thermal_type": thermal_type,
                "unit": "celsius",
                "scale_divisor": divisor,
                "initial_raw_value": initial_raw,
            }
        )
        found_thermal_types.add(thermal_type)

    return {
        "schema_version": "deepsafe.platform-thermal-sources/v1",
        "discovered_at_utc": utc_now(),
        "available": bool(sources),
        "best_effort": True,
        "failure_policy": "record_unavailable_or_blank; never abort benchmark",
        "dell_smm_hwmon_path": str(dell_path) if dell_path is not None else None,
        "requested_thermal_zone_types": list(requested_thermal_types),
        "missing_thermal_zone_types": sorted(
            set(requested_thermal_types) - found_thermal_types
        ),
        "columns": ["timestamp", *(source["column"] for source in sources)],
        "sources": sources,
        "discovery_errors": errors,
    }


def platform_thermal_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "available": manifest.get("available"),
        "columns": manifest.get("columns", []),
        "sources": [
            {
                key: source.get(key)
                for key in (
                    "column",
                    "source_path",
                    "resolved_source_path",
                    "kind",
                    "thermal_type",
                    "unit",
                    "scale_divisor",
                )
            }
            for source in manifest.get("sources", [])
        ],
    }


def read_platform_thermal_row(
    manifest: dict[str, Any], timestamp: str
) -> tuple[list[str], list[str]]:
    row = [timestamp]
    errors: list[str] = []
    for source in manifest.get("sources", []):
        path = Path(source["source_path"])
        try:
            raw = float(path.read_text(encoding="utf-8").strip())
            divisor = float(source["scale_divisor"])
            if not math.isfinite(raw) or not math.isfinite(divisor) or divisor == 0:
                raise ValueError("non-finite value or invalid scale divisor")
            value = raw / divisor
            row.append(f"{value:.3f}")
        except (OSError, ValueError, ZeroDivisionError) as exc:
            row.append("")
            errors.append(f"{path}: {exc}")
    return row, errors


def query_gpu_identity(gpu_index: int) -> dict[str, str]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.total",
        "pci.bus_id",
    ]
    result = command(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed")
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise RuntimeError(f"Unexpected nvidia-smi identity row: {result.stdout!r}")
    return dict(zip(fields, (item.strip() for item in rows[0])))


def require_valid_gpu_identity(
    identity: dict[str, Any], *, gpu_index: int
) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise RuntimeError("GPU identity telemetry is not an object")
    normalized = {
        key: str(identity.get(key, "")).strip()
        for key in REQUIRED_GPU_IDENTITY_FIELDS
    }
    missing = [
        field
        for field in REQUIRED_GPU_IDENTITY_FIELDS
        if normalized.get(field, "").upper() in {"", "N/A", "[N/A]"}
    ]
    if missing:
        raise RuntimeError(
            "Required GPU identity telemetry is unavailable: " + ", ".join(missing)
        )
    if normalized["index"] != str(gpu_index):
        raise RuntimeError(
            f"GPU identity index drift: {normalized['index']} != {gpu_index}"
        )
    memory_total = optional_float(normalized["memory.total"])
    if memory_total is None or not math.isfinite(memory_total) or memory_total <= 0:
        raise RuntimeError("GPU identity memory.total is malformed")
    return normalized


def read_xid_log() -> dict[str, Any]:
    probes = [
        ["journalctl", "-k", "-b", "--no-pager", "--grep", "NVRM: Xid"],
        ["dmesg", "--color=never"],
    ]
    errors: list[str] = []
    for probe in probes:
        if shutil.which(probe[0]) is None:
            continue
        try:
            result = command(probe, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{probe[0]}: {exc}")
            continue
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode not in (0, 1) or "Operation not permitted" in combined:
            errors.append(f"{probe[0]}: {combined.strip()}")
            continue
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if "NVRM:" in line and "Xid" in line
        ]
        return {"available": True, "source": probe[0], "lines": lines, "count": len(lines)}
    return {"available": False, "source": None, "lines": [], "count": 0, "errors": errors}


def read_power_profile() -> dict[str, Any]:
    if shutil.which("powerprofilesctl") is None:
        return {"available": False, "value": None}
    result = command(["powerprofilesctl", "get"], timeout=10)
    if result.returncode != 0:
        return {
            "available": False,
            "value": None,
            "error": result.stderr.strip() or result.stdout.strip(),
        }
    return {"available": True, "value": result.stdout.strip()}


def new_xid_lines(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    if not before.get("available") or not after.get("available"):
        return []
    remaining = Counter(after.get("lines", [])) - Counter(before.get("lines", []))
    return list(remaining.elements())


def preflight(
    *,
    image: str,
    gpu_index: int,
    max_temperature_c: float,
    allow_non_performance_profile: bool,
    power_limit_drop_tolerance_w: float,
    slowdown_consecutive_samples: int,
    operating_policy_mode: str = DEFAULT_GPU_OPERATING_POLICY_MODE,
) -> dict[str, Any]:
    operating_policy_mode = validate_gpu_operating_policy_mode(
        operating_policy_mode
    )
    for executable in ("docker", "nvidia-smi", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable not found: {executable}")
    docker_info = command(["docker", "info"], timeout=30)
    if docker_info.returncode != 0:
        raise RuntimeError(f"Docker daemon unavailable: {docker_info.stderr.strip()}")
    image_inspect = command(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], timeout=30
    )
    if image_inspect.returncode != 0:
        raise RuntimeError(f"Docker image not found: {image}")

    power_profile = read_power_profile()
    if (
        power_profile.get("available")
        and power_profile.get("value") != "performance"
        and not allow_non_performance_profile
    ):
        raise RuntimeError(
            "powerprofilesctl is not in performance mode; use "
            "`powerprofilesctl set performance` or explicitly pass "
            "--allow-non-performance-profile"
        )

    gpu_identity = require_valid_gpu_identity(
        query_gpu_identity(gpu_index), gpu_index=gpu_index
    )

    gpu_rows: list[list[str]] = []
    for index in range(PREFLIGHT_GPU_SAMPLES):
        gpu_rows.append(query_gpu_row(gpu_index))
        if index + 1 < PREFLIGHT_GPU_SAMPLES:
            time.sleep(PREFLIGHT_GPU_SAMPLE_INTERVAL_SECONDS)
    gpu_snapshots = [gpu_row_snapshot(row) for row in gpu_rows]
    assessments = [
        assess_gpu_safety(
            snapshot,
            power_limit_drop_tolerance_w=power_limit_drop_tolerance_w,
        )
        for snapshot in gpu_snapshots
    ]
    safety_events: list[dict[str, Any]] = []
    diagnostic_events: list[dict[str, Any]] = []
    consecutive_slowdown = 0
    for sample_number, (snapshot, assessment) in enumerate(
        zip(gpu_snapshots, assessments), start=1
    ):
        if (
            str(snapshot.get("gpu_index", "")).strip() != gpu_identity["index"]
            or str(snapshot.get("gpu_name", "")).strip() != gpu_identity["name"]
        ):
            safety_events.append(
                safety_event(
                    "gpu_identity_drift",
                    "GPU CSV index/name differs from the attested preflight identity",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                )
            )
            break
        temperature = optional_float(snapshot.get("temperature_c"))
        if temperature is None or not math.isfinite(temperature):
            safety_events.append(
                safety_event(
                    "temperature_telemetry_unavailable",
                    "GPU temperature telemetry is unavailable during preflight",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                )
            )
            break
        if temperature >= max_temperature_c:
            event = classify_static_signal_event(
                safety_event(
                    "temperature_threshold",
                    f"GPU temperature {temperature:.1f} C reached threshold "
                    f"{max_temperature_c:.1f} C during preflight",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                ),
                operating_policy_mode=operating_policy_mode,
            )
            diagnostic_events.append(event)
            if operating_policy_mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE:
                safety_events.append(event)
                break
        if not assessment["power_limit_telemetry_complete"]:
            safety_events.append(
                safety_event(
                    "power_limit_telemetry_unavailable",
                    "Current/default GPU power-limit telemetry is unavailable during preflight",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                )
            )
            break
        if not required_slowdown_telemetry_valid(snapshot):
            safety_events.append(
                safety_event(
                    "slowdown_telemetry_unavailable",
                    "Required SW-thermal/HW-slowdown telemetry is malformed or unavailable",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                )
            )
            break
        if assessment["power_limit_drop_detected"]:
            event = classify_static_signal_event(
                safety_event(
                    "power_limit_below_default",
                    "GPU current/requested power limit is at least "
                    f"{power_limit_drop_tolerance_w:.1f} W below its default during preflight",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                ),
                operating_policy_mode=operating_policy_mode,
            )
            diagnostic_events.append(event)
            if operating_policy_mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE:
                safety_events.append(event)
                break
        if assessment["dangerous_slowdown_active"]:
            consecutive_slowdown += 1
        else:
            consecutive_slowdown = 0
        if consecutive_slowdown >= slowdown_consecutive_samples:
            event = classify_static_signal_event(
                safety_event(
                    "sustained_clock_slowdown",
                    "SW thermal or HW slowdown was active for "
                    f"{consecutive_slowdown} consecutive preflight samples",
                    sample_number=sample_number,
                    snapshot=snapshot,
                    assessment=assessment,
                ),
                operating_policy_mode=operating_policy_mode,
            )
            diagnostic_events.append(event)
            if operating_policy_mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE:
                safety_events.append(event)
                break
    xid = read_xid_log()
    if not xid.get("available"):
        raise RuntimeError(f"NVIDIA Xid log is unavailable: {xid.get('errors', [])}")
    platform_thermal_sources = discover_platform_thermal_sources()
    return {
        "status": "safety_abort" if safety_events else "ok",
        "checked_at_utc": utc_now(),
        "image": image,
        "image_id": image_inspect.stdout.strip(),
        "gpu_index": gpu_index,
        "gpu_snapshot": gpu_snapshots[-1],
        "gpu_snapshots": gpu_snapshots,
        "gpu_safety_assessments": assessments,
        "gpu_identity": gpu_identity,
        "power_profile": power_profile,
        "performance_profile_required": not allow_non_performance_profile,
        "max_temperature_c": max_temperature_c,
        "power_safety_policy": {
            "operating_policy_mode": operating_policy_mode,
            "hardware_protection_owner": "workstation_bios_ec_nvidia_driver",
            "static_signal_action": (
                "safety_abort"
                if operating_policy_mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
                else "record_measurement_quality_diagnostic"
            ),
            "power_limit_drop_tolerance_w": power_limit_drop_tolerance_w,
            "slowdown_consecutive_samples": slowdown_consecutive_samples,
            "preflight_samples": PREFLIGHT_GPU_SAMPLES,
            "preflight_sample_interval_seconds": (
                PREFLIGHT_GPU_SAMPLE_INTERVAL_SECONDS
            ),
            "power_limit_fields": [
                "power_requested_limit_w",
                "power_current_limit_w",
                "power_default_limit_w",
            ],
            "diagnostic_slowdown_flags": [
                "clock_event_sw_thermal_slowdown",
                "clock_event_hw_slowdown",
                "clock_event_hw_thermal_slowdown",
                "clock_event_hw_power_brake_slowdown",
            ],
            "abort_slowdown_flags": (
                [
                    "clock_event_sw_thermal_slowdown",
                    "clock_event_hw_slowdown",
                    "clock_event_hw_thermal_slowdown",
                    "clock_event_hw_power_brake_slowdown",
                ]
                if operating_policy_mode
                == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
                else []
            ),
            "required_telemetry_failure_action": "safety_abort",
        },
        "safety_events": safety_events,
        "diagnostic_events": diagnostic_events,
        "platform_thermal_sources": platform_thermal_sources,
        "xid": xid,
    }


class GpuMonitor(threading.Thread):
    def __init__(
        self,
        path: Path,
        event_path: Path,
        gpu_index: int,
        max_temperature_c: float,
        power_limit_drop_tolerance_w: float,
        slowdown_consecutive_samples: int,
        platform_thermal_path: Path | None = None,
        platform_thermal_manifest: dict[str, Any] | None = None,
        operating_policy_mode: str = DEFAULT_GPU_OPERATING_POLICY_MODE,
        expected_gpu_identity: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="scene-benchmark-gpu-monitor", daemon=True)
        self.path = path
        self.event_path = event_path
        self.gpu_index = gpu_index
        self.max_temperature_c = max_temperature_c
        self.power_limit_drop_tolerance_w = power_limit_drop_tolerance_w
        self.slowdown_consecutive_samples = slowdown_consecutive_samples
        self.operating_policy_mode = validate_gpu_operating_policy_mode(
            operating_policy_mode
        )
        self.expected_gpu_identity = (
            require_valid_gpu_identity(expected_gpu_identity, gpu_index=gpu_index)
            if expected_gpu_identity is not None
            else None
        )
        self.gpu_identity_checks = 0
        self.last_observed_gpu_identity: dict[str, str] | None = None
        self.platform_thermal_path = platform_thermal_path
        self.platform_thermal_manifest = platform_thermal_manifest
        self.stop_event = threading.Event()
        self.safety_reason: str | None = None
        self.safety_reason_code: str | None = None
        self.safety_event: dict[str, Any] | None = None
        self.diagnostic_events: list[dict[str, Any]] = []
        self.diagnostic_event_counts: Counter[str] = Counter()
        self.query_errors: list[str] = []
        self.consecutive_query_errors = 0
        self.consecutive_slowdown_samples = 0
        self.max_consecutive_slowdown_samples = 0
        self.slowdown_active_samples = 0
        self.power_limit_drop_samples = 0
        self.temperature_threshold_samples = 0
        self.maximum_temperature_c: float | None = None
        self.platform_thermal_samples = 0
        self.platform_thermal_error_count = 0
        self.platform_thermal_errors: list[str] = []
        self.samples = 0

    def stop(self) -> None:
        self.stop_event.set()

    def _abort(self, event: dict[str, Any]) -> None:
        if self.safety_event is not None:
            return
        self.safety_event = event
        self.safety_reason_code = str(event["code"])
        self.safety_reason = str(event["reason"])
        try:
            atomic_write_json(self.event_path, event)
        except Exception as exc:  # The status JSON still carries the event.
            self.query_errors.append(f"failed to persist GPU safety event: {exc}")
        self.stop_event.set()

    def _record_static_signal(self, event: dict[str, Any]) -> bool:
        """Record a static signal; return true only when legacy policy aborts."""
        classified = classify_static_signal_event(
            event, operating_policy_mode=self.operating_policy_mode
        )
        self.diagnostic_event_counts[str(classified["code"])] += 1
        if len(self.diagnostic_events) < 50:
            self.diagnostic_events.append(classified)
        if self.operating_policy_mode == LEGACY_STRICT_GPU_OPERATING_POLICY_MODE:
            self._abort(classified)
            return True
        return False

    def _check_gpu_identity(self) -> bool:
        if self.expected_gpu_identity is None:
            return True
        try:
            observed = require_valid_gpu_identity(
                query_gpu_identity(self.gpu_index), gpu_index=self.gpu_index
            )
        except Exception as exc:
            self._abort(
                {
                    "schema_version": "deepsafe.gpu-safety-event/v1",
                    "detected_at_utc": utc_now(),
                    "code": "gpu_identity_telemetry_failure",
                    "reason": f"Required GPU identity telemetry failed: {exc}",
                    "sample_number": self.samples,
                    "gpu_csv_line_number": self.samples + 1,
                    "expected_gpu_identity": self.expected_gpu_identity,
                }
            )
            return False
        self.gpu_identity_checks += 1
        self.last_observed_gpu_identity = observed
        if observed != self.expected_gpu_identity:
            self._abort(
                {
                    "schema_version": "deepsafe.gpu-safety-event/v1",
                    "detected_at_utc": utc_now(),
                    "code": "gpu_identity_drift",
                    "reason": "GPU identity/driver telemetry drifted during the run",
                    "sample_number": self.samples,
                    "gpu_csv_line_number": self.samples + 1,
                    "expected_gpu_identity": self.expected_gpu_identity,
                    "observed_gpu_identity": observed,
                }
            )
            return False
        return True

    def run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            handle = stack.enter_context(
                self.path.open("w", encoding="utf-8", newline="")
            )
            writer = csv.writer(handle)
            writer.writerow(GPU_CSV_HEADER)
            handle.flush()
            platform_handle = None
            platform_writer = None
            if self.platform_thermal_path is not None:
                self.platform_thermal_path.parent.mkdir(parents=True, exist_ok=True)
                platform_handle = stack.enter_context(
                    self.platform_thermal_path.open(
                        "w", encoding="utf-8", newline=""
                    )
                )
                platform_writer = csv.writer(platform_handle)
                platform_writer.writerow(
                    (self.platform_thermal_manifest or {}).get(
                        "columns", ["timestamp"]
                    )
                )
                platform_handle.flush()
            next_sample = time.monotonic()
            while not self.stop_event.is_set():
                try:
                    row = query_gpu_row(self.gpu_index)
                    writer.writerow(row)
                    handle.flush()
                    self.samples += 1
                    if (
                        self.expected_gpu_identity is not None
                        and (
                            self.samples == 1
                            or self.samples % GPU_IDENTITY_CHECK_INTERVAL_SAMPLES == 0
                        )
                        and not self._check_gpu_identity()
                    ):
                        break
                    if platform_writer is not None and platform_handle is not None:
                        platform_row, platform_errors = read_platform_thermal_row(
                            self.platform_thermal_manifest or {}, row[0]
                        )
                        platform_writer.writerow(platform_row)
                        platform_handle.flush()
                        self.platform_thermal_samples += 1
                        self.platform_thermal_error_count += len(platform_errors)
                        remaining_error_slots = max(
                            0, 50 - len(self.platform_thermal_errors)
                        )
                        self.platform_thermal_errors.extend(
                            platform_errors[:remaining_error_slots]
                        )
                    self.consecutive_query_errors = 0
                    snapshot = gpu_row_snapshot(row)
                    assessment = assess_gpu_safety(
                        snapshot,
                        power_limit_drop_tolerance_w=(
                            self.power_limit_drop_tolerance_w
                        ),
                    )
                    temperature = optional_float(snapshot.get("temperature_c"))
                    if temperature is None or not math.isfinite(temperature):
                        self._abort(
                            safety_event(
                                "temperature_telemetry_unavailable",
                                "GPU temperature telemetry is unavailable",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        )
                        break
                    self.maximum_temperature_c = (
                        temperature
                        if self.maximum_temperature_c is None
                        else max(self.maximum_temperature_c, temperature)
                    )
                    if temperature >= self.max_temperature_c:
                        self.temperature_threshold_samples += 1
                        if self._record_static_signal(
                            safety_event(
                                "temperature_threshold",
                                f"GPU temperature {temperature:.1f} C reached threshold "
                                f"{self.max_temperature_c:.1f} C",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        ):
                            break
                    if not assessment["power_limit_telemetry_complete"]:
                        self._abort(
                            safety_event(
                                "power_limit_telemetry_unavailable",
                                "Current/default GPU power-limit telemetry is unavailable",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        )
                        break
                    if not required_slowdown_telemetry_valid(snapshot):
                        self._abort(
                            safety_event(
                                "slowdown_telemetry_unavailable",
                                "Required SW-thermal/HW-slowdown telemetry is "
                                "malformed or unavailable",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        )
                        break
                    if assessment["power_limit_drop_detected"]:
                        self.power_limit_drop_samples += 1
                        if self._record_static_signal(
                            safety_event(
                                "power_limit_below_default",
                                "GPU current/requested power limit is at least "
                                f"{self.power_limit_drop_tolerance_w:.1f} W below its default",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        ):
                            break
                    if assessment["dangerous_slowdown_active"]:
                        self.consecutive_slowdown_samples += 1
                        self.slowdown_active_samples += 1
                    else:
                        self.consecutive_slowdown_samples = 0
                    self.max_consecutive_slowdown_samples = max(
                        self.max_consecutive_slowdown_samples,
                        self.consecutive_slowdown_samples,
                    )
                    if (
                        self.consecutive_slowdown_samples
                        == self.slowdown_consecutive_samples
                    ):
                        if self._record_static_signal(
                            safety_event(
                                "sustained_clock_slowdown",
                                "SW thermal or HW slowdown was active for "
                                f"{self.consecutive_slowdown_samples} consecutive samples",
                                sample_number=self.samples,
                                snapshot=snapshot,
                                assessment=assessment,
                            )
                        ):
                            break
                except Exception as exc:  # Monitoring failure is surfaced in status JSON.
                    self.query_errors.append(str(exc))
                    self.consecutive_query_errors += 1
                    if self.consecutive_query_errors >= MAX_CONSECUTIVE_GPU_QUERY_ERRORS:
                        event = {
                            "schema_version": "deepsafe.gpu-safety-event/v1",
                            "detected_at_utc": utc_now(),
                            "code": "gpu_telemetry_query_failure",
                            "reason": "GPU telemetry failed for "
                            f"{self.consecutive_query_errors} consecutive samples",
                            "sample_number": self.samples,
                            "gpu_csv_line_number": self.samples + 1,
                            "query_errors": self.query_errors[-self.consecutive_query_errors :],
                        }
                        self._abort(event)
                        break
                next_sample += 1.0
                self.stop_event.wait(max(0.0, next_sample - time.monotonic()))


def stop_attached_container(
    process: subprocess.Popen[Any], container_name: str, kill_grace: int
) -> tuple[int | None, str]:
    method = "sigint"
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=kill_grace)
    except subprocess.TimeoutExpired:
        method = "docker_rm_force"
        command(["docker", "rm", "-f", container_name], timeout=30)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            method = "sigkill"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    command(["docker", "rm", "-f", container_name], timeout=30)
    return process.returncode, method


def scan_fatal_log(path: Path) -> dict[str, int]:
    if not path.exists():
        return {name: 0 for name in FATAL_LOG_PATTERNS}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {name: len(pattern.findall(text)) for name, pattern in FATAL_LOG_PATTERNS.items()}


def count_active_perf_rows(path: Path, expected_streams: int) -> int:
    """Count structurally complete, non-zero DeepStream PERF rows."""
    if not path.exists():
        return 0
    active = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "**PERF:" not in line:
            continue
        pairs = PERF_PAIR.findall(line.partition("**PERF:")[2])
        if len(pairs) != expected_streams:
            continue
        if any(float(current) > 0 for current, _ in pairs):
            active += 1
    return active


def archive_previous_attempt(run_dir: Path) -> str | None:
    primary_candidates = [
        run_dir / "status.json",
        run_dir / "deepstream.log",
        run_dir / "gpu.csv",
        run_dir / "platform-thermal.csv",
        run_dir / "gpu-safety-event.json",
        run_dir / "runtime-container.json",
    ]
    if not any(path.exists() for path in primary_candidates):
        return None
    candidates = [*primary_candidates, run_dir / "deepstream.txt"]
    existing = [path for path in candidates if path.exists()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir / "attempts" / stamp
    suffix = 1
    while destination.exists():
        destination = run_dir / "attempts" / f"{stamp}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True)
    for path in existing:
        path.replace(destination / path.name)
    return project_relative(destination)


def should_skip(status_path: Path, fingerprint: str, force: bool = False) -> bool:
    if force or not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return status.get("status") == "complete" and status.get("fingerprint") == fingerprint


def docker_command(
    *, image: str, gpu_index: int, config_path: Path, container_name: str
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--gpus",
        f"device={gpu_index}",
        "--ipc=host",
        "--env",
        "GST_DEBUG=1",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "--volume",
        f"{PROJECT_ROOT}:/workspace:ro",
        "--volume",
        f"{PROJECT_ROOT / 'models'}:/models:ro",
        "--workdir",
        "/workspace",
        image,
        "deepstream-app",
        "-c",
        container_path(config_path),
    ]


def canonical_launch_command_sha256(launch_command: list[str]) -> str:
    if (
        not isinstance(launch_command, list)
        or not launch_command
        or not all(isinstance(value, str) and value for value in launch_command)
    ):
        raise RuntimeError("Scene launch command is malformed")
    rendered = json.dumps(
        launch_command,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _docker_result_detail(result: Any) -> str:
    stderr = str(getattr(result, "stderr", "") or "").strip()
    stdout = str(getattr(result, "stdout", "") or "").strip()
    return (stderr or stdout or "no diagnostic output")[:1000]


def _exact_scene_container_record(container_name: str) -> tuple[str, str] | None:
    """Return one exact-name container; Docker errors are never treated as absence."""

    result = command(
        [
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.ID}}\t{{.Names}}",
        ],
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot establish scene container presence; "
            f"docker ps returncode={result.returncode}; {_docker_result_detail(result)}"
        )
    lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) != 1:
        raise RuntimeError(
            f"Exact scene container lookup returned {len(lines)} records"
        )
    fields = lines[0].split("\t")
    if (
        len(fields) != 2
        or fields[1] != container_name
        or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None
    ):
        raise RuntimeError("Exact scene container lookup returned malformed identity")
    return fields[0], fields[1]


def _inspect_scene_container_item(container_id: str) -> dict[str, Any]:
    result = command(["docker", "inspect", container_id], timeout=15)
    if result.returncode != 0:
        raise RuntimeError(
            "Cannot inspect bound scene container; "
            f"returncode={result.returncode}; {_docker_result_detail(result)}"
        )
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise TypeError("inspect result must contain exactly one record")
        item = payload[0]
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Scene container inspect payload is malformed") from exc
    if not isinstance(item, dict) or item.get("Id") != container_id:
        raise RuntimeError("Scene container inspect identity differs")
    return item


def _runtime_string_vector(
    value: Any, context: str, *, allow_empty: bool = True
) -> list[str]:
    if value is None and allow_empty:
        result: list[str] = []
    elif isinstance(value, str):
        result = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        result = list(value)
    else:
        raise RuntimeError(f"Scene container {context} is malformed")
    if not allow_empty and not result:
        raise RuntimeError(f"Scene container {context} is empty")
    return result


def _expected_runtime_cmd(
    launch_command: list[str], resolved_image_id: str
) -> list[str]:
    positions = [
        index for index, value in enumerate(launch_command) if value == resolved_image_id
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(launch_command):
        raise RuntimeError("Resolved image ID is ambiguous in scene launch command")
    return launch_command[positions[0] + 1 :]


def _normalized_runtime_mounts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError("Scene container mount list is malformed")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("Scene container mount record is malformed")
        source = item.get("Source")
        destination = item.get("Destination")
        mount_type = item.get("Type")
        read_write = item.get("RW")
        if (
            not isinstance(source, str)
            or not isinstance(destination, str)
            or not isinstance(mount_type, str)
            or not isinstance(read_write, bool)
        ):
            raise RuntimeError("Scene container mount fields are malformed")
        normalized.append(
            {
                "type": mount_type,
                "source": str(Path(source).resolve()),
                "destination": destination,
                "read_write": read_write,
            }
        )
    return sorted(
        normalized, key=lambda item: (item["destination"], item["source"])
    )


def _normalized_gpu_device_request(value: Any, gpu_index: int) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError("Scene container must have exactly one GPU device request")
    request = value[0]
    driver = request.get("Driver")
    count = request.get("Count")
    device_ids = request.get("DeviceIDs")
    capabilities = request.get("Capabilities")
    options = request.get("Options")
    if driver not in {"", "nvidia"}:
        raise RuntimeError("Scene container GPU device-request driver differs")
    if isinstance(count, bool) or not isinstance(count, int) or count != 0:
        raise RuntimeError("Scene container GPU device-request count differs")
    if device_ids != [str(gpu_index)]:
        raise RuntimeError("Scene container GPU device ID differs")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or not all(
            isinstance(group, list)
            and all(isinstance(item, str) for item in group)
            for group in capabilities
        )
        or not any("gpu" in group for group in capabilities)
    ):
        raise RuntimeError("Scene container GPU capability request differs")
    if not isinstance(options, dict):
        raise RuntimeError("Scene container GPU request options are malformed")
    return {
        "driver": driver,
        "count": count,
        "device_ids": list(device_ids),
        "capabilities": capabilities,
        "options": options,
    }


def capture_runtime_container_attestation(
    *,
    launch_command: list[str],
    container_name: str,
    resolved_image_id: str,
    gpu_index: int,
) -> dict[str, Any] | None:
    """Capture the actual running scene container from exact Docker inspect data."""

    if (
        not isinstance(container_name, str)
        or not container_name
        or re.fullmatch(r"sha256:[0-9a-f]{64}", resolved_image_id) is None
        or isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise RuntimeError("Scene runtime identity is malformed")
    launch_sha256 = canonical_launch_command_sha256(launch_command)
    expected_cmd = _expected_runtime_cmd(launch_command, resolved_image_id)
    record = _exact_scene_container_record(container_name)
    if record is None:
        return None
    container_id, _ = record
    item = _inspect_scene_container_item(container_id)
    try:
        config = item["Config"]
        host_config = item["HostConfig"]
        state = item["State"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Scene runtime inspect payload is incomplete") from exc
    if (
        item.get("Name") != f"/{container_name}"
        or item.get("Image") != resolved_image_id
        or not isinstance(config, dict)
        or config.get("Image") != resolved_image_id
        or not isinstance(host_config, dict)
        or not isinstance(state, dict)
    ):
        raise RuntimeError("Scene runtime container image/name identity differs")

    cmd = _runtime_string_vector(config.get("Cmd"), "Config.Cmd", allow_empty=False)
    entrypoint = _runtime_string_vector(config.get("Entrypoint"), "Config.Entrypoint")
    path = item.get("Path")
    args = item.get("Args")
    if cmd != expected_cmd:
        raise RuntimeError("Scene runtime Config.Cmd differs from launch command")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(args, list)
        or not all(isinstance(value, str) for value in args)
        or [path, *args] != [*entrypoint, *cmd]
    ):
        raise RuntimeError("Scene runtime Path/Args differ from Entrypoint/Cmd")
    if config.get("WorkingDir") != "/workspace":
        raise RuntimeError("Scene runtime working directory differs")
    environment = config.get("Env")
    required_environment = {
        "GST_DEBUG=1",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
    }
    if (
        not isinstance(environment, list)
        or not all(isinstance(value, str) for value in environment)
        or not required_environment.issubset(environment)
    ):
        raise RuntimeError("Scene runtime required environment differs")

    expected_mounts = sorted(
        [
            {
                "type": "bind",
                "source": str(PROJECT_ROOT.resolve()),
                "destination": "/workspace",
                "read_write": False,
            },
            {
                "type": "bind",
                "source": str((PROJECT_ROOT / "models").resolve()),
                "destination": "/models",
                "read_write": False,
            },
        ],
        key=lambda mount: (mount["destination"], mount["source"]),
    )
    mounts = _normalized_runtime_mounts(item.get("Mounts"))
    if mounts != expected_mounts:
        raise RuntimeError("Scene runtime read-only mount contract differs")
    if host_config.get("AutoRemove") is not True or host_config.get("IpcMode") != "host":
        raise RuntimeError("Scene runtime lifecycle/IPC contract differs")
    device_request = _normalized_gpu_device_request(
        host_config.get("DeviceRequests"), gpu_index
    )
    started_at = state.get("StartedAt")
    if (
        state.get("Running") is not True
        or not isinstance(started_at, str)
        or not started_at
    ):
        raise RuntimeError("Scene runtime container was not running when attested")

    return {
        "schema_version": RUNTIME_CONTAINER_ATTESTATION_SCHEMA,
        "status": "verified_running",
        "captured_at_utc": utc_now(),
        "launch_command_sha256": launch_sha256,
        "container_id": container_id,
        "container_name": container_name,
        "image_id": resolved_image_id,
        "config_image": config["Image"],
        "entrypoint": entrypoint,
        "cmd": cmd,
        "actual_process": {"path": path, "args": list(args)},
        "working_dir": config["WorkingDir"],
        "required_environment": sorted(required_environment),
        "gpu_device_request": device_request,
        "mounts": mounts,
        "host_config": {"auto_remove": True, "ipc_mode": "host"},
        "state": {"running": True, "started_at_utc": started_at},
    }


def capture_and_write_runtime_container_attestation(
    path: Path,
    *,
    launch_command: list[str],
    container_name: str,
    resolved_image_id: str,
    gpu_index: int,
) -> dict[str, Any] | None:
    attestation = capture_runtime_container_attestation(
        launch_command=launch_command,
        container_name=container_name,
        resolved_image_id=resolved_image_id,
        gpu_index=gpu_index,
    )
    if attestation is not None:
        atomic_write_json(path, attestation)
    return attestation


def runtime_container_attestation_failure_reasons(
    attestation: dict[str, Any] | None,
    error: str | None,
    path: Path,
) -> list[str]:
    if attestation is None:
        if error is not None:
            return [f"runtime_container_attestation_error={error}"]
        return ["runtime_container_attestation_missing"]
    if not path.is_file():
        return ["runtime_container_attestation_artifact_missing"]
    return []


def run_one(
    *,
    scene: dict[str, Any],
    size: int,
    output_root: Path,
    duration: int,
    warmup: int,
    perf_interval: int,
    startup_timeout: int,
    streams: int,
    image: str,
    gpu_index: int,
    kill_grace: int,
    max_temperature_c: float,
    power_limit_drop_tolerance_w: float,
    slowdown_consecutive_samples: int,
    campaign_manifest: dict[str, Any],
    preflight_report: dict[str, Any],
    ds9_compatibility_receipt_path: Path = DEFAULT_DS9_COMPATIBILITY_RECEIPT,
    force: bool,
) -> dict[str, Any]:
    compatibility_binding = require_runtime_compatibility(
        resolve_project_path(ds9_compatibility_receipt_path),
        project_root=PROJECT_ROOT,
        requested_image=image,
        resolved_image_id=str(preflight_report.get("image_id")),
    )
    video_path = resolve_project_path(scene["video_path"])
    video_metadata = probe_video(video_path)
    model_profile = validate_person_profile(size)
    run_dir = output_root / scene["id"] / str(size)
    config_path = run_dir / "deepstream.txt"
    status_path = run_dir / "status.json"
    config_text = render_deepstream_config(
        config_path,
        video_path=video_path,
        video_metadata=video_metadata,
        size=size,
        streams=streams,
        perf_interval_seconds=perf_interval,
        write=False,
    )
    fingerprint, fingerprint_input = build_fingerprint(
        scene=scene,
        video_metadata=video_metadata,
        model_profile=model_profile,
        config_text=config_text,
        duration=duration,
        warmup=warmup,
        perf_interval=perf_interval,
        startup_timeout=startup_timeout,
        streams=streams,
        image=image,
        image_id=preflight_report["image_id"],
        gpu_index=gpu_index,
        gpu_identity=preflight_report["gpu_identity"],
        power_profile=preflight_report["power_profile"],
        max_temperature_c=max_temperature_c,
        power_safety_policy=preflight_report["power_safety_policy"],
        preflight_power_limits_w=preflight_report["gpu_safety_assessments"][-1][
            "power_limits_w"
        ],
        platform_thermal_sources=preflight_report["platform_thermal_sources"],
        ds9_runtime_compatibility=compatibility_binding,
    )
    if should_skip(status_path, fingerprint, force=force):
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        existing["resume_action"] = "skipped_matching_complete_run"
        print(f"SKIP {scene['id']} / {size}: matching complete run")
        return existing

    archived = archive_previous_attempt(run_dir)
    config_text = render_deepstream_config(
        config_path,
        video_path=video_path,
        video_metadata=video_metadata,
        size=size,
        streams=streams,
        perf_interval_seconds=perf_interval,
    )
    log_path = run_dir / "deepstream.log"
    gpu_path = run_dir / "gpu.csv"
    platform_thermal_path = run_dir / "platform-thermal.csv"
    safety_event_path = run_dir / "gpu-safety-event.json"
    runtime_container_path = run_dir / "runtime-container.json"
    start_utc = utc_now()
    run_xid_before = read_xid_log()
    run_token = hashlib.sha256(f"{scene['id']}-{size}-{start_utc}".encode()).hexdigest()[:10]
    container_name = f"deepsafe-scene-{size}-{run_token}"
    if (
        preflight_report.get("image") != image
        or not isinstance(preflight_report.get("image_id"), str)
        or not preflight_report["image_id"].startswith("sha256:")
    ):
        raise RuntimeError(
            "Scene benchmark preflight does not bind the requested tag to an immutable image ID"
        )
    docker_args = docker_command(
        image=preflight_report["image_id"],
        gpu_index=gpu_index,
        config_path=config_path,
        container_name=container_name,
    )
    mux_width, mux_height = active_area_dimensions(
        int(video_metadata["width"]), int(video_metadata["height"]), size
    )
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "fingerprint": fingerprint,
        "fingerprint_input": fingerprint_input,
        "scene": {
            "id": scene["id"],
            "benchmark_type": scene["benchmark_type"],
            "security_camera_relevance": scene["security_camera_relevance"],
            "source_manifest_id": scene["source_manifest_id"],
            "source_metadata": scene["source_metadata"],
            "notes": scene["notes"],
        },
        "video": video_metadata,
        "model": model_profile,
        "simulation": {
            "streams": streams,
            "source_type": 3,
            "num_sources": streams,
            "file_loop": True,
            "sink_type": 1,
            "sink_sync": False,
            "requested_duration_seconds": duration,
            "warmup_seconds": warmup,
            "perf_interval_seconds": perf_interval,
            "startup_timeout_seconds": startup_timeout,
            "measurement_clock": "steady_state_monotonic_wall_clock",
            "streammux": {
                "width": mux_width,
                "height": mux_height,
                "enable_padding": False,
                "policy": "source_aspect_fit_to_model_active_area",
            },
        },
        "accuracy_separation": campaign_manifest["accuracy_policy"],
        "timing": {"started_at_utc": start_utc},
        "safety": {"preflight": preflight_report, "run_xid_before": run_xid_before},
        "process": {
            "container_name": container_name,
            "requested_image": image,
            "resolved_image_id": preflight_report["image_id"],
            "docker_command": docker_args,
            "launch_command_sha256": canonical_launch_command_sha256(docker_args),
            "runtime_container_id": None,
            "exit_code": None,
        },
        "runtime_container_attestation": None,
        "runtime_container_attestation_error": None,
        "artifacts": {
            "config": project_relative(config_path),
            "deepstream_log": project_relative(log_path),
            "gpu_csv": project_relative(gpu_path),
            "platform_thermal_csv": project_relative(platform_thermal_path),
            "gpu_safety_event": project_relative(safety_event_path),
            "runtime_container": project_relative(runtime_container_path),
            "status": project_relative(status_path),
            "archived_previous_attempt": archived,
        },
        "ds9_runtime_compatibility": compatibility_binding,
    }
    atomic_write_json(status_path, status)
    print(
        f"RUN  {scene['id']} / {size}: {streams} streams, {duration}s, "
        f"{video_metadata['width']}x{video_metadata['height']} H.264"
    )

    preflight_gpu_identity = preflight_report.get("gpu_identity")
    monitor_expected_gpu_identity = (
        preflight_gpu_identity
        if isinstance(preflight_gpu_identity, dict)
        and all(field in preflight_gpu_identity for field in REQUIRED_GPU_IDENTITY_FIELDS)
        else None
    )
    monitor = GpuMonitor(
        gpu_path,
        safety_event_path,
        gpu_index,
        max_temperature_c,
        power_limit_drop_tolerance_w,
        slowdown_consecutive_samples,
        platform_thermal_path=platform_thermal_path,
        platform_thermal_manifest=preflight_report["platform_thermal_sources"],
        operating_policy_mode=preflight_report.get("power_safety_policy", {}).get(
            "operating_policy_mode", DEFAULT_GPU_OPERATING_POLICY_MODE
        ),
        expected_gpu_identity=monitor_expected_gpu_identity,
    )
    process: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    premature_exit = False
    interrupted = False
    run_exception: str | None = None
    termination_method: str | None = None
    runtime_container_attestation: dict[str, Any] | None = None
    runtime_container_attestation_error: str | None = None
    first_active_perf_seen_monotonic: float | None = None
    warmup_deadline_monotonic: float | None = None
    warmup_boundary_perf_rows: int | None = None
    measurement_started_monotonic: float | None = None
    measurement_ended_monotonic: float | None = None
    measurement_started_at_utc: str | None = None
    measurement_ended_at_utc: str | None = None
    measurement_start_perf_rows: int | None = None
    measurement_end_perf_rows: int | None = None
    measurement_start_gpu_samples: int | None = None
    measurement_end_gpu_samples: int | None = None
    measurement_complete = False
    measurement_timeout_reason: str | None = None
    requested_perf_intervals = math.ceil(duration / perf_interval)
    requested_gpu_samples = math.ceil(duration)
    startup_deadline = started + startup_timeout
    tail_grace = max(10.0, 3.0 * perf_interval)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        try:
            monitor.start()
            process = subprocess.Popen(
                docker_args,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                if runtime_container_attestation is None:
                    try:
                        runtime_container_attestation = (
                            capture_and_write_runtime_container_attestation(
                                runtime_container_path,
                                launch_command=docker_args,
                                container_name=container_name,
                                resolved_image_id=preflight_report["image_id"],
                                gpu_index=gpu_index,
                            )
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        runtime_container_attestation_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        break
                if process.poll() is not None:
                    premature_exit = not measurement_complete
                    break
                if monitor.safety_reason:
                    break
                now = time.monotonic()
                active_perf_rows = count_active_perf_rows(log_path, streams)

                if first_active_perf_seen_monotonic is None:
                    if active_perf_rows > 0:
                        first_active_perf_seen_monotonic = now
                        warmup_deadline_monotonic = now + warmup
                        if warmup == 0:
                            measurement_started_monotonic = now
                            measurement_started_at_utc = utc_now()
                            measurement_start_perf_rows = active_perf_rows
                            measurement_start_gpu_samples = monitor.samples
                    elif now >= startup_deadline:
                        measurement_timeout_reason = (
                            f"no active {streams}-stream PERF row within "
                            f"{startup_timeout}s startup timeout"
                        )
                        break
                elif measurement_started_monotonic is None:
                    assert warmup_deadline_monotonic is not None
                    if now >= warmup_deadline_monotonic:
                        if warmup_boundary_perf_rows is None:
                            # The first row after this boundary starts a wholly
                            # steady-state measurement interval.
                            warmup_boundary_perf_rows = active_perf_rows
                        elif active_perf_rows > warmup_boundary_perf_rows:
                            measurement_started_monotonic = now
                            measurement_started_at_utc = utc_now()
                            measurement_start_perf_rows = active_perf_rows
                            measurement_start_gpu_samples = monitor.samples
                        elif now >= warmup_deadline_monotonic + tail_grace:
                            measurement_timeout_reason = (
                                "no complete PERF boundary after warmup"
                            )
                            break
                else:
                    measurement_deadline = measurement_started_monotonic + duration
                    if now >= measurement_deadline:
                        assert measurement_start_perf_rows is not None
                        assert measurement_start_gpu_samples is not None
                        perf_rows_in_window = active_perf_rows - measurement_start_perf_rows
                        gpu_samples_in_window = monitor.samples - measurement_start_gpu_samples
                        if (
                            perf_rows_in_window >= requested_perf_intervals
                            and gpu_samples_in_window >= requested_gpu_samples
                        ):
                            measurement_complete = True
                            measurement_ended_monotonic = now
                            measurement_ended_at_utc = utc_now()
                            measurement_end_perf_rows = active_perf_rows
                            measurement_end_gpu_samples = monitor.samples
                            break
                        if now >= measurement_deadline + tail_grace:
                            measurement_timeout_reason = (
                                "steady-state wall-clock deadline passed without "
                                f"complete telemetry (PERF {perf_rows_in_window}/"
                                f"{requested_perf_intervals}, GPU {gpu_samples_in_window}/"
                                f"{requested_gpu_samples})"
                            )
                            break
                time.sleep(0.25)
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            run_exception = f"{type(exc).__name__}: {exc}"
        finally:
            # Freeze the measurement bounds before shutdown/kill-grace output;
            # teardown PERF and GPU samples must never enter the result.
            if measurement_end_perf_rows is None:
                measurement_end_perf_rows = count_active_perf_rows(log_path, streams)
            if measurement_end_gpu_samples is None:
                measurement_end_gpu_samples = monitor.samples
            if measurement_ended_monotonic is None:
                measurement_ended_monotonic = time.monotonic()
                measurement_ended_at_utc = utc_now()
            if process is not None and process.poll() is None:
                _, termination_method = stop_attached_container(
                    process, container_name, kill_grace
                )
            elif process is not None:
                command(["docker", "rm", "-f", container_name], timeout=30)
            monitor.stop()
            if monitor.ident is not None:
                monitor.join(timeout=5)

    ended = time.monotonic()
    exit_code = process.returncode if process is not None else None
    post_xid = read_xid_log()
    post_power_profile = read_power_profile()
    new_xids = new_xid_lines(run_xid_before, post_xid)
    perf_start = (
        measurement_start_perf_rows
        if measurement_start_perf_rows is not None
        else measurement_end_perf_rows
    )
    gpu_start = (
        measurement_start_gpu_samples
        if measurement_start_gpu_samples is not None
        else measurement_end_gpu_samples
    )
    perf = parse_perf(
        log_path,
        streams,
        duration,
        perf_interval,
        active_row_start=perf_start or 0,
        active_row_end=measurement_end_perf_rows,
    )
    gpu = parse_gpu_metrics(
        gpu_path,
        duration,
        sample_start=gpu_start or 0,
        sample_end=measurement_end_gpu_samples,
    )
    fatal_log_matches = scan_fatal_log(log_path)

    failure_reasons: list[str] = []
    safety_reasons: list[str] = []
    if interrupted:
        failure_reasons.append("operator_interrupt")
    if run_exception:
        failure_reasons.append(f"run_exception={run_exception}")
    failure_reasons.extend(
        runtime_container_attestation_failure_reasons(
            runtime_container_attestation,
            runtime_container_attestation_error,
            runtime_container_path,
        )
    )
    if monitor.safety_reason:
        safety_reasons.append(monitor.safety_reason)
    if not run_xid_before.get("available") or not post_xid.get("available"):
        safety_reasons.append("nvidia_xid_log_unavailable")
    if new_xids:
        safety_reasons.append(f"new_nvidia_xid_events={len(new_xids)}")
    if (
        preflight_report.get("performance_profile_required")
        and not post_power_profile.get("available")
    ):
        safety_reasons.append("post_run_power_profile_unavailable")
    elif (
        preflight_report.get("performance_profile_required")
        and post_power_profile.get("value") != "performance"
    ):
        safety_reasons.append(
            f"power_profile_changed_to={post_power_profile.get('value')}"
        )
    if measurement_timeout_reason:
        failure_reasons.append(f"measurement_timeout={measurement_timeout_reason}")
    if not measurement_complete:
        failure_reasons.append("steady_state_measurement_window_incomplete")
    if premature_exit:
        failure_reasons.append("deepstream_exited_before_requested_duration")
    if exit_code not in EXPECTED_EXIT_CODES:
        failure_reasons.append(f"unexpected_exit_code={exit_code}")
    if perf.get("status") != "ok":
        failure_reasons.append(f"throughput_status={perf.get('status')}")
    if perf.get("header_streams") != streams:
        failure_reasons.append(
            f"deepstream_stream_count={perf.get('header_streams')} expected={streams}"
        )
    if perf.get("inactive_stream_ids"):
        failure_reasons.append(
            f"inactive_stream_ids={perf.get('inactive_stream_ids')}"
        )
    if perf.get("streams_with_nonpositive_fps"):
        failure_reasons.append(
            "streams_with_nonpositive_fps="
            f"{perf.get('streams_with_nonpositive_fps')}"
        )
    if gpu.get("status") != "ok":
        failure_reasons.append(f"gpu_status={gpu.get('status')}")
    if any(fatal_log_matches.values()):
        failure_reasons.append("fatal_patterns_in_deepstream_log")

    if interrupted:
        final_status = "interrupted"
    elif safety_reasons:
        final_status = "safety_abort"
    elif failure_reasons:
        final_status = "failed"
    else:
        final_status = "complete"
    failure_reasons = [*safety_reasons, *failure_reasons]

    measurement_wall_seconds = None
    if measurement_started_monotonic is not None and measurement_ended_monotonic is not None:
        measurement_wall_seconds = round(
            measurement_ended_monotonic - measurement_started_monotonic, 3
        )
    if measurement_wall_seconds is None or measurement_wall_seconds < duration:
        failure_reasons.append(
            f"measurement_wall_seconds={measurement_wall_seconds} expected_at_least={duration}"
        )
        if final_status == "complete":
            final_status = "failed"

    status.update(
        {
            "status": final_status,
            "resume_action": "executed",
            "timing": {
                "started_at_utc": start_utc,
                "finished_at_utc": utc_now(),
                "wall_time_seconds": round(ended - started, 3),
                "requested_duration_seconds": duration,
                "warmup_seconds": warmup,
                "startup_timeout_seconds": startup_timeout,
                "first_active_perf_seen_after_process_start_seconds": (
                    round(first_active_perf_seen_monotonic - started, 3)
                    if first_active_perf_seen_monotonic is not None
                    else None
                ),
                "measurement_started_at_utc": measurement_started_at_utc,
                "measurement_finished_at_utc": measurement_ended_at_utc,
                "measurement_wall_time_seconds": measurement_wall_seconds,
                "measurement_complete": measurement_complete,
                "measurement_timeout_reason": measurement_timeout_reason,
                "measurement_perf_row_bounds": [
                    measurement_start_perf_rows,
                    measurement_end_perf_rows,
                ],
                "measurement_gpu_sample_bounds": [
                    measurement_start_gpu_samples,
                    measurement_end_gpu_samples,
                ],
            },
            "process": {
                **status["process"],
                "exit_code": exit_code,
                "expected_exit_codes": sorted(EXPECTED_EXIT_CODES),
                "premature_exit": premature_exit,
                "termination_method": termination_method,
                "runtime_container_id": (
                    runtime_container_attestation.get("container_id")
                    if isinstance(runtime_container_attestation, dict)
                    else None
                ),
            },
            "runtime_container_attestation": runtime_container_attestation,
            "runtime_container_attestation_error": (
                runtime_container_attestation_error
            ),
            "safety": {
                "preflight": preflight_report,
                "run_xid_before": run_xid_before,
                "post_run_xid": post_xid,
                "post_run_power_profile": post_power_profile,
                "new_xid_lines": new_xids,
                "monitor_abort_reason_code": monitor.safety_reason_code,
                "monitor_abort_reason": monitor.safety_reason,
                "monitor_safety_event": monitor.safety_event,
                "temperature_abort_reason": (
                    monitor.safety_reason
                    if monitor.safety_reason_code == "temperature_threshold"
                    else None
                ),
                "monitor_query_errors": monitor.query_errors,
                "monitor_samples": monitor.samples,
                "gpu_identity_checks": getattr(monitor, "gpu_identity_checks", 0),
                "last_observed_gpu_identity": getattr(
                    monitor, "last_observed_gpu_identity", None
                ),
                "max_consecutive_monitor_query_errors": (
                    MAX_CONSECUTIVE_GPU_QUERY_ERRORS
                ),
                "power_safety_policy": preflight_report["power_safety_policy"],
                "monitor_diagnostic_events_first_50": getattr(
                    monitor, "diagnostic_events", []
                ),
                "monitor_diagnostic_event_counts": dict(
                    sorted(
                        getattr(monitor, "diagnostic_event_counts", {}).items()
                    )
                ),
                "temperature_threshold_samples": (
                    getattr(monitor, "temperature_threshold_samples", 0)
                ),
                "maximum_temperature_c": getattr(
                    monitor, "maximum_temperature_c", None
                ),
                "power_limit_drop_samples": monitor.power_limit_drop_samples,
                "slowdown_active_samples": monitor.slowdown_active_samples,
                "max_consecutive_slowdown_samples": (
                    monitor.max_consecutive_slowdown_samples
                ),
                "platform_thermal": {
                    "sources": preflight_report["platform_thermal_sources"],
                    "samples": monitor.platform_thermal_samples,
                    "read_error_count": monitor.platform_thermal_error_count,
                    "read_errors_first_50": monitor.platform_thermal_errors,
                },
            },
            "throughput": perf,
            "gpu": gpu,
            "deepstream_log_fatal_matches": fatal_log_matches,
            "failure_reasons": failure_reasons,
        }
    )
    atomic_write_json(status_path, status)
    aggregate = perf.get("aggregate_current_fps", {})
    print(
        f"{final_status.upper():<12} {scene['id']} / {size}: "
        f"aggregate mean={aggregate.get('mean')} p05={aggregate.get('p05')} "
        f"p95={aggregate.get('p95')} FPS"
    )
    if interrupted:
        raise KeyboardInterrupt
    return status


def write_matrix_summary(
    output_root: Path,
    *,
    selected_scenes: list[dict[str, Any]],
    sizes: list[int],
    duration: int,
    warmup: int,
    streams: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for scene in selected_scenes:
        for size in sizes:
            status_path = output_root / scene["id"] / str(size) / "status.json"
            state = "pending"
            fingerprint = None
            if status_path.is_file():
                try:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                    state = payload.get("status", "unknown")
                    fingerprint = payload.get("fingerprint")
                except (OSError, json.JSONDecodeError):
                    state = "invalid_status_json"
            counts[state] += 1
            runs.append(
                {
                    "scene_id": scene["id"],
                    "benchmark_type": scene["benchmark_type"],
                    "model_input_size": size,
                    "status": state,
                    "fingerprint": fingerprint,
                    "status_path": project_relative(status_path),
                }
            )
    summary = {
        "schema_version": "deepsafe.scene-benchmark-matrix/v1",
        "generated_at_utc": utc_now(),
        "streams": streams,
        "duration_seconds_per_run": duration,
        "warmup_seconds_per_run": warmup,
        "selected_scenes": len(selected_scenes),
        "selected_sizes": sizes,
        "expected_runs": len(selected_scenes) * len(sizes),
        "status_counts": dict(sorted(counts.items())),
        "runs": runs,
    }
    atomic_write_json(output_root / "matrix-summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable 12-camera x scene x YOLO11s 640/960 DeepStream benchmark"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenes", nargs="+", help="Scene ids; default is the full manifest")
    parser.add_argument("--sizes", nargs="+", type=int, choices=(640, 960))
    parser.add_argument("--duration", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--perf-interval", type=int)
    parser.add_argument("--startup-timeout", type=int)
    parser.add_argument("--image", default="deepsafe-deepstream:9.0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--reentry-evidence",
        type=Path,
        default=DEFAULT_REENTRY_EVIDENCE,
        help="current fail-closed sustained-load re-entry evidence bundle",
    )
    parser.add_argument(
        "--ds9-compatibility-receipt",
        type=Path,
        default=DEFAULT_DS9_COMPATIBILITY_RECEIPT,
        help="current production-ready DeepStream 9 compatibility receipt",
    )
    parser.add_argument("--kill-grace", type=int, default=8)
    parser.add_argument(
        "--gpu-operating-policy-mode",
        choices=GPU_OPERATING_POLICY_MODES,
        default=DEFAULT_GPU_OPERATING_POLICY_MODE,
        help=(
            "workstation_managed records temperature/power/slowdown signals without "
            "aborting; legacy_strict preserves the previous static-threshold aborts"
        ),
    )
    parser.add_argument(
        "--max-temperature-c",
        type=float,
        default=86.0,
        help=(
            "GPU temperature diagnostic threshold; abort threshold only in "
            "legacy_strict mode (default: 86)"
        ),
    )
    parser.add_argument(
        "--power-limit-drop-tolerance-w",
        type=float,
        default=DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W,
        help=(
            "Flag when current/requested GPU power limit is this many watts below "
            "the default; abort only in legacy_strict mode (default: 5)"
        ),
    )
    parser.add_argument(
        "--slowdown-consecutive-samples",
        type=int,
        default=DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES,
        help=(
            "Flag after this many consecutive SW-thermal/HW-slowdown samples; "
            "abort only in legacy_strict mode (default: 2)"
        ),
    )
    parser.add_argument("--allow-non-performance-profile", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun matching complete runs")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run manifest smoke scene for 15 seconds at 640 input",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = resolve_project_path(args.manifest)
    manifest = load_manifest(manifest_path)
    campaign = manifest["campaign"]
    streams = int(campaign["streams"])
    scene_by_id = {scene["id"]: scene for scene in manifest["scenes"]}

    if args.list_scenes:
        for scene in manifest["scenes"]:
            print(
                f"{scene['id']}\t{scene['benchmark_type']}\t"
                f"{scene['security_camera_relevance']}\t{scene['video_path']}"
            )
        return 0

    if args.smoke:
        if (
            args.scenes
            or args.sizes
            or args.duration is not None
            or args.warmup is not None
            or args.perf_interval is not None
            or args.startup_timeout is not None
        ):
            raise SystemExit("--smoke cannot be combined with scene/size/duration overrides")
        selected_ids = [campaign["smoke_scene_id"]]
        sizes = [int(campaign["smoke_model_input_size"])]
        duration = int(campaign["smoke_duration_seconds"])
        warmup = int(campaign["smoke_warmup_seconds"])
        perf_interval = int(campaign["smoke_perf_interval_seconds"])
        startup_timeout = int(campaign["startup_timeout_seconds"])
        output_root = resolve_project_path(args.output or DEFAULT_SMOKE_OUTPUT)
    else:
        selected_ids = args.scenes or [scene["id"] for scene in manifest["scenes"]]
        sizes = args.sizes or [int(value) for value in campaign["model_input_sizes"]]
        duration = (
            int(campaign["duration_seconds"])
            if args.duration is None
            else args.duration
        )
        warmup = (
            int(campaign["warmup_seconds"])
            if args.warmup is None
            else args.warmup
        )
        perf_interval = (
            int(campaign["perf_interval_seconds"])
            if args.perf_interval is None
            else args.perf_interval
        )
        startup_timeout = (
            int(campaign["startup_timeout_seconds"])
            if args.startup_timeout is None
            else args.startup_timeout
        )
        output_root = resolve_project_path(args.output or DEFAULT_OUTPUT)

    if (
        duration <= 0
        or warmup < 0
        or perf_interval <= 0
        or startup_timeout <= 0
        or args.kill_grace <= 0
        or not math.isfinite(args.max_temperature_c)
        or not 0 < args.max_temperature_c <= 95
        or not math.isfinite(args.power_limit_drop_tolerance_w)
        or args.power_limit_drop_tolerance_w <= 0
        or args.slowdown_consecutive_samples <= 0
    ):
        raise SystemExit(
            "duration, perf interval, startup timeout and kill grace must be positive; "
            "warmup must be non-negative; max temperature must be finite and in (0, 95]; "
            "power-limit tolerance and slowdown consecutive samples must be positive"
        )
    if perf_interval > duration:
        raise SystemExit("perf interval cannot exceed measurement duration")
    unknown = [scene_id for scene_id in selected_ids if scene_id not in scene_by_id]
    if unknown:
        raise SystemExit(f"Unknown scene ids: {', '.join(unknown)}")
    selected_scenes = [scene_by_id[scene_id] for scene_id in selected_ids]
    project_relative(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_cache: dict[str, dict[str, Any]] = {}
    for scene in selected_scenes:
        video_path = resolve_project_path(scene["video_path"])
        metadata = video_cache.setdefault(scene["id"], probe_video(video_path))
        for size in sizes:
            validate_person_profile(size)
            if args.generate_only:
                render_deepstream_config(
                    output_root / scene["id"] / str(size) / "deepstream.txt",
                    video_path=video_path,
                    video_metadata=metadata,
                    size=size,
                    streams=streams,
                    perf_interval_seconds=perf_interval,
                )
    if args.generate_only:
        write_ds9_pending_report(
            output_root / "ds9-runtime-compatibility-pending.json",
            requested_image=args.image,
            project_root=PROJECT_ROOT,
            launch_scope="scene-benchmark-generate-only",
        )
        summary = write_matrix_summary(
            output_root,
            selected_scenes=selected_scenes,
            sizes=sizes,
            duration=duration,
            warmup=warmup,
            streams=streams,
        )
        print(
            f"Generated {summary['expected_runs']} DeepStream .txt configs under "
            f"{project_relative(output_root)}; no GPU run started."
        )
        return 0

    compatibility_path = resolve_project_path(args.ds9_compatibility_receipt)
    try:
        early_compatibility = prevalidate_runtime_compatibility(
            compatibility_path,
            project_root=PROJECT_ROOT,
            requested_image=args.image,
        )
    except Exception as exc:
        atomic_write_json(
            output_root / "preflight.json",
            {
                "status": "ds9_runtime_compatibility_blocked",
                "checked_at_utc": utc_now(),
                "requested_image": args.image,
                "receipt": project_relative(compatibility_path),
                "error": f"{type(exc).__name__}: {exc}",
                "gpu_process_started": False,
            },
        )
        write_matrix_summary(
            output_root,
            selected_scenes=selected_scenes,
            sizes=sizes,
            duration=duration,
            warmup=warmup,
            streams=streams,
        )
        print(f"SAFETY STOP: {exc}", file=sys.stderr)
        return 3

    try:
        with gpu_lock(args.gpu_index) as lock_metadata:
            # Local import avoids a module-import cycle: the evidence collector
            # intentionally reuses this runner's read-only telemetry parsers.
            from validation.gpu_reentry_evidence import (
                ReentryEvidenceError,
                require_reentry_evidence,
            )

            try:
                reentry_receipt = require_reentry_evidence(
                    resolve_project_path(args.reentry_evidence),
                    project_root=PROJECT_ROOT,
                )
            except ReentryEvidenceError as exc:
                atomic_write_json(
                    output_root / "preflight.json",
                    {
                        "status": "reentry_blocked",
                        "checked_at_utc": utc_now(),
                        "gpu_index": args.gpu_index,
                        "gpu_lock": lock_metadata,
                        "reentry_evidence": project_relative(
                            resolve_project_path(args.reentry_evidence)
                        ),
                        "error": f"{type(exc).__name__}: {exc}",
                        "gpu_process_started": False,
                    },
                )
                write_matrix_summary(
                    output_root,
                    selected_scenes=selected_scenes,
                    sizes=sizes,
                    duration=duration,
                    warmup=warmup,
                    streams=streams,
                )
                print(f"SAFETY STOP: {exc}", file=sys.stderr)
                return 3
            preflight_report = preflight(
                image=args.image,
                gpu_index=args.gpu_index,
                max_temperature_c=args.max_temperature_c,
                allow_non_performance_profile=args.allow_non_performance_profile,
                power_limit_drop_tolerance_w=(
                    args.power_limit_drop_tolerance_w
                ),
                slowdown_consecutive_samples=(
                    args.slowdown_consecutive_samples
                ),
                operating_policy_mode=args.gpu_operating_policy_mode,
            )
            preflight_report["gpu_lock"] = lock_metadata
            preflight_report["gpu_reentry_evidence"] = reentry_receipt
            preflight_report["ds9_runtime_compatibility_prevalidated"] = (
                early_compatibility
            )
            atomic_write_json(output_root / "preflight.json", preflight_report)
            if preflight_report.get("status") != "ok":
                write_matrix_summary(
                    output_root,
                    selected_scenes=selected_scenes,
                    sizes=sizes,
                    duration=duration,
                    warmup=warmup,
                    streams=streams,
                )
                print(
                    "SAFETY STOP: preflight GPU power/thermal gate rejected the run: "
                    f"{preflight_report.get('safety_events')}",
                    file=sys.stderr,
                )
                return 3
            failures = 0
            safety_stop = False
            try:
                for scene in selected_scenes:
                    for size in sizes:
                        result = run_one(
                            scene=scene,
                            size=size,
                            output_root=output_root,
                            duration=duration,
                            warmup=warmup,
                            perf_interval=perf_interval,
                            startup_timeout=startup_timeout,
                            streams=streams,
                            image=args.image,
                            gpu_index=args.gpu_index,
                            kill_grace=args.kill_grace,
                            max_temperature_c=args.max_temperature_c,
                            power_limit_drop_tolerance_w=(
                                args.power_limit_drop_tolerance_w
                            ),
                            slowdown_consecutive_samples=(
                                args.slowdown_consecutive_samples
                            ),
                            campaign_manifest=manifest,
                            preflight_report=preflight_report,
                            ds9_compatibility_receipt_path=compatibility_path,
                            force=args.force,
                        )
                        if result.get("status") != "complete":
                            failures += 1
                            if result.get("status") == "safety_abort":
                                safety_stop = True
                            if args.fail_fast:
                                raise RuntimeError(
                                    f"Run failed: {scene['id']} / {size}: "
                                    f"{result.get('failure_reasons')}"
                                )
                        write_matrix_summary(
                            output_root,
                            selected_scenes=selected_scenes,
                            sizes=sizes,
                            duration=duration,
                            warmup=warmup,
                            streams=streams,
                        )
                        if safety_stop:
                            print(
                                "SAFETY STOP: remaining matrix runs were not started",
                                file=sys.stderr,
                            )
                            break
                    if safety_stop:
                        break
            finally:
                write_matrix_summary(
                    output_root,
                    selected_scenes=selected_scenes,
                    sizes=sizes,
                    duration=duration,
                    warmup=warmup,
                    streams=streams,
                )
    except RuntimeError as exc:
        if "locked by another DeepSafe campaign" in str(exc):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
