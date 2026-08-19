#!/usr/bin/env python3
"""Two-pass, fail-closed DeepStream-Yolo parser image bootstrap.

The default mode only writes an inert plan.  ``--execute-discovery`` builds the
``parser-audit-export`` target without accepting a parser digest and measures
the exported ELF.  ``--execute-production`` accepts only the immutable
discovery receipt, rebuilds the production target with that measured digest,
and records Docker's immutable image ID.  Neither execution mode requests a
GPU; this controller never runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA_VERSION = "deepsafe.ds9-parser-bootstrap-plan/v1"
DISCOVERY_SCHEMA_VERSION = "deepsafe.ds9-parser-discovery-receipt/v1"
PRODUCTION_SCHEMA_VERSION = "deepsafe.ds9-parser-production-build-receipt/v1"
SOURCE_SCHEMA_VERSION = "deepsafe.ds9-parser-source-lineage/v3"
BUILD_LINEAGE_SCHEMA_VERSION = "deepsafe.deepstream-build-lineage/v3"
MAX_RECEIPT_AGE = timedelta(hours=24)
BUILD_THERMAL_SAMPLE_SECONDS = 1.0
BUILD_TELEMETRY_SCHEMA_VERSION = "deepsafe.ds9-parser-build-telemetry/v2"
BUILD_POLICY_SCHEMA_VERSION = "deepsafe.ds9-parser-build-operating-policy/v1"
BUILD_OPERATING_POLICY_ID = "workstation_managed"
BUILD_SLOWDOWN_FLAG_NAMES = (
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
)
MAX_JSON_BYTES = 4 * 1024 * 1024

BASE_TAG = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
SOURCE_REPOSITORY = "https://github.com/marcoslucianops/DeepStream-Yolo.git"
SOURCE_COMMIT = "2894babce8e75c49115dbe0c7b516289ed853565"
SOURCE_TREE = "1740cc4bc7e925f30e4eea0160064bfde729f8d8"
SOURCE_PATCH = Path(
    "deepstream/patches/deepstream-yolo-ds9-cuda-kernel-proof.patch"
)
SOURCE_PATCH_SHA256 = "dd85619bf62da249d17d99e967caf53de96367a0f139286c52a86f5e67b7623e"
PATCHED_SOURCE_PATH = "nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu"
UPSTREAM_SOURCE_SHA256 = "a63299206550f1f8dd413cb9328352db304cdc020a47451170fd4c7eda0adf4d"
PATCHED_SOURCE_SHA256 = "642e2875d67c3528c7ea301bcd1e973ea31fb019865f2aa116122583e9a765e3"
PATCHED_SOURCE_TREE = "753acbd2995f9f8c0b791f6152f0793baa11b71a"
PARSER_BUILD_MAKEFILE_PATH = "nvdsinfer_custom_impl_Yolo/Makefile"
PARSER_BUILD_MAKEFILE_SHA256 = (
    "fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0"
)
PARSER_CUDA_CUBIN_ARCHITECTURE = "sm_86"
PARSER_CUDA_PTX_ARCHITECTURE = "compute_86"
PARSER_CUDA_GENCODE_FLAGS = (
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_86,code=compute_86",
)
PARSER_BUILD_COMMAND = (
    "nice -n 10 make -C nvdsinfer_custom_impl_Yolo -j2 CUDA_VER=13.1 "
    "'CUFLAGS=-I/opt/nvidia/deepstream/deepstream/sources/includes "
    "-I/usr/local/cuda-13.1/include "
    "-gencode=arch=compute_86,code=sm_86 "
    "-gencode=arch=compute_86,code=compute_86'"
)
PARSER_BUILD_COMMAND_SHA256 = (
    "e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1"
)
PARSER_POST_LINK_TOOL_PATH = "/usr/bin/x86_64-linux-gnu-strip"
PARSER_POST_LINK_TOOL_SHA256 = (
    "4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d"
)
PARSER_POST_LINK_TOOL_VERSION = "GNU strip (GNU Binutils for Ubuntu) 2.42"
PARSER_POST_LINK_COMMAND = (
    "/usr/bin/x86_64-linux-gnu-strip --strip-unneeded "
    "nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
)
PARSER_POST_LINK_COMMAND_SHA256 = (
    "5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca"
)
PARSER_POST_LINK_REMOVED_SECTIONS = (".symtab", ".strtab")
PARSER_POST_LINK_RETAINED_SECTIONS = (".dynsym", ".dynstr")
KERNEL_PROOF_SCHEMA_VERSION = "deepsafe.ds9-cuda-kernel-proof/v1"
DOCKERFILE = Path("deepstream/Dockerfile")
DOCKERIGNORE = Path(".dockerignore")
CONTROLLER = Path("validation/ds9_parser_bootstrap.py")
RUNTIME_CONTROLLER = Path("validation/ds9_runtime_compatibility.py")
CONTROL_MANIFEST = Path("deepstream/runtime-control-manifest.json")
PARSER_EXPORT = Path("parser/libnvdsinfer_custom_impl_Yolo.so")
PARSER_DIGEST_EXPORT = Path("parser/parser.sha256")
SOURCE_LINEAGE_EXPORT = Path("parser/source-lineage.json")

SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
NONCE_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_TAG_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9._-]+")


class ParserBootstrapError(RuntimeError):
    """Raised when parser build evidence is not reproducible or immutable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ParserBootstrapError(f"{label} must be an RFC3339 timestamp")
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise ParserBootstrapError(f"{label} must be an RFC3339 timestamp") from exc
    if result.tzinfo is None:
        raise ParserBootstrapError(f"{label} must include a timezone")
    return result.astimezone(timezone.utc)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside_root(path: Path, project_root: Path) -> Path:
    root = project_root.resolve()
    candidate = root / path if not path.is_absolute() else path
    resolved = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ParserBootstrapError(f"path leaves project root: {resolved}") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ParserBootstrapError(f"path contains a symlink: {current}")
    return resolved


def _relative(path: Path, project_root: Path) -> str:
    return _inside_root(path, project_root).relative_to(project_root.resolve()).as_posix()


def file_pin(path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ParserBootstrapError(f"cannot open pinned file: {resolved}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ParserBootstrapError(f"pinned path is not a regular file: {resolved}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_mode,
        item.st_nlink,
    )
    if identity(before) != identity(after):
        raise ParserBootstrapError(f"file changed while hashing: {resolved}")
    return {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_json(
    path: Path, *, project_root: Path, immutable: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ParserBootstrapError(f"cannot open JSON: {resolved}") from exc
    content = bytearray()
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ParserBootstrapError("JSON must be a regular file")
        if before.st_size < 0 or before.st_size > MAX_JSON_BYTES:
            raise ParserBootstrapError("JSON exceeds its byte limit")
        if immutable and (
            stat.S_IMODE(before.st_mode) != 0o440 or before.st_nlink != 1
        ):
            raise ParserBootstrapError(
                "receipt must be regular, mode 0440, one hard link"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            content.extend(block)
            digest.update(block)
            if len(content) > MAX_JSON_BYTES:
                raise ParserBootstrapError("JSON exceeds its byte limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_mode,
        item.st_nlink,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise ParserBootstrapError("JSON changed while reading")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ParserBootstrapError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParserBootstrapError(f"invalid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ParserBootstrapError("JSON root must be an object")
    pin = {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }
    return payload, pin


def _read_exact_digest(path: Path, *, project_root: Path) -> str:
    """Read one lowercase SHA-256 line through a stable, no-follow FD."""

    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ParserBootstrapError("cannot open exported parser digest") from exc
    content = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 65:
            raise ParserBootstrapError("exported parser digest file is malformed")
        while True:
            block = os.read(descriptor, 128)
            if not block:
                break
            content.extend(block)
            if len(content) > 65:
                raise ParserBootstrapError("exported parser digest file is oversized")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_mode,
        item.st_nlink,
    )
    if identity(before) != identity(after):
        raise ParserBootstrapError("exported parser digest changed while reading")
    try:
        rendered = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ParserBootstrapError("exported parser digest is not ASCII") from exc
    if not rendered.endswith("\n") or rendered.count("\n") != 1:
        raise ParserBootstrapError("exported parser digest must be one newline-terminated line")
    digest = rendered[:-1]
    if SHA256_RE.fullmatch(digest) is None:
        raise ParserBootstrapError("exported parser digest content is invalid")
    return digest


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if exclusive:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o440,
        )
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                if written <= 0:
                    raise OSError("short receipt write")
                offset += written
            os.fchmod(descriptor, 0o440)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _reject_placeholder_hex(value: str, label: str, *, image_prefix: bool) -> str:
    raw = value.removeprefix("sha256:") if image_prefix else value
    pattern = IMAGE_ID_RE if image_prefix else SHA256_RE
    if pattern.fullmatch(value) is None:
        raise ParserBootstrapError(f"{label} is not an exact lowercase SHA-256")
    if len(set(raw)) < 4 or raw in {"0" * 64, "f" * 64}:
        raise ParserBootstrapError(f"{label} looks like a placeholder")
    return value


def validate_base_ref(base_ref: str) -> tuple[str, str]:
    prefix = f"{BASE_TAG}@"
    if not base_ref.startswith(prefix):
        raise ParserBootstrapError("base ref must pin the exact DeepStream 9 tag by digest")
    digest = _reject_placeholder_hex(
        base_ref[len(prefix) :], "base image digest", image_prefix=True
    )
    return base_ref, digest


def _input_pins(project_root: Path) -> dict[str, dict[str, Any]]:
    pins = {
        "dockerfile": file_pin(project_root / DOCKERFILE, project_root=project_root),
        "dockerignore": file_pin(project_root / DOCKERIGNORE, project_root=project_root),
        "source_patch": file_pin(
            project_root / SOURCE_PATCH, project_root=project_root
        ),
        "bootstrap_controller": file_pin(
            project_root / CONTROLLER, project_root=project_root
        ),
        "runtime_controller": file_pin(
            project_root / RUNTIME_CONTROLLER, project_root=project_root
        ),
        "runtime_control_manifest": file_pin(
            project_root / CONTROL_MANIFEST, project_root=project_root
        ),
    }
    if pins["source_patch"]["sha256"] != SOURCE_PATCH_SHA256:
        raise ParserBootstrapError("live CUDA kernel proof patch SHA differs")
    return pins


def build_operating_policy() -> dict[str, Any]:
    """Return the exact workstation-managed parser-build operating policy."""

    return {
        "schema_version": BUILD_POLICY_SCHEMA_VERSION,
        "id": BUILD_OPERATING_POLICY_ID,
        "revision": 1,
        "hardware_protection_owner": "workstation_bios_ec_gpu_driver",
        "temperature_threshold_enforcement": "informational_only",
        "telemetry_read_failure": "abort_build",
        "sample_interval_seconds": BUILD_THERMAL_SAMPLE_SECONDS,
        "telemetry_channels": {
            "platform_temperature_c": "required",
            "gpu_temperature_c": "required",
            "gpu_power": "record",
            "gpu_slowdown": "record",
        },
        "process_group_termination_on_telemetry_failure": [
            "SIGTERM",
            "SIGKILL_after_10s",
        ],
        "docker_gpu_device_request": False,
        "gpu_telemetry_query_is_read_only": True,
    }


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ParserBootstrapError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ParserBootstrapError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ParserBootstrapError(f"{label} must be a finite number")
    return number


def _optional_nonnegative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParserBootstrapError(f"{label} must be a stored numeric value")
    return _finite_number(value, label, nonnegative=True)


def _stored_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParserBootstrapError(f"{label} must be a stored numeric value")
    return _finite_number(value, label)


def _optional_gpu_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered or rendered.upper() in {"N/A", "[N/A]"}:
        return None
    return _finite_number(rendered, label, nonnegative=True)


def _gpu_flag(value: Any, label: str) -> bool:
    rendered = str(value).strip().casefold()
    if rendered in {"active", "yes", "true", "1"}:
        return True
    if rendered in {"not active", "no", "false", "0"}:
        return False
    raise ParserBootstrapError(f"{label} is unreadable")


def _validate_gpu_telemetry(payload: Any) -> tuple[int, str]:
    expected_keys = {
        "sample_timestamp",
        "gpu_index",
        "gpu_name",
        "temperature_c",
        "power_draw_w",
        "power_limits_w",
        "power_limit_telemetry_complete",
        "pstate",
        "clock_event_reasons_active_mask",
        "clock_event_sw_power_cap",
        "slowdown_flags",
        "dangerous_slowdown_active",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ParserBootstrapError("GPU telemetry fields are malformed")
    timestamp = payload.get("sample_timestamp")
    name = payload.get("gpu_name")
    index = payload.get("gpu_index")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ParserBootstrapError("GPU telemetry timestamp is unreadable")
    if not isinstance(name, str) or not name.strip():
        raise ParserBootstrapError("GPU identity name is unreadable")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ParserBootstrapError("GPU identity index is malformed")
    _stored_finite_number(payload.get("temperature_c"), "GPU temperature")
    _optional_nonnegative_number(payload.get("power_draw_w"), "GPU power draw")

    power_limits = payload.get("power_limits_w")
    if not isinstance(power_limits, dict) or set(power_limits) != {
        "requested",
        "current",
        "default",
    }:
        raise ParserBootstrapError("GPU power-limit telemetry is malformed")
    for field in ("requested", "current", "default"):
        _optional_nonnegative_number(
            power_limits.get(field), f"GPU {field} power limit"
        )
    complete = payload.get("power_limit_telemetry_complete")
    if not isinstance(complete, bool) or complete != (
        power_limits.get("current") is not None
        and power_limits.get("default") is not None
    ):
        raise ParserBootstrapError("GPU power-limit completeness differs")

    pstate = payload.get("pstate")
    active_mask = payload.get("clock_event_reasons_active_mask")
    if not isinstance(pstate, str) or not pstate.strip():
        raise ParserBootstrapError("GPU pstate telemetry is unreadable")
    if not isinstance(active_mask, str) or not active_mask.strip():
        raise ParserBootstrapError("GPU clock-event mask is unreadable")
    if not isinstance(payload.get("clock_event_sw_power_cap"), bool):
        raise ParserBootstrapError("GPU software power-cap flag is malformed")

    flags = payload.get("slowdown_flags")
    if not isinstance(flags, dict) or set(flags) != set(BUILD_SLOWDOWN_FLAG_NAMES):
        raise ParserBootstrapError("GPU slowdown telemetry fields are malformed")
    if any(not isinstance(flags.get(field), bool) for field in BUILD_SLOWDOWN_FLAG_NAMES):
        raise ParserBootstrapError("GPU slowdown telemetry is malformed")
    dangerous = payload.get("dangerous_slowdown_active")
    if not isinstance(dangerous, bool) or dangerous != any(flags.values()):
        raise ParserBootstrapError("GPU slowdown aggregate differs")
    return index, name.strip()


def _validate_build_sample(
    payload: Any, *, expected_gpu_identity: tuple[int, str] | None = None
) -> tuple[int, str]:
    expected_keys = {
        "phase",
        "sampled_at_utc",
        "temperatures_c",
        "max_temperature_c",
        "source_manifest",
        "gpu_telemetry",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ParserBootstrapError("parser build telemetry sample is malformed")
    if payload.get("phase") not in {"preflight", "runtime", "post-run"}:
        raise ParserBootstrapError("parser build telemetry phase is malformed")
    _parse_time(payload.get("sampled_at_utc"), "build sample sampled_at_utc")
    if not isinstance(payload.get("source_manifest"), dict):
        raise ParserBootstrapError("platform thermal source manifest is malformed")

    temperatures = payload.get("temperatures_c")
    if not isinstance(temperatures, dict) or not temperatures:
        raise ParserBootstrapError("platform/GPU temperatures are unavailable")
    numeric_temperatures: dict[str, float] = {}
    for name, value in temperatures.items():
        if not isinstance(name, str) or not name:
            raise ParserBootstrapError("temperature source identity is malformed")
        numeric_temperatures[name] = _stored_finite_number(
            value, f"temperature source {name}"
        )
    maximum = _stored_finite_number(
        payload.get("max_temperature_c"), "maximum observed temperature"
    )
    if maximum != max(numeric_temperatures.values()):
        raise ParserBootstrapError("maximum observed temperature differs")

    identity = _validate_gpu_telemetry(payload.get("gpu_telemetry"))
    gpu_key = f"gpu_{identity[0]}_c"
    gpu_temperature = _stored_finite_number(
        payload["gpu_telemetry"].get("temperature_c"), "GPU temperature"
    )
    if numeric_temperatures.get(gpu_key) != gpu_temperature:
        raise ParserBootstrapError("GPU temperature/source binding differs")
    if expected_gpu_identity is not None and identity != expected_gpu_identity:
        raise ParserBootstrapError("GPU identity changed during parser build")
    return identity


def platform_thermal_snapshot() -> dict[str, Any]:
    """Read platform and GPU temperature/power/slowdown telemetry."""

    from validation.scene_benchmark.run_matrix import (
        discover_platform_thermal_sources,
        gpu_row_snapshot,
        query_gpu_row,
        read_platform_thermal_row,
    )

    manifest = discover_platform_thermal_sources()
    row, errors = read_platform_thermal_row(manifest, utc_now())
    columns = list(manifest.get("columns", []))
    values: dict[str, float] = {}
    for column, raw in zip(columns[1:], row[1:]):
        if not column.endswith("_c"):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{column}: missing/non-numeric")
            continue
        if not math.isfinite(value):
            errors.append(f"{column}: non-finite")
            continue
        values[column] = value
    if not manifest.get("available") or not values or errors:
        raise ParserBootstrapError(
            "platform thermal sources are unavailable/incomplete: " + "; ".join(errors)
        )

    try:
        gpu_snapshot = gpu_row_snapshot(query_gpu_row(0))
        gpu_index = int(gpu_snapshot["gpu_index"])
        gpu_name = gpu_snapshot["gpu_name"].strip()
        gpu_temperature = _finite_number(
            gpu_snapshot["temperature_c"], "GPU temperature"
        )
        slowdown_flags = {
            name: _gpu_flag(gpu_snapshot.get(name), name)
            for name in BUILD_SLOWDOWN_FLAG_NAMES
        }
        power_limits = {
            "requested": _optional_gpu_number(
                gpu_snapshot.get("power_requested_limit_w"),
                "GPU requested power limit",
            ),
            "current": _optional_gpu_number(
                gpu_snapshot.get("power_current_limit_w"),
                "GPU current power limit",
            ),
            "default": _optional_gpu_number(
                gpu_snapshot.get("power_default_limit_w"),
                "GPU default power limit",
            ),
        }
        gpu_telemetry = {
            "sample_timestamp": str(gpu_snapshot.get("timestamp", "")).strip(),
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "temperature_c": gpu_temperature,
            "power_draw_w": _optional_gpu_number(
                gpu_snapshot.get("power_draw_w"), "GPU power draw"
            ),
            "power_limits_w": power_limits,
            "power_limit_telemetry_complete": (
                power_limits["current"] is not None
                and power_limits["default"] is not None
            ),
            "pstate": str(gpu_snapshot.get("pstate", "")).strip(),
            "clock_event_reasons_active_mask": str(
                gpu_snapshot.get("clock_event_reasons_active_mask", "")
            ).strip(),
            "clock_event_sw_power_cap": _gpu_flag(
                gpu_snapshot.get("clock_event_sw_power_cap"),
                "clock_event_sw_power_cap",
            ),
            "slowdown_flags": slowdown_flags,
            "dangerous_slowdown_active": any(slowdown_flags.values()),
        }
        _validate_gpu_telemetry(gpu_telemetry)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ParserBootstrapError(
            f"GPU temperature/power/slowdown telemetry is unavailable: {exc}"
        ) from exc
    values[f"gpu_{gpu_index}_c"] = gpu_temperature
    return {
        "sampled_at_utc": row[0],
        "temperatures_c": values,
        "max_temperature_c": max(values.values()),
        "source_manifest": manifest,
        "gpu_telemetry": gpu_telemetry,
    }


def discovery_command(base_ref: str, session_root: Path, project_root: Path) -> list[str]:
    _, digest = validate_base_ref(base_ref)
    session = _inside_root(session_root, project_root)
    destination = _relative(session / "pass-1-discovery/export", project_root)
    return [
        "docker",
        "build",
        "--file",
        DOCKERFILE.as_posix(),
        "--target",
        "parser-audit-export",
        "--no-cache",
        "--pull=false",
        "--progress=plain",
        "--build-arg",
        f"DEEPSTREAM_BASE_REF={base_ref}",
        "--build-arg",
        f"DEEPSTREAM_BASE_DIGEST={digest}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_COMMIT={SOURCE_COMMIT}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_TREE={SOURCE_TREE}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCH_SHA256={SOURCE_PATCH_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256={UPSTREAM_SOURCE_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256={PATCHED_SOURCE_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCHED_TREE={PATCHED_SOURCE_TREE}",
        "--output",
        f"type=local,dest={destination}",
        ".",
    ]


def production_command(
    *,
    base_ref: str,
    parser_sha256: str,
    image_tag: str,
    session_root: Path,
    project_root: Path,
) -> list[str]:
    _, digest = validate_base_ref(base_ref)
    _reject_placeholder_hex(parser_sha256, "measured parser SHA", image_prefix=False)
    if IMAGE_TAG_RE.fullmatch(image_tag) is None or "@" in image_tag:
        raise ParserBootstrapError("production image tag is invalid")
    pins = _input_pins(project_root)
    iidfile = _relative(
        _inside_root(session_root, project_root) / "pass-2-production/image-id.txt",
        project_root,
    )
    return [
        "docker",
        "build",
        "--file",
        DOCKERFILE.as_posix(),
        "--target",
        "runtime",
        "--no-cache",
        "--pull=false",
        "--progress=plain",
        "--iidfile",
        iidfile,
        "--tag",
        image_tag,
        "--build-arg",
        f"DEEPSTREAM_BASE_REF={base_ref}",
        "--build-arg",
        f"DEEPSTREAM_BASE_DIGEST={digest}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_COMMIT={SOURCE_COMMIT}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_TREE={SOURCE_TREE}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCH_SHA256={SOURCE_PATCH_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256={UPSTREAM_SOURCE_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256={PATCHED_SOURCE_SHA256}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PATCHED_TREE={PATCHED_SOURCE_TREE}",
        "--build-arg",
        f"DEEPSTREAM_YOLO_PARSER_SHA256={parser_sha256}",
        "--build-arg",
        "DEEPSAFE_RUNTIME_CONTROLLER_SHA256="
        + pins["runtime_controller"]["sha256"],
        "--build-arg",
        "DEEPSAFE_RUNTIME_CONTROL_MANIFEST_SHA256="
        + pins["runtime_control_manifest"]["sha256"],
        "--build-arg",
        "DEEPSAFE_DOCKERIGNORE_SHA256=" + pins["dockerignore"]["sha256"],
        ".",
    ]


def make_plan(
    *,
    base_ref: str,
    image_tag: str,
    session_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    base_ref, digest = validate_base_ref(base_ref)
    if IMAGE_TAG_RE.fullmatch(image_tag) is None or "@" in image_tag:
        raise ParserBootstrapError("production image tag is invalid")
    session = _inside_root(session_root, project_root)
    command = discovery_command(base_ref, session, project_root)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "pending_explicit_discovery",
        "production_ready": False,
        "created_at_utc": utc_now(),
        "base_ref": base_ref,
        "base_digest": digest,
        "source": source_lineage_contract(),
        "session_root": _relative(session, project_root),
        "image_tag": image_tag,
        "inputs": _input_pins(project_root),
        "pass_1": {
            "purpose": "measure_parser_sha_from_exported_binary",
            "accepts_expected_parser_sha": False,
            "command": command,
            "command_sha256": canonical_sha256(command),
        },
        "pass_2": {
            "purpose": "rebuild_and_require_pass_1_measured_parser_sha",
            "parser_sha_source": "immutable_pass_1_discovery_receipt_only",
            "command": None,
        },
        "docker_called": False,
        "gpu_requested": False,
        "inference_started": False,
    }


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    thermal_path: Path,
    project_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    thermal_probe: Callable[[], dict[str, Any]] = platform_thermal_snapshot,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["DOCKER_BUILDKIT"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    gpu_identity: tuple[int, str] | None = None

    def sample(*, phase: str) -> dict[str, Any]:
        nonlocal gpu_identity
        observed = thermal_probe()
        if not isinstance(observed, dict):
            raise ParserBootstrapError("build telemetry probe returned no object")
        entry = {
            "phase": phase,
            "sampled_at_utc": observed.get("sampled_at_utc"),
            "temperatures_c": observed.get("temperatures_c"),
            "max_temperature_c": observed.get("max_temperature_c"),
            "source_manifest": observed.get("source_manifest"),
            "gpu_telemetry": observed.get("gpu_telemetry"),
        }
        gpu_identity = _validate_build_sample(
            entry, expected_gpu_identity=gpu_identity
        )
        samples.append(entry)
        return entry

    def write_report(*, status: str, abort_reason: str | None) -> dict[str, Any]:
        report = {
            "schema_version": BUILD_TELEMETRY_SCHEMA_VERSION,
            "status": status,
            "policy": build_operating_policy(),
            "max_observed_temperature_c": (
                max(item["max_temperature_c"] for item in samples)
                if samples
                else None
            ),
            "abort_reason": abort_reason,
            "samples": samples,
        }
        _write_json(thermal_path, report, exclusive=False)
        return report

    try:
        sample(phase="preflight")
    except Exception as exc:
        abort_reason = f"telemetry_error={type(exc).__name__}: {exc}"
        write_report(
            status="telemetry_unavailable_before_start",
            abort_reason=abort_reason,
        )
        raise ParserBootstrapError(
            "parser build preflight telemetry is unavailable or malformed: "
            + abort_reason
        ) from exc

    returncode: int
    aborted = False
    abort_reason: str | None = None
    if runner is not None:
        completed = runner(
            list(command),
            cwd=project_root,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        returncode = completed.returncode
        try:
            sample(phase="post-run")
        except Exception as exc:
            aborted = True
            abort_reason = f"telemetry_error={type(exc).__name__}: {exc}"
    else:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = popen_factory(
                list(command),
                cwd=project_root,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def terminate_build() -> None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait(timeout=5)

            while process.poll() is None:
                sleeper(BUILD_THERMAL_SAMPLE_SECONDS)
                try:
                    sample(phase="runtime")
                except Exception as exc:
                    aborted = True
                    abort_reason = f"telemetry_error={type(exc).__name__}: {exc}"
                    terminate_build()
                    break
            returncode = int(process.returncode if process.returncode is not None else -1)
        if abort_reason is None:
            try:
                sample(phase="post-run")
            except Exception as exc:
                aborted = True
                abort_reason = f"telemetry_error={type(exc).__name__}: {exc}"

    telemetry_report = write_report(
        status=(
            "telemetry_abort"
            if aborted
            else ("complete" if returncode == 0 else "build_failed")
        ),
        abort_reason=abort_reason,
    )
    if aborted:
        raise ParserBootstrapError(
            "parser build aborted on telemetry loss or malformed evidence: "
            + str(abort_reason)
        )
    if returncode != 0:
        raise ParserBootstrapError(f"Docker build failed with exit {returncode}")
    return telemetry_report


def source_lineage_contract() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "patched_source_path": PATCHED_SOURCE_PATH,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "patch_path": SOURCE_PATCH.as_posix(),
        "patch_sha256": SOURCE_PATCH_SHA256,
        "patched_source_sha256": PATCHED_SOURCE_SHA256,
        "patched_tree": PATCHED_SOURCE_TREE,
        "build_makefile_path": PARSER_BUILD_MAKEFILE_PATH,
        "build_makefile_sha256": PARSER_BUILD_MAKEFILE_SHA256,
        "build_makefile_modified": False,
        "cuda_cubin_architecture": PARSER_CUDA_CUBIN_ARCHITECTURE,
        "cuda_ptx_architecture": PARSER_CUDA_PTX_ARCHITECTURE,
        "cuda_gencode_flags": list(PARSER_CUDA_GENCODE_FLAGS),
        "instrumentation_schema": KERNEL_PROOF_SCHEMA_VERSION,
        "build_command": PARSER_BUILD_COMMAND,
        "build_command_sha256": PARSER_BUILD_COMMAND_SHA256,
        "post_link_tool_path": PARSER_POST_LINK_TOOL_PATH,
        "post_link_tool_sha256": PARSER_POST_LINK_TOOL_SHA256,
        "post_link_tool_version": PARSER_POST_LINK_TOOL_VERSION,
        "post_link_command": PARSER_POST_LINK_COMMAND,
        "post_link_command_sha256": PARSER_POST_LINK_COMMAND_SHA256,
        "post_link_removed_sections": list(PARSER_POST_LINK_REMOVED_SECTIONS),
        "post_link_retained_sections": list(PARSER_POST_LINK_RETAINED_SECTIONS),
    }


def _validate_source_lineage(payload: Mapping[str, Any]) -> None:
    expected = source_lineage_contract()
    if dict(payload) != expected:
        raise ParserBootstrapError("exported parser source lineage differs")


def _validate_thermal_report(
    path: Path, *, project_root: Path
) -> dict[str, Any]:
    payload, _ = _read_json(path, project_root=project_root, immutable=False)
    if set(payload) != {
        "schema_version",
        "status",
        "policy",
        "max_observed_temperature_c",
        "abort_reason",
        "samples",
    }:
        raise ParserBootstrapError("parser build telemetry report is malformed")
    if payload.get("schema_version") != BUILD_TELEMETRY_SCHEMA_VERSION:
        raise ParserBootstrapError("parser build telemetry schema differs")
    if (
        payload.get("status") != "complete"
        or payload.get("abort_reason") is not None
        or payload.get("policy") != build_operating_policy()
    ):
        raise ParserBootstrapError("parser build telemetry status/policy differs")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ParserBootstrapError("parser build telemetry coverage is incomplete")
    if samples[0].get("phase") != "preflight" or samples[-1].get("phase") != "post-run":
        raise ParserBootstrapError("parser build telemetry phase coverage differs")
    gpu_identity: tuple[int, str] | None = None
    maxima: list[float] = []
    for item in samples:
        gpu_identity = _validate_build_sample(
            item, expected_gpu_identity=gpu_identity
        )
        maxima.append(
            _stored_finite_number(
                item.get("max_temperature_c"), "maximum observed temperature"
            )
        )
    observed_maximum = _stored_finite_number(
        payload.get("max_observed_temperature_c"),
        "report maximum observed temperature",
    )
    if observed_maximum != max(maxima):
        raise ParserBootstrapError("parser build telemetry maximum differs")
    return payload


def execute_discovery(
    *,
    base_ref: str,
    image_tag: str,
    session_root: Path,
    project_root: Path = PROJECT_ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    thermal_probe: Callable[[], dict[str, Any]] = platform_thermal_snapshot,
) -> dict[str, Any]:
    session = _inside_root(session_root, project_root)
    pass_root = session / "pass-1-discovery"
    if pass_root.exists():
        raise ParserBootstrapError("pass-1 output already exists; nonce/session is single-use")
    pass_root.mkdir(parents=True, exist_ok=False)
    command = discovery_command(base_ref, session, project_root)
    log_path = pass_root / "docker-build.log"
    thermal_path = pass_root / "platform-thermal.json"
    thermal = _run_logged(
        command,
        log_path=log_path,
        thermal_path=thermal_path,
        project_root=project_root,
        runner=runner,
        thermal_probe=thermal_probe,
    )

    export = pass_root / "export"
    parser_path = export / PARSER_EXPORT
    digest_path = export / PARSER_DIGEST_EXPORT
    source_path = export / SOURCE_LINEAGE_EXPORT
    parser_pin = file_pin(parser_path, project_root=project_root)
    if parser_pin["bytes"] <= 0:
        raise ParserBootstrapError("exported parser ELF is empty")
    exported_digest = _read_exact_digest(digest_path, project_root=project_root)
    if exported_digest != parser_pin["sha256"]:
        raise ParserBootstrapError("exported parser digest does not match binary")
    _reject_placeholder_hex(exported_digest, "measured parser SHA", image_prefix=False)
    source_payload, _ = _read_json(
        source_path, project_root=project_root, immutable=False
    )
    _validate_source_lineage(source_payload)
    base_ref, base_digest = validate_base_ref(base_ref)
    created = _parse_time(utc_now(), "created_at_utc")
    receipt = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "status": "parser_sha_measured",
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (created + MAX_RECEIPT_AGE)
        .isoformat()
        .replace("+00:00", "Z"),
        "base_ref": base_ref,
        "base_digest": base_digest,
        "source": dict(source_payload),
        "session_root": _relative(session, project_root),
        "image_tag": image_tag,
        "pass_number": 1,
        "expected_parser_sha_input": None,
        "measured_parser_sha256": exported_digest,
        "command": command,
        "command_sha256": canonical_sha256(command),
        "environment": {"DOCKER_BUILDKIT": "1", "gpu_requested": False},
        "thermal_policy": thermal["policy"],
        "inputs": _input_pins(project_root),
        "artifacts": {
            "parser": parser_pin,
            "parser_digest": file_pin(digest_path, project_root=project_root),
            "source_lineage": file_pin(source_path, project_root=project_root),
            "raw_build_log": file_pin(log_path, project_root=project_root),
            "platform_thermal": file_pin(thermal_path, project_root=project_root),
        },
    }
    _write_json(pass_root / "discovery-receipt.json", receipt, exclusive=True)
    return receipt


def validate_discovery_receipt(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, receipt_pin = _read_json(path, project_root=project_root, immutable=True)
    if payload.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        raise ParserBootstrapError("discovery receipt schema differs")
    if payload.get("status") != "parser_sha_measured" or payload.get("pass_number") != 1:
        raise ParserBootstrapError("discovery receipt status/pass differs")
    created = _parse_time(payload.get("created_at_utc"), "created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= created or expires - created > MAX_RECEIPT_AGE:
        raise ParserBootstrapError("discovery receipt lifetime exceeds 24h")
    if current < created or current >= expires:
        raise ParserBootstrapError("discovery receipt is not current")
    base_ref, digest = validate_base_ref(str(payload.get("base_ref")))
    if payload.get("base_digest") != digest:
        raise ParserBootstrapError("discovery base ref/digest differ")
    if payload.get("expected_parser_sha_input") is not None:
        raise ParserBootstrapError("pass-1 illegally accepted an expected parser SHA")
    session = _inside_root(Path(str(payload.get("session_root"))), project_root)
    command = discovery_command(base_ref, session, project_root)
    if payload.get("command") != command or payload.get("command_sha256") != canonical_sha256(command):
        raise ParserBootstrapError("discovery command contract differs")
    if payload.get("inputs") != _input_pins(project_root):
        raise ParserBootstrapError("discovery build inputs changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "parser",
        "parser_digest",
        "source_lineage",
        "raw_build_log",
        "platform_thermal",
    }:
        raise ParserBootstrapError("discovery artifact set differs")
    live: dict[str, dict[str, Any]] = {}
    for name, expected in artifacts.items():
        if not isinstance(expected, dict):
            raise ParserBootstrapError("discovery artifact pin is malformed")
        observed = file_pin(project_root / str(expected.get("path", "")), project_root=project_root)
        if observed != expected:
            raise ParserBootstrapError(f"discovery artifact changed: {name}")
        live[name] = observed
    thermal = _validate_thermal_report(
        project_root / live["platform_thermal"]["path"], project_root=project_root
    )
    if payload.get("thermal_policy") != thermal["policy"]:
        raise ParserBootstrapError("discovery thermal policy binding differs")
    measured = str(payload.get("measured_parser_sha256"))
    _reject_placeholder_hex(measured, "measured parser SHA", image_prefix=False)
    if live["parser"]["sha256"] != measured:
        raise ParserBootstrapError("receipt parser SHA differs from exported ELF")
    digest_content = _read_exact_digest(
        project_root / live["parser_digest"]["path"], project_root=project_root
    )
    if digest_content != measured:
        raise ParserBootstrapError("receipt parser digest content differs from ELF")
    source_payload, _ = _read_json(
        project_root / live["source_lineage"]["path"],
        project_root=project_root,
        immutable=False,
    )
    _validate_source_lineage(source_payload)
    if payload.get("source") != source_payload:
        raise ParserBootstrapError("receipt source lineage differs from export")
    return payload, receipt_pin


def execute_production(
    *,
    discovery_receipt: Path,
    image_tag: str,
    project_root: Path = PROJECT_ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    thermal_probe: Callable[[], dict[str, Any]] = platform_thermal_snapshot,
) -> dict[str, Any]:
    discovery, discovery_pin = validate_discovery_receipt(
        discovery_receipt, project_root=project_root
    )
    if image_tag != discovery.get("image_tag"):
        raise ParserBootstrapError("production image tag differs from pass-1 receipt")
    session = _inside_root(Path(discovery["session_root"]), project_root)
    pass_root = session / "pass-2-production"
    if pass_root.exists():
        raise ParserBootstrapError("pass-2 output already exists; session is single-use")
    pass_root.mkdir(parents=True, exist_ok=False)
    command = production_command(
        base_ref=discovery["base_ref"],
        parser_sha256=discovery["measured_parser_sha256"],
        image_tag=image_tag,
        session_root=session,
        project_root=project_root,
    )
    log_path = pass_root / "docker-build.log"
    thermal_path = pass_root / "platform-thermal.json"
    thermal = _run_logged(
        command,
        log_path=log_path,
        thermal_path=thermal_path,
        project_root=project_root,
        runner=runner,
        thermal_probe=thermal_probe,
    )
    iid_path = pass_root / "image-id.txt"
    try:
        image_id = iid_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ParserBootstrapError("production build did not write image ID") from exc
    _reject_placeholder_hex(image_id, "production image ID", image_prefix=True)
    inspect_runner = runner or subprocess.run
    inspected = inspect_runner(
        ["docker", "image", "inspect", image_id],
        cwd=project_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    inspect_path = pass_root / "docker-image-inspect.json"
    inspect_path.write_text(inspected.stdout or "", encoding="utf-8")
    if inspected.returncode != 0:
        raise ParserBootstrapError("production image inspect failed")
    try:
        values = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise ParserBootstrapError("production image inspect JSON is invalid") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ParserBootstrapError("production image inspect must contain one image")
    image = values[0]
    labels = (image.get("Config") or {}).get("Labels")
    expected_parser = discovery["measured_parser_sha256"]
    pins = _input_pins(project_root)
    expected_labels = {
        "com.deepsafe.build-lineage.schema": BUILD_LINEAGE_SCHEMA_VERSION,
        "com.deepsafe.deepstream-yolo.parser-sha256": expected_parser,
        "com.deepsafe.deepstream.base-ref": discovery["base_ref"],
        "com.deepsafe.deepstream.base-digest": discovery["base_digest"],
        "com.deepsafe.deepstream-yolo.patch-sha256": SOURCE_PATCH_SHA256,
        "com.deepsafe.deepstream-yolo.upstream-source-sha256": UPSTREAM_SOURCE_SHA256,
        "com.deepsafe.deepstream-yolo.patched-source-sha256": PATCHED_SOURCE_SHA256,
        "com.deepsafe.deepstream-yolo.patched-tree": PATCHED_SOURCE_TREE,
        "com.deepsafe.deepstream-yolo.parser-build-makefile-sha256": PARSER_BUILD_MAKEFILE_SHA256,
        "com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture": PARSER_CUDA_CUBIN_ARCHITECTURE,
        "com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture": PARSER_CUDA_PTX_ARCHITECTURE,
        "com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags": ";".join(PARSER_CUDA_GENCODE_FLAGS),
        "com.deepsafe.deepstream-yolo.parser-build-command-sha256": PARSER_BUILD_COMMAND_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-path": PARSER_POST_LINK_TOOL_PATH,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256": PARSER_POST_LINK_TOOL_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-version": PARSER_POST_LINK_TOOL_VERSION,
        "com.deepsafe.deepstream-yolo.parser-post-link-command": PARSER_POST_LINK_COMMAND,
        "com.deepsafe.deepstream-yolo.parser-post-link-command-sha256": PARSER_POST_LINK_COMMAND_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-removed-sections": ";".join(
            PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        "com.deepsafe.deepstream-yolo.parser-post-link-retained-sections": ";".join(
            PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        "com.deepsafe.cuda-kernel-proof.schema": KERNEL_PROOF_SCHEMA_VERSION,
        "com.deepsafe.dockerignore.sha256": pins["dockerignore"]["sha256"],
        "com.deepsafe.runtime-compatibility-controller.sha256": pins[
            "runtime_controller"
        ]["sha256"],
        "com.deepsafe.runtime-control-manifest.sha256": pins[
            "runtime_control_manifest"
        ]["sha256"],
    }
    if image.get("Id") != image_id or not isinstance(labels, dict):
        raise ParserBootstrapError("production image metadata differs")
    for label, expected in expected_labels.items():
        if labels.get(label) != expected:
            raise ParserBootstrapError(f"production image label differs: {label}")
    created = _parse_time(utc_now(), "created_at_utc")
    receipt = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "status": "candidate_image_built",
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (created + MAX_RECEIPT_AGE)
        .isoformat()
        .replace("+00:00", "Z"),
        "pass_number": 2,
        "two_pass_complete": True,
        "session_root": discovery["session_root"],
        "image_tag": image_tag,
        "resolved_image_id": image_id,
        "base_ref": discovery["base_ref"],
        "base_digest": discovery["base_digest"],
        "parser_sha256": expected_parser,
        "parser_sha_source": discovery_pin,
        "discovery_receipt": discovery_pin,
        "source": discovery["source"],
        "command": command,
        "command_sha256": canonical_sha256(command),
        "environment": {"DOCKER_BUILDKIT": "1", "gpu_requested": False},
        "thermal_policy": thermal["policy"],
        "inputs": pins,
        "image": {
            "id": image_id,
            "architecture": image.get("Architecture"),
            "os": image.get("Os"),
            "repo_digests": image.get("RepoDigests") or [],
            "labels": labels,
        },
        "artifacts": {
            "raw_build_log": file_pin(log_path, project_root=project_root),
            "image_id": file_pin(iid_path, project_root=project_root),
            "raw_image_inspect": file_pin(inspect_path, project_root=project_root),
            "platform_thermal": file_pin(thermal_path, project_root=project_root),
        },
    }
    _write_json(pass_root / "production-build-receipt.json", receipt, exclusive=True)
    return receipt


def validate_production_receipt(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    resolved_image_id: str | None = None,
    parser_sha256: str | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, pin = _read_json(path, project_root=project_root, immutable=True)
    if payload.get("schema_version") != PRODUCTION_SCHEMA_VERSION:
        raise ParserBootstrapError("production build receipt schema differs")
    if payload.get("status") != "candidate_image_built" or payload.get("pass_number") != 2:
        raise ParserBootstrapError("production build receipt status/pass differs")
    if payload.get("two_pass_complete") is not True:
        raise ParserBootstrapError("production build was not two-pass")
    created = _parse_time(payload.get("created_at_utc"), "created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= created or expires - created > MAX_RECEIPT_AGE or not (created <= current < expires):
        raise ParserBootstrapError("production build receipt is stale or overlong")
    image_id = str(payload.get("resolved_image_id"))
    _reject_placeholder_hex(image_id, "production image ID", image_prefix=True)
    if resolved_image_id is not None and image_id != resolved_image_id:
        raise ParserBootstrapError("production build receipt image differs")
    measured = str(payload.get("parser_sha256"))
    _reject_placeholder_hex(measured, "production parser SHA", image_prefix=False)
    if parser_sha256 is not None and measured != parser_sha256:
        raise ParserBootstrapError("production build receipt parser differs")
    discovery_pin = payload.get("discovery_receipt")
    if not isinstance(discovery_pin, dict):
        raise ParserBootstrapError("production receipt has no discovery receipt pin")
    discovery_path = project_root / str(discovery_pin.get("path", ""))
    discovery, live_discovery_pin = validate_discovery_receipt(
        discovery_path, project_root=project_root, now=current
    )
    if live_discovery_pin != discovery_pin or payload.get("parser_sha_source") != discovery_pin:
        raise ParserBootstrapError("production parser SHA source changed")
    if discovery.get("measured_parser_sha256") != measured:
        raise ParserBootstrapError("production parser differs from pass-1 measurement")
    for field in ("base_ref", "base_digest", "source", "session_root", "image_tag"):
        if payload.get(field) != discovery.get(field):
            raise ParserBootstrapError(
                f"production/discovery lineage differs: {field}"
            )
    session = _inside_root(Path(str(payload.get("session_root"))), project_root)
    command = production_command(
        base_ref=payload["base_ref"],
        parser_sha256=measured,
        image_tag=payload["image_tag"],
        session_root=session,
        project_root=project_root,
    )
    if payload.get("command") != command or payload.get("command_sha256") != canonical_sha256(command):
        raise ParserBootstrapError("production build command differs")
    if payload.get("inputs") != _input_pins(project_root):
        raise ParserBootstrapError("production build inputs changed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "raw_build_log",
        "image_id",
        "raw_image_inspect",
        "platform_thermal",
    }:
        raise ParserBootstrapError("production artifact set differs")
    for name, expected in artifacts.items():
        if not isinstance(expected, dict):
            raise ParserBootstrapError("production artifact pin is malformed")
        if file_pin(project_root / str(expected.get("path", "")), project_root=project_root) != expected:
            raise ParserBootstrapError(f"production build artifact changed: {name}")
    thermal = _validate_thermal_report(
        project_root / artifacts["platform_thermal"]["path"], project_root=project_root
    )
    if payload.get("thermal_policy") != thermal["policy"]:
        raise ParserBootstrapError("production thermal policy binding differs")
    iid = (project_root / artifacts["image_id"]["path"]).read_text(encoding="ascii").strip()
    if iid != image_id:
        raise ParserBootstrapError("production image ID file differs")
    try:
        inspected = json.loads(
            (project_root / artifacts["raw_image_inspect"]["path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ParserBootstrapError("production raw image inspect is invalid") from exc
    if not isinstance(inspected, list) or len(inspected) != 1 or not isinstance(
        inspected[0], dict
    ):
        raise ParserBootstrapError("production raw image inspect set differs")
    observed_image = inspected[0]
    observed_labels = (observed_image.get("Config") or {}).get("Labels")
    receipt_image = payload.get("image")
    if not isinstance(receipt_image, dict) or receipt_image != {
        "id": observed_image.get("Id"),
        "architecture": observed_image.get("Architecture"),
        "os": observed_image.get("Os"),
        "repo_digests": observed_image.get("RepoDigests") or [],
        "labels": observed_labels,
    }:
        raise ParserBootstrapError("production receipt image differs from raw inspect")
    expected_labels = {
        "com.deepsafe.build-lineage.schema": BUILD_LINEAGE_SCHEMA_VERSION,
        "com.deepsafe.deepstream-yolo.parser-sha256": measured,
        "com.deepsafe.deepstream.base-ref": payload["base_ref"],
        "com.deepsafe.deepstream.base-digest": payload["base_digest"],
        "com.deepsafe.deepstream-yolo.patch-sha256": SOURCE_PATCH_SHA256,
        "com.deepsafe.deepstream-yolo.upstream-source-sha256": UPSTREAM_SOURCE_SHA256,
        "com.deepsafe.deepstream-yolo.patched-source-sha256": PATCHED_SOURCE_SHA256,
        "com.deepsafe.deepstream-yolo.patched-tree": PATCHED_SOURCE_TREE,
        "com.deepsafe.deepstream-yolo.parser-build-makefile-sha256": PARSER_BUILD_MAKEFILE_SHA256,
        "com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture": PARSER_CUDA_CUBIN_ARCHITECTURE,
        "com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture": PARSER_CUDA_PTX_ARCHITECTURE,
        "com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags": ";".join(PARSER_CUDA_GENCODE_FLAGS),
        "com.deepsafe.deepstream-yolo.parser-build-command-sha256": PARSER_BUILD_COMMAND_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-path": PARSER_POST_LINK_TOOL_PATH,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256": PARSER_POST_LINK_TOOL_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-version": PARSER_POST_LINK_TOOL_VERSION,
        "com.deepsafe.deepstream-yolo.parser-post-link-command": PARSER_POST_LINK_COMMAND,
        "com.deepsafe.deepstream-yolo.parser-post-link-command-sha256": PARSER_POST_LINK_COMMAND_SHA256,
        "com.deepsafe.deepstream-yolo.parser-post-link-removed-sections": ";".join(
            PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        "com.deepsafe.deepstream-yolo.parser-post-link-retained-sections": ";".join(
            PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        "com.deepsafe.cuda-kernel-proof.schema": KERNEL_PROOF_SCHEMA_VERSION,
        "com.deepsafe.dockerignore.sha256": payload["inputs"]["dockerignore"][
            "sha256"
        ],
        "com.deepsafe.runtime-compatibility-controller.sha256": payload[
            "inputs"
        ]["runtime_controller"]["sha256"],
        "com.deepsafe.runtime-control-manifest.sha256": payload["inputs"][
            "runtime_control_manifest"
        ]["sha256"],
    }
    if not isinstance(observed_labels, dict) or observed_image.get("Id") != image_id:
        raise ParserBootstrapError("production raw image identity differs")
    for label, expected in expected_labels.items():
        if observed_labels.get(label) != expected:
            raise ParserBootstrapError(f"production raw image label differs: {label}")
    return payload, pin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-discovery", action="store_true")
    mode.add_argument("--execute-production", action="store_true")
    parser.add_argument("--base-ref")
    parser.add_argument("--image-tag", default="deepsafe-deepstream:9.0")
    parser.add_argument("--session-root", type=Path)
    parser.add_argument("--discovery-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.execute_production:
            if args.discovery_receipt is None:
                raise ParserBootstrapError("--discovery-receipt is required")
            result = execute_production(
                discovery_receipt=args.discovery_receipt,
                image_tag=args.image_tag,
                project_root=root,
            )
        else:
            if args.base_ref is None or args.session_root is None:
                raise ParserBootstrapError("--base-ref and --session-root are required")
            if args.execute_discovery:
                result = execute_discovery(
                    base_ref=args.base_ref,
                    image_tag=args.image_tag,
                    session_root=args.session_root,
                    project_root=root,
                )
            else:
                result = make_plan(
                    base_ref=args.base_ref,
                    image_tag=args.image_tag,
                    session_root=args.session_root,
                    project_root=root,
                )
                output = args.output or (_inside_root(args.session_root, root) / "plan.json")
                _write_json(_inside_root(output, root), result, exclusive=False)
    except (OSError, subprocess.SubprocessError, ParserBootstrapError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
