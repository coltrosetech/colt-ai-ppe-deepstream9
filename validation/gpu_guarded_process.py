"""Run one attached DeepStream Docker process behind the shared GPU guards.

The accuracy/review campaigns are normally short, but they must still use the
same continuous GPU telemetry as the long throughput and endurance campaigns.
The default workstation-managed policy records temperature, power-limit and
slowdown signals without replacing BIOS/EC/driver protection with static
software aborts.  Telemetry failure, identity drift, driver/Xid faults, process
failure and pin-integrity faults remain fail-closed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from validation.ds9_runtime_compatibility import (
    DEFAULT_RECEIPT as DEFAULT_DS9_COMPATIBILITY_RECEIPT,
    require_static_candidate_compatibility,
    require_runtime_compatibility,
)
from validation.gpu_reentry_evidence import (
    LEGACY_STRICT_PHYSICAL_POLICY_ID,
    WORKSTATION_MANAGED_POLICY_ID,
    operating_policy_contract,
    require_reentry_evidence,
)
from validation.scene_benchmark.run_matrix import (
    DEFAULT_GPU_OPERATING_POLICY_MODE,
    GpuMonitor,
    GPU_CSV_HEADER,
    LEGACY_STRICT_GPU_OPERATING_POLICY_MODE,
    atomic_write_json,
    new_xid_lines,
    preflight,
    query_gpu_identity,
    read_power_profile,
    read_xid_log,
    scan_fatal_log,
    stop_attached_container,
)


SCHEMA_VERSION = "deepsafe.gpu-guarded-process/v1"
RECEIPT_SCHEMA_VERSION = "deepsafe.gpu-guard-artifact-receipt/v1"
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
DEFAULT_MAX_TEMPERATURE_C = 82.0
DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W = 5.0
DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES = 2
DEFAULT_KILL_GRACE_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 0.25
DEFAULT_CONTAINER_INSPECT_TIMEOUT_SECONDS = 10.0
DEFAULT_CONTAINER_INSPECT_POLL_INTERVAL_SECONDS = 0.05
GPU_TELEMETRY_INTERVAL_SECONDS = 1.0
# The monitor samples immediately and then every second.  A process ending a
# few scheduler ticks after an integer-second boundary can therefore have one
# valid sample per complete second while ``ceil(duration)`` asks for a sample
# that was not yet due.  Half an interval matches the existing 1.5 s maximum
# edge/gap replay tolerance without hiding a genuinely missed 1 Hz sample.
GPU_TELEMETRY_ENDPOINT_TOLERANCE_SECONDS = 1.5
GPU_TELEMETRY_COUNT_GRACE_SECONDS = (
    GPU_TELEMETRY_ENDPOINT_TOLERANCE_SECONDS - GPU_TELEMETRY_INTERVAL_SECONDS
)
COMPATIBILITY_MODES = ("production", "static_candidate_smoke")
STATIC_CANDIDATE_SMOKE_WORKER = "/app/validation/ds9_gpu_smoke.py"
STATIC_DIAGNOSTIC_EVENT_CODES = frozenset(
    {
        "temperature_threshold",
        "power_limit_below_default",
        "sustained_clock_slowdown",
    }
)
GPU_IDENTITY_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "memory.total",
    "pci.bus_id",
)
GPU_SLOWDOWN_FIELDS = (
    "clock_event_sw_thermal_slowdown",
    "clock_event_hw_slowdown",
    "clock_event_hw_thermal_slowdown",
    "clock_event_hw_power_brake_slowdown",
)


class GpuGuardError(RuntimeError):
    """Raised when a guarded process cannot start or finish acceptance-safely."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


class ContainerNotVisibleError(RuntimeError):
    """Raised while Docker has not yet published a newly requested container."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"guard artifact must stay inside project root: {resolved}") from exc


def _write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)


def _replace_requested_image(
    command: Sequence[str], requested_image: str, resolved_image_id: str
) -> list[str]:
    """Bind the actual Docker process to the preflight-resolved immutable ID."""

    if IMAGE_ID_RE.fullmatch(resolved_image_id) is None:
        raise ValueError("Docker preflight returned an invalid immutable image ID")
    indices = [index for index, value in enumerate(command) if value == requested_image]
    immutable_indices = [
        index for index, value in enumerate(command) if value == resolved_image_id
    ]
    if not indices and len(immutable_indices) == 1:
        return list(command)
    if len(indices) != 1 or immutable_indices:
        raise ValueError(
            "Docker command must contain either one requested tag or one exact resolved ID"
        )
    result = list(command)
    result[indices[0]] = resolved_image_id
    return result


def inspect_running_container_image(container_name: str) -> str:
    """Return Docker's immutable image ID for the live named container."""

    completed = subprocess.run(
        [
            "docker",
            "inspect",
            "--type",
            "container",
            "--format={{.Image}}",
            container_name,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = "\n".join(
            part.strip() for part in (completed.stderr, completed.stdout) if part.strip()
        )
        if re.search(r"\bNo such (?:container|object)\b", diagnostic, re.IGNORECASE):
            raise ContainerNotVisibleError(
                f"Docker has not published container {container_name!r} yet"
            )
        bounded = diagnostic[:1000] if diagnostic else "no diagnostic output"
        raise RuntimeError(
            "Docker container image inspection failed "
            f"with exit code {completed.returncode}: {bounded}"
        )
    image_id = completed.stdout.strip()
    if IMAGE_ID_RE.fullmatch(image_id) is None:
        raise RuntimeError("running container returned an invalid immutable image ID")
    return image_id


def wait_for_running_container_image(
    container_name: str,
    process: subprocess.Popen[Any],
    *,
    inspector: Callable[[str], str] = inspect_running_container_image,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    timeout_seconds: float = DEFAULT_CONTAINER_INSPECT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_CONTAINER_INSPECT_POLL_INTERVAL_SECONDS,
) -> tuple[str, dict[str, Any]]:
    """Wait for Docker's create/name publication, then prove the live image ID.

    ``docker run`` is an attached client, but ``Popen`` returning does not mean
    the daemon has completed ``POST /containers/create``.  A missing named
    container is therefore transient only while that client remains alive and
    only inside this small bounded startup window.  Every other inspect error,
    an early client exit, an invalid ID, and timeout remain fail-closed.
    """

    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("container inspect timeout and poll interval must be positive")
    started = monotonic()
    deadline = started + timeout_seconds
    max_attempts = max(1, math.ceil(timeout_seconds / poll_interval_seconds) + 1)
    attempts = 0
    while True:
        attempts += 1
        try:
            image_id = inspector(container_name)
        except ContainerNotVisibleError as exc:
            exit_code = process.poll()
            now = monotonic()
            if exit_code is not None:
                raise RuntimeError(
                    "attached Docker process exited with code "
                    f"{exit_code} before the named container became inspectable"
                ) from exc
            if now >= deadline or attempts >= max_attempts:
                raise TimeoutError(
                    "named Docker container did not become inspectable within "
                    f"{timeout_seconds:.3f} seconds"
                ) from exc
            sleeper(min(poll_interval_seconds, max(0.0, deadline - now)))
            continue
        if IMAGE_ID_RE.fullmatch(image_id) is None:
            raise RuntimeError("running container returned an invalid immutable image ID")
        return image_id, {
            "attempts": attempts,
            "wait_seconds": round(max(0.0, monotonic() - started), 6),
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        }


def _artifact_pin(
    path: Path, project_root: Path, *, allow_empty: bool
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = _safe_relative(resolved, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"guard artifact changed while hashing: {relative}")
    if before.st_size == 0 and not allow_empty:
        raise RuntimeError(f"guard artifact is unexpectedly empty: {relative}")
    return {
        "path": relative,
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
        "allow_empty": allow_empty,
    }


def _receipt_pin(path: Path, project_root: Path) -> dict[str, Any]:
    pin = _artifact_pin(path, project_root, allow_empty=False)
    pin.pop("allow_empty")
    return pin


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
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
                raise OSError("short write while persisting guard receipt")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _guard_operating_policy(reentry_receipt: dict[str, Any]) -> dict[str, Any]:
    """Resolve the exact policy already authorized by re-entry verification."""

    supplied = reentry_receipt.get("operating_policy")
    verification = reentry_receipt.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    verified_policy = verification.get("operating_policy")
    if supplied is None and verified_policy is None:
        return operating_policy_contract(LEGACY_STRICT_PHYSICAL_POLICY_ID)
    if not isinstance(supplied, dict) or supplied != verified_policy:
        raise RuntimeError("re-entry operating-policy receipt is missing or mismatched")
    policy_id = supplied.get("id")
    try:
        expected = operating_policy_contract(str(policy_id))
    except ValueError as exc:
        raise RuntimeError("re-entry operating-policy receipt is unknown") from exc
    if supplied != expected:
        raise RuntimeError("re-entry operating-policy receipt differs from exact contract")
    return expected


def _native_gpu_policy_mode(operating_policy_id: str) -> str:
    if operating_policy_id == WORKSTATION_MANAGED_POLICY_ID:
        return DEFAULT_GPU_OPERATING_POLICY_MODE
    if operating_policy_id == LEGACY_STRICT_PHYSICAL_POLICY_ID:
        return LEGACY_STRICT_GPU_OPERATING_POLICY_MODE
    raise RuntimeError("guard operating policy cannot be mapped to the GPU monitor")


def _identity_fields(value: Any, *, context: str) -> dict[str, str]:
    if isinstance(value, dict) and isinstance(value.get("fields"), dict):
        if value.get("available") is not True:
            raise RuntimeError(f"{context} GPU identity is unavailable")
        value = value["fields"]
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} GPU identity is missing")
    normalized: dict[str, str] = {}
    for field in GPU_IDENTITY_FIELDS:
        rendered = value.get(field)
        if not isinstance(rendered, (str, int, float)) or isinstance(rendered, bool):
            raise RuntimeError(f"{context} GPU identity field is malformed: {field}")
        rendered = str(rendered).strip()
        if not rendered:
            raise RuntimeError(f"{context} GPU identity field is empty: {field}")
        normalized[field] = rendered
    return normalized


def _bind_gpu_identity(
    reentry_receipt: dict[str, Any], preflight_identity: Any, gpu_index: int
) -> dict[str, str]:
    reentry_identity = _identity_fields(
        reentry_receipt.get("gpu_identity"), context="re-entry"
    )
    current_identity = _identity_fields(preflight_identity, context="preflight")
    if reentry_identity != current_identity:
        raise RuntimeError("preflight GPU identity differs from re-entry GPU identity")
    if current_identity["index"] != str(gpu_index):
        raise RuntimeError("preflight GPU identity index differs from requested GPU")
    return current_identity


def _preflight_disposition(
    preflight_report: dict[str, Any], operating_policy_id: str
) -> tuple[bool, list[dict[str, Any]], str]:
    status = preflight_report.get("status")
    events = preflight_report.get("safety_events")
    if not isinstance(events, list) or any(not isinstance(item, dict) for item in events):
        return False, [], "preflight safety-event list is malformed"
    native_diagnostics = preflight_report.get("diagnostic_events", [])
    if not isinstance(native_diagnostics, list) or any(
        not isinstance(item, dict) for item in native_diagnostics
    ):
        return False, [], "preflight diagnostic-event list is malformed"
    if status == "ok" and events == []:
        if not native_diagnostics:
            return True, [], "clean"
        native_codes = [item.get("code") for item in native_diagnostics]
        native_diagnostics_valid = (
            operating_policy_id == WORKSTATION_MANAGED_POLICY_ID
            and all(
                isinstance(code, str) and code in STATIC_DIAGNOSTIC_EVENT_CODES
                for code in native_codes
            )
            and all(
                item.get("operating_policy_mode")
                == DEFAULT_GPU_OPERATING_POLICY_MODE
                and item.get("measurement_quality_signal") is True
                and item.get("disposition")
                == "record_only_workstation_hardware_managed"
                for item in native_diagnostics
            )
        )
        if native_diagnostics_valid:
            return (
                True,
                native_diagnostics,
                "native_record_only_workstation_static_signals",
            )
        return False, [], "unknown or policy-mismatched preflight diagnostics"
    codes = [item.get("code") for item in events]
    diagnostic_only = (
        operating_policy_id == WORKSTATION_MANAGED_POLICY_ID
        and status == "safety_abort"
        and bool(events)
        and all(isinstance(code, str) and code in STATIC_DIAGNOSTIC_EVENT_CODES for code in codes)
    )
    if diagnostic_only:
        normalized = [
            {
                **event,
                "operating_policy_mode": DEFAULT_GPU_OPERATING_POLICY_MODE,
                "measurement_quality_signal": True,
                "disposition": "record_only_workstation_hardware_managed",
            }
            for event in events
        ]
        return True, normalized, "record_only_workstation_static_signals"
    return False, [], "blocking_or_unknown_preflight_condition"


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        rendered = float(text)
    except ValueError:
        return None
    return rendered if math.isfinite(rendered) else None


def _expected_telemetry_samples(duration_seconds: float) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("GPU telemetry duration must be finite and non-negative")
    adjusted_duration = max(
        0.0, duration_seconds - GPU_TELEMETRY_COUNT_GRACE_SECONDS
    )
    return max(
        1, math.ceil(adjusted_duration / GPU_TELEMETRY_INTERVAL_SECONDS)
    )


def _gpu_csv_summary(
    path: Path,
    duration_seconds: float,
    *,
    expected_identity: dict[str, str],
    operating_policy_id: str,
    max_temperature_c: float,
    power_limit_drop_tolerance_w: float,
) -> dict[str, Any]:
    expected_minimum = _expected_telemetry_samples(duration_seconds)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != list(GPU_CSV_HEADER):
        raise RuntimeError("GPU telemetry CSV header differs from the 1 Hz contract")
    samples = rows[1:]
    if any(len(row) != len(GPU_CSV_HEADER) for row in samples):
        raise RuntimeError("GPU telemetry CSV contains a malformed row")
    malformed_samples: list[dict[str, Any]] = []
    malformed_sample_count = 0
    identity_drift_samples: list[int] = []
    identity_drift_sample_count = 0
    temperatures: list[float] = []
    temperature_threshold_samples = 0
    power_limit_drop_samples = 0
    slowdown_active_samples = 0
    consecutive_slowdown = 0
    max_consecutive_slowdown = 0
    numeric_fields = (
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
        "power_current_limit_w",
        "power_default_limit_w",
    )
    for sample_number, row in enumerate(samples, start=1):
        snapshot = dict(zip(GPU_CSV_HEADER, row))
        sample_problems: list[str] = []
        if not snapshot.get("timestamp", "").strip():
            sample_problems.append("timestamp")
        numeric = {field: _as_finite_float(snapshot.get(field)) for field in numeric_fields}
        sample_problems.extend(
            field for field, value in numeric.items() if value is None
        )
        accepted_slowdown_values = {
            "active",
            "not active",
            "yes",
            "no",
            "true",
            "false",
            "1",
            "0",
        }
        if any(
            str(snapshot.get(field, "")).strip().casefold()
            not in accepted_slowdown_values
            for field in GPU_SLOWDOWN_FIELDS
        ):
            sample_problems.append("slowdown_flags")
        if numeric["gpu_utilization_percent"] is not None and not (
            0 <= numeric["gpu_utilization_percent"] <= 100
        ):
            sample_problems.append("gpu_utilization_percent_range")
        if numeric["memory_utilization_percent"] is not None and not (
            0 <= numeric["memory_utilization_percent"] <= 100
        ):
            sample_problems.append("memory_utilization_percent_range")
        if numeric["memory_used_mib"] is not None and numeric["memory_used_mib"] < 0:
            sample_problems.append("memory_used_mib_range")
        if numeric["memory_total_mib"] is not None and numeric["memory_total_mib"] <= 0:
            sample_problems.append("memory_total_mib_range")
        for field in ("power_current_limit_w", "power_default_limit_w"):
            if numeric[field] is not None and numeric[field] <= 0:
                sample_problems.append(f"{field}_range")
        memory_total = numeric["memory_total_mib"]
        identity_drift = (
            snapshot.get("gpu_index", "").strip() != expected_identity["index"]
            or snapshot.get("gpu_name", "").strip() != expected_identity["name"]
            or memory_total is None
            or memory_total != _as_finite_float(expected_identity["memory.total"])
        )
        if identity_drift:
            identity_drift_sample_count += 1
            if len(identity_drift_samples) < 50:
                identity_drift_samples.append(sample_number)
        if sample_problems:
            malformed_sample_count += 1
            if len(malformed_samples) < 50:
                malformed_samples.append(
                    {"sample_number": sample_number, "fields": sample_problems}
                )
            consecutive_slowdown = 0
            continue

        temperature = numeric["temperature_c"]
        assert temperature is not None
        temperatures.append(temperature)
        if temperature >= max_temperature_c:
            temperature_threshold_samples += 1

        default_limit = numeric["power_default_limit_w"]
        current_limit = numeric["power_current_limit_w"]
        requested_limit = _as_finite_float(snapshot.get("power_requested_limit_w"))
        comparable_limits = [current_limit]
        if requested_limit is not None:
            comparable_limits.append(requested_limit)
        if default_limit is not None and any(
            default_limit - limit >= power_limit_drop_tolerance_w
            for limit in comparable_limits
            if limit is not None
        ):
            power_limit_drop_samples += 1

        slowdown_active = any(
            str(snapshot[field]).strip().casefold() in {"active", "yes", "true", "1"}
            for field in GPU_SLOWDOWN_FIELDS
        )
        if slowdown_active:
            slowdown_active_samples += 1
            consecutive_slowdown += 1
        else:
            consecutive_slowdown = 0
        max_consecutive_slowdown = max(
            max_consecutive_slowdown, consecutive_slowdown
        )
    return {
        "interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
        "endpoint_tolerance_seconds": GPU_TELEMETRY_ENDPOINT_TOLERANCE_SECONDS,
        "sample_count_grace_seconds": GPU_TELEMETRY_COUNT_GRACE_SECONDS,
        "process_duration_seconds": round(duration_seconds, 6),
        "expected_minimum_samples": expected_minimum,
        "csv_samples": len(samples),
        "coverage_satisfied": len(samples) >= expected_minimum,
        "first_sample_at_utc": samples[0][0] if samples else None,
        "last_sample_at_utc": samples[-1][0] if samples else None,
        "telemetry_valid": malformed_sample_count == 0,
        "malformed_sample_count": malformed_sample_count,
        "malformed_samples_first_50": malformed_samples,
        "identity_drift_sample_count": identity_drift_sample_count,
        "identity_drift_samples_first_50": identity_drift_samples,
        "quality_diagnostics": {
            "operating_policy_id": operating_policy_id,
            "signal_action": (
                "record_only"
                if operating_policy_id == WORKSTATION_MANAGED_POLICY_ID
                else "software_abort"
            ),
            "maximum_temperature_c": max(temperatures) if temperatures else None,
            "temperature_threshold_c": max_temperature_c,
            "temperature_threshold_samples": temperature_threshold_samples,
            "power_limit_drop_tolerance_w": power_limit_drop_tolerance_w,
            "power_limit_drop_samples": power_limit_drop_samples,
            "slowdown_active_samples": slowdown_active_samples,
            "max_consecutive_slowdown_samples": max_consecutive_slowdown,
        },
    }


def _failure_report(
    *,
    report_path: Path,
    report: dict[str, Any],
    reason: str,
    error: BaseException | None = None,
) -> None:
    report["status"] = "blocked_before_start"
    report["finished_at_utc"] = utc_now()
    report["failure_reasons"] = [reason]
    if error is not None:
        report["error"] = f"{type(error).__name__}: {error}"
    _write_report(report_path, report)


def run_guarded_docker(
    command: Sequence[str],
    *,
    project_root: Path,
    artifact_root: Path,
    log_path: Path,
    container_name: str,
    image: str,
    gpu_index: int,
    reentry_evidence_path: Path,
    ds9_compatibility_receipt_path: Path = DEFAULT_DS9_COMPATIBILITY_RECEIPT,
    compatibility_mode: str = "production",
    max_temperature_c: float = DEFAULT_MAX_TEMPERATURE_C,
    power_limit_drop_tolerance_w: float = DEFAULT_POWER_LIMIT_DROP_TOLERANCE_W,
    slowdown_consecutive_samples: int = DEFAULT_SLOWDOWN_CONSECUTIVE_SAMPLES,
    kill_grace_seconds: int = DEFAULT_KILL_GRACE_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    container_image_inspector: Callable[[str], str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute an attached Docker command with continuous fail-closed telemetry.

    A current re-entry bundle and a fresh hardware preflight are required before
    ``Popen`` is called. Under ``workstation_managed``, 1 Hz temperature,
    power-limit and slowdown signals are record-only while BIOS/EC/driver owns
    protection. Missing/malformed telemetry remains an abort. The selectable
    legacy policy retains static signal aborts. Postflight rejects identity
    drift, new Xids, an unreadable/changed Linux power profile, fatal DeepStream
    log patterns, missing telemetry, and a non-zero process exit.
    """

    if not command or command[0] != "docker":
        raise ValueError("guarded command must be an attached Docker invocation")
    if not container_name.strip():
        raise ValueError("container_name must be non-empty")
    if gpu_index < 0:
        raise ValueError("gpu_index cannot be negative")
    if not math.isfinite(max_temperature_c) or max_temperature_c <= 0:
        raise ValueError("max_temperature_c must be positive")
    if (
        not math.isfinite(power_limit_drop_tolerance_w)
        or power_limit_drop_tolerance_w <= 0
    ):
        raise ValueError("power_limit_drop_tolerance_w must be positive")
    if slowdown_consecutive_samples <= 0:
        raise ValueError("slowdown_consecutive_samples must be positive")
    if kill_grace_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("kill grace and poll interval must be positive")
    if compatibility_mode not in COMPATIBILITY_MODES:
        raise ValueError("unsupported DS9 compatibility mode")
    if compatibility_mode == "static_candidate_smoke":
        required_tokens = {
            "--pull=never",
            "--network=none",
            "--read-only",
            "--inside-container",
            STATIC_CANDIDATE_SMOKE_WORKER,
        }
        if not required_tokens.issubset(set(command)):
            raise ValueError("static-candidate mode is restricted to the closed smoke command")
        immutable_image_tokens = [
            value for value in command if IMAGE_ID_RE.fullmatch(str(value)) is not None
        ]
        if (
            "--privileged" in command
            or list(command).count(image) != 0
            or len(immutable_image_tokens) != 1
        ):
            raise ValueError("static-candidate smoke command/image binding is unsafe")
    if container_image_inspector is None:
        container_image_inspector = inspect_running_container_image

    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    log_path = log_path.resolve()
    ds9_compatibility_receipt_path = (
        project_root / ds9_compatibility_receipt_path
        if not ds9_compatibility_receipt_path.is_absolute()
        else ds9_compatibility_receipt_path
    ).resolve()
    _safe_relative(artifact_root, project_root)
    _safe_relative(log_path, project_root)
    _safe_relative(ds9_compatibility_receipt_path, project_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = artifact_root / "gpu-guard-report.json"
    preflight_path = artifact_root / "gpu-preflight.json"
    gpu_csv_path = artifact_root / "gpu.csv"
    platform_csv_path = artifact_root / "platform-thermal.csv"
    safety_event_path = artifact_root / "gpu-safety-event.json"
    receipt_path = artifact_root / "gpu-guard-artifact-receipt.json"
    guard_started_at = utc_now()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "checking",
        "started_at_utc": guard_started_at,
        "finished_at_utc": None,
        "gpu_index": gpu_index,
        "container_name": container_name,
        "image": image,
        "requested_image": image,
        "compatibility_mode": compatibility_mode,
        "operating_policy_id": None,
        "resolved_image_id": None,
        "requested_command": list(command),
        "command": None,
        "policy": {
            "operating_policy_id": None,
            "max_temperature_c": max_temperature_c,
            "power_limit_drop_tolerance_w": power_limit_drop_tolerance_w,
            "slowdown_consecutive_samples": slowdown_consecutive_samples,
            "gpu_sample_interval_seconds": 1.0,
            "kill_grace_seconds": kill_grace_seconds,
            "fail_closed": True,
            "temperature_power_slowdown_action": None,
            "unreadable_telemetry_action": "abort",
            "hardware_protection_owner": None,
        },
        "diagnostics": {
            "preflight_static_signals": [],
            "runtime_static_signals": None,
            "runtime_monitor_events": [],
            "runtime_monitor_event_counts": {},
        },
        "artifacts": {
            "report": _safe_relative(report_path, project_root),
            "preflight": _safe_relative(preflight_path, project_root),
            "gpu_csv": _safe_relative(gpu_csv_path, project_root),
            "platform_thermal_csv": _safe_relative(platform_csv_path, project_root),
            "gpu_safety_event": _safe_relative(safety_event_path, project_root),
            "deepstream_log": _safe_relative(log_path, project_root),
            "artifact_receipt": _safe_relative(receipt_path, project_root),
            "ds9_runtime_compatibility_receipt": _safe_relative(
                ds9_compatibility_receipt_path, project_root
            ),
        },
        "process": {
            "started": False,
            "command": None,
            "container_image_id": None,
            "container_image_inspection": {
                "status": "pending",
                "attempts": 0,
                "wait_seconds": None,
                "timeout_seconds": DEFAULT_CONTAINER_INSPECT_TIMEOUT_SECONDS,
                "poll_interval_seconds": (
                    DEFAULT_CONTAINER_INSPECT_POLL_INTERVAL_SECONDS
                ),
                "started_at_utc": None,
                "verified_at_utc": None,
            },
            "exit_code": None,
            "termination_method": None,
        },
        "timeline": {
            "guard_started_at_utc": guard_started_at,
            "reentry_verified_at_utc": None,
            "preflight_checked_at_utc": None,
            "ds9_compatibility_verified_at_utc": None,
            "process_started_at_utc": None,
            "process_finished_at_utc": None,
            "postflight_checked_at_utc": None,
            "guard_finished_at_utc": None,
        },
        "failure_reasons": [],
    }
    _write_report(report_path, report)

    try:
        reentry_receipt = require_reentry_evidence(
            reentry_evidence_path,
            project_root=project_root,
        )
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_reentry_evidence_blocked",
            error=exc,
        )
        raise GpuGuardError("GPU re-entry evidence blocked execution", report) from exc
    report["reentry_evidence"] = reentry_receipt
    report["timeline"]["reentry_verified_at_utc"] = utc_now()
    try:
        operating_policy = _guard_operating_policy(reentry_receipt)
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_reentry_operating_policy_invalid",
            error=exc,
        )
        raise GpuGuardError("GPU re-entry operating policy is invalid", report) from exc
    operating_policy_id = str(operating_policy["id"])
    workstation_managed = operating_policy_id == WORKSTATION_MANAGED_POLICY_ID
    native_gpu_policy_mode = _native_gpu_policy_mode(operating_policy_id)
    report["operating_policy"] = operating_policy
    report["operating_policy_id"] = operating_policy_id
    report["policy"].update(
        {
            "operating_policy_id": operating_policy_id,
            "temperature_power_slowdown_action": (
                "record_only" if workstation_managed else "software_abort"
            ),
            "hardware_protection_owner": operating_policy[
                "hardware_protection_owner"
            ],
        }
    )
    _write_report(report_path, report)

    try:
        preflight_report = preflight(
            image=image,
            gpu_index=gpu_index,
            max_temperature_c=max_temperature_c,
            allow_non_performance_profile=False,
            power_limit_drop_tolerance_w=power_limit_drop_tolerance_w,
            slowdown_consecutive_samples=slowdown_consecutive_samples,
            operating_policy_mode=native_gpu_policy_mode,
        )
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_preflight_failed",
            error=exc,
        )
        raise GpuGuardError("GPU preflight failed before execution", report) from exc
    if not isinstance(preflight_report, dict):
        error = RuntimeError("GPU preflight report root is malformed")
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_preflight_failed",
            error=error,
        )
        raise GpuGuardError("GPU preflight failed before execution", report) from error
    preflight_allowed, preflight_diagnostics, preflight_disposition = (
        _preflight_disposition(preflight_report, operating_policy_id)
    )
    if preflight_allowed and preflight_diagnostics:
        preflight_report["collector_status"] = preflight_report.get("status")
        preflight_report["status"] = "ok"
        preflight_report["diagnostic_events"] = preflight_diagnostics
        preflight_report["safety_events"] = []
    preflight_report["guard_operating_policy_id"] = operating_policy_id
    preflight_report["guard_disposition"] = preflight_disposition
    _write_report(preflight_path, preflight_report)
    report["diagnostics"]["preflight_static_signals"] = preflight_diagnostics
    report["preflight"] = {
        "status": preflight_report.get("status"),
        "checked_at_utc": preflight_report.get("checked_at_utc"),
        "requested_image": preflight_report.get("image"),
        "resolved_image_id": preflight_report.get("image_id"),
        "gpu_identity": preflight_report.get("gpu_identity"),
        "power_profile": preflight_report.get("power_profile"),
        "power_safety_policy": preflight_report.get("power_safety_policy"),
        "safety_events": preflight_report.get("safety_events", []),
    }
    if not preflight_allowed:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_preflight_safety_abort",
        )
        raise GpuGuardError("GPU preflight requested a safety abort", report)

    try:
        expected_gpu_identity = _bind_gpu_identity(
            reentry_receipt, preflight_report.get("gpu_identity"), gpu_index
        )
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="gpu_identity_drift_before_start",
            error=exc,
        )
        raise GpuGuardError("GPU identity drift blocked execution", report) from exc
    report["gpu_identity_binding"] = {
        "status": "exact_match",
        "fields": expected_gpu_identity,
    }

    resolved_image_id = preflight_report.get("image_id")
    try:
        if compatibility_mode == "production":
            ds9_compatibility = require_runtime_compatibility(
                ds9_compatibility_receipt_path,
                project_root=project_root,
                requested_image=image,
                resolved_image_id=str(resolved_image_id),
            )
        else:
            ds9_compatibility = require_static_candidate_compatibility(
                ds9_compatibility_receipt_path,
                project_root=project_root,
                requested_image=image,
                resolved_image_id=str(resolved_image_id),
            )
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="ds9_runtime_compatibility_blocked",
            error=exc,
        )
        raise GpuGuardError(
            "DeepStream 9 runtime compatibility blocked execution", report
        ) from exc
    report["ds9_runtime_compatibility"] = ds9_compatibility
    report["timeline"]["ds9_compatibility_verified_at_utc"] = utc_now()
    try:
        executed_command = _replace_requested_image(
            command, image, str(resolved_image_id)
        )
    except Exception as exc:
        _failure_report(
            report_path=report_path,
            report=report,
            reason="immutable_image_command_binding_failed",
            error=exc,
        )
        raise GpuGuardError("immutable Docker image binding failed", report) from exc
    report["resolved_image_id"] = resolved_image_id
    report["command"] = executed_command
    report["process"]["command"] = executed_command
    report["timeline"]["preflight_checked_at_utc"] = preflight_report.get(
        "checked_at_utc"
    )

    xid_before = read_xid_log()
    if not xid_before.get("available"):
        _failure_report(
            report_path=report_path,
            report=report,
            reason="xid_log_unavailable_before_start",
        )
        raise GpuGuardError("NVIDIA Xid log is unavailable before execution", report)
    report["pre_run"] = {"xid": xid_before}

    monitor = GpuMonitor(
        gpu_csv_path,
        safety_event_path,
        gpu_index,
        max_temperature_c,
        power_limit_drop_tolerance_w,
        slowdown_consecutive_samples,
        platform_thermal_path=platform_csv_path,
        platform_thermal_manifest=preflight_report.get("platform_thermal_sources"),
        operating_policy_mode=native_gpu_policy_mode,
        expected_gpu_identity=expected_gpu_identity,
    )

    def monitor_event_is_blocking() -> bool:
        if not monitor.safety_reason:
            return False
        return not (
            workstation_managed
            and monitor.safety_reason_code in STATIC_DIAGNOSTIC_EVENT_CODES
        )

    process: subprocess.Popen[Any] | None = None
    run_exception: BaseException | None = None
    process_started_monotonic: float | None = None
    process_finished_monotonic: float | None = None
    report["status"] = "running"
    _write_report(report_path, report)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        try:
            monitor.start()
            process_started_monotonic = time.monotonic()
            process = popen_factory(
                executed_command,
                cwd=project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report["process"]["started"] = True
            report["timeline"]["process_started_at_utc"] = utc_now()
            inspection_report = report["process"]["container_image_inspection"]
            inspection_report["started_at_utc"] = utc_now()
            try:
                running_image_id, inspection_facts = wait_for_running_container_image(
                    container_name,
                    process,
                    inspector=container_image_inspector,
                    sleeper=sleeper,
                )
            except BaseException:
                inspection_report["status"] = "failed"
                raise
            inspection_report.update(inspection_facts)
            inspection_report["status"] = "verified"
            inspection_report["verified_at_utc"] = utc_now()
            if running_image_id != resolved_image_id:
                inspection_report["status"] = "image_mismatch"
                raise RuntimeError(
                    "running container image differs from the preflight-resolved image ID"
                )
            report["process"]["container_image_id"] = running_image_id
            _write_report(report_path, report)
            while process.poll() is None:
                if monitor_event_is_blocking():
                    _, method = stop_attached_container(
                        process, container_name, kill_grace_seconds
                    )
                    report["process"]["termination_method"] = method
                    break
                sleeper(poll_interval_seconds)
        except BaseException as exc:  # cleanup is mandatory even for interrupts
            run_exception = exc
        finally:
            if process is not None and process.poll() is None:
                try:
                    _, method = stop_attached_container(
                        process, container_name, kill_grace_seconds
                    )
                    report["process"]["termination_method"] = method
                except BaseException as exc:
                    if run_exception is None:
                        run_exception = exc
            monitor.stop()
            if monitor.ident is not None:
                monitor.join(timeout=5)
            process_finished_monotonic = time.monotonic()
            report["timeline"]["process_finished_at_utc"] = utc_now()

    report["process"]["exit_code"] = process.returncode if process is not None else None
    xid_after = read_xid_log()
    power_profile_after = read_power_profile()
    postflight_gpu_identity: dict[str, str] | None = None
    postflight_gpu_identity_error: str | None = None
    try:
        postflight_gpu_identity = _identity_fields(
            query_gpu_identity(gpu_index), context="postflight"
        )
    except Exception as exc:
        postflight_gpu_identity_error = f"{type(exc).__name__}: {exc}"
    new_xids = new_xid_lines(xid_before, xid_after)
    fatal_patterns = scan_fatal_log(log_path)
    duration_seconds = (
        max(0.0, process_finished_monotonic - process_started_monotonic)
        if process_started_monotonic is not None
        and process_finished_monotonic is not None
        else 0.0
    )
    try:
        telemetry_coverage = _gpu_csv_summary(
            gpu_csv_path,
            duration_seconds,
            expected_identity=expected_gpu_identity,
            operating_policy_id=operating_policy_id,
            max_temperature_c=max_temperature_c,
            power_limit_drop_tolerance_w=power_limit_drop_tolerance_w,
        )
    except Exception as exc:
        telemetry_coverage = {
            "interval_seconds": GPU_TELEMETRY_INTERVAL_SECONDS,
            "endpoint_tolerance_seconds": (
                GPU_TELEMETRY_ENDPOINT_TOLERANCE_SECONDS
            ),
            "sample_count_grace_seconds": GPU_TELEMETRY_COUNT_GRACE_SECONDS,
            "process_duration_seconds": round(duration_seconds, 6),
            "expected_minimum_samples": _expected_telemetry_samples(
                duration_seconds
            ),
            "csv_samples": 0,
            "coverage_satisfied": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    failure_reasons: list[str] = []
    if run_exception is not None:
        failure_reasons.append(f"run_exception={type(run_exception).__name__}: {run_exception}")
    if process is None:
        failure_reasons.append("docker_process_not_started")
    elif process.returncode != 0:
        failure_reasons.append(f"docker_exit_code={process.returncode}")
    if monitor_event_is_blocking():
        failure_reasons.append(
            f"gpu_safety_abort={monitor.safety_reason_code}: {monitor.safety_reason}"
        )
    if monitor.query_errors:
        failure_reasons.append(
            f"gpu_telemetry_query_errors={len(monitor.query_errors)}"
        )
    if monitor.samples < 1:
        failure_reasons.append("gpu_telemetry_samples=0")
    if (
        not telemetry_coverage.get("coverage_satisfied")
        or telemetry_coverage.get("csv_samples") != monitor.samples
    ):
        failure_reasons.append("gpu_telemetry_1hz_coverage_failed")
    if telemetry_coverage.get("telemetry_valid") is not True:
        failure_reasons.append("gpu_telemetry_malformed_or_unreadable")
    if telemetry_coverage.get("identity_drift_sample_count", 0) != 0:
        failure_reasons.append("gpu_identity_drift_in_runtime_telemetry")
    if hasattr(monitor, "gpu_identity_checks"):
        if getattr(monitor, "gpu_identity_checks", 0) < 1:
            failure_reasons.append("gpu_identity_not_rechecked_during_run")
        elif getattr(monitor, "last_observed_gpu_identity", None) != expected_gpu_identity:
            failure_reasons.append("gpu_identity_drift_during_run")
    if postflight_gpu_identity_error is not None:
        failure_reasons.append("gpu_identity_unavailable_after_run")
    elif postflight_gpu_identity != expected_gpu_identity:
        failure_reasons.append("gpu_identity_drift_after_run")
    if not xid_after.get("available"):
        failure_reasons.append("xid_log_unavailable_after_run")
    if new_xids:
        failure_reasons.append(f"new_nvidia_xid_events={len(new_xids)}")
    if not power_profile_after.get("available"):
        failure_reasons.append("power_profile_unavailable_after_run")
    elif power_profile_after.get("value") != "performance":
        failure_reasons.append(
            f"power_profile_changed_to={power_profile_after.get('value')}"
        )
    if any(fatal_patterns.values()):
        failure_reasons.append("fatal_patterns_in_deepstream_log")

    postflight_checked_at = utc_now()
    guard_finished_at = utc_now()
    report["timeline"]["postflight_checked_at_utc"] = postflight_checked_at
    report["timeline"]["guard_finished_at_utc"] = guard_finished_at

    report.update(
        {
            "status": "safety_abort" if monitor_event_is_blocking() else (
                "failed" if failure_reasons else "complete"
            ),
            "finished_at_utc": guard_finished_at,
            "failure_reasons": failure_reasons,
            "telemetry": {
                "samples": monitor.samples,
                "query_error_count": len(monitor.query_errors),
                "query_errors_first_50": monitor.query_errors[:50],
                "gpu_identity_checks": getattr(monitor, "gpu_identity_checks", 0),
                "last_observed_gpu_identity": getattr(
                    monitor, "last_observed_gpu_identity", None
                ),
                "power_limit_drop_samples": telemetry_coverage.get(
                    "quality_diagnostics", {}
                ).get("power_limit_drop_samples", monitor.power_limit_drop_samples),
                "slowdown_active_samples": telemetry_coverage.get(
                    "quality_diagnostics", {}
                ).get("slowdown_active_samples", monitor.slowdown_active_samples),
                "max_consecutive_slowdown_samples": (
                    telemetry_coverage.get("quality_diagnostics", {}).get(
                        "max_consecutive_slowdown_samples",
                        monitor.max_consecutive_slowdown_samples,
                    )
                ),
                "platform_thermal_samples": monitor.platform_thermal_samples,
                "platform_thermal_read_error_count": (
                    monitor.platform_thermal_error_count
                ),
                "safety_event": (
                    monitor.safety_event if monitor_event_is_blocking() else None
                ),
                "coverage": telemetry_coverage,
            },
            "postflight": {
                "xid": xid_after,
                "new_xid_lines": new_xids,
                "power_profile": power_profile_after,
                "gpu_identity": postflight_gpu_identity,
                "gpu_identity_error": postflight_gpu_identity_error,
                "fatal_log_patterns": fatal_patterns,
            },
        }
    )
    report["diagnostics"]["runtime_static_signals"] = telemetry_coverage.get(
        "quality_diagnostics"
    )
    report["diagnostics"]["runtime_monitor_events"] = list(
        getattr(monitor, "diagnostic_events", [])
    )
    report["diagnostics"]["runtime_monitor_event_counts"] = dict(
        getattr(monitor, "diagnostic_event_counts", {})
    )
    if (
        workstation_managed
        and monitor.safety_event is not None
        and not monitor_event_is_blocking()
    ):
        report["diagnostics"]["monitor_static_signal_event"] = monitor.safety_event
    receipt_error: Exception | None = None
    try:
        receipt_artifacts = {
            "preflight": _artifact_pin(
                preflight_path, project_root, allow_empty=False
            ),
            "gpu_csv": _artifact_pin(
                gpu_csv_path, project_root, allow_empty=False
            ),
            "platform_thermal_csv": _artifact_pin(
                platform_csv_path, project_root, allow_empty=False
            ),
            "deepstream_log": _artifact_pin(
                log_path, project_root, allow_empty=True
            ),
        }
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "created_at_utc": guard_finished_at,
            "guard_status": report["status"],
            "requested_image": image,
            "compatibility_mode": compatibility_mode,
            "operating_policy_id": operating_policy_id,
            "operating_policy": operating_policy,
            "resolved_image_id": resolved_image_id,
            "requested_command": list(command),
            "executed_command": executed_command,
            "running_container": {
                "name": container_name,
                "image_id": report["process"]["container_image_id"],
            },
            "timeline": report["timeline"],
            "reentry_evidence": reentry_receipt,
            "ds9_runtime_compatibility": ds9_compatibility,
            "diagnostics": report["diagnostics"],
            "artifacts": receipt_artifacts,
            "safety_event": (
                {
                    "present": True,
                    "disposition": "blocking_abort",
                    "artifact": _artifact_pin(
                        safety_event_path, project_root, allow_empty=False
                    ),
                }
                if safety_event_path.is_file() and monitor_event_is_blocking()
                else {
                    "present": False,
                    "disposition": None,
                    "path": _safe_relative(safety_event_path, project_root),
                }
            ),
            "record_only_diagnostic_event": (
                {
                    "present": True,
                    "disposition": "record_only",
                    "artifact": _artifact_pin(
                        safety_event_path, project_root, allow_empty=False
                    ),
                }
                if safety_event_path.is_file() and not monitor_event_is_blocking()
                else {
                    "present": False,
                    "disposition": None,
                    "path": _safe_relative(safety_event_path, project_root),
                }
            ),
        }
        _write_exclusive_json(receipt_path, receipt)
        report["artifact_receipt"] = _receipt_pin(receipt_path, project_root)
    except Exception as exc:
        receipt_error = exc
    if receipt_error is not None:
        report["failure_reasons"].append(
            f"guard_artifact_receipt_failed={type(receipt_error).__name__}: {receipt_error}"
        )
        report["status"] = "failed"
    _write_report(report_path, report)
    if report["failure_reasons"]:
        raise GpuGuardError(
            "guarded DeepStream process did not finish acceptance-safely: "
            + "; ".join(report["failure_reasons"]),
            report,
        )
    return report
