"""Collect and verify a read-only GPU execution re-entry evidence bundle.

This module only reads telemetry and files.  It has no command for changing GPU
power, Linux/BIOS profiles, fans, or for starting a benchmark.  A complete
bundle is a technical observation, not cryptographic operator authentication.
Execution authority comes from the user's explicit instruction, never this tool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from validation.scene_benchmark.run_matrix import (
    GPU_CSV_HEADER,
    GPU_QUERY_FIELDS,
    assess_gpu_safety,
    discover_platform_thermal_sources,
    gpu_row_snapshot,
    read_platform_thermal_row,
)


SCHEMA_VERSION = "deepsafe.gpu-reentry-evidence/v1"
DECLARATION_SCHEMA_VERSION = "deepsafe.gpu-reentry-declaration/v1"
VERIFICATION_SCHEMA_VERSION = "deepsafe.gpu-reentry-verification/v1"
OPERATING_POLICY_SCHEMA_VERSION = "deepsafe.gpu-reentry-operating-policy/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

WORKSTATION_MANAGED_POLICY_ID = "workstation_managed"
LEGACY_STRICT_PHYSICAL_POLICY_ID = "legacy_strict_physical"
DEFAULT_OPERATING_POLICY_ID = WORKSTATION_MANAGED_POLICY_ID

WORKSTATION_REQUIRED_GATE_IDS = [
    "operating_policy_contract",
    "fresh_idle_gpu_telemetry",
    "linux_power_profiles_performance",
    "ac_mains_online",
    "no_xid_current_boot",
    "current_guard_code_hashes",
]
WORKSTATION_INFORMATIONAL_GATE_IDS = [
    "operator_declaration_integrity",
    "bios_thermal_management",
    "original_240w_adapter_declaration",
    "cooling_and_epsa_acknowledgement",
    "idle_gpu_quality_diagnostics",
    "platform_fan_acpi_snapshot",
    "incident_provenance_integrity",
]
LEGACY_REQUIRED_GATE_IDS = [
    *WORKSTATION_REQUIRED_GATE_IDS,
    *WORKSTATION_INFORMATIONAL_GATE_IDS,
]

OPERATING_POLICY_CONTRACTS: dict[str, dict[str, Any]] = {
    WORKSTATION_MANAGED_POLICY_ID: {
        "schema_version": OPERATING_POLICY_SCHEMA_VERSION,
        "id": WORKSTATION_MANAGED_POLICY_ID,
        "revision": 1,
        "execution_authority": "explicit_user_instruction",
        "hardware_protection_owner": "workstation_bios_ec_gpu_driver",
        "required_gate_ids": WORKSTATION_REQUIRED_GATE_IDS,
        "informational_gate_ids": WORKSTATION_INFORMATIONAL_GATE_IDS,
    },
    LEGACY_STRICT_PHYSICAL_POLICY_ID: {
        "schema_version": OPERATING_POLICY_SCHEMA_VERSION,
        "id": LEGACY_STRICT_PHYSICAL_POLICY_ID,
        "revision": 1,
        "execution_authority": "explicit_user_instruction",
        "hardware_protection_owner": "operator_plus_active_software_guards",
        "required_gate_ids": LEGACY_REQUIRED_GATE_IDS,
        "informational_gate_ids": [],
    },
}

THERMAL_MANAGEMENT_PATH = Path(
    "/sys/class/firmware-attributes/dell-wmi-sysman/attributes/"
    "ThermalManagement/current_value"
)
THERMAL_MANAGEMENT_POSSIBLE_VALUES_PATH = THERMAL_MANAGEMENT_PATH.with_name(
    "possible_values"
)
THERMAL_MANAGEMENT_REQUIRED_VALUE = "UltraPerformance"
THERMAL_MANAGEMENT_SUDO_COMMAND = f"sudo cat {THERMAL_MANAGEMENT_PATH}"
PLATFORM_PROFILE_PATH = Path("/sys/firmware/acpi/platform_profile")
SYS_VENDOR_PATH = Path("/sys/class/dmi/id/sys_vendor")
KERNEL_OSRELEASE_PATH = Path("/proc/sys/kernel/osrelease")
DELL_PC_MODULE_ROOT = Path("/sys/module/dell_pc")
DELL_PC_MODULE_SRCVERSION_PATH = DELL_PC_MODULE_ROOT / "srcversion"
DELL_PC_MODULE_INITSTATE_PATH = DELL_PC_MODULE_ROOT / "initstate"
DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH = Path(
    "/sys/devices/faux/dell-pc/platform-profile/platform-profile-0/name"
)
DELL_PLATFORM_PROFILE_PROVIDER_PROFILE_PATH = (
    DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH.with_name("profile")
)
DELL_PLATFORM_PROFILE_MAPPING_SCHEMA_VERSION = (
    "deepsafe.dell-platform-profile-thermal-mapping/v1"
)
DELL_SYS_VENDORS = ("Dell Inc.", "Dell Computer Corporation")
DELL_KERNEL_OSRELEASE_PATTERN = r"^6\.17\.[0-9]+(?:[-+][A-Za-z0-9._-]+)?$"
DELL_PC_MODULE_SRCVERSION_ALLOWLIST = ("370814FE904C30776223695",)
DELL_PLATFORM_PROFILE_MAPPING_SOURCE = {
    "implementation": "Linux dell-pc driver",
    "source_revision": "Linux v6.17",
    "source_commit": "e5f0a698b34ed76002dc5cff3804a61c80233a7a",
    "source_path": "drivers/platform/x86/dell/dell-pc.c",
    "source_url": (
        "https://raw.githubusercontent.com/torvalds/linux/"
        "e5f0a698b34ed76002dc5cff3804a61c80233a7a/"
        "drivers/platform/x86/dell/dell-pc.c"
    ),
    "firmware_interface": "Dell SMBIOS SELECT_THERMAL_MANAGEMENT",
    "semantic_mapping": "PLATFORM_PROFILE_PERFORMANCE -> DELL_PERFORMANCE",
    "runtime_binding": (
        "Linux 6.17.x plus live /sys/module/dell_pc with an allowlisted srcversion"
    ),
}
DELL_PLATFORM_PROFILE_REQUIRED_CONDITIONS = {
    "sys_vendor_values": list(DELL_SYS_VENDORS),
    "kernel_osrelease_pattern": DELL_KERNEL_OSRELEASE_PATTERN,
    "module_name": "dell_pc",
    "module_srcversion_values": list(DELL_PC_MODULE_SRCVERSION_ALLOWLIST),
    "module_initstate": "live",
    "provider_name": "dell-pc",
    "provider_profile": "performance",
    "acpi_platform_profile": "performance",
    "wmi_possible_value": THERMAL_MANAGEMENT_REQUIRED_VALUE,
}
POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")

GPU_IDLE_SAMPLES = 2
GPU_IDLE_SAMPLE_INTERVAL_SECONDS = 1.0
# A workstation campaign may spend more than two hours on the canonical 24-run
# FPS matrix after immutable image/smoke binding.  Four hours keeps that single
# technical identity baseline usable for one campaign while every run still
# performs fresh preflight, continuous telemetry, identity checks and
# Xid/fatal-log postflight.
GPU_IDLE_MAX_AGE_SECONDS = 4 * 60 * 60
PHYSICAL_DECLARATION_MAX_AGE_SECONDS = 24 * 60 * 60
GPU_IDLE_MAX_UTILIZATION_PERCENT = 10.0
GPU_IDLE_MAX_TEMPERATURE_C = 65.0
POWER_LIMIT_DROP_TOLERANCE_W = 5.0
GPU_COMPUTE_PROCESS_FIELDS = ("pid", "process_name", "gpu_uuid")

GUARD_PATHS = (
    "validation/gpu_reentry_evidence.py",
    "validation/scene_benchmark/run_matrix.py",
    "validation/endurance/supervisor.py",
    "validation/endurance/campaign.json",
    "validation/run_caviar.py",
    "validation/run_caviar_batch.py",
    "validation/open_video_review.py",
    "validation/run_loaf.py",
    "validation/run_loaf_batch.py",
    "validation/gpu_guarded_process.py",
)
INCIDENT_SOURCE_HASHES = {
    "validation/results/scene-benchmark/people_waiting_crosswalk/640/gpu.csv": (
        "8e8a03845599a392f80f852480c6b31658b6a4aa43ec29b2ec19c44f4d3a9d4b"
    ),
    "validation/results/scene-benchmark/people_waiting_crosswalk/640/deepstream.log": (
        "5d154fb1490e75ad3564e7ca65a35c78fbcf158ec77da541e6222535c73c362f"
    ),
    "validation/results/scene-benchmark/people_waiting_crosswalk/640/status.json": (
        "51b25b92a2a0603405cae5bded71d19dfa239286576df3ecd1282209399aa35d"
    ),
}
INCIDENT_DOCUMENT_PATH = "docs/gpu-power-throttle-incident.md"

CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


class ReentryEvidenceError(RuntimeError):
    """Raised when a GPU execution path presents incomplete or stale evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def operating_policy_contract(
    policy_id: str = DEFAULT_OPERATING_POLICY_ID,
) -> dict[str, Any]:
    """Return an isolated copy of one exact, versioned operating policy."""

    try:
        return deepcopy(OPERATING_POLICY_CONTRACTS[policy_id])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown GPU re-entry operating policy: {policy_id!r}") from exc


def _resolve_operating_policy(
    evidence: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Validate the embedded policy without trusting a caller-selected mode.

    V1 artifacts made before operating policies were explicit retain their
    original strict semantics.  Any *present* policy must match a complete known
    contract and its collection-policy binding exactly.
    """

    if "operating_policy" not in evidence:
        return (
            operating_policy_contract(LEGACY_STRICT_PHYSICAL_POLICY_ID),
            "legacy v1 evidence has no explicit policy and is interpreted strictly",
        )
    supplied = evidence.get("operating_policy")
    if not isinstance(supplied, dict):
        return None, "operating_policy must be an exact versioned object"
    policy_id = supplied.get("id")
    if not isinstance(policy_id, str) or policy_id not in OPERATING_POLICY_CONTRACTS:
        return None, "operating_policy id is missing or unknown"
    expected = OPERATING_POLICY_CONTRACTS[policy_id]
    if supplied != expected:
        return None, f"operating_policy contract differs from exact {policy_id!r} policy"
    collection = evidence.get("collection_policy")
    if not isinstance(collection, dict):
        return None, "collection_policy is missing for an explicit operating policy"
    if collection.get("operating_policy_id") != policy_id:
        return None, "collection_policy operating-policy binding is missing or mismatched"
    return deepcopy(expected), f"exact {policy_id!r} operating policy is embedded and bound"


def _project_path(project_root: Path, relative: str) -> Path:
    candidate = (project_root / relative).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path leaves project root: {relative}") from exc
    return candidate


def _is_read_only_command(command: Sequence[str]) -> bool:
    if not command:
        return False
    if list(command) == ["powerprofilesctl", "get"]:
        return True
    if command[0] == "nvidia-smi":
        gpu_query = (
            len(command) == 5
            and command[1] == "-i"
            and str(command[2]).isdigit()
            and command[3].startswith("--query-gpu=")
            and bool(command[3].partition("=")[2])
            and command[4] == "--format=csv,noheader,nounits"
        )
        compute_query = (
            len(command) == 5
            and command[1] == "-i"
            and str(command[2]).isdigit()
            and command[3]
            == f"--query-compute-apps={','.join(GPU_COMPUTE_PROCESS_FIELDS)}"
            and command[4] == "--format=csv,noheader,nounits"
        )
        return gpu_query or compute_query
    if list(command) == [
        "journalctl",
        "--quiet",
        "-k",
        "-b",
        "--no-pager",
        "--grep",
        "NVRM: Xid",
    ]:
        return True
    if list(command) == ["dmesg", "--color=never"]:
        return True
    return False


def run_read_only_command(
    command: Sequence[str], timeout: float = 10.0
) -> subprocess.CompletedProcess[str]:
    if not _is_read_only_command(command):
        raise ValueError(f"command is not on the read-only evidence allowlist: {command!r}")
    return subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def declaration_template() -> dict[str, Any]:
    return {
        "schema_version": DECLARATION_SCHEMA_VERSION,
        "instructions": (
            "Fill only facts personally checked by the operator. Null remains blocking "
            "under legacy_strict_physical and is informational under the default "
            "workstation_managed policy; do not copy example confirmations."
        ),
        "declared_by": None,
        "declared_at_utc": None,
        "bios_thermal_management": {
            "observed_value": None,
            "observed_at_utc": None,
            "evidence_command": THERMAL_MANAGEMENT_SUDO_COMMAND,
        },
        "adapter": {
            "original_dell_adapter_confirmed": None,
            "direct_barrel_connection_confirmed": None,
            "rated_output_w": None,
            "output_voltage_v": None,
            "output_current_a": None,
            "label_or_asset_note": None,
        },
        "cooling_and_epsa": {
            "air_inlet_and_exhaust_clear_confirmed": None,
            "machine_elevated_for_airflow_confirmed": None,
            "epsa_thermal_test_completed": None,
            "epsa_result": None,
            "checked_at_utc": None,
            "note": None,
        },
    }


def _command_result(
    command: Sequence[str], runner: CommandRunner, timeout: float = 10.0
) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {
            "available": False,
            "command": list(command),
            "returncode": None,
            "stdout": None,
            "stdout_on_error": None,
            "stderr": None,
            "error": f"executable not found: {command[0]}",
        }
    try:
        result = runner(command, timeout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {
            "available": False,
            "command": list(command),
            "returncode": None,
            "stdout": None,
            "stdout_on_error": None,
            "stderr": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    raw_stdout = result.stdout.strip()
    raw_stderr = result.stderr.strip()
    return {
        "available": result.returncode == 0,
        "command": list(command),
        "returncode": result.returncode,
        "stdout": raw_stdout if result.returncode == 0 else None,
        "stdout_on_error": raw_stdout if result.returncode != 0 else None,
        "stderr": raw_stderr,
        "error": (
            None
            if result.returncode == 0
            else (raw_stderr or raw_stdout or "command failed")
        ),
    }


def _csv_row(result: dict[str, Any], fields: Sequence[str]) -> dict[str, str] | None:
    if not result.get("available") or not isinstance(result.get("stdout"), str):
        return None
    rows = list(csv.reader(result["stdout"].splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        result["available"] = False
        result["error"] = "unexpected CSV field count"
        return None
    return dict(zip(fields, (item.strip() for item in rows[0])))


def _csv_rows(
    result: dict[str, Any], fields: Sequence[str]
) -> list[dict[str, str]] | None:
    if not result.get("available") or not isinstance(result.get("stdout"), str):
        return None
    if not result["stdout"]:
        return []
    rows = list(csv.reader(result["stdout"].splitlines()))
    if any(len(row) != len(fields) for row in rows):
        result["available"] = False
        result["error"] = "unexpected CSV field count"
        return None
    return [
        dict(zip(fields, (item.strip() for item in row)))
        for row in rows
    ]


def collect_gpu_identity(
    gpu_index: int, runner: CommandRunner = run_read_only_command
) -> dict[str, Any]:
    fields = ["index", "uuid", "name", "pci.bus_id", "driver_version", "memory.total"]
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = _command_result(command, runner)
    row = _csv_row(result, fields)
    return {
        "available": row is not None,
        "fields": row,
        "error": result.get("error"),
        "read_only_command": command,
    }


def collect_compute_processes(
    gpu_index: int, runner: CommandRunner = run_read_only_command
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-compute-apps={','.join(GPU_COMPUTE_PROCESS_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = _command_result(command, runner)
    rows = _csv_rows(result, GPU_COMPUTE_PROCESS_FIELDS)
    return {
        "available": rows is not None,
        "processes": rows if rows is not None else [],
        "count": len(rows) if rows is not None else None,
        "error": result.get("error"),
        "read_only_command": command,
    }


def collect_idle_gpu_telemetry(
    gpu_index: int,
    *,
    runner: CommandRunner = run_read_only_command,
    sample_count: int = GPU_IDLE_SAMPLES,
    interval_seconds: float = GPU_IDLE_SAMPLE_INTERVAL_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if sample_count < 2:
        raise ValueError("at least two idle GPU samples are required")
    if interval_seconds < 0:
        raise ValueError("sample interval cannot be negative")
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    for index in range(sample_count):
        result = _command_result(command, runner)
        row = _csv_row(result, GPU_QUERY_FIELDS)
        if row is None:
            errors.append(result.get("error") or "nvidia-smi sample unavailable")
        else:
            ordered = [row[field] for field in GPU_QUERY_FIELDS]
            snapshot = gpu_row_snapshot(ordered)
            samples.append(
                {
                    "snapshot": snapshot,
                    "safety_assessment": assess_gpu_safety(
                        snapshot,
                        power_limit_drop_tolerance_w=POWER_LIMIT_DROP_TOLERANCE_W,
                    ),
                }
            )
        if index + 1 < sample_count:
            sleeper(interval_seconds)
    return {
        "available": len(samples) == sample_count and not errors,
        "requested_sample_count": sample_count,
        "sample_interval_seconds": interval_seconds,
        "samples": samples,
        "errors": errors,
        "read_only_command": command,
        "idle_policy": {
            "max_age_seconds": GPU_IDLE_MAX_AGE_SECONDS,
            "max_gpu_utilization_percent": GPU_IDLE_MAX_UTILIZATION_PERCENT,
            "max_temperature_c": GPU_IDLE_MAX_TEMPERATURE_C,
            "power_limit_drop_tolerance_w": POWER_LIMIT_DROP_TOLERANCE_W,
            "power_limit_telemetry_required": True,
            "dangerous_slowdown_flags_must_be_inactive": True,
        },
    }


def _read_text(path: Path) -> dict[str, Any]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {
            "path": str(path),
            "readable": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"path": str(path), "readable": True, "value": value, "error": None}


def _read_directory_name(path: Path) -> dict[str, Any]:
    try:
        if not path.is_dir():
            raise FileNotFoundError(f"module directory is missing: {path}")
    except OSError as exc:
        return {
            "path": str(path),
            "readable": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": str(path),
        "readable": True,
        "value": path.name,
        "error": None,
    }


def _firmware_enumeration_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(item.strip() for item in value.split(";") if item.strip())


def _dell_platform_profile_mapping_problems(mapping: Any) -> list[str]:
    """Independently validate the strict Dell-only BIOS fallback record."""

    if not isinstance(mapping, dict):
        return ["Dell platform-profile mapping evidence is missing"]

    problems: list[str] = []
    if mapping.get("schema_version") != DELL_PLATFORM_PROFILE_MAPPING_SCHEMA_VERSION:
        problems.append("Dell mapping schema is missing or unsupported")
    if mapping.get("mapping_source") != DELL_PLATFORM_PROFILE_MAPPING_SOURCE:
        problems.append("Dell driver source mapping is missing or changed")
    if mapping.get("required_conditions") != DELL_PLATFORM_PROFILE_REQUIRED_CONDITIONS:
        problems.append("Dell mapping required-condition contract is missing or changed")

    observations = mapping.get("observations")
    observations = observations if isinstance(observations, dict) else {}
    expected_paths = {
        "sys_vendor": SYS_VENDOR_PATH,
        "kernel_osrelease": KERNEL_OSRELEASE_PATH,
        "module_name": DELL_PC_MODULE_ROOT,
        "module_srcversion": DELL_PC_MODULE_SRCVERSION_PATH,
        "module_initstate": DELL_PC_MODULE_INITSTATE_PATH,
        "provider_name": DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH,
        "provider_profile": DELL_PLATFORM_PROFILE_PROVIDER_PROFILE_PATH,
        "acpi_platform_profile": PLATFORM_PROFILE_PATH,
        "wmi_possible_values": THERMAL_MANAGEMENT_POSSIBLE_VALUES_PATH,
    }
    values: dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        record = observations.get(name)
        if not isinstance(record, dict):
            problems.append(f"{name} observation is missing")
            continue
        if record.get("path") != str(expected_path):
            problems.append(f"{name} did not come from the required path")
        if record.get("readable") is not True or record.get("error") is not None:
            problems.append(f"{name} is not a clean readable observation")
            continue
        values[name] = record.get("value")

    if values.get("sys_vendor") not in DELL_SYS_VENDORS:
        problems.append("sys_vendor is not an exact supported Dell value")
    kernel_osrelease = values.get("kernel_osrelease")
    if not isinstance(kernel_osrelease, str) or re.fullmatch(
        DELL_KERNEL_OSRELEASE_PATTERN, kernel_osrelease
    ) is None:
        problems.append("running kernel osrelease is not an allowed Linux 6.17.x build")
    if values.get("module_name") != "dell_pc":
        problems.append("loaded module identity is not exactly dell_pc")
    if values.get("module_srcversion") not in DELL_PC_MODULE_SRCVERSION_ALLOWLIST:
        problems.append("dell_pc srcversion is not in the reviewed allowlist")
    if values.get("module_initstate") != "live":
        problems.append("dell_pc module is absent or not live")
    if values.get("provider_name") != "dell-pc":
        problems.append("platform-profile provider is not exactly dell-pc")
    if values.get("provider_profile") != "performance":
        problems.append("live dell-pc provider profile is not performance")
    if values.get("acpi_platform_profile") != "performance":
        problems.append("ACPI platform_profile is not performance")
    if THERMAL_MANAGEMENT_REQUIRED_VALUE not in _firmware_enumeration_values(
        values.get("wmi_possible_values")
    ):
        problems.append("WMI ThermalManagement enum does not contain UltraPerformance")
    return problems


def collect_dell_platform_profile_mapping() -> dict[str, Any]:
    """Collect the complete Dell-only mapping without running commands or sudo."""

    mapping: dict[str, Any] = {
        "schema_version": DELL_PLATFORM_PROFILE_MAPPING_SCHEMA_VERSION,
        "mapping_source": dict(DELL_PLATFORM_PROFILE_MAPPING_SOURCE),
        "required_conditions": {
            **DELL_PLATFORM_PROFILE_REQUIRED_CONDITIONS,
            "sys_vendor_values": list(DELL_SYS_VENDORS),
            "module_srcversion_values": list(
                DELL_PC_MODULE_SRCVERSION_ALLOWLIST
            ),
        },
        "observations": {
            "sys_vendor": _read_text(SYS_VENDOR_PATH),
            "kernel_osrelease": _read_text(KERNEL_OSRELEASE_PATH),
            "module_name": _read_directory_name(DELL_PC_MODULE_ROOT),
            "module_srcversion": _read_text(DELL_PC_MODULE_SRCVERSION_PATH),
            "module_initstate": _read_text(DELL_PC_MODULE_INITSTATE_PATH),
            "provider_name": _read_text(DELL_PLATFORM_PROFILE_PROVIDER_NAME_PATH),
            "provider_profile": _read_text(
                DELL_PLATFORM_PROFILE_PROVIDER_PROFILE_PATH
            ),
            "acpi_platform_profile": _read_text(PLATFORM_PROFILE_PATH),
            "wmi_possible_values": _read_text(
                THERMAL_MANAGEMENT_POSSIBLE_VALUES_PATH
            ),
        },
    }
    problems = _dell_platform_profile_mapping_problems(mapping)
    mapping["strict_conditions_met"] = not problems
    mapping["failed_conditions"] = problems
    return mapping


def collect_bios_thermal_management() -> dict[str, Any]:
    result = _read_text(THERMAL_MANAGEMENT_PATH)
    result.update(
        {
            "required_value": THERMAL_MANAGEMENT_REQUIRED_VALUE,
            "evidence_missing_command": THERMAL_MANAGEMENT_SUDO_COMMAND,
            "command_executed_by_collector": False,
            "dell_platform_profile_mapping": (
                collect_dell_platform_profile_mapping()
            ),
        }
    )
    return result


def collect_linux_power_profiles(
    runner: CommandRunner = run_read_only_command,
) -> dict[str, Any]:
    command = ["powerprofilesctl", "get"]
    result = _command_result(command, runner)
    return {
        "powerprofilesctl": {
            "available": result.get("available", False),
            "value": result.get("stdout"),
            "error": result.get("error"),
            "read_only_command": command,
        },
        "platform_profile": _read_text(PLATFORM_PROFILE_PATH),
        "required_value": "performance",
    }


def collect_ac_online(power_supply_root: Path = POWER_SUPPLY_ROOT) -> dict[str, Any]:
    supplies: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        candidates = sorted(power_supply_root.iterdir())
    except OSError as exc:
        candidates = []
        errors.append(f"{type(exc).__name__}: {exc}")
    for candidate in candidates:
        supply_type = _read_text(candidate / "type")
        if not supply_type["readable"]:
            continue
        online = _read_text(candidate / "online")
        if not online["readable"] and supply_type["value"] == "Battery":
            continue
        online_value = None
        if online["readable"] and online["value"] in {"0", "1"}:
            online_value = online["value"] == "1"
        supplies.append(
            {
                "name": candidate.name,
                "type": supply_type["value"],
                "online": online_value,
                "online_path": online["path"],
                "error": online["error"],
            }
        )
    mains = [item for item in supplies if item["type"] == "Mains"]
    return {
        "available": bool(mains),
        "mains_online": any(item["online"] is True for item in mains),
        "supplies": supplies,
        "errors": errors,
    }


def collect_platform_thermal() -> dict[str, Any]:
    try:
        manifest = discover_platform_thermal_sources()
        row, errors = read_platform_thermal_row(manifest, utc_now())
    except (OSError, ValueError, ZeroDivisionError) as exc:
        return {
            "available": False,
            "manifest": None,
            "values": {},
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    values: dict[str, float | None | str] = {}
    for column, raw in zip(manifest.get("columns", []), row):
        if column == "timestamp":
            values[column] = raw
            continue
        try:
            values[column] = float(raw) if raw != "" else None
        except ValueError:
            values[column] = None
            errors.append(f"{column}: non-numeric value {raw!r}")
    return {
        "available": bool(manifest.get("available")) and not errors,
        "manifest": manifest,
        "values": values,
        "errors": errors,
        "best_effort_sensor_names": True,
    }


def collect_xid_log(runner: CommandRunner = run_read_only_command) -> dict[str, Any]:
    probes = [
        [
            "journalctl",
            "--quiet",
            "-k",
            "-b",
            "--no-pager",
            "--grep",
            "NVRM: Xid",
        ],
        ["dmesg", "--color=never"],
    ]
    errors: list[str] = []
    for command in probes:
        result = _command_result(command, runner, timeout=15.0)
        if result.get("returncode") == 0 and result.get("error") in (None, ""):
            lines = [
                line.strip()
                for line in (result.get("stdout") or "").splitlines()
                if "NVRM:" in line and "Xid" in line
            ]
            return {
                "available": True,
                "source": command[0],
                "count": len(lines),
                "lines": lines,
                "read_only_command": command,
                "errors": errors,
            }
        # journalctl commonly returns 1 for no matches. ``--quiet`` makes that
        # result unambiguous: both output channels stay empty, while permission
        # and journal errors still carry text and fail closed.
        if (
            command[0] == "journalctl"
            and result.get("returncode") == 1
            and result.get("stdout_on_error") == ""
            and result.get("stderr") == ""
        ):
            return {
                "available": True,
                "source": "journalctl",
                "count": 0,
                "lines": [],
                "read_only_command": command,
                "errors": errors,
            }
        errors.append(f"{command[0]}: {result.get('error')}")
    return {"available": False, "source": None, "count": 0, "lines": [], "errors": errors}


def _file_records(project_root: Path, paths: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = _project_path(project_root, relative)
        if not path.is_file():
            records.append(
                {"path": relative, "present": False, "bytes": None, "sha256": None}
            )
            continue
        records.append(
            {
                "path": relative,
                "present": True,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def collect_incident_provenance(project_root: Path) -> dict[str, Any]:
    sources = _file_records(project_root, INCIDENT_SOURCE_HASHES)
    for source in sources:
        source["expected_sha256"] = INCIDENT_SOURCE_HASHES[source["path"]]
        source["hash_matches_incident_record"] = (
            source["present"] and source["sha256"] == source["expected_sha256"]
        )
    document = _file_records(project_root, [INCIDENT_DOCUMENT_PATH])[0]
    return {
        "incident_date": "2026-07-16",
        "incident_code": "gpu_power_thermal_regime_change",
        "capacity_result_policy": "interrupted run is invalid and excluded",
        "source_artifacts": sources,
        "incident_document": document,
    }


def load_declaration(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        return declaration_template(), {
            "path": None,
            "present": False,
            "sha256": None,
            "error": "no declaration path supplied",
        }
    if not path.is_file():
        return declaration_template(), {
            "path": str(path),
            "present": False,
            "sha256": None,
            "error": "declaration file does not exist",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return declaration_template(), {
            "path": str(path),
            "present": True,
            "sha256": sha256_file(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return declaration_template(), {
            "path": str(path),
            "present": True,
            "sha256": sha256_file(path),
            "error": "declaration root must be an object",
        }
    return payload, {
        "path": str(path),
        "present": True,
        "sha256": sha256_file(path),
        "error": None,
    }


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        rendered = float(value)
    except ValueError:
        return None
    return rendered if math.isfinite(rendered) else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        rendered = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if rendered.tzinfo is None:
        return None
    return rendered.astimezone(timezone.utc)


def _is_recent(moment: datetime | None, now: datetime, max_age_seconds: float) -> bool:
    if moment is None:
        return False
    age = (now - moment).total_seconds()
    return -60 <= age <= max_age_seconds


def _gate(
    gate_id: str,
    passed: bool,
    detail: str,
    *evidence_refs: str,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "required": bool(required),
        "passed": bool(passed),
        "detail": detail,
        "evidence_refs": list(evidence_refs),
    }


def _records_match_current(
    records: Any, project_root: Path, required_paths: Sequence[str]
) -> tuple[bool, str]:
    if not isinstance(records, list):
        return False, "hash records are missing"
    problems: list[str] = []
    required_set = set(required_paths)
    if len(required_set) != len(required_paths):
        return False, "required guard path contract contains duplicates"
    by_path: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            problems.append("malformed guard hash record")
            continue
        relative = item["path"]
        if relative in by_path:
            problems.append(f"duplicate guard hash record: {relative}")
            continue
        by_path[relative] = item
    extras = sorted(set(by_path) - required_set)
    if extras:
        problems.append(f"unexpected guard hash records: {', '.join(extras)}")
    for relative in required_paths:
        item = by_path.get(relative)
        path = _project_path(project_root, relative)
        if (
            item is None
            or item.get("present") is not True
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) is None
            or isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or item.get("bytes", -1) < 0
        ):
            problems.append(f"missing recorded hash: {relative}")
        elif not path.is_file():
            problems.append(f"current file missing: {relative}")
        else:
            try:
                current_stat = path.stat()
                current_hash = sha256_file(path)
            except OSError as exc:
                problems.append(
                    f"current guard file is unreadable: {relative}: {type(exc).__name__}"
                )
                continue
            if current_stat.st_size != item["bytes"]:
                problems.append(f"recorded size is stale: {relative}")
            elif current_hash != item["sha256"]:
                problems.append(f"recorded hash is stale: {relative}")
    return not problems, "; ".join(problems) if problems else "all required hashes match current files"


def _declaration_matches_file(
    evidence: dict[str, Any], project_root: Path
) -> tuple[bool, str]:
    record = evidence.get("operator_declaration_file")
    declaration = evidence.get("operator_declaration")
    if not isinstance(record, dict) or not isinstance(declaration, dict):
        return False, "declaration record or embedded declaration is missing"
    if declaration.get("schema_version") != DECLARATION_SCHEMA_VERSION:
        return False, "declaration schema is missing or unsupported"
    path_text = record.get("path")
    if not record.get("present") or record.get("error") or not isinstance(path_text, str):
        return False, "operator declaration file is absent or unreadable"
    raw_path = Path(path_text)
    try:
        path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else _project_path(project_root, path_text)
        )
        path.relative_to(project_root.resolve())
    except ValueError:
        return False, "operator declaration path leaves the project root"
    if not path.is_file():
        return False, "operator declaration file is no longer present"
    try:
        current_hash = sha256_file(path)
    except OSError as exc:
        return False, f"operator declaration file is unreadable: {type(exc).__name__}: {exc}"
    if current_hash != record.get("sha256"):
        return False, "operator declaration file hash is stale"
    try:
        current_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"operator declaration cannot be parsed: {exc}"
    if current_payload != declaration:
        return False, "embedded declaration does not match its hashed file"
    return True, "embedded declaration exactly matches its current hashed file"


def verify_evidence(
    evidence: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not isinstance(evidence, dict):
        evidence = {}
    gates: list[dict[str, Any]] = []
    operating_policy, operating_policy_detail = _resolve_operating_policy(evidence)
    evidence_schema_ok = evidence.get("schema_version") == SCHEMA_VERSION
    operating_policy_ok = operating_policy is not None and evidence_schema_ok
    if not evidence_schema_ok:
        operating_policy_detail = (
            f"{operating_policy_detail}; evidence schema is missing or unsupported"
        )
    required_gate_ids = (
        set(operating_policy["required_gate_ids"])
        if operating_policy is not None
        else set(LEGACY_REQUIRED_GATE_IDS)
    )

    def gate_is_required(gate_id: str) -> bool:
        return gate_id in required_gate_ids

    gates.append(
        _gate(
            "operating_policy_contract",
            operating_policy_ok,
            operating_policy_detail,
            "operating_policy",
            "collection_policy.operating_policy_id",
            required=True,
        )
    )
    declaration = evidence.get("operator_declaration")
    declaration = declaration if isinstance(declaration, dict) else {}
    declared_by = declaration.get("declared_by")
    declared_at = _parse_time(declaration.get("declared_at_utc"))
    attribution_ok = (
        isinstance(declared_by, str)
        and bool(declared_by.strip())
        and _is_recent(declared_at, now, PHYSICAL_DECLARATION_MAX_AGE_SECONDS)
    )

    declaration_ok, declaration_detail = _declaration_matches_file(
        evidence, project_root
    )
    gates.append(
        _gate(
            "operator_declaration_integrity",
            declaration_ok,
            declaration_detail,
            "operator_declaration_file",
            "operator_declaration",
            required=gate_is_required("operator_declaration_integrity"),
        )
    )

    bios = evidence.get("bios_thermal_management")
    bios = bios if isinstance(bios, dict) else {}
    live_bios_readable = bios.get("readable") is True
    live_bios_ok = (
        live_bios_readable
        and bios.get("path") == str(THERMAL_MANAGEMENT_PATH)
        and bios.get("error") is None
        and str(bios.get("value", "")).strip() == THERMAL_MANAGEMENT_REQUIRED_VALUE
    )
    dell_mapping = bios.get("dell_platform_profile_mapping")
    dell_mapping_problems = _dell_platform_profile_mapping_problems(dell_mapping)
    if isinstance(dell_mapping, dict):
        if dell_mapping.get("strict_conditions_met") is not True:
            dell_mapping_problems.append(
                "collector did not record every strict Dell condition as met"
            )
        if dell_mapping.get("failed_conditions") != []:
            dell_mapping_problems.append(
                "collector recorded failed strict Dell conditions"
            )
    dell_mapping_ok = not dell_mapping_problems
    declared_bios = declaration.get("bios_thermal_management")
    declared_bios = declared_bios if isinstance(declared_bios, dict) else {}
    declared_bios_ok = (
        attribution_ok
        and declared_bios.get("observed_value") == THERMAL_MANAGEMENT_REQUIRED_VALUE
        and declared_bios.get("evidence_command") == THERMAL_MANAGEMENT_SUDO_COMMAND
        and _is_recent(
            _parse_time(declared_bios.get("observed_at_utc")),
            now,
            PHYSICAL_DECLARATION_MAX_AGE_SECONDS,
        )
    )
    # A readable root-only current_value is authoritative.  Fallback evidence
    # must never override a live contradictory BIOS value.
    declared_bios_accepted = not live_bios_readable and declared_bios_ok
    dell_mapping_accepted = not live_bios_readable and dell_mapping_ok
    bios_ok = live_bios_ok or declared_bios_accepted or dell_mapping_accepted
    if live_bios_ok:
        bios_detail = "UltraPerformance was read directly from the root-only BIOS attribute"
    elif live_bios_readable:
        bios_detail = (
            "the directly readable root-only BIOS attribute did not confirm "
            "UltraPerformance; fallback evidence is not accepted"
        )
    elif declared_bios_accepted:
        bios_detail = "operator-recorded sudo output confirms UltraPerformance"
    elif dell_mapping_accepted:
        bios_detail = (
            "strict Dell mapping confirms the Dell performance thermal state: "
            "exact Dell sys_vendor, reviewed Linux 6.17.x/live allowlisted "
            "dell_pc module, dell-pc provider, live provider and ACPI performance "
            "profiles, and WMI UltraPerformance enum"
        )
    else:
        mapping_summary = "; ".join(dell_mapping_problems[:3])
        if mapping_summary:
            mapping_summary = f"; Dell fallback rejected: {mapping_summary}"
        bios_detail = (
            "missing UltraPerformance evidence; run exactly: "
            f"{THERMAL_MANAGEMENT_SUDO_COMMAND}{mapping_summary}"
        )
    gates.append(
        _gate(
            "bios_thermal_management",
            bios_ok,
            bios_detail,
            "bios_thermal_management",
            "bios_thermal_management.dell_platform_profile_mapping",
            "operator_declaration.bios_thermal_management",
            required=gate_is_required("bios_thermal_management"),
        )
    )

    adapter = declaration.get("adapter")
    adapter = adapter if isinstance(adapter, dict) else {}
    adapter_ok = (
        attribution_ok
        and adapter.get("original_dell_adapter_confirmed") is True
        and adapter.get("direct_barrel_connection_confirmed") is True
        and _as_float(adapter.get("rated_output_w")) == 240.0
        and _as_float(adapter.get("output_voltage_v")) == 19.5
        and _as_float(adapter.get("output_current_a")) == 12.3
    )
    gates.append(
        _gate(
            "original_240w_adapter_declaration",
            adapter_ok,
            (
                "signed declaration confirms original direct Dell 240 W, 19.5 V, 12.3 A barrel adapter"
                if adapter_ok
                else "signed, <=24h original/direct Dell 240 W, 19.5 V, 12.3 A barrel-adapter declaration is incomplete"
            ),
            "operator_declaration.adapter",
            required=gate_is_required("original_240w_adapter_declaration"),
        )
    )

    cooling = declaration.get("cooling_and_epsa")
    cooling = cooling if isinstance(cooling, dict) else {}
    cooling_ok = (
        attribution_ok
        and cooling.get("air_inlet_and_exhaust_clear_confirmed") is True
        and cooling.get("machine_elevated_for_airflow_confirmed") is True
        and cooling.get("epsa_thermal_test_completed") is True
        and str(cooling.get("epsa_result", "")).strip().casefold() == "pass"
        and _is_recent(
            _parse_time(cooling.get("checked_at_utc")),
            now,
            PHYSICAL_DECLARATION_MAX_AGE_SECONDS,
        )
    )
    gates.append(
        _gate(
            "cooling_and_epsa_acknowledgement",
            cooling_ok,
            (
                "signed airflow/vent and passing ePSA thermal acknowledgement is complete"
                if cooling_ok
                else "airflow/vent checks and a signed <=24h passing ePSA thermal result are incomplete"
            ),
            "operator_declaration.cooling_and_epsa",
            required=gate_is_required("cooling_and_epsa_acknowledgement"),
        )
    )

    identity = evidence.get("gpu_identity")
    identity = identity if isinstance(identity, dict) else {}
    identity_fields = (
        identity.get("fields") if isinstance(identity.get("fields"), dict) else {}
    )
    identity_problems: list[str] = []
    if identity.get("available") is not True:
        identity_problems.append("GPU identity query is unavailable")
    required_identity_fields = (
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "driver_version",
        "memory.total",
    )
    for field in required_identity_fields:
        value = identity_fields.get(field)
        if not isinstance(value, str) or not value.strip():
            identity_problems.append(f"GPU identity field is missing: {field}")
    gpu_index = evidence.get("gpu_index", 0)
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        identity_problems.append("gpu_index is malformed")
    elif str(identity_fields.get("index", "")).strip() != str(gpu_index):
        identity_problems.append("GPU identity index differs from requested gpu_index")
    memory_total = _as_float(identity_fields.get("memory.total"))
    if memory_total is None or memory_total <= 0:
        identity_problems.append("GPU identity memory.total is not a positive number")

    idle = evidence.get("idle_gpu_telemetry")
    idle = idle if isinstance(idle, dict) else {}
    collected_at = _parse_time(evidence.get("collected_at_utc"))
    age_seconds = (now - collected_at).total_seconds() if collected_at else None
    samples = idle.get("samples") if isinstance(idle.get("samples"), list) else []
    telemetry_problems: list[str] = list(identity_problems)
    quality_problems: list[str] = []
    workstation_managed = (
        operating_policy is not None
        and operating_policy.get("id") == WORKSTATION_MANAGED_POLICY_ID
    )
    compute = evidence.get("compute_processes")
    compute = compute if isinstance(compute, dict) else {}
    compute_processes = (
        compute.get("processes")
        if isinstance(compute.get("processes"), list)
        else None
    )
    expected_compute_command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-compute-apps={','.join(GPU_COMPUTE_PROCESS_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    compute_query_readable = (
        compute.get("available") is True
        and compute.get("error") is None
        and compute.get("read_only_command") == expected_compute_command
        and compute_processes is not None
        and compute.get("count") == len(compute_processes)
        and all(
            isinstance(item, dict)
            and str(item.get("pid", "")).isdigit()
            and bool(str(item.get("process_name", "")).strip())
            and item.get("gpu_uuid") == identity_fields.get("uuid")
            for item in compute_processes
        )
    )
    no_compute_clients = compute_query_readable and compute_processes == []
    if workstation_managed and compute_query_readable and compute_processes:
        telemetry_problems.append(
            "one or more CUDA compute clients are present on the selected GPU"
        )
    if idle.get("available") is not True or len(samples) < GPU_IDLE_SAMPLES:
        telemetry_problems.append("two complete nvidia-smi samples are unavailable")
    if age_seconds is None or age_seconds < -60 or age_seconds > GPU_IDLE_MAX_AGE_SECONDS:
        telemetry_problems.append(
            "idle telemetry is missing, future-dated, or older than 15 minutes"
        )
    for index, item in enumerate(samples, start=1):
        if not isinstance(item, dict):
            telemetry_problems.append(f"sample {index} is malformed")
            continue
        snapshot = item.get("snapshot") if isinstance(item.get("snapshot"), dict) else {}
        assessment = (
            item.get("safety_assessment")
            if isinstance(item.get("safety_assessment"), dict)
            else {}
        )
        timestamp = snapshot.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp.strip():
            telemetry_problems.append(f"sample {index} timestamp is missing")
        if str(snapshot.get("gpu_index", "")).strip() != str(gpu_index):
            telemetry_problems.append(f"sample {index} GPU index is missing or mismatched")
        if snapshot.get("gpu_name") != identity_fields.get("name"):
            telemetry_problems.append(f"sample {index} GPU name is missing or mismatched")
        utilization = _as_float(snapshot.get("gpu_utilization_percent"))
        memory_utilization = _as_float(snapshot.get("memory_utilization_percent"))
        memory_used = _as_float(snapshot.get("memory_used_mib"))
        sample_memory_total = _as_float(snapshot.get("memory_total_mib"))
        temperature = _as_float(snapshot.get("temperature_c"))
        if utilization is None or not 0 <= utilization <= 100:
            telemetry_problems.append(f"sample {index} GPU utilization is unreadable")
        elif utilization > GPU_IDLE_MAX_UTILIZATION_PERCENT:
            if workstation_managed and no_compute_clients:
                quality_problems.append(
                    f"sample {index} graphics/other GPU utilization is above the "
                    "comparable-idle ceiling while the readable compute-client "
                    "query is empty"
                )
            else:
                suffix = (
                    " and compute-client absence is not proven"
                    if workstation_managed
                    else ""
                )
                telemetry_problems.append(
                    f"sample {index} GPU utilization is above the comparable-idle "
                    f"ceiling{suffix}"
                )
        if memory_utilization is None or not 0 <= memory_utilization <= 100:
            telemetry_problems.append(f"sample {index} memory utilization is unreadable")
        if memory_used is None or memory_used < 0:
            telemetry_problems.append(f"sample {index} memory.used is unreadable")
        if sample_memory_total is None or sample_memory_total <= 0:
            telemetry_problems.append(f"sample {index} memory.total is unreadable")
        if temperature is None or temperature >= GPU_IDLE_MAX_TEMPERATURE_C:
            quality_problems.append(f"sample {index} GPU temperature is not below 65 C")
        expected_assessment = assess_gpu_safety(
            snapshot,
            power_limit_drop_tolerance_w=POWER_LIMIT_DROP_TOLERANCE_W,
        )
        if assessment != expected_assessment:
            quality_problems.append(
                f"sample {index} recorded quality assessment differs from raw telemetry"
            )
        if expected_assessment.get("power_limit_telemetry_complete") is not True:
            quality_problems.append(
                f"sample {index} current/default power limits are incomplete"
            )
        if expected_assessment.get("power_limit_drop_detected") is True:
            quality_problems.append(f"sample {index} power limit is below default")
        if expected_assessment.get("dangerous_slowdown_active") is True:
            quality_problems.append(
                f"sample {index} firmware/driver slowdown flag is active"
            )
        slowdown_fields = (
            "clock_event_sw_thermal_slowdown",
            "clock_event_hw_slowdown",
            "clock_event_hw_thermal_slowdown",
            "clock_event_hw_power_brake_slowdown",
        )
        if any(
            not isinstance(snapshot.get(field), str)
            or not snapshot.get(field, "").strip()
            for field in slowdown_fields
        ):
            quality_problems.append(
                f"sample {index} firmware/driver slowdown flags are incomplete"
            )
    gates.append(
        _gate(
            "fresh_idle_gpu_telemetry",
            not telemetry_problems,
            (
                "; ".join(telemetry_problems)
                if telemetry_problems
                else "fresh readable GPU identity and comparable-idle telemetry are complete"
            ),
            "gpu_identity",
            "idle_gpu_telemetry",
            "compute_processes",
            required=gate_is_required("fresh_idle_gpu_telemetry"),
        )
    )
    gates.append(
        _gate(
            "idle_gpu_quality_diagnostics",
            not quality_problems,
            (
                "; ".join(quality_problems)
                if quality_problems
                else "idle temperature, power-limit and slowdown diagnostics are clean"
            ),
            "idle_gpu_telemetry",
            "compute_processes",
            required=gate_is_required("idle_gpu_quality_diagnostics"),
        )
    )

    profiles = evidence.get("linux_power_profiles")
    profiles = profiles if isinstance(profiles, dict) else {}
    ppctl = profiles.get("powerprofilesctl")
    ppctl = ppctl if isinstance(ppctl, dict) else {}
    platform_profile = profiles.get("platform_profile")
    platform_profile = platform_profile if isinstance(platform_profile, dict) else {}
    profiles_ok = (
        ppctl.get("available") is True
        and ppctl.get("value") == "performance"
        and platform_profile.get("readable") is True
        and platform_profile.get("value") == "performance"
    )
    gates.append(
        _gate(
            "linux_power_profiles_performance",
            profiles_ok,
            "both Linux powerprofilesctl and ACPI platform_profile are performance" if profiles_ok else "both Linux power profiles must be readable and equal performance",
            "linux_power_profiles",
            required=gate_is_required("linux_power_profiles_performance"),
        )
    )

    ac = evidence.get("ac_power")
    ac = ac if isinstance(ac, dict) else {}
    ac_ok = ac.get("available") is True and ac.get("mains_online") is True
    gates.append(
        _gate(
            "ac_mains_online",
            ac_ok,
            "a Mains power_supply is online" if ac_ok else "no readable online Mains power_supply evidence",
            "ac_power",
            required=gate_is_required("ac_mains_online"),
        )
    )

    platform = evidence.get("platform_thermal")
    platform = platform if isinstance(platform, dict) else {}
    values = platform.get("values") if isinstance(platform.get("values"), dict) else {}
    required_thermal_values = (
        "dell_fan1_rpm",
        "dell_fan2_rpm",
        "thermal_tvga_c",
        "thermal_tcpu_c",
        "thermal_tskn_c",
    )
    platform_ok = platform.get("available") is True and all(
        (_as_float(values.get(name)) or 0.0) > 0.0 for name in required_thermal_values
    )
    gates.append(
        _gate(
            "platform_fan_acpi_snapshot",
            platform_ok,
            "both Dell fans and TVGA/TCPU/TSKN temperatures are readable" if platform_ok else "Dell fan RPM and TVGA/TCPU/TSKN idle values are incomplete",
            "platform_thermal",
            required=gate_is_required("platform_fan_acpi_snapshot"),
        )
    )

    xid = evidence.get("xid_current_boot")
    xid = xid if isinstance(xid, dict) else {}
    xid_source = xid.get("source")
    xid_command = xid.get("read_only_command")
    expected_xid_commands = {
        "journalctl": [
            "journalctl",
            "--quiet",
            "-k",
            "-b",
            "--no-pager",
            "--grep",
            "NVRM: Xid",
        ],
        "dmesg": ["dmesg", "--color=never"],
    }
    xid_ok = (
        xid.get("available") is True
        and not isinstance(xid.get("count"), bool)
        and xid.get("count") == 0
        and xid.get("lines") == []
        and xid_source in expected_xid_commands
        and xid_command == expected_xid_commands.get(xid_source)
        and _is_read_only_command(xid_command)
    )
    gates.append(
        _gate(
            "no_xid_current_boot",
            xid_ok,
            "current boot Xid log is readable and empty" if xid_ok else "current boot Xid evidence is unavailable or non-empty",
            "xid_current_boot",
            required=gate_is_required("no_xid_current_boot"),
        )
    )

    guard_ok, guard_detail = _records_match_current(
        evidence.get("guard_code"), project_root, GUARD_PATHS
    )
    gates.append(
        _gate(
            "current_guard_code_hashes",
            guard_ok,
            guard_detail,
            "guard_code",
            required=gate_is_required("current_guard_code_hashes"),
        )
    )

    incident = evidence.get("incident_provenance")
    incident = incident if isinstance(incident, dict) else {}
    sources = incident.get("source_artifacts")
    sources = sources if isinstance(sources, list) else []
    by_path = {item.get("path"): item for item in sources if isinstance(item, dict)}
    incident_problems: list[str] = []
    for relative, expected in INCIDENT_SOURCE_HASHES.items():
        item = by_path.get(relative)
        current = _project_path(project_root, relative)
        if item is None or item.get("sha256") != expected:
            incident_problems.append(f"recorded incident hash missing/mismatched: {relative}")
        else:
            try:
                current_hash = sha256_file(current) if current.is_file() else None
            except OSError:
                current_hash = None
            if current_hash != expected:
                incident_problems.append(
                    f"current incident artifact missing/mutated: {relative}"
                )
    document = incident.get("incident_document")
    document = document if isinstance(document, dict) else {}
    document_path = _project_path(project_root, INCIDENT_DOCUMENT_PATH)
    try:
        document_hash = (
            sha256_file(document_path) if document_path.is_file() else None
        )
    except OSError:
        document_hash = None
    if document.get("sha256") != document_hash or document_hash is None:
        incident_problems.append("incident runbook hash is missing or stale")
    gates.append(
        _gate(
            "incident_provenance_integrity",
            not incident_problems,
            "; ".join(incident_problems) if incident_problems else "incident artifacts and runbook match the evidence bundle",
            "incident_provenance",
            required=gate_is_required("incident_provenance_integrity"),
        )
    )

    emitted_ids = [item["id"] for item in gates]
    expected_ids = set(LEGACY_REQUIRED_GATE_IDS)
    contract_complete = (
        len(emitted_ids) == len(set(emitted_ids))
        and set(emitted_ids) == expected_ids
    )
    if not contract_complete:
        policy_gate = gates[0]
        policy_gate["passed"] = False
        policy_gate["detail"] = (
            f"{policy_gate['detail']}; verifier gate set differs from the versioned "
            "policy contract"
        )

    failed = [item for item in gates if item["required"] and not item["passed"]]
    status = "blocked" if failed else "ready_for_operator_review"
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified_at_utc": now.isoformat(),
        "status": status,
        "operating_policy": deepcopy(operating_policy),
        "all_required_evidence_present": not failed,
        "sustained_load_authorized": False,
        "authorization_policy": (
            "Execution authority is the user's explicit instruction. This read-only "
            "technical bundle does not cryptographically authenticate identity, grant "
            "authority, change workstation settings, or start a benchmark."
        ),
        "gates": gates,
        "failed_gate_ids": [item["id"] for item in failed],
    }


def require_reentry_evidence(
    report_path: Path | str,
    *,
    project_root: Path = PROJECT_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail closed unless a current technical evidence artifact is ready.

    GPU execution entry points call this immediately before doing GPU work.
    Passing this prerequisite neither authenticates an operator nor grants load
    authority.  The user's explicit instruction is execution authority.
    """

    path = Path(report_path)
    if not path.is_absolute():
        path = _project_path(project_root, str(path))
    else:
        path = path.resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ReentryEvidenceError(
                "re-entry evidence report must be inside the project root"
            ) from exc
    if not path.is_file():
        raise ReentryEvidenceError(f"re-entry evidence report is missing: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReentryEvidenceError(f"cannot read re-entry evidence: {exc}") from exc
    if not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA_VERSION:
        raise ReentryEvidenceError("re-entry evidence schema is missing or unsupported")
    verification = verify_evidence(evidence, project_root=project_root, now=now)
    if verification["status"] != "ready_for_operator_review":
        failed = ", ".join(verification["failed_gate_ids"])
        raise ReentryEvidenceError(
            "GPU sustained-load re-entry evidence is blocked; failed gates: "
            f"{failed or 'unknown'}"
        )
    return {
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "collected_at_utc": evidence.get("collected_at_utc"),
        "gpu_identity": evidence.get("gpu_identity"),
        "operating_policy": verification.get("operating_policy"),
        "verification": verification,
        "load_authority_granted_by_gate": False,
        "execution_authority": {
            "source": "explicit_user_instruction",
            "granted_by_this_evidence": False,
            "cryptographic_identity_authentication": False,
        },
    }


def collect_evidence(
    *,
    project_root: Path = PROJECT_ROOT,
    declaration_path: Path | None = None,
    gpu_index: int = 0,
    operating_policy_id: str = DEFAULT_OPERATING_POLICY_ID,
    runner: CommandRunner = run_read_only_command,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    operating_policy = operating_policy_contract(operating_policy_id)
    if declaration_path is not None:
        if declaration_path.is_absolute():
            declaration_path = declaration_path.resolve()
            try:
                declaration_path.relative_to(project_root)
            except ValueError as exc:
                raise ValueError("operator declaration must stay inside project root") from exc
        else:
            declaration_path = _project_path(project_root, str(declaration_path))
    declaration, declaration_file = load_declaration(declaration_path)
    collected_at = utc_now()
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "collected_not_verified",
        "collected_at_utc": collected_at,
        "operating_policy": operating_policy,
        "collection_policy": {
            "read_only": True,
            "gpu_stress_performed": False,
            "settings_changed": False,
            "sudo_executed": False,
            "benchmark_started": False,
            "gate_auto_clear_supported": False,
            "operating_policy_id": operating_policy_id,
        },
        "gpu_index": gpu_index,
        "gpu_identity": collect_gpu_identity(gpu_index, runner),
        "compute_processes": collect_compute_processes(gpu_index, runner),
        "idle_gpu_telemetry": collect_idle_gpu_telemetry(
            gpu_index, runner=runner, sleeper=sleeper
        ),
        "linux_power_profiles": collect_linux_power_profiles(runner),
        "ac_power": collect_ac_online(),
        "bios_thermal_management": collect_bios_thermal_management(),
        "platform_thermal": collect_platform_thermal(),
        "xid_current_boot": collect_xid_log(runner),
        "operator_declaration_file": declaration_file,
        "operator_declaration": declaration,
        "guard_code": _file_records(project_root, GUARD_PATHS),
        "incident_provenance": collect_incident_provenance(project_root),
    }
    evidence["verification"] = verify_evidence(evidence, project_root=project_root)
    evidence["status"] = evidence["verification"]["status"]
    return evidence


def render_markdown(evidence: dict[str, Any]) -> str:
    verification = evidence.get("verification", {})
    operating_policy = verification.get("operating_policy") or {}
    identity = evidence.get("gpu_identity", {}).get("fields") or {}
    samples = evidence.get("idle_gpu_telemetry", {}).get("samples") or []
    last_snapshot = samples[-1].get("snapshot", {}) if samples else {}
    compute = evidence.get("compute_processes", {})
    bios = evidence.get("bios_thermal_management", {})
    dell_mapping = bios.get("dell_platform_profile_mapping", {})
    dell_observations = dell_mapping.get("observations", {})
    dell_vendor = dell_observations.get("sys_vendor", {})
    dell_kernel = dell_observations.get("kernel_osrelease", {})
    dell_module = dell_observations.get("module_name", {})
    dell_module_srcversion = dell_observations.get("module_srcversion", {})
    dell_module_initstate = dell_observations.get("module_initstate", {})
    dell_provider = dell_observations.get("provider_name", {})
    dell_profile = dell_observations.get("provider_profile", {})
    dell_acpi_profile = dell_observations.get("acpi_platform_profile", {})
    dell_wmi_values = dell_observations.get("wmi_possible_values", {})
    mapping_source = dell_mapping.get("mapping_source", {})
    profiles = evidence.get("linux_power_profiles", {})
    ppctl = profiles.get("powerprofilesctl", {})
    platform_profile = profiles.get("platform_profile", {})
    ac = evidence.get("ac_power", {})
    thermal = evidence.get("platform_thermal", {}).get("values", {})
    lines = [
        "# GPU sürekli-yük yeniden giriş kanıtı",
        "",
        f"- Durum: `{verification.get('status', 'blocked')}`",
        f"- İşletim politikası: `{operating_policy.get('id')}`",
        "- Yük yetkisi: kullanıcının açık talimatıdır; bu teknik kanıt kimliği kriptografik olarak doğrulamaz, yetki vermez ve test başlatmaz.",
        f"- Toplama zamanı (UTC): `{evidence.get('collected_at_utc')}`",
        f"- GPU: `{identity.get('name')}`; driver `{identity.get('driver_version')}`; UUID `{identity.get('uuid')}`",
        f"- Son boşta örnek: util `%{last_snapshot.get('gpu_utilization_percent')}`, sıcaklık `{last_snapshot.get('temperature_c')} C`, current/default limit `{last_snapshot.get('power_current_limit_w')}/{last_snapshot.get('power_default_limit_w')} W`",
        f"- Compute istemcileri: sorgu okunabilir `{compute.get('available')}`, adet `{compute.get('count')}`",
        f"- Linux profilleri: powerprofilesctl `{ppctl.get('value')}`, platform_profile `{platform_profile.get('value')}`; Mains AC online `{ac.get('mains_online')}`",
        f"- Platform boşta: fan1/fan2 `{thermal.get('dell_fan1_rpm')}/{thermal.get('dell_fan2_rpm')} RPM`, TVGA/TCPU/TSKN `{thermal.get('thermal_tvga_c')}/{thermal.get('thermal_tcpu_c')}/{thermal.get('thermal_tskn_c')} C`",
        f"- BIOS ThermalManagement: `{bios.get('value')}` (doğrudan okunabilir: `{bios.get('readable')}`)",
        f"- Strict Dell BIOS alternatifi: `{dell_mapping.get('strict_conditions_met')}`; sys_vendor `{dell_vendor.get('value')}`, provider `{dell_provider.get('value')}`, provider/ACPI profil `{dell_profile.get('value')}/{dell_acpi_profile.get('value')}`, WMI enum `{dell_wmi_values.get('value')}`",
        f"- Dell runtime bağı: kernel `{dell_kernel.get('value')}`, modül `{dell_module.get('value')}` srcversion `{dell_module_srcversion.get('value')}` state `{dell_module_initstate.get('value')}`",
        f"- Dell eşleme kaynağı: `{mapping_source.get('source_revision')}` commit `{mapping_source.get('source_commit')}` `{mapping_source.get('source_path')}`; `{mapping_source.get('semantic_mapping')}`",
        "",
        "## Teknik kapılar ve tanılamalar",
        "",
        "| Kapı | Zorunlu | Sonuç | Ayrıntı |",
        "|---|---:|---|---|",
    ]
    for gate in verification.get("gates", []):
        result = (
            "PASS"
            if gate.get("passed")
            else ("BLOCK" if gate.get("required") else "INFO")
        )
        detail = str(gate.get("detail", "")).replace("|", "\\|")
        lines.append(
            f"| `{gate.get('id')}` | `{str(bool(gate.get('required'))).lower()}` | {result} | {detail} |"
        )
    lines.extend(
        [
            "",
            "## Eksik BIOS kanıtı komutu",
            "",
            f"Collector bu komutu çalıştırmadı. Doğrudan root-only değer tercih edilen kanıttır. Gerekirse operatörün çalıştıracağı tam komut: `{bios.get('evidence_missing_command', THERMAL_MANAGEMENT_SUDO_COMMAND)}`",
            "",
            "`legacy_strict_physical` modunda root-only değer okunamıyorsa BIOS kapısı yalnız eksiksiz strict Dell eşlemesiyle veya gerçekten gözlemlenmiş deklarasyonla geçebilir. `workstation_managed` modunda BIOS, adaptör, ePSA ve operator deklarasyonu yalnız bilgi amaçlıdır.",
            "",
            "## Güvenlik sınırı",
            "",
            "Bu rapor yalnız salt-okunur teknik gözlemdir. BIOS, EC ve GPU sürücüsü workstation donanım korumasını yürütür. `ready_for_operator_review` durumu kimlik doğrulaması veya yetki makbuzu değildir; yürütme yetkisi kullanıcının açık talimatıdır.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="write a null physical-observation declaration template")
    template_parser.add_argument("--output", type=Path, required=True)

    collect_parser = subparsers.add_parser("collect", help="collect read-only current evidence")
    collect_parser.add_argument("--declaration", type=Path)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--markdown", type=Path, required=True)
    collect_parser.add_argument("--gpu-index", type=int, default=0)
    collect_parser.add_argument(
        "--operating-policy",
        choices=tuple(OPERATING_POLICY_CONTRACTS),
        default=DEFAULT_OPERATING_POLICY_ID,
        help="workstation-managed by default; legacy strict physical gates remain selectable",
    )

    verify_parser = subparsers.add_parser("verify", help="fail closed while any required evidence is missing/stale")
    verify_parser.add_argument("--report", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "template":
        _write_json(args.output, declaration_template())
        print(json.dumps({"status": "blocking_template_written", "output": str(args.output)}))
        return 0
    if args.command == "collect":
        evidence = collect_evidence(
            declaration_path=args.declaration,
            gpu_index=args.gpu_index,
            operating_policy_id=args.operating_policy,
        )
        _write_json(args.output, evidence)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(evidence), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "sustained_load_authorized": False,
                    "failed_gate_ids": evidence["verification"]["failed_gate_ids"],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        evidence = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read evidence report: {exc}") from exc
    verification = verify_evidence(evidence)
    if args.output:
        _write_json(args.output, verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["status"] == "ready_for_operator_review" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
