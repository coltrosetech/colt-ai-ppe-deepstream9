"""Read-only admin projection for the independently accepted GPU Lease V5."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "022480fcc82b80e2a707a97f595fa14aa2737c7ef15262b53c84e9cdb21fd120"
PINS = {
    "validation/gpu_lease_v5.py": (
        90_029,
        "c06aae2aaadca1d8a0f126874da9a3fd8b8f7c21177abdc7cf4e9892e07740b9",
    ),
    "validation/contracts/gpu-lease-v5.json": (
        3_581,
        "039322d84b3ad495e6bf4aa16966c11c9f1dfda981908908bd7f77cc0b58b38d",
    ),
    "validation/schemas/gpu-lease-workload-plan-v5.schema.json": (
        6_201,
        "7b2530ba038fef6b7011920eb565e6f5a43dc7f7ec4b7e04d40da7cccf7c7e5a",
    ),
    "validation/schemas/gpu-lease-activation-receipt-v5.schema.json": (
        6_353,
        "21b4fa59647ae89b371ac297d4ce8e4debe12742dd2d47eafd1a3e8a3513e284",
    ),
}


class GPULeaseV5StatusError(RuntimeError):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise GPULeaseV5StatusError(message)


def _roots() -> tuple[Path, Path]:
    workspace = Path(os.getenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", ROOT))
    validation = Path(os.getenv("DEEPSAFE_VALIDATION_ROOT", workspace / "validation/results"))
    return workspace, validation


def _host_tool_paths() -> dict[str, Path]:
    return {
        "docker_cli": Path(
            os.getenv("DEEPSAFE_GPU_LEASE_HOST_DOCKER", "/usr/bin/docker")
        ),
        "nvidia_smi": Path(
            os.getenv("DEEPSAFE_GPU_LEASE_HOST_NVIDIA_SMI", "/usr/bin/nvidia-smi")
        ),
    }


def _observe_host_tool(path: Path) -> dict[str, Any]:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"host tool is not regular: {path}")
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        _expect(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"host tool changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _host_tool_status(contract: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    all_current = True
    paths = _host_tool_paths()
    for key in ("docker_cli", "nvidia_smi"):
        expected = contract["tools"][key]
        current = _observe_host_tool(paths[key])
        matches = (
            current["bytes"] == expected["bytes"]
            and current["sha256"] == expected["sha256"]
        )
        all_current = all_current and matches
        observed[key] = {
            "path": expected["path"],
            "expected_bytes": expected["bytes"],
            "expected_sha256": expected["sha256"],
            "current_bytes": current["bytes"],
            "current_sha256": current["sha256"],
            "matches_contract": matches,
        }
    return {"current": all_current, "tools": observed}


def _read_exact(path: Path, pin: tuple[int, str]) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _expect(stat.S_ISREG(before.st_mode), f"not regular: {path}")
        _expect(before.st_nlink == 1, f"unexpected link count: {path}")
        _expect(before.st_mode & 0o222 == 0, f"writable control: {path}")
        _expect(before.st_size == pin[0], f"control byte count differs: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        _expect(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"control changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    _expect(len(raw) == pin[0] and digest.hexdigest() == pin[1], f"control pin differs: {path}")
    return raw


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _expect(key not in result, f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GPULeaseV5StatusError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPULeaseV5StatusError(f"invalid JSON: {label}") from exc
    _expect(isinstance(value, dict), f"JSON root differs: {label}")
    return value


def _validate_contract(contract: dict[str, Any]) -> None:
    _expect(contract.get("schema_version") == "deepsafe.gpu-lease-contract/v5", "schema differs")
    _expect(contract.get("status") == "locked-no-live-plan", "contract status differs")
    _expect(contract.get("contract_fingerprint_sha256") == FINGERPRINT, "fingerprint differs")
    _expect(
        contract["implementation"]
        == {
            "path": "validation/gpu_lease_v5.py",
            "bytes": PINS["validation/gpu_lease_v5.py"][0],
            "sha256": PINS["validation/gpu_lease_v5.py"][1],
        },
        "implementation projection differs",
    )
    schemas = contract["schemas"]
    for key, relative in (
        ("workload_plan", "validation/schemas/gpu-lease-workload-plan-v5.schema.json"),
        ("activation_receipt", "validation/schemas/gpu-lease-activation-receipt-v5.schema.json"),
    ):
        _expect(
            schemas[key]
            == {"path": relative, "bytes": PINS[relative][0], "sha256": PINS[relative][1]},
            f"schema projection differs: {key}",
        )
    policy = contract["activation_policy"]
    _expect(policy["plan_required"] is True, "plan requirement differs")
    _expect(policy["default_plan"] is False, "default plan overclaim")
    _expect(policy["published_live_plan"] is False, "live plan overclaim")
    _expect(policy["verification_and_execution_separate"] is True, "two-stage policy differs")
    _expect(policy["caller_argument_overrides"] is False, "argv override policy differs")
    _expect(contract["receipt_policy"]["published_receipt_in_this_delivery"] is False, "receipt overclaim")


def _unavailable() -> dict[str, Any]:
    return {
        "label": "GPU Lease V5 exact-plan yürütme kilidi",
        "state": "unavailable_integrity_error",
        "reason": "gpu_lease_v5_control_integrity_failed",
        "available": False,
        "contract_verified": False,
        "host_tools_current": False,
        "host_tools": {},
        "current_host_replay_eligible": False,
        "current_host_replay_blocker": "GPU Lease V5 control integrity failed",
        "live_plan_published": False,
        "execution_authorized": False,
        "read_only": True,
        "execution_actions_available": False,
        "production_ready": False,
        "evidence": [],
        "caveats": ["GPU Lease V5 kontrol paketi exact-pin doğrulamasından geçmedi."],
    }


def load_gpu_lease_v5_status() -> dict[str, Any]:
    workspace, validation_root = _roots()
    try:
        raw: dict[str, bytes] = {}
        for relative, pin in PINS.items():
            raw[relative] = _read_exact(workspace / relative, pin)
        contract = _strict_object(
            raw["validation/contracts/gpu-lease-v5.json"],
            "GPU Lease V5 contract",
        )
        _validate_contract(contract)
        result_root = validation_root / "gpu-leases/v5"
        info = result_root.stat()
        _expect(stat.S_ISDIR(info.st_mode), "V5 result root missing")
        _expect(stat.S_IMODE(info.st_mode) == 0o700, "V5 result root mode differs")
        _expect(not any(result_root.iterdir()), "V5 result root is not empty")
        host_tools = _host_tool_status(contract)
    except (OSError, KeyError, TypeError, GPULeaseV5StatusError):
        return _unavailable()

    host_tools_current = host_tools["current"]
    return {
        "label": "GPU Lease V5 exact-plan yürütme kilidi",
        "state": (
            "contract_verified_no_live_plan"
            if host_tools_current
            else "contract_exact_host_tool_pins_stale"
        ),
        "reason": (
            "no_published_live_plan_execution_closed"
            if host_tools_current
            else "docker_and_nvidia_smi_pins_drifted_requalification_required"
        ),
        "available": True,
        "contract_verified": True,
        "host_tools_current": host_tools_current,
        "host_tools": host_tools["tools"],
        "current_host_replay_eligible": host_tools_current,
        "current_host_replay_blocker": (
            None if host_tools_current else "trusted executable hash differs before activation"
        ),
        "contract_fingerprint_sha256": FINGERPRINT,
        "base_contract_fingerprint_sha256": contract["base_v4"]["contract_fingerprint_sha256"],
        "live_plan_published": False,
        "activation_receipt_published": False,
        "execution_authorized": False,
        "gpu_or_docker_called_during_validation": False,
        "read_only": True,
        "execution_actions_available": False,
        "production_ready": False,
        "policy": {
            "plan_required": True,
            "external_file_and_semantic_pins_required": True,
            "verification_and_execution_separate": True,
            "foreground_direct_docker_run_only": True,
            "same_uid_boundary": contract["boundary_policy"]["same_uid"],
        },
        "tests": {
            "focused_passed": 26,
            "regression_passed": 164,
            "failed": 0,
            "independent_review": "pass",
            "scope": "frozen_acceptance_baseline_not_current_host_replay",
            "p0": 0,
            "p1": 0,
            "p2": 0,
        },
        "caveats": [
            "Bu kart yalnız execution-closed sözleşmeyi gösterir; canlı GPU planı veya çalıştırma düğmesi yoktur.",
            (
                "V5 host tool pinleri günceldir."
                if host_tools_current
                else "V5 sözleşmesi exact olsa da host Docker ve nvidia-smi pinleri drift etti; V5 activation fail-closed olur ve successor/requalification gerekir."
            ),
            "26/164 test sayıları frozen bağımsız kabul baseline'ıdır; güncel host replay sonucu değildir.",
            "Gerçek plan ayrıca exact model/config, image digest, argv digest, GPU UUID ve owner pinlerini sağlamalıdır.",
        ],
        "evidence": [],
    }
