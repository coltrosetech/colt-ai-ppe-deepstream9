"""Fail-closed DeepStream 9.0 parallel three-model topology scaffold.

This module intentionally does not start Docker, query a GPU, or run inference.
It renders the configuration format used by NVIDIA's parallel inference
reference application only after every runtime/model artifact is present and
hash-bound.  A blocked plan never leaves launchable configuration files behind.

The actual process launcher is deliberately outside this module.  A caller may
request a launch authorization payload with :func:`require_launch_ready`, but
that function returns an argv/environment contract and never invokes it.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.strict_json import StrictJSONError, loads as strict_json_loads


CONTRACT_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-contract/v1"
PLAN_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-plan/v1"
FUSION_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-fusion/v1"
CAPABILITY_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-runtime-capabilities/v1"
PUBLICATION_RECEIPT_SCHEMA_VERSION = (
    "deepsafe.deepstream-fusion-publication-receipt/v1"
)
POSE_ASSOCIATION_SCHEMA_VERSION = "deepsafe.pose.postprocess-contract/v1"
PPE_ASSOCIATION_SCHEMA_VERSION = "deepsafe.ppe.postprocess-contract/v1"

ACTIVE_DEEPSTREAM_VERSION = "9.0.0"
MIGRATION_DEEPSTREAM_VERSION = "9.1"
MIGRATION_STATUS = "separate_post_campaign_qualification_required"
PARALLEL_PATTERN = "nvidia_parallel_inference_nvdsmetamux"
MAX_SOURCES = 12
PUBLICATION_RECEIPT_NAME = "publication-receipt.json"
PUBLICATION_PRIMITIVE = (
    "renameat2(RENAME_NOREPLACE)+post_rename_fd_identity+"
    "descriptor_relative_replay"
)
PUBLICATION_BUNDLE_NAMES = {
    "build-provenance.json",
    "capability-manifest.json",
    "deepstream-parallel-infer",
    "fusion-runtime.conf",
    "libdeepsafe_fusion.so.1",
    PUBLICATION_RECEIPT_NAME,
}

DEFAULT_CONTRACT = PROJECT_ROOT / "deepstream/full-stack-contract.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation/results/deepstream-full-stack/dry-run-640"

ROLE_ORDER = ("person", "pose", "ppe")
ROLE_GIE_IDS = {"person": 1, "pose": 2, "ppe": 3}
ROLE_ARTIFACTS = {
    "person": ("infer_config", "engine"),
    "pose": ("infer_config", "engine", "postprocess_library", "association_contract"),
    "ppe": ("infer_config", "engine", "postprocess_library", "association_contract"),
}
RUNTIME_ARTIFACTS = (
    "parallel_app_binary",
    "fusion_plugin",
    "capability_manifest",
    "publication_receipt",
)
JSON_ARTIFACT_REQUIREMENTS = {
    "runtime.capability_manifest": {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "status": "runtime_ready",
    },
    "runtime.publication_receipt": {
        "schema_version": PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "status": "published",
    },
    "pose.association_contract": {
        "schema_version": POSE_ASSOCIATION_SCHEMA_VERSION,
        "status": "runtime_ready",
    },
    "ppe.association_contract": {
        "schema_version": PPE_ASSOCIATION_SCHEMA_VERSION,
        "status": "runtime_ready",
    },
}
ARTIFACT_BINDINGS = (
    (
        "runtime.capability_manifest",
        "parallel_app_binary_sha256",
        "runtime.parallel_app_binary",
    ),
    (
        "runtime.capability_manifest",
        "fusion_plugin_sha256",
        "runtime.fusion_plugin",
    ),
    (
        "pose.association_contract",
        "postprocess_library_sha256",
        "pose.postprocess_library",
    ),
    (
        "ppe.association_contract",
        "postprocess_library_sha256",
        "ppe.postprocess_library",
    ),
)
GENERATED_NAMES = (
    "sources.csv",
    "metamux.txt",
    "tracker.yml",
    "parallel-inference.yml",
    "fusion-runtime.conf",
    "fusion-contract.json",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
CONTAINER_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
URI_TEXT_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9+.-]*://[A-Za-z0-9._~:/\[\]@!$&'()*+,;=%-]+$"
)
INVALID_PERCENT_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
MAX_PATH_BYTES = 4096
MAX_TEXT_ARTIFACT_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


class FullStackContractError(ValueError):
    """The topology, artifact contract, or rendered plan is invalid."""


class LaunchRejected(FullStackContractError):
    """Launch authorization was requested for a fail-closed plan."""


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullStackContractError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FullStackContractError(
            f"{where} keys mismatch; missing={missing}, extra={extra}"
        )
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: Any, where: str, minimum: int, maximum: int) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        raise FullStackContractError(
            f"{where} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _require_bool(value: Any, where: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise FullStackContractError(f"{where} must be boolean")
    if expected is not None and value is not expected:
        raise FullStackContractError(f"{where} must be {str(expected).lower()}")
    return value


def _read_strict_json(path: Path, *, max_bytes: int = 1024 * 1024) -> dict[str, Any]:
    raw = _read_stable_regular_bytes(path, max_bytes=max_bytes)
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FullStackContractError(f"invalid strict JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullStackContractError(f"JSON artifact root must be an object: {path}")
    return value


def _read_stable_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read one bounded regular-file snapshot without following symlinks."""

    try:
        info = path.lstat()
    except (OSError, ValueError) as exc:
        raise FullStackContractError(f"cannot stat file {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FullStackContractError(f"file must be regular and not a symlink: {path}")
    if info.st_nlink != 1:
        raise FullStackContractError(f"hard-linked file is not an isolated snapshot: {path}")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise FullStackContractError(
            f"file size outside (0, {max_bytes}] bytes: {path}"
        )
    try:
        if path.resolve(strict=True) != path.absolute():
            raise FullStackContractError(f"file path escapes or contains a symlink: {path}")
    except OSError as exc:
        raise FullStackContractError(f"cannot resolve file {path}: {exc}") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as exc:
        raise FullStackContractError(f"cannot open stable file snapshot {path}: {exc}") from exc
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or any(getattr(info, field) != getattr(before, field) for field in identity_fields)
        ):
            raise FullStackContractError(f"file changed before snapshot: {path}")
        try:
            descriptor_target = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        except OSError as exc:
            raise FullStackContractError(
                f"cannot resolve stable file descriptor for {path}: {exc}"
            ) from exc
        if descriptor_target != path.absolute():
            raise FullStackContractError(f"file descriptor path drifted: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise FullStackContractError(f"file exceeds {max_bytes} bytes: {path}")
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise FullStackContractError(f"file disappeared during snapshot: {path}") from exc
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(before, field) != getattr(current, field)
            for field in identity_fields
        ):
            raise FullStackContractError(f"file changed during snapshot: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise FullStackContractError(f"cannot read stable file snapshot {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise FullStackContractError(f"cannot atomically write {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _utf8_size(value: str, where: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise FullStackContractError(f"{where} must be valid UTF-8 text") from exc


def _relative_path(value: str, where: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
    ):
        raise FullStackContractError(f"{where} must be a non-empty relative path")
    if (
        _utf8_size(value, where) > MAX_PATH_BYTES
        or "\\" in value
        or value.startswith("//")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise FullStackContractError(f"{where} must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FullStackContractError(f"{where} must be normalized and project-relative")
    return path


def _container_path(value: str, where: str) -> str:
    if (
        not isinstance(value, str)
        or value == "/"
    ):
        raise FullStackContractError(
            f"{where} must be an injection-safe absolute container path"
        )
    if (
        _utf8_size(value, where) > MAX_PATH_BYTES
        or CONTAINER_PATH_RE.fullmatch(value) is None
        or "//" in value
    ):
        raise FullStackContractError(
            f"{where} must be an injection-safe absolute container path"
        )
    path = Path(value)
    if path.as_posix() != value or any(part in {".", ".."} for part in path.parts):
        raise FullStackContractError(f"{where} must be normalized")
    return value


def _plan_file_path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FullStackContractError(f"{where} must be a non-empty normalized path")
    if (
        _utf8_size(value, where) > MAX_PATH_BYTES
        or "\\" in value
        or value.startswith("//")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise FullStackContractError(f"{where} must be a non-empty normalized path")
    path = Path(value)
    if path.is_absolute():
        if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
            raise FullStackContractError(f"{where} must be a non-empty normalized path")
        return path
    return _relative_path(value, where)


def _validate_json_requirements(value: Any, where: str) -> None:
    if value is None:
        return
    requirements = _exact_keys(value, {"schema_version", "status"}, where)
    for key in ("schema_version", "status"):
        if not isinstance(requirements[key], str) or not requirements[key]:
            raise FullStackContractError(f"{where}.{key} must be a non-empty string")


def _validate_artifact_slot(value: Any, where: str) -> None:
    slot = _exact_keys(
        value,
        {"host_path", "container_path", "sha256", "json_requirements"},
        where,
    )
    if slot["host_path"] is None:
        if any(slot[key] is not None for key in ("container_path", "sha256", "json_requirements")):
            raise FullStackContractError(
                f"{where} unconfigured slot must use null for every field"
            )
        return
    _relative_path(slot["host_path"], f"{where}.host_path")
    _container_path(slot["container_path"], f"{where}.container_path")
    if not isinstance(slot["sha256"], str) or not SHA256_RE.fullmatch(slot["sha256"]):
        raise FullStackContractError(f"{where}.sha256 must be lowercase SHA-256")
    _validate_json_requirements(slot["json_requirements"], f"{where}.json_requirements")


def _validate_artifact_json_policy(slot: dict[str, Any], label: str, where: str) -> None:
    expected = JSON_ARTIFACT_REQUIREMENTS.get(label)
    requirements = slot["json_requirements"]
    if expected is None:
        if requirements is not None:
            raise FullStackContractError(
                f"{where}.json_requirements must be null for a non-JSON artifact"
            )
        return
    if slot["host_path"] is None:
        # The all-null unconfigured form was already enforced by the slot validator.
        return
    if requirements != expected:
        raise FullStackContractError(
            f"{where}.json_requirements must exactly require status "
            f"{expected['status']} for {expected['schema_version']}"
        )


def validate_contract(value: Any) -> dict[str, Any]:
    """Validate the exact, versioned full-stack input contract."""

    contract = _exact_keys(
        value,
        {"schema_version", "runtime", "limits", "topology", "profiles"},
        "contract",
    )
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise FullStackContractError("unsupported full-stack contract schema")

    runtime = _exact_keys(
        contract["runtime"],
        {
            "active_deepstream_version",
            "migration_deepstream_version",
            "migration_status",
            "parallel_app_binary",
            "fusion_plugin",
            "capability_manifest",
            "publication_receipt",
        },
        "contract.runtime",
    )
    if runtime["active_deepstream_version"] != ACTIVE_DEEPSTREAM_VERSION:
        raise FullStackContractError(
            f"active runtime must remain DeepStream {ACTIVE_DEEPSTREAM_VERSION}"
        )
    if runtime["migration_deepstream_version"] != MIGRATION_DEEPSTREAM_VERSION:
        raise FullStackContractError("DeepStream 9.1 must remain the migration target")
    if runtime["migration_status"] != MIGRATION_STATUS:
        raise FullStackContractError("DeepStream 9.1 requires a separate qualification lane")
    for name in RUNTIME_ARTIFACTS:
        where = f"contract.runtime.{name}"
        _validate_artifact_slot(runtime[name], where)
        _validate_artifact_json_policy(runtime[name], f"runtime.{name}", where)

    limits = _exact_keys(
        contract["limits"],
        {"max_sources", "max_batch_size"},
        "contract.limits",
    )
    if limits["max_sources"] != MAX_SOURCES or limits["max_batch_size"] != MAX_SOURCES:
        raise FullStackContractError("source and camera-batch limits must both equal 12")

    topology = _exact_keys(
        contract["topology"],
        {
            "pattern",
            "nvdsmetamux_required",
            "metamux_active_pad",
            "metamux_pts_tolerance_us",
            "headless",
            "perf_measurement_interval_seconds",
            "component_latency_measurement",
            "streammux_width",
            "streammux_height",
            "tracker_library",
            "tracker_config",
        },
        "contract.topology",
    )
    if topology["pattern"] != PARALLEL_PATTERN:
        raise FullStackContractError("official NVIDIA parallel-inference pattern is mandatory")
    _require_bool(topology["nvdsmetamux_required"], "contract.topology.nvdsmetamux_required", True)
    if topology["metamux_active_pad"] != "sink_0":
        raise FullStackContractError("person/tracker branch sink_0 must be the active metamux pad")
    _require_int(
        topology["metamux_pts_tolerance_us"],
        "contract.topology.metamux_pts_tolerance_us",
        1,
        10_000_000,
    )
    _require_bool(topology["headless"], "contract.topology.headless", True)
    _require_int(
        topology["perf_measurement_interval_seconds"],
        "contract.topology.perf_measurement_interval_seconds",
        1,
        60,
    )
    _require_bool(
        topology["component_latency_measurement"],
        "contract.topology.component_latency_measurement",
        True,
    )
    _require_int(topology["streammux_width"], "contract.topology.streammux_width", 1, 16384)
    _require_int(topology["streammux_height"], "contract.topology.streammux_height", 1, 16384)
    _container_path(topology["tracker_library"], "contract.topology.tracker_library")
    _container_path(topology["tracker_config"], "contract.topology.tracker_config")

    profiles = contract["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise FullStackContractError("contract.profiles must be a non-empty object")
    for profile_id, profile_value in profiles.items():
        if not isinstance(profile_id, str) or not PROFILE_RE.fullmatch(profile_id):
            raise FullStackContractError(f"invalid profile id: {profile_id!r}")
        profile = _exact_keys(
            profile_value,
            {"input_width", "input_height", "models"},
            f"contract.profiles.{profile_id}",
        )
        _require_int(profile["input_width"], f"contract.profiles.{profile_id}.input_width", 1, 16384)
        _require_int(profile["input_height"], f"contract.profiles.{profile_id}.input_height", 1, 16384)
        models = _exact_keys(
            profile["models"], set(ROLE_ORDER), f"contract.profiles.{profile_id}.models"
        )
        for role in ROLE_ORDER:
            model = _exact_keys(
                models[role],
                {
                    "role",
                    "gie_unique_id",
                    "inference_mode",
                    "batch_semantics",
                    "tracker",
                    "metadata_output",
                    "association_target_gie",
                    "artifacts",
                },
                f"contract.profiles.{profile_id}.models.{role}",
            )
            if model["role"] != role:
                raise FullStackContractError(f"model role mismatch for {role}")
            if (
                not _is_int(model["gie_unique_id"])
                or model["gie_unique_id"] != ROLE_GIE_IDS[role]
            ):
                raise FullStackContractError(
                    f"{role} gie_unique_id must equal {ROLE_GIE_IDS[role]}"
                )
            if model["inference_mode"] != "full_frame_primary":
                raise FullStackContractError(f"{role} must be full-frame PGIE, not SGIE/ROI")
            if model["batch_semantics"] != "camera_batch":
                raise FullStackContractError(f"{role} must batch cameras")
            expected_tracker = "nvdcf" if role == "person" else "none"
            if model["tracker"] != expected_tracker:
                raise FullStackContractError(f"{role} tracker must be {expected_tracker}")
            expected_meta = "nvds_infer_tensor_meta" if role == "pose" else "nvds_object_meta"
            if model["metadata_output"] != expected_meta:
                raise FullStackContractError(f"{role} metadata output must be {expected_meta}")
            expected_target = None if role == "person" else ROLE_GIE_IDS["person"]
            target = model["association_target_gie"]
            target_valid = (
                target is None
                if expected_target is None
                else _is_int(target) and target == expected_target
            )
            if not target_valid:
                raise FullStackContractError(
                    f"{role} association target must be canonical person GIE 1"
                )
            artifacts = _exact_keys(
                model["artifacts"],
                set(ROLE_ARTIFACTS[role]),
                f"contract.profiles.{profile_id}.models.{role}.artifacts",
            )
            for artifact_name in ROLE_ARTIFACTS[role]:
                where = (
                    f"contract.profiles.{profile_id}.models.{role}.artifacts."
                    f"{artifact_name}"
                )
                _validate_artifact_slot(
                    artifacts[artifact_name],
                    where,
                )
                _validate_artifact_json_policy(
                    artifacts[artifact_name], f"{role}.{artifact_name}", where
                )
    return contract


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FullStackContractError(f"cannot read full-stack contract {path}: {exc}") from exc
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FullStackContractError(f"invalid strict contract JSON: {exc}") from exc
    return validate_contract(value), hashlib.sha256(raw).hexdigest()


def _project_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _probe_artifact(
    label: str,
    slot: dict[str, Any],
    *,
    project_root: Path,
) -> tuple[dict[str, Any], bytes | None]:
    result = {
        "label": label,
        "host_path": slot["host_path"],
        "container_path": slot["container_path"],
        "expected_sha256": slot["sha256"],
        "observed_sha256": None,
        "size_bytes": None,
        "status": "unconfigured",
    }
    if slot["host_path"] is None:
        return result, None
    path = project_root / slot["host_path"]
    try:
        info = path.lstat()
    except OSError:
        result["status"] = "missing"
        return result, None
    try:
        if path.resolve(strict=True) != path.absolute():
            result["status"] = "path_escape_or_symlink"
            return result, None
        path.resolve(strict=True).relative_to(project_root.resolve())
    except (OSError, ValueError):
        result["status"] = "path_escape_or_symlink"
        return result, None
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        result["status"] = "not_regular_file"
        return result, None
    if info.st_nlink != 1:
        result["status"] = "hardlink_forbidden"
        return result, None
    if info.st_size <= 0:
        result["status"] = "empty_file"
        return result, None

    capture_bytes = slot["json_requirements"] is not None or label.endswith(
        ".infer_config"
    )
    if capture_bytes and info.st_size > MAX_TEXT_ARTIFACT_BYTES:
        result["status"] = "text_artifact_too_large"
        return result, None

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    # A regular file could be swapped for a FIFO after lstat(). O_NONBLOCK keeps
    # the fail-closed probe from hanging before fstat() detects that race.
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        result["status"] = "changed_during_probe"
        return result, None
    snapshot: bytes | None = None
    try:
        before = os.fstat(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or any(getattr(info, field) != getattr(before, field) for field in identity_fields)
        ):
            result["status"] = "changed_during_probe"
            return result, None
        try:
            descriptor_target = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        except OSError:
            result["status"] = "changed_during_probe"
            return result, None
        if descriptor_target != path.absolute():
            result["status"] = "path_escape_or_symlink"
            return result, None

        digest = hashlib.sha256()
        captured = bytearray() if capture_bytes else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if captured is not None:
                captured.extend(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            result["status"] = "changed_during_probe"
            return result, None
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(before, field) != getattr(current, field)
            for field in identity_fields
        ):
            result["status"] = "changed_during_probe"
            return result, None
        observed = digest.hexdigest()
        snapshot = bytes(captured) if captured is not None else None
    except OSError:
        result["status"] = "changed_during_probe"
        return result, None
    finally:
        os.close(descriptor)

    result["observed_sha256"] = observed
    result["size_bytes"] = before.st_size
    if observed != slot["sha256"]:
        result["status"] = "sha256_mismatch"
        return result, snapshot
    if label == "runtime.parallel_app_binary" and before.st_mode & 0o111 == 0:
        result["status"] = "not_executable"
        return result, snapshot
    requirements = slot["json_requirements"]
    if requirements is not None:
        try:
            artifact_json = strict_json_loads(snapshot)
        except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError, TypeError):
            result["status"] = "invalid_json"
            return result, snapshot
        if not isinstance(artifact_json, dict):
            result["status"] = "invalid_json"
            return result, snapshot
        for key, expected in requirements.items():
            if artifact_json.get(key) != expected:
                result["status"] = f"json_{key}_mismatch"
                return result, snapshot
        semantic_error = _artifact_json_semantic_error(label, artifact_json)
        if semantic_error is not None:
            result["status"] = semantic_error
            return result, snapshot
    result["status"] = "ready"
    return result, snapshot


def _artifact_json_semantic_error(label: str, value: dict[str, Any]) -> str | None:
    if label == "runtime.capability_manifest":
        required = {
            "deepstream_version": ACTIVE_DEEPSTREAM_VERSION,
            "parallel_pattern": PARALLEL_PATTERN,
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            return "runtime_capability_claim_mismatch"
        features = value.get("features")
        required_features = {
            "nvdsmetamux",
            "full_frame_camera_batch",
            "nvdcf_tracker",
            "pose_tensor_track_association",
            "ppe_person_association",
            "headless_performance",
            "component_latency",
        }
        if not isinstance(features, dict) or set(features) != required_features:
            return "runtime_capability_features_mismatch"
        if any(features[key] is not True for key in required_features):
            return "runtime_capability_feature_disabled"
        if value.get("fusion_plugin_ready") is not True:
            return "runtime_fusion_plugin_not_ready"
        if value.get("gpu_integration_validated") is not True:
            return "runtime_gpu_integration_not_validated"
        if value.get("runtime_ready") is not True:
            return "runtime_ready_claim_mismatch"
        static_evidence = value.get("static_evidence")
        if not isinstance(static_evidence, dict):
            return "runtime_static_evidence_missing"
        if static_evidence.get("legacy_openpose_probe_registered") is not False:
            return "runtime_legacy_openpose_probe_registered"
        fusion_probe_count = static_evidence.get(
            "canonical_fusion_probe_install_count"
        )
        if not _is_int(fusion_probe_count) or fusion_probe_count != 1:
            return "runtime_canonical_fusion_probe_count_mismatch"
        publication = value.get("publication_contract")
        if publication != {
            "receipt_schema_version": PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "receipt_name": PUBLICATION_RECEIPT_NAME,
            "primitive": PUBLICATION_PRIMITIVE,
            "directory_identity_bound": True,
            "post_rename_inode_verified": True,
            "descriptor_relative_artifact_replay": True,
            "exact_file_set_verified": True,
            "canonical_path_reopened": True,
        }:
            return "runtime_publication_contract_mismatch"
        fusion = value.get("fusion_contract")
        required_fusion = {
            "abi_version": "0x00010000",
            "canonical_person_gie_id": 1,
            "pose_tensor_gie_id": 2,
            "ppe_object_gie_id": 3,
            "max_sources": 12,
            "missing_ppe_means": "unknown",
            "unknown_generates_violation": False,
            "ambiguous_or_occluded": "unknown_unassociated",
            "fp16_pose_adapter": True,
            "fp32_pose_adapter": True,
        }
        if fusion != required_fusion:
            return "runtime_fusion_contract_mismatch"
    elif label == "runtime.publication_receipt":
        required_claims = {
            "schema_version": PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "status": "published",
            "primitive": PUBLICATION_PRIMITIVE,
            "post_rename_inode_verified": True,
            "descriptor_relative_artifact_replay": True,
            "exact_file_set_verified": True,
            "canonical_path_reopened": True,
            "exact_file_names": sorted(PUBLICATION_BUNDLE_NAMES),
        }
        if any(value.get(key) != expected for key, expected in required_claims.items()):
            return "runtime_publication_receipt_claim_mismatch"
        for key in ("publication_id", "publication_plan_sha256", "publisher_sha256"):
            if not isinstance(value.get(key), str) or not value[key]:
                return "runtime_publication_receipt_claim_mismatch"
        if not SHA256_RE.fullmatch(value["publication_plan_sha256"]):
            return "runtime_publication_receipt_claim_mismatch"
        if not SHA256_RE.fullmatch(value["publisher_sha256"]):
            return "runtime_publication_receipt_claim_mismatch"
        try:
            _relative_path(value.get("destination"), "publication receipt destination")
        except FullStackContractError:
            return "runtime_publication_receipt_claim_mismatch"
        identity = value.get("directory_identity")
        if not isinstance(identity, dict) or set(identity) != {"device", "inode"}:
            return "runtime_publication_receipt_identity_mismatch"
        if (
            not _is_int(identity["device"])
            or identity["device"] < 0
            or not _is_int(identity["inode"])
            or identity["inode"] < 1
        ):
            return "runtime_publication_receipt_identity_mismatch"
        files = value.get("files")
        expected_files = PUBLICATION_BUNDLE_NAMES - {PUBLICATION_RECEIPT_NAME}
        if not isinstance(files, dict) or set(files) != expected_files:
            return "runtime_publication_receipt_files_mismatch"
        for metadata in files.values():
            if not isinstance(metadata, dict) or set(metadata) != {
                "sha256",
                "size_bytes",
                "mode",
            }:
                return "runtime_publication_receipt_files_mismatch"
            if (
                not isinstance(metadata["sha256"], str)
                or SHA256_RE.fullmatch(metadata["sha256"]) is None
                or not _is_int(metadata["size_bytes"])
                or metadata["size_bytes"] < 1
                or not isinstance(metadata["mode"], str)
                or re.fullmatch(r"0[0-7]{3}", metadata["mode"]) is None
                or int(metadata["mode"], 8) & 0o022
            ):
                return "runtime_publication_receipt_files_mismatch"
    elif label == "pose.association_contract":
        association = value.get("track_association")
        if not isinstance(association, dict):
            return "pose_track_association_missing"
        if association.get("identity_source") != "canonical_person_tracker":
            return "pose_identity_source_mismatch"
        if association.get("duplicate_track_id_policy") != "fail_closed":
            return "pose_duplicate_track_policy_mismatch"
    elif label == "ppe.association_contract":
        identity = value.get("identity_boundary")
        if not isinstance(identity, dict):
            return "ppe_identity_boundary_missing"
        if identity.get("track_id_source") != "canonical_ephemeral_person_tracker":
            return "ppe_identity_source_mismatch"
        if identity.get("biometric_identity") is not False:
            return "ppe_biometric_identity_forbidden"
        association = value.get("association")
        if not isinstance(association, dict):
            return "ppe_person_association_missing"
        if association.get("duplicate_track_id_policy") != "fail_closed":
            return "ppe_duplicate_track_policy_mismatch"
        if association.get("confirmed_tracks_required") is not True:
            return "ppe_confirmed_tracks_required"
        observation = value.get("equipment_observation")
        if not isinstance(observation, dict):
            return "ppe_equipment_observation_missing"
        if observation.get("equipment") != ["helmet", "hi_vis"]:
            return "ppe_required_attributes_mismatch"
        if (
            observation.get("missing_detection_means") != "unknown"
            or observation.get("unknown_generates_violation") is not False
        ):
            return "ppe_unknown_policy_mismatch"
    return None


def _parse_infer_config(content: bytes, label: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=True,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=None,
    )
    parser.optionxform = str
    try:
        parser.read_string(content.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise FullStackContractError(f"invalid nvinfer config {label}: {exc}") from exc
    if "property" not in parser:
        raise FullStackContractError(f"nvinfer config lacks [property]: {label}")
    return parser


def _infer_config_errors(
    role: str,
    config_snapshot: bytes,
    engine_container_path: str,
    *,
    source_count: int,
    max_batch_size: int,
) -> list[str]:
    try:
        properties = _parse_infer_config(config_snapshot, f"{role}.infer_config")["property"]
    except FullStackContractError:
        return [f"{role}.infer_config:invalid_key_file"]
    errors: list[str] = []
    expected_values = {
        "model-engine-file": engine_container_path,
        "gie-unique-id": str(ROLE_GIE_IDS[role]),
        "process-mode": "1",
    }
    if role in {"person", "ppe"}:
        expected_values["network-type"] = "0"
    for key, expected in expected_values.items():
        if properties.get(key) != expected:
            errors.append(f"{role}.infer_config:{key}_mismatch")
    try:
        configured_batch = int(properties.get("batch-size", ""), 10)
    except ValueError:
        configured_batch = -1
    if not source_count <= configured_batch <= max_batch_size:
        errors.append(f"{role}.infer_config:batch_size_not_camera_batch_compatible")
    if role == "pose" and properties.get("output-tensor-meta") != "1":
        errors.append("pose.infer_config:output_tensor_meta_required")
    return errors


def _normalize_sources(sources: Iterable[str], max_sources: int) -> list[str]:
    values = list(sources)
    if not 1 <= len(values) <= max_sources:
        raise FullStackContractError(f"source count must be in [1, {max_sources}]")
    normalized: list[str] = []
    for index, uri in enumerate(values):
        if not isinstance(uri, str) or not uri:
            raise FullStackContractError(f"source {index} URI is empty or too long")
        if _utf8_size(uri, f"source {index} URI") > MAX_PATH_BYTES:
            raise FullStackContractError(f"source {index} URI is empty or too long")
        if any(character.isspace() or ord(character) == 127 for character in uri):
            raise FullStackContractError(
                f"source {index} URI contains whitespace/control characters"
            )
        try:
            parsed = urlsplit(uri)
            parsed_port = parsed.port
            parsed_hostname = parsed.hostname
        except ValueError as exc:
            raise FullStackContractError(f"source {index} URI is malformed: {exc}") from exc
        if (
            parsed.scheme not in {"file", "rtsp", "rtsps"}
            or not uri.startswith(f"{parsed.scheme}://")
        ):
            raise FullStackContractError(
                f"source {index} must use lowercase file://, rtsp://, or rtsps://"
            )
        if parsed.username is not None or parsed.password is not None:
            raise FullStackContractError(
                f"source {index} URI userinfo/credentials are forbidden in persisted plans"
            )
        if parsed.query or parsed.fragment:
            raise FullStackContractError(
                f"source {index} URI query/fragment is forbidden in persisted plans"
            )
        if parsed.scheme.lower() == "file" and parsed.netloc not in {"", "localhost"}:
            raise FullStackContractError(
                f"source {index} file URI authority must be empty or localhost"
            )
        if parsed.scheme.lower() == "file" and not parsed.path.startswith("/"):
            raise FullStackContractError(f"source {index} file URI must be absolute")
        if parsed.scheme.lower() in {"rtsp", "rtsps"} and (
            not parsed.netloc or not parsed_hostname or parsed.netloc.endswith(":")
        ):
            raise FullStackContractError(f"source {index} RTSP URI must include an authority")
        if URI_TEXT_RE.fullmatch(uri) is None or INVALID_PERCENT_ESCAPE_RE.search(uri):
            raise FullStackContractError(
                f"source {index} URI contains non-RFC-safe or malformed text"
            )
        # Accessing .port above also rejects non-numeric/out-of-range ports.
        _ = parsed_port
        normalized.append(uri)
    return normalized


def _artifact_checks(
    contract: dict[str, Any],
    profile_id: str,
    *,
    project_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, bytes],
]:
    checks: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    snapshots: dict[str, bytes] = {}
    runtime = contract["runtime"]
    for name in RUNTIME_ARTIFACTS:
        label = f"runtime.{name}"
        check, snapshot = _probe_artifact(label, runtime[name], project_root=project_root)
        checks.append(check)
        by_label[label] = check
        if snapshot is not None:
            snapshots[label] = snapshot
    models = contract["profiles"][profile_id]["models"]
    for role in ROLE_ORDER:
        for name in ROLE_ARTIFACTS[role]:
            label = f"{role}.{name}"
            check, snapshot = _probe_artifact(
                label, models[role]["artifacts"][name], project_root=project_root
            )
            checks.append(check)
            by_label[label] = check
            if snapshot is not None:
                snapshots[label] = snapshot
    return checks, by_label, snapshots


def _read_bundle_entry(
    directory_fd: int, name: str, *, maximum: int
) -> tuple[bytes, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum
        ):
            raise FullStackContractError("bundle entry is not an isolated bounded file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise FullStackContractError("bundle entry exceeds size bound")
        after = os.fstat(descriptor)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise FullStackContractError("bundle entry changed during replay")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _open_bundle_directory(project_root: Path, destination: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(project_root.resolve(), flags)
    try:
        for part in destination.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise FullStackContractError("publication destination is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _publication_receipt_error(
    contract: dict[str, Any],
    *,
    project_root: Path,
    by_label: dict[str, dict[str, Any]],
    snapshots: dict[str, bytes],
) -> str | None:
    label = "runtime.publication_receipt"
    if by_label[label]["status"] != "ready":
        return None
    snapshot = snapshots.get(label)
    try:
        receipt = strict_json_loads(snapshot)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError, TypeError):
        return "stable_json_snapshot_missing"
    if not isinstance(receipt, dict):
        return "stable_json_snapshot_missing"
    runtime = contract["runtime"]
    try:
        destination = _relative_path(
            receipt["destination"], "runtime publication receipt destination"
        )
    except (KeyError, FullStackContractError):
        return "destination_mismatch"
    expected_host_names = {
        "parallel_app_binary": "deepstream-parallel-infer",
        "fusion_plugin": "libdeepsafe_fusion.so.1",
        "capability_manifest": "capability-manifest.json",
        "publication_receipt": PUBLICATION_RECEIPT_NAME,
    }
    expected_container_names = dict(expected_host_names)
    host_parents: set[Path] = set()
    container_parents: set[Path] = set()
    for slot_name, expected_name in expected_host_names.items():
        slot = runtime[slot_name]
        if slot["host_path"] is None or slot["container_path"] is None:
            return "runtime_slot_unconfigured"
        host_path = Path(slot["host_path"])
        container_path = Path(slot["container_path"])
        if host_path.name != expected_name or container_path.name != expected_container_names[slot_name]:
            return "runtime_slot_name_mismatch"
        host_parents.add(host_path.parent)
        container_parents.add(container_path.parent)
    if host_parents != {destination} or len(container_parents) != 1:
        return "destination_mismatch"
    try:
        directory_fd = _open_bundle_directory(project_root, destination)
    except (OSError, FullStackContractError):
        return "directory_open_failed"
    try:
        before = os.fstat(directory_fd)
        identity = receipt["directory_identity"]
        if (
            before.st_dev != identity["device"]
            or before.st_ino != identity["inode"]
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            return "directory_identity_mismatch"
        if sorted(os.listdir(directory_fd)) != sorted(PUBLICATION_BUNDLE_NAMES):
            return "exact_file_set_mismatch"
        anchored_receipt, receipt_info = _read_bundle_entry(
            directory_fd, PUBLICATION_RECEIPT_NAME, maximum=MAX_TEXT_ARTIFACT_BYTES
        )
        if (
            anchored_receipt != snapshot
            or hashlib.sha256(anchored_receipt).hexdigest()
            != runtime["publication_receipt"]["sha256"]
            or stat.S_IMODE(receipt_info.st_mode) & 0o022
        ):
            return "receipt_snapshot_mismatch"
        anchored: dict[str, bytes] = {}
        for name, metadata in sorted(receipt["files"].items()):
            content, info = _read_bundle_entry(
                directory_fd, name, maximum=MAX_ARTIFACT_BYTES
            )
            if (
                len(content) != metadata["size_bytes"]
                or hashlib.sha256(content).hexdigest() != metadata["sha256"]
                or f"{stat.S_IMODE(info.st_mode):04o}" != metadata["mode"]
            ):
                return f"descriptor_replay_mismatch:{name}"
            anchored[name] = content
        artifact_slots = {
            "deepstream-parallel-infer": "parallel_app_binary",
            "libdeepsafe_fusion.so.1": "fusion_plugin",
            "capability-manifest.json": "capability_manifest",
        }
        for file_name, slot_name in artifact_slots.items():
            if (
                receipt["files"][file_name]["sha256"]
                != runtime[slot_name]["sha256"]
            ):
                return f"contract_hash_mismatch:{file_name}"
        if (
            receipt["files"]["fusion-runtime.conf"]["sha256"]
            != hashlib.sha256(_fusion_runtime_config()).hexdigest()
        ):
            return "fusion_config_hash_mismatch"
        try:
            capability = strict_json_loads(anchored["capability-manifest.json"])
        except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError):
            return "capability_snapshot_invalid"
        if (
            not isinstance(capability, dict)
            or capability.get("build_provenance_sha256")
            != receipt["files"]["build-provenance.json"]["sha256"]
        ):
            return "provenance_binding_mismatch"
        if sorted(os.listdir(directory_fd)) != sorted(PUBLICATION_BUNDLE_NAMES):
            return "exact_file_set_changed_during_replay"
        after = os.fstat(directory_fd)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            return "directory_changed_during_replay"
    except (KeyError, OSError, FullStackContractError, TypeError, ValueError):
        return "descriptor_replay_failed"
    finally:
        os.close(directory_fd)
    return None


def _derive_readiness_blockers(
    contract: dict[str, Any],
    profile_id: str,
    *,
    source_count: int,
    project_root: Path,
    checks: list[dict[str, Any]],
    by_label: dict[str, dict[str, Any]],
    snapshots: dict[str, bytes],
) -> list[str]:
    blockers = [
        f"artifact:{check['label']}:{check['status']}"
        for check in checks
        if check["status"] != "ready"
    ]
    publication_error = _publication_receipt_error(
        contract,
        project_root=project_root,
        by_label=by_label,
        snapshots=snapshots,
    )
    if publication_error is not None:
        blockers.append(f"runtime.publication_receipt:{publication_error}")
    for json_label, binding_key, target_label in ARTIFACT_BINDINGS:
        if (
            by_label[json_label]["status"] != "ready"
            or by_label[target_label]["status"] != "ready"
        ):
            continue
        snapshot = snapshots.get(json_label)
        try:
            artifact_json = strict_json_loads(snapshot)
        except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError, TypeError):
            blockers.append(f"{json_label}:stable_json_snapshot_missing")
            continue
        if (
            not isinstance(artifact_json, dict)
            or artifact_json.get(binding_key)
            != by_label[target_label]["observed_sha256"]
        ):
            blockers.append(f"{json_label}:{binding_key}_mismatch")
    capability_snapshot = snapshots.get("runtime.capability_manifest")
    if by_label["runtime.capability_manifest"]["status"] == "ready":
        try:
            capability = strict_json_loads(capability_snapshot)
        except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError, TypeError):
            blockers.append("runtime.capability_manifest:stable_json_snapshot_missing")
        else:
            expected_config_sha256 = hashlib.sha256(
                _fusion_runtime_config()
            ).hexdigest()
            if (
                not isinstance(capability, dict)
                or capability.get("fusion_config_sha256")
                != expected_config_sha256
            ):
                blockers.append(
                    "runtime.capability_manifest:fusion_config_sha256_mismatch"
                )
    for role in ROLE_ORDER:
        if (
            by_label[f"{role}.infer_config"]["status"] == "ready"
            and by_label[f"{role}.engine"]["status"] == "ready"
        ):
            config_snapshot = snapshots.get(f"{role}.infer_config")
            if config_snapshot is None:
                blockers.append(f"{role}.infer_config:stable_snapshot_missing")
                continue
            engine_container_path = by_label[f"{role}.engine"]["container_path"]
            blockers.extend(
                _infer_config_errors(
                    role,
                    config_snapshot,
                    engine_container_path,
                    source_count=source_count,
                    max_batch_size=contract["limits"]["max_batch_size"],
                )
            )
    return sorted(set(blockers))


def _branch_plan(
    role: str,
    source_ids: list[int],
    model: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "branch_index": ROLE_GIE_IDS[role] - 1,
        "metamux_sink_pad": f"sink_{ROLE_GIE_IDS[role] - 1}",
        "gie_unique_id": ROLE_GIE_IDS[role],
        "inference_mode": "full_frame_primary",
        "batch_semantics": "camera_batch",
        "source_ids": source_ids,
        "tracker": model["tracker"],
        "metadata_output": model["metadata_output"],
        "association_target_gie": model["association_target_gie"],
    }


def _plan_contract_path(contract_path: Path, project_root: Path) -> str:
    return _project_relative(contract_path, project_root)


def _safe_output_dir(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    endurance_current = (project_root / "validation/results/endurance/current").resolve()
    if resolved == endurance_current or endurance_current in resolved.parents:
        raise FullStackContractError("full-stack output may not touch endurance/current")
    return resolved


def build_plan(
    contract_path: Path,
    *,
    profile_id: str,
    sources: Iterable[str],
    output_dir: Path,
    container_output_dir: str = "/opt/deepsafe/generated/full-stack",
    project_root: Path = PROJECT_ROOT,
    authorize_launch: bool = False,
) -> dict[str, Any]:
    """Build and persist a deterministic dry-run or authorization plan.

    Missing or invalid artifacts are represented as blockers.  When any blocker
    exists, only ``full-stack-plan.json`` is written; runtime configuration files
    are neither created nor retained.
    """

    project_root = project_root.resolve()
    output_dir = _safe_output_dir(output_dir, project_root)
    _container_path(container_output_dir, "container_output_dir")
    contract, contract_sha = load_contract(contract_path)
    if profile_id not in contract["profiles"]:
        raise FullStackContractError(f"unknown profile: {profile_id}")
    source_uris = _normalize_sources(sources, contract["limits"]["max_sources"])
    source_ids = list(range(len(source_uris)))
    checks, by_label, snapshots = _artifact_checks(
        contract, profile_id, project_root=project_root
    )
    profile = contract["profiles"][profile_id]
    blockers = _derive_readiness_blockers(
        contract,
        profile_id,
        source_count=len(source_uris),
        project_root=project_root,
        checks=checks,
        by_label=by_label,
        snapshots=snapshots,
    )
    execution_ready = not blockers
    mode = "authorize_launch" if authorize_launch else "dry_run"
    topology = contract["topology"]
    models = profile["models"]
    generator_bytes = _read_stable_regular_bytes(
        Path(__file__), max_bytes=MAX_TEXT_ARTIFACT_BYTES
    )

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "deterministic": True,
        "mode": mode,
        "contract": {
            "path": _plan_contract_path(contract_path, project_root),
            "sha256": contract_sha,
        },
        "generator": {
            "path": _project_relative(Path(__file__), project_root),
            "sha256": hashlib.sha256(generator_bytes).hexdigest(),
        },
        "profile": {
            "id": profile_id,
            "input_width": profile["input_width"],
            "input_height": profile["input_height"],
        },
        "runtime": {
            "active_deepstream_version": contract["runtime"]["active_deepstream_version"],
            "migration_deepstream_version": contract["runtime"]["migration_deepstream_version"],
            "migration_status": contract["runtime"]["migration_status"],
            "parallel_pattern": topology["pattern"],
        },
        "sources": {
            "count": len(source_uris),
            "ids": source_ids,
            "uris": source_uris,
            "batch_size": len(source_uris),
            "max_batch_size": contract["limits"]["max_batch_size"],
        },
        "topology": {
            "branches": [
                _branch_plan(role, source_ids, models[role]) for role in ROLE_ORDER
            ],
            "nvdsmetamux": {
                "required": True,
                "active_pad": topology["metamux_active_pad"],
                "pts_tolerance_us": topology["metamux_pts_tolerance_us"],
                "model_unique_ids": [ROLE_GIE_IDS[role] for role in ROLE_ORDER],
                "source_ids": source_ids,
            },
            "streammux": {
                "batch_size": len(source_uris),
                "width": topology["streammux_width"],
                "height": topology["streammux_height"],
            },
            "sink": {"type": "fakesink", "headless": True, "sync": False, "qos": False},
            "performance": {
                "enabled": True,
                "interval_seconds": topology["perf_measurement_interval_seconds"],
                "component_latency_measurement": True,
                "environment": {
                    "NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT": "1",
                    "NVDS_ENABLE_LATENCY_MEASUREMENT": "1",
                },
            },
        },
        "artifact_checks": checks,
        "readiness_blockers": blockers,
        "execution_ready": execution_ready,
        "launch_authorized": bool(execution_ready and authorize_launch),
        "safety": {
            "docker_called": False,
            "gpu_process_started": False,
            "inference_started": False,
        },
        "planned_outputs": {
            "container_directory": container_output_dir.rstrip("/"),
            "files": list(GENERATED_NAMES),
        },
        "rendered_outputs": [],
        "launch": None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, bytes] | None = None
    if execution_ready:
        bundle = _render_bundle(plan, contract)
        rendered: list[dict[str, Any]] = []
        for name in GENERATED_NAMES:
            content = bundle[name]
            rendered.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        plan["rendered_outputs"] = rendered
        container_dir = plan["planned_outputs"]["container_directory"]
        plan["launch"] = {
            "argv": [
                by_label["runtime.parallel_app_binary"]["container_path"],
                "-c",
                f"{container_dir}/parallel-inference.yml",
            ],
            "environment": {
                **plan["topology"]["performance"]["environment"],
                "DEEPSAFE_FUSION_CONFIG": f"{container_dir}/fusion-runtime.conf",
                "DEEPSAFE_FUSION_CONFIG_SHA256": hashlib.sha256(
                    bundle["fusion-runtime.conf"]
                ).hexdigest(),
            },
        }
    validate_plan(plan)
    if not execution_ready:
        # Invalidate any older authorization before removing stale generated
        # configs. This ordering stays fail-closed even if cleanup is interrupted.
        _atomic_write(output_dir / "full-stack-plan.json", _json_bytes(plan))
        cleanup_errors: list[str] = []
        for name in GENERATED_NAMES:
            path = output_dir / name
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(f"{name}:{exc}")
        if cleanup_errors:
            raise FullStackContractError(
                "blocked plan persisted but stale output cleanup failed: "
                + ", ".join(cleanup_errors)
            )
        return plan
    if bundle is not None:
        for name in GENERATED_NAMES:
            _atomic_write(output_dir / name, bundle[name])
    _atomic_write(output_dir / "full-stack-plan.json", _json_bytes(plan))
    return plan


def _source_csv(plan: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["enable", "type", "uri", "num-sources", "gpu-id", "cudadec-memtype"])
    for uri in plan["sources"]["uris"]:
        source_type = 4 if urlsplit(uri).scheme.lower() in {"rtsp", "rtsps"} else 3
        writer.writerow([1, source_type, uri, 1, 0, 0])
    return output.getvalue().encode("utf-8")


def _metamux_config(plan: dict[str, Any]) -> bytes:
    meta = plan["topology"]["nvdsmetamux"]
    ids = ";".join(str(value) for value in meta["source_ids"])
    lines = [
        "[property]",
        "enable=1",
        f"active-pad={meta['active_pad']}",
        f"pts-tolerance={meta['pts_tolerance_us']}",
        "",
        "[user-configs]",
        "",
        "[group-0]",
    ]
    lines.extend(f"src-ids-model-{gie_id}={ids}" for gie_id in meta["model_unique_ids"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _check_by_label(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["label"]: check for check in plan["artifact_checks"]}


def _parallel_yaml(plan: dict[str, Any]) -> bytes:
    checks = _check_by_label(plan)
    sources = plan["sources"]
    source_ids = ";".join(str(value) for value in sources["ids"])
    container_dir = plan["planned_outputs"]["container_directory"]
    live_source = int(
        any(urlsplit(uri).scheme.lower() in {"rtsp", "rtsps"} for uri in sources["uris"])
    )
    lines = [
        "application:",
        "  enable-perf-measurement: 1",
        f"  perf-measurement-interval-sec: {plan['topology']['performance']['interval_seconds']}",
        "",
        "tiled-display:",
        "  enable: 0",
        "",
        "source:",
        f"  csv-file-path: {container_dir}/sources.csv",
        "",
        "sink0:",
        "  enable: 1",
        "  type: 1",
        "  sync: 0",
        "  qos: 0",
        "",
        "osd:",
        "  enable: 0",
        "",
        "streammux:",
        "  gpu-id: 0",
        f"  live-source: {live_source}",
        f"  batch-size: {sources['batch_size']}",
        "  batched-push-timeout: 40000",
        f"  width: {plan['topology']['streammux']['width']}",
        f"  height: {plan['topology']['streammux']['height']}",
        "  enable-padding: 0",
        "  nvbuf-memory-type: 0",
        "",
    ]
    for branch_index, role in enumerate(ROLE_ORDER):
        gie_id = ROLE_GIE_IDS[role]
        lines.extend(
            [
                f"primary-gie{branch_index}:",
                "  enable: 1",
                "  plugin-type: 0",
                "  gpu-id: 0",
                f"  batch-size: {sources['batch_size']}",
                "  interval: 0",
                f"  gie-unique-id: {gie_id}",
                "  nvbuf-memory-type: 0",
                f"  config-file: {checks[f'{role}.infer_config']['container_path']}",
                "",
                f"branch{branch_index}:",
                f"  pgie-id: {gie_id}",
                f"  src-ids: {source_ids}",
                "",
                f"tracker{branch_index}:",
                f"  enable: {1 if role == 'person' else 0}",
            ]
        )
        if role == "person":
            lines.extend(
                [
                    f"  cfg-file-path: {container_dir}/tracker.yml",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "meta-mux:",
            "  enable: 1",
            f"  config-file: {container_dir}/metamux.txt",
            "",
            "tests:",
            "  file-loop: 0",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _tracker_yaml(contract: dict[str, Any]) -> bytes:
    topology = contract["topology"]
    lines = [
        "tracker:",
        "  tracker-width: 640",
        "  tracker-height: 384",
        f"  ll-lib-file: {topology['tracker_library']}",
        f"  ll-config-file: {topology['tracker_config']}",
        "  gpu-id: 0",
        "  enable-batch-process: 1",
        "  enable-past-frame: 0",
        "  display-tracking-id: 0",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _fusion_contract(plan: dict[str, Any]) -> bytes:
    checks = _check_by_label(plan)
    value = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "source_ids": plan["sources"]["ids"],
        "canonical_person": {
            "gie_unique_id": ROLE_GIE_IDS["person"],
            "tracker": "nvdcf",
            "identity_metadata": "nvds_object_meta.object_id",
        },
        "pose": {
            "gie_unique_id": ROLE_GIE_IDS["pose"],
            "metadata_input": "NvDsInferTensorMeta",
            "postprocess_library": checks["pose.postprocess_library"]["container_path"],
            "association_contract": checks["pose.association_contract"]["container_path"],
            "association_target_gie": ROLE_GIE_IDS["person"],
        },
        "ppe": {
            "gie_unique_id": ROLE_GIE_IDS["ppe"],
            "metadata_input": "NvDsObjectMeta",
            "postprocess_library": checks["ppe.postprocess_library"]["container_path"],
            "association_contract": checks["ppe.association_contract"]["container_path"],
            "association_target_gie": ROLE_GIE_IDS["person"],
            "required_attributes": ["helmet", "hi_vis"],
        },
        "duplicate_track_id_policy": "fail_closed",
        "unmatched_metadata_policy": "retain_diagnostic_do_not_emit_compliance_event",
    }
    return _json_bytes(value)


def _fusion_runtime_config() -> bytes:
    lines = [
        "schema_version=deepsafe.fusion-runtime-config/v1",
        "person_gie_id=1",
        "pose_gie_id=2",
        "ppe_gie_id=3",
        "person_class_id=0",
        "ppe_helmet_present_class_id=0",
        "ppe_helmet_absent_class_id=1",
        "ppe_hi_vis_present_class_id=2",
        "ppe_hi_vis_absent_class_id=3",
        "max_sources=12",
        "max_batch_size=12",
        "max_persons_per_frame=512",
        "max_pose_detections=300",
        "max_ppe_observations=1024",
        "pose_detection_threshold=0.25",
        "pose_keypoint_threshold=0.25",
        "pose_ambiguity_margin=0.05",
        "minimum_pose_iou=0.30",
        "minimum_pose_coverage=0.50",
        "minimum_person_height_px=24.0",
        "minimum_tracker_confidence=0.00",
        "ppe_minimum_confidence=0.25",
        "ppe_minimum_person_coverage=0.50",
        "ppe_minimum_zone_coverage=0.20",
        "ppe_ambiguity_margin=0.05",
        "ppe_minimum_presence_visible_fraction=0.05",
        "ppe_minimum_absence_visible_fraction=0.50",
        "reject_nonmonotonic_pts=true",
        "missing_ppe_policy=unknown",
        "ambiguous_policy=unknown_unassociated",
        "occluded_policy=unknown_unassociated",
        "duplicate_meta_policy=reject_frame",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _render_bundle(
    plan: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, bytes]:
    if not plan["execution_ready"]:
        raise FullStackContractError("cannot render runtime config for a blocked plan")
    return {
        "sources.csv": _source_csv(plan),
        "metamux.txt": _metamux_config(plan),
        "tracker.yml": _tracker_yaml(contract),
        "parallel-inference.yml": _parallel_yaml(plan),
        "fusion-runtime.conf": _fusion_runtime_config(),
        "fusion-contract.json": _fusion_contract(plan),
    }


PLAN_KEYS = {
    "schema_version",
    "deterministic",
    "mode",
    "contract",
    "generator",
    "profile",
    "runtime",
    "sources",
    "topology",
    "artifact_checks",
    "readiness_blockers",
    "execution_ready",
    "launch_authorized",
    "safety",
    "planned_outputs",
    "rendered_outputs",
    "launch",
}


def validate_plan(plan: Any) -> dict[str, Any]:
    """Validate exact invariants which must hold before launch authorization."""

    value = _exact_keys(plan, PLAN_KEYS, "plan")
    if value["schema_version"] != PLAN_SCHEMA_VERSION or value["deterministic"] is not True:
        raise FullStackContractError("invalid plan schema or deterministic marker")
    if value["mode"] not in {"dry_run", "authorize_launch"}:
        raise FullStackContractError("invalid plan mode")
    contract_pin = _exact_keys(value["contract"], {"path", "sha256"}, "plan.contract")
    _plan_file_path(contract_pin["path"], "plan.contract.path")
    if not isinstance(contract_pin["sha256"], str) or not SHA256_RE.fullmatch(contract_pin["sha256"]):
        raise FullStackContractError("plan.contract.sha256 must be lowercase SHA-256")
    generator_pin = _exact_keys(value["generator"], {"path", "sha256"}, "plan.generator")
    _plan_file_path(generator_pin["path"], "plan.generator.path")
    if not isinstance(generator_pin["sha256"], str) or not SHA256_RE.fullmatch(generator_pin["sha256"]):
        raise FullStackContractError("plan.generator.sha256 must be lowercase SHA-256")
    profile = _exact_keys(value["profile"], {"id", "input_width", "input_height"}, "plan.profile")
    if not isinstance(profile["id"], str) or not PROFILE_RE.fullmatch(profile["id"]):
        raise FullStackContractError("invalid plan profile id")
    _require_int(profile["input_width"], "plan.profile.input_width", 1, 16384)
    _require_int(profile["input_height"], "plan.profile.input_height", 1, 16384)

    runtime = _exact_keys(
        value["runtime"],
        {
            "active_deepstream_version",
            "migration_deepstream_version",
            "migration_status",
            "parallel_pattern",
        },
        "plan.runtime",
    )
    if runtime != {
        "active_deepstream_version": ACTIVE_DEEPSTREAM_VERSION,
        "migration_deepstream_version": MIGRATION_DEEPSTREAM_VERSION,
        "migration_status": MIGRATION_STATUS,
        "parallel_pattern": PARALLEL_PATTERN,
    }:
        raise FullStackContractError("plan runtime or migration lane drifted")

    sources = _exact_keys(
        value["sources"],
        {"count", "ids", "uris", "batch_size", "max_batch_size"},
        "plan.sources",
    )
    count = _require_int(sources["count"], "plan.sources.count", 1, MAX_SOURCES)
    if sources["ids"] != list(range(count)):
        raise FullStackContractError("source IDs must be contiguous 0..N-1")
    if not all(_is_int(source_id) for source_id in sources["ids"]):
        raise FullStackContractError("source IDs must be exact integers")
    if not isinstance(sources["uris"], list) or len(sources["uris"]) != count:
        raise FullStackContractError("source URI count mismatch")
    _normalize_sources(sources["uris"], MAX_SOURCES)
    if sources["batch_size"] != count or sources["max_batch_size"] != MAX_SOURCES:
        raise FullStackContractError("camera batch must equal N and remain <=12")

    topology = _exact_keys(
        value["topology"],
        {"branches", "nvdsmetamux", "streammux", "sink", "performance"},
        "plan.topology",
    )
    branches = topology["branches"]
    if not isinstance(branches, list) or len(branches) != 3:
        raise FullStackContractError("exactly three parallel inference branches are required")
    observed_ids: list[int] = []
    for index, role in enumerate(ROLE_ORDER):
        branch = _exact_keys(
            branches[index],
            {
                "role",
                "branch_index",
                "metamux_sink_pad",
                "gie_unique_id",
                "inference_mode",
                "batch_semantics",
                "source_ids",
                "tracker",
                "metadata_output",
                "association_target_gie",
            },
            f"plan.topology.branches[{index}]",
        )
        expected_target = None if role == "person" else 1
        expected = {
            "role": role,
            "branch_index": index,
            "metamux_sink_pad": f"sink_{index}",
            "gie_unique_id": ROLE_GIE_IDS[role],
            "inference_mode": "full_frame_primary",
            "batch_semantics": "camera_batch",
            "source_ids": sources["ids"],
            "tracker": "nvdcf" if role == "person" else "none",
            "metadata_output": "nvds_infer_tensor_meta" if role == "pose" else "nvds_object_meta",
            "association_target_gie": expected_target,
        }
        if branch != expected:
            raise FullStackContractError(f"{role} branch topology drifted")
        if not _is_int(branch["gie_unique_id"]):
            raise FullStackContractError(f"{role} GIE unique ID must be an exact integer")
        observed_ids.append(branch["gie_unique_id"])
    if len(set(observed_ids)) != 3:
        raise FullStackContractError("GIE unique IDs must be unique")

    meta = _exact_keys(
        topology["nvdsmetamux"],
        {"required", "active_pad", "pts_tolerance_us", "model_unique_ids", "source_ids"},
        "plan.topology.nvdsmetamux",
    )
    if (
        meta["required"] is not True
        or meta["active_pad"] != "sink_0"
        or meta["model_unique_ids"] != [1, 2, 3]
        or meta["source_ids"] != sources["ids"]
    ):
        raise FullStackContractError("nvdsmetamux contract drifted")
    if not all(_is_int(gie_id) for gie_id in meta["model_unique_ids"]):
        raise FullStackContractError("nvdsmetamux model IDs must be exact integers")
    _require_int(meta["pts_tolerance_us"], "plan.topology.nvdsmetamux.pts_tolerance_us", 1, 10_000_000)
    streammux = _exact_keys(topology["streammux"], {"batch_size", "width", "height"}, "plan.topology.streammux")
    if streammux["batch_size"] != count:
        raise FullStackContractError("streammux batch must equal source count")
    _require_int(streammux["width"], "plan.topology.streammux.width", 1, 16384)
    _require_int(streammux["height"], "plan.topology.streammux.height", 1, 16384)
    if topology["sink"] != {
        "type": "fakesink",
        "headless": True,
        "sync": False,
        "qos": False,
    }:
        raise FullStackContractError("headless performance sink contract drifted")
    performance = _exact_keys(
        topology["performance"],
        {"enabled", "interval_seconds", "component_latency_measurement", "environment"},
        "plan.topology.performance",
    )
    if performance["enabled"] is not True or performance["component_latency_measurement"] is not True:
        raise FullStackContractError("performance and component latency measurement are mandatory")
    _require_int(performance["interval_seconds"], "plan.topology.performance.interval_seconds", 1, 60)
    if performance["environment"] != {
        "NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT": "1",
        "NVDS_ENABLE_LATENCY_MEASUREMENT": "1",
    }:
        raise FullStackContractError("latency environment contract drifted")

    if not isinstance(value["artifact_checks"], list) or len(value["artifact_checks"]) != 14:
        raise FullStackContractError("exactly fourteen artifact checks are required")
    expected_labels = [f"runtime.{name}" for name in RUNTIME_ARTIFACTS]
    expected_labels.extend(
        f"{role}.{name}" for role in ROLE_ORDER for name in ROLE_ARTIFACTS[role]
    )
    labels: list[str] = []
    statuses: list[str] = []
    for index, check_value in enumerate(value["artifact_checks"]):
        check = _exact_keys(
            check_value,
            {
                "label",
                "host_path",
                "container_path",
                "expected_sha256",
                "observed_sha256",
                "size_bytes",
                "status",
            },
            f"plan.artifact_checks[{index}]",
        )
        labels.append(check["label"])
        statuses.append(check["status"])
        if not isinstance(check["label"], str) or not isinstance(check["status"], str):
            raise FullStackContractError("artifact check label/status must be strings")
        if check["host_path"] is None:
            if (
                check["status"] != "unconfigured"
                or any(
                    check[key] is not None
                    for key in (
                        "container_path",
                        "expected_sha256",
                        "observed_sha256",
                        "size_bytes",
                    )
                )
            ):
                raise FullStackContractError(
                    "unconfigured artifact check must use the exact all-null form"
                )
            continue
        _relative_path(check["host_path"], f"plan.artifact_checks[{index}].host_path")
        _container_path(
            check["container_path"],
            f"plan.artifact_checks[{index}].container_path",
        )
        if (
            not isinstance(check["expected_sha256"], str)
            or not SHA256_RE.fullmatch(check["expected_sha256"])
        ):
            raise FullStackContractError("configured artifact expected SHA-256 is invalid")
        if check["observed_sha256"] is not None and (
            not isinstance(check["observed_sha256"], str)
            or not SHA256_RE.fullmatch(check["observed_sha256"])
        ):
            raise FullStackContractError("artifact observed SHA-256 is invalid")
        if check["size_bytes"] is not None and (
            not _is_int(check["size_bytes"]) or check["size_bytes"] < 1
        ):
            raise FullStackContractError("artifact observed size is invalid")
        if check["status"] == "ready" and (
            check["observed_sha256"] != check["expected_sha256"]
            or check["size_bytes"] is None
        ):
            raise FullStackContractError(
                "ready artifact check must have matching hash and positive size"
            )
    if labels != expected_labels:
        raise FullStackContractError("artifact check labels/order drifted")
    if (
        not isinstance(value["readiness_blockers"], list)
        or not all(isinstance(item, str) and item for item in value["readiness_blockers"])
        or value["readiness_blockers"] != sorted(set(value["readiness_blockers"]))
    ):
        raise FullStackContractError("readiness blockers must be a sorted unique list")
    expected_ready = not value["readiness_blockers"] and all(status == "ready" for status in statuses)
    if value["execution_ready"] is not expected_ready:
        raise FullStackContractError("execution_ready does not match fail-closed blockers")
    expected_authorized = expected_ready and value["mode"] == "authorize_launch"
    if value["launch_authorized"] is not expected_authorized:
        raise FullStackContractError("launch_authorized does not match mode/readiness")
    if value["safety"] != {
        "docker_called": False,
        "gpu_process_started": False,
        "inference_started": False,
    }:
        raise FullStackContractError("static scaffold safety claims drifted")
    outputs = _exact_keys(
        value["planned_outputs"], {"container_directory", "files"}, "plan.planned_outputs"
    )
    _container_path(outputs["container_directory"], "plan.planned_outputs.container_directory")
    if outputs["files"] != list(GENERATED_NAMES):
        raise FullStackContractError("planned output file set drifted")
    if expected_ready:
        if (
            not isinstance(value["rendered_outputs"], list)
            or not all(isinstance(item, dict) for item in value["rendered_outputs"])
            or [item.get("name") for item in value["rendered_outputs"]]
            != list(GENERATED_NAMES)
        ):
            raise FullStackContractError("ready plan requires every rendered output pin")
        for item in value["rendered_outputs"]:
            _exact_keys(item, {"name", "sha256", "size_bytes"}, "plan.rendered_output")
            if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
                raise FullStackContractError("rendered output SHA-256 is invalid")
            _require_int(item["size_bytes"], "plan.rendered_output.size_bytes", 1, 10_000_000)
        launch = _exact_keys(value["launch"], {"argv", "environment"}, "plan.launch")
        if not isinstance(launch["argv"], list) or len(launch["argv"]) != 3:
            raise FullStackContractError("launch argv contract is invalid")
        checks_by_label = _check_by_label(value)
        expected_argv = [
            checks_by_label["runtime.parallel_app_binary"]["container_path"],
            "-c",
            f"{outputs['container_directory']}/parallel-inference.yml",
        ]
        if launch["argv"] != expected_argv:
            raise FullStackContractError("launch config argv drifted")
        if launch["environment"] != {
            **performance["environment"],
            "DEEPSAFE_FUSION_CONFIG": f"{outputs['container_directory']}/fusion-runtime.conf",
            "DEEPSAFE_FUSION_CONFIG_SHA256": hashlib.sha256(
                _fusion_runtime_config()
            ).hexdigest(),
        }:
            raise FullStackContractError("launch environment drifted")
    else:
        if value["rendered_outputs"] != [] or value["launch"] is not None:
            raise FullStackContractError("blocked plan may not contain configs or launch command")
    return value


def verify_persisted_plan(
    plan_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    plan = validate_plan(_read_strict_json(plan_path))
    contract_path = Path(plan["contract"]["path"])
    if not contract_path.is_absolute():
        contract_path = project_root / contract_path
    contract, observed_contract_sha = load_contract(contract_path)
    if observed_contract_sha != plan["contract"]["sha256"]:
        raise FullStackContractError("live contract SHA-256 differs from plan pin")
    generator_path = Path(plan["generator"]["path"])
    if not generator_path.is_absolute():
        generator_path = project_root / generator_path
    expected_generator_path = Path(__file__).resolve()
    if generator_path.resolve() != expected_generator_path:
        raise FullStackContractError("plan generator path is not this full-stack implementation")
    generator_bytes = _read_stable_regular_bytes(
        generator_path, max_bytes=MAX_TEXT_ARTIFACT_BYTES
    )
    if hashlib.sha256(generator_bytes).hexdigest() != plan["generator"]["sha256"]:
        raise FullStackContractError("live generator SHA-256 differs from plan pin")
    if plan["profile"]["id"] not in contract["profiles"]:
        raise FullStackContractError("plan profile disappeared from live contract")
    live_checks, live_by_label, live_snapshots = _artifact_checks(
        contract, plan["profile"]["id"], project_root=project_root
    )
    if live_checks != plan["artifact_checks"]:
        raise FullStackContractError("live artifact replay differs from plan")
    live_blockers = _derive_readiness_blockers(
        contract,
        plan["profile"]["id"],
        source_count=plan["sources"]["count"],
        project_root=project_root,
        checks=live_checks,
        by_label=live_by_label,
        snapshots=live_snapshots,
    )
    if live_blockers != plan["readiness_blockers"]:
        raise FullStackContractError("live readiness derivation differs from plan")
    if plan["execution_ready"]:
        output_dir = plan_path.parent
        bundle = _render_bundle(plan, contract)
        pins = {item["name"]: item for item in plan["rendered_outputs"]}
        for name in GENERATED_NAMES:
            path = output_dir / name
            try:
                actual = _read_stable_regular_bytes(path, max_bytes=10_000_000)
            except FullStackContractError as exc:
                raise FullStackContractError(f"missing rendered output {name}: {exc}") from exc
            if actual != bundle[name]:
                raise FullStackContractError(f"rendered output content drift: {name}")
            if len(actual) != pins[name]["size_bytes"] or hashlib.sha256(actual).hexdigest() != pins[name]["sha256"]:
                raise FullStackContractError(f"rendered output pin mismatch: {name}")
    return plan


def require_launch_ready(
    plan_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return a launch contract only after strict replay; never start a process."""

    try:
        plan = verify_persisted_plan(plan_path, project_root=project_root)
    except FullStackContractError as exc:
        raise LaunchRejected(f"full-stack launch replay rejected: {exc}") from exc
    if plan["mode"] != "authorize_launch":
        raise LaunchRejected("dry-run plan cannot authorize launch")
    if not plan["execution_ready"] or not plan["launch_authorized"]:
        blockers = ", ".join(plan["readiness_blockers"]) or "unknown readiness drift"
        raise LaunchRejected(f"full-stack launch rejected: {blockers}")

    return {
        "argv": list(plan["launch"]["argv"]),
        "environment": dict(plan["launch"]["environment"]),
        "contract_sha256": plan["contract"]["sha256"],
        "rendered_outputs": list(plan["rendered_outputs"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render/validate a fail-closed DeepStream 9.0 three-model parallel plan"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", default="640")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--container-output-dir", default="/opt/deepsafe/generated/full-stack"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--authorize-launch",
        action="store_true",
        help="request authorization output; this command still never launches a process",
    )
    parser.add_argument(
        "--validate-plan",
        type=Path,
        help="strictly replay an existing plan instead of generating one",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_plan is not None:
            plan = verify_persisted_plan(args.validate_plan, project_root=args.project_root)
        else:
            if not args.source:
                raise FullStackContractError("at least one --source URI is required")
            plan = build_plan(
                args.contract,
                profile_id=args.profile,
                sources=args.source,
                output_dir=args.output_dir,
                container_output_dir=args.container_output_dir,
                project_root=args.project_root,
                authorize_launch=args.authorize_launch,
            )
            if args.authorize_launch:
                require_launch_ready(
                    args.output_dir / "full-stack-plan.json",
                    project_root=args.project_root,
                )
    except LaunchRejected as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except FullStackContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": plan["schema_version"],
                "mode": plan["mode"],
                "profile": plan["profile"]["id"],
                "sources": plan["sources"]["count"],
                "execution_ready": plan["execution_ready"],
                "launch_authorized": plan["launch_authorized"],
                "blocker_count": len(plan["readiness_blockers"]),
                "docker_called": False,
                "gpu_process_started": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
