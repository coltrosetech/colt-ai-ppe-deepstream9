#!/usr/bin/env python3
"""Audited DeepStream 9 parser/engine GPU smoke and parity harness.

The default mode is inert with respect to Docker and the GPU: it only renders
and hashes a candidate contract.  A workload is possible only with explicit
``--execute`` plus a current, immutable operator authorization, GPU re-entry
evidence, two-pass parser build receipt and static DS9 candidate receipt.  The
four DeepStream runs execute behind :mod:`validation.gpu_guarded_process`.

The same file has a hidden container-worker mode.  It is copied into the
immutable image and may only run the exact engine-only configs supplied by the
host contract.  Production evidence is accepted by raw replay, never by
trusting a producer-written ``pass`` string.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from validation.ds9_parser_bootstrap import (
    PRODUCTION_SCHEMA_VERSION as PARSER_BUILD_SCHEMA_VERSION,
    source_lineage_contract,
    validate_production_receipt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-plan/v1"
AUTH_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-authorization/v1"
CLAIM_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-session-claim/v1"
CONTRACT_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-probe-contract/v1"
CONTAINER_IDENTITY_SCHEMA_VERSION = "deepsafe.ds9-container-process-identity/v1"
INNER_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-inner-run/v1"
EVIDENCE_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-evidence/v1"
STATIC_RECEIPT_SCHEMA_VERSION = "deepsafe.ds9-runtime-compatibility-receipt/v1"
MAX_AUTH_AGE = timedelta(hours=24)
MAX_EVIDENCE_AGE = timedelta(hours=24)

REQUIRED_CHECKS = (
    "cuda_parser_kernel_launch_sm86",
    "deepstream_640_engine_deserialize_no_fallback",
    "deepstream_960_engine_deserialize_no_fallback",
    "cpu_cuda_parser_parity_640",
    "cpu_cuda_parser_parity_960",
)
PROFILES = (640, 960)
PARSERS = ("cuda", "cpu")
PARSER_FUNCTIONS = {
    "cuda": "NvDsInferParseYoloCuda",
    "cpu": "NvDsInferParseYolo",
}
RUN_IDS = tuple(f"{profile}-{parser}" for profile in PROFILES for parser in PARSERS)

DEFAULT_IMAGE = "deepsafe-deepstream:9.0"
DEFAULT_VIDEO = Path("data/derived/caviar/CAVIARDATA1/Walk1/Walk1.mp4")
DEFAULT_REENTRY = Path("validation/results/gpu-reentry/current/evidence.json")
DEFAULT_STATIC_CANDIDATE = Path(
    "validation/results/ds9-runtime-compatibility/candidate/receipt.json"
)
DEFAULT_PARSER_BUILD = Path(
    "validation/results/ds9-runtime-compatibility/parser-bootstrap/"
    "pass-2-production/production-build-receipt.json"
)
PARSER_LIBRARY = Path(
    "/opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/"
    "libnvdsinfer_custom_impl_Yolo.so"
)
CONTAINER_WORKER = Path("/app/validation/ds9_gpu_smoke.py")
SESSION_PREFIX = Path(
    "validation/results/ds9-runtime-compatibility/gpu-smoke/sessions"
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
NONCE_RE = SHA256_RE
GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F-]{36}")
KITTI_NAME_RE = re.compile(r"00_000_[0-9]{6}\.txt")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
POSITIVE_ENGINE_PATTERNS = (
    r"deserializeEngineAndBackend\(\).+deserialized trt engine from\s*:",
    r"generateBackendContext\(\).+Use deserialized engine model:",
)
ENGINE_FALLBACK_PATTERNS = (
    r"failed\s+to\s+deserialize",
    r"failed\s+to\s+create.*engine",
    r"(?<!de)\bserializ(?:e|ed|ing)\b.{0,80}\bengine\b",
    r"building.*engine",
    r"build.*engine.*from",
    r"onnx.*pars",
    r"fallback",
)
CUDA_ERROR_PATTERNS = (
    r"cudaError",
    r"CUDA failure",
    r"unspecified launch failure",
    r"illegal memory access",
    r"no kernel image",
)
REQUIRED_NVIDIA_FD_PATTERNS = (
    r"^/dev/nvidia[0-9]+$",
    r"^/dev/nvidiactl$",
    r"^/dev/nvidia-uvm$",
)
GPU_FD_SAMPLE_SECONDS = 0.05
GPU_FD_TERMINAL_EXIT_GRACE_SECONDS = 0.5
GPU_FD_TERMINAL_EXIT_GRACE_NS = 500_000_000
GPU_FD_TERMINAL_ROOT_ERRNOS = (errno.EACCES, errno.ENOENT)
GPU_FD_TERMINAL_EXCEPTION_TYPES = {
    errno.EACCES: "PermissionError",
    errno.ENOENT: "FileNotFoundError",
}

PARITY_BBOX_ABS_TOLERANCE_PX = 0.25
PARITY_CONFIDENCE_ABS_TOLERANCE = 1e-4
PARITY_MIN_IOU = 0.999
MIN_TOTAL_DETECTIONS = 1
KERNEL_PROOF_SCHEMA_VERSION = "deepsafe.ds9-cuda-kernel-proof/v1"
KERNEL_PROOF_MARKER_NAME = "cuda-parser-kernel.json"
KERNEL_PROOF_KERNEL_NAME = "decodeTensorYoloCuda"
KERNEL_PROOF_ENV = {
    "enable": "DEEPSAFE_DS9_CUDA_PROOF",
    "campaign_nonce": "DEEPSAFE_DS9_CAMPAIGN_NONCE",
    "run_id": "DEEPSAFE_DS9_RUN_ID",
    "marker_path": "DEEPSAFE_DS9_CUDA_MARKER_PATH",
    "disable_ptx_jit": "CUDA_DISABLE_PTX_JIT",
}
MAX_JSON_BYTES = 512 * 1024
MAX_KERNEL_MARKER_BYTES = 8 * 1024
MAX_RAW_LOG_BYTES = 8 * 1024 * 1024
MAX_GPU_IDENTITY_BYTES = 16 * 1024
MAX_KITTI_FILE_BYTES = 1024 * 1024
MAX_KITTI_TOTAL_BYTES = 64 * 1024 * 1024
MAX_KITTI_FILES = 10_000
MAX_KITTI_LINE_BYTES = 4096


class Ds9GpuSmokeError(RuntimeError):
    """Raised when smoke evidence cannot safely authorize DS9 production."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Ds9GpuSmokeError(f"{label} must be an RFC3339 timestamp")
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise Ds9GpuSmokeError(f"{label} must be an RFC3339 timestamp") from exc
    if result.tzinfo is None:
        raise Ds9GpuSmokeError(f"{label} must include a timezone")
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
        raise Ds9GpuSmokeError(f"artifact leaves project root: {resolved}") from exc
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise Ds9GpuSmokeError(f"artifact path contains a symlink: {current}")
    return resolved


def _relative(path: Path, project_root: Path) -> str:
    return _inside_root(path, project_root).relative_to(project_root.resolve()).as_posix()


def container_process_identity() -> dict[str, Any]:
    """Return the exact unprivileged host identity used inside the smoke container."""

    return {
        "schema_version": CONTAINER_IDENTITY_SCHEMA_VERSION,
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "selection": "host_effective_identity",
        "capabilities": "drop_all",
    }


def file_pin(path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise Ds9GpuSmokeError(f"cannot open pinned artifact: {resolved}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Ds9GpuSmokeError(f"artifact is not a regular file: {resolved}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
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
        raise Ds9GpuSmokeError(f"artifact changed while hashing: {resolved}")
    return {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_bounded_file(
    path: Path, *, project_root: Path, max_bytes: int, label: str
) -> tuple[bytes, dict[str, Any], os.stat_result]:
    """Read a regular file once through a no-follow FD and pin those bytes."""

    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise Ds9GpuSmokeError(f"cannot open {label}: {resolved}") from exc
    digest = hashlib.sha256()
    content = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Ds9GpuSmokeError(f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise Ds9GpuSmokeError(f"{label} exceeds its byte limit")
        while True:
            block = os.read(descriptor, min(1024 * 1024, max_bytes + 1))
            if not block:
                break
            content.extend(block)
            digest.update(block)
            if len(content) > max_bytes:
                raise Ds9GpuSmokeError(f"{label} exceeds its byte limit")
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
        raise Ds9GpuSmokeError(f"{label} changed while reading")
    pin = {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }
    return bytes(content), pin, before


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Ds9GpuSmokeError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_json(
    path: Path,
    *,
    project_root: Path,
    immutable: bool,
    max_bytes: int = MAX_JSON_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content, pin, metadata = _read_bounded_file(
        path, project_root=project_root, max_bytes=max_bytes, label="JSON evidence"
    )
    if immutable and (
        stat.S_IMODE(metadata.st_mode) != 0o440
        or metadata.st_nlink != 1
    ):
        raise Ds9GpuSmokeError("JSON evidence must be mode 0440 with one hard link")
    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ds9GpuSmokeError(f"invalid JSON artifact: {pin['path']}") from exc
    if not isinstance(payload, dict):
        raise Ds9GpuSmokeError("JSON artifact root must be an object")
    return payload, pin


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("short atomic JSON write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                raise OSError("short immutable JSON write")
            offset += written
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pin_matches(pin: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise Ds9GpuSmokeError("artifact pin fields differ")
    observed = file_pin(project_root / str(pin.get("path", "")), project_root=project_root)
    if observed != dict(pin):
        raise Ds9GpuSmokeError(f"artifact changed after pin: {pin.get('path')}")
    return observed


def _model_paths(project_root: Path) -> dict[int, dict[str, Path]]:
    return {
        profile: {
            "engine": project_root
            / f"models/person/{profile}/yolo11s_b12_gpu0_fp16.engine",
            "labels": project_root / f"models/person/{profile}/labels.txt",
        }
        for profile in PROFILES
    }


def render_infer_config(profile: int, parser: str) -> str:
    if profile not in PROFILES or parser not in PARSERS:
        raise ValueError("unsupported smoke profile/parser")
    filtered = ";".join(str(value) for value in range(1, 80))
    # Deliberately omit onnx-file.  A deserialization problem therefore cannot
    # silently rebuild/fall back to ONNX and still produce a passing run.
    return f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
model-engine-file=/models/{profile}.engine
labelfile-path=/models/{profile}.labels
batch-size=1
network-mode=2
num-detected-classes=80
filter-out-class-ids={filtered}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name={PARSER_FUNCTIONS[parser]}
custom-lib-path={PARSER_LIBRARY.as_posix()}
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
"""


def render_deepstream_config(profile: int, parser: str) -> str:
    run_id = f"{profile}-{parser}"
    return f"""[application]
enable-perf-measurement=1
perf-measurement-interval-sec=1
gie-kitti-output-dir=/evidence/{run_id}/kitti

[tiled-display]
enable=0

[source0]
enable=1
type=2
uri=file:///input/smoke.mp4
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
batch-size=1
batched-push-timeout=40000
width={profile}
height={profile}
enable-padding=1
nvbuf-memory-type=0

[primary-gie]
enable=1
gpu-id=0
batch-size=1
interval=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=/contract/infer-{run_id}.txt

[tests]
file-loop=0
"""


def _write_or_verify(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise Ds9GpuSmokeError(f"existing generated input differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        os.write(descriptor, encoded)
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _offline_static_candidate(
    path: Path, *, project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, pin = _read_json(path, project_root=project_root, immutable=True)
    if payload.get("schema_version") != STATIC_RECEIPT_SCHEMA_VERSION:
        raise Ds9GpuSmokeError("static candidate receipt schema differs")
    if payload.get("status") != "pending_gpu_smoke" or payload.get("production_ready") is not False:
        raise Ds9GpuSmokeError("static candidate must be pending GPU smoke")
    image = payload.get("image")
    static = payload.get("static_probe")
    if not isinstance(image, dict) or IMAGE_ID_RE.fullmatch(
        str(image.get("resolved_image_id"))
    ) is None:
        raise Ds9GpuSmokeError("static candidate has no immutable image ID")
    if not isinstance(static, dict) or static.get("status") != "pass":
        raise Ds9GpuSmokeError("static candidate probe is not passing")
    evidence = static.get("evidence")
    facts = evidence.get("facts") if isinstance(evidence, dict) else None
    if (
        not isinstance(facts, dict)
        or facts.get("sm86_cubin_present") is not True
        or facts.get("sm86_only_cubin_set") is not True
        or facts.get("compute86_only_ptx_set") is not True
    ):
        raise Ds9GpuSmokeError(
            "static candidate lacks exact sm_86 cubin/compute_86 PTX evidence"
        )
    created = _parse_time(payload.get("created_at_utc"), "static created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "static expires_at_utc")
    now = datetime.now(timezone.utc)
    if expires <= created or expires - created > MAX_EVIDENCE_AGE or not (created <= now < expires):
        raise Ds9GpuSmokeError("static candidate receipt is stale or overlong")
    return payload, pin


def _kernel_proof_environment(nonce: str, run_id: str) -> dict[str, str]:
    if run_id not in RUN_IDS:
        raise Ds9GpuSmokeError("kernel proof run ID is outside the closed matrix")
    if not run_id.endswith("-cuda"):
        return {}
    return {
        KERNEL_PROOF_ENV["enable"]: "1",
        KERNEL_PROOF_ENV["campaign_nonce"]: nonce,
        KERNEL_PROOF_ENV["run_id"]: run_id,
        KERNEL_PROOF_ENV["marker_path"]: (
            f"/evidence/{run_id}/{KERNEL_PROOF_MARKER_NAME}"
        ),
        KERNEL_PROOF_ENV["disable_ptx_jit"]: "1",
    }


def _kernel_proof_policy() -> dict[str, Any]:
    return {
        "schema_version": KERNEL_PROOF_SCHEMA_VERSION,
        "kernel": KERNEL_PROOF_KERNEL_NAME,
        "enabled_runs": ["640-cuda", "960-cuda"],
        "cpu_environment_must_be_unset": list(KERNEL_PROOF_ENV.values()),
        "marker_name": KERNEL_PROOF_MARKER_NAME,
        "marker_mode": "0440",
        "marker_hard_links": 1,
        "exclusive_create_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW"],
        "required_binary_version": 86,
        "required_ptx_version": 86,
        "ptx_jit_disabled_for_enabled_runs": True,
        "proof_calls_in_order": [
            "cudaGetLastError",
            "cudaDeviceSynchronize",
            "cudaFuncGetAttributes(decodeTensorYoloCuda)",
        ],
        "generic_signals_are_not_kernel_proof": [
            "nvidia_device_fds",
            "detections",
            "parser_function_config",
            "cuda_error_free_log",
        ],
    }


def _gpu_fd_policy() -> dict[str, Any]:
    return {
        "scope": "each_deepstream_app_process",
        "required_patterns": list(REQUIRED_NVIDIA_FD_PATTERNS),
        "sample_interval_seconds": GPU_FD_SAMPLE_SECONDS,
        "minimum_samples": 1,
        "build_stage_exempt": True,
        "terminal_teardown": {
            "classification": "same_process_proc_fd_root_terminal_teardown",
            "proc_fd_root_template": "/proc/{pid}/fd",
            "allowed_root_errors": [
                {
                    "errno": value,
                    "errno_name": errno.errorcode[value],
                    "exception_type": GPU_FD_TERMINAL_EXCEPTION_TYPES[value],
                }
                for value in GPU_FD_TERMINAL_ROOT_ERRNOS
            ],
            "exit_grace_seconds": GPU_FD_TERMINAL_EXIT_GRACE_SECONDS,
            "exit_grace_monotonic_ns": GPU_FD_TERMINAL_EXIT_GRACE_NS,
            "require_complete_required_fd_set_before_error": True,
            "require_zero_prior_read_errors": True,
            "require_terminal_error_as_final_sample": True,
            "required_exit_returncode": 0,
        },
    }


def _definition(
    *,
    image: str,
    resolved_image_id: str,
    gpu_index: int,
    gpu_uuid: str,
    nonce: str,
    session_root: Path,
    video_pin: Mapping[str, Any],
    model_pins: Mapping[str, Any],
    generated_pins: Mapping[str, Any],
    static_pin: Mapping[str, Any],
    parser_build_pin: Mapping[str, Any],
    parser_sha256: str,
    runtime_control_manifest_sha256: str,
    container_identity: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "requested_image": image,
        "resolved_image_id": resolved_image_id,
        "gpu": {"index": gpu_index, "uuid": gpu_uuid, "compute_capability": "8.6"},
        "campaign_nonce": nonce,
        "session_id": f"ds9-gpu-smoke-{nonce}",
        "session_root": _relative(session_root, project_root),
        "container_process_identity": dict(container_identity),
        "required_checks": list(REQUIRED_CHECKS),
        "parser": {
            "path": PARSER_LIBRARY.as_posix(),
            "sha256": parser_sha256,
            "static_sm86_cubin_required": True,
            "source_lineage": source_lineage_contract(),
        },
        "runtime_control_manifest_sha256": runtime_control_manifest_sha256,
        "static_candidate_receipt": dict(static_pin),
        "parser_production_build_receipt": dict(parser_build_pin),
        "inputs": {"video": dict(video_pin), "models": dict(model_pins)},
        "generated_inputs": dict(generated_pins),
        "runs": [
            {
                "run_id": run_id,
                "profile": int(run_id.split("-", 1)[0]),
                "parser": run_id.split("-", 1)[1],
                "parser_function": PARSER_FUNCTIONS[run_id.split("-", 1)[1]],
                "argv": [
                    "deepstream-app",
                    "-c",
                    f"/contract/deepstream-{run_id}.txt",
                ],
                "environment": _kernel_proof_environment(nonce, run_id),
                "kernel_marker": (
                    f"{run_id}/{KERNEL_PROOF_MARKER_NAME}"
                    if run_id.endswith("-cuda")
                    else None
                ),
            }
            for run_id in RUN_IDS
        ],
        "environment": {
            "CUDA_MODULE_LOADING": "EAGER",
            "CUDA_LAUNCH_BLOCKING": "1",
        },
        "parity_thresholds": {
            "bbox_abs_tolerance_px": PARITY_BBOX_ABS_TOLERANCE_PX,
            "confidence_abs_tolerance": PARITY_CONFIDENCE_ABS_TOLERANCE,
            "minimum_iou": PARITY_MIN_IOU,
            "minimum_total_detections_per_run": MIN_TOTAL_DETECTIONS,
            "require_equal_frame_set": True,
            "require_equal_detection_count": True,
            "require_all_detections_matched": True,
        },
        "engine_policy": {
            "onnx_in_config": False,
            "positive_log_patterns": list(POSITIVE_ENGINE_PATTERNS),
            "forbidden_log_patterns": list(ENGINE_FALLBACK_PATTERNS),
            "engine_sha_must_be_stable": True,
        },
        "gpu_fd_policy": _gpu_fd_policy(),
        "kernel_proof_policy": _kernel_proof_policy(),
    }


def build_plan(
    *,
    session_root: Path,
    image: str,
    gpu_index: int,
    gpu_uuid: str,
    static_candidate_receipt: Path,
    parser_build_receipt: Path,
    authorization: Path | None,
    video: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if gpu_index < 0:
        raise Ds9GpuSmokeError("GPU index cannot be negative")
    if GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise Ds9GpuSmokeError("GPU UUID must be explicit and exact")
    session = _inside_root(session_root, project_root)
    try:
        nonce = session.relative_to(project_root / SESSION_PREFIX).as_posix()
    except ValueError as exc:
        raise Ds9GpuSmokeError("session root must be under the DS9 smoke prefix") from exc
    if "/" in nonce or NONCE_RE.fullmatch(nonce) is None:
        raise Ds9GpuSmokeError("session root basename must be a 64-hex nonce")
    inputs_dir = session / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for run_id in RUN_IDS:
        profile_text, parser = run_id.split("-", 1)
        profile = int(profile_text)
        _write_or_verify(
            inputs_dir / f"infer-{run_id}.txt",
            render_infer_config(profile, parser),
        )
        _write_or_verify(
            inputs_dir / f"deepstream-{run_id}.txt",
            render_deepstream_config(profile, parser),
        )

    video_path = _inside_root(video, project_root)
    video_pin = file_pin(video_path, project_root=project_root)
    models = _model_paths(project_root)
    model_pins: dict[str, Any] = {}
    for profile, paths in models.items():
        model_pins[str(profile)] = {
            name: file_pin(path, project_root=project_root)
            for name, path in paths.items()
        }
    generated_pins = {
        path.name: file_pin(path, project_root=project_root)
        for path in sorted(inputs_dir.glob("*.txt"))
    }
    blockers: list[str] = []
    try:
        static, static_pin = _offline_static_candidate(
            static_candidate_receipt, project_root=project_root
        )
    except (OSError, Ds9GpuSmokeError) as exc:
        blockers.append(f"static_candidate_invalid={type(exc).__name__}: {exc}")
        static = None
        static_pin = None
    parser_build = None
    parser_build_pin = None
    if static is not None:
        image_data = static["image"]
        resolved_image_id = image_data["resolved_image_id"]
        parser_sha = image_data.get("labels", {}).get(
            "com.deepsafe.deepstream-yolo.parser-sha256"
        )
        control_sha = static.get("runtime_controls", {}).get("pin", {}).get("sha256")
        try:
            parser_build, parser_build_pin = validate_production_receipt(
                parser_build_receipt,
                project_root=project_root,
                resolved_image_id=resolved_image_id,
                parser_sha256=parser_sha,
            )
        except (OSError, ValueError, Ds9GpuSmokeError, Exception) as exc:
            # The broad catch is intentional here: the bootstrap module has its
            # own typed fail-closed exception and plan generation must report a
            # blocker rather than claim readiness.
            blockers.append(f"parser_build_invalid={type(exc).__name__}: {exc}")
    else:
        resolved_image_id = None
        parser_sha = None
        control_sha = None
    definition = None
    definition_sha = None
    process_identity = container_process_identity()
    if all(
        value is not None
        for value in (
            resolved_image_id,
            parser_sha,
            control_sha,
            static_pin,
            parser_build_pin,
        )
    ):
        definition = _definition(
            image=image,
            resolved_image_id=resolved_image_id,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            nonce=nonce,
            session_root=session,
            video_pin=video_pin,
            model_pins=model_pins,
            generated_pins=generated_pins,
            static_pin=static_pin,
            parser_build_pin=parser_build_pin,
            parser_sha256=parser_sha,
            runtime_control_manifest_sha256=control_sha,
            container_identity=process_identity,
            project_root=project_root,
        )
        definition_sha = canonical_sha256(definition)
        contract = dict(definition)
        contract["definition_sha256"] = definition_sha
        contract_path = inputs_dir / "probe-contract.json"
        content = json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_or_verify(contract_path, content)
        contract_pin = file_pin(contract_path, project_root=project_root)
    else:
        contract_pin = None

    auth_summary = None
    if authorization is None:
        blockers.append("operator_authorization_missing")
    elif definition_sha is None:
        blockers.append("operator_authorization_cannot_bind_incomplete_definition")
    else:
        try:
            auth, auth_pin = validate_authorization(
                authorization,
                project_root=project_root,
                expected_nonce=nonce,
                expected_session_root=_relative(session, project_root),
                expected_image_id=resolved_image_id,
                expected_definition_sha256=definition_sha,
                expected_static_receipt_sha256=static_pin["sha256"],
                expected_parser_build_receipt_sha256=parser_build_pin["sha256"],
                expected_gpu_index=gpu_index,
                expected_gpu_uuid=gpu_uuid,
            )
            auth_summary = {"authorization": auth, "pin": auth_pin}
        except (OSError, Ds9GpuSmokeError) as exc:
            blockers.append(f"operator_authorization_invalid={type(exc).__name__}: {exc}")

    status = "ready_for_authorized_execution" if not blockers else "blocked"
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "production_ready": False,
        "created_at_utc": utc_now(),
        "session_root": _relative(session, project_root),
        "campaign_nonce": nonce,
        "requested_image": image,
        "resolved_image_id": resolved_image_id,
        "gpu": {"index": gpu_index, "uuid": gpu_uuid},
        "container_process_identity": process_identity,
        "required_checks": list(REQUIRED_CHECKS),
        "definition": definition,
        "definition_sha256": definition_sha,
        "contract": contract_pin,
        "authorization": auth_summary,
        "blockers": blockers,
        "execution": {
            "explicit_execute_required": True,
            "guarded_process_required": True,
            "single_use": True,
        },
        "dry_run": {
            "docker_called": False,
            "gpu_process_started": False,
            "gpu_telemetry_queried": False,
        },
    }
    _atomic_json(session / "plan.json", plan)
    return plan


def validate_authorization(
    path: Path,
    *,
    project_root: Path,
    expected_nonce: str,
    expected_session_root: str,
    expected_image_id: str,
    expected_definition_sha256: str,
    expected_static_receipt_sha256: str,
    expected_parser_build_receipt_sha256: str,
    expected_gpu_index: int,
    expected_gpu_uuid: str,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, pin = _read_json(path, project_root=project_root, immutable=True)
    expected_fields = {
        "schema_version",
        "status",
        "operator_identity",
        "campaign_nonce",
        "session_id",
        "authorized_session_root",
        "issued_at_utc",
        "expires_at_utc",
        "resolved_image_id",
        "smoke_definition_sha256",
        "static_candidate_receipt_sha256",
        "parser_production_build_receipt_sha256",
        "gpu_index",
        "gpu_uuid",
        "approved_checks",
        "single_use",
    }
    if set(payload) != expected_fields:
        raise Ds9GpuSmokeError("authorization fields differ from the closed contract")
    if payload.get("schema_version") != AUTH_SCHEMA_VERSION or payload.get("status") != "approved":
        raise Ds9GpuSmokeError("authorization schema/status differs")
    identity = payload.get("operator_identity")
    if not isinstance(identity, str) or not (3 <= len(identity.strip()) <= 128):
        raise Ds9GpuSmokeError("authorization operator identity is invalid")
    expected = {
        "campaign_nonce": expected_nonce,
        "session_id": f"ds9-gpu-smoke-{expected_nonce}",
        "authorized_session_root": expected_session_root,
        "resolved_image_id": expected_image_id,
        "smoke_definition_sha256": expected_definition_sha256,
        "static_candidate_receipt_sha256": expected_static_receipt_sha256,
        "parser_production_build_receipt_sha256": expected_parser_build_receipt_sha256,
        "gpu_index": expected_gpu_index,
        "gpu_uuid": expected_gpu_uuid,
        "approved_checks": list(REQUIRED_CHECKS),
        "single_use": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise Ds9GpuSmokeError(f"authorization binding differs: {key}")
    issued = _parse_time(payload.get("issued_at_utc"), "authorization issued_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "authorization expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= issued or expires - issued > MAX_AUTH_AGE:
        raise Ds9GpuSmokeError("authorization lifetime exceeds 24h")
    if current < issued or current >= expires:
        raise Ds9GpuSmokeError("authorization is not current")
    return payload, pin


def build_docker_command(
    *,
    plan: Mapping[str, Any],
    project_root: Path,
    video: Path,
) -> tuple[list[str], str]:
    nonce = str(plan["campaign_nonce"])
    session = _inside_root(Path(str(plan["session_root"])), project_root)
    raw = session / "raw"
    inputs = session / "inputs"
    models = _model_paths(project_root)
    container_name = f"deepsafe-ds9-smoke-{nonce[:16]}"
    image = str(plan["resolved_image_id"])
    if IMAGE_ID_RE.fullmatch(image) is None:
        raise Ds9GpuSmokeError("smoke command requires an immutable image ID")
    process_identity = plan.get("container_process_identity")
    if process_identity != container_process_identity():
        raise Ds9GpuSmokeError("smoke container process identity differs from host")
    uid = int(process_identity["uid"])
    gid = int(process_identity["gid"])
    expected_input_names = {
        "probe-contract.json",
        *{f"infer-{run_id}.txt" for run_id in RUN_IDS},
        *{f"deepstream-{run_id}.txt" for run_id in RUN_IDS},
    }
    observed_input_names = {item.name for item in inputs.iterdir()}
    if observed_input_names != expected_input_names:
        raise Ds9GpuSmokeError("smoke container input file set differs")
    for item in inputs.iterdir():
        metadata = os.lstat(item)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o440
            or metadata.st_nlink != 1
            or metadata.st_uid != uid
            or metadata.st_gid != gid
        ):
            raise Ds9GpuSmokeError(
                f"smoke container input ownership/mode differs: {item.name}"
            )
    raw_metadata = os.lstat(raw)
    if (
        not stat.S_ISDIR(raw_metadata.st_mode)
        or raw_metadata.st_uid != uid
        or raw_metadata.st_gid != gid
        or stat.S_IMODE(raw_metadata.st_mode) & 0o700 != 0o700
    ):
        raise Ds9GpuSmokeError("smoke raw output ownership/mode differs")
    gpu_index = int(plan["gpu"]["index"])
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--pull=never",
        "--network=none",
        "--gpus",
        f"device={gpu_index}",
        "--user",
        f"{uid}:{gid}",
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "512",
        "--ipc=private",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=536870912",
        "--env",
        "CUDA_MODULE_LOADING=EAGER",
        "--env",
        "CUDA_LAUNCH_BLOCKING=1",
        "--mount",
        f"type=bind,src={inputs},dst=/contract,readonly",
        "--mount",
        f"type=bind,src={raw},dst=/evidence",
        "--mount",
        f"type=bind,src={_inside_root(video, project_root)},dst=/input/smoke.mp4,readonly",
    ]
    for profile in PROFILES:
        command.extend(
            [
                "--mount",
                f"type=bind,src={models[profile]['engine']},dst=/models/{profile}.engine,readonly",
                "--mount",
                f"type=bind,src={models[profile]['labels']},dst=/models/{profile}.labels,readonly",
            ]
        )
    command.extend(
        [
            image,
            "python3",
            CONTAINER_WORKER.as_posix(),
            "--inside-container",
            "--contract",
            "/contract/probe-contract.json",
            "--inner-output",
            "/evidence",
        ]
    )
    return command, container_name


def validate_contract_semantics(
    contract: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "requested_image",
        "resolved_image_id",
        "gpu",
        "campaign_nonce",
        "session_id",
        "session_root",
        "container_process_identity",
        "required_checks",
        "parser",
        "runtime_control_manifest_sha256",
        "static_candidate_receipt",
        "parser_production_build_receipt",
        "inputs",
        "generated_inputs",
        "runs",
        "environment",
        "parity_thresholds",
        "engine_policy",
        "gpu_fd_policy",
        "kernel_proof_policy",
        "definition_sha256",
    }
    if set(contract) != expected_fields or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise Ds9GpuSmokeError("smoke contract field/schema set differs")
    definition = dict(contract)
    definition_sha = definition.pop("definition_sha256", None)
    if canonical_sha256(definition) != definition_sha:
        raise Ds9GpuSmokeError("smoke contract definition hash differs")
    nonce = contract.get("campaign_nonce")
    if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        raise Ds9GpuSmokeError("smoke contract nonce is invalid")
    session = _inside_root(Path(str(contract.get("session_root"))), project_root)
    if contract.get("session_id") != f"ds9-gpu-smoke-{nonce}":
        raise Ds9GpuSmokeError("smoke contract session ID differs")
    if contract.get("container_process_identity") != container_process_identity():
        raise Ds9GpuSmokeError("smoke contract container process identity differs")
    if contract.get("required_checks") != list(REQUIRED_CHECKS):
        raise Ds9GpuSmokeError("smoke contract check set differs")
    if IMAGE_ID_RE.fullmatch(str(contract.get("resolved_image_id"))) is None:
        raise Ds9GpuSmokeError("smoke contract image ID is not immutable")
    gpu = contract.get("gpu")
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"index", "uuid", "compute_capability"}
        or not isinstance(gpu.get("index"), int)
        or gpu["index"] < 0
        or GPU_UUID_RE.fullmatch(str(gpu.get("uuid"))) is None
        or gpu.get("compute_capability") != "8.6"
    ):
        raise Ds9GpuSmokeError("smoke contract GPU binding differs")
    parser = contract.get("parser")
    if (
        parser
        != {
            "path": PARSER_LIBRARY.as_posix(),
            "sha256": parser.get("sha256") if isinstance(parser, dict) else None,
            "static_sm86_cubin_required": True,
            "source_lineage": source_lineage_contract(),
        }
        or SHA256_RE.fullmatch(str(parser.get("sha256"))) is None
    ):
        raise Ds9GpuSmokeError("smoke contract parser binding differs")
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"video", "models"}:
        raise Ds9GpuSmokeError("smoke contract input set differs")
    video_pin = _pin_matches(inputs["video"], project_root=project_root)
    expected_model_pins: dict[str, Any] = {}
    models = _model_paths(project_root)
    for profile in PROFILES:
        expected_model_pins[str(profile)] = {
            name: file_pin(path, project_root=project_root)
            for name, path in models[profile].items()
        }
    if inputs.get("models") != expected_model_pins:
        raise Ds9GpuSmokeError("smoke contract model pins differ from live inputs")
    generated = contract.get("generated_inputs")
    expected_generated_names = {
        f"infer-{run_id}.txt" for run_id in RUN_IDS
    } | {f"deepstream-{run_id}.txt" for run_id in RUN_IDS}
    if not isinstance(generated, dict) or set(generated) != expected_generated_names:
        raise Ds9GpuSmokeError("smoke generated config set differs")
    live_generated: dict[str, Any] = {}
    for name in sorted(expected_generated_names):
        path = session / "inputs" / name
        if name.startswith("infer-"):
            run_id = name.removeprefix("infer-").removesuffix(".txt")
            profile_text, parser_name = run_id.split("-", 1)
            expected_text = render_infer_config(int(profile_text), parser_name)
        else:
            run_id = name.removeprefix("deepstream-").removesuffix(".txt")
            profile_text, parser_name = run_id.split("-", 1)
            expected_text = render_deepstream_config(int(profile_text), parser_name)
        if path.read_text(encoding="utf-8") != expected_text:
            raise Ds9GpuSmokeError(f"smoke generated config content differs: {name}")
        live_generated[name] = file_pin(path, project_root=project_root)
    if generated != live_generated:
        raise Ds9GpuSmokeError("smoke generated config pins changed")
    expected = _definition(
        image=str(contract["requested_image"]),
        resolved_image_id=str(contract["resolved_image_id"]),
        gpu_index=gpu["index"],
        gpu_uuid=gpu["uuid"],
        nonce=nonce,
        session_root=session,
        video_pin=video_pin,
        model_pins=expected_model_pins,
        generated_pins=live_generated,
        static_pin=contract["static_candidate_receipt"],
        parser_build_pin=contract["parser_production_build_receipt"],
        parser_sha256=parser["sha256"],
        runtime_control_manifest_sha256=str(
            contract["runtime_control_manifest_sha256"]
        ),
        container_identity=contract["container_process_identity"],
        project_root=project_root,
    )
    if definition != expected:
        raise Ds9GpuSmokeError("smoke contract semantics differ from closed definition")
    return definition


def _inner_file_sha(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _inner_contract_pin_matches(path: Path, pin: Mapping[str, Any]) -> None:
    size, digest = _inner_file_sha(path)
    if size != pin.get("bytes") or digest != pin.get("sha256"):
        raise Ds9GpuSmokeError(f"container input pin differs: {path}")


def _structured_fd_error(
    *, scope: str, path: Path, exc: OSError, observed_at_monotonic_ns: int
) -> dict[str, Any]:
    error_number = exc.errno if isinstance(exc.errno, int) else -1
    return {
        "scope": scope,
        "path": path.as_posix(),
        "errno": error_number,
        "errno_name": errno.errorcode.get(error_number, "UNKNOWN"),
        "exception_type": type(exc).__name__,
        "observed_at_monotonic_ns": observed_at_monotonic_ns,
    }


def _nvidia_fd_sample(pid: int, sample_index: int) -> dict[str, Any]:
    observed: set[str] = set()
    errors: list[dict[str, Any]] = []
    fd_root = Path(f"/proc/{pid}/fd")
    started_at = time.monotonic_ns()
    try:
        entries = list(fd_root.iterdir())
    except OSError as exc:
        error_at = time.monotonic_ns()
        errors.append(
            _structured_fd_error(
                scope="proc_fd_root",
                path=fd_root,
                exc=exc,
                observed_at_monotonic_ns=error_at,
            )
        )
    else:
        for entry in entries:
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                # The descriptor may close between the directory snapshot and
                # readlink.  This is not a root-read failure and proves no FD.
                continue
            except OSError as exc:
                errors.append(
                    _structured_fd_error(
                        scope="fd_entry",
                        path=entry,
                        exc=exc,
                        observed_at_monotonic_ns=time.monotonic_ns(),
                    )
                )
                continue
            if target.startswith("/dev/nvidia"):
                observed.add(target)
    completed_at = time.monotonic_ns()
    return {
        "sample_index": sample_index,
        "pid": pid,
        "proc_fd_root": fd_root.as_posix(),
        "started_at_monotonic_ns": started_at,
        "completed_at_monotonic_ns": completed_at,
        "observed_nvidia_device_fds": sorted(observed),
        "read_errors": errors,
    }


def _required_nvidia_fd_set_complete(observed: Iterable[str]) -> bool:
    values = tuple(observed)
    return all(
        any(re.fullmatch(pattern, item) for item in values)
        for pattern in REQUIRED_NVIDIA_FD_PATTERNS
    )


def _terminal_fd_error_candidate(
    *,
    sample: Mapping[str, Any],
    pid: int,
    observed_before_error: set[str],
    prior_read_errors: Sequence[Mapping[str, Any]],
) -> bool:
    errors = sample.get("read_errors")
    expected_root = f"/proc/{pid}/fd"
    if (
        not isinstance(errors, list)
        or len(errors) != 1
        or prior_read_errors
        or sample.get("pid") != pid
        or sample.get("proc_fd_root") != expected_root
        or not _required_nvidia_fd_set_complete(observed_before_error)
    ):
        return False
    error = errors[0]
    if not isinstance(error, dict):
        return False
    error_number = error.get("errno")
    return (
        error.get("scope") == "proc_fd_root"
        and error.get("path") == expected_root
        and error_number in GPU_FD_TERMINAL_ROOT_ERRNOS
        and error.get("errno_name") == errno.errorcode[error_number]
        and error.get("exception_type")
        == GPU_FD_TERMINAL_EXCEPTION_TYPES[error_number]
    )


def _observe_process_exit_within_terminal_grace(
    process: subprocess.Popen[str], *, error_observed_at_monotonic_ns: int
) -> tuple[int, int] | None:
    deadline = error_observed_at_monotonic_ns + GPU_FD_TERMINAL_EXIT_GRACE_NS
    while True:
        observed_at = time.monotonic_ns()
        if observed_at > deadline:
            return None
        returncode = process.poll()
        if returncode is not None:
            return observed_at, int(returncode)
        remaining_seconds = (deadline - observed_at) / 1_000_000_000
        time.sleep(min(GPU_FD_SAMPLE_SECONDS, remaining_seconds))


def _terminal_fd_evidence(
    *,
    sample: Mapping[str, Any],
    root_error: Mapping[str, Any],
    observed_before_error: set[str],
    exit_observed_at_monotonic_ns: int,
    exit_returncode: int,
) -> dict[str, Any]:
    error_at = int(root_error["observed_at_monotonic_ns"])
    return {
        "classification": "same_process_proc_fd_root_terminal_teardown",
        "sample_index": sample["sample_index"],
        "pid": sample["pid"],
        "proc_fd_root": sample["proc_fd_root"],
        "root_error": dict(root_error),
        "required_fds_observed_before_error": sorted(observed_before_error),
        "prior_read_error_count": 0,
        "error_observed_at_monotonic_ns": error_at,
        "exit_observed_at_monotonic_ns": exit_observed_at_monotonic_ns,
        "elapsed_to_exit_monotonic_ns": (
            exit_observed_at_monotonic_ns - error_at
        ),
        "exit_grace_monotonic_ns": GPU_FD_TERMINAL_EXIT_GRACE_NS,
        "exit_returncode": exit_returncode,
    }


def _inside_container(contract_path: Path, output_root: Path) -> dict[str, Any]:
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ds9GpuSmokeError("container smoke contract is invalid") from exc
    if not isinstance(contract, dict) or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise Ds9GpuSmokeError("container smoke contract schema differs")
    if contract.get("container_process_identity") != container_process_identity():
        raise Ds9GpuSmokeError("container process identity differs from contract")
    if contract.get("required_checks") != list(REQUIRED_CHECKS):
        raise Ds9GpuSmokeError("container required check set differs")
    if contract.get("gpu_fd_policy") != _gpu_fd_policy():
        raise Ds9GpuSmokeError("container GPU FD policy differs")
    expected_definition_sha = contract.pop("definition_sha256", None)
    if canonical_sha256(contract) != expected_definition_sha:
        raise Ds9GpuSmokeError("container contract definition hash differs")
    contract["definition_sha256"] = expected_definition_sha
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise Ds9GpuSmokeError("container raw output must start empty")

    _inner_contract_pin_matches(Path("/input/smoke.mp4"), contract["inputs"]["video"])
    for profile in PROFILES:
        model = contract["inputs"]["models"][str(profile)]
        _inner_contract_pin_matches(Path(f"/models/{profile}.engine"), model["engine"])
        _inner_contract_pin_matches(Path(f"/models/{profile}.labels"), model["labels"])
    for name, pin in contract["generated_inputs"].items():
        _inner_contract_pin_matches(Path("/contract") / name, pin)
    _, parser_sha = _inner_file_sha(PARSER_LIBRARY)
    if parser_sha != contract["parser"]["sha256"]:
        raise Ds9GpuSmokeError("container parser SHA differs from contract")

    identity_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    identity = subprocess.run(
        identity_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    identity_log = output_root / "gpu-identity.log"
    identity_log.write_text(identity.stdout or "", encoding="utf-8")
    if identity.returncode != 0:
        raise Ds9GpuSmokeError("container GPU identity query failed")

    run_results: dict[str, Any] = {}
    engine_before = {
        profile: _inner_file_sha(Path(f"/models/{profile}.engine"))[1]
        for profile in PROFILES
    }
    for run in contract["runs"]:
        run_id = run["run_id"]
        if run_id not in RUN_IDS or run["argv"] != [
            "deepstream-app",
            "-c",
            f"/contract/deepstream-{run_id}.txt",
        ]:
            raise Ds9GpuSmokeError("container run command differs")
        expected_run_environment = _kernel_proof_environment(
            contract["campaign_nonce"], run_id
        )
        expected_marker = (
            f"{run_id}/{KERNEL_PROOF_MARKER_NAME}"
            if run_id.endswith("-cuda")
            else None
        )
        if (
            run.get("environment") != expected_run_environment
            or run.get("kernel_marker") != expected_marker
        ):
            raise Ds9GpuSmokeError("container kernel proof run contract differs")
        run_root = output_root / run_id
        kitti = run_root / "kitti"
        kitti.mkdir(parents=True, exist_ok=False)
        log_path = run_root / "deepstream.log"
        observed_fds: set[str] = set()
        fd_errors: list[dict[str, Any]] = []
        fd_samples: list[dict[str, Any]] = []
        fd_terminal_evidence: dict[str, Any] | None = None
        run_environment = {**os.environ, **contract["environment"]}
        for environment_name in KERNEL_PROOF_ENV.values():
            run_environment.pop(environment_name, None)
        run_environment.update(expected_run_environment)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.Popen(
                run["argv"],
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=run_environment,
                start_new_session=True,
            )
            process_pid = process.pid
            while process.poll() is None:
                observed_before_error = set(observed_fds)
                sample = _nvidia_fd_sample(process.pid, len(fd_samples) + 1)
                fd_samples.append(sample)
                observed_fds.update(sample["observed_nvidia_device_fds"])
                errors = sample["read_errors"]
                if errors:
                    if _terminal_fd_error_candidate(
                        sample=sample,
                        pid=process.pid,
                        observed_before_error=observed_before_error,
                        prior_read_errors=fd_errors,
                    ):
                        terminal_exit = _observe_process_exit_within_terminal_grace(
                            process,
                            error_observed_at_monotonic_ns=errors[0][
                                "observed_at_monotonic_ns"
                            ],
                        )
                        if terminal_exit is not None and terminal_exit[1] == 0:
                            fd_terminal_evidence = _terminal_fd_evidence(
                                sample=sample,
                                root_error=errors[0],
                                observed_before_error=observed_before_error,
                                exit_observed_at_monotonic_ns=terminal_exit[0],
                                exit_returncode=terminal_exit[1],
                            )
                            break
                    fd_errors.extend(errors)
                    break
                time.sleep(GPU_FD_SAMPLE_SECONDS)
            if process.returncode is None:
                returncode = int(process.wait())
            else:
                returncode = int(process.returncode)
        if returncode != 0:
            raise Ds9GpuSmokeError(f"DeepStream smoke run failed: {run_id}")
        marker_path = run_root / KERNEL_PROOF_MARKER_NAME
        if run_id.endswith("-cuda"):
            try:
                marker_metadata = os.lstat(marker_path)
            except OSError as exc:
                raise Ds9GpuSmokeError(
                    f"CUDA kernel proof marker missing: {run_id}"
                ) from exc
            if (
                not stat.S_ISREG(marker_metadata.st_mode)
                or stat.S_IMODE(marker_metadata.st_mode) != 0o440
                or marker_metadata.st_nlink != 1
            ):
                raise Ds9GpuSmokeError(
                    f"CUDA kernel proof marker is not immutable: {run_id}"
                )
        elif marker_path.exists() or marker_path.is_symlink():
            raise Ds9GpuSmokeError(f"CPU run emitted a CUDA proof marker: {run_id}")
        run_results[run_id] = {
            "argv": run["argv"],
            "returncode": returncode,
            "pid": process_pid,
            "log": f"{run_id}/deepstream.log",
            "kitti": f"{run_id}/kitti",
            "nvidia_device_fds_observed": sorted(observed_fds),
            "gpu_fd_sample_count": len(fd_samples),
            "gpu_fd_samples": fd_samples,
            "gpu_fd_read_errors": fd_errors,
            "gpu_fd_terminal_evidence": fd_terminal_evidence,
            "kernel_marker": expected_marker,
            "kernel_proof_environment": expected_run_environment,
        }
    engine_after = {
        profile: _inner_file_sha(Path(f"/models/{profile}.engine"))[1]
        for profile in PROFILES
    }
    if engine_before != engine_after:
        raise Ds9GpuSmokeError("engine changed during container smoke")
    result = {
        "schema_version": INNER_SCHEMA_VERSION,
        "status": "completed_for_host_replay",
        "created_at_utc": utc_now(),
        "campaign_nonce": contract["campaign_nonce"],
        "session_id": contract["session_id"],
        "resolved_image_id": contract["resolved_image_id"],
        "definition_sha256": expected_definition_sha,
        "parser_sha256": parser_sha,
        "gpu_identity_command": identity_command,
        "gpu_identity_log": "gpu-identity.log",
        "engine_sha256_before": {str(k): v for k, v in engine_before.items()},
        "engine_sha256_after": {str(k): v for k, v in engine_after.items()},
        "runs": run_results,
    }
    (output_root / "inner-run.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_gpu_identity(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise Ds9GpuSmokeError("GPU identity log must contain exactly one device")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise Ds9GpuSmokeError("GPU identity row field count differs")
    try:
        index = int(fields[0])
    except ValueError as exc:
        raise Ds9GpuSmokeError("GPU identity index is invalid") from exc
    if GPU_UUID_RE.fullmatch(fields[1]) is None or fields[3] != "8.6":
        raise Ds9GpuSmokeError("GPU UUID/compute capability differs")
    return {"index": index, "uuid": fields[1], "name": fields[2], "compute_capability": fields[3]}


def _parse_kitti_file(
    path: Path, *, project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    content, pin, _ = _read_bounded_file(
        path,
        project_root=project_root,
        max_bytes=MAX_KITTI_FILE_BYTES,
        label="KITTI output",
    )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise Ds9GpuSmokeError(f"cannot read KITTI output: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if len(line.encode("utf-8")) > MAX_KITTI_LINE_BYTES:
            raise Ds9GpuSmokeError(
                f"KITTI line exceeds byte limit: {path}:{line_number}"
            )
        fields = line.split()
        if len(fields) != 16:
            raise Ds9GpuSmokeError(f"KITTI field count differs: {path}:{line_number}")
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise Ds9GpuSmokeError(f"KITTI numeric value invalid: {path}:{line_number}") from exc
        if not all(math.isfinite(value) for value in values):
            raise Ds9GpuSmokeError(f"KITTI non-finite value: {path}:{line_number}")
        left, top, right, bottom = values[3:7]
        confidence = values[14]
        if right < left or bottom < top or not (0.0 <= confidence <= 1.0):
            raise Ds9GpuSmokeError(f"KITTI bbox/confidence invalid: {path}:{line_number}")
        detections.append(
            {
                "class": fields[0],
                "bbox": [left, top, right, bottom],
                "confidence": confidence,
            }
        )
    return detections, pin


def _load_kitti_dir(
    path: Path, *, project_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not path.is_dir() or path.is_symlink():
        raise Ds9GpuSmokeError(f"KITTI output directory invalid: {path}")
    entries = sorted(path.iterdir(), key=lambda item: item.name)
    if (
        not entries
        or len(entries) > MAX_KITTI_FILES
        or any(
            item.is_symlink()
            or not item.is_file()
            or KITTI_NAME_RE.fullmatch(item.name) is None
            for item in entries
        )
    ):
        raise Ds9GpuSmokeError(f"KITTI output file set invalid: {path}")
    parsed: dict[str, list[dict[str, Any]]] = {}
    pins: list[dict[str, Any]] = []
    total_bytes = 0
    for item in entries:
        detections, pin = _parse_kitti_file(item, project_root=project_root)
        parsed[item.name] = detections
        pins.append(pin)
        total_bytes += pin["bytes"]
        if total_bytes > MAX_KITTI_TOTAL_BYTES:
            raise Ds9GpuSmokeError("KITTI output tree exceeds its byte limit")
    if [item.name for item in entries] != sorted(item.name for item in path.iterdir()):
        raise Ds9GpuSmokeError("KITTI output directory changed while replaying")
    return parsed, pins


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else (1.0 if first == second else 0.0)


def _parity(
    cpu: Mapping[str, list[dict[str, Any]]],
    cuda: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    frames_equal = set(cpu) == set(cuda)
    total_cpu = sum(len(items) for items in cpu.values())
    total_cuda = sum(len(items) for items in cuda.values())
    matched = 0
    max_bbox_delta = 0.0
    max_conf_delta = 0.0
    min_iou = 1.0
    unmatched_frames: list[str] = []
    for frame in sorted(set(cpu) | set(cuda)):
        cpu_items = cpu.get(frame, [])
        cuda_items = cuda.get(frame, [])
        if len(cpu_items) != len(cuda_items):
            unmatched_frames.append(frame)
            continue
        available = set(range(len(cuda_items)))
        frame_ok = True
        for item in cpu_items:
            candidates: list[tuple[float, float, float, int]] = []
            for index in available:
                other = cuda_items[index]
                if item["class"] != other["class"]:
                    continue
                bbox_delta = max(
                    abs(a - b) for a, b in zip(item["bbox"], other["bbox"])
                )
                confidence_delta = abs(item["confidence"] - other["confidence"])
                overlap = _iou(item["bbox"], other["bbox"])
                if (
                    bbox_delta <= PARITY_BBOX_ABS_TOLERANCE_PX
                    and confidence_delta <= PARITY_CONFIDENCE_ABS_TOLERANCE
                    and overlap >= PARITY_MIN_IOU
                ):
                    candidates.append((bbox_delta, confidence_delta, -overlap, index))
            if not candidates:
                frame_ok = False
                break
            bbox_delta, confidence_delta, negative_overlap, selected = min(candidates)
            available.remove(selected)
            matched += 1
            max_bbox_delta = max(max_bbox_delta, bbox_delta)
            max_conf_delta = max(max_conf_delta, confidence_delta)
            min_iou = min(min_iou, -negative_overlap)
        if not frame_ok or available:
            unmatched_frames.append(frame)
    passed = (
        frames_equal
        and total_cpu == total_cuda
        and total_cpu >= MIN_TOTAL_DETECTIONS
        and matched == total_cpu
        and not unmatched_frames
    )
    return {
        "status": "pass" if passed else "fail",
        "frame_sets_equal": frames_equal,
        "frames_cpu": len(cpu),
        "frames_cuda": len(cuda),
        "detections_cpu": total_cpu,
        "detections_cuda": total_cuda,
        "matched_detections": matched,
        "unmatched_frame_count": len(set(unmatched_frames)),
        "unmatched_frames_first_50": sorted(set(unmatched_frames))[:50],
        "max_bbox_abs_delta_px": max_bbox_delta,
        "max_confidence_abs_delta": max_conf_delta,
        "minimum_matched_iou": min_iou if matched else None,
        "thresholds": {
            "bbox_abs_tolerance_px": PARITY_BBOX_ABS_TOLERANCE_PX,
            "confidence_abs_tolerance": PARITY_CONFIDENCE_ABS_TOLERANCE,
            "minimum_iou": PARITY_MIN_IOU,
            "minimum_total_detections": MIN_TOTAL_DETECTIONS,
        },
    }


def _validate_kernel_marker(
    path: Path,
    *,
    run_id: str,
    expected_pid: int,
    contract: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker, pin = _read_json(
        path,
        project_root=project_root,
        immutable=True,
        max_bytes=MAX_KERNEL_MARKER_BYTES,
    )
    expected_fields = {
        "binary_version",
        "campaign_nonce",
        "cuda_device_synchronize",
        "cuda_func_get_attributes",
        "cuda_get_last_error",
        "kernel",
        "launch_count_at_marker",
        "marker_path",
        "marker_write_count",
        "number_of_blocks",
        "output_size",
        "pid",
        "ptx_version",
        "run_id",
        "schema_version",
        "threads_per_block",
    }
    if set(marker) != expected_fields:
        raise Ds9GpuSmokeError(f"CUDA kernel marker field set differs: {run_id}")
    expected_exact = {
        "schema_version": KERNEL_PROOF_SCHEMA_VERSION,
        "campaign_nonce": contract["campaign_nonce"],
        "run_id": run_id,
        "pid": expected_pid,
        "kernel": KERNEL_PROOF_KERNEL_NAME,
        "binary_version": 86,
        "cuda_get_last_error": 0,
        "cuda_device_synchronize": 0,
        "cuda_func_get_attributes": 0,
        "launch_count_at_marker": 1,
        "marker_write_count": 1,
        "threads_per_block": 1024,
        "marker_path": f"/evidence/{run_id}/{KERNEL_PROOF_MARKER_NAME}",
    }
    for name, expected in expected_exact.items():
        if marker.get(name) != expected:
            raise Ds9GpuSmokeError(
                f"CUDA kernel marker binding differs: {run_id}:{name}"
            )
    output_size = marker.get("output_size")
    blocks = marker.get("number_of_blocks")
    ptx_version = marker.get("ptx_version")
    if (
        isinstance(output_size, bool)
        or not isinstance(output_size, int)
        or output_size <= 0
        or isinstance(blocks, bool)
        or not isinstance(blocks, int)
        or blocks != output_size // 1024 + 1
        or isinstance(ptx_version, bool)
        or not isinstance(ptx_version, int)
        or ptx_version != 86
    ):
        raise Ds9GpuSmokeError(f"CUDA kernel marker launch facts differ: {run_id}")
    return marker, pin


def _validate_gpu_fd_run_evidence(
    *, run_data: Mapping[str, Any], process_pid: int, policy: Any
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if policy != _gpu_fd_policy():
        failures.append("contract_policy_differs")

    observed_fds = run_data.get("nvidia_device_fds_observed")
    sample_count = run_data.get("gpu_fd_sample_count")
    samples = run_data.get("gpu_fd_samples")
    blocking_errors = run_data.get("gpu_fd_read_errors")
    terminal = run_data.get("gpu_fd_terminal_evidence")
    if (
        not isinstance(observed_fds, list)
        or not all(isinstance(item, str) for item in observed_fds)
        or observed_fds != sorted(set(observed_fds))
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or not isinstance(samples, list)
        or sample_count != len(samples)
        or not isinstance(blocking_errors, list)
    ):
        return False, [*failures, "top_level_structure_differs"]

    expected_root = f"/proc/{process_pid}/fd"
    sample_fields = {
        "sample_index",
        "pid",
        "proc_fd_root",
        "started_at_monotonic_ns",
        "completed_at_monotonic_ns",
        "observed_nvidia_device_fds",
        "read_errors",
    }
    error_fields = {
        "scope",
        "path",
        "errno",
        "errno_name",
        "exception_type",
        "observed_at_monotonic_ns",
    }
    aggregate_observed: set[str] = set()
    aggregate_errors: list[tuple[int, dict[str, Any]]] = []
    previous_completed_at: int | None = None
    for expected_index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict) or set(sample) != sample_fields:
            failures.append(f"sample_{expected_index}_field_set_differs")
            continue
        started_at = sample.get("started_at_monotonic_ns")
        completed_at = sample.get("completed_at_monotonic_ns")
        sample_observed = sample.get("observed_nvidia_device_fds")
        sample_errors = sample.get("read_errors")
        if (
            sample.get("sample_index") != expected_index
            or sample.get("pid") != process_pid
            or sample.get("proc_fd_root") != expected_root
            or isinstance(started_at, bool)
            or not isinstance(started_at, int)
            or started_at < 0
            or isinstance(completed_at, bool)
            or not isinstance(completed_at, int)
            or completed_at < started_at
            or (
                previous_completed_at is not None
                and started_at < previous_completed_at
            )
            or not isinstance(sample_observed, list)
            or not all(isinstance(item, str) for item in sample_observed)
            or sample_observed != sorted(set(sample_observed))
            or any(
                not item.startswith("/dev/nvidia") or "\x00" in item
                for item in sample_observed
            )
            or not isinstance(sample_errors, list)
        ):
            failures.append(f"sample_{expected_index}_structure_differs")
            continue
        previous_completed_at = completed_at
        aggregate_observed.update(sample_observed)
        for error_index, error_data in enumerate(sample_errors, start=1):
            if not isinstance(error_data, dict) or set(error_data) != error_fields:
                failures.append(
                    f"sample_{expected_index}_error_{error_index}_field_set_differs"
                )
                continue
            error_number = error_data.get("errno")
            error_at = error_data.get("observed_at_monotonic_ns")
            error_scope = error_data.get("scope")
            error_path = error_data.get("path")
            path_ok = (
                error_scope == "proc_fd_root" and error_path == expected_root
            ) or (
                error_scope == "fd_entry"
                and isinstance(error_path, str)
                and re.fullmatch(rf"{re.escape(expected_root)}/[0-9]+", error_path)
                is not None
            )
            if (
                isinstance(error_number, bool)
                or not isinstance(error_number, int)
                or error_data.get("errno_name")
                != errno.errorcode.get(error_number, "UNKNOWN")
                or not isinstance(error_data.get("exception_type"), str)
                or not error_data["exception_type"]
                or isinstance(error_at, bool)
                or not isinstance(error_at, int)
                or not (started_at <= error_at <= completed_at)
                or not path_ok
            ):
                failures.append(
                    f"sample_{expected_index}_error_{error_index}_structure_differs"
                )
                continue
            aggregate_errors.append((expected_index, error_data))

    if failures:
        return False, failures
    if observed_fds != sorted(aggregate_observed):
        failures.append("aggregate_observed_fd_set_differs")
    required_complete = _required_nvidia_fd_set_complete(aggregate_observed)
    if not required_complete:
        failures.append("required_fd_set_incomplete")

    accepted_terminal = False
    if terminal is None:
        expected_blocking_errors = [item[1] for item in aggregate_errors]
        if aggregate_errors:
            failures.append("read_error_not_accepted_as_terminal_teardown")
    else:
        terminal_fields = {
            "classification",
            "sample_index",
            "pid",
            "proc_fd_root",
            "root_error",
            "required_fds_observed_before_error",
            "prior_read_error_count",
            "error_observed_at_monotonic_ns",
            "exit_observed_at_monotonic_ns",
            "elapsed_to_exit_monotonic_ns",
            "exit_grace_monotonic_ns",
            "exit_returncode",
        }
        if not isinstance(terminal, dict) or set(terminal) != terminal_fields:
            failures.append("terminal_evidence_field_set_differs")
        elif len(aggregate_errors) != 1:
            failures.append("terminal_evidence_requires_exactly_one_read_error")
        else:
            terminal_sample_index, root_error = aggregate_errors[0]
            terminal_sample = samples[terminal_sample_index - 1]
            observed_before_error = {
                item
                for prior_sample in samples[: terminal_sample_index - 1]
                for item in prior_sample["observed_nvidia_device_fds"]
            }
            error_number = root_error.get("errno")
            exit_at = terminal.get("exit_observed_at_monotonic_ns")
            root_error_ok = (
                terminal_sample_index == len(samples)
                and terminal_sample["read_errors"] == [root_error]
                and root_error.get("scope") == "proc_fd_root"
                and root_error.get("path") == expected_root
                and error_number in GPU_FD_TERMINAL_ROOT_ERRNOS
                and root_error.get("exception_type")
                == GPU_FD_TERMINAL_EXCEPTION_TYPES.get(error_number)
                and _required_nvidia_fd_set_complete(observed_before_error)
            )
            exit_ok = (
                not isinstance(exit_at, bool)
                and isinstance(exit_at, int)
                and exit_at >= terminal_sample["completed_at_monotonic_ns"]
                and 0
                <= exit_at - root_error["observed_at_monotonic_ns"]
                <= GPU_FD_TERMINAL_EXIT_GRACE_NS
                and terminal.get("exit_returncode") == 0
                and run_data.get("returncode") == 0
            )
            if root_error_ok and exit_ok:
                expected_terminal = _terminal_fd_evidence(
                    sample=terminal_sample,
                    root_error=root_error,
                    observed_before_error=observed_before_error,
                    exit_observed_at_monotonic_ns=exit_at,
                    exit_returncode=0,
                )
                accepted_terminal = terminal == expected_terminal
            if not accepted_terminal:
                failures.append("terminal_evidence_semantics_differ")
        expected_blocking_errors = [] if accepted_terminal else [
            item[1] for item in aggregate_errors
        ]

    if blocking_errors != expected_blocking_errors:
        failures.append("blocking_read_error_set_differs")
    passed = not failures and required_complete and (
        not aggregate_errors or accepted_terminal
    )
    return passed, failures


def replay_raw_session(
    *,
    session_root: Path,
    contract: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    session = _inside_root(session_root, project_root)
    raw = session / "raw"
    if not raw.is_dir() or raw.is_symlink():
        raise Ds9GpuSmokeError("raw smoke root is not a regular directory")
    expected_raw_entries = set(RUN_IDS) | {"inner-run.json", "gpu-identity.log"}
    if {item.name for item in raw.iterdir()} != expected_raw_entries:
        raise Ds9GpuSmokeError("raw smoke root contains missing or extra artifacts")
    inner_path = raw / "inner-run.json"
    inner, inner_pin = _read_json(
        inner_path, project_root=project_root, immutable=False
    )
    if inner.get("schema_version") != INNER_SCHEMA_VERSION or inner.get("status") != "completed_for_host_replay":
        raise Ds9GpuSmokeError("inner run status/schema differs")
    expected_inner = {
        "campaign_nonce": contract["campaign_nonce"],
        "session_id": contract["session_id"],
        "resolved_image_id": contract["resolved_image_id"],
        "definition_sha256": contract["definition_sha256"],
        "parser_sha256": contract["parser"]["sha256"],
    }
    for key, value in expected_inner.items():
        if inner.get(key) != value:
            raise Ds9GpuSmokeError(f"inner run binding differs: {key}")
    if set(inner.get("runs", {})) != set(RUN_IDS):
        raise Ds9GpuSmokeError("inner run set differs")
    if contract.get("kernel_proof_policy") != _kernel_proof_policy():
        raise Ds9GpuSmokeError("kernel proof policy differs from the closed contract")
    expected_engine_sha = {
        str(profile): contract["inputs"]["models"][str(profile)]["engine"]["sha256"]
        for profile in PROFILES
    }
    if inner.get("engine_sha256_before") != expected_engine_sha or inner.get("engine_sha256_after") != expected_engine_sha:
        raise Ds9GpuSmokeError("inner engine SHA changed/differs")

    identity_path = raw / "gpu-identity.log"
    identity_content, identity_pin, _ = _read_bounded_file(
        identity_path,
        project_root=project_root,
        max_bytes=MAX_GPU_IDENTITY_BYTES,
        label="GPU identity log",
    )
    try:
        identity_text = identity_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Ds9GpuSmokeError("GPU identity log is not UTF-8") from exc
    identity = _parse_gpu_identity(identity_text)
    if identity["index"] != contract["gpu"]["index"] or identity["uuid"] != contract["gpu"]["uuid"]:
        raise Ds9GpuSmokeError("live GPU identity differs from contract")

    run_evidence: dict[str, Any] = {}
    parsed_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    engine_profile_pass: dict[int, bool] = {}
    cuda_logs_safe = True
    gpu_fd_all_runs = True
    kernel_markers: dict[str, dict[str, Any]] = {}
    for run_id in RUN_IDS:
        run_data = inner["runs"][run_id]
        expected_argv = ["deepstream-app", "-c", f"/contract/deepstream-{run_id}.txt"]
        expected_environment = _kernel_proof_environment(
            contract["campaign_nonce"], run_id
        )
        expected_marker_relative = (
            f"{run_id}/{KERNEL_PROOF_MARKER_NAME}"
            if run_id.endswith("-cuda")
            else None
        )
        process_pid = run_data.get("pid") if isinstance(run_data, dict) else None
        expected_run_data_fields = {
            "argv",
            "returncode",
            "pid",
            "log",
            "kitti",
            "nvidia_device_fds_observed",
            "gpu_fd_sample_count",
            "gpu_fd_samples",
            "gpu_fd_read_errors",
            "gpu_fd_terminal_evidence",
            "kernel_marker",
            "kernel_proof_environment",
        }
        if (
            not isinstance(run_data, dict)
            or set(run_data) != expected_run_data_fields
            or run_data.get("argv") != expected_argv
            or run_data.get("returncode") != 0
            or isinstance(process_pid, bool)
            or not isinstance(process_pid, int)
            or process_pid <= 0
            or run_data.get("kernel_proof_environment") != expected_environment
            or run_data.get("kernel_marker") != expected_marker_relative
            or run_data.get("log") != f"{run_id}/deepstream.log"
            or run_data.get("kitti") != f"{run_id}/kitti"
        ):
            raise Ds9GpuSmokeError(f"inner command/exit differs: {run_id}")
        run_root = raw / run_id
        if not run_root.is_dir() or run_root.is_symlink():
            raise Ds9GpuSmokeError(f"raw run directory is invalid: {run_id}")
        expected_run_entries = {"deepstream.log", "kitti"}
        if run_id.endswith("-cuda"):
            expected_run_entries.add(KERNEL_PROOF_MARKER_NAME)
        if {item.name for item in run_root.iterdir()} != expected_run_entries:
            raise Ds9GpuSmokeError(
                f"raw run contains missing/extra artifacts: {run_id}"
            )
        observed_fds = run_data.get("nvidia_device_fds_observed")
        fd_samples = run_data.get("gpu_fd_sample_count")
        fd_sample_evidence = run_data.get("gpu_fd_samples")
        fd_errors = run_data.get("gpu_fd_read_errors")
        fd_terminal_evidence = run_data.get("gpu_fd_terminal_evidence")
        fd_ok, fd_failures = _validate_gpu_fd_run_evidence(
            run_data=run_data,
            process_pid=process_pid,
            policy=contract.get("gpu_fd_policy"),
        )
        gpu_fd_all_runs = gpu_fd_all_runs and fd_ok
        log_path = run_root / "deepstream.log"
        log_content, log_pin, _ = _read_bounded_file(
            log_path,
            project_root=project_root,
            max_bytes=MAX_RAW_LOG_BYTES,
            label=f"DeepStream log {run_id}",
        )
        try:
            decoded_log = log_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Ds9GpuSmokeError(
                f"DeepStream log is not UTF-8: {run_id}"
            ) from exc
        log_text = ANSI_RE.sub("", decoded_log)
        positives = {
            pattern: len(re.findall(pattern, log_text, flags=re.IGNORECASE | re.DOTALL))
            for pattern in POSITIVE_ENGINE_PATTERNS
        }
        forbidden = {
            pattern: len(re.findall(pattern, log_text, flags=re.IGNORECASE | re.DOTALL))
            for pattern in ENGINE_FALLBACK_PATTERNS
        }
        cuda_errors = {
            pattern: len(re.findall(pattern, log_text, flags=re.IGNORECASE))
            for pattern in CUDA_ERROR_PATTERNS
        }
        profile = int(run_id.split("-", 1)[0])
        expected_engine_path = f"/models/{profile}.engine"
        path_mentions = log_text.count(expected_engine_path)
        engine_ok = (
            all(count >= 1 for count in positives.values())
            and all(count == 0 for count in forbidden.values())
            and all(count == 0 for count in cuda_errors.values())
            and path_mentions >= 2
            and fd_ok
        )
        engine_profile_pass[profile] = engine_profile_pass.get(profile, True) and engine_ok
        if run_id.endswith("-cuda") and any(cuda_errors.values()):
            cuda_logs_safe = False
        parsed, kitti_pins = _load_kitti_dir(
            run_root / "kitti", project_root=project_root
        )
        parsed_runs[run_id] = parsed
        marker_evidence = None
        if run_id.endswith("-cuda"):
            marker_payload, marker_pin = _validate_kernel_marker(
                run_root / KERNEL_PROOF_MARKER_NAME,
                run_id=run_id,
                expected_pid=process_pid,
                contract=contract,
                project_root=project_root,
            )
            marker_evidence = {"payload": marker_payload, "pin": marker_pin}
            kernel_markers[run_id] = marker_evidence
        run_evidence[run_id] = {
            "argv": expected_argv,
            "pid": process_pid,
            "log": log_pin,
            "engine_log": {
                "positive_pattern_counts": positives,
                "forbidden_pattern_counts": forbidden,
                "cuda_error_pattern_counts": cuda_errors,
                "expected_engine_path_mentions": path_mentions,
                "status": "pass" if engine_ok else "fail",
            },
            "kitti_files": kitti_pins,
            "kitti_tree_sha256": canonical_sha256(kitti_pins),
            "frame_count": len(parsed),
            "detection_count": sum(len(value) for value in parsed.values()),
            "gpu_fd": {
                "status": "pass" if fd_ok else "fail",
                "sample_count": fd_samples,
                "samples": fd_sample_evidence,
                "observed": observed_fds,
                "read_errors": fd_errors,
                "terminal_teardown": fd_terminal_evidence,
                "validation_failures": fd_failures,
                "required_patterns": list(REQUIRED_NVIDIA_FD_PATTERNS),
            },
            "cuda_kernel_marker": marker_evidence,
        }
    parity = {
        str(profile): _parity(
            parsed_runs[f"{profile}-cpu"], parsed_runs[f"{profile}-cuda"]
        )
        for profile in PROFILES
    }
    cuda_detections = sum(
        run_evidence[f"{profile}-cuda"]["detection_count"] for profile in PROFILES
    )
    kernel_pass = identity["compute_capability"] == "8.6" and set(
        kernel_markers
    ) == {"640-cuda", "960-cuda"}
    checks = {
        "cuda_parser_kernel_launch_sm86": "pass" if kernel_pass else "fail",
        "deepstream_640_engine_deserialize_no_fallback": (
            "pass" if engine_profile_pass.get(640) else "fail"
        ),
        "deepstream_960_engine_deserialize_no_fallback": (
            "pass" if engine_profile_pass.get(960) else "fail"
        ),
        "cpu_cuda_parser_parity_640": parity["640"]["status"],
        "cpu_cuda_parser_parity_960": parity["960"]["status"],
    }
    if {item.name for item in raw.iterdir()} != expected_raw_entries:
        raise Ds9GpuSmokeError("raw smoke root changed while replaying")
    raw_artifacts = {
        "inner_run": inner_pin,
        "gpu_identity_log": identity_pin,
        "runs": run_evidence,
    }
    metrics = {
        "gpu_identity": identity,
        "cuda_parser": {
            "static_sm86_cubin": True,
            "cuda_output_detections": cuda_detections,
            "cuda_logs_safe": cuda_logs_safe,
            "gpu_fd_all_deepstream_runs": gpu_fd_all_runs,
            "kernel_proof_method": "source_pinned_immediate_post_launch_marker",
            "ptx_jit_disabled_for_kernel_proof": True,
            "valid_kernel_marker_count": len(kernel_markers),
            "required_kernel_marker_runs": ["640-cuda", "960-cuda"],
            "kernel_markers": kernel_markers,
            "generic_gpu_fds_used_for_kernel_proof": False,
            "detections_used_for_kernel_proof": False,
            "parser_config_used_for_kernel_proof": False,
            "cuda_log_used_for_kernel_proof": False,
        },
        "engine_profiles": {
            str(profile): {"status": "pass" if engine_profile_pass.get(profile) else "fail"}
            for profile in PROFILES
        },
        "parity": parity,
    }
    return checks, metrics, raw_artifacts


def execute(
    *,
    plan: Mapping[str, Any],
    authorization_path: Path,
    static_candidate_receipt: Path,
    parser_build_receipt: Path,
    reentry_evidence: Path,
    video: Path,
    project_root: Path = PROJECT_ROOT,
    guarded_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if plan.get("status") != "ready_for_authorized_execution" or plan.get("blockers") != []:
        raise Ds9GpuSmokeError("smoke plan is not ready for authorized execution")
    session = _inside_root(Path(plan["session_root"]), project_root)
    contract_path = session / "inputs/probe-contract.json"
    contract, contract_pin = _read_json(
        contract_path, project_root=project_root, immutable=True
    )
    if contract_pin != plan.get("contract") or contract.get("definition_sha256") != plan.get("definition_sha256"):
        raise Ds9GpuSmokeError("live smoke contract differs from plan")
    definition = dict(contract)
    definition.pop("definition_sha256", None)
    if canonical_sha256(definition) != contract["definition_sha256"] or definition != plan["definition"]:
        raise Ds9GpuSmokeError("smoke definition changed after authorization")
    auth, auth_pin = validate_authorization(
        authorization_path,
        project_root=project_root,
        expected_nonce=plan["campaign_nonce"],
        expected_session_root=plan["session_root"],
        expected_image_id=plan["resolved_image_id"],
        expected_definition_sha256=plan["definition_sha256"],
        expected_static_receipt_sha256=contract["static_candidate_receipt"]["sha256"],
        expected_parser_build_receipt_sha256=contract[
            "parser_production_build_receipt"
        ]["sha256"],
        expected_gpu_index=plan["gpu"]["index"],
        expected_gpu_uuid=plan["gpu"]["uuid"],
    )
    if plan.get("authorization", {}).get("pin") != auth_pin:
        raise Ds9GpuSmokeError("authorization changed after plan")

    # These checks intentionally precede the single-use claim.  Invalid/stale
    # external evidence must not consume an otherwise valid operator nonce.
    from validation.ds9_runtime_compatibility import require_static_candidate_compatibility
    from validation.gpu_reentry_evidence import require_reentry_evidence

    reentry = require_reentry_evidence(
        _inside_root(reentry_evidence, project_root), project_root=project_root
    )
    static = require_static_candidate_compatibility(
        _inside_root(static_candidate_receipt, project_root),
        project_root=project_root,
        requested_image=plan["requested_image"],
        resolved_image_id=plan["resolved_image_id"],
    )
    parser_build, parser_build_pin = validate_production_receipt(
        _inside_root(parser_build_receipt, project_root),
        project_root=project_root,
        resolved_image_id=plan["resolved_image_id"],
        parser_sha256=contract["parser"]["sha256"],
    )
    if parser_build_pin != contract["parser_production_build_receipt"]:
        raise Ds9GpuSmokeError("parser production build receipt changed")

    claim_path = session / "session-claim.json"
    raw = session / "raw"
    evidence_path = session / "gpu-smoke-evidence.json"
    if raw.exists() or evidence_path.exists():
        raise Ds9GpuSmokeError("single-use smoke session already has execution output")
    claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claimed_at_utc": utc_now(),
        "campaign_nonce": plan["campaign_nonce"],
        "session_id": contract["session_id"],
        "authorized_session_root": plan["session_root"],
        "authorization": auth_pin,
        "contract": contract_pin,
        "single_use": True,
    }
    _exclusive_json(claim_path, claim)
    claim_pin = file_pin(claim_path, project_root=project_root)
    raw.mkdir(parents=True, exist_ok=False)
    command, container_name = build_docker_command(
        plan=plan, project_root=project_root, video=video
    )
    if guarded_runner is None:
        from validation.gpu_guarded_process import run_guarded_docker

        guarded_runner = run_guarded_docker
    guard_root = session / "guard"
    guard_report = guarded_runner(
        command,
        project_root=project_root,
        artifact_root=guard_root,
        log_path=guard_root / "container-stdout.log",
        container_name=container_name,
        image=plan["requested_image"],
        gpu_index=plan["gpu"]["index"],
        reentry_evidence_path=_inside_root(reentry_evidence, project_root),
        ds9_compatibility_receipt_path=_inside_root(
            static_candidate_receipt, project_root
        ),
        compatibility_mode="static_candidate_smoke",
    )
    if guard_report.get("status") != "complete":
        raise Ds9GpuSmokeError("GPU guard did not complete safely")
    checks, metrics, raw_artifacts = replay_raw_session(
        session_root=session, contract=contract, project_root=project_root
    )
    created = _parse_time(utc_now(), "created_at_utc")
    status = "pass" if all(value == "pass" for value in checks.values()) else "fail"
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (created + MAX_EVIDENCE_AGE)
        .isoformat()
        .replace("+00:00", "Z"),
        "resolved_image_id": plan["resolved_image_id"],
        "runtime_control_manifest_sha256": contract[
            "runtime_control_manifest_sha256"
        ],
        "campaign_nonce": plan["campaign_nonce"],
        "session_id": contract["session_id"],
        "session_root": plan["session_root"],
        "checks": checks,
        "metrics": metrics,
        "bindings": {
            "operator_authorization": auth_pin,
            "session_claim": claim_pin,
            "probe_contract": contract_pin,
            "static_candidate_receipt": contract["static_candidate_receipt"],
            "parser_production_build_receipt": parser_build_pin,
            "reentry_evidence": file_pin(
                _inside_root(reentry_evidence, project_root),
                project_root=project_root,
            ),
        },
        "guard": {
            "report": file_pin(guard_root / "gpu-guard-report.json", project_root=project_root),
            "artifact_receipt": file_pin(
                guard_root / "gpu-guard-artifact-receipt.json",
                project_root=project_root,
            ),
            "container_stdout": file_pin(
                guard_root / "container-stdout.log", project_root=project_root
            ),
            "requested_command": command,
            "requested_command_sha256": canonical_sha256(command),
        },
        "raw_artifacts": raw_artifacts,
    }
    _exclusive_json(evidence_path, evidence)
    return evidence


def validate_production_evidence(
    payload: Mapping[str, Any],
    *,
    evidence_path: Path,
    project_root: Path,
    resolved_image_id: str,
    runtime_control_manifest_sha256: str,
    expected_parser_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ds9GpuSmokeError("GPU smoke evidence schema differs")
    if payload.get("status") != "pass":
        raise Ds9GpuSmokeError("GPU smoke evidence is not passing")
    if payload.get("resolved_image_id") != resolved_image_id:
        raise Ds9GpuSmokeError("GPU smoke evidence image differs")
    if payload.get("runtime_control_manifest_sha256") != runtime_control_manifest_sha256:
        raise Ds9GpuSmokeError("GPU smoke control-manifest binding differs")
    created = _parse_time(payload.get("created_at_utc"), "smoke created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "smoke expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= created or expires - created > MAX_EVIDENCE_AGE or not (created <= current < expires):
        raise Ds9GpuSmokeError("GPU smoke evidence is stale or overlong")
    nonce = payload.get("campaign_nonce")
    if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        raise Ds9GpuSmokeError("GPU smoke nonce is invalid")
    expected_session = f"ds9-gpu-smoke-{nonce}"
    expected_root = (SESSION_PREFIX / nonce).as_posix()
    if payload.get("session_id") != expected_session or payload.get("session_root") != expected_root:
        raise Ds9GpuSmokeError("GPU smoke nonce/session binding differs")
    session = _inside_root(Path(expected_root), project_root)
    if _inside_root(evidence_path, project_root) != session / "gpu-smoke-evidence.json":
        raise Ds9GpuSmokeError("GPU smoke evidence path is not nonce-isolated")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "operator_authorization",
        "session_claim",
        "probe_contract",
        "static_candidate_receipt",
        "parser_production_build_receipt",
        "reentry_evidence",
    }:
        raise Ds9GpuSmokeError("GPU smoke binding set differs")
    for key in (
        "operator_authorization",
        "session_claim",
        "probe_contract",
        "static_candidate_receipt",
        "parser_production_build_receipt",
        "reentry_evidence",
    ):
        _pin_matches(bindings[key], project_root=project_root)
    contract, contract_pin = _read_json(
        project_root / bindings["probe_contract"]["path"],
        project_root=project_root,
        immutable=True,
    )
    if contract_pin != bindings["probe_contract"]:
        raise Ds9GpuSmokeError("GPU smoke contract pin differs")
    definition = validate_contract_semantics(contract, project_root=project_root)
    definition_sha = contract["definition_sha256"]
    if contract.get("resolved_image_id") != resolved_image_id:
        raise Ds9GpuSmokeError("GPU smoke contract image differs")
    if contract.get("parser", {}).get("sha256") != expected_parser_sha256:
        raise Ds9GpuSmokeError("GPU smoke parser SHA differs")
    if contract.get("runtime_control_manifest_sha256") != runtime_control_manifest_sha256:
        raise Ds9GpuSmokeError("GPU smoke contract controls differ")
    if contract.get("required_checks") != list(REQUIRED_CHECKS):
        raise Ds9GpuSmokeError("GPU smoke contract check set differs")
    auth, auth_pin = validate_authorization(
        project_root / bindings["operator_authorization"]["path"],
        project_root=project_root,
        expected_nonce=nonce,
        expected_session_root=expected_root,
        expected_image_id=resolved_image_id,
        expected_definition_sha256=definition_sha,
        expected_static_receipt_sha256=bindings["static_candidate_receipt"]["sha256"],
        expected_parser_build_receipt_sha256=bindings[
            "parser_production_build_receipt"
        ]["sha256"],
        expected_gpu_index=contract["gpu"]["index"],
        expected_gpu_uuid=contract["gpu"]["uuid"],
        now=current,
    )
    if auth_pin != bindings["operator_authorization"]:
        raise Ds9GpuSmokeError("GPU smoke authorization pin differs")
    claim, claim_pin = _read_json(
        project_root / bindings["session_claim"]["path"],
        project_root=project_root,
        immutable=True,
    )
    expected_claim = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "claimed_at_utc": claim.get("claimed_at_utc"),
        "campaign_nonce": nonce,
        "session_id": expected_session,
        "authorized_session_root": expected_root,
        "authorization": auth_pin,
        "contract": contract_pin,
        "single_use": True,
    }
    _parse_time(claim.get("claimed_at_utc"), "claim claimed_at_utc")
    if claim != expected_claim or claim_pin != bindings["session_claim"]:
        raise Ds9GpuSmokeError("GPU smoke session claim differs")
    parser_build, parser_pin = validate_production_receipt(
        project_root / bindings["parser_production_build_receipt"]["path"],
        project_root=project_root,
        resolved_image_id=resolved_image_id,
        parser_sha256=expected_parser_sha256,
        now=current,
    )
    if parser_pin != bindings["parser_production_build_receipt"]:
        raise Ds9GpuSmokeError("GPU smoke parser build pin differs")
    static_payload, static_pin = _offline_static_candidate(
        project_root / bindings["static_candidate_receipt"]["path"],
        project_root=project_root,
    )
    if static_pin != bindings["static_candidate_receipt"]:
        raise Ds9GpuSmokeError("GPU smoke static candidate pin differs")
    if static_payload["image"]["resolved_image_id"] != resolved_image_id:
        raise Ds9GpuSmokeError("GPU smoke static candidate image differs")
    checks, metrics, raw_artifacts = replay_raw_session(
        session_root=session, contract=contract, project_root=project_root
    )
    if checks != payload.get("checks") or any(value != "pass" for value in checks.values()):
        raise Ds9GpuSmokeError("GPU smoke checks differ from raw replay")
    if metrics != payload.get("metrics") or raw_artifacts != payload.get("raw_artifacts"):
        raise Ds9GpuSmokeError("GPU smoke metrics/artifact index differs from raw replay")
    guard = payload.get("guard")
    if not isinstance(guard, dict) or set(guard) != {
        "report",
        "artifact_receipt",
        "container_stdout",
        "requested_command",
        "requested_command_sha256",
    }:
        raise Ds9GpuSmokeError("GPU guard evidence set differs")
    for key in ("report", "artifact_receipt", "container_stdout"):
        _pin_matches(guard[key], project_root=project_root)
    expected_command, expected_container_name = build_docker_command(
        plan={
            "campaign_nonce": nonce,
            "session_root": expected_root,
            "requested_image": contract["requested_image"],
            "resolved_image_id": resolved_image_id,
            "gpu": contract["gpu"],
            "container_process_identity": contract["container_process_identity"],
        },
        project_root=project_root,
        video=project_root / contract["inputs"]["video"]["path"],
    )
    if (
        guard["requested_command"] != expected_command
        or canonical_sha256(expected_command) != guard["requested_command_sha256"]
    ):
        raise Ds9GpuSmokeError("GPU guard command pin differs")
    guard_report, _ = _read_json(
        project_root / guard["report"]["path"], project_root=project_root, immutable=False
    )
    expected_guard_fields = {
        "status": "complete",
        "compatibility_mode": "static_candidate_smoke",
        "requested_image": contract["requested_image"],
        "image": contract["requested_image"],
        "resolved_image_id": resolved_image_id,
        "container_name": expected_container_name,
        "requested_command": expected_command,
        "command": expected_command,
        "failure_reasons": [],
    }
    for key, value in expected_guard_fields.items():
        if guard_report.get(key) != value:
            raise Ds9GpuSmokeError(f"GPU guard report binding differs: {key}")
    process = guard_report.get("process")
    if not isinstance(process, dict) or any(
        (
            process.get("started") is not True,
            process.get("command") != expected_command,
            process.get("container_image_id") != resolved_image_id,
            process.get("exit_code") != 0,
        )
    ):
        raise Ds9GpuSmokeError("GPU guard process binding differs")
    ds9_guard = guard_report.get("ds9_runtime_compatibility")
    if (
        not isinstance(ds9_guard, dict)
        or ds9_guard.get("status")
        != "static_candidate_ready_for_guarded_gpu_smoke"
        or ds9_guard.get("production_ready") is not False
        or ds9_guard.get("resolved_image_id") != resolved_image_id
        or ds9_guard.get("receipt") != bindings["static_candidate_receipt"]
    ):
        raise Ds9GpuSmokeError("GPU guard static-candidate receipt binding differs")
    guard_receipt, guard_receipt_pin = _read_json(
        project_root / guard["artifact_receipt"]["path"],
        project_root=project_root,
        immutable=True,
    )
    if guard_receipt_pin != guard["artifact_receipt"]:
        raise Ds9GpuSmokeError("GPU guard artifact receipt pin differs")
    expected_receipt_fields = {
        "schema_version": "deepsafe.gpu-guard-artifact-receipt/v1",
        "guard_status": "complete",
        "requested_image": contract["requested_image"],
        "compatibility_mode": "static_candidate_smoke",
        "resolved_image_id": resolved_image_id,
        "requested_command": expected_command,
        "executed_command": expected_command,
        "running_container": {
            "name": expected_container_name,
            "image_id": resolved_image_id,
        },
        "ds9_runtime_compatibility": ds9_guard,
    }
    for key, value in expected_receipt_fields.items():
        if guard_receipt.get(key) != value:
            raise Ds9GpuSmokeError(f"GPU guard artifact receipt differs: {key}")
    if (
        guard_report.get("artifact_receipt") != guard["artifact_receipt"]
        or guard_receipt.get("timeline") != guard_report.get("timeline")
        or guard_receipt.get("reentry_evidence")
        != guard_report.get("reentry_evidence")
    ):
        raise Ds9GpuSmokeError("GPU guard report/receipt cross-binding differs")
    receipt_artifacts = guard_receipt.get("artifacts")
    expected_artifact_names = {
        "preflight",
        "gpu_csv",
        "platform_thermal_csv",
        "deepstream_log",
    }
    if not isinstance(receipt_artifacts, dict) or set(receipt_artifacts) != expected_artifact_names:
        raise Ds9GpuSmokeError("GPU guard receipt artifact set differs")
    for name, expected_pin in receipt_artifacts.items():
        if not isinstance(expected_pin, dict) or set(expected_pin) != {
            "path",
            "bytes",
            "sha256",
            "allow_empty",
        }:
            raise Ds9GpuSmokeError(f"GPU guard artifact pin fields differ: {name}")
        expected_allow_empty = name == "deepstream_log"
        if expected_pin["allow_empty"] is not expected_allow_empty:
            raise Ds9GpuSmokeError(f"GPU guard empty-artifact policy differs: {name}")
        live_pin = file_pin(
            project_root / expected_pin["path"], project_root=project_root
        )
        if live_pin != {
            key: expected_pin[key] for key in ("path", "bytes", "sha256")
        }:
            raise Ds9GpuSmokeError(f"GPU guard raw artifact changed: {name}")
        report_path = guard_report.get("artifacts", {}).get(name)
        if isinstance(report_path, str) and report_path != expected_pin["path"]:
            raise Ds9GpuSmokeError(f"GPU guard report artifact path differs: {name}")
    safety = guard_receipt.get("safety_event")
    if not isinstance(safety, dict) or safety.get("present") is not False:
        raise Ds9GpuSmokeError("passing GPU smoke cannot contain a safety event")
    return {
        "status": "pass",
        "checks": checks,
        "campaign_nonce": nonce,
        "session_id": expected_session,
        "gpu_identity": metrics["gpu_identity"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--inner-output", type=Path)
    parser.add_argument("--session-root", type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--static-candidate-receipt", type=Path, default=DEFAULT_STATIC_CANDIDATE)
    parser.add_argument("--parser-build-receipt", type=Path, default=DEFAULT_PARSER_BUILD)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--reentry-evidence", type=Path, default=DEFAULT_REENTRY)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.inside_container:
            if args.contract is None or args.inner_output is None:
                raise Ds9GpuSmokeError("container worker requires contract/output")
            result = _inside_container(args.contract, args.inner_output)
        else:
            if args.session_root is None or args.gpu_uuid is None:
                raise Ds9GpuSmokeError("--session-root and --gpu-uuid are required")
            root = args.project_root.resolve()
            plan = build_plan(
                session_root=args.session_root,
                image=args.image,
                gpu_index=args.gpu_index,
                gpu_uuid=args.gpu_uuid,
                static_candidate_receipt=args.static_candidate_receipt,
                parser_build_receipt=args.parser_build_receipt,
                authorization=args.authorization,
                video=args.video,
                project_root=root,
            )
            if not args.execute:
                result = plan
            else:
                if args.authorization is None:
                    raise Ds9GpuSmokeError("--execute requires --authorization")
                result = execute(
                    plan=plan,
                    authorization_path=args.authorization,
                    static_candidate_receipt=args.static_candidate_receipt,
                    parser_build_receipt=args.parser_build_receipt,
                    reentry_evidence=args.reentry_evidence,
                    video=args.video,
                    project_root=root,
                )
    except (OSError, subprocess.SubprocessError, Ds9GpuSmokeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if isinstance(result, dict) and result.get("status") == "fail":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
