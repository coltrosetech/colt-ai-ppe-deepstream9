#!/usr/bin/env python3
"""Fail-closed DeepStream 9 runtime compatibility release gate.

The default mode is deliberately inert: it writes a pending report without
calling Docker or touching a GPU.  ``--execute-static-probe`` is the only mode
that may inspect an already-local image and run a network-isolated, GPU-free
container probe.  Static success is still not production approval; a separate
immutable GPU smoke/parity artifact must close every required GPU check.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "deepsafe.ds9-runtime-compatibility-receipt/v1"
PENDING_SCHEMA_VERSION = "deepsafe.ds9-runtime-compatibility-pending/v1"
STATIC_PROBE_SCHEMA_VERSION = "deepsafe.ds9-static-container-probe/v1"
GPU_SMOKE_SCHEMA_VERSION = "deepsafe.ds9-gpu-smoke-evidence/v1"
CONTROL_MANIFEST_SCHEMA_VERSION = "deepsafe.runtime-control-manifest/v1"
BUILD_LINEAGE_SCHEMA_VERSION = "deepsafe.deepstream-build-lineage/v3"

DEFAULT_RECEIPT = Path(
    "validation/results/ds9-runtime-compatibility/current/receipt.json"
)
DEFAULT_OUTPUT_DIR = Path("validation/results/ds9-runtime-compatibility/current")
RUNTIME_CONTROL_MANIFEST = Path("deepstream/runtime-control-manifest.json")
PARSER_LIBRARY = Path(
    "/opt/deepsafe/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/"
    "libnvdsinfer_custom_impl_Yolo.so"
)
BUILD_LINEAGE_MANIFEST = Path("/opt/deepsafe/build-lineage.json")
CONTAINER_CONTROL_MANIFEST = Path("/opt/deepsafe/runtime-control-manifest.json")
CONTAINER_CONTROLLER = Path("/app/validation/ds9_runtime_compatibility.py")
CONTAINER_GPU_SMOKE_HARNESS = Path("/app/validation/ds9_gpu_smoke.py")
CONTAINER_DOCKERIGNORE_POLICY = Path("/opt/deepsafe/dockerignore.policy")

DEEPSTREAM_BASE_TAG = "nvcr.io/nvidia/deepstream:9.0-triton-multiarch"
DEEPSTREAM_VERSION = "9.0.0"
CUDA_VERSION = "13.1"
TENSORRT_VERSION = "10.14.1.48"
GSTREAMER_VERSION = "1.24.2"
NVIDIA_DRIVER_VERSION = "590.48.01"
DEEPSTREAM_YOLO_REPOSITORY = "https://github.com/marcoslucianops/DeepStream-Yolo.git"
DEEPSTREAM_YOLO_COMMIT = "2894babce8e75c49115dbe0c7b516289ed853565"
DEEPSTREAM_YOLO_TREE = "1740cc4bc7e925f30e4eea0160064bfde729f8d8"
DEEPSTREAM_YOLO_PATCH_SHA256 = "dd85619bf62da249d17d99e967caf53de96367a0f139286c52a86f5e67b7623e"
DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256 = "a63299206550f1f8dd413cb9328352db304cdc020a47451170fd4c7eda0adf4d"
DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256 = "642e2875d67c3528c7ea301bcd1e973ea31fb019865f2aa116122583e9a765e3"
DEEPSTREAM_YOLO_PATCHED_TREE = "753acbd2995f9f8c0b791f6152f0793baa11b71a"
DEEPSTREAM_YOLO_PARSER_BUILD_MAKEFILE_SHA256 = (
    "fd2c03b810b8dae9d9d3a60b503616bbf6ed67a6f614843dd6a29f7f87ff8ad0"
)
DEEPSTREAM_YOLO_PARSER_CUDA_CUBIN_ARCHITECTURE = "sm_86"
DEEPSTREAM_YOLO_PARSER_CUDA_PTX_ARCHITECTURE = "compute_86"
DEEPSTREAM_YOLO_PARSER_CUDA_GENCODE_FLAGS = (
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_86,code=compute_86",
)
DEEPSTREAM_YOLO_PARSER_BUILD_COMMAND_SHA256 = (
    "e244df2d9c424fe7d027d62205ff21c820b58eb2cc00aa61df2d32cbfe329ac1"
)
DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_PATH = "/usr/bin/x86_64-linux-gnu-strip"
DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_SHA256 = (
    "4dad0d12aa5d6a49b117b4551b897175ad5b43b9525e8f9efd661133a1c8ea0d"
)
DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_VERSION = (
    "GNU strip (GNU Binutils for Ubuntu) 2.42"
)
DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND = (
    "/usr/bin/x86_64-linux-gnu-strip --strip-unneeded "
    "nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
)
DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND_SHA256 = (
    "5e19627d403e984d9e349f8c81332f7280a1ac8477d56056c9b27be63aeef7ca"
)
DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS = (".symtab", ".strtab")
DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS = (".dynsym", ".dynstr")
CUDA_KERNEL_PROOF_SCHEMA_VERSION = "deepsafe.ds9-cuda-kernel-proof/v1"
EXPECTED_ARCHITECTURE = "amd64"
EXPECTED_OS = "linux"
EXPECTED_PARSER_CUDA_ENTRY_INDEXES = (1, 2, 3, 4)
EXPECTED_NVINFER_PATH = (
    "/opt/nvidia/deepstream/deepstream-9.0/lib/gst-plugins/"
    "libnvdsgst_infer.so"
)
DEEPSTREAM_VERSION_MANIFEST = Path(
    "/opt/nvidia/deepstream/deepstream-9.0/version"
)
# The DeepStream container installs its SDK tree outside dpkg ownership.  Pin
# the exact NVIDIA-supplied DS9 nvinfer binary instead of pretending that a
# package database owns it.  This digest was measured from the immutable base
# image named by ``DEEPSTREAM_BASE_TAG``/its required base digest lineage.
EXPECTED_NVINFER_SHA256 = (
    "3f72d4352d178c9acfccc5fc19d9b250dd4ac9d28e3dc9b74f8623cb7db854c1"
)
EXPECTED_NVINFER_BYTES = 318552
EXPECTED_NVINFER_NEEDED = (
    "libnvds_infer.so",
    "libnvds_meta.so",
    "libnvdsgst_meta.so",
    "libnvdsgst_helper.so",
    "libnvdsgst_customhelper.so",
    "libnvbufsurftransform.so",
    "libnvbufsurface.so",
    "libyaml-cpp.so.0.8",
    "libcudart.so.13",
    "libcuda.so.1",
    "libgstreamer-1.0.so.0",
    "libgstbase-1.0.so.0",
    "libglib-2.0.so.0",
    "libgobject-2.0.so.0",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
)
NVINFER_DESCRIPTOR_SYMBOLS = (
    "gst_plugin_nvdsgst_infer_get_desc",
    "gst_plugin_nvdsgst_infer_register",
)
NVINFER_STATIC_VERIFICATION_SCOPE = "static_binary_metadata_only"
REQUIRED_SYMBOLS = (
    "NvDsInferParseYoloCuda",
    "NvDsInferYoloCudaEngineGet",
)
REQUIRED_GPU_SMOKE_CHECKS = (
    "cuda_parser_kernel_launch_sm86",
    "deepstream_640_engine_deserialize_no_fallback",
    "deepstream_960_engine_deserialize_no_fallback",
    "cpu_cuda_parser_parity_640",
    "cpu_cuda_parser_parity_960",
)
MAX_RECEIPT_AGE = timedelta(hours=24)
MAX_COMPATIBILITY_JSON_BYTES = 4 * 1024 * 1024
MAX_NM_ORIGINAL_STDOUT_BYTES = 4 * 1024 * 1024
MAX_NM_PROJECTED_STDOUT_BYTES = 4096
MAX_NM_PROJECTED_MATCHES = 32
NM_PROJECTION_POLICY = "exact_required_defined_export_lines/v1"
MAX_CUOBJDUMP_LIST_BYTES = 16 * 1024
MAX_READELF_SECTION_BYTES = 64 * 1024
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

# This exact set is projected into a manifest that is baked into the immutable
# image.  A receipt cannot therefore coherently re-pin changed launch code
# without also changing the image ID and its build-lineage labels.
RUNTIME_CONTROL_PATHS: dict[str, str] = {
    "docker_build_context_policy": ".dockerignore",
    "ds9_runtime_compatibility": "validation/ds9_runtime_compatibility.py",
    "ds9_runtime_compatibility_schema": (
        "validation/schemas/ds9-runtime-compatibility-v1.schema.json"
    ),
    "deepstream_image_build": "deepstream/Dockerfile",
    "gpu_guard": "validation/gpu_guarded_process.py",
    "ds9_parser_bootstrap": "validation/ds9_parser_bootstrap.py",
    "ds9_parser_bootstrap_schema": (
        "validation/schemas/ds9-parser-bootstrap-v1.schema.json"
    ),
    "ds9_gpu_smoke": "validation/ds9_gpu_smoke.py",
    "ds9_gpu_smoke_authorization_schema": (
        "validation/schemas/ds9-gpu-smoke-authorization-v1.schema.json"
    ),
    "ds9_gpu_smoke_evidence_schema": (
        "validation/schemas/ds9-gpu-smoke-evidence-v1.schema.json"
    ),
    "ds9_cuda_kernel_proof_schema": (
        "validation/schemas/ds9-cuda-kernel-proof-v1.schema.json"
    ),
    "ds9_cuda_kernel_source_patch": (
        "deepstream/patches/deepstream-yolo-ds9-cuda-kernel-proof.patch"
    ),
    "scene_benchmark_runner": "validation/scene_benchmark/run_matrix.py",
    "caviar_runner": "validation/run_caviar.py",
    "caviar_batch_runner": "validation/run_caviar_batch.py",
    "loaf_runner": "validation/run_loaf.py",
    "loaf_batch_runner": "validation/run_loaf_batch.py",
    "person_quality_policy": "validation/person_quality_policy.py",
}

LABELS = {
    "schema": "com.deepsafe.build-lineage.schema",
    "base_ref": "com.deepsafe.deepstream.base-ref",
    "base_digest": "com.deepsafe.deepstream.base-digest",
    "deepstream": "com.deepsafe.deepstream.version",
    "cuda": "com.deepsafe.cuda.version",
    "tensorrt": "com.deepsafe.tensorrt.version",
    "gstreamer": "com.deepsafe.gstreamer.version",
    "repository": "com.deepsafe.deepstream-yolo.repository",
    "commit": "com.deepsafe.deepstream-yolo.commit",
    "tree": "com.deepsafe.deepstream-yolo.tree",
    "patch_sha256": "com.deepsafe.deepstream-yolo.patch-sha256",
    "upstream_source_sha256": (
        "com.deepsafe.deepstream-yolo.upstream-source-sha256"
    ),
    "patched_source_sha256": (
        "com.deepsafe.deepstream-yolo.patched-source-sha256"
    ),
    "patched_tree": "com.deepsafe.deepstream-yolo.patched-tree",
    "parser_build_makefile_sha256": (
        "com.deepsafe.deepstream-yolo.parser-build-makefile-sha256"
    ),
    "parser_cuda_cubin_architecture": (
        "com.deepsafe.deepstream-yolo.parser-cuda-cubin-architecture"
    ),
    "parser_cuda_ptx_architecture": (
        "com.deepsafe.deepstream-yolo.parser-cuda-ptx-architecture"
    ),
    "parser_cuda_gencode_flags": (
        "com.deepsafe.deepstream-yolo.parser-cuda-gencode-flags"
    ),
    "parser_build_command_sha256": (
        "com.deepsafe.deepstream-yolo.parser-build-command-sha256"
    ),
    "parser_post_link_tool_path": (
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-path"
    ),
    "parser_post_link_tool_sha256": (
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-sha256"
    ),
    "parser_post_link_tool_version": (
        "com.deepsafe.deepstream-yolo.parser-post-link-tool-version"
    ),
    "parser_post_link_command": (
        "com.deepsafe.deepstream-yolo.parser-post-link-command"
    ),
    "parser_post_link_command_sha256": (
        "com.deepsafe.deepstream-yolo.parser-post-link-command-sha256"
    ),
    "parser_post_link_removed_sections": (
        "com.deepsafe.deepstream-yolo.parser-post-link-removed-sections"
    ),
    "parser_post_link_retained_sections": (
        "com.deepsafe.deepstream-yolo.parser-post-link-retained-sections"
    ),
    "kernel_proof_schema": "com.deepsafe.cuda-kernel-proof.schema",
    "parser_sha256": "com.deepsafe.deepstream-yolo.parser-sha256",
    "controller_sha256": "com.deepsafe.runtime-compatibility-controller.sha256",
    "control_manifest_sha256": "com.deepsafe.runtime-control-manifest.sha256",
    "dockerignore_sha256": "com.deepsafe.dockerignore.sha256",
}


class Ds9CompatibilityError(RuntimeError):
    """Raised when static/runtime compatibility evidence is not acceptance-safe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Ds9CompatibilityError(f"{label} must be an RFC3339 timestamp")
    rendered = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise Ds9CompatibilityError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise Ds9CompatibilityError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_sha256(value: Any) -> str:
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
        raise Ds9CompatibilityError(
            f"compatibility artifact must stay inside project root: {resolved}"
        ) from exc
    current = root
    for component in relative.parts:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise Ds9CompatibilityError(
                f"compatibility artifact path cannot contain a symlink: {current}"
            )
    return resolved


def _relative(path: Path, project_root: Path) -> str:
    return _inside_root(path, project_root).relative_to(project_root.resolve()).as_posix()


def make_file_pin(path: Path, *, project_root: Path) -> dict[str, Any]:
    """Hash a regular, non-symlink file through one stable descriptor."""

    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise Ds9CompatibilityError(f"cannot open pinned file: {resolved}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Ds9CompatibilityError(f"pinned artifact is not regular: {resolved}")
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
        raise Ds9CompatibilityError(f"artifact changed while hashing: {resolved}")
    return {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_json_stable(
    path: Path, *, project_root: Path, immutable: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _inside_root(path, project_root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise Ds9CompatibilityError(f"cannot open compatibility JSON: {resolved}") from exc
    digest = hashlib.sha256()
    content = bytearray()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Ds9CompatibilityError("compatibility JSON must be a regular file")
        if before.st_size < 0 or before.st_size > MAX_COMPATIBILITY_JSON_BYTES:
            raise Ds9CompatibilityError("compatibility JSON exceeds its byte limit")
        if immutable:
            if stat.S_IMODE(before.st_mode) != 0o440:
                raise Ds9CompatibilityError("compatibility receipt must have mode 0440")
            if before.st_nlink != 1:
                raise Ds9CompatibilityError("compatibility receipt must have one hard link")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            content.extend(chunk)
            digest.update(chunk)
            if len(content) > MAX_COMPATIBILITY_JSON_BYTES:
                raise Ds9CompatibilityError(
                    "compatibility JSON exceeds its byte limit"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
        after.st_nlink,
    ):
        raise Ds9CompatibilityError("compatibility JSON changed while reading")
    if len(content) != before.st_size:
        raise Ds9CompatibilityError("compatibility JSON size changed while reading")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Ds9CompatibilityError(
                    f"duplicate compatibility JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ds9CompatibilityError("compatibility JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise Ds9CompatibilityError("compatibility JSON must contain an object")
    return payload, {
        "path": _relative(resolved, project_root),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while persisting compatibility report")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _exclusive_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while persisting compatibility receipt")
            offset += written
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def runtime_control_pins(project_root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    return {
        name: make_file_pin(project_root / relative, project_root=project_root)
        for name, relative in RUNTIME_CONTROL_PATHS.items()
    }


def validate_runtime_control_manifest(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    path = project_root / RUNTIME_CONTROL_MANIFEST
    payload, pin = _read_json_stable(path, project_root=project_root, immutable=False)
    if payload.get("schema_version") != CONTROL_MANIFEST_SCHEMA_VERSION:
        raise Ds9CompatibilityError("runtime control manifest schema differs")
    artifacts = payload.get("artifacts")
    live = runtime_control_pins(project_root)
    if not isinstance(artifacts, dict) or artifacts != live:
        raise Ds9CompatibilityError("runtime control manifest differs from live controls")
    expected_projection = {
        "schema_version": CONTROL_MANIFEST_SCHEMA_VERSION,
        "artifacts": live,
    }
    if set(payload) != set(expected_projection) or payload != expected_projection:
        raise Ds9CompatibilityError("runtime control manifest contains unapproved fields")
    return {
        "pin": pin,
        "projection_sha256": _canonical_sha256(expected_projection),
        "artifacts": live,
    }


def build_static_probe_command(resolved_image_id: str) -> list[str]:
    if IMAGE_ID_RE.fullmatch(resolved_image_id) is None:
        raise Ds9CompatibilityError("static probe requires an immutable sha256 image ID")
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--entrypoint",
        "python3",
        resolved_image_id,
        CONTAINER_CONTROLLER.as_posix(),
        "--inside-container-probe",
    ]


def _probe_command_contract() -> dict[str, Any]:
    return {
        "argv_prefix": [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--entrypoint",
            "python3",
        ],
        "image": "immutable sha256:<64 lowercase hex> resolved by docker image inspect",
        "argv_suffix": [
            CONTAINER_CONTROLLER.as_posix(),
            "--inside-container-probe",
        ],
        "gpu_devices_requested": False,
        "network": "none",
        "pull_policy": "never",
    }


def make_pending_report(
    *,
    requested_image: str,
    project_root: Path = PROJECT_ROOT,
    launch_scope: str,
) -> dict[str, Any]:
    controls = validate_runtime_control_manifest(project_root)
    return {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": "pending_static_probe",
        "production_ready": False,
        "created_at_utc": utc_now(),
        "requested_image": requested_image,
        "launch_scope": launch_scope,
        "static_probe_command_contract": _probe_command_contract(),
        "runtime_controls": controls,
        "pending": {
            "static_container_probe": True,
            "gpu_smoke_status": "pending_gpu_smoke",
            "gpu_smoke_checks": list(REQUIRED_GPU_SMOKE_CHECKS),
        },
        "docker_called": False,
        "gpu_process_started": False,
    }


def write_pending_report(
    path: Path,
    *,
    requested_image: str,
    project_root: Path = PROJECT_ROOT,
    launch_scope: str,
) -> dict[str, Any]:
    resolved = _inside_root(path, project_root)
    payload = make_pending_report(
        requested_image=requested_image,
        project_root=project_root,
        launch_scope=launch_scope,
    )
    _atomic_json(resolved, payload)
    return payload


def inspect_local_image(
    image: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    completed = runner(
        ["docker", "image", "inspect", image],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        values = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Ds9CompatibilityError("docker image inspect returned invalid JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise Ds9CompatibilityError("docker image inspect must return exactly one image")
    value = values[0]
    labels = (value.get("Config") or {}).get("Labels")
    if not isinstance(labels, dict):
        raise Ds9CompatibilityError("image has no build-lineage labels")
    result = {
        "requested_image": image,
        "resolved_image_id": value.get("Id"),
        "architecture": value.get("Architecture"),
        "os": value.get("Os"),
        "repo_digests": value.get("RepoDigests") or [],
        "labels": labels,
    }
    if IMAGE_ID_RE.fullmatch(str(result["resolved_image_id"])) is None:
        raise Ds9CompatibilityError("docker image inspect returned an invalid image ID")
    return result


def validate_image_lineage(
    image: Mapping[str, Any], *, controls: Mapping[str, Any]
) -> dict[str, Any]:
    if image.get("architecture") != EXPECTED_ARCHITECTURE or image.get("os") != EXPECTED_OS:
        raise Ds9CompatibilityError("image OS/architecture is not exact linux/amd64")
    labels = image.get("labels")
    if not isinstance(labels, dict):
        raise Ds9CompatibilityError("image labels are missing")
    controller_pin = controls["artifacts"]["ds9_runtime_compatibility"]
    expected_exact = {
        LABELS["schema"]: BUILD_LINEAGE_SCHEMA_VERSION,
        LABELS["deepstream"]: DEEPSTREAM_VERSION,
        LABELS["cuda"]: CUDA_VERSION,
        LABELS["tensorrt"]: TENSORRT_VERSION,
        LABELS["gstreamer"]: GSTREAMER_VERSION,
        LABELS["repository"]: DEEPSTREAM_YOLO_REPOSITORY,
        LABELS["commit"]: DEEPSTREAM_YOLO_COMMIT,
        LABELS["tree"]: DEEPSTREAM_YOLO_TREE,
        LABELS["patch_sha256"]: DEEPSTREAM_YOLO_PATCH_SHA256,
        LABELS["upstream_source_sha256"]: DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256,
        LABELS["patched_source_sha256"]: DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256,
        LABELS["patched_tree"]: DEEPSTREAM_YOLO_PATCHED_TREE,
        LABELS[
            "parser_build_makefile_sha256"
        ]: DEEPSTREAM_YOLO_PARSER_BUILD_MAKEFILE_SHA256,
        LABELS[
            "parser_cuda_cubin_architecture"
        ]: DEEPSTREAM_YOLO_PARSER_CUDA_CUBIN_ARCHITECTURE,
        LABELS[
            "parser_cuda_ptx_architecture"
        ]: DEEPSTREAM_YOLO_PARSER_CUDA_PTX_ARCHITECTURE,
        LABELS["parser_cuda_gencode_flags"]: ";".join(
            DEEPSTREAM_YOLO_PARSER_CUDA_GENCODE_FLAGS
        ),
        LABELS[
            "parser_build_command_sha256"
        ]: DEEPSTREAM_YOLO_PARSER_BUILD_COMMAND_SHA256,
        LABELS[
            "parser_post_link_tool_path"
        ]: DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_PATH,
        LABELS[
            "parser_post_link_tool_sha256"
        ]: DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_SHA256,
        LABELS[
            "parser_post_link_tool_version"
        ]: DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_VERSION,
        LABELS[
            "parser_post_link_command"
        ]: DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND,
        LABELS[
            "parser_post_link_command_sha256"
        ]: DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND_SHA256,
        LABELS["parser_post_link_removed_sections"]: ";".join(
            DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        LABELS["parser_post_link_retained_sections"]: ";".join(
            DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        LABELS["kernel_proof_schema"]: CUDA_KERNEL_PROOF_SCHEMA_VERSION,
        LABELS["controller_sha256"]: controller_pin["sha256"],
        LABELS["control_manifest_sha256"]: controls["pin"]["sha256"],
        LABELS["dockerignore_sha256"]: controls["artifacts"]
        ["docker_build_context_policy"]["sha256"],
    }
    for name, expected in expected_exact.items():
        if labels.get(name) != expected:
            raise Ds9CompatibilityError(f"image lineage label differs: {name}")
    base_digest = labels.get(LABELS["base_digest"])
    base_ref = labels.get(LABELS["base_ref"])
    parser_sha = labels.get(LABELS["parser_sha256"])
    if IMAGE_ID_RE.fullmatch(str(base_digest)) is None:
        raise Ds9CompatibilityError("base image digest label is not immutable")
    if base_ref != f"{DEEPSTREAM_BASE_TAG}@{base_digest}":
        raise Ds9CompatibilityError("base image ref/digest labels are inconsistent")
    if SHA256_RE.fullmatch(str(parser_sha)) is None:
        raise Ds9CompatibilityError("parser SHA label is invalid")
    return dict(image)


def _run_static_command(argv: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(argv),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-131072:],
        "stderr": completed.stderr[-131072:],
    }


def _run_bounded_cuobjdump_list_command(argv: Sequence[str]) -> dict[str, Any]:
    """Capture a complete small cuobjdump listing; never retain a tail."""

    completed = subprocess.run(
        list(argv),
        check=False,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes = bytes(completed.stdout)
    stderr_bytes = bytes(completed.stderr)
    if (
        len(stdout_bytes) > MAX_CUOBJDUMP_LIST_BYTES
        or len(stderr_bytes) > MAX_CUOBJDUMP_LIST_BYTES
    ):
        raise Ds9CompatibilityError("cuobjdump listing exceeds its full-capture limit")
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
        stderr = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Ds9CompatibilityError("cuobjdump listing is not UTF-8") from exc
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _nm_command_argv() -> list[str]:
    return ["nm", "-D", "--defined-only", PARSER_LIBRARY.as_posix()]


def _parser_section_command_argv() -> list[str]:
    return [
        "/usr/bin/readelf",
        "--section-headers",
        "--wide",
        PARSER_LIBRARY.as_posix(),
    ]


def _run_bounded_readelf_sections_command(argv: Sequence[str]) -> dict[str, Any]:
    """Capture the complete parser section table without tail truncation."""

    completed = subprocess.run(
        list(argv),
        check=False,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes = bytes(completed.stdout)
    stderr_bytes = bytes(completed.stderr)
    if (
        len(stdout_bytes) > MAX_READELF_SECTION_BYTES
        or len(stderr_bytes) > MAX_READELF_SECTION_BYTES
    ):
        raise Ds9CompatibilityError("parser readelf section table exceeds its limit")
    try:
        stdout = stdout_bytes.decode("utf-8", errors="strict")
        stderr = stderr_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Ds9CompatibilityError("parser readelf section table is not UTF-8") from exc
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _parse_parser_post_link_section_counts(stdout: str) -> dict[str, int]:
    names = re.findall(r"^\s*\[\s*[0-9]+\]\s+(\.[^\s]+)\s+", stdout, re.MULTILINE)
    required = (
        *DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS,
        *DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS,
    )
    return {name: names.count(name) for name in required}


def _line_names_required_export(line: str) -> bool:
    columns = line.rsplit(maxsplit=1)
    return len(columns) == 2 and columns[1] in REQUIRED_SYMBOLS


def _run_bounded_nm_symbol_command(argv: Sequence[str]) -> dict[str, Any]:
    """Capture complete nm stdout, then retain only exact target-symbol lines.

    The complete output is measured and hashed before projection.  No tail or
    prefix truncation participates in symbol discovery.  The JSON evidence is
    bounded independently; an oversized/undecodable capture or projection is
    represented explicitly and rejected by :func:`validate_static_probe`.
    """

    completed = subprocess.run(
        list(argv),
        check=False,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes = bytes(completed.stdout)
    stderr_bytes = bytes(completed.stderr)
    within_limit = len(stdout_bytes) <= MAX_NM_ORIGINAL_STDOUT_BYTES
    try:
        decoded = stdout_bytes.decode("utf-8", errors="strict")
        utf8_decoded = True
    except UnicodeDecodeError:
        decoded = ""
        utf8_decoded = False

    matching_lines: list[str] = []
    if within_limit and utf8_decoded:
        matching_lines = [
            line for line in decoded.splitlines() if _line_names_required_export(line)
        ]
    projection_complete = (
        within_limit
        and utf8_decoded
        and len(matching_lines) <= MAX_NM_PROJECTED_MATCHES
    )
    projected = "\n".join(matching_lines)
    if projected:
        projected += "\n"
    if len(projected.encode("utf-8")) > MAX_NM_PROJECTED_STDOUT_BYTES:
        projection_complete = False
    if not projection_complete:
        projected = ""

    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": projected,
        "stderr": stderr_bytes[-131072:].decode("utf-8", errors="replace"),
        "stdout_original_bytes": len(stdout_bytes),
        "stdout_original_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_original_utf8_decoded": utf8_decoded,
        "stdout_original_within_limit": within_limit,
        "stdout_projection_policy": NM_PROJECTION_POLICY,
        "stdout_projection_complete": projection_complete,
        "stdout_projection_match_count": len(matching_lines),
    }


def _parse_nm_symbol_projection(stdout: str) -> dict[str, Any]:
    counts = {name: 0 for name in REQUIRED_SYMBOLS}
    invalid_lines: list[str] = []
    record = re.compile(r"^([0-9a-fA-F]{16}) T (\S+)$")
    for line in stdout.splitlines():
        match = record.fullmatch(line)
        if match is None or match.group(2) not in counts:
            invalid_lines.append(line)
            continue
        counts[match.group(2)] += 1
    return {"counts": counts, "invalid_lines": invalid_lines}


def _first_version(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else None


def _parse_cuobjdump_elf_entries(stdout: str) -> list[dict[str, Any]]:
    """Parse only tool-labelled ``--list-elf`` stdout records.

    SASS, PTX and stderr are deliberately excluded: an ``sm_86`` token in
    those channels is not evidence that the ELF contains an sm_86 cubin.
    """

    record = re.compile(r"^\s*ELF file\s+([0-9]+):\s+(\S+)\s*$", re.IGNORECASE)
    name_record = re.compile(
        r"^libnvdsinfer_custom_impl_Yolo\.([0-9]+)\.sm_([0-9]+)\.cubin$"
    )
    entries: list[dict[str, Any]] = []
    observed_indexes: set[int] = set()
    for line in stdout.splitlines():
        match = record.fullmatch(line)
        if not line.strip():
            continue
        if match is None:
            raise Ds9CompatibilityError("cuobjdump ELF listing contains an invalid line")
        index = int(match.group(1))
        name = match.group(2)
        if index <= 0 or index in observed_indexes:
            raise Ds9CompatibilityError("cuobjdump ELF listing indexes are invalid")
        observed_indexes.add(index)
        name_match = name_record.fullmatch(name)
        if name_match is None or int(name_match.group(1)) != index:
            raise Ds9CompatibilityError("cuobjdump ELF listing name/index differs")
        entries.append(
            {
                "index": index,
                "name": name,
                "architecture": int(name_match.group(2)),
            }
        )
    return entries


def _parse_cuobjdump_ptx_entries(stdout: str) -> list[dict[str, Any]]:
    """Parse bounded ``cuobjdump --list-ptx`` records without PTX-body tails."""

    record = re.compile(r"^\s*PTX file\s+([0-9]+):\s+(\S+)\s*$", re.IGNORECASE)
    name_record = re.compile(
        r"^libnvdsinfer_custom_impl_Yolo\.([0-9]+)\.sm_([0-9]+)\.ptx$"
    )
    entries: list[dict[str, Any]] = []
    observed_indexes: set[int] = set()
    for line in stdout.splitlines():
        match = record.fullmatch(line)
        if not line.strip():
            continue
        if match is None:
            raise Ds9CompatibilityError("cuobjdump PTX listing contains an invalid line")
        index = int(match.group(1))
        name = match.group(2)
        if index <= 0 or index in observed_indexes:
            raise Ds9CompatibilityError("cuobjdump PTX listing indexes are invalid")
        observed_indexes.add(index)
        name_match = name_record.fullmatch(name)
        if name_match is None or int(name_match.group(1)) != index:
            raise Ds9CompatibilityError("cuobjdump PTX listing name/index differs")
        entries.append(
            {
                "index": index,
                "name": name,
                "architecture": int(name_match.group(2)),
            }
        )
    return entries


def _nvinfer_static_command_argv() -> dict[str, list[str]]:
    """Return the exact non-loading nvinfer artifact inspection contract."""

    plugin = EXPECTED_NVINFER_PATH
    return {
        "deepstream_version_manifest": [
            "cat",
            DEEPSTREAM_VERSION_MANIFEST.as_posix(),
        ],
        "nvinfer_realpath": ["readlink", "--canonicalize-existing", plugin],
        "nvinfer_stat": [
            "stat",
            "--format=%f|%s|%a|%u|%g",
            plugin,
        ],
        "nvinfer_sha256": ["sha256sum", "--binary", plugin],
        "nvinfer_readelf_header": [
            "readelf",
            "--file-header",
            "--wide",
            plugin,
        ],
        "nvinfer_readelf_dynamic": [
            "readelf",
            "--dynamic",
            "--wide",
            plugin,
        ],
        "nvinfer_readelf_symbols": [
            "readelf",
            "--dyn-syms",
            "--wide",
            plugin,
        ],
        "nvinfer_strings": ["strings", "--all", "--bytes=4", plugin],
    }


def _parse_nvinfer_elf_header(stdout: str) -> dict[str, str | None]:
    def field(label: str) -> str | None:
        match = re.search(
            rf"^\s*{re.escape(label)}:\s*(.+?)\s*$",
            stdout,
            flags=re.MULTILINE,
        )
        return match.group(1) if match else None

    elf_type = field("Type")
    return {
        "class": field("Class"),
        "data": field("Data"),
        "type": elf_type.split()[0] if elf_type else None,
        "machine": field("Machine"),
    }


def _parse_nvinfer_symbol_counts(stdout: str) -> dict[str, int]:
    counts = {name: 0 for name in NVINFER_DESCRIPTOR_SYMBOLS}
    for line in stdout.splitlines():
        columns = line.split()
        if len(columns) < 8:
            continue
        name = columns[-1]
        if (
            name in counts
            and columns[3] == "FUNC"
            and columns[4] == "GLOBAL"
            and columns[6] != "UND"
        ):
            counts[name] += 1
    return counts


def _parse_nvinfer_sha256(stdout: str) -> str | None:
    match = re.fullmatch(
        rf"([0-9a-f]{{64}}) \*{re.escape(EXPECTED_NVINFER_PATH)}\n?",
        stdout,
    )
    return match.group(1) if match else None


def _parse_nvinfer_stat(stdout: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"([0-9a-f]+)\|([0-9]+)\|([0-7]+)\|([0-9]+)\|([0-9]+)\n?",
        stdout,
    )
    if match is None:
        return {
            "file_type": None,
            "raw_mode_hex": None,
            "bytes": None,
            "permissions_octal": None,
            "uid": None,
            "gid": None,
        }
    raw_mode = int(match.group(1), 16)
    return {
        "file_type": "regular_file" if raw_mode & 0o170000 == 0o100000 else None,
        "raw_mode_hex": match.group(1),
        "bytes": int(match.group(2)),
        "permissions_octal": match.group(3),
        "uid": int(match.group(4)),
        "gid": int(match.group(5)),
    }


def _derive_nvinfer_static_facts(
    commands: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive nvinfer facts solely from captured, non-loading command stdout."""

    manifest_stdout = str(commands["deepstream_version_manifest"].get("stdout", ""))
    manifest_version = _first_version(
        r"^Version:\s*([0-9.]+)\s*$", manifest_stdout
    )
    dynamic_stdout = str(commands["nvinfer_readelf_dynamic"].get("stdout", ""))
    strings_lines = str(commands["nvinfer_strings"].get("stdout", "")).splitlines()
    embedded_markers = ("nvinfer plugin", DEEPSTREAM_VERSION, "nvdsgst_infer")
    return {
        "nvinfer_verification_scope": NVINFER_STATIC_VERIFICATION_SCOPE,
        "nvinfer_runtime_plugin_load_attempted": False,
        "nvinfer_artifact_path": str(
            commands["nvinfer_realpath"].get("stdout", "")
        ).rstrip("\n"),
        "nvinfer_artifact_stat": _parse_nvinfer_stat(
            str(commands["nvinfer_stat"].get("stdout", ""))
        ),
        "nvinfer_artifact_sha256": _parse_nvinfer_sha256(
            str(commands["nvinfer_sha256"].get("stdout", ""))
        ),
        "nvinfer_sdk_manifest_path": DEEPSTREAM_VERSION_MANIFEST.as_posix(),
        "nvinfer_sdk_manifest_version": manifest_version,
        "nvinfer_elf_identity": _parse_nvinfer_elf_header(
            str(commands["nvinfer_readelf_header"].get("stdout", ""))
        ),
        "nvinfer_elf_needed": re.findall(
            r"Shared library: \[([^]]+)\]", dynamic_stdout
        ),
        "nvinfer_descriptor_symbol_counts": _parse_nvinfer_symbol_counts(
            str(commands["nvinfer_readelf_symbols"].get("stdout", ""))
        ),
        "nvinfer_embedded_descriptor_marker_counts": {
            marker: strings_lines.count(marker) for marker in embedded_markers
        },
        "nvinfer_version_binding": (
            "exact_binary_sha256_plus_embedded_marker_plus_sdk_version_manifest"
        ),
    }


def _inside_container_probe() -> dict[str, Any]:
    parser_path = PARSER_LIBRARY
    commands = {
        "deepstream": _run_static_command(["deepstream-app", "--version-all"]),
        "tensorrt": _run_static_command(
            ["dpkg-query", "-W", "-f=${Version}", "libnvinfer10"]
        ),
        "cuda": _run_static_command(
            ["/usr/local/cuda-13.1/bin/nvcc", "--version"]
        ),
        "gstreamer": _run_static_command(["gst-launch-1.0", "--version"]),
        "readelf": _run_static_command(["readelf", "-d", parser_path.as_posix()]),
        "readelf_sections": _run_bounded_readelf_sections_command(
            _parser_section_command_argv()
        ),
        "ldd": _run_static_command(["ldd", parser_path.as_posix()]),
        "nm": _run_bounded_nm_symbol_command(_nm_command_argv()),
        "cuobjdump_elf": _run_bounded_cuobjdump_list_command(
            ["/usr/local/cuda-13.1/bin/cuobjdump", "--list-elf", parser_path.as_posix()]
        ),
        "cuobjdump_ptx": _run_bounded_cuobjdump_list_command(
            ["/usr/local/cuda-13.1/bin/cuobjdump", "--list-ptx", parser_path.as_posix()]
        ),
        "cuobjdump_sass": _run_static_command(
            ["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-sass", parser_path.as_posix()]
        ),
    }
    commands.update(
        {
            name: _run_static_command(argv)
            for name, argv in _nvinfer_static_command_argv().items()
        }
    )
    parser_sha: str | None = None
    try:
        parser_sha = hashlib.sha256(parser_path.read_bytes()).hexdigest()
    except OSError:
        pass

    dlsym: dict[str, bool] = {name: False for name in REQUIRED_SYMBOLS}
    dlsym_error: str | None = None
    try:
        library = ctypes.CDLL(parser_path.as_posix(), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        dlsym = {name: getattr(library, name, None) is not None for name in REQUIRED_SYMBOLS}
    except (OSError, AttributeError) as exc:
        dlsym_error = f"{type(exc).__name__}: {exc}"

    abi_source = r'''
#include "nvdsinfer_custom_impl.h"
extern "C" bool NvDsInferParseYoloCuda(
    std::vector<NvDsInferLayerInfo> const &,
    NvDsInferNetworkInfo const &,
    NvDsInferParseDetectionParams const &,
    std::vector<NvDsInferObjectDetectionInfo> &);
extern "C" bool NvDsInferYoloCudaEngineGet(
    nvinfer1::IBuilder * const,
    nvinfer1::IBuilderConfig * const,
    const NvDsInferContextInitParams * const,
    nvinfer1::DataType,
    nvinfer1::ICudaEngine *&);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloCuda);
CHECK_CUSTOM_ENGINE_CREATE_FUNC_PROTOTYPE(NvDsInferYoloCudaEngineGet);
'''
    with tempfile.TemporaryDirectory(prefix="deepsafe-ds9-abi-") as temporary:
        source = Path(temporary) / "probe.cpp"
        output = Path(temporary) / "probe.o"
        source.write_text(abi_source, encoding="utf-8")
        commands["abi_compile"] = _run_static_command(
            [
                "g++",
                "-std=c++17",
                "-Werror",
                "-c",
                source.as_posix(),
                "-I/opt/nvidia/deepstream/deepstream/sources/includes",
                "-I/usr/local/cuda-13.1/include",
                "-o",
                output.as_posix(),
            ]
        )

    build_lineage: dict[str, Any] | None = None
    control_manifest_sha: str | None = None
    dockerignore_sha: str | None = None
    gpu_smoke_harness_sha: str | None = None
    file_errors: list[str] = []
    try:
        candidate = json.loads(BUILD_LINEAGE_MANIFEST.read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            build_lineage = candidate
    except (OSError, json.JSONDecodeError) as exc:
        file_errors.append(f"build_lineage={type(exc).__name__}: {exc}")
    try:
        control_manifest_sha = hashlib.sha256(
            CONTAINER_CONTROL_MANIFEST.read_bytes()
        ).hexdigest()
    except OSError as exc:
        file_errors.append(f"control_manifest={type(exc).__name__}: {exc}")
    try:
        dockerignore_sha = hashlib.sha256(
            CONTAINER_DOCKERIGNORE_POLICY.read_bytes()
        ).hexdigest()
    except OSError as exc:
        file_errors.append(f"dockerignore={type(exc).__name__}: {exc}")
    try:
        gpu_smoke_harness_sha = hashlib.sha256(
            CONTAINER_GPU_SMOKE_HARNESS.read_bytes()
        ).hexdigest()
    except OSError as exc:
        file_errors.append(f"gpu_smoke_harness={type(exc).__name__}: {exc}")

    deepstream_text = "\n".join(
        (commands["deepstream"]["stdout"], commands["deepstream"]["stderr"])
    )
    gst_text = "\n".join(
        (commands["gstreamer"]["stdout"], commands["gstreamer"]["stderr"])
    )
    nm_projection = _parse_nm_symbol_projection(commands["nm"]["stdout"])
    parser_section_counts = _parse_parser_post_link_section_counts(
        commands["readelf_sections"]["stdout"]
    )
    needed = re.findall(r"Shared library: \[([^]]+)\]", commands["readelf"]["stdout"])
    ldd_missing = [
        line.strip()
        for line in commands["ldd"]["stdout"].splitlines()
        if "not found" in line.casefold()
    ]
    cubin_elf_entries = _parse_cuobjdump_elf_entries(
        commands["cuobjdump_elf"]["stdout"]
    )
    ptx_entries = _parse_cuobjdump_ptx_entries(
        commands["cuobjdump_ptx"]["stdout"]
    )
    ptx_targets = sorted(
        {str(entry["architecture"]) for entry in ptx_entries if entry["architecture"] is not None}
    )
    sm86_present = any(
        entry["architecture"] == 86 for entry in cubin_elf_entries
    )
    sm86_only_cubin_set = [
        entry["index"] for entry in cubin_elf_entries
    ] == list(EXPECTED_PARSER_CUDA_ENTRY_INDEXES) and all(
        entry["architecture"] == 86 for entry in cubin_elf_entries
    )
    compute86_only_ptx_set = [
        entry["index"] for entry in ptx_entries
    ] == list(EXPECTED_PARSER_CUDA_ENTRY_INDEXES) and all(
        entry["architecture"] == 86 for entry in ptx_entries
    )
    forward_ptx_present = any(int(value) <= 86 for value in ptx_targets)
    facts = {
        "deepstream_app_version": _first_version(
            r"deepstream-app\s+version\s+([0-9.]+)", deepstream_text
        ),
        "deepstream_sdk_version": _first_version(
            r"DeepStreamSDK\s+([0-9.]+)", deepstream_text
        ),
        "tensorrt_version": _first_version(
            r"([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)",
            commands["tensorrt"]["stdout"],
        ),
        "cuda_version": _first_version(
            r"release\s+([0-9]+\.[0-9]+)", commands["cuda"]["stdout"]
        ),
        "gstreamer_version": _first_version(r"version\s+([0-9.]+)", gst_text),
        "parser_sha256": parser_sha,
        "parser_post_link_section_counts": parser_section_counts,
        "readelf_needed": needed,
        "ldd_missing": ldd_missing,
        "nm_symbol_counts": nm_projection["counts"],
        "nm_symbol_projection_invalid_lines": nm_projection["invalid_lines"],
        "nm_original_stdout": {
            "bytes": commands["nm"]["stdout_original_bytes"],
            "sha256": commands["nm"]["stdout_original_sha256"],
            "utf8_decoded": commands["nm"]["stdout_original_utf8_decoded"],
            "within_limit": commands["nm"]["stdout_original_within_limit"],
        },
        "dlsym": dlsym,
        "dlsym_error": dlsym_error,
        "abi_compile_passed": commands["abi_compile"]["returncode"] == 0,
        "sm86_cubin_present": sm86_present,
        "sm86_only_cubin_set": sm86_only_cubin_set,
        "cubin_elf_entries": cubin_elf_entries,
        "ptx_entries": ptx_entries,
        "ptx_targets": ptx_targets,
        "compute86_only_ptx_set": compute86_only_ptx_set,
        "forward_compatible_ptx_present": forward_ptx_present,
        "build_lineage": build_lineage,
        "runtime_control_manifest_sha256": control_manifest_sha,
        "dockerignore_sha256": dockerignore_sha,
        "gpu_smoke_harness_sha256": gpu_smoke_harness_sha,
        "file_errors": file_errors,
    }
    facts.update(_derive_nvinfer_static_facts(commands))
    return {
        "schema_version": STATIC_PROBE_SCHEMA_VERSION,
        "commands": commands,
        "facts": facts,
    }


def validate_static_probe(
    probe: Mapping[str, Any], *, image: Mapping[str, Any], controls: Mapping[str, Any]
) -> dict[str, Any]:
    if probe.get("schema_version") != STATIC_PROBE_SCHEMA_VERSION:
        raise Ds9CompatibilityError("static probe schema differs")
    commands = probe.get("commands")
    facts = probe.get("facts")
    if not isinstance(commands, dict) or not isinstance(facts, dict):
        raise Ds9CompatibilityError("static probe commands/facts are missing")
    required_commands = {
        "deepstream",
        "tensorrt",
        "cuda",
        "gstreamer",
        "readelf",
        "readelf_sections",
        "ldd",
        "nm",
        "cuobjdump_elf",
        "cuobjdump_ptx",
        "cuobjdump_sass",
        "abi_compile",
        *_nvinfer_static_command_argv(),
    }
    if set(commands) != required_commands:
        raise Ds9CompatibilityError("static probe command set differs")
    common_command_keys = {"argv", "returncode", "stdout", "stderr"}
    nm_command_keys = common_command_keys | {
        "stdout_original_bytes",
        "stdout_original_sha256",
        "stdout_original_utf8_decoded",
        "stdout_original_within_limit",
        "stdout_projection_policy",
        "stdout_projection_complete",
        "stdout_projection_match_count",
    }
    for name, value in commands.items():
        if (
            not isinstance(value, dict)
            or set(value) != (nm_command_keys if name == "nm" else common_command_keys)
            or not isinstance(value.get("argv"), list)
            or not all(isinstance(item, str) for item in value["argv"])
            or value.get("returncode") != 0
            or not isinstance(value.get("stdout"), str)
            or not isinstance(value.get("stderr"), str)
            or len(value["stdout"]) > 131072
            or len(value["stderr"]) > 131072
        ):
            raise Ds9CompatibilityError(
                f"static probe command evidence is invalid: {name}"
            )
    for name in ("cuobjdump_elf", "cuobjdump_ptx"):
        if (
            len(commands[name]["stdout"].encode("utf-8"))
            > MAX_CUOBJDUMP_LIST_BYTES
            or len(commands[name]["stderr"].encode("utf-8"))
            > MAX_CUOBJDUMP_LIST_BYTES
            or commands[name]["stderr"] != ""
        ):
            raise Ds9CompatibilityError(
                f"static cuobjdump full listing limit/stderr differs: {name}"
            )
    if (
        commands["readelf_sections"].get("argv")
        != _parser_section_command_argv()
        or len(commands["readelf_sections"]["stdout"].encode("utf-8"))
        > MAX_READELF_SECTION_BYTES
        or len(commands["readelf_sections"]["stderr"].encode("utf-8"))
        > MAX_READELF_SECTION_BYTES
        or commands["readelf_sections"]["stderr"] != ""
    ):
        raise Ds9CompatibilityError("static parser section-table evidence differs")
    nm_evidence = commands["nm"]
    nm_original_bytes = nm_evidence.get("stdout_original_bytes")
    nm_match_count = nm_evidence.get("stdout_projection_match_count")
    if (
        nm_evidence.get("argv") != _nm_command_argv()
        or not isinstance(nm_original_bytes, int)
        or isinstance(nm_original_bytes, bool)
        or not 0 < nm_original_bytes <= MAX_NM_ORIGINAL_STDOUT_BYTES
        or nm_original_bytes < len(nm_evidence["stdout"].encode("utf-8"))
        or SHA256_RE.fullmatch(str(nm_evidence.get("stdout_original_sha256")))
        is None
        or nm_evidence.get("stdout_original_utf8_decoded") is not True
        or nm_evidence.get("stdout_original_within_limit") is not True
        or nm_evidence.get("stdout_projection_policy") != NM_PROJECTION_POLICY
        or nm_evidence.get("stdout_projection_complete") is not True
        or not isinstance(nm_match_count, int)
        or isinstance(nm_match_count, bool)
        or not 0 <= nm_match_count <= MAX_NM_PROJECTED_MATCHES
        or len(nm_evidence["stdout"].encode("utf-8"))
        > MAX_NM_PROJECTED_STDOUT_BYTES
        or nm_match_count != len(nm_evidence["stdout"].splitlines())
    ):
        raise Ds9CompatibilityError("bounded parser nm evidence is invalid")
    nm_projection = _parse_nm_symbol_projection(nm_evidence["stdout"])
    if (
        nm_projection["invalid_lines"] != []
        or nm_projection["counts"] != {name: 1 for name in REQUIRED_SYMBOLS}
    ):
        raise Ds9CompatibilityError("parser nm export projection differs")
    if any(
        "gst-inspect-1.0" in command["argv"] for command in commands.values()
    ):
        raise Ds9CompatibilityError(
            "static nvinfer probe must not attempt runtime plugin loading"
        )
    for name, expected_argv in _nvinfer_static_command_argv().items():
        if commands[name].get("argv") != expected_argv:
            raise Ds9CompatibilityError(
                f"static nvinfer artifact command differs: {name}"
            )
    expected_cuda_commands = {
        "cuobjdump_elf": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--list-elf",
            PARSER_LIBRARY.as_posix(),
        ],
        "cuobjdump_ptx": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--list-ptx",
            PARSER_LIBRARY.as_posix(),
        ],
        "cuobjdump_sass": [
            "/usr/local/cuda-13.1/bin/cuobjdump",
            "--dump-sass",
            PARSER_LIBRARY.as_posix(),
        ],
    }
    for name, expected_argv in expected_cuda_commands.items():
        if commands[name].get("argv") != expected_argv:
            raise Ds9CompatibilityError(f"static CUDA probe command differs: {name}")
    expected_facts = {
        "deepstream_app_version": DEEPSTREAM_VERSION,
        "deepstream_sdk_version": DEEPSTREAM_VERSION,
        "tensorrt_version": TENSORRT_VERSION,
        "cuda_version": CUDA_VERSION,
        "gstreamer_version": GSTREAMER_VERSION,
    }
    for name, expected in expected_facts.items():
        if facts.get(name) != expected:
            raise Ds9CompatibilityError(f"static runtime version/path differs: {name}")
    derived_nvinfer = _derive_nvinfer_static_facts(commands)
    for name, observed in derived_nvinfer.items():
        if facts.get(name) != observed:
            raise Ds9CompatibilityError(
                f"static nvinfer fact differs from raw evidence: {name}"
            )
    expected_nvinfer = {
        "nvinfer_verification_scope": NVINFER_STATIC_VERIFICATION_SCOPE,
        "nvinfer_runtime_plugin_load_attempted": False,
        "nvinfer_artifact_path": EXPECTED_NVINFER_PATH,
        "nvinfer_artifact_stat": {
            "file_type": "regular_file",
            "raw_mode_hex": "81ed",
            "bytes": EXPECTED_NVINFER_BYTES,
            "permissions_octal": "755",
            "uid": 0,
            "gid": 0,
        },
        "nvinfer_artifact_sha256": EXPECTED_NVINFER_SHA256,
        "nvinfer_sdk_manifest_path": DEEPSTREAM_VERSION_MANIFEST.as_posix(),
        "nvinfer_sdk_manifest_version": DEEPSTREAM_VERSION,
        "nvinfer_elf_identity": {
            "class": "ELF64",
            "data": "2's complement, little endian",
            "type": "DYN",
            "machine": "Advanced Micro Devices X86-64",
        },
        "nvinfer_elf_needed": list(EXPECTED_NVINFER_NEEDED),
        "nvinfer_descriptor_symbol_counts": {
            name: 1 for name in NVINFER_DESCRIPTOR_SYMBOLS
        },
        "nvinfer_embedded_descriptor_marker_counts": {
            "nvinfer plugin": 1,
            DEEPSTREAM_VERSION: 1,
            "nvdsgst_infer": 1,
        },
        "nvinfer_version_binding": (
            "exact_binary_sha256_plus_embedded_marker_plus_sdk_version_manifest"
        ),
    }
    for name, expected in expected_nvinfer.items():
        if derived_nvinfer.get(name) != expected:
            raise Ds9CompatibilityError(
                f"static nvinfer artifact/version/ABI differs: {name}"
            )
    if "gst_nvinfer_filename" in facts or "gst_nvinfer_version" in facts:
        raise Ds9CompatibilityError(
            "legacy runtime-loaded nvinfer facts are not valid static evidence"
        )
    labels = image["labels"]
    if facts.get("parser_sha256") != labels.get(LABELS["parser_sha256"]):
        raise Ds9CompatibilityError("parser binary SHA differs from image lineage label")
    section_counts = _parse_parser_post_link_section_counts(
        commands["readelf_sections"]["stdout"]
    )
    expected_section_counts = {
        **{
            name: 0
            for name in DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS
        },
        **{
            name: 1
            for name in DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS
        },
    }
    if (
        facts.get("parser_post_link_section_counts") != section_counts
        or section_counts != expected_section_counts
    ):
        raise Ds9CompatibilityError(
            "parser post-link ELF section canonicalization differs"
        )
    if facts.get("runtime_control_manifest_sha256") != controls["pin"]["sha256"]:
        raise Ds9CompatibilityError("container control manifest differs from host/image")
    if facts.get("dockerignore_sha256") != controls["artifacts"][
        "docker_build_context_policy"
    ]["sha256"]:
        raise Ds9CompatibilityError("container .dockerignore policy differs from host/image")
    if facts.get("gpu_smoke_harness_sha256") != controls["artifacts"][
        "ds9_gpu_smoke"
    ]["sha256"]:
        raise Ds9CompatibilityError("container GPU smoke harness differs from controls")
    if facts.get("ldd_missing") != []:
        raise Ds9CompatibilityError("parser has unresolved shared-library dependencies")
    needed = facts.get("readelf_needed")
    if not isinstance(needed, list) or not any("libnvinfer" in item for item in needed):
        raise Ds9CompatibilityError("parser ELF does not bind TensorRT")
    if not any("libcudart" in item for item in needed):
        raise Ds9CompatibilityError("parser ELF does not bind the CUDA runtime")
    if facts.get("nm_symbol_counts") != {name: 1 for name in REQUIRED_SYMBOLS}:
        raise Ds9CompatibilityError("parser exported symbol counts differ")
    if facts.get("nm_symbol_projection_invalid_lines") != []:
        raise Ds9CompatibilityError("parser nm projection contains invalid lines")
    expected_nm_original = {
        "bytes": nm_evidence["stdout_original_bytes"],
        "sha256": nm_evidence["stdout_original_sha256"],
        "utf8_decoded": True,
        "within_limit": True,
    }
    if facts.get("nm_original_stdout") != expected_nm_original:
        raise Ds9CompatibilityError("parser nm original-output metadata differs")
    if facts.get("dlsym") != {name: True for name in REQUIRED_SYMBOLS}:
        raise Ds9CompatibilityError("parser dlsym checks failed")
    if facts.get("dlsym_error") is not None or facts.get("abi_compile_passed") is not True:
        raise Ds9CompatibilityError("parser DS9 ABI probe failed")
    cubin_entries = _parse_cuobjdump_elf_entries(commands["cuobjdump_elf"]["stdout"])
    ptx_entries = _parse_cuobjdump_ptx_entries(commands["cuobjdump_ptx"]["stdout"])
    ptx_targets = sorted(
        {str(entry["architecture"]) for entry in ptx_entries if entry["architecture"] is not None}
    )
    sm86_present = any(entry["architecture"] == 86 for entry in cubin_entries)
    sm86_only_cubin_set = [
        entry["index"] for entry in cubin_entries
    ] == list(EXPECTED_PARSER_CUDA_ENTRY_INDEXES) and all(
        entry["architecture"] == 86 for entry in cubin_entries
    )
    compute86_only_ptx_set = [
        entry["index"] for entry in ptx_entries
    ] == list(EXPECTED_PARSER_CUDA_ENTRY_INDEXES) and all(
        entry["architecture"] == 86 for entry in ptx_entries
    )
    forward_ptx_present = any(int(value) <= 86 for value in ptx_targets)
    if (
        facts.get("cubin_elf_entries") != cubin_entries
        or facts.get("ptx_entries") != ptx_entries
        or facts.get("ptx_targets") != ptx_targets
        or facts.get("sm86_cubin_present") is not sm86_present
        or facts.get("sm86_only_cubin_set") is not sm86_only_cubin_set
        or facts.get("compute86_only_ptx_set") is not compute86_only_ptx_set
        or facts.get("forward_compatible_ptx_present") is not forward_ptx_present
    ):
        raise Ds9CompatibilityError("structured cuobjdump facts differ from stdout")
    if not sm86_only_cubin_set or not compute86_only_ptx_set:
        raise Ds9CompatibilityError(
            "parser CUDA entry set is not exact four-entry sm_86 cubin plus compute_86 PTX"
        )
    if facts.get("file_errors") != []:
        raise Ds9CompatibilityError("container build-lineage files were unreadable")
    lineage = facts.get("build_lineage")
    if not isinstance(lineage, dict):
        raise Ds9CompatibilityError("container build-lineage manifest is missing")
    expected_lineage = {
        "schema_version": BUILD_LINEAGE_SCHEMA_VERSION,
        "base_ref": labels[LABELS["base_ref"]],
        "base_digest": labels[LABELS["base_digest"]],
        "deepstream_version": DEEPSTREAM_VERSION,
        "cuda_version": CUDA_VERSION,
        "tensorrt_version": TENSORRT_VERSION,
        "gstreamer_version": GSTREAMER_VERSION,
        "deepstream_yolo_repository": DEEPSTREAM_YOLO_REPOSITORY,
        "deepstream_yolo_commit": DEEPSTREAM_YOLO_COMMIT,
        "deepstream_yolo_tree": DEEPSTREAM_YOLO_TREE,
        "deepstream_yolo_patch_sha256": DEEPSTREAM_YOLO_PATCH_SHA256,
        "deepstream_yolo_upstream_source_sha256": DEEPSTREAM_YOLO_UPSTREAM_SOURCE_SHA256,
        "deepstream_yolo_patched_source_sha256": DEEPSTREAM_YOLO_PATCHED_SOURCE_SHA256,
        "deepstream_yolo_patched_tree": DEEPSTREAM_YOLO_PATCHED_TREE,
        "deepstream_yolo_parser_build_makefile_sha256": DEEPSTREAM_YOLO_PARSER_BUILD_MAKEFILE_SHA256,
        "deepstream_yolo_parser_cuda_cubin_architecture": DEEPSTREAM_YOLO_PARSER_CUDA_CUBIN_ARCHITECTURE,
        "deepstream_yolo_parser_cuda_ptx_architecture": DEEPSTREAM_YOLO_PARSER_CUDA_PTX_ARCHITECTURE,
        "deepstream_yolo_parser_cuda_gencode_flags": ";".join(
            DEEPSTREAM_YOLO_PARSER_CUDA_GENCODE_FLAGS
        ),
        "deepstream_yolo_parser_build_command_sha256": DEEPSTREAM_YOLO_PARSER_BUILD_COMMAND_SHA256,
        "deepstream_yolo_parser_post_link_tool_path": DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_PATH,
        "deepstream_yolo_parser_post_link_tool_sha256": DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_SHA256,
        "deepstream_yolo_parser_post_link_tool_version": DEEPSTREAM_YOLO_PARSER_POST_LINK_TOOL_VERSION,
        "deepstream_yolo_parser_post_link_command": DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND,
        "deepstream_yolo_parser_post_link_command_sha256": DEEPSTREAM_YOLO_PARSER_POST_LINK_COMMAND_SHA256,
        "deepstream_yolo_parser_post_link_removed_sections": ";".join(
            DEEPSTREAM_YOLO_PARSER_POST_LINK_REMOVED_SECTIONS
        ),
        "deepstream_yolo_parser_post_link_retained_sections": ";".join(
            DEEPSTREAM_YOLO_PARSER_POST_LINK_RETAINED_SECTIONS
        ),
        "cuda_kernel_proof_schema": CUDA_KERNEL_PROOF_SCHEMA_VERSION,
        "parser_sha256": labels[LABELS["parser_sha256"]],
        "controller_sha256": controls["artifacts"]["ds9_runtime_compatibility"][
            "sha256"
        ],
        "runtime_control_manifest_sha256": controls["pin"]["sha256"],
        "dockerignore_sha256": controls["artifacts"][
            "docker_build_context_policy"
        ]["sha256"],
    }
    if lineage != expected_lineage:
        raise Ds9CompatibilityError("container build-lineage manifest differs")
    return {
        "schema_version": STATIC_PROBE_SCHEMA_VERSION,
        "status": "pass",
        "facts": dict(facts),
        "commands": dict(commands),
    }


def _load_gpu_smoke_evidence(
    path: Path,
    *,
    project_root: Path,
    resolved_image_id: str,
    controls: Mapping[str, Any],
    expected_parser_sha256: str,
) -> dict[str, Any]:
    payload, pin = _read_json_stable(path, project_root=project_root, immutable=True)
    if payload.get("schema_version") != GPU_SMOKE_SCHEMA_VERSION:
        raise Ds9CompatibilityError("GPU smoke evidence schema differs")
    if payload.get("status") != "pass" or payload.get("resolved_image_id") != resolved_image_id:
        raise Ds9CompatibilityError("GPU smoke evidence image/status differs")
    if payload.get("runtime_control_manifest_sha256") != controls["pin"]["sha256"]:
        raise Ds9CompatibilityError("GPU smoke evidence control binding differs")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_GPU_SMOKE_CHECKS):
        raise Ds9CompatibilityError("GPU smoke check set differs")
    if any(value != "pass" for value in checks.values()):
        raise Ds9CompatibilityError("one or more GPU smoke checks did not pass")
    try:
        from validation.ds9_gpu_smoke import validate_production_evidence

        replay = validate_production_evidence(
            payload,
            evidence_path=_inside_root(path, project_root),
            project_root=project_root,
            resolved_image_id=resolved_image_id,
            runtime_control_manifest_sha256=controls["pin"]["sha256"],
            expected_parser_sha256=expected_parser_sha256,
        )
    except Exception as exc:
        raise Ds9CompatibilityError(
            f"GPU smoke raw replay failed: {type(exc).__name__}: {exc}"
        ) from exc
    if replay.get("status") != "pass" or replay.get("checks") != checks:
        raise Ds9CompatibilityError("GPU smoke replay/check summary differs")
    return {"status": "pass", "checks": checks, "evidence": pin}


def create_static_receipt(
    *,
    requested_image: str,
    image: Mapping[str, Any],
    probe_command: Sequence[str],
    probe: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
    gpu_smoke_evidence: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    controls = validate_runtime_control_manifest(project_root)
    normalized_image = validate_image_lineage(image, controls=controls)
    resolved_image_id = normalized_image["resolved_image_id"]
    expected_command = build_static_probe_command(resolved_image_id)
    if list(probe_command) != expected_command:
        raise Ds9CompatibilityError("static probe command differs from the closed contract")
    static = validate_static_probe(probe, image=normalized_image, controls=controls)
    if gpu_smoke_evidence is None:
        gpu = {
            "status": "pending_gpu_smoke",
            "checks": {name: "pending" for name in REQUIRED_GPU_SMOKE_CHECKS},
            "evidence": None,
        }
        status = "pending_gpu_smoke"
        production_ready = False
    else:
        gpu = _load_gpu_smoke_evidence(
            gpu_smoke_evidence,
            project_root=project_root,
            resolved_image_id=resolved_image_id,
            controls=controls,
            expected_parser_sha256=normalized_image["labels"][
                LABELS["parser_sha256"]
            ],
        )
        status = "production_ready"
        production_ready = True
    created = _parse_time(created_at_utc or utc_now(), "created_at_utc")
    expires = created + MAX_RECEIPT_AGE
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "production_ready": production_ready,
        "created_at_utc": created.isoformat().replace("+00:00", "Z"),
        "expires_at_utc": expires.isoformat().replace("+00:00", "Z"),
        "requested_image": requested_image,
        "image": normalized_image,
        "runtime_controls": controls,
        "static_probe": {
            "status": "pass",
            "command": expected_command,
            "command_sha256": _canonical_sha256(expected_command),
            "evidence": static,
        },
        "gpu_smoke": gpu,
    }


def require_static_candidate_compatibility(
    receipt_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    requested_image: str,
    resolved_image_id: str,
    now: datetime | None = None,
    image_inspector: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify the static-only candidate accepted solely by the GPU smoke guard.

    This function never upgrades the candidate to production.  It exists to
    break the unavoidable bootstrap cycle: real CUDA/parity evidence must run
    against an image that has passed every GPU-free DS9/ABI/lineage check but
    does not yet have its own GPU smoke receipt.
    """

    payload, receipt_pin = _read_json_stable(
        receipt_path, project_root=project_root, immutable=True
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Ds9CompatibilityError("DS9 static candidate receipt schema differs")
    if (
        payload.get("status") != "pending_gpu_smoke"
        or payload.get("production_ready") is not False
    ):
        raise Ds9CompatibilityError("DS9 smoke bootstrap requires a static-only candidate")
    created = _parse_time(payload.get("created_at_utc"), "created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= created or expires - created > MAX_RECEIPT_AGE:
        raise Ds9CompatibilityError("DS9 static candidate lifetime exceeds 24h")
    if current < created or current >= expires:
        raise Ds9CompatibilityError("DS9 static candidate is not current")
    if payload.get("requested_image") != requested_image:
        raise Ds9CompatibilityError("DS9 static candidate requested image differs")
    image = payload.get("image")
    if not isinstance(image, dict) or image.get("resolved_image_id") != resolved_image_id:
        raise Ds9CompatibilityError("DS9 static candidate immutable image ID differs")
    inspector = image_inspector or inspect_local_image
    observed = inspector(resolved_image_id)
    for key in ("resolved_image_id", "architecture", "os", "labels"):
        if observed.get(key) != image.get(key):
            raise Ds9CompatibilityError(
                f"live static candidate image metadata differs: {key}"
            )
    controls = validate_runtime_control_manifest(project_root)
    if payload.get("runtime_controls") != controls:
        raise Ds9CompatibilityError("DS9 static candidate controls changed")
    validate_image_lineage(image, controls=controls)
    static = payload.get("static_probe")
    expected_command = build_static_probe_command(resolved_image_id)
    if not isinstance(static, dict) or static.get("status") != "pass":
        raise Ds9CompatibilityError("DS9 static candidate probe is not passing")
    if static.get("command") != expected_command or static.get(
        "command_sha256"
    ) != _canonical_sha256(expected_command):
        raise Ds9CompatibilityError("DS9 static candidate command binding differs")
    evidence = static.get("evidence")
    if not isinstance(evidence, dict):
        raise Ds9CompatibilityError("DS9 static candidate evidence is missing")
    validated_static = validate_static_probe(evidence, image=image, controls=controls)
    # The production check is specifically sm_86, not merely forward PTX.
    if validated_static["facts"].get("sm86_cubin_present") is not True:
        raise Ds9CompatibilityError("DS9 GPU smoke candidate lacks exact sm_86 cubin")
    gpu = payload.get("gpu_smoke")
    expected_gpu = {
        "status": "pending_gpu_smoke",
        "checks": {name: "pending" for name in REQUIRED_GPU_SMOKE_CHECKS},
        "evidence": None,
    }
    if gpu != expected_gpu:
        raise Ds9CompatibilityError("DS9 static candidate GPU state was rewritten")
    return {
        "status": "static_candidate_ready_for_guarded_gpu_smoke",
        "production_ready": False,
        "receipt": receipt_pin,
        "resolved_image_id": resolved_image_id,
        "runtime_control_manifest_sha256": controls["pin"]["sha256"],
        "parser_sha256": image["labels"][LABELS["parser_sha256"]],
        "required_gpu_smoke_checks": list(REQUIRED_GPU_SMOKE_CHECKS),
    }


def require_runtime_compatibility(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    project_root: Path = PROJECT_ROOT,
    requested_image: str,
    resolved_image_id: str,
    now: datetime | None = None,
    image_inspector: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify a current production-ready receipt before any GPU process starts."""

    payload, receipt_pin = _read_json_stable(
        receipt_path, project_root=project_root, immutable=True
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise Ds9CompatibilityError("DS9 compatibility receipt schema differs")
    if payload.get("status") != "production_ready" or payload.get("production_ready") is not True:
        raise Ds9CompatibilityError(
            "DS9 compatibility is not production-ready (GPU smoke remains pending)"
        )
    created = _parse_time(payload.get("created_at_utc"), "created_at_utc")
    expires = _parse_time(payload.get("expires_at_utc"), "expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= created or expires - created > MAX_RECEIPT_AGE:
        raise Ds9CompatibilityError("DS9 compatibility receipt lifetime exceeds 24h")
    if current < created or current >= expires:
        raise Ds9CompatibilityError("DS9 compatibility receipt is not current")
    if payload.get("requested_image") != requested_image:
        raise Ds9CompatibilityError("DS9 receipt requested image differs")
    image = payload.get("image")
    if not isinstance(image, dict) or image.get("resolved_image_id") != resolved_image_id:
        raise Ds9CompatibilityError("DS9 receipt immutable image ID differs")
    if image_inspector is None:
        image_inspector = inspect_local_image
    observed_image = image_inspector(resolved_image_id)
    for key in ("resolved_image_id", "architecture", "os", "labels"):
        if observed_image.get(key) != image.get(key):
            raise Ds9CompatibilityError(
                f"live immutable image metadata differs from receipt: {key}"
            )
    controls = validate_runtime_control_manifest(project_root)
    if payload.get("runtime_controls") != controls:
        raise Ds9CompatibilityError("DS9 receipt control hashes differ from live controls")
    validate_image_lineage(image, controls=controls)
    static = payload.get("static_probe")
    if not isinstance(static, dict) or static.get("status") != "pass":
        raise Ds9CompatibilityError("DS9 static probe is not passing")
    expected_command = build_static_probe_command(resolved_image_id)
    if static.get("command") != expected_command:
        raise Ds9CompatibilityError("DS9 static probe command was coherently rewritten")
    if static.get("command_sha256") != _canonical_sha256(expected_command):
        raise Ds9CompatibilityError("DS9 static probe command pin differs")
    evidence = static.get("evidence")
    if not isinstance(evidence, dict):
        raise Ds9CompatibilityError("DS9 static probe evidence is missing")
    # Recompute semantic checks; embedded `status=pass` is never trusted alone.
    validate_static_probe(evidence, image=image, controls=controls)
    gpu = payload.get("gpu_smoke")
    if not isinstance(gpu, dict) or gpu.get("status") != "pass":
        raise Ds9CompatibilityError("DS9 GPU smoke evidence is not passing")
    checks = gpu.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_GPU_SMOKE_CHECKS):
        raise Ds9CompatibilityError("DS9 GPU smoke check set differs")
    if any(value != "pass" for value in checks.values()):
        raise Ds9CompatibilityError("DS9 GPU smoke contains a non-passing check")
    evidence_pin = gpu.get("evidence")
    if not isinstance(evidence_pin, dict):
        raise Ds9CompatibilityError("DS9 GPU smoke artifact pin is missing")
    live_gpu, live_pin = _read_json_stable(
        project_root / evidence_pin.get("path", ""),
        project_root=project_root,
        immutable=True,
    )
    if live_pin != evidence_pin:
        raise Ds9CompatibilityError("DS9 GPU smoke artifact changed after receipt")
    _load_gpu_smoke_evidence(
        project_root / evidence_pin["path"],
        project_root=project_root,
        resolved_image_id=resolved_image_id,
        controls=controls,
        expected_parser_sha256=image["labels"][LABELS["parser_sha256"]],
    )
    return {
        "status": "production_ready",
        "receipt": receipt_pin,
        "resolved_image_id": resolved_image_id,
        "runtime_control_manifest_sha256": controls["pin"]["sha256"],
        "gpu_smoke": live_gpu,
    }


def prevalidate_runtime_compatibility(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    project_root: Path = PROJECT_ROOT,
    requested_image: str,
    now: datetime | None = None,
    image_inspector: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate freshness/controls before an execution nonce or GPU lock is used.

    The definitive launch boundary must still call
    :func:`require_runtime_compatibility` with Docker's independently resolved
    image ID.  This early check deliberately trusts the receipt only for that
    ID so an invalid/expired receipt cannot consume a single-use campaign.
    """

    payload, _ = _read_json_stable(
        receipt_path, project_root=project_root, immutable=True
    )
    image = payload.get("image")
    resolved_image_id = image.get("resolved_image_id") if isinstance(image, dict) else None
    if not isinstance(resolved_image_id, str):
        raise Ds9CompatibilityError("DS9 receipt has no immutable image ID")
    inspector = image_inspector or inspect_local_image
    requested_image_observation = inspector(requested_image)
    if requested_image_observation.get("resolved_image_id") != resolved_image_id:
        raise Ds9CompatibilityError(
            "requested image tag no longer resolves to the receipt image ID"
        )
    return require_runtime_compatibility(
        receipt_path,
        project_root=project_root,
        requested_image=requested_image,
        resolved_image_id=resolved_image_id,
        now=now,
        image_inspector=inspector,
    )


def execute_static_probe(
    *,
    requested_image: str,
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
    gpu_smoke_evidence: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    output = _inside_root(output_dir, project_root)
    controls = validate_runtime_control_manifest(project_root)
    image = inspect_local_image(requested_image, runner=runner)
    validate_image_lineage(image, controls=controls)
    command = build_static_probe_command(image["resolved_image_id"])
    completed = runner(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise Ds9CompatibilityError(
            f"static container probe failed with exit code {completed.returncode}"
        )
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Ds9CompatibilityError("static container probe returned invalid JSON") from exc
    if not isinstance(probe, dict):
        raise Ds9CompatibilityError("static container probe must return an object")
    receipt = create_static_receipt(
        requested_image=requested_image,
        image=image,
        probe_command=command,
        probe=probe,
        project_root=project_root,
        gpu_smoke_evidence=gpu_smoke_evidence,
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "static-probe.json", probe)
    _exclusive_receipt(output / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-static-probe", action="store_true")
    mode.add_argument("--inside-container-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--image", default="deepsafe-deepstream:9.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--gpu-smoke-evidence", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.inside_container_probe:
        print(json.dumps(_inside_container_probe(), ensure_ascii=False, sort_keys=True))
        return 0
    project_root = args.project_root.resolve()
    output_dir = _inside_root(args.output_dir, project_root)
    if not args.execute_static_probe:
        pending = write_pending_report(
            output_dir / "pending-report.json",
            requested_image=args.image,
            project_root=project_root,
            launch_scope="standalone_static_compatibility_probe",
        )
        print(json.dumps(pending, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        receipt = execute_static_probe(
            requested_image=args.image,
            output_dir=output_dir,
            project_root=project_root,
            gpu_smoke_evidence=args.gpu_smoke_evidence,
        )
    except (Ds9CompatibilityError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["production_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
