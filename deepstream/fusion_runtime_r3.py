"""Descriptor-bound, no-overwrite republication of the DS9 fusion R2 bundle.

R1/R2 remain immutable historical publications.  R3 republishes the exact R2
runtime binaries with a new provenance/capability pair and an inode-bound
publication receipt.  The stage directory FD remains open across renameat2;
the destination and canonical path are then reopened and replayed relative to
directory FDs before publication is accepted.

This module never starts Docker, queries a GPU, loads a model, or runs
inference.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepstream import fusion_runtime as r1
from deepstream import fusion_runtime_r2 as r2
from validation.strict_json import StrictJSONError, loads as strict_json_loads


PLAN_SCHEMA_VERSION = "deepsafe.deepstream-fusion-publication-plan/v1"
RECEIPT_SCHEMA_VERSION = "deepsafe.deepstream-fusion-publication-receipt/v1"
PROVENANCE_SCHEMA_VERSION = "deepsafe.deepstream-fusion-runtime-provenance/v3"
DEFAULT_PLAN = PROJECT_ROOT / "deepstream/fusion-runtime-r3-publication-plan.json"
FROZEN_PLAN_SHA256 = "5b14391a10f1bcd2a7500abb165de5d45002845df4262db720158ad6bb9a9d8e"
EXPECTED_DESTINATION = "models/runtime/deepsafe-fusion-ds9-9946965e-r3"
RECEIPT_NAME = "publication-receipt.json"
PRIMITIVE = (
    "renameat2(RENAME_NOREPLACE)+post_rename_fd_identity+"
    "descriptor_relative_replay"
)
SOURCE_NAMES = {
    "build-provenance.json",
    "capability-manifest.json",
    "deepstream-parallel-infer",
    "fusion-runtime.conf",
    "libdeepsafe_fusion.so.1",
}
PUBLISHED_NAMES = SOURCE_NAMES | {RECEIPT_NAME}
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class FusionRuntimeR3Error(RuntimeError):
    """The R3 publication plan, source bundle, receipt, or replay was invalid."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _exact(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise FusionRuntimeR3Error(f"{where} keys mismatch")
    return value


def _strict_json(content: bytes, where: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(content)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FusionRuntimeR3Error(f"invalid strict JSON in {where}: {exc}") from exc
    if not isinstance(value, dict):
        raise FusionRuntimeR3Error(f"{where} root must be an object")
    return value


def _relative(value: Any, where: str) -> Path:
    try:
        return r1._relative(value, where)
    except r1.FusionRuntimeError as exc:
        raise FusionRuntimeR3Error(str(exc)) from exc


def _read(path: Path, maximum: int) -> bytes:
    try:
        return r1._read(path, maximum=maximum)
    except r1.FusionRuntimeError as exc:
        raise FusionRuntimeR3Error(str(exc)) from exc


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.lstat().st_mode):04o}"


def load_frozen_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
    content = _read(path, MAX_JSON_BYTES)
    digest = _sha256(content)
    if digest != FROZEN_PLAN_SHA256:
        raise FusionRuntimeR3Error("R3 publication plan SHA-256 mismatch")
    plan = _strict_json(content, "R3 publication plan")
    _validate_plan(plan)
    return plan, digest


def _validate_plan(value: Any) -> dict[str, Any]:
    plan = _exact(
        value,
        {"schema_version", "plan_id", "source_publication", "publisher", "publication", "safety"},
        "plan",
    )
    if (
        plan["schema_version"] != PLAN_SCHEMA_VERSION
        or plan["plan_id"]
        != "deepsafe-ds9-canonical-fusion-inode-bound-publication-r3"
    ):
        raise FusionRuntimeR3Error("unsupported R3 publication plan")
    source = _exact(
        plan["source_publication"],
        {
            "path",
            "build_plan_sha256",
            "provenance_sha256",
            "capability_manifest_sha256",
            "files",
        },
        "plan.source_publication",
    )
    if source["path"] != r2.EXPECTED_DESTINATION:
        raise FusionRuntimeR3Error("R3 source must be the exact R2 publication")
    if source["build_plan_sha256"] != r2.FROZEN_PLAN_SHA256:
        raise FusionRuntimeR3Error("R2 build-plan pin drifted")
    files = source["files"]
    if not isinstance(files, list) or len(files) != len(SOURCE_NAMES):
        raise FusionRuntimeR3Error("R3 source file list mismatch")
    names: list[str] = []
    for index, raw in enumerate(files):
        item = _exact(raw, {"name", "sha256", "bytes", "mode"}, f"source file {index}")
        name = item["name"]
        if name not in SOURCE_NAMES or not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool):
            raise FusionRuntimeR3Error("invalid R3 source file entry")
        if item["bytes"] < 1 or item["mode"] not in {"0440", "0550"}:
            raise FusionRuntimeR3Error("invalid R3 source file size/mode")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise FusionRuntimeR3Error("invalid R3 source file hash")
        names.append(name)
    if names != sorted(SOURCE_NAMES):
        raise FusionRuntimeR3Error("R3 source files must be exact and sorted")
    publisher = _exact(
        plan["publisher"],
        {
            "receipt_schema_version",
            "directory_identity_bound",
            "post_rename_inode_verified",
            "descriptor_relative_artifact_replay",
            "exact_file_set_verified",
            "canonical_path_reopened",
        },
        "plan.publisher",
    )
    if publisher != {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "directory_identity_bound": True,
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
    }:
        raise FusionRuntimeR3Error("R3 publisher requirements drifted")
    publication = _exact(
        plan["publication"],
        {
            "destination",
            "no_overwrite",
            "primitive",
            "receipt_name",
            "directory_mode",
            "executable_mode",
            "config_mode",
            "json_mode",
        },
        "plan.publication",
    )
    if publication != {
        "destination": EXPECTED_DESTINATION,
        "no_overwrite": True,
        "primitive": PRIMITIVE,
        "receipt_name": RECEIPT_NAME,
        "directory_mode": "0550",
        "executable_mode": "0550",
        "config_mode": "0440",
        "json_mode": "0440",
    }:
        raise FusionRuntimeR3Error("R3 publication contract drifted")
    safety = _exact(
        plan["safety"],
        {
            "artifact_rebuild",
            "container_started",
            "network_used",
            "gpu_device_injected",
            "gpu_queried",
            "inference_executed",
            "model_or_engine_loaded",
            "endurance_executed",
        },
        "plan.safety",
    )
    if any(value is not False for value in safety.values()):
        raise FusionRuntimeR3Error("R3 republication safety flags must all be false")
    return plan


def _source_files(plan: dict[str, Any], project_root: Path) -> dict[str, tuple[bytes, int]]:
    inspected = r2.inspect_publication(project_root=project_root)
    source = plan["source_publication"]
    if (
        inspected["build_plan_sha256"] != source["build_plan_sha256"]
        or inspected["provenance_sha256"] != source["provenance_sha256"]
        or inspected["capability_manifest_sha256"]
        != source["capability_manifest_sha256"]
    ):
        raise FusionRuntimeR3Error("live R2 inspector pins differ from R3 plan")
    root = project_root / _relative(source["path"], "R2 source publication")
    result: dict[str, tuple[bytes, int]] = {}
    for item in source["files"]:
        path = root / item["name"]
        content = _read(path, MAX_ARTIFACT_BYTES)
        if (
            len(content) != item["bytes"]
            or _sha256(content) != item["sha256"]
            or _mode(path) != item["mode"]
        ):
            raise FusionRuntimeR3Error(f"live R2 source drifted: {item['name']}")
        result[item["name"]] = (content, int(item["mode"], 8))
    return result


def _open_chain(root: Path, relative: Path, *, create: bool) -> int:
    descriptor = os.open(root, DIRECTORY_FLAGS)
    try:
        for part in relative.parts:
            if not part or part in {".", ".."}:
                raise FusionRuntimeR3Error("unsafe directory component")
            if create:
                try:
                    os.mkdir(part, mode=0o750, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise FusionRuntimeR3Error("publication chain component is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_file(directory_fd: int, name: str, content: bytes, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FusionRuntimeR3Error("short publication write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_fd_file(directory_fd: int, name: str, maximum: int) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(name, FILE_FLAGS, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1:
            raise FusionRuntimeR3Error(f"published entry is not an isolated regular file: {name}")
        if before.st_size > maximum:
            raise FusionRuntimeR3Error(f"published entry exceeds size bound: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise FusionRuntimeR3Error(f"published entry exceeds size bound: {name}")
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in identity):
            raise FusionRuntimeR3Error(f"published entry changed during replay: {name}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _rename_noreplace(parent_fd: int, source_name: str, target_name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise FusionRuntimeR3Error("renameat2 unavailable; refusing weaker publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(target_name),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FusionRuntimeR3Error("publication destination already exists")
        raise FusionRuntimeR3Error(f"renameat2 publication failed: {os.strerror(error)}")


def _directory_identity(info: os.stat_result) -> dict[str, int]:
    return {"device": info.st_dev, "inode": info.st_ino}


def _file_manifest(files: dict[str, tuple[bytes, int]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": _sha256(content),
            "size_bytes": len(content),
            "mode": f"{mode:04o}",
        }
        for name, (content, mode) in sorted(files.items())
    }


def _receipt_value(
    *,
    plan: dict[str, Any],
    plan_sha256: str,
    builder_sha256: str,
    directory_info: os.stat_result,
    files: dict[str, tuple[bytes, int]],
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "published",
        "publication_id": plan["plan_id"],
        "destination": plan["publication"]["destination"],
        "publication_plan_sha256": plan_sha256,
        "publisher_sha256": builder_sha256,
        "primitive": PRIMITIVE,
        "directory_identity": _directory_identity(directory_info),
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
        "exact_file_names": sorted(PUBLISHED_NAMES),
        "files": _file_manifest(files),
    }


def _validate_receipt(value: Any, plan: dict[str, Any], plan_sha256: str, builder_sha256: str) -> dict[str, Any]:
    receipt = _exact(
        value,
        {
            "schema_version",
            "status",
            "publication_id",
            "destination",
            "publication_plan_sha256",
            "publisher_sha256",
            "primitive",
            "directory_identity",
            "post_rename_inode_verified",
            "descriptor_relative_artifact_replay",
            "exact_file_set_verified",
            "canonical_path_reopened",
            "exact_file_names",
            "files",
        },
        "publication receipt",
    )
    expected = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "published",
        "publication_id": plan["plan_id"],
        "destination": plan["publication"]["destination"],
        "publication_plan_sha256": plan_sha256,
        "publisher_sha256": builder_sha256,
        "primitive": PRIMITIVE,
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
        "exact_file_names": sorted(PUBLISHED_NAMES),
    }
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        raise FusionRuntimeR3Error("publication receipt claim mismatch")
    identity = _exact(receipt["directory_identity"], {"device", "inode"}, "receipt directory identity")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in identity.values()):
        raise FusionRuntimeR3Error("invalid receipt directory identity")
    if set(receipt["files"]) != SOURCE_NAMES:
        raise FusionRuntimeR3Error("receipt file manifest mismatch")
    for name, raw in receipt["files"].items():
        item = _exact(raw, {"sha256", "size_bytes", "mode"}, f"receipt file {name}")
        if (
            not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or not isinstance(item["size_bytes"], int)
            or isinstance(item["size_bytes"], bool)
            or item["size_bytes"] < 1
            or item["mode"] not in {"0440", "0550"}
        ):
            raise FusionRuntimeR3Error("invalid receipt file metadata")
    return receipt


def _replay_directory(
    directory_fd: int,
    *,
    expected_identity: dict[str, int],
    expected_files: dict[str, tuple[bytes, int]],
) -> None:
    before = os.fstat(directory_fd)
    if not stat.S_ISDIR(before.st_mode) or _directory_identity(before) != expected_identity:
        raise FusionRuntimeR3Error("post-rename destination inode differs from stage FD")
    if stat.S_IMODE(before.st_mode) != 0o550:
        raise FusionRuntimeR3Error("published directory mode drifted")
    if sorted(os.listdir(directory_fd)) != sorted(expected_files):
        raise FusionRuntimeR3Error("published exact file set drifted")
    for name, (expected, mode) in sorted(expected_files.items()):
        actual, info = _read_fd_file(directory_fd, name, MAX_ARTIFACT_BYTES)
        if actual != expected or stat.S_IMODE(info.st_mode) != mode:
            raise FusionRuntimeR3Error(f"descriptor-relative replay drifted: {name}")
    if sorted(os.listdir(directory_fd)) != sorted(expected_files):
        raise FusionRuntimeR3Error("published exact file set changed during replay")
    after = os.fstat(directory_fd)
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise FusionRuntimeR3Error("published directory changed during replay")


def _cleanup_owned_stage(parent_fd: int, name: str, identity: dict[str, int]) -> None:
    try:
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return
    try:
        if _directory_identity(os.fstat(descriptor)) != identity:
            return
        os.fchmod(descriptor, 0o700)
        for entry in os.listdir(descriptor):
            try:
                os.unlink(entry, dir_fd=descriptor)
            except OSError:
                return
    finally:
        os.close(descriptor)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        pass


def _publish(
    plan: dict[str, Any],
    *,
    project_root: Path,
    plan_sha256: str,
    builder_sha256: str,
    files: dict[str, tuple[bytes, int]],
) -> tuple[Path, bytes]:
    if set(files) != SOURCE_NAMES:
        raise FusionRuntimeR3Error("R3 pre-receipt publication file set drifted")
    destination = _relative(plan["publication"]["destination"], "R3 destination")
    project_root = project_root.resolve(strict=True)
    parent_fd = _open_chain(project_root, destination.parent, create=True)
    stage_name = ""
    stage_fd = -1
    stage_identity: dict[str, int] = {}
    renamed = False
    try:
        for _ in range(16):
            candidate = f".fusion-runtime-r3-stage-{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            stage_name = candidate
            stage_fd = os.open(stage_name, DIRECTORY_FLAGS, dir_fd=parent_fd)
            break
        if stage_fd < 0:
            raise FusionRuntimeR3Error("could not allocate exclusive R3 stage directory")
        for name, (content, mode) in sorted(files.items()):
            _write_file(stage_fd, name, content, mode)
        directory_info = os.fstat(stage_fd)
        stage_identity = _directory_identity(directory_info)
        receipt_value = _receipt_value(
            plan=plan,
            plan_sha256=plan_sha256,
            builder_sha256=builder_sha256,
            directory_info=directory_info,
            files=files,
        )
        receipt = _json_bytes(receipt_value)
        all_files = dict(files)
        all_files[RECEIPT_NAME] = (receipt, 0o440)
        _write_file(stage_fd, RECEIPT_NAME, receipt, 0o440)
        os.fchmod(stage_fd, 0o550)
        os.fsync(stage_fd)

        _rename_noreplace(parent_fd, stage_name, destination.name)
        renamed = True
        target_fd = os.open(destination.name, DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            if _directory_identity(os.fstat(target_fd)) != stage_identity:
                raise FusionRuntimeR3Error("post-rename destination inode differs from held stage FD")
            _replay_directory(
                target_fd,
                expected_identity=stage_identity,
                expected_files=all_files,
            )
        finally:
            os.close(target_fd)
        os.fsync(parent_fd)

        canonical_parent_fd = _open_chain(project_root, destination.parent, create=False)
        try:
            canonical_fd = os.open(destination.name, DIRECTORY_FLAGS, dir_fd=canonical_parent_fd)
            try:
                if _directory_identity(os.fstat(canonical_fd)) != stage_identity:
                    raise FusionRuntimeR3Error("canonical destination path does not resolve to published inode")
                _replay_directory(
                    canonical_fd,
                    expected_identity=stage_identity,
                    expected_files=all_files,
                )
            finally:
                os.close(canonical_fd)
        finally:
            os.close(canonical_parent_fd)
        return project_root / destination, receipt
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        if stage_name and not renamed and stage_identity:
            _cleanup_owned_stage(parent_fd, stage_name, stage_identity)
        os.close(parent_fd)


def _publication_contract(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_name": RECEIPT_NAME,
        "primitive": PRIMITIVE,
        "directory_identity_bound": True,
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
    }


def publish_runtime(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    builder_path = Path(__file__).resolve(strict=True)
    builder_snapshot = _read(builder_path, MAX_JSON_BYTES)
    builder_sha256 = _sha256(builder_snapshot)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    source_files = _source_files(plan, project_root)

    source_manifest = _file_manifest(source_files)
    runtime_artifacts = {
        name: source_manifest[name]
        for name in (
            "deepstream-parallel-infer",
            "fusion-runtime.conf",
            "libdeepsafe_fusion.so.1",
        )
    }
    provenance_value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "descriptor_bound_republication_passed",
        "publication_plan_sha256": plan_sha256,
        "publisher": {
            "path": "deepstream/fusion_runtime_r3.py",
            "sha256": builder_sha256,
        },
        "source_publication": {
            "path": plan["source_publication"]["path"],
            "build_plan_sha256": plan["source_publication"]["build_plan_sha256"],
            "provenance_sha256": plan["source_publication"]["provenance_sha256"],
            "capability_manifest_sha256": plan["source_publication"]["capability_manifest_sha256"],
            "files": source_manifest,
        },
        "publication_contract": _publication_contract(plan),
        "artifacts": runtime_artifacts,
        "safety": dict(plan["safety"]),
    }
    provenance = _json_bytes(provenance_value)
    source_capability = _strict_json(
        source_files["capability-manifest.json"][0], "R2 capability manifest"
    )
    capability_value = copy.deepcopy(source_capability)
    capability_value["build_provenance_sha256"] = _sha256(provenance)
    capability_value["publication_contract"] = _publication_contract(plan)
    static_evidence = capability_value.get("static_evidence")
    if not isinstance(static_evidence, dict):
        raise FusionRuntimeR3Error("R2 capability static evidence is missing")
    static_evidence["descriptor_bound_publication_receipt"] = True
    capability = _json_bytes(capability_value)
    files = {
        "deepstream-parallel-infer": source_files["deepstream-parallel-infer"],
        "libdeepsafe_fusion.so.1": source_files["libdeepsafe_fusion.so.1"],
        "fusion-runtime.conf": source_files["fusion-runtime.conf"],
        "build-provenance.json": (provenance, 0o440),
        "capability-manifest.json": (capability, 0o440),
    }
    if _read(builder_path, MAX_JSON_BYTES) != builder_snapshot:
        raise FusionRuntimeR3Error("R3 publisher changed during transaction")
    published, receipt = _publish(
        plan,
        project_root=project_root,
        plan_sha256=plan_sha256,
        builder_sha256=builder_sha256,
        files=files,
    )
    return {
        "published": published.relative_to(project_root).as_posix(),
        "publication_plan_sha256": plan_sha256,
        "publisher_sha256": builder_sha256,
        "source_build_plan_sha256": plan["source_publication"]["build_plan_sha256"],
        "artifacts": runtime_artifacts,
        "provenance_sha256": _sha256(provenance),
        "capability_manifest_sha256": _sha256(capability),
        "publication_receipt_sha256": _sha256(receipt),
        "publication_primitive": PRIMITIVE,
        "gpu_integration_validated": False,
        "runtime_ready": False,
    }


def _inspect_directory(
    plan: dict[str, Any], plan_sha256: str, builder_sha256: str, project_root: Path
) -> tuple[dict[str, Any], dict[str, bytes]]:
    destination = _relative(plan["publication"]["destination"], "R3 destination")
    parent_fd = _open_chain(project_root, destination.parent, create=False)
    try:
        directory_fd = os.open(destination.name, DIRECTORY_FLAGS, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        directory_before = os.fstat(directory_fd)
        receipt_content, receipt_info = _read_fd_file(directory_fd, RECEIPT_NAME, MAX_JSON_BYTES)
        if stat.S_IMODE(receipt_info.st_mode) != 0o440:
            raise FusionRuntimeR3Error("publication receipt mode drifted")
        receipt = _validate_receipt(
            _strict_json(receipt_content, "publication receipt"),
            plan,
            plan_sha256,
            builder_sha256,
        )
        if _directory_identity(os.fstat(directory_fd)) != receipt["directory_identity"]:
            raise FusionRuntimeR3Error("live publication directory inode differs from receipt")
        if sorted(os.listdir(directory_fd)) != sorted(PUBLISHED_NAMES):
            raise FusionRuntimeR3Error("live R3 exact file set drifted")
        snapshots: dict[str, bytes] = {RECEIPT_NAME: receipt_content}
        for name, expected in sorted(receipt["files"].items()):
            content, info = _read_fd_file(directory_fd, name, MAX_ARTIFACT_BYTES)
            if (
                _sha256(content) != expected["sha256"]
                or len(content) != expected["size_bytes"]
                or f"{stat.S_IMODE(info.st_mode):04o}" != expected["mode"]
            ):
                raise FusionRuntimeR3Error(f"live R3 receipt binding drifted: {name}")
            snapshots[name] = content
        if sorted(os.listdir(directory_fd)) != sorted(PUBLISHED_NAMES):
            raise FusionRuntimeR3Error("live R3 exact file set changed during replay")
        directory_after = os.fstat(directory_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(directory_before, field) != getattr(directory_after, field)
            for field in stable_fields
        ):
            raise FusionRuntimeR3Error("live R3 directory changed during replay")
        return receipt, snapshots
    finally:
        os.close(directory_fd)


def inspect_publication(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    builder_sha256 = _sha256(_read(Path(__file__).resolve(strict=True), MAX_JSON_BYTES))
    _source_files(plan, project_root)
    receipt, snapshots = _inspect_directory(
        plan, plan_sha256, builder_sha256, project_root
    )
    provenance = _strict_json(snapshots["build-provenance.json"], "R3 provenance")
    capability = _strict_json(snapshots["capability-manifest.json"], "R3 capability")
    if (
        provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or provenance.get("status") != "descriptor_bound_republication_passed"
        or provenance.get("publication_plan_sha256") != plan_sha256
        or provenance.get("publisher")
        != {"path": "deepstream/fusion_runtime_r3.py", "sha256": builder_sha256}
        or provenance.get("publication_contract") != _publication_contract(plan)
        or provenance.get("safety") != plan["safety"]
    ):
        raise FusionRuntimeR3Error("R3 provenance binding drifted")
    if (
        capability.get("build_provenance_sha256")
        != _sha256(snapshots["build-provenance.json"])
        or capability.get("publication_contract") != _publication_contract(plan)
        or capability.get("static_evidence", {}).get(
            "descriptor_bound_publication_receipt"
        )
        is not True
        or capability.get("fusion_plugin_ready") is not True
        or capability.get("gpu_integration_validated") is not False
        or capability.get("runtime_ready") is not False
        or capability.get("status") != "blocked"
    ):
        raise FusionRuntimeR3Error("R3 capability binding drifted")
    source = {item["name"]: item for item in plan["source_publication"]["files"]}
    for name in ("deepstream-parallel-infer", "fusion-runtime.conf", "libdeepsafe_fusion.so.1"):
        if _sha256(snapshots[name]) != source[name]["sha256"]:
            raise FusionRuntimeR3Error(f"R3 runtime artifact differs from R2: {name}")
    return {
        "published": plan["publication"]["destination"],
        "publication_plan_sha256": plan_sha256,
        "publisher_sha256": builder_sha256,
        "source_build_plan_sha256": plan["source_publication"]["build_plan_sha256"],
        "provenance_sha256": _sha256(snapshots["build-provenance.json"]),
        "capability_manifest_sha256": _sha256(snapshots["capability-manifest.json"]),
        "publication_receipt_sha256": _sha256(snapshots[RECEIPT_NAME]),
        "directory_identity": receipt["directory_identity"],
        "publication_primitive": receipt["primitive"],
        "post_rename_inode_verified": True,
        "descriptor_relative_artifact_replay": True,
        "exact_file_set_verified": True,
        "canonical_path_reopened": True,
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "runtime_ready": False,
    }


def verify_prerequisites(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    files = _source_files(plan, project_root)
    return {
        "publication_plan_sha256": plan_sha256,
        "source_files_verified": len(files),
        "source_build_plan_sha256": plan["source_publication"]["build_plan_sha256"],
        "publication_primitive": PRIMITIVE,
        **plan["safety"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("publish")
    commands.add_parser("inspect")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_prerequisites(plan_path=args.plan)
        elif args.command == "publish":
            result = publish_runtime(plan_path=args.plan)
        else:
            result = inspect_publication(plan_path=args.plan)
    except (FusionRuntimeR3Error, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
