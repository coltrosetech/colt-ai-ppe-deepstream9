"""Auditable, CPU-only builder for NVIDIA's DS9 parallel-inference app.

The builder reads every upstream build input directly from a frozen Git object,
uses an immutable local DeepStream image with networking and GPU injection
disabled, performs static ELF/ABI probes, and publishes one immutable directory
with ``renameat2(RENAME_NOREPLACE)``.  It deliberately does not run the built
application, inference, a model, a TensorRT engine, or an endurance workload.

The upstream program does not consume ``DEEPSAFE_FUSION_CONFIG`` and does not
implement DeepSafe's pose/PPE-to-person association boundary.  Consequently a
successful build still emits a fail-closed capability manifest whose status is
``blocked`` and whose two association features are false.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.strict_json import StrictJSONError, loads as strict_json_loads


PLAN_SCHEMA_VERSION = "deepsafe.deepstream-parallel-runtime-build-plan/v1"
PROVENANCE_SCHEMA_VERSION = "deepsafe.deepstream-parallel-runtime-provenance/v1"
CAPABILITY_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-runtime-capabilities/v1"
DEFAULT_PLAN = PROJECT_ROOT / "deepstream/parallel-runtime-build-plan.json"
FROZEN_PLAN_SHA256 = "ea22142ac35a10c84f89f5d452153b95379495f258a8533ff432bd267f4310d1"
EXPECTED_COMMIT = "9946965e8adb1aa93b1b66983ec4196351c9190c"
EXPECTED_TREE = "4c4382ed32fad08767d01cd7bfbf464bd6be0e37"
EXPECTED_IMAGE = (
    "deepsafe-deepstream@"
    "sha256:96aedaba7ebb8d50359a7f73db251d46a81fd23e42c7c7ae215542795f88d663"
)
EXPECTED_IMAGE_ID = (
    "sha256:96aedaba7ebb8d50359a7f73db251d46a81fd23e42c7c7ae215542795f88d663"
)
PARALLEL_PATTERN = "nvidia_parallel_inference_nvdsmetamux"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_PLAN_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 32 * 1024 * 1024


class ParallelRuntimeError(RuntimeError):
    """The frozen source, image, build, probe, or publication is invalid."""


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParallelRuntimeError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise ParallelRuntimeError(
            f"{where} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _normalized_relative(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ParallelRuntimeError(f"{where} must be a normalized relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(SAFE_PART_RE.fullmatch(part) is None for part in path.parts)
    ):
        raise ParallelRuntimeError(f"{where} must be a normalized relative path")
    return path


def _read_stable_regular(path: Path, *, max_bytes: int) -> bytes:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise ParallelRuntimeError(f"cannot stat {path}: {exc}") from exc
    if (
        not stat.S_ISREG(initial.st_mode)
        or stat.S_ISLNK(initial.st_mode)
        or initial.st_nlink != 1
        or initial.st_size <= 0
        or initial.st_size > max_bytes
    ):
        raise ParallelRuntimeError(f"unsafe or out-of-range regular file: {path}")
    try:
        if path.resolve(strict=True) != path.absolute():
            raise ParallelRuntimeError(f"path contains a symlink: {path}")
    except OSError as exc:
        raise ParallelRuntimeError(f"cannot resolve {path}: {exc}") from exc

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    identity = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    try:
        before = os.fstat(descriptor)
        if any(getattr(initial, key) != getattr(before, key) for key in identity):
            raise ParallelRuntimeError(f"file changed before snapshot: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ParallelRuntimeError(f"file exceeds byte limit: {path}")
        after = os.fstat(descriptor)
        current = path.lstat()
        if any(
            getattr(before, key) != getattr(after, key)
            or getattr(before, key) != getattr(current, key)
            for key in identity
        ):
            raise ParallelRuntimeError(f"file changed during snapshot: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise ParallelRuntimeError(f"cannot snapshot {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _template_sha256(argv: Iterable[str]) -> str:
    return _sha256(_json_bytes(list(argv)))


def load_frozen_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
    raw = _read_stable_regular(path, max_bytes=MAX_PLAN_BYTES)
    digest = _sha256(raw)
    if digest != FROZEN_PLAN_SHA256:
        raise ParallelRuntimeError(
            f"build plan SHA-256 mismatch: expected {FROZEN_PLAN_SHA256}, observed {digest}"
        )
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise ParallelRuntimeError(f"invalid strict build plan JSON: {exc}") from exc
    return validate_plan(value), digest


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _exact_keys(
        value,
        {"schema_version", "plan_id", "source", "container", "build", "probes", "publication"},
        "plan",
    )
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ParallelRuntimeError("unsupported build plan schema")
    if plan["plan_id"] != "nvidia-deepstream-parallel-inference-ds9-static-build-r1":
        raise ParallelRuntimeError("unexpected build plan id")

    source = _exact_keys(
        plan["source"],
        {"repository_path", "commit", "tree", "build_root", "license", "files"},
        "plan.source",
    )
    _normalized_relative(source["repository_path"], "plan.source.repository_path")
    _normalized_relative(source["build_root"], "plan.source.build_root")
    if source["commit"] != EXPECTED_COMMIT or source["tree"] != EXPECTED_TREE:
        raise ParallelRuntimeError("source commit/tree is not the frozen NVIDIA checkout")
    license_value = _exact_keys(
        source["license"],
        {
            "spdx",
            "repository_license_path",
            "repository_license_git_blob",
            "repository_license_sha256",
            "app_license_path",
            "app_license_git_blob",
            "app_license_sha256",
        },
        "plan.source.license",
    )
    if license_value["spdx"] != "Apache-2.0":
        raise ParallelRuntimeError("upstream license must remain Apache-2.0")
    for prefix in ("repository_license", "app_license"):
        _normalized_relative(license_value[f"{prefix}_path"], f"plan.source.license.{prefix}_path")
        if GIT_OID_RE.fullmatch(str(license_value[f"{prefix}_git_blob"])) is None:
            raise ParallelRuntimeError(f"invalid {prefix} Git blob")
        if SHA256_RE.fullmatch(str(license_value[f"{prefix}_sha256"])) is None:
            raise ParallelRuntimeError(f"invalid {prefix} SHA-256")

    files = source["files"]
    if not isinstance(files, list) or len(files) != 8:
        raise ParallelRuntimeError("exactly eight frozen build inputs are required")
    paths: list[str] = []
    for index, item_value in enumerate(files):
        item = _exact_keys(
            item_value,
            {"path", "mode", "git_blob", "sha256", "bytes"},
            f"plan.source.files[{index}]",
        )
        path = _normalized_relative(item["path"], f"plan.source.files[{index}].path")
        if not path.is_relative_to(Path(source["build_root"])):
            raise ParallelRuntimeError("build input escapes frozen source root")
        if item["mode"] not in {"100644", "100755"}:
            raise ParallelRuntimeError("unsupported upstream Git file mode")
        if GIT_OID_RE.fullmatch(str(item["git_blob"])) is None:
            raise ParallelRuntimeError("invalid build input Git blob")
        if SHA256_RE.fullmatch(str(item["sha256"])) is None:
            raise ParallelRuntimeError("invalid build input SHA-256")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or not 1 <= item["bytes"] <= MAX_SOURCE_BYTES:
            raise ParallelRuntimeError("invalid build input byte count")
        paths.append(path.as_posix())
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ParallelRuntimeError("build inputs must be unique and sorted")

    container = _exact_keys(
        plan["container"],
        {"image_reference", "image_id", "architecture", "operating_system"},
        "plan.container",
    )
    if (
        container["image_reference"] != EXPECTED_IMAGE
        or container["image_id"] != EXPECTED_IMAGE_ID
        or container["architecture"] != "amd64"
        or container["operating_system"] != "linux"
    ):
        raise ParallelRuntimeError("container identity is not the frozen local DS9 image")

    build = _exact_keys(
        plan["build"],
        {"binary_name", "docker_argv_template", "network", "nvidia_container_runtime", "source_date_epoch", "uses_gpu"},
        "plan.build",
    )
    if build["binary_name"] != "deepstream-parallel-infer":
        raise ParallelRuntimeError("unexpected output binary name")
    if (
        build["network"] != "none"
        or build["nvidia_container_runtime"] is not False
        or build["source_date_epoch"] != 0
        or build["uses_gpu"] is not False
    ):
        raise ParallelRuntimeError("build must be networkless and GPU-free")
    _validate_docker_template(build["docker_argv_template"], build=True)

    probes = _exact_keys(plan["probes"], {"docker_common_argv_template", "tools"}, "plan.probes")
    _validate_docker_template(probes["docker_common_argv_template"], build=False)
    tools = _exact_keys(probes["tools"], {"readelf", "objdump", "ldd", "strings"}, "plan.probes.tools")
    for name, argv in tools.items():
        if not isinstance(argv, list) or len(argv) < 2 or any(not isinstance(token, str) or not token for token in argv):
            raise ParallelRuntimeError(f"invalid {name} probe argv")
        if argv[0] != f"/usr/bin/{name}":
            raise ParallelRuntimeError(f"unexpected {name} tool path")
        if argv[-1] != "/probe/deepstream-parallel-infer":
            raise ParallelRuntimeError(f"{name} must probe the mounted binary")

    publication = _exact_keys(
        plan["publication"],
        {"destination", "directory_mode", "executable_mode", "json_mode", "no_overwrite", "primitive"},
        "plan.publication",
    )
    _normalized_relative(publication["destination"], "plan.publication.destination")
    if publication != {
        "destination": "models/runtime/deepstream-parallel-infer-ds9-9946965e-r2",
        "directory_mode": "0550",
        "executable_mode": "0550",
        "json_mode": "0440",
        "no_overwrite": True,
        "primitive": "renameat2(RENAME_NOREPLACE)",
    }:
        raise ParallelRuntimeError("publication policy drifted")
    return plan


def _validate_docker_template(value: Any, *, build: bool) -> None:
    if not isinstance(value, list) or any(not isinstance(token, str) or not token for token in value):
        raise ParallelRuntimeError("Docker argv template must be a non-empty string list")
    required = {
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--runtime=runc",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=NVIDIA_DRIVER_CAPABILITIES=none",
        "--env=CUDA_VISIBLE_DEVICES=-1",
    }
    if not build:
        required.add("--read-only")
    if not required.issubset(set(value)):
        raise ParallelRuntimeError("Docker argv omits a required network/GPU isolation token")
    forbidden_prefixes = ("--gpus", "--device", "--runtime=nvidia")
    if any(token.startswith(forbidden_prefixes) for token in value):
        raise ParallelRuntimeError("Docker argv requests a device or NVIDIA runtime")
    placeholder = "{source_dir}" if build else "{binary_path}"
    if sum(token.count(placeholder) for token in value) != 1:
        raise ParallelRuntimeError("Docker argv placeholder count drifted")
    if build and sum(token.count("{output_dir}") for token in value) != 1:
        raise ParallelRuntimeError("build output placeholder count drifted")


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[bytes]:
    if any(not isinstance(token, str) or not token or "\x00" in token for token in argv):
        raise ParallelRuntimeError("refusing malformed subprocess argv")
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ParallelRuntimeError(f"command failed to execute: {argv[0]}: {exc}") from exc
    if len(result.stdout) > MAX_COMMAND_OUTPUT_BYTES or len(result.stderr) > MAX_COMMAND_OUTPUT_BYTES:
        raise ParallelRuntimeError(f"command output exceeded audit limit: {argv[0]}")
    if result.returncode != 0:
        tail = result.stderr[-2000:].decode("utf-8", "replace")
        raise ParallelRuntimeError(
            f"command exited {result.returncode}: {argv[0]}; stderr tail: {tail}"
        )
    return result


def _git(repo: Path, args: list[str], *, timeout: int = 30) -> bytes:
    return _run(["git", "-C", str(repo), *args], timeout=timeout).stdout


def _repo_path(plan: dict[str, Any], project_root: Path) -> Path:
    relative = _normalized_relative(plan["source"]["repository_path"], "repository_path")
    path = project_root / relative
    try:
        if path.resolve(strict=True) != path.absolute():
            raise ParallelRuntimeError("source repository path contains a symlink")
        path.resolve(strict=True).relative_to(project_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ParallelRuntimeError(f"source repository path is unsafe: {exc}") from exc
    if not path.is_dir():
        raise ParallelRuntimeError("source repository is missing")
    return path


def verify_source(plan: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    repo = _repo_path(plan, project_root)
    source = plan["source"]
    head = _git(repo, ["rev-parse", "HEAD"]).decode().strip()
    tree = _git(repo, ["rev-parse", "HEAD^{tree}"]).decode().strip()
    if head != source["commit"] or tree != source["tree"]:
        raise ParallelRuntimeError("checked-out NVIDIA source HEAD/tree drifted")

    snapshots: dict[str, bytes] = {}
    entries = list(source["files"])
    license_value = source["license"]
    for prefix in ("repository_license", "app_license"):
        entries.append(
            {
                "path": license_value[f"{prefix}_path"],
                "git_blob": license_value[f"{prefix}_git_blob"],
                "sha256": license_value[f"{prefix}_sha256"],
                "bytes": None,
                "mode": None,
            }
        )

    for item in entries:
        path = item["path"]
        observed_blob = _git(repo, ["rev-parse", f"{source['commit']}:{path}"]).decode().strip()
        if observed_blob != item["git_blob"]:
            raise ParallelRuntimeError(f"Git blob drifted: {path}")
        content = _git(repo, ["cat-file", "blob", observed_blob])
        if _sha256(content) != item["sha256"]:
            raise ParallelRuntimeError(f"source SHA-256 drifted: {path}")
        if item["bytes"] is not None and len(content) != item["bytes"]:
            raise ParallelRuntimeError(f"source byte count drifted: {path}")
        snapshots[path] = content

    build_content = b"\n".join(snapshots[item["path"]] for item in source["files"])
    evidence = _source_feature_evidence(build_content)
    return {
        "repository_path": source["repository_path"],
        "commit": head,
        "tree": tree,
        "license": "Apache-2.0",
        "license_files_verified": True,
        "build_input_count": len(source["files"]),
        "build_inputs": [
            {
                "path": item["path"],
                "git_blob": item["git_blob"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for item in source["files"]
        ],
        "static_feature_evidence": evidence,
        "_snapshots": snapshots,
    }


def _source_feature_evidence(content: bytes) -> dict[str, Any]:
    tokens = {
        "nvdsmetamux_factory": b'gst_element_factory_make ("nvdsmetamux"',
        "parallel_primary_gie_parser": b'pgie_str = "primary-gie"',
        "branch_source_selection": b'branch_str = "branch"',
        "tracker_bin_creation": b"create_tracking_bin",
        "performance_measurement": b"enable_perf_measurement",
        "body_pose_probe": b"body_pose_gie_src_pad_buffer_probe",
        "deepsafe_fusion_config": b"DEEPSAFE_FUSION_CONFIG",
        "helmet": b"helmet",
        "hi_vis": b"hi_vis",
        "ppe_person_association": b"ppe_person_association",
    }
    return {name: content.count(token) for name, token in tokens.items()}


def verify_image(plan: dict[str, Any]) -> dict[str, Any]:
    image = plan["container"]["image_reference"]
    result = _run(["docker", "image", "inspect", image], timeout=30)
    try:
        values = strict_json_loads(result.stdout)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise ParallelRuntimeError(f"invalid Docker image inspection JSON: {exc}") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ParallelRuntimeError("unexpected Docker image inspection result")
    value = values[0]
    repo_digests = value.get("RepoDigests")
    if (
        value.get("Id") != plan["container"]["image_id"]
        or value.get("Architecture") != plan["container"]["architecture"]
        or value.get("Os") != plan["container"]["operating_system"]
        or not isinstance(repo_digests, list)
        or image not in repo_digests
    ):
        raise ParallelRuntimeError("local Docker image identity/digest drifted")
    return {
        "image_reference": image,
        "image_id": value["Id"],
        "architecture": value["Architecture"],
        "operating_system": value["Os"],
        "repo_digest_verified": True,
    }


def _export_source(plan: dict[str, Any], verification: dict[str, Any], destination: Path) -> None:
    source_root = Path(plan["source"]["build_root"])
    snapshots = verification["_snapshots"]
    destination.mkdir(mode=0o750)
    for item in plan["source"]["files"]:
        relative = Path(item["path"]).relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o750 if item["mode"] == "100755" else 0o640)
        try:
            content = snapshots[item["path"]]
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _replace_placeholders(argv: list[str], replacements: dict[str, str]) -> list[str]:
    result: list[str] = []
    for token in argv:
        replaced = token
        for placeholder, value in replacements.items():
            replaced = replaced.replace("{" + placeholder + "}", value)
        if "{" in replaced or "}" in replaced:
            raise ParallelRuntimeError("unresolved Docker argv placeholder")
        result.append(replaced)
    return result


def _probe_argv(plan: dict[str, Any], name: str, binary_path: Path) -> tuple[list[str], list[str]]:
    common_template = list(plan["probes"]["docker_common_argv_template"])
    tool = list(plan["probes"]["tools"][name])
    template = [
        *common_template,
        f"--entrypoint={tool[0]}",
        plan["container"]["image_reference"],
        *tool[1:],
    ]
    actual = _replace_placeholders(template, {"binary_path": str(binary_path)})
    return template, actual


def _parse_readelf(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", "strict")

    def capture(pattern: str, label: str) -> str:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            raise ParallelRuntimeError(f"readelf output lacks {label}")
        return match.group(1).strip()

    elf_class = capture(r"^\s*Class:\s*(.+)$", "ELF class")
    data = capture(r"^\s*Data:\s*(.+)$", "ELF data encoding")
    file_type = capture(r"^\s*Type:\s*(.+)$", "ELF type")
    machine = capture(r"^\s*Machine:\s*(.+)$", "ELF machine")
    interpreter = capture(r"Requesting program interpreter:\s*([^\]]+)\]", "interpreter")
    if (
        elf_class != "ELF64"
        or "little endian" not in data
        or not file_type.startswith("DYN")
        or machine not in {"Advanced Micro Devices X86-64", "AMD x86-64"}
    ):
        raise ParallelRuntimeError("built binary has an unexpected ELF ABI")
    needed = sorted(set(re.findall(r"Shared library:\s*\[([^\]]+)\]", text)))
    if not needed:
        raise ParallelRuntimeError("built binary has no ELF NEEDED entries")
    runpath_matches = re.findall(r"Library r(?:un)?path:\s*\[([^\]]+)\]", text, re.IGNORECASE)
    build_id_match = re.search(r"Build ID:\s*([0-9a-f]+)", text)
    return {
        "class": elf_class,
        "data": data,
        "type": file_type,
        "machine": machine,
        "interpreter": interpreter,
        "needed_sonames": needed,
        "runpaths": sorted(set(runpath_matches)),
        "build_id": build_id_match.group(1) if build_id_match else None,
        "readelf_output_sha256": _sha256(content),
    }


def _parse_ldd(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", "strict")
    unresolved = sorted(
        line.strip().split()[0]
        for line in text.splitlines()
        if "=> not found" in line
    )
    if unresolved:
        raise ParallelRuntimeError(f"unresolved DS9 ABI dependencies: {unresolved}")
    resolved: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=>" in stripped:
            resolved.append(stripped.split("=>", 1)[0].strip())
        else:
            first = stripped.split()[0]
            if first.endswith(".so") or ".so." in first:
                resolved.append(first)
    return {
        "unresolved_sonames": unresolved,
        "resolved_sonames": sorted(set(resolved)),
        "ldd_output_sha256": _sha256(content),
    }


def _run_probes(plan: dict[str, Any], binary_path: Path) -> dict[str, Any]:
    outputs: dict[str, bytes] = {}
    argv_templates: dict[str, list[str]] = {}
    for name in ("readelf", "objdump", "ldd", "strings"):
        template, actual = _probe_argv(plan, name, binary_path)
        argv_templates[name] = template
        outputs[name] = _run(actual, timeout=60).stdout
    elf = _parse_readelf(outputs["readelf"])
    abi = _parse_ldd(outputs["ldd"])
    strings = outputs["strings"]
    binary_tokens = {
        "nvdsmetamux": strings.count(b"nvdsmetamux"),
        "deepsafe_fusion_config": strings.count(b"DEEPSAFE_FUSION_CONFIG"),
        "helmet": strings.count(b"helmet"),
        "hi_vis": strings.count(b"hi_vis"),
        "ppe_person_association": strings.count(b"ppe_person_association"),
    }
    if binary_tokens["nvdsmetamux"] < 1:
        raise ParallelRuntimeError("built binary lacks the nvdsmetamux static token")
    if binary_tokens["deepsafe_fusion_config"] != 0:
        raise ParallelRuntimeError("unexpected DEEPSAFE_FUSION_CONFIG implementation appeared")
    return {
        "scope": "static_elf_and_container_abi_only",
        "argv_templates": argv_templates,
        "argv_template_sha256": {
            name: _template_sha256(value) for name, value in argv_templates.items()
        },
        "elf": elf,
        "abi": abi,
        "objdump_output_sha256": _sha256(outputs["objdump"]),
        "strings_output_sha256": _sha256(strings),
        "binary_token_counts": binary_tokens,
    }


def _capability_manifest(
    *,
    binary_sha256: str,
    binary_size: int,
    provenance_sha256: str,
    source_evidence: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    source_static = source_evidence["static_feature_evidence"]
    binary_tokens = probe["binary_token_counts"]
    nvdsmetamux = source_static["nvdsmetamux_factory"] >= 1 and binary_tokens["nvdsmetamux"] >= 1
    camera_batch = (
        source_static["parallel_primary_gie_parser"] >= 1
        and source_static["branch_source_selection"] >= 1
    )
    tracker = source_static["tracker_bin_creation"] >= 1
    performance = source_static["performance_measurement"] >= 1
    fusion_consumed = (
        source_static["deepsafe_fusion_config"] >= 1
        or binary_tokens["deepsafe_fusion_config"] >= 1
    )
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "status": "blocked",
        "deepstream_version": "9.0.0",
        "parallel_pattern": PARALLEL_PATTERN,
        "parallel_app_binary_sha256": binary_sha256,
        "parallel_app_binary_size_bytes": binary_size,
        "build_provenance_sha256": provenance_sha256,
        "features": {
            "nvdsmetamux": nvdsmetamux,
            "full_frame_camera_batch": camera_batch,
            "nvdcf_tracker": tracker,
            "pose_tensor_track_association": False,
            "ppe_person_association": False,
            "headless_performance": performance,
            "component_latency": True,
        },
        "static_evidence": {
            "deepsafe_fusion_config_consumed": fusion_consumed,
            "source_token_counts": source_static,
            "binary_token_counts": binary_tokens,
            "elf_abi_probe_passed": True,
            "unresolved_sonames": probe["abi"]["unresolved_sonames"],
        },
        "verification_scope": {
            "build_executed": True,
            "container_network": "none",
            "container_runtime": "runc",
            "gpu_device_injected": False,
            "gpu_runtime_probe_executed": False,
            "inference_executed": False,
            "model_or_engine_loaded": False,
            "endurance_executed": False,
        },
        "blockers": [
            {
                "code": "deepsafe_fusion_config_not_consumed",
                "detail": "The pinned upstream binary has no DEEPSAFE_FUSION_CONFIG consumer.",
            },
            {
                "code": "pose_tensor_track_association_not_implemented",
                "detail": "The body-pose sample postprocess is not DeepSafe canonical person-track association.",
            },
            {
                "code": "ppe_person_association_not_implemented",
                "detail": "The pinned upstream application has no helmet/hi-vis to person-track association layer.",
            },
            {
                "code": "gpu_runtime_probe_not_executed",
                "detail": "This authorized lane is CPU-only and did not inject or query a GPU runtime.",
            },
            {
                "code": "full_stack_inference_not_executed",
                "detail": "No model, TensorRT engine, inference stream, or endurance workload was run.",
            },
        ],
        "runtime_ready": False,
    }


def _write_exclusive(path: Path, content: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_chain(root: Path, relative: Path) -> Path:
    root = root.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    current = root
    try:
        for part in relative.parts:
            if SAFE_PART_RE.fullmatch(part) is None:
                raise ParallelRuntimeError("unsafe publication parent component")
            try:
                os.mkdir(part, mode=0o750, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise ParallelRuntimeError("publication parent is not a directory")
            os.close(descriptor)
            descriptor = next_descriptor
            current = current / part
        return current
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(parent: Path, source_name: str, target_name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise ParallelRuntimeError("renameat2 is unavailable; refusing weaker publication")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            descriptor,
            os.fsencode(source_name),
            descriptor,
            os.fsencode(target_name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise ParallelRuntimeError("publication destination already exists")
            raise ParallelRuntimeError(f"renameat2 publication failed: {os.strerror(error)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(
    plan: dict[str, Any],
    *,
    project_root: Path,
    binary: bytes,
    provenance: bytes,
    capability: bytes,
) -> Path:
    destination_relative = _normalized_relative(plan["publication"]["destination"], "destination")
    parent = _ensure_directory_chain(project_root, destination_relative.parent)
    target_name = destination_relative.name
    stage = Path(tempfile.mkdtemp(prefix=".parallel-runtime-stage-", dir=parent))
    try:
        _write_exclusive(stage / "deepstream-parallel-infer", binary, 0o550)
        _write_exclusive(stage / "build-provenance.json", provenance, 0o440)
        _write_exclusive(stage / "capability-manifest.json", capability, 0o440)
        os.chmod(stage, 0o550)
        stage_descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_directory_noreplace(parent, stage.name, target_name)
        return parent / target_name
    except Exception:
        try:
            os.chmod(stage, 0o750)
            shutil.rmtree(stage)
        except OSError:
            pass
        raise


def build_runtime(
    *,
    plan_path: Path = DEFAULT_PLAN,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    builder_path = Path(__file__).resolve(strict=True)
    builder_snapshot = _read_stable_regular(builder_path, max_bytes=4 * 1024 * 1024)
    builder_sha256 = _sha256(builder_snapshot)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    source_verification = verify_source(plan, project_root=project_root)
    image_verification = verify_image(plan)

    with tempfile.TemporaryDirectory(prefix="deepsafe-ds9-parallel-") as temporary_name:
        temporary = Path(temporary_name)
        source_dir = temporary / "source"
        output_dir = temporary / "output"
        _export_source(plan, source_verification, source_dir)
        output_dir.mkdir(mode=0o750)
        build_template = list(plan["build"]["docker_argv_template"])
        build_argv = _replace_placeholders(
            build_template,
            {"source_dir": str(source_dir), "output_dir": str(output_dir)},
        )
        build_result = _run(build_argv, timeout=900)
        binary_path = output_dir / plan["build"]["binary_name"]
        binary = _read_stable_regular(binary_path, max_bytes=128 * 1024 * 1024)
        binary_info = binary_path.lstat()
        if binary_info.st_mode & 0o111 == 0:
            raise ParallelRuntimeError("built binary is not executable")
        probe = _run_probes(plan, binary_path)
        binary_after_probe = _read_stable_regular(binary_path, max_bytes=128 * 1024 * 1024)
        if binary_after_probe != binary:
            raise ParallelRuntimeError("built binary changed during the static/ABI probe")

    if _read_stable_regular(builder_path, max_bytes=4 * 1024 * 1024) != builder_snapshot:
        raise ParallelRuntimeError("builder changed during the build/probe transaction")

    binary_sha256 = _sha256(binary)
    source_public = {key: value for key, value in source_verification.items() if key != "_snapshots"}
    provenance_value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "build_and_static_abi_probe_passed",
        "build_plan_sha256": plan_sha256,
        "builder": {
            "path": "deepstream/parallel_runtime.py",
            "sha256": builder_sha256,
        },
        "source": source_public,
        "container": image_verification,
        "build": {
            "docker_argv_template": build_template,
            "docker_argv_template_sha256": _template_sha256(build_template),
            "stdout_sha256": _sha256(build_result.stdout),
            "stderr_sha256": _sha256(build_result.stderr),
            "returncode": build_result.returncode,
            "network": "none",
            "container_runtime": "runc",
            "gpu_requested": False,
        },
        "artifact": {
            "name": plan["build"]["binary_name"],
            "sha256": binary_sha256,
            "size_bytes": len(binary),
            "elf_type": probe["elf"]["type"],
            "elf_machine": probe["elf"]["machine"],
        },
        "probe": probe,
        "safety": {
            "docker_network_disabled": True,
            "nvidia_runtime_used": False,
            "gpu_device_injected": False,
            "gpu_queried": False,
            "inference_executed": False,
            "model_or_engine_loaded": False,
            "endurance_executed": False,
        },
    }
    provenance = _json_bytes(provenance_value)
    capability_value = _capability_manifest(
        binary_sha256=binary_sha256,
        binary_size=len(binary),
        provenance_sha256=_sha256(provenance),
        source_evidence=source_verification,
        probe=probe,
    )
    capability = _json_bytes(capability_value)
    published = _publish_directory(
        plan,
        project_root=project_root,
        binary=binary,
        provenance=provenance,
        capability=capability,
    )
    return {
        "published": published.relative_to(project_root.resolve()).as_posix(),
        "build_plan_sha256": plan_sha256,
        "binary_sha256": binary_sha256,
        "binary_size_bytes": len(binary),
        "provenance_sha256": _sha256(provenance),
        "capability_manifest_sha256": _sha256(capability),
        "capability_status": capability_value["status"],
        "runtime_ready": capability_value["runtime_ready"],
        "blocker_codes": [item["code"] for item in capability_value["blockers"]],
        "safety": provenance_value["safety"],
    }


def inspect_publication(
    *,
    plan_path: Path = DEFAULT_PLAN,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    plan, plan_sha256 = load_frozen_plan(plan_path)
    destination = project_root / _normalized_relative(plan["publication"]["destination"], "destination")
    names = {
        "deepstream-parallel-infer",
        "build-provenance.json",
        "capability-manifest.json",
    }
    try:
        destination_info = destination.lstat()
    except OSError as exc:
        raise ParallelRuntimeError(f"published runtime directory is missing: {exc}") from exc
    if (
        not stat.S_ISDIR(destination_info.st_mode)
        or stat.S_ISLNK(destination_info.st_mode)
        or stat.S_IMODE(destination_info.st_mode) != 0o550
        or destination.resolve(strict=True) != destination.absolute()
        or set(path.name for path in destination.iterdir()) != names
    ):
        raise ParallelRuntimeError("published runtime directory is missing or has unexpected entries")
    expected_modes = {
        "deepstream-parallel-infer": 0o550,
        "build-provenance.json": 0o440,
        "capability-manifest.json": 0o440,
    }
    for name, expected_mode in expected_modes.items():
        if stat.S_IMODE((destination / name).lstat().st_mode) != expected_mode:
            raise ParallelRuntimeError(f"published artifact mode drifted: {name}")
    snapshots = {
        name: _read_stable_regular(
            destination / name,
            max_bytes=128 * 1024 * 1024 if name == "deepstream-parallel-infer" else MAX_PLAN_BYTES,
        )
        for name in sorted(names)
    }
    try:
        provenance = strict_json_loads(snapshots["build-provenance.json"])
        capability = strict_json_loads(snapshots["capability-manifest.json"])
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise ParallelRuntimeError(f"published JSON is invalid: {exc}") from exc
    binary_sha256 = _sha256(snapshots["deepstream-parallel-infer"])
    current_builder_sha256 = _sha256(
        _read_stable_regular(Path(__file__).resolve(strict=True), max_bytes=4 * 1024 * 1024)
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("build_plan_sha256") != plan_sha256
        or provenance.get("builder", {}).get("path") != "deepstream/parallel_runtime.py"
        or not isinstance(provenance.get("builder", {}).get("sha256"), str)
        or SHA256_RE.fullmatch(provenance["builder"]["sha256"]) is None
        or provenance["builder"]["sha256"] != current_builder_sha256
        or provenance.get("artifact", {}).get("sha256") != binary_sha256
        or not isinstance(capability, dict)
        or capability.get("parallel_app_binary_sha256") != binary_sha256
        or capability.get("build_provenance_sha256") != _sha256(snapshots["build-provenance.json"])
        or capability.get("status") != "blocked"
        or capability.get("runtime_ready") is not False
        or capability.get("features", {}).get("pose_tensor_track_association") is not False
        or capability.get("features", {}).get("ppe_person_association") is not False
    ):
        raise ParallelRuntimeError("published runtime provenance/capability binding failed")
    return {
        "published": plan["publication"]["destination"],
        "build_plan_sha256": plan_sha256,
        "binary_sha256": binary_sha256,
        "binary_size_bytes": len(snapshots["deepstream-parallel-infer"]),
        "provenance_sha256": _sha256(snapshots["build-provenance.json"]),
        "capability_manifest_sha256": _sha256(snapshots["capability-manifest.json"]),
        "capability_status": capability["status"],
        "runtime_ready": capability["runtime_ready"],
        "blocker_codes": [item["code"] for item in capability["blockers"]],
    }


def verify_prerequisites(
    *,
    plan_path: Path = DEFAULT_PLAN,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    plan, plan_sha256 = load_frozen_plan(plan_path)
    source = verify_source(plan, project_root=project_root)
    image = verify_image(plan)
    return {
        "build_plan_sha256": plan_sha256,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "source_inputs_verified": source["build_input_count"],
        "image_id": image["image_id"],
        "network": "none",
        "uses_gpu": False,
        "inference_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/probe the frozen NVIDIA DeepStream 9 parallel app without GPU or network"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify frozen Git inputs and the local image only")
    subparsers.add_parser("build", help="CPU-build, statically probe, and publish once")
    subparsers.add_parser("inspect", help="verify the immutable published artifacts")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_prerequisites(plan_path=args.plan)
        elif args.command == "build":
            result = build_runtime(plan_path=args.plan)
        else:
            result = inspect_publication(plan_path=args.plan)
    except ParallelRuntimeError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
