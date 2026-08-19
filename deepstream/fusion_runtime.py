"""Hermetic DS9 build/probe publisher for the canonical DeepSafe fusion layer.

The authorized lane compiles and runs model-free metadata tests only.  It never
injects a GPU, loads a model/engine, runs inference, or claims runtime readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepstream import parallel_runtime as upstream_runtime
from validation.strict_json import StrictJSONError, loads as strict_json_loads


PLAN_SCHEMA_VERSION = "deepsafe.deepstream-fusion-runtime-build-plan/v1"
PROVENANCE_SCHEMA_VERSION = "deepsafe.deepstream-fusion-runtime-provenance/v1"
CAPABILITY_SCHEMA_VERSION = "deepsafe.deepstream-full-stack-runtime-capabilities/v1"
DEFAULT_PLAN = PROJECT_ROOT / "deepstream/fusion-runtime-build-plan.json"
FROZEN_PLAN_SHA256 = "3fb57248a97de1ebd6eb90dcc2451befc3c05f9db5bc788fca15182ce3e8cd2a"
EXPECTED_UPSTREAM_PLAN_SHA256 = upstream_runtime.FROZEN_PLAN_SHA256
EXPECTED_IMAGE = upstream_runtime.EXPECTED_IMAGE
EXPECTED_IMAGE_ID = upstream_runtime.EXPECTED_IMAGE_ID
EXPECTED_COMMIT = upstream_runtime.EXPECTED_COMMIT
EXPECTED_TREE = upstream_runtime.EXPECTED_TREE
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
EXPECTED_EXPORTS = {
    "deepsafe_fusion_abi_version_v1",
    "deepsafe_fusion_create_from_env_v1",
    "deepsafe_fusion_create_v1",
    "deepsafe_fusion_destroy_v1",
    "deepsafe_fusion_frame_meta_view_v1",
    "deepsafe_fusion_nvds_meta_copy_v1",
    "deepsafe_fusion_nvds_meta_release_v1",
    "deepsafe_fusion_nvds_meta_type_v1",
    "deepsafe_fusion_process_gst_buffer_v1",
}
PUBLISHED_NAMES = {
    "build-provenance.json",
    "capability-manifest.json",
    "deepstream-parallel-infer",
    "fusion-runtime.conf",
    "libdeepsafe_fusion.so.1",
}


class FusionRuntimeError(RuntimeError):
    """A frozen source, image, build, probe, or publication was invalid."""


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FusionRuntimeError(f"{where} must be an object")
    if set(value) != expected:
        raise FusionRuntimeError(
            f"{where} keys mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _template_sha256(value: list[str]) -> str:
    return _sha256(_json_bytes(value))


def _read(path: Path, *, maximum: int) -> bytes:
    try:
        return upstream_runtime._read_stable_regular(path, max_bytes=maximum)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc


def _relative(value: Any, where: str) -> Path:
    try:
        return upstream_runtime._normalized_relative(value, where)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc


def _run(argv: list[str], timeout: int = 900):
    try:
        return upstream_runtime._run(argv, timeout=timeout)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc


def _replace(argv: list[str], values: dict[str, str]) -> list[str]:
    try:
        return upstream_runtime._replace_placeholders(argv, values)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc


def _validate_docker_template(value: Any, *, build: bool) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or "\0" in item for item in value
    ):
        raise FusionRuntimeError("Docker argv template must be a string list")
    required = {
        "docker", "run", "--rm", "--pull=never", "--network=none",
        "--runtime=runc", "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=NVIDIA_DRIVER_CAPABILITIES=none",
        "--env=CUDA_VISIBLE_DEVICES=-1",
    }
    if not build:
        required.add("--read-only")
    if not required.issubset(set(value)):
        raise FusionRuntimeError("Docker argv omits network/GPU isolation")
    if any(item.startswith(("--gpus", "--device", "--runtime=nvidia")) for item in value):
        raise FusionRuntimeError("Docker argv requests GPU/device access")
    placeholders = ("{source_dir}", "{output_dir}") if build else ("{artifact_dir}",)
    if any(sum(item.count(name) for item in value) != 1 for name in placeholders):
        raise FusionRuntimeError("Docker argv placeholder count drifted")
    return value


def validate_plan(value: Any) -> dict[str, Any]:
    plan = _exact_keys(
        value,
        {"schema_version", "plan_id", "upstream", "local_inputs", "container", "build", "probes", "publication"},
        "plan",
    )
    if plan["schema_version"] != PLAN_SCHEMA_VERSION or plan["plan_id"] != "deepsafe-ds9-canonical-fusion-static-build-r1":
        raise FusionRuntimeError("unsupported fusion build plan")
    upstream = _exact_keys(plan["upstream"], {"plan_path", "plan_sha256", "commit", "tree"}, "plan.upstream")
    _relative(upstream["plan_path"], "plan.upstream.plan_path")
    if upstream != {
        "plan_path": "deepstream/parallel-runtime-build-plan.json",
        "plan_sha256": EXPECTED_UPSTREAM_PLAN_SHA256,
        "commit": EXPECTED_COMMIT,
        "tree": EXPECTED_TREE,
    }:
        raise FusionRuntimeError("upstream source reference drifted")

    inputs = plan["local_inputs"]
    if not isinstance(inputs, list) or len(inputs) != 39:
        raise FusionRuntimeError("exactly 39 local build inputs are required")
    paths: list[str] = []
    for index, raw in enumerate(inputs):
        item = _exact_keys(raw, {"path", "sha256", "bytes"}, f"plan.local_inputs[{index}]")
        path = _relative(item["path"], f"plan.local_inputs[{index}].path")
        if path.parts[0] not in {"deepstream", "models"}:
            raise FusionRuntimeError("local input is outside approved roots")
        if SHA256_RE.fullmatch(str(item["sha256"])) is None:
            raise FusionRuntimeError("invalid local input SHA-256")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or not 1 <= item["bytes"] <= MAX_SOURCE_BYTES:
            raise FusionRuntimeError("invalid local input byte count")
        paths.append(path.as_posix())
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise FusionRuntimeError("local inputs must be unique and sorted")
    required_inputs = {
        "deepstream/fusion/build-runtime.sh",
        "deepstream/fusion/exports.map",
        "deepstream/fusion/tests/test_ds9_metadata.cpp",
        "deepstream/patches/deepsafe-fusion-ds9-app.patch",
        "models/pose/postprocess/src/association.cpp",
        "models/ppe/postprocess/src/association.cpp",
    }
    if not required_inputs.issubset(paths):
        raise FusionRuntimeError("required fusion/pose/PPE build input is absent")

    container = _exact_keys(plan["container"], {"image_reference", "image_id", "architecture", "operating_system"}, "plan.container")
    if container != {
        "image_reference": EXPECTED_IMAGE,
        "image_id": EXPECTED_IMAGE_ID,
        "architecture": "amd64",
        "operating_system": "linux",
    }:
        raise FusionRuntimeError("container identity drifted")
    build = _exact_keys(plan["build"], {"docker_argv_template", "network", "container_runtime", "uses_gpu", "runs_inference", "test_count", "outputs"}, "plan.build")
    _validate_docker_template(build["docker_argv_template"], build=True)
    if build["network"] != "none" or build["container_runtime"] != "runc" or build["uses_gpu"] is not False or build["runs_inference"] is not False or build["test_count"] != 6:
        raise FusionRuntimeError("build safety/test contract drifted")
    if build["outputs"] != ["deepstream-parallel-infer", "fusion-runtime.conf", "libdeepsafe_fusion.so.1"]:
        raise FusionRuntimeError("build outputs drifted")

    probes = _exact_keys(plan["probes"], {"docker_common_argv_template", "tools"}, "plan.probes")
    _validate_docker_template(probes["docker_common_argv_template"], build=False)
    expected_tools = {"app_ldd", "app_readelf", "app_strings", "plugin_ldd", "plugin_nm", "plugin_readelf", "plugin_strings"}
    tools = _exact_keys(probes["tools"], expected_tools, "plan.probes.tools")
    for name, argv in tools.items():
        if not isinstance(argv, list) or len(argv) < 2 or any(not isinstance(item, str) or not item for item in argv):
            raise FusionRuntimeError(f"invalid probe command: {name}")
        if argv[0] not in {"/usr/bin/ldd", "/usr/bin/readelf", "/usr/bin/strings", "/usr/bin/nm"} or not argv[-1].startswith("/probe/"):
            raise FusionRuntimeError(f"unapproved probe command: {name}")

    publication = _exact_keys(plan["publication"], {"destination", "no_overwrite", "primitive", "directory_mode", "executable_mode", "config_mode", "json_mode"}, "plan.publication")
    _relative(publication["destination"], "plan.publication.destination")
    if publication != {
        "destination": "models/runtime/deepsafe-fusion-ds9-9946965e-r1",
        "no_overwrite": True,
        "primitive": "renameat2(RENAME_NOREPLACE)",
        "directory_mode": "0550",
        "executable_mode": "0550",
        "config_mode": "0440",
        "json_mode": "0440",
    }:
        raise FusionRuntimeError("publication policy drifted")
    return plan


def load_frozen_plan(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
    raw = _read(path, maximum=MAX_JSON_BYTES)
    digest = _sha256(raw)
    if digest != FROZEN_PLAN_SHA256:
        raise FusionRuntimeError(
            f"fusion build plan SHA-256 mismatch: expected {FROZEN_PLAN_SHA256}, observed {digest}"
        )
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FusionRuntimeError(f"invalid strict build plan JSON: {exc}") from exc
    return validate_plan(value), digest


def _verify_local_inputs(plan: dict[str, Any], project_root: Path) -> dict[str, Any]:
    snapshots: dict[str, bytes] = {}
    public: list[dict[str, Any]] = []
    for item in plan["local_inputs"]:
        relative = _relative(item["path"], "local input")
        path = project_root / relative
        content = _read(path, maximum=MAX_SOURCE_BYTES)
        if len(content) != item["bytes"] or _sha256(content) != item["sha256"]:
            raise FusionRuntimeError(f"local input drifted: {relative.as_posix()}")
        snapshots[relative.as_posix()] = content
        public.append(dict(item))
    return {"count": len(public), "files": public, "_snapshots": snapshots}


def _upstream(plan: dict[str, Any], project_root: Path):
    upstream_path = project_root / _relative(plan["upstream"]["plan_path"], "upstream plan")
    try:
        upstream_plan, digest = upstream_runtime.load_frozen_plan(upstream_path)
        source = upstream_runtime.verify_source(upstream_plan, project_root=project_root)
        image = upstream_runtime.verify_image(upstream_plan)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc
    if digest != plan["upstream"]["plan_sha256"]:
        raise FusionRuntimeError("referenced upstream plan hash drifted")
    return upstream_plan, source, image


def _write_snapshot(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _export_sources(
    staging: Path,
    plan: dict[str, Any],
    local: dict[str, Any],
    upstream_plan: dict[str, Any],
    upstream: dict[str, Any],
) -> None:
    staging.mkdir(mode=0o750)
    for path, content in local["_snapshots"].items():
        _write_snapshot(staging / path, content)
    upstream_runtime._export_source(upstream_plan, upstream, staging / "upstream-app")


def _probe_argv(plan: dict[str, Any], name: str, artifact_dir: Path) -> tuple[list[str], list[str]]:
    command = list(plan["probes"]["tools"][name])
    template = [
        *plan["probes"]["docker_common_argv_template"],
        f"--entrypoint={command[0]}",
        plan["container"]["image_reference"],
        *command[1:],
    ]
    return template, _replace(template, {"artifact_dir": str(artifact_dir)})


def _parse_plugin_readelf(content: bytes) -> dict[str, Any]:
    text = content.decode("utf-8", "strict")
    def capture(pattern: str, label: str) -> str:
        match = re.search(pattern, text, re.MULTILINE)
        if match is None:
            raise FusionRuntimeError(f"plugin readelf lacks {label}")
        return match.group(1).strip()
    elf_class = capture(r"^\s*Class:\s*(.+)$", "class")
    machine = capture(r"^\s*Machine:\s*(.+)$", "machine")
    file_type = capture(r"^\s*Type:\s*(.+)$", "type")
    soname = capture(r"\(SONAME\).*\[([^]]+)\]", "SONAME")
    if elf_class != "ELF64" or not file_type.startswith("DYN") or machine not in {"Advanced Micro Devices X86-64", "AMD x86-64"} or soname != "libdeepsafe_fusion.so.1":
        raise FusionRuntimeError("fusion plugin ELF/SONAME is invalid")
    needed = sorted(set(re.findall(r"\(NEEDED\).*\[([^]]+)\]", text)))
    return {"class": elf_class, "type": file_type, "machine": machine, "soname": soname, "needed_sonames": needed, "readelf_output_sha256": _sha256(content)}


def _parse_exports(content: bytes) -> list[str]:
    exports: set[str] = set()
    for line in content.decode("utf-8", "strict").splitlines():
        fields = line.split()
        if fields:
            name = fields[-1].split("@", 1)[0]
            if name.startswith("deepsafe_fusion_"):
                exports.add(name)
    if exports != EXPECTED_EXPORTS:
        raise FusionRuntimeError(f"fusion C ABI export set drifted: {sorted(exports)}")
    return sorted(exports)


def _run_probes(plan: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    outputs: dict[str, bytes] = {}
    templates: dict[str, list[str]] = {}
    for name in sorted(plan["probes"]["tools"]):
        template, actual = _probe_argv(plan, name, artifact_dir)
        templates[name] = template
        outputs[name] = _run(actual, timeout=60).stdout
    try:
        app_elf = upstream_runtime._parse_readelf(outputs["app_readelf"])
        app_abi = upstream_runtime._parse_ldd(outputs["app_ldd"])
        plugin_abi = upstream_runtime._parse_ldd(outputs["plugin_ldd"])
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc
    plugin_elf = _parse_plugin_readelf(outputs["plugin_readelf"])
    exports = _parse_exports(outputs["plugin_nm"])
    app_strings = outputs["app_strings"]
    plugin_strings = outputs["plugin_strings"]
    app_tokens = {
        "fusion_process_symbol": app_strings.count(b"deepsafe_fusion_process_gst_buffer_v1"),
        "fusion_library_soname": app_strings.count(b"libdeepsafe_fusion.so.1"),
        "nvdsmetamux": app_strings.count(b"nvdsmetamux"),
    }
    plugin_tokens = {
        "config_path_environment": plugin_strings.count(b"DEEPSAFE_FUSION_CONFIG"),
        "config_hash_environment": plugin_strings.count(b"DEEPSAFE_FUSION_CONFIG_SHA256"),
        "canonical_meta_type": plugin_strings.count(b"DEEPSAFE.FUSION.CANONICAL.V1"),
    }
    if (
        "libdeepsafe_fusion.so.1" not in app_elf["needed_sonames"]
        or not any("$ORIGIN" in value for value in app_elf["runpaths"])
        or min(app_tokens.values()) < 1
        or min(plugin_tokens.values()) < 1
    ):
        raise FusionRuntimeError("derivative app/plugin static binding evidence is incomplete")
    return {
        "scope": "cpu_build_model_free_ds9_metadata_tests_and_static_elf_abi",
        "argv_templates": templates,
        "argv_template_sha256": {name: _template_sha256(value) for name, value in templates.items()},
        "app": {"elf": app_elf, "abi": app_abi, "token_counts": app_tokens, "strings_output_sha256": _sha256(app_strings)},
        "plugin": {"elf": plugin_elf, "abi": plugin_abi, "exported_c_abi": exports, "token_counts": plugin_tokens, "strings_output_sha256": _sha256(plugin_strings)},
    }


def _capability(artifacts: dict[str, dict[str, Any]], provenance_sha256: str, probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "status": "blocked",
        "deepstream_version": "9.0.0",
        "parallel_pattern": "nvidia_parallel_inference_nvdsmetamux",
        "parallel_app_binary_sha256": artifacts["deepstream-parallel-infer"]["sha256"],
        "fusion_plugin_sha256": artifacts["libdeepsafe_fusion.so.1"]["sha256"],
        "fusion_config_sha256": artifacts["fusion-runtime.conf"]["sha256"],
        "build_provenance_sha256": provenance_sha256,
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "features": {
            "nvdsmetamux": True,
            "full_frame_camera_batch": True,
            "nvdcf_tracker": True,
            "pose_tensor_track_association": True,
            "ppe_person_association": True,
            "headless_performance": True,
            "component_latency": True,
        },
        "fusion_contract": {
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
        },
        "static_evidence": {
            "deepsafe_fusion_config_consumed": True,
            "exact_config_sha256_required": True,
            "model_free_ds9_metadata_tests_passed": True,
            "elf_abi_probe_passed": True,
            "exported_c_abi": probe["plugin"]["exported_c_abi"],
            "app_unresolved_sonames": probe["app"]["abi"]["unresolved_sonames"],
            "plugin_unresolved_sonames": probe["plugin"]["abi"]["unresolved_sonames"],
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
            {"code": "gpu_integration_not_validated", "detail": "No GPU was injected or queried in this authorized static/model-free lane."},
            {"code": "full_stack_inference_not_executed", "detail": "No model, TensorRT engine, camera stream, or inference pipeline was run."},
            {"code": "endurance_not_executed", "detail": "The required GPU endurance qualification remains separate."},
        ],
        "runtime_ready": False,
    }


def _publish(plan: dict[str, Any], project_root: Path, files: dict[str, tuple[bytes, int]]) -> Path:
    destination = _relative(plan["publication"]["destination"], "destination")
    try:
        parent = upstream_runtime._ensure_directory_chain(project_root, destination.parent)
    except upstream_runtime.ParallelRuntimeError as exc:
        raise FusionRuntimeError(str(exc)) from exc
    stage = Path(tempfile.mkdtemp(prefix=".fusion-runtime-stage-", dir=parent))
    try:
        if set(files) != PUBLISHED_NAMES:
            raise FusionRuntimeError("publication file set drifted")
        for name, (content, mode) in files.items():
            upstream_runtime._write_exclusive(stage / name, content, mode)
        os.chmod(stage, 0o550)
        descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            upstream_runtime._rename_directory_noreplace(parent, stage.name, destination.name)
        except upstream_runtime.ParallelRuntimeError as exc:
            raise FusionRuntimeError(str(exc)) from exc
        return parent / destination.name
    except Exception:
        try:
            os.chmod(stage, 0o750)
            shutil.rmtree(stage)
        except OSError:
            pass
        raise


def build_runtime(*, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    builder_path = Path(__file__).resolve(strict=True)
    builder_snapshot = _read(builder_path, maximum=MAX_SOURCE_BYTES)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    local = _verify_local_inputs(plan, project_root)
    upstream_plan, upstream, image = _upstream(plan, project_root)

    with tempfile.TemporaryDirectory(prefix="deepsafe-ds9-fusion-") as temporary_name:
        temporary = Path(temporary_name)
        source_dir = temporary / "source"
        output_dir = temporary / "output"
        _export_sources(source_dir, plan, local, upstream_plan, upstream)
        output_dir.mkdir(mode=0o750)
        template = list(plan["build"]["docker_argv_template"])
        actual = _replace(template, {"source_dir": str(source_dir), "output_dir": str(output_dir)})
        build_result = _run(actual, timeout=1200)
        if b"100% tests passed, 0 tests failed out of 6" not in build_result.stdout:
            raise FusionRuntimeError("exact six-test CTest success evidence is absent")
        snapshots = {
            name: _read(output_dir / name, maximum=MAX_ARTIFACT_BYTES)
            for name in plan["build"]["outputs"]
        }
        if snapshots["fusion-runtime.conf"] != local["_snapshots"]["deepstream/fusion/default-runtime.conf"]:
            raise FusionRuntimeError("published fusion config is not the frozen exact config")
        probe = _run_probes(plan, output_dir)
        for name, before in snapshots.items():
            if _read(output_dir / name, maximum=MAX_ARTIFACT_BYTES) != before:
                raise FusionRuntimeError(f"artifact changed during probes: {name}")

    if _read(builder_path, maximum=MAX_SOURCE_BYTES) != builder_snapshot:
        raise FusionRuntimeError("builder changed during build transaction")
    artifacts = {
        name: {"sha256": _sha256(content), "size_bytes": len(content)}
        for name, content in snapshots.items()
    }
    upstream_public = {key: value for key, value in upstream.items() if key != "_snapshots"}
    local_public = {key: value for key, value in local.items() if key != "_snapshots"}
    provenance_value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "build_and_model_free_ds9_metadata_tests_passed",
        "build_plan_sha256": plan_sha256,
        "builder": {"path": "deepstream/fusion_runtime.py", "sha256": _sha256(builder_snapshot)},
        "upstream": {"plan_sha256": plan["upstream"]["plan_sha256"], "source": upstream_public},
        "local_inputs": local_public,
        "container": image,
        "build": {
            "docker_argv_template": template,
            "docker_argv_template_sha256": _template_sha256(template),
            "stdout_sha256": _sha256(build_result.stdout),
            "stderr_sha256": _sha256(build_result.stderr),
            "returncode": build_result.returncode,
            "ctest_passed": 6,
        },
        "artifacts": artifacts,
        "probe": probe,
        "safety": {
            "docker_network_disabled": True,
            "container_runtime": "runc",
            "nvidia_runtime_used": False,
            "gpu_device_injected": False,
            "gpu_queried": False,
            "inference_executed": False,
            "model_or_engine_loaded": False,
            "endurance_executed": False,
        },
    }
    provenance = _json_bytes(provenance_value)
    capability_value = _capability(artifacts, _sha256(provenance), probe)
    capability = _json_bytes(capability_value)
    published = _publish(
        plan,
        project_root,
        {
            "deepstream-parallel-infer": (snapshots["deepstream-parallel-infer"], 0o550),
            "libdeepsafe_fusion.so.1": (snapshots["libdeepsafe_fusion.so.1"], 0o550),
            "fusion-runtime.conf": (snapshots["fusion-runtime.conf"], 0o440),
            "build-provenance.json": (provenance, 0o440),
            "capability-manifest.json": (capability, 0o440),
        },
    )
    return {
        "published": published.relative_to(project_root).as_posix(),
        "build_plan_sha256": plan_sha256,
        "artifacts": artifacts,
        "provenance_sha256": _sha256(provenance),
        "capability_manifest_sha256": _sha256(capability),
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "runtime_ready": False,
        "blocker_codes": [item["code"] for item in capability_value["blockers"]],
    }


def inspect_publication(*, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    plan, plan_sha256 = load_frozen_plan(plan_path)
    destination = project_root / _relative(plan["publication"]["destination"], "destination")
    try:
        info = destination.lstat()
    except OSError as exc:
        raise FusionRuntimeError(f"publication is missing: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o550 or destination.resolve(strict=True) != destination.absolute() or {path.name for path in destination.iterdir()} != PUBLISHED_NAMES:
        raise FusionRuntimeError("publication directory/file set is invalid")
    modes = {
        "deepstream-parallel-infer": 0o550,
        "libdeepsafe_fusion.so.1": 0o550,
        "fusion-runtime.conf": 0o440,
        "build-provenance.json": 0o440,
        "capability-manifest.json": 0o440,
    }
    snapshots: dict[str, bytes] = {}
    for name, mode in modes.items():
        path = destination / name
        if stat.S_IMODE(path.lstat().st_mode) != mode:
            raise FusionRuntimeError(f"published mode drifted: {name}")
        snapshots[name] = _read(path, maximum=MAX_ARTIFACT_BYTES)
    try:
        provenance = strict_json_loads(snapshots["build-provenance.json"])
        capability = strict_json_loads(snapshots["capability-manifest.json"])
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FusionRuntimeError(f"published JSON is invalid: {exc}") from exc
    artifacts = {
        name: {"sha256": _sha256(snapshots[name]), "size_bytes": len(snapshots[name])}
        for name in plan["build"]["outputs"]
    }
    builder_sha = _sha256(_read(Path(__file__).resolve(strict=True), maximum=MAX_SOURCE_BYTES))
    if (
        not isinstance(provenance, dict)
        or provenance.get("build_plan_sha256") != plan_sha256
        or provenance.get("builder") != {"path": "deepstream/fusion_runtime.py", "sha256": builder_sha}
        or provenance.get("artifacts") != artifacts
        or not isinstance(capability, dict)
        or capability.get("status") != "blocked"
        or capability.get("runtime_ready") is not False
        or capability.get("fusion_plugin_ready") is not True
        or capability.get("gpu_integration_validated") is not False
        or capability.get("parallel_app_binary_sha256") != artifacts["deepstream-parallel-infer"]["sha256"]
        or capability.get("fusion_plugin_sha256") != artifacts["libdeepsafe_fusion.so.1"]["sha256"]
        or capability.get("fusion_config_sha256") != artifacts["fusion-runtime.conf"]["sha256"]
        or capability.get("build_provenance_sha256") != _sha256(snapshots["build-provenance.json"])
    ):
        raise FusionRuntimeError("publication provenance/capability binding failed")
    return {
        "published": plan["publication"]["destination"],
        "build_plan_sha256": plan_sha256,
        "artifacts": artifacts,
        "provenance_sha256": _sha256(snapshots["build-provenance.json"]),
        "capability_manifest_sha256": _sha256(snapshots["capability-manifest.json"]),
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "runtime_ready": False,
        "blocker_codes": [item["code"] for item in capability["blockers"]],
    }


def verify_prerequisites(*, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    plan, digest = load_frozen_plan(plan_path)
    local = _verify_local_inputs(plan, project_root.resolve(strict=True))
    _, upstream, image = _upstream(plan, project_root.resolve(strict=True))
    return {
        "build_plan_sha256": digest,
        "local_inputs_verified": local["count"],
        "upstream_commit": upstream["commit"],
        "upstream_tree": upstream["tree"],
        "image_id": image["image_id"],
        "network": "none",
        "container_runtime": "runc",
        "uses_gpu": False,
        "runs_inference": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/probe frozen DS9 canonical fusion without GPU or inference")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    commands.add_parser("build")
    commands.add_parser("inspect")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_prerequisites(plan_path=args.plan)
        elif args.command == "build":
            result = build_runtime(plan_path=args.plan)
        else:
            result = inspect_publication(plan_path=args.plan)
    except FusionRuntimeError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
