"""Superseding DS9 fusion publisher with the incompatible OpenPose probe removed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepstream import fusion_runtime as base
from validation.strict_json import StrictJSONError, loads as strict_json_loads


OVERLAY_SCHEMA_VERSION = "deepsafe.deepstream-fusion-runtime-build-plan-overlay/v1"
PROVENANCE_SCHEMA_VERSION = "deepsafe.deepstream-fusion-runtime-provenance/v2"
DEFAULT_PLAN = PROJECT_ROOT / "deepstream/fusion-runtime-r2-build-plan.json"
FROZEN_PLAN_SHA256 = "d84fae28b35514c86c3536c693174e9083f5ea35d27aaa1ef28109bb6ca2ed39"
EXPECTED_BASE_PLAN_SHA256 = base.FROZEN_PLAN_SHA256
EXPECTED_BUILD_SCRIPT = "/build/src/deepstream/fusion-r2/build-runtime.sh"
EXPECTED_DESTINATION = "models/runtime/deepsafe-fusion-ds9-9946965e-r2"


class FusionRuntimeR2Error(RuntimeError):
    """The r2 overlay, source, build, probe, or publication was invalid."""


def _error(exc: Exception) -> FusionRuntimeR2Error:
    return FusionRuntimeR2Error(str(exc))


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise FusionRuntimeR2Error(f"{where} keys mismatch")
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read(path: Path, maximum: int) -> bytes:
    try:
        return base._read(path, maximum=maximum)
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc


def validate_overlay(value: Any) -> dict[str, Any]:
    plan = _exact_keys(
        value,
        {
            "schema_version",
            "plan_id",
            "base_plan",
            "input_replacements",
            "build_script_container_path",
            "publication",
            "required_static_evidence",
        },
        "overlay",
    )
    if plan["schema_version"] != OVERLAY_SCHEMA_VERSION or plan["plan_id"] != "deepsafe-ds9-canonical-fusion-static-build-r2":
        raise FusionRuntimeR2Error("unsupported r2 overlay")
    if plan["base_plan"] != {
        "path": "deepstream/fusion-runtime-build-plan.json",
        "sha256": EXPECTED_BASE_PLAN_SHA256,
    }:
        raise FusionRuntimeR2Error("r2 base plan drifted")
    replacements = plan["input_replacements"]
    if not isinstance(replacements, list) or len(replacements) != 2:
        raise FusionRuntimeR2Error("r2 requires exactly two source replacements")
    expected = {
        "deepstream/fusion/build-runtime.sh": {
            "path": "deepstream/fusion-r2/build-runtime.sh",
            "sha256": "86638e942ffdd21e3d332266b3d4a9088582ed49389cd884d5edecbcd0dcb7c2",
            "bytes": 1736,
        },
        "deepstream/patches/deepsafe-fusion-ds9-app.patch": {
            "path": "deepstream/patches/deepsafe-fusion-ds9-app-r2.patch",
            "sha256": "ea4081eb91047bd8ecc37e1018294f5b25f87d15a6d6d7dc3e465b037a1beb60",
            "bytes": 4699,
        },
    }
    observed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(replacements):
        item = _exact_keys(raw, {"remove_path", "add"}, f"replacement[{index}]")
        addition = _exact_keys(item["add"], {"path", "sha256", "bytes"}, f"replacement[{index}].add")
        try:
            base._relative(item["remove_path"], "remove_path")
            base._relative(addition["path"], "add.path")
        except base.FusionRuntimeError as exc:
            raise _error(exc) from exc
        observed[item["remove_path"]] = dict(addition)
    if observed != expected:
        raise FusionRuntimeR2Error("r2 source replacement set drifted")
    if plan["build_script_container_path"] != EXPECTED_BUILD_SCRIPT:
        raise FusionRuntimeR2Error("r2 build script path drifted")
    publication = plan["publication"]
    expected_publication = {
        "destination": EXPECTED_DESTINATION,
        "no_overwrite": True,
        "primitive": "renameat2(RENAME_NOREPLACE)",
        "directory_mode": "0550",
        "executable_mode": "0550",
        "config_mode": "0440",
        "json_mode": "0440",
    }
    if publication != expected_publication:
        raise FusionRuntimeR2Error("r2 publication policy drifted")
    if plan["required_static_evidence"] != {
        "legacy_openpose_probe_registered": False,
        "canonical_fusion_probe_install_count": 1,
    }:
        raise FusionRuntimeR2Error("r2 static evidence policy drifted")
    return plan


def load_frozen_overlay(path: Path = DEFAULT_PLAN) -> tuple[dict[str, Any], str]:
    raw = _read(path, base.MAX_JSON_BYTES)
    digest = _sha256(raw)
    if digest != FROZEN_PLAN_SHA256:
        raise FusionRuntimeR2Error(
            f"r2 overlay SHA-256 mismatch: expected {FROZEN_PLAN_SHA256}, observed {digest}"
        )
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FusionRuntimeR2Error(f"invalid strict r2 overlay: {exc}") from exc
    return validate_overlay(value), digest


def _materialize(
    overlay: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], str]:
    base_path = project_root / overlay["base_plan"]["path"]
    try:
        plan, digest = base.load_frozen_plan(base_path)
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc
    if digest != overlay["base_plan"]["sha256"]:
        raise FusionRuntimeR2Error("r2 base plan hash mismatch")
    plan = copy.deepcopy(plan)
    by_path = {item["path"]: dict(item) for item in plan["local_inputs"]}
    for replacement in overlay["input_replacements"]:
        if by_path.pop(replacement["remove_path"], None) is None:
            raise FusionRuntimeR2Error("r2 removal input is absent from base plan")
        addition = dict(replacement["add"])
        if addition["path"] in by_path:
            raise FusionRuntimeR2Error("r2 addition collides with base input")
        by_path[addition["path"]] = addition
    plan["local_inputs"] = [by_path[path] for path in sorted(by_path)]
    if len(plan["local_inputs"]) != 39:
        raise FusionRuntimeR2Error("r2 effective input count drifted")
    plan["build"]["docker_argv_template"][-1] = overlay["build_script_container_path"]
    try:
        base._validate_docker_template(plan["build"]["docker_argv_template"], build=True)
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc
    plan["publication"] = copy.deepcopy(overlay["publication"])
    return plan, digest


def _static_source_evidence(source_dir: Path) -> dict[str, Any]:
    app = _read(
        source_dir / "upstream-app/deepstream_parallel_infer_app.cpp",
        base.MAX_SOURCE_BYTES,
    )
    legacy = app.count(
        b"body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER"
    )
    canonical = app.count(b"deepsafe_fusion_app_hook_install(")
    if legacy != 0 or canonical != 1:
        raise FusionRuntimeR2Error(
            f"patched app probe boundary drifted: legacy={legacy}, canonical={canonical}"
        )
    return {
        "legacy_openpose_probe_registered": False,
        "canonical_fusion_probe_install_count": canonical,
        "patched_app_sha256": _sha256(app),
        "patched_app_size_bytes": len(app),
    }


def _capability(
    artifacts: dict[str, dict[str, Any]],
    provenance_sha256: str,
    probe: dict[str, Any],
    static_source: dict[str, Any],
) -> dict[str, Any]:
    value = base._capability(artifacts, provenance_sha256, probe)
    value["static_evidence"].update(static_source)
    value["blockers"] = [
        {
            "code": "gpu_integration_not_validated",
            "detail": "No GPU was injected or queried in this authorized static/model-free lane.",
        },
        {
            "code": "full_stack_inference_not_executed",
            "detail": "No model, TensorRT engine, camera stream, or inference pipeline was run.",
        },
        {
            "code": "endurance_not_executed",
            "detail": "The required GPU endurance qualification remains separate.",
        },
    ]
    return value


def build_runtime(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    builder_path = Path(__file__).resolve(strict=True)
    builder_snapshot = _read(builder_path, base.MAX_SOURCE_BYTES)
    overlay, overlay_sha = load_frozen_overlay(plan_path)
    effective, base_plan_sha = _materialize(overlay, project_root)
    try:
        local = base._verify_local_inputs(effective, project_root)
        upstream_plan, upstream, image = base._upstream(effective, project_root)
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc

    with tempfile.TemporaryDirectory(prefix="deepsafe-ds9-fusion-r2-") as name:
        temporary = Path(name)
        source_dir = temporary / "source"
        output_dir = temporary / "output"
        try:
            base._export_sources(
                source_dir, effective, local, upstream_plan, upstream
            )
        except base.FusionRuntimeError as exc:
            raise _error(exc) from exc
        output_dir.mkdir(mode=0o750)
        template = list(effective["build"]["docker_argv_template"])
        try:
            actual = base._replace(
                template,
                {"source_dir": str(source_dir), "output_dir": str(output_dir)},
            )
            result = base._run(actual, timeout=1200)
        except base.FusionRuntimeError as exc:
            raise _error(exc) from exc
        if b"100% tests passed, 0 tests failed out of 6" not in result.stdout:
            raise FusionRuntimeR2Error("r2 exact six-test success evidence is absent")
        static_source = _static_source_evidence(source_dir)
        snapshots = {
            artifact: _read(output_dir / artifact, base.MAX_ARTIFACT_BYTES)
            for artifact in effective["build"]["outputs"]
        }
        if snapshots["fusion-runtime.conf"] != local["_snapshots"][
            "deepstream/fusion/default-runtime.conf"
        ]:
            raise FusionRuntimeR2Error("r2 fusion config differs from frozen input")
        try:
            probe = base._run_probes(effective, output_dir)
        except base.FusionRuntimeError as exc:
            raise _error(exc) from exc
        for artifact, content in snapshots.items():
            if _read(output_dir / artifact, base.MAX_ARTIFACT_BYTES) != content:
                raise FusionRuntimeR2Error(f"r2 artifact changed during probe: {artifact}")

    if _read(builder_path, base.MAX_SOURCE_BYTES) != builder_snapshot:
        raise FusionRuntimeR2Error("r2 builder changed during transaction")
    artifacts = {
        artifact: {"sha256": _sha256(content), "size_bytes": len(content)}
        for artifact, content in snapshots.items()
    }
    upstream_public = {key: value for key, value in upstream.items() if key != "_snapshots"}
    local_public = {key: value for key, value in local.items() if key != "_snapshots"}
    provenance_value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "build_model_free_ds9_metadata_tests_and_legacy_probe_removal_passed",
        "build_plan_sha256": overlay_sha,
        "base_build_plan_sha256": base_plan_sha,
        "builder": {
            "path": "deepstream/fusion_runtime_r2.py",
            "sha256": _sha256(builder_snapshot),
        },
        "upstream": {
            "plan_sha256": effective["upstream"]["plan_sha256"],
            "source": upstream_public,
        },
        "local_inputs": local_public,
        "container": image,
        "build": {
            "docker_argv_template": template,
            "docker_argv_template_sha256": base._template_sha256(template),
            "stdout_sha256": _sha256(result.stdout),
            "stderr_sha256": _sha256(result.stderr),
            "returncode": result.returncode,
            "ctest_passed": 6,
        },
        "static_source_evidence": static_source,
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
    capability_value = _capability(
        artifacts, _sha256(provenance), probe, static_source
    )
    capability = _json_bytes(capability_value)
    try:
        published = base._publish(
            effective,
            project_root,
            {
                "deepstream-parallel-infer": (
                    snapshots["deepstream-parallel-infer"],
                    0o550,
                ),
                "libdeepsafe_fusion.so.1": (
                    snapshots["libdeepsafe_fusion.so.1"],
                    0o550,
                ),
                "fusion-runtime.conf": (snapshots["fusion-runtime.conf"], 0o440),
                "build-provenance.json": (provenance, 0o440),
                "capability-manifest.json": (capability, 0o440),
            },
        )
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc
    return {
        "published": published.relative_to(project_root).as_posix(),
        "build_plan_sha256": overlay_sha,
        "base_build_plan_sha256": base_plan_sha,
        "artifacts": artifacts,
        "provenance_sha256": _sha256(provenance),
        "capability_manifest_sha256": _sha256(capability),
        "legacy_openpose_probe_registered": False,
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "runtime_ready": False,
        "blocker_codes": [item["code"] for item in capability_value["blockers"]],
    }


def inspect_publication(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    overlay, overlay_sha = load_frozen_overlay(plan_path)
    effective, base_plan_sha = _materialize(overlay, project_root)
    destination = project_root / EXPECTED_DESTINATION
    try:
        info = destination.lstat()
    except OSError as exc:
        raise FusionRuntimeR2Error(f"r2 publication missing: {exc}") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o550
        or destination.resolve(strict=True) != destination.absolute()
        or {path.name for path in destination.iterdir()} != base.PUBLISHED_NAMES
    ):
        raise FusionRuntimeR2Error("r2 publication directory is invalid")
    modes = {
        "deepstream-parallel-infer": 0o550,
        "libdeepsafe_fusion.so.1": 0o550,
        "fusion-runtime.conf": 0o440,
        "build-provenance.json": 0o440,
        "capability-manifest.json": 0o440,
    }
    snapshots: dict[str, bytes] = {}
    for artifact, mode in modes.items():
        path = destination / artifact
        if stat.S_IMODE(path.lstat().st_mode) != mode:
            raise FusionRuntimeR2Error(f"r2 mode drifted: {artifact}")
        snapshots[artifact] = _read(path, base.MAX_ARTIFACT_BYTES)
    try:
        provenance = strict_json_loads(snapshots["build-provenance.json"])
        capability = strict_json_loads(snapshots["capability-manifest.json"])
    except (UnicodeDecodeError, StrictJSONError, json.JSONDecodeError) as exc:
        raise FusionRuntimeR2Error(f"r2 published JSON invalid: {exc}") from exc
    artifacts = {
        artifact: {
            "sha256": _sha256(snapshots[artifact]),
            "size_bytes": len(snapshots[artifact]),
        }
        for artifact in effective["build"]["outputs"]
    }
    builder_sha = _sha256(_read(Path(__file__).resolve(strict=True), base.MAX_SOURCE_BYTES))
    static_source = {
        "legacy_openpose_probe_registered": False,
        "canonical_fusion_probe_install_count": 1,
    }
    if (
        not isinstance(provenance, dict)
        or provenance.get("build_plan_sha256") != overlay_sha
        or provenance.get("base_build_plan_sha256") != base_plan_sha
        or provenance.get("builder")
        != {"path": "deepstream/fusion_runtime_r2.py", "sha256": builder_sha}
        or provenance.get("artifacts") != artifacts
        or any(
            provenance.get("static_source_evidence", {}).get(key) != value
            for key, value in static_source.items()
        )
        or not isinstance(capability, dict)
        or capability.get("status") != "blocked"
        or capability.get("fusion_plugin_ready") is not True
        or capability.get("gpu_integration_validated") is not False
        or capability.get("runtime_ready") is not False
        or capability.get("parallel_app_binary_sha256")
        != artifacts["deepstream-parallel-infer"]["sha256"]
        or capability.get("fusion_plugin_sha256")
        != artifacts["libdeepsafe_fusion.so.1"]["sha256"]
        or capability.get("fusion_config_sha256")
        != artifacts["fusion-runtime.conf"]["sha256"]
        or capability.get("build_provenance_sha256")
        != _sha256(snapshots["build-provenance.json"])
        or capability.get("static_evidence", {}).get(
            "legacy_openpose_probe_registered"
        )
        is not False
    ):
        raise FusionRuntimeR2Error("r2 provenance/capability binding failed")
    return {
        "published": EXPECTED_DESTINATION,
        "build_plan_sha256": overlay_sha,
        "base_build_plan_sha256": base_plan_sha,
        "artifacts": artifacts,
        "provenance_sha256": _sha256(snapshots["build-provenance.json"]),
        "capability_manifest_sha256": _sha256(
            snapshots["capability-manifest.json"]
        ),
        "legacy_openpose_probe_registered": False,
        "fusion_plugin_ready": True,
        "gpu_integration_validated": False,
        "runtime_ready": False,
        "blocker_codes": [item["code"] for item in capability["blockers"]],
    }


def verify_prerequisites(
    *, plan_path: Path = DEFAULT_PLAN, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    overlay, overlay_sha = load_frozen_overlay(plan_path)
    effective, base_plan_sha = _materialize(overlay, project_root)
    try:
        local = base._verify_local_inputs(effective, project_root)
        _, upstream, image = base._upstream(effective, project_root)
    except base.FusionRuntimeError as exc:
        raise _error(exc) from exc
    patch = local["_snapshots"][
        "deepstream/patches/deepsafe-fusion-ds9-app-r2.patch"
    ]
    if patch.count(
        b"-      body_pose_gie_src_pad_buffer_probe, GST_PAD_PROBE_TYPE_BUFFER,"
    ) != 2:
        raise FusionRuntimeR2Error("r2 patch does not remove both legacy probes")
    return {
        "build_plan_sha256": overlay_sha,
        "base_build_plan_sha256": base_plan_sha,
        "local_inputs_verified": local["count"],
        "upstream_commit": upstream["commit"],
        "upstream_tree": upstream["tree"],
        "image_id": image["image_id"],
        "legacy_openpose_probe_removals_pinned": 2,
        "network": "none",
        "container_runtime": "runc",
        "uses_gpu": False,
        "runs_inference": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/probe superseding DS9 fusion app without legacy OpenPose probe"
    )
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
    except FusionRuntimeR2Error as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
