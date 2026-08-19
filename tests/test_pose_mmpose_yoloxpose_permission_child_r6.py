from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_permission_child_r6 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T02:40:00+00:00",
        docker_version="29.4.0",
        buildx_version=(
            "github.com/docker/buildx v0.33.1 "
            "f00d000000000000000000000000000000000000"
        ),
    )


def _selected(*, manifest_readable: bool) -> dict:
    manifest = {
        "exists": True,
        "mode": "0o444" if manifest_readable else "0o440",
        "uid": 0,
        "gid": 0,
        "read_access": manifest_readable,
        "execute_access": False,
        "write_mode_bits": False,
        "open_read": manifest_readable,
        "open_error": None if manifest_readable else "PermissionError",
    }
    regular = {
        "exists": True,
        "mode": "0o444",
        "uid": 0,
        "gid": 0,
        "read_access": True,
        "execute_access": False,
        "write_mode_bits": False,
        "open_read": True,
        "open_error": None,
    }
    python = {
        "exists": True,
        "mode": "0o555",
        "uid": 0,
        "gid": 0,
        "read_access": True,
        "execute_access": True,
        "write_mode_bits": False,
    }
    result = {path: copy.deepcopy(regular) for path in (
        "/opt/deepsafe/provenance/local-source-wheels.json",
        "/opt/deepsafe/provenance/requirements-local-sources.lock.txt",
        "/opt/deepsafe/provenance/sources/mmdeploy-bc75c9d6c8940aa03d0e1e5b5962bd930478ba77.tar",
        "/opt/deepsafe/environment_probe.py",
        "/opt/deepsafe/environment_probe_base.py",
        "/opt/deepsafe/local_source_contract.py",
        "/opt/deepsafe/build-probe.json",
        "/opt/src/mmdeploy/mmdeploy/__init__.py",
        "/opt/src/mmpose/mmpose/__init__.py",
    )}
    result["/opt/deepsafe/provenance/manifest.json"] = manifest
    result["/opt/deepsafe-export/bin/python"] = python
    result[
        "/opt/deepsafe-export/lib/python3.8/site-packages/mmcv/_ext.cpython-38-x86_64-linux-gnu.so"
    ] = copy.deepcopy(regular)
    return result


def _inventory(phase: str) -> dict:
    post = phase == "post_fix"
    return {
        "schema_version": lane.INVENTORY_SCHEMA,
        "phase": phase,
        "effective_uid": 1000,
        "effective_gid": 1000,
        "roots": ["/opt/deepsafe", "/opt/deepsafe-export", "/opt/src"],
        "counts": {"directories": 10, "regular_files": 20, "symlinks": 2},
        "unreadable_files": [] if post else ["/opt/deepsafe/provenance/manifest.json"],
        "unreadable_file_count_capped": 0 if post else 1,
        "untraversable_directories": [],
        "untraversable_directory_count_capped": 0,
        "writable_mode_paths": [],
        "writable_mode_path_count_capped": 0,
        "walk_errors": [],
        "selected": _selected(manifest_readable=post),
        "root_mount_options": ["ro", "relatime"],
        "root_read_only": True,
        "network_interfaces": ["lo"],
        "gpu_device_nodes": [],
        "gpu_api_query_executed": False,
        "gpu_compute_executed": False,
    }


def test_r5_permission_failure_is_frozen_and_replayed() -> None:
    evidence = lane._validate_r5_evidence()
    assert evidence["attempt_id"] == "child-v4-probe-r5-001"
    assert evidence["image_id"] == lane.R4_IMAGE_ID
    assert evidence["build_performed"] is False
    assert evidence["runtime_probe_exit_code"] == 2
    assert evidence["gpu_exposed"] is False
    assert evidence["gpu_api_queried"] is False


def test_contract_is_permission_only_and_does_not_overclaim() -> None:
    contract = lane._validate_contract()
    scope = contract["change_scope"]
    assert scope["dependency_build"] is False
    assert scope["dependency_install"] is False
    assert scope["source_wheel_build"] is False
    assert scope["model_export"] is False
    assert scope["permission_change_root"] == "/opt/deepsafe"
    assert scope["permission_add"] == "a+rX"
    assert scope["permission_remove"] == "a-w"
    assert all(value is False for value in contract["observed"].values())


def test_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane._impl.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_inventory_build_probe"
    assert result["image_built"] is False
    assert result["gpu_exposed"] is False


def test_build_is_exact_parent_offline_and_has_no_dependency_context(plan: dict) -> None:
    command = plan["commands"]["build"]
    assert "--network=none" in command
    assert "--pull=false" in command
    assert "--provenance=false" in command
    assert f"BASE_IMAGE={lane.PARENT_REFERENCE}" in command
    assert "--gpus" not in command
    assert all("wheelbundle=" not in item for item in command)
    assert all("childhelpers=" not in item for item in command)
    source = inspect.getsource(lane._build_command)
    assert "pip" not in source
    assert "apt" not in source


def test_dockerfile_has_only_permission_repair_and_closed_mode_assertions() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    assert lane.PARENT_REFERENCE in source
    assert "chmod -R a+rX,a-w /opt/deepsafe" in source
    assert "-perm /022" in source
    assert "-type f ! -perm -004" in source
    assert "-type d ! -perm -005" in source
    for forbidden in ("apt-get", "apt ", "pip install", "uv pip", "wget ", "curl "):
        assert forbidden not in source


@pytest.mark.parametrize(
    "name",
    [
        "pre_fix_rootless_inventory",
        "post_fix_rootless_inventory_template",
        "runtime_probe_template",
    ],
)
def test_rootless_commands_are_read_only_network_none_and_gpu_free(plan: dict, name: str) -> None:
    command = plan["commands"][name]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert command[command.index("--user") + 1] == "1000:1000"
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "--gpus" not in command


def test_runtime_mount_uses_valid_long_syntax(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    mount = command[command.index("--mount") + 1]
    assert mount.startswith("type=bind,src=")
    assert mount.endswith("dst=/receipt")
    assert all("=" in field for field in mount.split(","))
    assert not mount.endswith(",rw")


def test_synthetic_pre_and_post_inventory_contracts_pass() -> None:
    assert lane._validate_inventory(_inventory("pre_fix"), "pre_fix")["phase"] == "pre_fix"
    assert lane._validate_inventory(_inventory("post_fix"), "post_fix")["phase"] == "post_fix"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest_readable_pre", "unexpectedly readable"),
        ("manifest_mode_pre", "manifest mode differs"),
        ("manifest_owner_pre", "owner differs"),
        ("root_uid", "UID differs"),
        ("write_mode", "write mode bits"),
        ("gpu_node", "GPU devices"),
        ("network", "network differs"),
        ("root_rw", "root is writable"),
        ("unreadable_post", "unreadable files remain"),
        ("untraversable_post", "untraversable dirs remain"),
        ("walk_error_post", "walk errors remain"),
        ("selected_missing_post", "selected path missing"),
        ("selected_unreadable_post", "selected path unreadable"),
    ],
)
def test_inventory_rejects_access_and_isolation_mutations(mutation: str, message: str) -> None:
    phase = "pre_fix" if mutation.endswith("_pre") else "post_fix"
    value = _inventory(phase)
    manifest = value["selected"]["/opt/deepsafe/provenance/manifest.json"]
    if mutation == "manifest_readable_pre":
        manifest["read_access"] = True
    elif mutation == "manifest_mode_pre":
        manifest["mode"] = "0o444"
    elif mutation == "manifest_owner_pre":
        manifest["uid"] = 1000
    elif mutation == "root_uid":
        value["effective_uid"] = 0
    elif mutation == "write_mode":
        value["writable_mode_paths"] = ["/opt/deepsafe/build-probe.json"]
    elif mutation == "gpu_node":
        value["gpu_device_nodes"] = ["/dev/nvidia0"]
    elif mutation == "network":
        value["network_interfaces"] = ["eth0", "lo"]
    elif mutation == "root_rw":
        value["root_read_only"] = False
    elif mutation == "unreadable_post":
        value["unreadable_files"] = ["/opt/deepsafe/build-probe.json"]
    elif mutation == "untraversable_post":
        value["untraversable_directories"] = ["/opt/deepsafe/provenance"]
    elif mutation == "walk_error_post":
        value["walk_errors"] = [{"path": "/opt/deepsafe", "error": "PermissionError"}]
    elif mutation == "selected_missing_post":
        value["selected"]["/opt/deepsafe/build-probe.json"]["exists"] = False
    else:
        value["selected"]["/opt/deepsafe/build-probe.json"]["read_access"] = False
    with pytest.raises(lane.PermissionChildError, match=message):
        lane._validate_inventory(value, phase)


def test_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--network=none")
    tampered["plan_sha256"] = lane._impl.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.PermissionChildError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_validator_parses_as_python38_and_never_queries_gpu() -> None:
    source = Path(lane.__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=lane.__file__, feature_version=(3, 8))
    assert "torch.cuda." not in source
    assert "nvidia-smi" not in source
    assert "pynvml" not in source
    for command in (
        lane._inventory_command(lane.R4_IMAGE_ID, "pre_fix"),
        lane._probe_command(lane.R4_IMAGE_ID, Path("/tmp/receipt"), "a" * 64, "child-v6-test"),
    ):
        assert "--gpus" not in command

