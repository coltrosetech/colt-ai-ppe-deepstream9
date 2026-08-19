from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_permission_child_r7 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T03:00:00+00:00",
        docker_version="29.4.0",
        buildx_version="github.com/docker/buildx v0.33.0 test",
    )


def test_r6_failure_is_frozen_with_pre_inventory_pass() -> None:
    evidence = lane._validate_r6_failure()
    assert evidence["attempt_id"] == "child-v6-permissions-r6-001"
    assert evidence["stage"] == "permission_child_build"
    assert evidence["build_exit_code"] == 1
    assert evidence["pre_fix_inventory_passed"] is True
    assert evidence["dependency_build"] is False
    assert evidence["gpu_exposed"] is False


def test_contract_changes_only_parent_resolution_and_does_not_overclaim() -> None:
    value = lane._validate_contract()
    assert value["parent"]["local_reference"] == lane.PARENT_LOCAL_REF
    assert value["parent"]["required_image_id"] == lane.R4_IMAGE_ID
    assert value["parent"]["binding_checks"] == [
        "immediately_before_build",
        "immediately_after_build",
    ]
    scope = value["change_scope"]
    assert scope["dependency_build"] is False
    assert scope["dependency_install"] is False
    assert scope["source_wheel_build"] is False
    assert scope["model_export"] is False
    assert scope["permission_change_root"] == "/opt/deepsafe"
    assert all(item is False for item in value["observed"].values())


def test_local_parent_tag_is_exact_frozen_r4_image() -> None:
    _, _, binding = lane._inspect_local_binding()
    assert binding == {
        "local_reference": lane.PARENT_LOCAL_REF,
        "image_id": lane.R4_IMAGE_ID,
        "size": lane.PARENT_SIZE,
        "layer_count": lane.PARENT_LAYER_COUNT,
        "layers_sha256": "efe058012e9e792f3a0a03929cdf7a8e6d8ecd597e9e5c2573d32aacd67f8381",
        "export_device": "cpu",
        "gpu_exposed": "false",
    }


def test_local_parent_binding_rejects_moved_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lane._impl._docker_image_inspect

    def changed(reference: str):
        value = original(reference)
        if reference == lane.PARENT_LOCAL_REF and value is not None:
            value = copy.deepcopy(value)
            value["Id"] = "sha256:" + "0" * 64
        return value

    monkeypatch.setattr(lane._impl, "_docker_image_inspect", changed)
    with pytest.raises((lane.LocalBindingChildError, lane._r6.PermissionChildError), match="moved|binding"):
        lane._inspect_local_binding()


def test_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane._impl.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_local_binding_inventory_build_probe"
    assert result["parent_local_reference"] == lane.PARENT_LOCAL_REF
    assert result["parent_image_id"] == lane.R4_IMAGE_ID
    assert result["image_built"] is False
    assert result["gpu_exposed"] is False


def test_plan_pins_r6_plan_receipt_build_log_and_pre_inventory(plan: dict) -> None:
    expected = {
        "validator_r6": lane.R6_VALIDATOR_SHA256,
        "plan_r6": lane.R6_PLAN_FILE_SHA256,
        "attempt_receipt_r6": lane.R6_RECEIPT_FILE_SHA256,
        "build_log_r6": lane.R6_BUILD_LOG_SHA256,
        "pre_inventory_r6": lane.R6_PRE_INVENTORY_SHA256,
    }
    for name, sha256 in expected.items():
        assert plan["inputs"][name]["sha256"] == sha256


def test_build_uses_local_tag_offline_without_dependency_context(plan: dict) -> None:
    command = plan["commands"]["build"]
    assert "--network=none" in command
    assert "--pull=false" in command
    assert "--provenance=false" in command
    assert f"BASE_IMAGE={lane.PARENT_LOCAL_REF}" in command
    assert "--gpus" not in command
    assert all("wheelbundle=" not in item for item in command)
    assert all("childhelpers=" not in item for item in command)
    source = inspect.getsource(lane._build_command)
    assert "pip" not in source
    assert "apt" not in source


def test_plan_declares_parent_binding_before_and_after_build(plan: dict) -> None:
    commands = plan["commands"]
    expected = ["docker", "image", "inspect", lane.PARENT_LOCAL_REF]
    assert commands["pre_build_parent_binding"] == expected
    assert commands["post_build_parent_binding"] == expected
    assert plan["parent"]["binding_checks"] == [
        "immediately_before_build",
        "immediately_after_build",
    ]


def test_dockerfile_is_permission_only_and_uses_local_parent() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    assert f"ARG BASE_IMAGE={lane.PARENT_LOCAL_REF}" in source
    assert "chmod -R a+rX,a-w /opt/deepsafe" in source
    assert "-perm /022" in source
    assert "-type f ! -perm -004" in source
    assert "-type d ! -perm -005" in source
    assert 'deepsafe.child.revision="v7-local-binding"' in source
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
def test_runtime_commands_are_read_only_network_none_and_gpu_free(plan: dict, name: str) -> None:
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


def _successful_build_log() -> str:
    return "\n".join(
        [
            "#4 [internal] load metadata for docker.io/library/deepsafe-mmpose-yoloxpose-export:child-v4",
            "#4 DONE 0.0s",
            "#5 [1/1] RUN chmod -R a+rX,a-w /opt/deepsafe && true",
            "#5 DONE 1.0s",
            "#6 exporting to image",
            "#6 naming to docker.io/library/deepsafe-mmpose-yoloxpose-export:child-v7-local-binding done",
        ]
    )


def test_build_log_policy_accepts_local_permission_export(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text(_successful_build_log(), encoding="utf-8")
    assert lane._validate_build_log(path) == {
        "local_parent_reference_present": True,
        "permission_layer_present": True,
        "remote_pull_or_download_markers": [],
    }


@pytest.mark.parametrize(
    "marker",
    [
        "pull access denied",
        "authorization failed",
        "insufficient_scope",
        "Pulling from repository",
        "Downloading layer",
        "Download complete",
        "failed to fetch",
        "unexpected status from HEAD request",
    ],
)
def test_build_log_policy_rejects_remote_pull_and_download_markers(
    tmp_path: Path, marker: str
) -> None:
    path = tmp_path / "build.log"
    path.write_text(_successful_build_log() + "\n" + marker, encoding="utf-8")
    with pytest.raises(lane.LocalBindingChildError, match="remote pull/download marker"):
        lane._validate_build_log(path)


def test_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--pull=false")
    tampered["plan_sha256"] = lane._impl.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.LocalBindingChildError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_attempt_id_is_bounded() -> None:
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("child-v7-local-binding-r7-001")
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("../escape") is None
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("UPPER") is None


def test_validator_parses_as_python38_and_never_queries_gpu() -> None:
    source = Path(lane.__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=lane.__file__, feature_version=(3, 8))
    assert "torch.cuda." not in source
    assert "nvidia-smi" not in source
    assert "pynvml" not in source
    assert "--gpus" not in lane._build_command()
    assert "--gpus" not in lane._r6._probe_command(
        lane.R4_IMAGE_ID, Path("/tmp/receipt"), "a" * 64, "child-v7-test"
    )

