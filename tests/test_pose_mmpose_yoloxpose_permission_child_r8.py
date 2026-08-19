from __future__ import annotations

import ast
import copy
import inspect
import os
import stat
import subprocess
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_permission_child_r8 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T03:10:00+00:00",
        docker_version="29.4.0",
        buildx_version="github.com/docker/buildx v0.33.0 test",
    )


def test_r7_failure_is_frozen_and_local_parent_resolution_passed() -> None:
    evidence = lane._validate_r7_failure()
    assert evidence["attempt_id"] == "child-v7-local-binding-r7-001"
    assert evidence["stage"] == "permission_child_build"
    assert evidence["build_exit_code"] == 1
    assert evidence["local_parent_resolution_passed"] is True
    assert evidence["pre_fix_inventory_passed"] is True
    assert evidence["dependency_build"] is False
    assert evidence["gpu_exposed"] is False


def test_contract_is_type_closed_and_does_not_overclaim() -> None:
    value = lane._validate_contract()
    assert value["parent"]["local_reference"] == lane.PARENT_LOCAL_REF
    assert value["parent"]["required_image_id"] == lane.R4_IMAGE_ID
    assert value["mode_policy"]["write_bit_audit_types"] == [
        "regular_file",
        "directory",
    ]
    assert value["mode_policy"]["write_bit_mask"] == "0222"
    assert value["mode_policy"]["all_audited_types_require_no_write_bits"] is True
    assert value["change_scope"]["dependency_build"] is False
    assert value["change_scope"]["dependency_install"] is False
    assert value["change_scope"]["model_export"] is False
    assert all(item is False for item in value["observed"].values())


def test_type_aware_find_command_has_closed_regular_file_directory_predicate() -> None:
    command = lane._mode_audit_command(["/opt/deepsafe", "/opt/src"])
    assert command == [
        "find",
        "/opt/deepsafe",
        "/opt/src",
        "-xdev",
        "(",
        "-type",
        "f",
        "-o",
        "-type",
        "d",
        ")",
        "-perm",
        "/222",
        "-print",
    ]


def _run_mode_audit(root: Path) -> list[str]:
    completed = subprocess.run(
        lane._mode_audit_command([str(root)]),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.splitlines()


def test_real_find_exempts_0777_symlink_but_rejects_writable_file_and_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime-root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    target = nested / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    link = root / "python"
    link.symlink_to(target)
    target.chmod(0o444)
    nested.chmod(0o555)
    root.chmod(0o555)
    try:
        assert stat.S_IMODE(os.lstat(link).st_mode) == 0o777
        assert _run_mode_audit(root) == []

        for writable_mode in (0o644, 0o464, 0o446):
            target.chmod(writable_mode)
            assert _run_mode_audit(root) == [str(target)]

        target.chmod(0o444)
        nested.chmod(0o755)
        assert _run_mode_audit(root) == [str(nested)]
    finally:
        root.chmod(0o755)
        nested.chmod(0o755)
        target.chmod(0o644)


def test_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane._impl.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_symlink_aware_inventory_build_probe"
    assert result["parent_image_id"] == lane.R4_IMAGE_ID
    assert result["symlink_mode_exempt"] is True
    assert result["image_built"] is False
    assert result["gpu_exposed"] is False


def test_plan_pins_r7_failure_chain(plan: dict) -> None:
    expected = {
        "validator_r7": lane.R7_VALIDATOR_SHA256,
        "plan_r7": lane.R7_PLAN_FILE_SHA256,
        "attempt_receipt_r7": lane.R7_RECEIPT_FILE_SHA256,
        "build_log_r7": lane.R7_BUILD_LOG_SHA256,
        "pre_inventory_r7": lane.R7_PRE_INVENTORY_SHA256,
        "parent_binding_r7": lane.R7_PARENT_BEFORE_SHA256,
    }
    for name, sha256 in expected.items():
        assert plan["inputs"][name]["sha256"] == sha256


def test_build_is_exact_local_parent_offline_and_dependency_free(plan: dict) -> None:
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


def test_dockerfile_has_symlink_aware_write_gate_and_closed_access_gates() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    assert f"ARG BASE_IMAGE={lane.PARENT_LOCAL_REF}" in source
    assert "chmod -R a+rX,a-w /opt/deepsafe" in source
    assert r"\( -type f -o -type d \) -perm /222" in source
    assert "-type f ! -perm -004" in source
    assert "-type d ! -perm -005" in source
    assert 'deepsafe.child.revision="v8-symlink-aware"' in source
    assert 'deepsafe.permission.policy="a+rX,a-w;symlink-mode-exempt"' in source
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
            "#5 [1/1] RUN chmod -R a+rX,a-w /opt/deepsafe && find roots -xdev ( -type f -o -type d ) -perm /222",
            "#5 DONE 1.0s",
            "#6 exporting to image",
            "#6 naming to docker.io/library/deepsafe-mmpose-yoloxpose-export:child-v8-symlink-aware done",
        ]
    )


def test_build_log_policy_requires_type_aware_predicate(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text(_successful_build_log(), encoding="utf-8")
    result = lane._validate_build_log(path)
    assert result["remote_pull_or_download_markers"] == []
    assert result["write_bit_types"] == ["regular_file", "directory"]
    assert result["symlink_mode_exempt"] is True


def test_build_log_policy_rejects_missing_type_predicate(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text(
        _successful_build_log().replace("-type f -o -type d", "-type l"),
        encoding="utf-8",
    )
    with pytest.raises(lane.SymlinkAwareChildError, match="symlink-aware predicate"):
        lane._validate_build_log(path)


@pytest.mark.parametrize(
    "marker",
    ["pull access denied", "authorization failed", "Downloading layer", "failed to fetch"],
)
def test_build_log_policy_still_rejects_remote_markers(
    tmp_path: Path, marker: str
) -> None:
    path = tmp_path / "build.log"
    path.write_text(_successful_build_log() + "\n" + marker, encoding="utf-8")
    with pytest.raises(lane._r7.LocalBindingChildError, match="remote pull/download marker"):
        lane._validate_build_log(path)


def test_r7_runtime_override_is_scoped_and_restored() -> None:
    original = {
        "IMAGE_REF": lane._r7.IMAGE_REF,
        "ATTEMPT_SCHEMA": lane._r7.ATTEMPT_SCHEMA,
        "_build_command": lane._r7._build_command,
    }
    with lane._r7_runtime_overrides():
        assert lane._r7.IMAGE_REF == lane.IMAGE_REF
        assert lane._r7.ATTEMPT_SCHEMA == lane.ATTEMPT_SCHEMA
        assert lane._r7._build_command is lane._build_command
    assert lane._r7.IMAGE_REF == original["IMAGE_REF"]
    assert lane._r7.ATTEMPT_SCHEMA == original["ATTEMPT_SCHEMA"]
    assert lane._r7._build_command is original["_build_command"]


def test_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["mode_policy"]["symlink_mode_exempt"] = False
    tampered["plan_sha256"] = lane._impl.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.SymlinkAwareChildError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_attempt_id_is_bounded() -> None:
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("child-v8-symlink-aware-r8-001")
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("../escape") is None
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("UPPER") is None


def test_validator_parses_as_python38_and_never_queries_gpu() -> None:
    source = Path(lane.__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=lane.__file__, feature_version=(3, 8))
    assert "torch.cuda." not in source
    assert "nvidia-smi" not in source
    assert "pynvml" not in source
    assert "--gpus" not in lane._build_command()
    assert "--gpus" not in lane._r7._r6._probe_command(
        lane.R4_IMAGE_ID, Path("/tmp/receipt"), "a" * 64, "child-v8-test"
    )
