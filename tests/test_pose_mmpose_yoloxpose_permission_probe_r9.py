from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_permission_probe_r9 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T03:30:00+00:00",
        docker_version="29.4.0",
    )


def test_r8_successful_build_and_runner_failure_are_independently_frozen() -> None:
    evidence = lane._validate_r8_failure()
    assert evidence["attempt_id"] == "child-v8-symlink-aware-r8-001"
    assert evidence["build_exit_code"] == 0
    assert evidence["build_log_validation"]["permission_run_passed"] is True
    assert evidence["build_log_validation"]["write_bit_mask"] == "0222"
    assert evidence["build_log_validation"]["exported_image_id"] == lane.IMAGE_ID
    assert evidence["runner_status"] == "failed_after_successful_build"
    assert evidence["runner_error"] == "maximum recursion depth exceeded"
    assert evidence["image_reusable_without_build"] is True
    assert evidence["gpu_exposed"] is False


def test_recursion_root_cause_is_reproduced_and_r9_pointer_is_not_dynamic() -> None:
    original = lane._r8._r7._validate_build_log
    assert lane._FROZEN_R7_BUILD_LOG_VALIDATOR is original
    with lane._r8._r7_runtime_overrides():
        assert lane._r8._r7._validate_build_log is lane._r8._validate_build_log
        with pytest.raises(RecursionError):
            lane._r8._validate_build_log(lane.R8_BUILD_LOG)
        assert lane._FROZEN_R7_BUILD_LOG_VALIDATOR is original
    assert lane._r8._r7._validate_build_log is original
    assert lane._validate_r8_build_log(lane.R8_BUILD_LOG)["permission_run_passed"] is True


def test_contract_is_probe_only_and_does_not_overclaim() -> None:
    value = lane._validate_contract()
    assert value["image"]["required_image_id"] == lane.IMAGE_ID
    assert value["image"]["immutable_reference"] == lane.IMAGE_REPO_DIGEST
    scope = value["execution_scope"]
    assert scope["docker_pull"] is False
    assert scope["docker_build"] is False
    assert scope["dependency_build"] is False
    assert scope["dependency_install"] is False
    assert scope["model_export"] is False
    assert scope["post_fix_rootless_inventory"] is True
    assert scope["runtime_probe"] is True
    assert all(item is False for item in value["observed"].values())


def test_exact_image_tag_id_digest_size_layers_lineage_and_labels() -> None:
    _, _, _, image = lane._inspect_exact_image()
    assert image["image_id"] == lane.IMAGE_ID
    assert image["local_reference"] == lane.IMAGE_REF
    assert image["immutable_reference"] == lane.IMAGE_REPO_DIGEST
    assert image["size"] == lane.IMAGE_SIZE
    assert image["layer_count"] == lane.IMAGE_LAYER_COUNT
    assert image["parent_layer_count"] == 26
    assert image["parent_layer_prefix_match"] is True
    assert image["official_base_layer_prefix_match"] is True
    assert image["labels"]["deepsafe.child.revision"] == "v8-symlink-aware"
    assert image["labels"]["deepsafe.permission.policy"] == "a+rX,a-w;symlink-mode-exempt"
    assert image["labels"]["deepsafe.gpu.exposed"] == "false"


def test_exact_image_binding_rejects_moved_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lane._impl._docker_image_inspect

    def changed(reference: str):
        value = original(reference)
        if reference == lane.IMAGE_REF and value is not None:
            value = copy.deepcopy(value)
            value["Id"] = "sha256:" + "0" * 64
        return value

    monkeypatch.setattr(lane._impl, "_docker_image_inspect", changed)
    with pytest.raises(lane.ProbeOnlyR9Error, match="binding differs"):
        lane._inspect_exact_image()


def test_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane._impl.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_exact_image_probe_only"
    assert result["image_id"] == lane.IMAGE_ID
    assert result["image_pulled"] is False
    assert result["image_built"] is False
    assert result["gpu_exposed"] is False


def test_plan_pins_complete_r8_failure_and_build_evidence(plan: dict) -> None:
    expected = {
        "validator_r8": lane.R8_VALIDATOR_SHA256,
        "plan_r8": lane.R8_PLAN_FILE_SHA256,
        "attempt_receipt_r8": lane.R8_RECEIPT_FILE_SHA256,
        "build_log_r8": lane.R8_BUILD_LOG_SHA256,
        "pre_inventory_r8": lane.R8_PRE_INVENTORY_SHA256,
        "parent_binding_r8": lane.R8_PARENT_BEFORE_SHA256,
        "base_inspect_r8": lane.R8_BASE_INSPECT_SHA256,
    }
    for name, sha256 in expected.items():
        assert plan["inputs"][name]["sha256"] == sha256


def test_plan_and_executor_have_no_pull_or_build_path(plan: dict) -> None:
    assert plan["commands"]["docker_pull"] is None
    assert plan["commands"]["docker_build"] is None
    source = inspect.getsource(lane.execute_attempt)
    assert "_build_command" not in source
    assert "buildx" not in source
    assert '["docker", "pull"' not in source


@pytest.mark.parametrize(
    "name", ["post_fix_rootless_inventory", "runtime_probe_template"]
)
def test_probe_commands_are_exact_image_read_only_network_none_gpu_free(
    plan: dict, name: str
) -> None:
    command = plan["commands"][name]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert command[command.index("--user") + 1] == "1000:1000"
    assert command[command.index("--entrypoint") + 2] == lane.IMAGE_ID
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


def test_build_log_validator_rejects_missing_complete_write_mask(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text(
        lane.R8_BUILD_LOG.read_text(encoding="utf-8").replace("-perm /222", "-perm /022"),
        encoding="utf-8",
    )
    with pytest.raises(lane.ProbeOnlyR9Error, match="complete write-bit mask"):
        lane._validate_r8_build_log(path)


def test_build_log_validator_still_rejects_remote_marker(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text(
        lane.R8_BUILD_LOG.read_text(encoding="utf-8") + "\nDownloading layer\n",
        encoding="utf-8",
    )
    with pytest.raises(lane._r8._r7.LocalBindingChildError, match="remote pull/download marker"):
        lane._validate_r8_build_log(path)


def test_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["docker_build"] = ["docker", "build"]
    tampered["plan_sha256"] = lane._impl.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ProbeOnlyR9Error, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_attempt_id_is_bounded() -> None:
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("child-v8-probe-r9-001")
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("../escape") is None
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("UPPER") is None


def test_verify_attempt_rejects_failed_receipt(tmp_path: Path) -> None:
    root = tmp_path / "child-v8-probe-r9-bad"
    root.mkdir()
    receipt = {
        "schema_version": lane.ATTEMPT_SCHEMA,
        "status": "failed",
        "attempt_id": root.name,
    }
    path = root / "attempt-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(lane.ProbeOnlyR9Error, match="did not pass"):
        lane.verify_attempt(path)


def test_published_r9_attempt_is_immutable_and_independently_verifies() -> None:
    path = lane.ATTEMPTS_ROOT / "child-v8-probe-r9-001" / "attempt-receipt.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "906d816e4cee1df91b501eceb7fcfbc85092671102d4778a91a04492c0ed5f0f"
    )
    result = lane.verify_attempt(path)
    assert result["valid"] is True
    assert result["image_id"] == lane.IMAGE_ID
    assert result["image_pulled"] is False
    assert result["image_built"] is False
    assert result["exact_image_binding_stable"] is True
    assert result["rootless_mode_inventory_passed"] is True
    assert result["runtime_probe_passed"] is True
    assert result["gpu_exposed"] is False


def test_validator_parses_as_python38_and_never_queries_gpu() -> None:
    source = Path(lane.__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=lane.__file__, feature_version=(3, 8))
    assert "torch.cuda." not in source
    assert "nvidia-smi" not in source
    assert "pynvml" not in source
    for command in (
        lane._inventory_command(),
        lane._probe_command(Path("/tmp/receipt"), "a" * 64, "child-v8-probe-r9-test"),
    ):
        assert "--gpus" not in command
