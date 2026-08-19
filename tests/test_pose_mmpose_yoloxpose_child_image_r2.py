from __future__ import annotations

import ast
import copy
import json

import pytest

from validation import pose_mmpose_yoloxpose_child_image_r2 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T02:00:00+00:00",
        base_local_observed=True,
        docker_version="29.4.0",
        buildx_version=(
            "github.com/docker/buildx v0.33.0 "
            "f7897eba028583e0071642db3c011e860444f8cf"
        ),
    )


def test_r2_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_build_ready"
    assert result["gpu_exposed"] is False
    assert result["model_exported"] is False


def test_r2_uses_new_image_and_preserves_exact_base(plan: dict) -> None:
    assert plan["image_ref"] == "deepsafe-mmpose-yoloxpose-export:child-v2"
    assert plan["base_image"]["immutable_reference"] == lane.BASE_REFERENCE
    assert plan["base_image"]["manifest_digest"] == lane.BASE_DIGEST
    assert plan["base_image"]["local_at_plan_creation"] is True
    assert plan["base_image"]["explicit_pull_required"] is False


def test_r2_pins_shared_runner_and_failed_attempt(plan: dict) -> None:
    assert plan["inputs"]["validator_base"]["sha256"] == lane.V1_VALIDATOR_SHA256
    assert (
        plan["inputs"]["failed_r1_attempt_receipt"]["sha256"]
        == lane.R1_FAILED_RECEIPT_FILE_SHA256
    )
    assert (
        plan["inputs"]["failed_r1_build_log"]["sha256"]
        == lane.R1_FAILED_BUILD_LOG_SHA256
    )
    assert plan["inputs"]["source_build_lock"]["path"].endswith(
        "child-image-v2/requirements-source-build.lock.txt"
    )


def test_failed_r1_evidence_is_self_consistent(plan: dict) -> None:
    repair = plan["repair_from"]
    assert repair["attempt_id"] == "child-v1-r1-001"
    assert repair["status"] == "failed"
    assert repair["stage_reached"] == "image_build"
    assert repair["build_exit_code"] == 1
    assert repair["gpu_exposed"] is False
    assert repair["gpu_api_queried"] is False
    assert "setup_requires" in repair["root_cause"]


def test_source_build_lock_is_minimal_exact_and_hashed() -> None:
    active = [
        line.strip()
        for line in lane.SOURCE_BUILD_LOCK.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert [line.split("==", 1)[0] for line in active] == [
        "cython",
        "numpy",
        "packaging",
    ]
    assert all(line.count("--hash=sha256:") == 1 for line in active)
    assert not any(line.startswith("torch==") for line in active)


def test_source_prerequisites_precede_source_wheel_build() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    install_at = source.index(
        "--requirement /opt/deepsafe-build/requirements-source-build.lock.txt"
    )
    wheel_at = source.index("/opt/deepsafe-export/bin/python -m pip wheel")
    assert install_at < wheel_at
    assert "! /opt/deepsafe-export/bin/python -c 'import torch'" in source
    assert 'deepsafe.child.revision="v2"' in source


def test_build_command_has_two_read_only_contexts_and_no_network(plan: dict) -> None:
    command = plan["commands"]["build"]
    contexts = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--build-context"
    ]
    assert f"wheelbundle={lane.WHEELHOUSE}" in contexts
    assert f"childhelpers={lane.CHILD_HELPERS_R1}" in contexts
    assert "--network=none" in command
    assert "--pull=false" in command
    assert "--gpus" not in command
    assert command[-1] == str(lane.CHILD_ROOT)


def test_runtime_probe_remains_read_only_cpu_only(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "--gpus" not in command


def test_r2_contract_records_repair_without_execution_overclaim() -> None:
    contract = json.loads(lane.CONTRACT.read_text(encoding="utf-8"))
    assert contract["image_ref"].endswith(":child-v2")
    assert contract["repair_from"]["attempt_id"] == "child-v1-r1-001"
    assert contract["build_policy"]["source_build_prerequisites_hash_locked"] is True
    assert contract["build_policy"]["torch_absent_during_source_wheel_build"] is True
    assert all(value is False for value in contract["observed"].values())


def test_r2_wrapper_and_helpers_parse_as_python38() -> None:
    for path in (
        lane.Path(lane.__file__),
        lane.BUNDLE_VERIFY,
        lane.MAKE_LOCAL_LOCK,
        lane.ENVIRONMENT_PROBE,
    ):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 8),
        )


def test_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--network=none")
    tampered["plan_sha256"] = lane.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ChildImageError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_r2_source_validation_replays_failure_and_bundle_evidence() -> None:
    result = lane._validate_sources()
    assert result["wheelhouse"]["valid"] is True
    assert result["failed_r1_attempt"]["status"] == "failed"
    assert result["failed_r1_attempt"]["build_exit_code"] == 1
