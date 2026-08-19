from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_child_probe_r5 as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T02:30:00+00:00",
        docker_version="29.4.0",
    )


def test_r4_successful_image_build_evidence_is_frozen() -> None:
    evidence = lane._validate_r4_evidence()
    assert evidence["build_exit_code"] == 0
    assert evidence["build_network"] == "none"
    assert evidence["runtime_probe_exit_code"] == 125
    assert evidence["runtime_probe_started"] is False
    assert evidence["image_reusable_without_build"] is True
    assert evidence["gpu_exposed"] is False
    assert evidence["gpu_api_queried"] is False


def test_exact_local_image_replays_frozen_id_size_layers_and_labels() -> None:
    _, _, projection = lane._inspect_exact_local_image()
    assert projection["image_id"] == lane.IMAGE_ID
    assert projection["size"] == lane.IMAGE_SIZE
    assert projection["base_layer_prefix_match"] is True
    assert projection["base_layer_count"] == 21
    assert projection["child_layer_count"] == 26
    assert projection["labels"]["deepsafe.gpu.exposed"] == "false"
    assert projection["labels"]["deepsafe.export.device"] == "cpu"


def test_r5_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane._impl.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["status"] == "planned_probe_ready"
    assert result["image_built"] is False
    assert result["image_pulled"] is False
    assert result["gpu_exposed"] is False


def test_r5_has_no_pull_or_build_command(plan: dict) -> None:
    assert plan["commands"]["docker_pull"] is None
    assert plan["commands"]["docker_build"] is None
    source = inspect.getsource(lane.execute_attempt)
    assert "_build_command" not in source
    assert "buildx" not in source
    assert '["docker", "pull"' not in source


def test_r5_mount_uses_valid_long_syntax_default_rw(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    mount = command[command.index("--mount") + 1]
    assert mount.startswith("type=bind,src=")
    assert mount.endswith("dst=/receipt")
    assert all("=" in field for field in mount.split(","))
    assert not mount.endswith(",rw")
    assert ",readonly=" not in mount


def test_r5_runtime_isolation_is_explicit_and_gpu_free(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "--gpus" not in command
    assert command[command.index("--entrypoint") + 1] == "/opt/deepsafe-export/bin/python"
    assert command[command.index("--entrypoint") + 2] == lane.IMAGE_ID


def test_r5_plan_pins_r4_plan_receipt_logs_and_inspects(plan: dict) -> None:
    expected = {
        "validator_r4": lane.R4_VALIDATOR_SHA256,
        "plan_r4": lane.R4_PLAN_FILE_SHA256,
        "attempt_receipt_r4": lane.R4_RECEIPT_FILE_SHA256,
        "build_log_r4": lane.R4_BUILD_LOG_SHA256,
        "probe_log_r4": lane.R4_PROBE_LOG_SHA256,
        "image_inspect_r4": lane.R4_IMAGE_INSPECT_SHA256,
        "base_inspect_r4": lane.R4_BASE_INSPECT_SHA256,
    }
    for name, sha256 in expected.items():
        assert plan["inputs"][name]["sha256"] == sha256


def test_r5_plan_records_probe_only_execution_boundary(plan: dict) -> None:
    boundary = plan["execution_boundary"]
    assert boundary["planner_only"] is True
    assert boundary["image_pulled"] is False
    assert boundary["image_built"] is False
    assert boundary["container_run"] is False
    assert boundary["gpu_exposed"] is False
    assert boundary["model_exported"] is False


def test_r5_detects_if_image_ref_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lane._impl._docker_image_inspect

    def changed(reference: str):
        value = original(reference)
        if reference == lane.IMAGE_REF and value is not None:
            value = copy.deepcopy(value)
            value["Id"] = "sha256:" + "0" * 64
        return value

    monkeypatch.setattr(lane._impl, "_docker_image_inspect", changed)
    with pytest.raises(lane.ProbeOnlyError, match="image ref moved"):
        lane._inspect_exact_local_image()


def test_r5_plan_tamper_fails_against_external_pin(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    command = tampered["commands"]["runtime_probe_template"]
    command.remove("--read-only")
    tampered["plan_sha256"] = lane._impl.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ProbeOnlyError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_r5_attempt_id_is_bounded() -> None:
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("child-v4-probe-r5-001")
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("../escape") is None
    assert lane._impl.ATTEMPT_ID_RE.fullmatch("UPPER") is None


def test_r5_validator_parses_as_python38_and_never_queries_gpu() -> None:
    source = Path(lane.__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=lane.__file__, feature_version=(3, 8))
    assert "torch.cuda." not in source
    assert "nvidia-smi" not in source
    assert "--gpus" not in lane._probe_command(
        Path("/tmp/receipt"), "a" * 64, "child-v4-probe-r5-test"
    )


def test_r5_frozen_r4_receipt_claims_build_pass_but_runtime_not_ready() -> None:
    receipt = json.loads(lane.R4_ATTEMPT_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["build"]["exit_code"] == 0
    assert receipt["probe"]["exit_code"] == 125
    assert receipt["conclusions"]["runtime_probe_passed"] is False
    assert receipt["conclusions"]["production_ready"] is False


def test_r5_verify_attempt_rejects_failed_receipt(tmp_path: Path) -> None:
    root = tmp_path / "child-v4-probe-r5-bad"
    root.mkdir()
    receipt = {
        "schema_version": lane.ATTEMPT_SCHEMA,
        "status": "failed",
        "attempt_id": root.name,
    }
    path = root / "attempt-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(lane.ProbeOnlyError, match="did not pass"):
        lane.verify_attempt(path)
