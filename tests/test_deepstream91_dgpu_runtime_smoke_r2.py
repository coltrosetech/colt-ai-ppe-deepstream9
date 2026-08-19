from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import deepstream91_dgpu_runtime_smoke_r2 as smoke


@pytest.fixture(scope="session")
def plan() -> dict:
    return smoke.strict_json(smoke.ROOT / smoke.PLAN_REL, "R2 V6 plan")


@pytest.fixture(scope="session")
def outer() -> dict:
    return smoke.strict_json(smoke.ROOT / smoke.OUTER_REL, "R2 outer acceptance")


@pytest.fixture(scope="session")
def handoff() -> dict:
    return smoke.strict_json(smoke.ROOT / smoke.HANDOFF_REL, "R2 author handoff")


@pytest.fixture(scope="session")
def handoff_schema() -> dict:
    return smoke.strict_json(smoke.ROOT / smoke.HANDOFF_SCHEMA_REL, "R2 handoff schema")


@pytest.fixture(scope="session")
def verified() -> dict:
    return smoke.verify_plan(simulate=False)


@pytest.fixture(scope="session")
def simulated() -> dict:
    return smoke.verify_plan(simulate=True)


@pytest.fixture(scope="session")
def worker_tree() -> ast.Module:
    raw = (smoke.ROOT / smoke.WORKER_REL).read_bytes()
    return ast.parse(raw.decode("utf-8", errors="strict"), filename=smoke.WORKER_REL)


def schema_errors(schema: dict, value: dict) -> list[jsonschema.ValidationError]:
    return list(jsonschema.Draft202012Validator(schema).iter_errors(value))


def worker_commands(tree: ast.Module) -> dict[str, list[str]]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "COMMAND_SPECS" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Tuple)
        commands: dict[str, list[str]] = {}
        for item in node.value.elts:
            assert isinstance(item, ast.Dict)
            fields = {
                ast.literal_eval(key): value
                for key, value in zip(item.keys, item.values)
                if key is not None
            }
            commands[ast.literal_eval(fields["probe_id"])] = ast.literal_eval(fields["argv"])
        return commands
    raise AssertionError("COMMAND_SPECS not found")


def test_strict_json_accepts_closed_object(tmp_path: Path) -> None:
    target = tmp_path / "ok.json"
    target.write_bytes(b'{"a":1}')
    assert smoke.strict_json(target, "ok") == {"a": 1}


def test_strict_json_rejects_duplicate_key(tmp_path: Path) -> None:
    target = tmp_path / "duplicate.json"
    target.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(smoke.SmokeR2AuthorError, match="duplicate JSON key"):
        smoke.strict_json(target, "duplicate")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite(tmp_path: Path, token: bytes) -> None:
    target = tmp_path / "nonfinite.json"
    target.write_bytes(b'{"a":' + token + b"}")
    with pytest.raises(smoke.SmokeR2AuthorError, match="non-finite JSON"):
        smoke.strict_json(target, "nonfinite")


def test_strict_json_rejects_nonobject(tmp_path: Path) -> None:
    target = tmp_path / "array.json"
    target.write_bytes(b"[]")
    with pytest.raises(smoke.SmokeR2AuthorError, match="JSON root differs"):
        smoke.strict_json(target, "array")


def test_plan_and_outer_are_exact_immutable_regular_files() -> None:
    for expected in (smoke.PLAN_PIN, smoke.OUTER_PIN):
        assert smoke.observed_pin(expected) == expected
        info = os.lstat(smoke.ROOT / expected["path"])
        assert stat.S_ISREG(info.st_mode)
        assert not stat.S_ISLNK(info.st_mode)


def test_plan_schema_is_closed_and_valid(plan: dict) -> None:
    schema = smoke.strict_json(
        smoke.ROOT / "validation/schemas/gpu-lease-workload-plan-v6.schema.json",
        "V6 plan schema",
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema_errors(schema, plan) == []
    mutated = copy.deepcopy(plan)
    mutated["unexpected"] = True
    assert schema_errors(schema, mutated)


def test_outer_schema_is_closed_and_valid(outer: dict) -> None:
    schema = smoke.strict_json(
        smoke.ROOT / "validation/schemas/gpu-lease-v6-outer-acceptance.schema.json",
        "V6 outer schema",
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema_errors(schema, outer) == []
    mutated = copy.deepcopy(outer)
    mutated["authority"]["unexpected"] = True
    assert schema_errors(schema, mutated)


def test_plan_fingerprint_is_exact(plan: dict) -> None:
    assert smoke.fingerprint(plan, "plan_fingerprint_sha256") == smoke.PLAN_FINGERPRINT
    assert plan["plan_fingerprint_sha256"] == smoke.PLAN_FINGERPRINT


def test_outer_fingerprint_is_exact(outer: dict) -> None:
    assert smoke.fingerprint(outer, "acceptance_fingerprint_sha256") == smoke.OUTER_FINGERPRINT
    assert outer["acceptance_fingerprint_sha256"] == smoke.OUTER_FINGERPRINT


def test_static_audit_is_nonterminal_and_nonexecuting() -> None:
    result = smoke.static_audit()
    assert result["status"] == "author_ready_nonterminal_no_live_launch"
    assert result["decision"] is None
    assert result["authority"]["live_launch_authorized"] is False
    assert result["authority"]["docker_executed"] is False
    assert result["authority"]["nvidia_smi_executed"] is False
    assert result["authority"]["gpu_executed"] is False


def test_exact_gpu_owner_and_time_bound(plan: dict) -> None:
    workload = plan["workload"]
    assert workload["gpu"] == {"index": 0, "uuid": smoke.GPU_UUID}
    assert workload["owner_kind"] == "legacy_validation"
    assert workload["ttl_seconds"] == 90
    assert workload["timeout_seconds"] == 90


def test_exact_image_digest_bound_once(plan: dict) -> None:
    workload = plan["workload"]
    assert workload["image"] == {
        "reference": smoke.IMAGE_REFERENCE,
        "digest": smoke.IMAGE_DIGEST,
    }
    assert workload["image_argv_index"] == 19
    assert workload["argv"][19] == smoke.IMAGE_REFERENCE
    assert workload["argv"].count(smoke.IMAGE_REFERENCE) == 1


def test_exact_argv_and_digest(plan: dict) -> None:
    argv = smoke.expected_docker_argv()
    assert len(argv) == 23
    assert plan["workload"]["argv"] == argv
    assert smoke.v1.command_argv_sha256(argv) == smoke.PLAN_ARGV_SHA256
    assert plan["workload"]["argv_sha256"] == smoke.PLAN_ARGV_SHA256


def test_foreground_envelope_has_no_auto_removal_or_detach(plan: dict) -> None:
    argv = plan["workload"]["argv"]
    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert "--rm" not in argv
    assert "--detach" not in argv
    assert "-d" not in argv


def test_network_and_pull_are_disabled(plan: dict) -> None:
    argv = plan["workload"]["argv"]
    assert "--network=none" in argv
    assert "--pull=never" in argv
    assert "--runtime=runc" in argv


def test_exact_gpu_is_the_only_visible_gpu(plan: dict) -> None:
    argv = plan["workload"]["argv"]
    assert f"--gpus=device={smoke.GPU_UUID}" in argv
    assert f"--env=NVIDIA_VISIBLE_DEVICES={smoke.GPU_UUID}" in argv
    assert "--env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in argv


def test_privilege_boundary_is_hardened(plan: dict) -> None:
    argv = plan["workload"]["argv"]
    for token in (
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=128",
        "--ipc=none",
        "--user=1000:1000",
    ):
        assert token in argv
    assert "--privileged" not in argv


def test_only_worker_and_output_are_bind_mounted(plan: dict) -> None:
    mounts = [item for item in plan["workload"]["argv"] if item.startswith("--mount=")]
    assert len(mounts) == 2
    assert sum("dst=/opt/deepsafe/runtime-smoke/probe.py,readonly" in item for item in mounts) == 1
    assert sum("dst=/output,rw" in item for item in mounts) == 1


def test_model_free_entrypoint_is_exact(plan: dict) -> None:
    argv = plan["workload"]["argv"]
    assert argv[18] == "--entrypoint=/usr/bin/python3"
    assert argv[20:] == [
        "/opt/deepsafe/runtime-smoke/probe.py",
        "--execute-runtime-smoke",
        "--authority-token=d0ec1aa72e4ab019fc9d8009dcc1817d390590cebfad82d64d2a7a4b10c0571d",
    ]
    joined = " ".join(argv).lower()
    assert all(token not in joined for token in (".onnx", ".engine", ".plan", "checkpoint"))


def test_outer_acceptance_binds_exact_workload(plan: dict, outer: dict) -> None:
    workload = plan["workload"]
    accepted = outer["workload_acceptance"]
    assert accepted["accepted"] is True
    assert accepted["plan_id"] == smoke.PLAN_ID
    assert accepted["gpu"] == workload["gpu"]
    assert accepted["owner_kind"] == workload["owner_kind"]
    assert accepted["argv_sha256"] == workload["argv_sha256"]
    assert accepted["image"] == workload["image"]


def test_outer_acceptance_records_user_notification(outer: dict) -> None:
    assert outer["user_notification"] == {
        "confirmed": True,
        "notification_id": "aee380dc6d0f5aeb1d77005cc83ca2261ab3ff2df0a48cab209af7c3912c2b08",
        "notified_at_utc": "2026-07-20T14:52:21Z",
    }
    assert outer["authority"] == {
        "decision": "ACCEPT",
        "scope": "exact-workload-plan-v6",
        "nontransferable": True,
    }


def test_v5_predecessor_is_explicitly_ineligible(plan: dict) -> None:
    predecessor = plan["v5_predecessor"]
    assert predecessor["activation_eligible"] is False
    assert predecessor["tool_drift_reason"] == "docker-and-nvidia-smi-exact-byte-pins-drifted"


def test_v6_review_pin_and_decision_are_exact(plan: dict) -> None:
    artifact = plan["workload"]["artifacts"][1]
    expected = smoke.V6_REVIEW_PIN
    assert artifact["path"] == str(smoke.ROOT / expected["path"])
    assert artifact["bytes"] == expected["bytes"]
    assert artifact["sha256"] == expected["sha256"]
    review = smoke.strict_json(smoke.ROOT / smoke.V6_REVIEW_REL, "V6 review")
    assert review["decision"] == "ACCEPT_STATIC_NON_EXECUTION_ONLY"
    assert review["review_fingerprint_sha256"] == smoke.V6_REVIEW_FINGERPRINT
    assert review["authority"]["gpu_or_docker_execution_authorized"] is False


def test_exact_r1_worker_pin_is_reused_without_change(plan: dict) -> None:
    assert smoke.observed_pin(smoke.WORKER_PIN) == smoke.WORKER_PIN
    artifact = plan["workload"]["artifacts"][0]
    assert artifact["path"] == str(smoke.ROOT / smoke.WORKER_REL)
    assert artifact["bytes"] == smoke.WORKER_PIN["bytes"]
    assert artifact["sha256"] == smoke.WORKER_PIN["sha256"]
    assert artifact["mode"] == smoke.WORKER_PIN["mode"]


def test_worker_commands_are_exact(worker_tree: ast.Module) -> None:
    assert worker_commands(worker_tree) == smoke.EXPECTED_PROBE_COMMANDS


@pytest.mark.parametrize("probe_id", sorted(smoke.EXPECTED_PROBE_COMMANDS))
def test_each_worker_command_is_literal_and_exact(worker_tree: ast.Module, probe_id: str) -> None:
    assert worker_commands(worker_tree)[probe_id] == smoke.EXPECTED_PROBE_COMMANDS[probe_id]


def test_worker_has_one_shell_free_subprocess_site(worker_tree: ast.Module) -> None:
    calls = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(calls) == 1
    assert calls[0].func.attr == "run"
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert "shell" not in keywords


def test_worker_has_no_network_or_model_imports(worker_tree: ast.Module) -> None:
    imports: set[str] = set()
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint({"socket", "requests", "urllib", "http", "torch", "onnx", "tensorflow"})


def test_worker_pipeline_is_four_buffer_model_free() -> None:
    command = smoke.EXPECTED_PROBE_COMMANDS["model_free_nvvideoconvert_pipeline"]
    assert "num-buffers=4" in command
    assert "nvvideoconvert" in command
    assert "fakesink" in command
    assert "sync=false" in command
    assert "nvinfer" not in command


def test_verify_plan_passes_without_execution(verified: dict) -> None:
    assert verified["status"] == "verified_not_executed"
    assert verified["authority_digest"] == smoke.PLAN_AUTHORITY_DIGEST
    assert verified["effective_argv_sha256"] == smoke.EFFECTIVE_ARGV_SHA256


def test_verify_only_path_never_calls_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("execute_workload_plan called by verify-only path")

    monkeypatch.setattr(smoke.v6, "execute_workload_plan", forbidden)
    assert smoke.verify_plan(simulate=False)["status"] == "verified_not_executed"


def test_simulation_is_held_before_any_device_call(simulated: dict) -> None:
    assert simulated["status"] == "simulated_held_activation_passed"
    assert simulated["isolated"] is True
    assert simulated["site_disabled"] is True
    assert simulated["gpu_probe"] == "simulated-no-device-call"
    assert simulated["driver_probe"] == "simulated-no-device-call"


def test_simulation_keeps_v5_paths_ineligible(simulated: dict) -> None:
    assert simulated["v5_api_after_replay"] is False
    assert simulated["v5_live_after_replay"] is False
    assert simulated["outer_workload_after_replay"] is True
    assert simulated["user_notification_after_replay"] is True


def test_controller_exposes_no_live_launch_operation() -> None:
    choices = smoke.parser()._subparsers._group_actions[0].choices
    assert set(choices) == {"audit", "verify-plan", "simulate-held-activation", "verify-handoff"}
    assert all(token not in choices for token in ("launch", "run", "execute", "activate"))


def test_r2_output_tree_is_fresh_and_absent() -> None:
    assert not (smoke.ROOT / smoke.OUTPUT_TREE_REL).exists()
    assert not (smoke.ROOT / smoke.OUTPUT_REL).exists()


def test_handoff_schema_is_closed_and_valid(handoff: dict, handoff_schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(handoff_schema)
    assert schema_errors(handoff_schema, handoff) == []
    mutated = copy.deepcopy(handoff)
    mutated["unexpected"] = True
    assert schema_errors(handoff_schema, mutated)


def test_handoff_fingerprint_is_exact(handoff: dict) -> None:
    assert handoff["handoff_fingerprint_sha256"] == smoke.fingerprint(
        handoff, "handoff_fingerprint_sha256"
    )


def test_handoff_artifact_pins_replay_exactly(handoff: dict) -> None:
    for expected in handoff["subject"]["artifacts"]:
        assert smoke.observed_pin(expected) == expected


def test_handoff_authority_is_nonterminal(handoff: dict) -> None:
    assert handoff["decision"] is None
    authority = handoff["authority"]
    assert authority["user_notification_accepted"] is True
    assert authority["outer_workload_accepted"] is True
    assert authority["v6_static_review_accepted"] is True
    assert authority["v5_activation_eligible"] is False
    assert authority["independent_r2_review_accepted"] is False
    assert authority["live_launch_authorized"] is False
    assert authority["production_authorized"] is False


def test_handoff_records_no_runtime_side_effects(handoff: dict) -> None:
    authority = handoff["authority"]
    assert authority["docker_executed"] is False
    assert authority["nvidia_smi_executed"] is False
    assert authority["gpu_executed"] is False
    assert authority["runtime_result_present"] is False
    assert authority["runtime_result_accepted"] is False
    assert all(value is False for value in handoff["permissions"].values())


def test_handoff_rejects_live_authority_mutation(handoff: dict, handoff_schema: dict) -> None:
    mutated = copy.deepcopy(handoff)
    mutated["authority"]["live_launch_authorized"] = True
    mutated["permissions"]["docker_run"] = True
    assert schema_errors(handoff_schema, mutated)


def test_handoff_exact_workload_matches_plan(handoff: dict, plan: dict) -> None:
    workload = handoff["exact_workload"]
    assert workload["argv"] == plan["workload"]["argv"]
    assert workload["argv_sha256"] == plan["workload"]["argv_sha256"]
    assert workload["gpu"] == plan["workload"]["gpu"]
    assert workload["image"] == plan["workload"]["image"]


def test_handoff_requires_separate_review_and_live_gate(handoff: dict) -> None:
    assert handoff["required_next"] == [
        "separate_independent_review_of_exact_r2_author_bundle",
        "fresh_secure_output_directory_materialization",
        "separate_explicit_live_launch_authorization",
        "current_boot_and_tool_replay_at_launch",
        "terminal_v6_activation_receipt",
        "runtime_smoke_result_and_separate_independent_result_acceptance",
    ]


def test_verify_handoff_is_nonterminal() -> None:
    assert smoke.verify_handoff() == {
        "status": "verified_nonterminal_author_handoff",
        "decision": None,
        "live_launch_authorized": False,
    }


def test_documentation_keeps_live_execution_out_of_scope() -> None:
    text = (smoke.ROOT / smoke.DOC_REL).read_text(encoding="utf-8")
    for phrase in (
        "No Docker command was executed",
        "No NVIDIA-SMI command was executed",
        "No GPU workload was executed",
        "separate independent review",
    ):
        assert phrase in text


def test_pin_helper_shape_and_digest() -> None:
    value = smoke.pin("a/b", 1, "0" * 64, "0440")
    assert value == {"path": "a/b", "bytes": 1, "sha256": "0" * 64, "mode": "0440"}
    assert hashlib.sha256(smoke.canonical_bytes({"a": 1})).hexdigest() == (
        "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    )
