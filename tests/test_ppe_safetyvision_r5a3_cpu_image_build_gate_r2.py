from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r2 as gate


STAMP = "2026-07-20T15:10:00Z"


@pytest.fixture(scope="session")
def audit() -> dict[str, object]:
    return gate.audit_current_state()


@pytest.fixture(scope="session")
def handoff() -> dict[str, object]:
    return gate.build_handoff(tests_run_count=2, tests_passed=84, prepared_at_utc=STAMP)


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["self_fingerprint"] = gate.fingerprint(value)
    return value


def test_strict_json_accepts_object() -> None:
    assert gate.strict_json(b'{"x":1}', label="valid") == {"x": 1}


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(gate.GateR2Error, match="duplicate"):
        gate.strict_json(b'{"x":1,"x":2}', label="duplicate")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite(token: bytes) -> None:
    with pytest.raises(gate.GateR2Error, match="non-finite"):
        gate.strict_json(b'{"x":' + token + b"}", label="nonfinite")


@pytest.mark.parametrize("raw", [b"", b"[]", b"\xef\xbb\xbf{}", b'{"x":"\xff"}'])
def test_strict_json_rejects_bad_envelope(raw: bytes) -> None:
    with pytest.raises(gate.GateR2Error):
        gate.strict_json(raw, label="bad")


@pytest.mark.parametrize("value", ["", "/x", "../x", "a/../x", "a//b", "a/./b", "a\\b", "a\x00b"])
def test_safe_parts_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(gate.GateR2Error, match="unsafe relative path"):
        gate._parts(value)


def test_held_descriptor_reader_returns_joint_exact_pin(tmp_path: Path) -> None:
    item = tmp_path / "item"
    item.write_bytes(b"exact")
    item.chmod(0o440)
    pin = {"path": "item", "bytes": 5, "sha256": hashlib.sha256(b"exact").hexdigest(), "mode": "0440"}
    with gate.AnchoredWorkspace(tmp_path) as workspace:
        assert workspace.read_pin(pin) == b"exact"


def test_held_descriptor_reader_rejects_symlink(tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"x")
    (tmp_path / "alias").symlink_to("target")
    with gate.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(OSError):
            workspace.read_regular("alias")


def test_held_descriptor_reader_rejects_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    os.link(target, tmp_path / "hard")
    with gate.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(gate.GateR2Error, match="single-link"):
            workspace.read_regular("target")


def test_held_descriptor_reader_rejects_deterministic_rename_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"a" * (1024 * 1024 + 1))
    target.chmod(0o440)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"b" * (1024 * 1024 + 1))
    replacement.chmod(0o440)
    real_pread = gate.os.pread
    replaced = False

    def raced(fd: int, length: int, offset: int) -> bytes:
        nonlocal replaced
        chunk = real_pread(fd, length, offset)
        if offset == 0 and not replaced:
            replacement.replace(target)
            replaced = True
        return chunk

    monkeypatch.setattr(gate.os, "pread", raced)
    with gate.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(gate.GateR2Error, match="inode or final name changed"):
            workspace.read_regular("item", limit=2 * 1024 * 1024)


@pytest.mark.parametrize("expected", [gate.DOCKER_BINARY_PIN, gate.BUILDX_BINARY_PIN])
def test_direct_tool_binary_joint_pins_are_exact(expected: dict[str, object]) -> None:
    assert gate.read_absolute_tool(expected) == expected


def test_plan_schema_is_draft_2020_exact_object_const() -> None:
    plan = json.loads((gate.ROOT / gate.PLAN_REL).read_text(encoding="utf-8"))
    schema = json.loads((gate.ROOT / gate.PLAN_SCHEMA_REL).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["const"] == plan
    jsonschema.Draft202012Validator(schema).validate(plan)


def test_plan_and_controller_fingerprints_replay() -> None:
    plan = json.loads((gate.ROOT / gate.PLAN_REL).read_text(encoding="utf-8"))
    controller = json.loads((gate.ROOT / gate.CONTROLLER_REL).read_text(encoding="utf-8"))
    assert gate.fingerprint(plan) == gate.PLAN_FINGERPRINT
    assert gate.fingerprint(controller) == gate.CONTROLLER_FINGERPRINT


@pytest.mark.parametrize("key", sorted(gate.REMEDIATIONS))
def test_plan_binds_each_r1_p1_remediation(key: str) -> None:
    plan = json.loads((gate.ROOT / gate.PLAN_REL).read_text(encoding="utf-8"))
    assert plan["r1_p1_remediations"][key] is True


def test_future_exact_build_projection_binds_every_field() -> None:
    assert set(gate.FUTURE_EXACT_BUILD) == {
        "argv_sha256",
        "network",
        "pull",
        "platform",
        "clean_environment",
        "timeout_seconds",
        "pre_stream_post_stream_post_build_context_replay_required",
        "fresh_absent_outputs_required",
        "oci_metadata_log_and_receipt_pins_required",
        "no_overwrite_publication_required",
        "executes_in_this_handoff",
    }
    assert gate.FUTURE_EXACT_BUILD["network"] == "none"
    assert gate.FUTURE_EXACT_BUILD["pull"] is False
    assert gate.FUTURE_EXACT_BUILD["executes_in_this_handoff"] is False


def test_r1_rejection_is_exact_preserved_lineage() -> None:
    with gate.AnchoredWorkspace(gate.ROOT) as workspace:
        receipt = workspace.read_json(gate.STATIC_PINS["r1_review"])
    gate.validate_r1_review(receipt)
    assert receipt["decision"] == "REJECT"
    assert receipt["severity_counts"] == {"P0": 0, "P1": 2, "P2": 0}
    assert all(value is False for value in receipt["authority"].values())


def test_a32_foundation_remains_context_only_and_exact() -> None:
    with gate.AnchoredWorkspace(gate.ROOT) as workspace:
        receipt = workspace.read_json(gate.STATIC_PINS["a32_foundation"])
    gate.validate_foundation(receipt)
    assert receipt["authority"]["phase_b_context_accepted"] is True
    assert receipt["authority"]["cpu_image_build_execution_authorized"] is False


def test_static_audit_is_nonterminal_and_remediated(audit: dict[str, object]) -> None:
    assert audit["status"] == "STATIC_R2_AUTHOR_GATE_VALID_R1_P1_REMEDIATED_EXECUTION_CLOSED"
    assert audit["source_controls"] == gate.source_controls()
    assert audit["context_lineage_exact"] is True
    assert audit["toolchain_evidence_exact"] is True


@pytest.mark.parametrize("key", sorted(gate.authority_matrix()))
def test_every_authority_bit_is_literal_false(key: str) -> None:
    assert gate.authority_matrix()[key] is False


def test_execute_gate_always_fails_closed() -> None:
    with pytest.raises(gate.GateR2Error, match="nonterminal"):
        gate.require_execution_gate()


def test_source_has_no_subprocess_network_model_gpu_or_runtime_import() -> None:
    source = inspect.getsource(gate)
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports.intersection({"subprocess", "socket", "requests", "torch", "onnx", "onnxruntime", "tensorrt", "ultralytics", "cv2", "gi"})


def test_source_has_no_writer_or_process_execution_surface() -> None:
    source = inspect.getsource(gate)
    tree = ast.parse(source)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not attributes.intersection({"run", "Popen", "system", "popen", "mkdir", "write_text", "write_bytes", "rename", "unlink", "chmod"})
    assert "O_CREAT" not in source and "O_WRONLY" not in source and "O_TRUNC" not in source


def test_source_read_pin_uses_one_read_regular_result_without_reopen() -> None:
    tree = ast.parse(inspect.getsource(gate))
    read_pin = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "read_pin")
    calls = [node.func.attr for node in ast.walk(read_pin) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert calls.count("read_regular") == 1
    assert "_parent" not in calls


def test_handoff_schema_is_closed_and_exact_for_fixed_nested_contracts() -> None:
    schema = json.loads((gate.ROOT / gate.HANDOFF_SCHEMA_REL).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["test_replay"]["const"] == gate.expected_test_replay()
    assert schema["properties"]["future_exact_build"]["const"] == gate.FUTURE_EXACT_BUILD
    assert schema["properties"]["authority"]["const"] == gate.authority_matrix()


def test_exact_handoff_builds_nonterminal_with_two_by_84(handoff: dict[str, object]) -> None:
    result = gate.validate_handoff(handoff)
    assert result["valid"] is True
    assert result["nonterminal"] is True
    assert result["authority_all_false"] is True
    assert [row["passed"] for row in handoff["test_replay"]["runs"]] == [84, 84]


@pytest.mark.parametrize("artifact", ["controller_source", "plan", "plan_schema", "handoff_schema", "static_controller", "tests", "documentation"])
def test_handoff_binds_every_author_artifact_exactly(handoff: dict[str, object], artifact: str) -> None:
    assert handoff["author_artifacts"][artifact] == gate.expected_author_artifacts()[artifact]


@pytest.mark.parametrize("run_count,passed", [(1, 84), (2, 83), (3, 84), (2, 85)])
def test_handoff_builder_rejects_any_nonexact_test_evidence(run_count: int, passed: int) -> None:
    with pytest.raises(gate.GateR2Error, match="exact 2x84"):
        gate.build_handoff(tests_run_count=run_count, tests_passed=passed, prepared_at_utc=STAMP)


@pytest.mark.parametrize(
    "mutation",
    [
        "test_count",
        "test_command",
        "test_gpu",
        "future_pull",
        "future_network",
        "future_extra",
        "artifact_pin",
        "authority",
        "static_audit",
    ],
)
def test_self_resigned_nested_drift_is_rejected(handoff: dict[str, object], mutation: str) -> None:
    changed = copy.deepcopy(handoff)
    if mutation == "test_count":
        changed["test_replay"]["runs"][0]["passed"] = 1
    elif mutation == "test_command":
        changed["test_replay"]["runs"][0]["command"] = "true"
    elif mutation == "test_gpu":
        changed["test_replay"]["runs"][0]["gpu_visibility_disabled"] = False
    elif mutation == "future_pull":
        changed["future_exact_build"]["pull"] = True
    elif mutation == "future_network":
        changed["future_exact_build"]["network"] = "host"
    elif mutation == "future_extra":
        changed["future_exact_build"]["unexpected"] = True
    elif mutation == "artifact_pin":
        changed["author_artifacts"]["tests"]["sha256"] = "0" * 64
    elif mutation == "authority":
        changed["authority"]["docker"] = True
    elif mutation == "static_audit":
        changed["static_audit"]["source_controls"]["joint_inode_pin"] = False
    _resign(changed)
    with pytest.raises(gate.GateR2Error):
        gate.validate_handoff(changed)


def test_self_resigned_top_level_extra_is_rejected(handoff: dict[str, object]) -> None:
    changed = copy.deepcopy(handoff)
    changed["unexpected"] = True
    _resign(changed)
    with pytest.raises(gate.GateR2Error, match="schema rejected"):
        gate.validate_handoff(changed)


def test_unresigned_handoff_drift_is_rejected(handoff: dict[str, object]) -> None:
    changed = copy.deepcopy(handoff)
    changed["prepared_at_utc"] = "2026-07-20T15:11:00Z"
    with pytest.raises(gate.GateR2Error, match="fingerprint differs"):
        gate.validate_handoff(changed)


def test_future_output_targets_remain_absent() -> None:
    with gate.AnchoredWorkspace(gate.ROOT) as workspace:
        for relative in gate.OUTPUT_ABSENCE_PATHS:
            workspace.require_absent(relative)


def test_main_audit_succeeds_without_execution() -> None:
    assert gate.main(["audit"]) == 0


def test_main_execute_returns_fail_closed() -> None:
    assert gate.main(["execute"]) == 2
