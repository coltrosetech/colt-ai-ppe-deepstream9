from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r1 as gate


STAMP = "2026-07-20T17:30:00Z"


def _load(relative: str) -> dict:
    return json.loads((gate.ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def plan() -> dict:
    return _load(gate.PLAN_REL)


@pytest.fixture(scope="session")
def schema() -> dict:
    return _load(gate.SCHEMA_REL)


@pytest.fixture(scope="session")
def controller() -> dict:
    return _load(gate.STATIC_CONTROLLER_REL)


@pytest.fixture(scope="session")
def foundation() -> dict:
    return _load(gate.FOUNDATION_REL)


@pytest.fixture(scope="session")
def static_audit() -> dict:
    return gate.load_and_verify_bundle(observe=False)


@pytest.fixture(scope="session")
def handoff(static_audit: dict) -> dict:
    return gate.build_handoff(
        tests_run_count=2,
        tests_passed=64,
        prepared_at_utc=STAMP,
        audit=static_audit,
    )


def test_strict_json_accepts_closed_object() -> None:
    assert gate.strict_json(b'{"a":1}', label="valid") == {"a": 1}


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(gate.GateR1Error, match="duplicate"):
        gate.strict_json(b'{"a":1,"a":2}', label="duplicate")


def test_strict_json_rejects_bom() -> None:
    with pytest.raises(gate.GateR1Error, match="BOM"):
        gate.strict_json(b"\xef\xbb\xbf{}", label="bom")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite(token: bytes) -> None:
    with pytest.raises(gate.GateR1Error, match="non-finite"):
        gate.strict_json(b'{"x":' + token + b"}", label="nonfinite")


def test_strict_json_rejects_invalid_utf8() -> None:
    with pytest.raises(gate.GateR1Error, match="invalid JSON"):
        gate.strict_json(b'{"x":"\xff"}', label="utf8")


def test_strict_json_rejects_non_object_root() -> None:
    with pytest.raises(gate.GateR1Error, match="root"):
        gate.strict_json(b"[]", label="array")


@pytest.mark.parametrize("path", ["", "/tmp/x", "../x", "a/../x", "a/./x", "a\x00b"])
def test_safe_parts_rejects_unsafe_path(path: str) -> None:
    with pytest.raises(gate.GateR1Error):
        gate.safe_parts(path)


def test_safe_parts_accepts_nested_relative_path() -> None:
    assert gate.safe_parts("validation/a.json") == ("validation", "a.json")


def test_fingerprint_is_canonical_and_excludes_only_named_field() -> None:
    first = {"b": 2, "a": 1, "self_fingerprint": "x"}
    second = {"a": 1, "b": 2, "self_fingerprint": "y"}
    assert gate.fingerprint(first) == gate.fingerprint(second)


def test_future_argv_digest_replays_exactly() -> None:
    assert gate.command_argv_sha256(gate.FUTURE_ARGV) == gate.FUTURE_ARGV_SHA256


@pytest.mark.parametrize("argv", [[], [""], ["ok", ""]])
def test_command_digest_rejects_malformed_argv(argv: list[str]) -> None:
    with pytest.raises(gate.GateR1Error):
        gate.command_argv_sha256(argv)


def test_plan_schema_is_draft_2020_exact_object_const(plan: dict, schema: dict) -> None:
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["const"] == plan
    Draft202012Validator(schema).validate(plan)


def test_plan_schema_rejects_extra_or_mutated_value(plan: dict, schema: dict) -> None:
    extra = copy.deepcopy(plan)
    extra["unexpected"] = True
    changed = copy.deepcopy(plan)
    changed["authority"]["build_execution"] = True
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(extra)
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(changed)


def test_plan_fingerprint_and_static_validation(plan: dict, schema: dict) -> None:
    assert plan["self_fingerprint"] == gate.PLAN_FINGERPRINT
    assert gate.fingerprint(plan) == gate.PLAN_FINGERPRINT
    assert gate.validate_plan(plan, schema)["schema_closed"] is True


def test_plan_authority_matrix_is_closed(plan: dict) -> None:
    assert plan["authority"]["independent_review_required"] is True
    assert all(
        value is False
        for key, value in plan["authority"].items()
        if key != "independent_review_required"
    )


def test_plan_binds_three_exact_trees(plan: dict) -> None:
    accepted = plan["foundation"]["accepted_context"]
    assert accepted["full_tree"] == gate.EXPECTED_TREES["full"]
    assert accepted["payload_excluding_context_receipt"] == gate.EXPECTED_TREES["payload"]
    assert accepted["runtime_rootfs"] == gate.EXPECTED_TREES["rootfs"]
    assert accepted["full_tree"]["regular_file_bytes"] == 2_235_796_669


def test_plan_binds_symlink_mode_and_special_closure(plan: dict) -> None:
    accepted = plan["foundation"]["accepted_context"]
    assert accepted["symlink_closure"]["symlinks"] == 47
    assert accepted["symlink_closure"]["dangling"] == 0
    assert accepted["symlink_closure"]["cycles"] == 0
    assert accepted["symlink_closure"]["root_escapes"] == 0
    assert accepted["mode_inventory"]["writable_directories_or_regular_files"] == 0
    assert accepted["mode_inventory"]["special_entries"] == 0


def test_future_stream_requires_pre_stream_post_stream_and_post_build_replay(plan: dict) -> None:
    stream = plan["future_build_contract"]["context_stream"]
    assert stream["pre_stream_full_tree_replay_required"] is True
    assert stream["stream_tree_must_equal_pre_stream_tree"] is True
    assert stream["post_stream_full_tree_replay_required"] is True
    assert stream["post_stream_tree_must_equal_pre_stream_tree"] is True
    assert stream["post_build_full_tree_replay_required"] is True
    assert stream["post_build_tree_must_equal_pre_stream_tree"] is True
    assert stream["context_copy"] is False
    assert stream["stream_bytes_pin"] is None
    assert stream["stream_sha256_pin"] is None


def test_future_argv_has_exact_offline_oci_flags(plan: dict) -> None:
    argv = plan["future_build_contract"]["invocation"]["argv"]
    assert argv == gate.FUTURE_ARGV
    assert "--network=none" in argv
    assert "--pull=false" in argv
    assert "--no-cache" in argv
    assert "--platform=linux/amd64" in argv
    assert "--progress=rawjson" in argv
    assert "--load" not in argv
    assert "--push" not in argv
    assert argv[-1] == "-"


def test_future_invocation_uses_clean_environment_and_supervision(plan: dict) -> None:
    invocation = plan["future_build_contract"]["invocation"]
    assert invocation["clean_environment"] == gate.EXPECTED_ENV
    assert invocation["caller_environment_inherited"] is False
    assert invocation["shell"] is False
    assert invocation["timeout_seconds"] == 7200
    assert invocation["supervision"] == {
        "start_new_session": True,
        "timeout_signal": "SIGTERM",
        "terminate_grace_seconds": 10,
        "kill_signal": "SIGKILL",
        "kill_grace_seconds": 30,
        "wait_and_reap_required": True,
        "success_exit_code": 0,
    }


def test_future_outputs_are_unpublished_and_required_absent(plan: dict) -> None:
    for output in plan["future_build_contract"]["outputs"].values():
        assert output["required_absent_now"] is True
        assert output["published_pin"] is None


def test_future_publication_is_no_overwrite_and_independently_accepted(plan: dict) -> None:
    publication = plan["future_build_contract"]["publication"]
    assert publication["file_creation"] == "openat_o_excl_o_nofollow"
    assert publication["file_and_directory_fsync_required"] is True
    assert publication["same_directory_renameat2_noreplace"] is True
    assert publication["parent_directory_fsync_required"] is True
    assert publication["overwrite_or_reuse_forbidden"] is True
    assert publication["execution_acceptance_receipt_requires_separate_independent_review"] is True


def test_static_controller_fingerprint_replays(controller: dict) -> None:
    assert controller["self_fingerprint"] == gate.CONTROLLER_FINGERPRINT
    assert gate.fingerprint(controller) == gate.CONTROLLER_FINGERPRINT
    assert gate.validate_controller(controller)["build_adapter_present"] is False


def test_static_controller_authority_and_claims_are_closed(controller: dict) -> None:
    assert controller["authority"]["model"] == "external_independent_review_required"
    assert all(value is False for key, value in controller["authority"].items() if key != "model")
    assert controller["execution_gate"]["authorized"] is False
    assert controller["execution_gate"]["build_adapter_present"] is False
    assert controller["claim_boundary"]["toolchain_observed_read_only"] is True
    assert all(value is False for key, value in controller["claim_boundary"].items() if key != "toolchain_observed_read_only")


def test_foundation_fingerprint_and_decision_replay(foundation: dict) -> None:
    assert gate.validate_foundation(foundation)["decision"] == "ACCEPT"
    assert foundation["review_fingerprint_sha256"] == gate.FOUNDATION_FINGERPRINT
    assert gate.fingerprint(foundation, "review_fingerprint_sha256") == gate.FOUNDATION_FINGERPRINT
    assert foundation["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}


def test_foundation_grants_only_next_gate_entry(foundation: dict) -> None:
    authority = foundation["authority"]
    assert authority["phase_b_context_accepted"] is True
    assert authority["separate_exact_cpu_image_build_gate_entry_allowed"] is True
    for key in (
        "cpu_image_build_execution_authorized", "docker_authorized",
        "context_copy_authorized", "checkpoint_deserialization_or_export_authorized",
        "model_or_onnx_load_authorized", "gpu_authorized",
        "tensorrt_or_deepstream_authorized", "quality_validated", "production_ready",
    ):
        assert authority[key] is False
    assert foundation["next_gate"]["image_build_executes_here"] is False


def test_foundation_review_source_pins_are_exact(foundation: dict) -> None:
    assert sorted(foundation["review_source_pins"], key=lambda item: item["path"]) == sorted(
        [gate.PINS["foundation_review_tests"], gate.PINS["foundation_review_source"], gate.PINS["foundation_review_schema"]],
        key=lambda item: item["path"],
    )


@pytest.mark.parametrize("name", sorted(gate.PINS))
def test_static_input_pin_is_exact(name: str) -> None:
    pin = gate.PINS[name]
    path = gate.ROOT / pin["path"]
    raw = path.read_bytes()
    info = path.stat()
    assert len(raw) == pin["bytes"]
    assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
    assert f"{stat.S_IMODE(info.st_mode):04o}" == pin["mode"]
    assert info.st_nlink == 1


def test_context_receipts_replay_without_full_tree_walk() -> None:
    context = _load(gate.CONTEXT_RECEIPT_REL)
    snapshot = _load(gate.SNAPSHOT_RECEIPT_REL)
    assert gate.verify_context_receipts(context, snapshot) == {
        "payload_exact": True,
        "rootfs_exact": True,
        "symlink_closure_exact": True,
    }


def test_anchored_reader_accepts_exact_regular_pin(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    item = root / "item.json"
    item.write_bytes(b"{}")
    item.chmod(0o440)
    pin = {"path": "item.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest(), "mode": "0440"}
    with gate.AnchoredWorkspace(root) as workspace:
        assert workspace.read_pin(pin) == b"{}"


def test_anchored_reader_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.write_bytes(b"x")
    (root / "link").symlink_to("target")
    with gate.AnchoredWorkspace(root) as workspace:
        with pytest.raises(OSError):
            workspace.read_bytes("link")


def test_anchored_reader_rejects_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.write_bytes(b"x")
    os.link(target, root / "hard")
    with gate.AnchoredWorkspace(root) as workspace:
        with pytest.raises(gate.GateR1Error, match="hard link"):
            workspace.read_bytes("target")


def test_anchored_reader_absence_check_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with gate.AnchoredWorkspace(root) as workspace:
        workspace.verify_absent("missing")
        (root / "present").write_bytes(b"x")
        with pytest.raises(gate.GateR1Error, match="exists"):
            workspace.verify_absent("present")


def test_readonly_allowlist_contains_only_five_exact_commands() -> None:
    assert len(gate.READ_ONLY_COMMANDS) == 5
    assert all(gate.allowed_readonly_argv(argv) for argv in gate.READ_ONLY_COMMANDS)


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/docker", "buildx", "build", "-"],
        ["/usr/bin/docker", "build", "."],
        ["/usr/bin/docker", "run", "ubuntu"],
        ["/usr/bin/docker", "pull", "ubuntu"],
        ["/usr/bin/docker", "image", "inspect", "ubuntu:latest"],
    ],
)
def test_readonly_allowlist_rejects_build_run_pull_and_unpinned_inspect(argv: list[str]) -> None:
    assert gate.allowed_readonly_argv(argv) is False


def test_disallowed_command_is_rejected_before_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not be reached")

    monkeypatch.setattr(gate.subprocess, "run", forbidden)
    with pytest.raises(gate.GateR1Error, match="allowlist"):
        gate.run_readonly_command(gate.FUTURE_ARGV)
    assert called is False


def test_parse_docker_client_exact_projection() -> None:
    raw = b'{"Platform":{"Name":"Docker Engine - Community"},"Version":"29.6.2","ApiVersion":"1.55","DefaultAPIVersion":"1.55","GitCommit":"dfc4efb","GoVersion":"go1.26.5","Os":"linux","Arch":"amd64","Context":"default"}'
    assert gate.parse_docker_client(raw) == gate.EXPECTED_OBSERVATIONS["docker_client"]


@pytest.mark.parametrize("raw", [b"", b"github.com/docker/buildx v0.35.0 bad", b"github.com/docker/buildx 0.35.0 " + b"a" * 40])
def test_parse_buildx_version_rejects_malformed(raw: bytes) -> None:
    with pytest.raises(gate.GateR1Error):
        gate.parse_buildx_version(raw)


def test_parse_buildx_inspect_exact_projection() -> None:
    raw = b"""Name: default
Driver: docker
Nodes:
Name: default
Status: running
BuildKit version: v0.31.2
Platforms: linux/amd64, linux/amd64/v2, linux/amd64/v3, linux/amd64/v4
Devices:
 Name: nvidia.com/gpu=0
 Automatically allowed: false
 Name: nvidia.com/gpu=GPU-8cbaba1c-2629-a732-f528-66f459089ef6
 Automatically allowed: false
 Name: nvidia.com/gpu=all
 Automatically allowed: false
"""
    assert gate.parse_buildx_inspect(raw) == gate.EXPECTED_OBSERVATIONS["builder"]


def test_parse_buildx_inspect_rejects_automatic_gpu_permission() -> None:
    raw = b"""Name: default
Driver: docker
Status: running
BuildKit version: v0.31.2
Platforms: linux/amd64
Devices:
 Name: nvidia.com/gpu=0
 Automatically allowed: true
"""
    parsed = gate.parse_buildx_inspect(raw)
    observations = copy.deepcopy(gate.EXPECTED_OBSERVATIONS)
    observations["builder"] = parsed
    with pytest.raises(gate.GateR1Error):
        gate.validate_observations(observations)


def test_parse_image_inspect_exact_projection() -> None:
    raw = json.dumps({
        "Id": gate.BASE_DIGEST,
        "RepoDigests": [gate.BASE_REFERENCE],
        "Os": "linux",
        "Architecture": "amd64",
        "Size": 29_744_763,
    }).encode()
    assert gate.parse_image_inspect(raw) == gate.EXPECTED_OBSERVATIONS["base_image"]


def test_parse_dpkg_exact_and_rejects_extra_line() -> None:
    assert gate.parse_dpkg(b"5:29.6.2-1~ubuntu.24.04~noble|amd64\n") == gate.EXPECTED_OBSERVATIONS["docker_package"]
    with pytest.raises(gate.GateR1Error):
        gate.parse_dpkg(b"v|amd64\nextra")


def test_validate_observations_is_exact_object_equality() -> None:
    assert gate.validate_observations(copy.deepcopy(gate.EXPECTED_OBSERVATIONS))["exact"] is True
    changed = copy.deepcopy(gate.EXPECTED_OBSERVATIONS)
    changed["builder"]["driver"] = "docker-container"
    with pytest.raises(gate.GateR1Error):
        gate.validate_observations(changed)


def test_live_toolchain_observation_matches_pin_read_only() -> None:
    observed = gate.observe_toolchain()
    assert observed == gate.EXPECTED_OBSERVATIONS


def test_static_audit_remains_nonterminal(static_audit: dict) -> None:
    assert static_audit["status"] == "STATIC_AUTHOR_GATE_VALID_EXECUTION_CLOSED"
    assert static_audit["context"]["full_tree_replay_performed_now"] is False
    assert static_audit["execution_gate"] == gate.execution_gate()
    assert static_audit["claim_boundary"] == gate.claim_boundary()


def test_literal_authority_constants_are_false() -> None:
    assert gate.BUILD_EXECUTION_AUTHORIZED is False
    assert gate.DOCKER_AUTHORIZED is False
    assert gate.CONTEXT_COPY_AUTHORIZED is False
    assert gate.MODEL_OR_ONNX_LOAD_AUTHORIZED is False
    assert gate.GPU_AUTHORIZED is False
    assert gate.TENSORRT_OR_DEEPSTREAM_AUTHORIZED is False
    assert gate.QUALITY_AUTHORIZED is False
    assert gate.PRODUCTION_AUTHORIZED is False
    assert gate.INDEPENDENT_REVIEW_ACCEPTED is False


def test_execute_gate_always_rejects() -> None:
    with pytest.raises(gate.GateR1Error, match="nonterminal"):
        gate.require_execution_gate()
    assert gate.main(["execute"]) == 2


def test_source_has_no_forbidden_runtime_imports_or_process_surface() -> None:
    source = (gate.ROOT / gate.THIS_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            calls.append(node)
    assert not ({"torch", "onnx", "onnxruntime", "tensorrt", "cv2", "gi", "ultralytics"} & imports)
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr in {"Popen", "call", "check_call", "check_output"} for call in calls)
    assert not any(isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "os" and call.func.attr in {"system", "popen", "spawnl", "spawnv"} for call in calls)


def test_source_has_one_subprocess_run_inside_allowlisted_observer() -> None:
    source = (gate.ROOT / gate.THIS_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    run_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(run_calls) == 1
    node: ast.AST = run_calls[0]
    while node in parents and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node = parents[node]
    assert isinstance(node, ast.FunctionDef)
    assert node.name == "run_readonly_command"


def test_handoff_is_closed_nonterminal_and_fingerprinted(handoff: dict) -> None:
    result = gate.validate_handoff(handoff)
    assert result["valid"] is True
    assert result["nonterminal"] is True
    assert handoff["self_fingerprint"] == gate.fingerprint(handoff)
    assert handoff["status"].startswith("NONTERMINAL_")
    assert all(value is False for value in handoff["authority"].values())


def test_handoff_records_two_non_authoritative_test_runs(handoff: dict) -> None:
    runs = handoff["test_replay"]["runs"]
    assert [run["run"] for run in runs] == [1, 2]
    assert all(run["passed"] == 64 and run["failed"] == 0 for run in runs)
    assert handoff["test_replay"]["self_reported_non_authoritative"] is True
    assert handoff["test_replay"]["independent_replay_required"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "ACCEPT"),
        ("authority", {"build_execution": True}),
        ("execution_gate", {"authorized": True}),
        ("claim_boundary", {"build_executed": True}),
    ],
)
def test_handoff_rejects_authority_widening(handoff: dict, field: str, value: object) -> None:
    changed = copy.deepcopy(handoff)
    changed[field] = value
    changed["self_fingerprint"] = gate.fingerprint(changed)
    with pytest.raises(gate.GateR1Error):
        gate.validate_handoff(changed)


def test_handoff_builder_requires_exactly_two_runs(static_audit: dict) -> None:
    with pytest.raises(gate.GateR1Error, match="exactly twice"):
        gate.build_handoff(tests_run_count=1, tests_passed=64, prepared_at_utc=STAMP, audit=static_audit)


def test_handoff_publication_is_atomic_frozen_and_no_overwrite(tmp_path: Path, handoff: dict) -> None:
    destination = tmp_path / "candidate" / "handoff.json"
    receipt = gate.publish_handoff(handoff, destination)
    assert receipt["publication"] == "same_directory_renameat2_noreplace"
    assert destination.is_file()
    assert f"{stat.S_IMODE(destination.stat().st_mode):04o}" == "0440"
    assert f"{stat.S_IMODE(destination.parent.stat().st_mode):04o}" == "0550"
    assert gate.verify_handoff(destination)["nonterminal"] is True
    with pytest.raises(gate.GateR1Error, match="collision"):
        gate.publish_handoff(handoff, destination)


def test_handoff_publication_rejects_wrong_filename(tmp_path: Path, handoff: dict) -> None:
    with pytest.raises(gate.GateR1Error, match="filename"):
        gate.publish_handoff(handoff, tmp_path / "candidate" / "receipt.json")


def test_show_commands_never_require_toolchain_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess forbidden")

    monkeypatch.setattr(gate.subprocess, "run", forbidden)
    assert gate.main(["show-plan"]) == 0
    assert gate.main(["show-controller"]) == 0
