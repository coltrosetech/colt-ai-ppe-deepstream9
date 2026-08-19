from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import jsonschema
import pytest

from validation import deepstream91_dgpu_runtime_smoke_r1 as smoke


WORKER_MODULE = "validation.deepstream91_dgpu_runtime_smoke_probe_r1"
V5_MODULE = "validation.gpu_lease_v5"


@pytest.fixture(scope="session")
def audit() -> dict:
    return smoke.audit_current_state()


@pytest.fixture(scope="session")
def handoff(audit: dict) -> dict:
    if (smoke.ROOT / smoke.HANDOFF_REL).exists():
        return smoke.load_handoff()
    return smoke.build_handoff(audit, independent_test_count=50)


@pytest.fixture(scope="session")
def handoff_schema() -> dict:
    with smoke.RepoReader() as reader:
        current = reader.current_pin(smoke.HANDOFF_SCHEMA_REL)
        return smoke.strict_json(reader.read_exact(current, "handoff schema"), "handoff schema")


@pytest.fixture(scope="session")
def result_schema() -> dict:
    with smoke.RepoReader() as reader:
        return smoke.strict_json(
            reader.read_exact(smoke.RESULT_SCHEMA_PIN, "result schema"), "result schema"
        )


@pytest.fixture(scope="session")
def worker_raw() -> bytes:
    with smoke.RepoReader() as reader:
        return reader.read_exact(smoke.WORKER_PIN, "probe worker")


def resign(value: dict) -> dict:
    value["self_fingerprint"]["canonical_sha256"] = smoke.fingerprint(
        value, "self_fingerprint"
    )
    return value


def test_strict_json_accepts_object() -> None:
    assert smoke.strict_json(b'{"a":1}', "ok") == {"a": 1}


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="duplicate JSON key"):
        smoke.strict_json(b'{"a":1,"a":2}', "duplicate")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite(token: bytes) -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="non-finite"):
        smoke.strict_json(b'{"a":' + token + b"}", "nonfinite")


def test_strict_json_rejects_bom() -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="envelope"):
        smoke.strict_json(b'\xef\xbb\xbf{"a":1}', "bom")


def test_strict_json_rejects_nonobject() -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="root differs"):
        smoke.strict_json(b"[]", "array")


def test_strict_json_rejects_invalid_utf8() -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="invalid JSON"):
        smoke.strict_json(b'{"a":"\xff"}', "utf8")


@pytest.mark.parametrize("path", ["", "/absolute", "a/../b", "a/./b", "a\\b", "a//b"])
def test_relative_path_rejects_unsafe_forms(path: str) -> None:
    with pytest.raises(smoke.SmokeAuthorError, match="unsafe relative path"):
        smoke._raw_parts(path)


def test_repo_reader_exact_regular(tmp_path: Path) -> None:
    target = tmp_path / "subject"
    target.write_bytes(b"subject")
    target.chmod(0o440)
    expected = smoke.pin("subject", 7, hashlib.sha256(b"subject").hexdigest(), "0440")
    with smoke.RepoReader(tmp_path) as reader:
        assert reader.read_exact(expected, "subject") == b"subject"


def test_repo_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "link").symlink_to(target.name)
    with smoke.RepoReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.current_pin("link")


def test_repo_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "file").write_bytes(b"x")
    (tmp_path / "link").symlink_to(directory.name, target_is_directory=True)
    with smoke.RepoReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.current_pin("link/file")


def test_repo_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"x")
    os.link(first, second)
    with smoke.RepoReader(tmp_path) as reader:
        with pytest.raises(smoke.SmokeAuthorError, match="control identity differs"):
            reader.current_pin("first")


@pytest.mark.parametrize("name", ["python_interpreter", "docker_cli", "nvidia_smi"])
def test_current_absolute_tool_pin_replays_without_execution(name: str) -> None:
    expected = smoke.CURRENT_ABSOLUTE_TOOL_PINS[name]
    assert smoke.read_absolute_pin(expected, name) == expected


def test_docker_pin_drift_is_exact() -> None:
    assert smoke.V5_EXPECTED_ABSOLUTE_TOOL_PINS["docker_cli"] == {
        "path": "/usr/bin/docker",
        "bytes": 45410374,
        "sha256": "aa5d6553089f1139456466c754d1f45d064dc2762c77c403d3bcc0e399eb4e1d",
    }
    assert smoke.CURRENT_ABSOLUTE_TOOL_PINS["docker_cli"] == {
        "path": "/usr/bin/docker",
        "bytes": 45356984,
        "sha256": "628af575ee8499596e7d266c221ee6bb74fbc20de3a81dd6fe5106f81e652db4",
    }


def test_nvidia_smi_pin_drift_is_exact() -> None:
    assert smoke.V5_EXPECTED_ABSOLUTE_TOOL_PINS["nvidia_smi"] == {
        "path": "/usr/bin/nvidia-smi",
        "bytes": 1260192,
        "sha256": "78c2d203c12619673ddd5ae9c487a79689c9461de40a9b948f3f4615f26dba29",
    }
    assert smoke.CURRENT_ABSOLUTE_TOOL_PINS["nvidia_smi"] == {
        "path": "/usr/bin/nvidia-smi",
        "bytes": 1259616,
        "sha256": "7896b7cdd9cb84b1e0fbc4baf91dfa3039b44ab6641cd08d7cc1dd23f81d6deb",
    }


def test_python_interpreter_pin_did_not_drift() -> None:
    assert smoke.CURRENT_ABSOLUTE_TOOL_PINS["python_interpreter"] == (
        smoke.V5_EXPECTED_ABSOLUTE_TOOL_PINS["python_interpreter"]
    )


def test_exact_docker_argv_digest() -> None:
    argv = smoke.expected_docker_argv()
    assert len(argv) == 25
    assert smoke.command_argv_sha256(argv) == smoke.PLAN_ARGV_SHA256


def test_exact_docker_image_position_and_digest() -> None:
    argv = smoke.expected_docker_argv()
    assert argv[19] == smoke.IMAGE_REFERENCE
    assert argv.count(smoke.IMAGE_REFERENCE) == 1
    assert all(item.startswith("-") for item in argv[2:19])


def test_docker_envelope_is_foreground_without_rm() -> None:
    argv = smoke.expected_docker_argv()
    assert "--rm" not in argv
    assert "--detach" not in argv
    assert "-d" not in argv
    assert argv[:2] == ["/usr/bin/docker", "run"]


def test_docker_envelope_has_no_network_or_pull() -> None:
    argv = smoke.expected_docker_argv()
    assert "--network=none" in argv
    assert "--pull=never" in argv
    assert "--runtime=runc" in argv


def test_docker_envelope_has_exact_gpu_identity() -> None:
    argv = smoke.expected_docker_argv()
    assert f"--gpus=device={smoke.GPU_UUID}" in argv
    assert f"--env=NVIDIA_VISIBLE_DEVICES={smoke.GPU_UUID}" in argv
    assert "--env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" in argv


def test_docker_envelope_has_hardened_privileges() -> None:
    argv = smoke.expected_docker_argv()
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert "--ipc=none" in argv
    assert "--pids-limit=128" in argv
    assert "--user=1000:1000" in argv


def test_docker_envelope_has_only_expected_bind_mounts() -> None:
    mounts = [item for item in smoke.expected_docker_argv() if item.startswith("--mount=")]
    assert len(mounts) == 2
    assert sum("dst=/output,rw" in item for item in mounts) == 1
    assert sum("dst=/opt/deepsafe/runtime-smoke/probe.py,readonly" in item for item in mounts) == 1


def test_worker_is_audited_as_inert_ast(worker_raw: bytes) -> None:
    result = smoke.audit_worker_source(worker_raw)
    assert result["parsed_as_inert_ast"] is True
    assert result["imported"] is False
    assert result["executed"] is False
    assert result["shell_execution"] is False
    assert result["subprocess_call_count"] == 1


@pytest.mark.parametrize("probe_id", sorted(smoke.EXPECTED_PROBE_COMMANDS))
def test_worker_probe_commands_are_exact(worker_raw: bytes, probe_id: str) -> None:
    result = smoke.audit_worker_source(worker_raw)
    observed = result["probe_commands"][probe_id]
    expected = smoke.EXPECTED_PROBE_COMMANDS[probe_id]
    assert observed["argv"] == expected
    assert observed["argv_sha256"] == smoke.command_argv_sha256(expected)


def test_worker_network_import_mutation_rejected(worker_raw: bytes) -> None:
    mutated = worker_raw.replace(b"import subprocess\n", b"import socket\nimport subprocess\n", 1)
    with pytest.raises(smoke.SmokeAuthorError, match="forbidden runtime"):
        smoke.audit_worker_source(mutated)


def test_worker_shell_mutation_rejected(worker_raw: bytes) -> None:
    mutated = worker_raw.replace(b"close_fds=True,", b"shell=True,\n            close_fds=True,", 1)
    with pytest.raises(smoke.SmokeAuthorError, match="shell execution"):
        smoke.audit_worker_source(mutated)


def test_worker_probe_command_mutation_rejected(worker_raw: bytes) -> None:
    mutated = worker_raw.replace(b'"nvinfer"', b'"fakeinfer"', 1)
    with pytest.raises(smoke.SmokeAuthorError, match="command envelope differs"):
        smoke.audit_worker_source(mutated)


def test_worker_and_v5_modules_were_never_imported(audit: dict) -> None:
    assert audit["resource_boundary"]["probe_worker_imported"] is False
    assert audit["resource_boundary"]["gpu_lease_v5_imported"] is False
    assert WORKER_MODULE not in sys.modules
    assert V5_MODULE not in sys.modules


def test_result_schema_is_valid_and_closed(result_schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(result_schema)
    result = smoke.schemas_recursively_closed(result_schema)
    assert result["object_schemas"] >= 5
    assert result["open_object_schemas"] == 0


def test_result_schema_forbids_inference_overclaim(result_schema: dict) -> None:
    boundary = result_schema["properties"]["execution_boundary"]["properties"]
    assert boundary["model_loaded"] == {"const": False}
    assert boundary["checkpoint_deserialized"] == {"const": False}
    assert boundary["onnx_deserialized"] == {"const": False}
    assert boundary["inference_executed"] == {"const": False}
    assert boundary["engine_built"] == {"const": False}


def test_handoff_schema_is_valid_and_closed(handoff_schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(handoff_schema)
    result = smoke.schemas_recursively_closed(handoff_schema)
    assert result["object_schemas"] >= 10
    assert result["open_object_schemas"] == 0


def test_r7_current_boot_driver_and_gpu_replayed(audit: dict) -> None:
    result = audit["foundations"]["driver595_r7_current_boot"]
    assert result["decision"] == "ACCEPT"
    assert result["current_boot_replayed"] is True
    assert result["boot_id"] == smoke.BOOT_ID
    assert result["driver_version"] == smoke.DRIVER_VERSION
    assert result["gpu_uuid"] == smoke.GPU_UUID
    assert result["deepstream_or_gpu_workload_authorized"] is False


def test_native_oci_manifest_and_config_are_exact(audit: dict) -> None:
    result = audit["foundations"]["deepstream91_native_image"]
    assert result["repo_digest_operand"] == smoke.IMAGE_REFERENCE
    assert result["manifest_digest"] == smoke.IMAGE_DIGEST
    assert result["config_digest"] == smoke.IMAGE_CONFIG_DIGEST
    assert result["local_oci_manifest_exact"] is True
    assert result["docker_cache_or_registry_resolution_performed"] is False
    assert result["gpu_runtime_qualified"] is False


def test_r2_acceptance_remains_cpu_definition_only(audit: dict) -> None:
    result = audit["foundations"]["deepstream91_preflight_r2"]
    assert result["decision"] == "ACCEPT"
    assert result["cpu_definition_only"] is True
    assert result["runtime_qualified"] is False


def test_v5_contract_remains_locked_and_not_activation_ready(audit: dict) -> None:
    result = audit["foundations"]["gpu_lease_v5"]
    assert result["contract_status"] == "locked-no-live-plan"
    assert result["global_default_and_published_plan"] is None
    assert result["host_tool_pins_match_contract"] is False
    assert result["docker_cli_pin_drift"] is True
    assert result["nvidia_smi_pin_drift"] is True
    assert result["activation_ready"] is False
    assert result["activation_authorized"] is False
    assert result["launch_authorized"] is False
    assert result["execution_authorized"] is False


def test_v6_successor_is_a_hard_gate(audit: dict) -> None:
    assert audit["findings"] == [smoke.HOST_TOOL_DRIFT_FINDING]
    assert smoke.HOST_TOOL_DRIFT_FINDING["required_gate"] == "gpu_lease_v6_successor_acceptance"
    assert "gpu_lease_v6_successor_acceptance" in audit["remaining_gates"]


def test_static_v5_candidate_plan_validates_but_is_not_outer_authority(audit: dict) -> None:
    result = audit["foundations"]["gpu_lease_v5"]
    assert result["inner_plan_execution_authorized_required_literal"] is True
    assert result["outer_acceptance"] is False
    assert result["independent_review_required"] is True
    assert result["activation_authorized"] is False


def test_static_plan_fingerprint_and_argv_are_exact() -> None:
    with smoke.RepoReader() as reader:
        plan = smoke.strict_json(reader.read_exact(smoke.PLAN_PIN, "plan"), "plan")
        schema = smoke.strict_json(
            reader.read_exact(smoke.LEASE_PINS["v5_plan_schema"], "V5 schema"), "V5 schema"
        )
    smoke.validate_schema(plan, schema, "plan")
    unsigned = dict(plan)
    unsigned.pop("plan_fingerprint_sha256")
    assert smoke.canonical_sha256(unsigned) == smoke.PLAN_FINGERPRINT
    assert plan["workload"]["argv"] == smoke.expected_docker_argv()
    assert plan["workload"]["argv_sha256"] == smoke.PLAN_ARGV_SHA256


def test_output_tree_is_absent_and_materialization_closed(audit: dict) -> None:
    boundary = audit["output_boundary"]
    assert boundary["tree_absent"] is True
    assert boundary["materialization_authorized"] is False
    path = smoke.ROOT / smoke.OUTPUT_TREE_REL
    assert not path.exists() and not path.is_symlink()


def test_author_audit_has_one_blocking_p1(audit: dict) -> None:
    assert audit["decision"] is None
    assert audit["severity_counts"] == {"P0": 0, "P1": 1, "P2": 0}
    assert audit["findings"] == [smoke.HOST_TOOL_DRIFT_FINDING]


def test_author_permissions_are_all_closed(audit: dict) -> None:
    assert audit["permissions"] == smoke.PERMISSIONS
    assert all(value is False for value in audit["permissions"].values())


def test_author_resource_boundary_records_no_runtime_calls(audit: dict) -> None:
    boundary = audit["resource_boundary"]
    assert boundary["author_controller_cpu_only"] is True
    assert boundary["proc_boot_id_read_only"] is True
    assert all(
        boundary[key] is False
        for key in set(boundary) - {"author_controller_cpu_only", "proc_boot_id_read_only"}
    )


def test_candidate_handoff_validates_and_verifies(handoff: dict, handoff_schema: dict) -> None:
    smoke.validate_schema(handoff, handoff_schema, "handoff")
    result = smoke.verify_handoff(handoff)
    assert result["outer_acceptance"] is False
    assert result["launch_authorized"] is False
    assert result["execution_authorized"] is False


def test_candidate_handoff_full_subject_replay(handoff: dict) -> None:
    result = smoke.verify_handoff(handoff, replay_subject=True)
    assert result["replayed_subject"] is True
    assert result["decision"] is None


def test_handoff_launch_overclaim_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["authority"]["launch_authorized"] = True
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_activation_overclaim_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["runtime_plan"]["activation_authorized"] = True
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_blocker_removal_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["findings"] = []
    value["severity_counts"]["P1"] = 0
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_v6_gate_removal_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["remaining_gates"].remove("gpu_lease_v6_successor_acceptance")
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_docker_permission_mutation_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["permissions"]["docker_run"] = True
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_worker_import_boundary_mutation_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["resource_boundary"]["probe_worker_imported"] = True
    with pytest.raises(smoke.SmokeAuthorError):
        smoke.verify_handoff(resign(value))


def test_handoff_network_host_argv_mutation_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["runtime_plan"]["argv"][3] = "--network=host"
    with pytest.raises(smoke.SmokeAuthorError, match="runtime command"):
        smoke.verify_handoff(resign(value))


def test_handoff_plan_pin_mutation_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["runtime_plan"]["v5_candidate_plan"]["sha256"] = "0" * 64
    with pytest.raises(smoke.SmokeAuthorError, match="runtime command"):
        smoke.verify_handoff(resign(value))


def test_handoff_double_test_mismatch_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["author_tests"]["run_2"]["passed"] -= 1
    with pytest.raises(smoke.SmokeAuthorError, match="double test replay"):
        smoke.verify_handoff(resign(value))


def test_handoff_fingerprint_mutation_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["self_fingerprint"]["canonical_sha256"] = "0" * 64
    with pytest.raises(smoke.SmokeAuthorError, match="fingerprint differs"):
        smoke.verify_handoff(value)


def test_handoff_unknown_root_field_rejected(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["unknown"] = False
    with pytest.raises(smoke.SmokeAuthorError, match="schema validation failed"):
        smoke.verify_handoff(resign(value))


def test_handoff_foundation_forgery_rejected_on_replay(handoff: dict) -> None:
    value = copy.deepcopy(handoff)
    value["foundations"]["gpu_lease_v5"]["activation_ready"] = True
    with pytest.raises(smoke.SmokeAuthorError, match="foundations replay differs"):
        smoke.verify_handoff(resign(value), replay_subject=True)


def test_publish_is_readonly_and_noreplace(tmp_path: Path, audit: dict) -> None:
    value = smoke.build_handoff(audit, independent_test_count=50)
    container = tmp_path / "container"
    container.mkdir()
    output = container / "run" / "handoff.json"
    size, digest = smoke.publish_handoff(output, value, expected_output=output)
    try:
        assert size == output.stat().st_size
        assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
        assert stat.S_IMODE(output.stat().st_mode) == 0o440
        assert stat.S_IMODE(output.parent.stat().st_mode) == 0o550
        with pytest.raises(smoke.SmokeAuthorError, match="already exists"):
            smoke.publish_handoff(output, value, expected_output=output)
    finally:
        output.parent.chmod(0o750)
        output.chmod(0o640)


def test_published_handoff_is_single_link_readonly() -> None:
    info = (smoke.ROOT / smoke.HANDOFF_REL).lstat()
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o440
    assert stat.S_IMODE((smoke.ROOT / smoke.HANDOFF_RUN_REL).stat().st_mode) == 0o550


def test_author_controls_have_expected_frozen_modes(audit: dict) -> None:
    modes = {item["path"]: item["mode"] for item in audit["author_artifacts"]}
    assert modes == {
        smoke.CONTROLLER_REL: "0555",
        smoke.WORKER_REL: "0555",
        smoke.PLAN_REL: "0440",
        smoke.RESULT_SCHEMA_REL: "0440",
        smoke.HANDOFF_SCHEMA_REL: "0440",
        smoke.TEST_REL: "0440",
        smoke.DOC_REL: "0440",
    }


def test_controller_has_no_runtime_subprocess_surface() -> None:
    tree = ast.parse((smoke.ROOT / smoke.CONTROLLER_REL).read_text("utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "subprocess" not in imports
    assert "docker" not in imports
    assert "socket" not in imports
