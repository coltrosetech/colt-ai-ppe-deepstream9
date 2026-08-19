from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import jsonschema
import pytest

from validation import person_rtdetrv4_tensorrt_r14i_independent_review as review


AUTHOR_MODULE = "validation.person_rtdetrv4_tensorrt_r14i"


@pytest.fixture(scope="session")
def audit() -> dict:
    return review.audit_current_state()


@pytest.fixture(scope="session")
def receipt(audit: dict) -> dict:
    if (review.ROOT / review.RECEIPT_REL).exists():
        return review.load_receipt()
    return review.build_receipt(audit, independent_tests=67)


@pytest.fixture(scope="session")
def schema() -> dict:
    with review.WorkspaceReader() as reader:
        pin = reader.pin_regular(review.SCHEMA_REL)
        return review.strict_json(reader.read_exact(pin, label="review schema"), "review schema")


def resign(value: dict) -> dict:
    value["review_fingerprint_sha256"] = review.fingerprint(
        value, "review_fingerprint_sha256"
    )
    return value


def test_strict_json_accepts_canonical_object() -> None:
    assert review.strict_json(b'{"a":1}', "ok") == {"a": 1}


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(review.R14IReviewError, match="duplicate JSON key"):
        review.strict_json(b'{"a":1,"a":2}', "duplicate")


def test_strict_json_rejects_nan() -> None:
    with pytest.raises(review.R14IReviewError, match="non-finite"):
        review.strict_json(b'{"a":NaN}', "nan")


def test_strict_json_rejects_infinity() -> None:
    with pytest.raises(review.R14IReviewError, match="non-finite"):
        review.strict_json(b'{"a":Infinity}', "infinity")


def test_strict_json_rejects_bom() -> None:
    with pytest.raises(review.R14IReviewError, match="envelope"):
        review.strict_json(b'\xef\xbb\xbf{"a":1}', "bom")


def test_strict_json_rejects_invalid_utf8() -> None:
    with pytest.raises(review.R14IReviewError, match="invalid JSON"):
        review.strict_json(b'{"a":"\xff"}', "utf8")


def test_strict_json_rejects_non_object_root() -> None:
    with pytest.raises(review.R14IReviewError, match="root differs"):
        review.strict_json(b"[]", "array")


@pytest.mark.parametrize("path", ["", "/absolute", "a/../b", "a\\b", "a/./b"])
def test_unsafe_paths_rejected(path: str) -> None:
    with pytest.raises(review.R14IReviewError, match="unsafe relative path"):
        review._parts(path)


def test_workspace_reader_replays_exact_regular(tmp_path: Path) -> None:
    target = tmp_path / "subject.bin"
    target.write_bytes(b"subject")
    target.chmod(0o440)
    pin = {
        "path": "subject.bin",
        "bytes": 7,
        "sha256": hashlib.sha256(b"subject").hexdigest(),
        "mode": "0440",
    }
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.read_exact(pin, label="subject") == b"subject"


def test_workspace_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "link").symlink_to(target.name)
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.pin_regular("link")


def test_workspace_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"x")
    (tmp_path / "link").symlink_to(real.name, target_is_directory=True)
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.pin_regular("link/file")


def test_workspace_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"x")
    os.link(first, second)
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.R14IReviewError, match="single-link"):
            reader.pin_regular("first")


def test_workspace_reader_rejects_mode_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "subject"
    target.write_bytes(b"x")
    target.chmod(0o600)
    pin = {
        "path": "subject",
        "bytes": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "mode": "0440",
    }
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.R14IReviewError, match="mode differs"):
            reader.read_exact(pin, label="mode")


def test_canonical_fingerprint_ignores_only_named_field() -> None:
    value = {"z": 1, "fingerprint": "ignored", "a": [2]}
    expected = hashlib.sha256(b'{"a":[2],"z":1}').hexdigest()
    assert review.fingerprint(value, "fingerprint") == expected


def test_exact_argv_hash_640() -> None:
    assert review.command_argv_sha256(review.expected_argv(640)) == (
        "47f91d4c3edd2e6fd9a22923c21b888b39e98fdf54cad4a054d0251738efd092"
    )


def test_exact_argv_hash_960() -> None:
    assert review.command_argv_sha256(review.expected_argv(960)) == (
        "e1d5c9c0a5ffe0159d66c894c02ed4f9a7a7bb79b37ee9fea2f4be79175e075e"
    )


@pytest.mark.parametrize("profile", [640, 960])
def test_expected_argv_is_networkless_readonly_uuid_bound(profile: int) -> None:
    argv = review.expected_argv(profile)
    assert "--network=none" in argv
    assert "--pull=never" in argv
    assert "--read-only" in argv
    assert f"--gpus=device={review.GPU_UUID}" in argv
    assert argv[13] == review.IMAGE_REFERENCE


@pytest.mark.parametrize("profile", [640, 960])
def test_expected_argv_has_batch_1_12_12_fp16(profile: int) -> None:
    argv = review.expected_argv(profile)
    assert "--fp16" in argv and "--noTF32" in argv
    assert f"--minShapes=images:1x3x{profile}x{profile},orig_target_sizes:1x2" in argv
    assert f"--optShapes=images:12x3x{profile}x{profile},orig_target_sizes:12x2" in argv
    assert f"--maxShapes=images:12x3x{profile}x{profile},orig_target_sizes:12x2" in argv


@pytest.mark.parametrize("profile", [640, 960])
def test_expected_output_set_has_nine_absent_targets(profile: int) -> None:
    outputs = review.expected_outputs(profile)
    assert len(outputs) == 9
    assert all(row == {
        "path": row["path"], "required_absent_now": True, "published_pin": None
    } for row in outputs.values())


def test_author_source_is_parsed_only() -> None:
    with review.WorkspaceReader() as reader:
        raw = reader.read_exact(
            review.AUTHOR_PINS["controller_source"], label="author source"
        )
    result = review.audit_author_source(raw)
    assert result["parsed_only_not_imported_or_executed"] is True
    assert result["execute_branch_calls_gate_then_has_no_adapter"] is True


def test_author_source_true_execution_flag_mutation_rejected() -> None:
    with review.WorkspaceReader() as reader:
        raw = reader.read_exact(
            review.AUTHOR_PINS["controller_source"], label="author source"
        )
    mutated = raw.replace(b"EXECUTION_AUTHORIZED = False", b"EXECUTION_AUTHORIZED = True", 1)
    with pytest.raises(review.R14IReviewError, match="execution flags differ"):
        review.audit_author_source(mutated)


def test_author_source_forbidden_import_mutation_rejected() -> None:
    with pytest.raises(review.R14IReviewError, match="imports execution surface"):
        review.audit_author_source(b"import torch\nEXECUTION_AUTHORIZED=False\n")


def test_author_source_filesystem_mutation_call_rejected() -> None:
    with review.WorkspaceReader() as reader:
        raw = reader.read_exact(
            review.AUTHOR_PINS["controller_source"], label="author source"
        )
    with pytest.raises(review.R14IReviewError, match="filesystem mutation"):
        review.audit_author_source(raw + b'\nos.remove("x")\n')


def test_author_source_writable_os_open_rejected() -> None:
    with review.WorkspaceReader() as reader:
        raw = reader.read_exact(
            review.AUTHOR_PINS["controller_source"], label="author source"
        )
    with pytest.raises(review.R14IReviewError, match="writable os.open"):
        review.audit_author_source(raw + b'\nos.open("x", os.O_WRONLY)\n')


def test_review_schema_is_draft_2020_12(schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_review_schema_is_recursively_closed(schema: dict) -> None:
    result = review.schemas_recursively_closed(schema)
    assert result["object_schemas"] >= 10
    assert result["open_object_schemas"] == 0


def test_handoff_exact_pin_and_fingerprint() -> None:
    with review.WorkspaceReader() as reader:
        value = review._load_json(reader, review.HANDOFF_PIN, "handoff")
    assert review.verify_handoff(value) == {
        "exact_keys": True,
        "self_fingerprint_valid": True,
        "terminal_accepted": False,
    }


def test_handoff_decision_mutation_rejected() -> None:
    with review.WorkspaceReader() as reader:
        value = review._load_json(reader, review.HANDOFF_PIN, "handoff")
    value["decision"] = "ACCEPT"
    with pytest.raises(review.R14IReviewError, match="terminal state differs"):
        review.verify_handoff(value)


def test_handoff_fingerprint_mutation_rejected() -> None:
    with review.WorkspaceReader() as reader:
        value = review._load_json(reader, review.HANDOFF_PIN, "handoff")
    value["self_fingerprint"]["canonical_sha256"] = "0" * 64
    with pytest.raises(review.R14IReviewError, match="self-fingerprint differs"):
        review.verify_handoff(value)


def test_full_audit_accepts_with_zero_findings(audit: dict) -> None:
    assert audit["decision"] == "ACCEPT"
    assert audit["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert audit["findings"] == []


def test_full_audit_replays_exact_r14h_lineage(audit: dict) -> None:
    assert audit["foundations"]["r14h"] == {
        "status": "frozen_accepted",
        "pins": review.R14H_PINS,
    }


def test_full_audit_keeps_preflight_scope_narrow(audit: dict) -> None:
    result = audit["foundations"]["deepstream91_preflight_r2"]
    assert result["decision"] == "ACCEPT"
    assert result["runtime_qualified"] is False
    assert result["lineage_files_rehashed"] == 13


def test_full_audit_keeps_r1c3_prepared_closed(audit: dict) -> None:
    result = audit["foundations"]["r1c3"]
    assert result["prepared_closed_only"] is True
    assert result["runtime_qualified"] is False
    assert result["person_onnx_repository_release_plan_and_embedded_probe_equal"] is True


def test_full_audit_keeps_gpu_lease_v5_locked(audit: dict) -> None:
    result = audit["foundations"]["gpu_lease_v5"]
    assert result["status"] == "locked-no-live-plan"
    assert result["live_plan_present"] is False
    assert result["activation_receipt_present"] is False


def test_full_audit_keeps_parser_abi_provenance_only(audit: dict) -> None:
    assert audit["controller"]["parser_abi_provenance_only"] is True
    assert audit["controller"]["parser_ds91_compatibility_accepted"] is False
    assert audit["model_inputs"]["parser_runtime_abi_accepted"] is False


@pytest.mark.parametrize("profile", ["640", "960"])
def test_full_audit_profiles_exact_and_outputs_absent(audit: dict, profile: str) -> None:
    result = audit["profiles"]["profiles"][profile]
    assert result["precision"] == {"fp16": True, "int8": False, "tf32": False}
    assert result["environment"] == {}
    assert len(result["mounts"]) == 1
    assert result["mounts"][0]["destination"] == "/output"
    assert result["mounts"][0]["read_only"] is False
    assert result["all_outputs_absent"] is True


def test_full_audit_all_execution_gates_false(audit: dict) -> None:
    assert audit["controller"]["execution_gate_all_false"] is True
    assert audit["profiles"]["execution_gate_all_false"] is True
    assert audit["remaining_gates"] == review.REQUIRED_GATES
    assert all(value is False for value in audit["permissions"].values())


def test_author_module_was_never_imported(audit: dict) -> None:
    assert audit["author_surface"]["parsed_only_not_imported_or_executed"] is True
    assert AUTHOR_MODULE not in sys.modules


def test_built_receipt_validates_against_schema(audit: dict, schema: dict) -> None:
    built = review.build_receipt(audit, independent_tests=30)
    review._validate_schema(built, schema, "built receipt")


def test_built_receipt_verifies_without_subject_replay(audit: dict) -> None:
    built = review.build_receipt(audit, independent_tests=30)
    assert review.verify_receipt(built)["execution_authorized"] is False


def test_published_receipt_fingerprint_is_valid(receipt: dict) -> None:
    assert receipt["review_fingerprint_sha256"] == review.fingerprint(
        receipt, "review_fingerprint_sha256"
    )


def test_published_receipt_replays_entire_subject(receipt: dict) -> None:
    result = review.verify_receipt(receipt, replay_subject=True)
    assert result["decision"] == "ACCEPT"
    assert result["replayed_subject"] is True
    assert result["execution_authorized"] is False


def test_receipt_authority_overclaim_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["authority_boundary"]["runtime_or_build_authority_granted"] = True
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_p0_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["severity_counts"]["P0"] = 1
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_finding_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["findings"] = [{"severity": "P2"}]
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_runtime_scope_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["accepted_scope"]["runtime_execution_authorized"] = True
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_removed_gate_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["remaining_gates"] = value["remaining_gates"][:-1]
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_gpu_permission_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["permissions"]["call_gpu"] = True
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_author_import_boundary_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["resource_boundary"]["author_code_imported"] = True
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_author_pin_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["subject"]["author_artifacts"]["controller_source"]["sha256"] = "0" * 64
    with pytest.raises(review.R14IReviewError, match="author pins differ"):
        review.verify_receipt(resign(value))


def test_receipt_author_surface_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["subject"]["author_surface"]["parsed_only_not_imported_or_executed"] = False
    with pytest.raises(review.R14IReviewError):
        review.verify_receipt(resign(value))


def test_receipt_test_count_mismatch_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["independent_test_replay"]["passed"] -= 1
    with pytest.raises(review.R14IReviewError, match="test replay differs"):
        review.verify_receipt(resign(value))


def test_receipt_fingerprint_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["review_fingerprint_sha256"] = "0" * 64
    with pytest.raises(review.R14IReviewError, match="fingerprint differs"):
        review.verify_receipt(value)


def test_receipt_unknown_root_field_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["unexpected"] = False
    with pytest.raises(review.R14IReviewError, match="schema validation failed"):
        review.verify_receipt(resign(value))


def test_receipt_profile_replay_mutation_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["verification"]["profiles"]["profiles"]["640"]["docker_argv"][2] = "--network=host"
    with pytest.raises(review.R14IReviewError, match="profile replay differs"):
        review.verify_receipt(resign(value), replay_subject=True)


def test_receipt_r1c3_runtime_overclaim_rejected(receipt: dict) -> None:
    value = copy.deepcopy(receipt)
    value["verification"]["foundations"]["r1c3"]["runtime_qualified"] = True
    with pytest.raises(review.R14IReviewError, match="foundation replay differs"):
        review.verify_receipt(resign(value), replay_subject=True)


def test_publish_is_readonly_and_noreplace(tmp_path: Path, audit: dict) -> None:
    built = review.build_receipt(audit, independent_tests=30)
    container = tmp_path / "container"
    container.mkdir()
    output = container / "run" / "receipt.json"
    size, digest = review.publish_receipt(output, built, expected_output=output)
    try:
        assert size == output.stat().st_size
        assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
        assert stat.S_IMODE(output.stat().st_mode) == 0o440
        assert stat.S_IMODE(output.parent.stat().st_mode) == 0o550
        with pytest.raises(review.R14IReviewError, match="already exists"):
            review.publish_receipt(output, built, expected_output=output)
    finally:
        output.parent.chmod(0o750)
        output.chmod(0o640)


def test_published_receipt_file_is_single_link_readonly() -> None:
    info = (review.ROOT / review.RECEIPT_REL).lstat()
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert stat.S_IMODE(info.st_mode) == 0o440


def test_review_controls_have_frozen_modes(audit: dict) -> None:
    modes = {row["path"]: row["mode"] for row in audit["review_control_pins"]}
    assert modes == {
        review.VERIFIER_REL: "0555",
        review.SCHEMA_REL: "0440",
        review.TEST_REL: "0440",
        review.DOC_REL: "0440",
    }
