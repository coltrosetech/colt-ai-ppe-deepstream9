from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r1 as author
from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r1_independent_review as review


@pytest.fixture(scope="session")
def audit() -> dict[str, object]:
    return review.audit_current_state()


@pytest.fixture(scope="session")
def receipt() -> dict[str, object]:
    return review.load_review()


def _resign_handoff(value: dict[str, object]) -> dict[str, object]:
    value["self_fingerprint"] = author.fingerprint(value)
    return value


def _resign_review(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def test_strict_json_rejects_duplicate_nonfinite_bom_bad_utf8_and_nonobject() -> None:
    samples = (
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b"\xef\xbb\xbf{}",
        b'{"x":"\xff"}',
        b"[]",
    )
    for raw in samples:
        with pytest.raises(review.IndependentReviewError):
            review.strict_json(raw, label="fixture")


def test_relative_path_parser_rejects_escape_absolute_alias_and_nul() -> None:
    for value in ("", "/x", "../x", "a/../x", "a//b", "a/./b", "a\\b", "a\x00b"):
        with pytest.raises(review.IndependentReviewError, match="unsafe relative path"):
            review._parts(value)


def test_independent_reader_binds_exact_bytes_and_metadata(tmp_path: Path) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"exact")
    target.chmod(0o440)
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.pin_regular("item") == {
            "path": "item",
            "bytes": 5,
            "sha256": hashlib.sha256(b"exact").hexdigest(),
            "mode": "0440",
        }


def test_independent_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "alias").symlink_to("target")
    os.link(target, tmp_path / "hard")
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.pin_regular("alias")
        with pytest.raises(review.IndependentReviewError, match="single-link"):
            reader.pin_regular("target")


def test_independent_reader_rejects_final_name_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"a" * (1024 * 1024 + 1))
    target.chmod(0o440)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"b" * (1024 * 1024 + 1))
    replacement.chmod(0o440)
    real_pread = review.os.pread
    replaced = False

    def raced(fd: int, length: int, offset: int) -> bytes:
        nonlocal replaced
        chunk = real_pread(fd, length, offset)
        if offset == 0 and not replaced:
            replacement.replace(target)
            replaced = True
        return chunk

    monkeypatch.setattr(review.os, "pread", raced)
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.IndependentReviewError, match="inode changed"):
            reader.pin_regular("item", limit=2 * 1024 * 1024)


def test_exact_author_closure_raw_pins_replay(audit: dict[str, object]) -> None:
    assert audit["author_closure"] == {**review.AUTHOR_PINS, "author_handoff": review.AUTHOR_HANDOFF_PIN}


def test_plan_schema_is_valid_exact_object_const(audit: dict[str, object]) -> None:
    schema = json.loads((review.ROOT / review.PLAN_SCHEMA_REL).read_text(encoding="utf-8"))
    plan = json.loads((review.ROOT / review.PLAN_REL).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["const"] == plan
    assert audit["plan"] == {
        "self_fingerprint": review.PLAN_FINGERPRINT,
        "schema_exact_object_const": True,
        "argv_sha256": review.FUTURE_ARGV_SHA256,
    }


def test_plan_schema_rejects_authority_and_future_argv_drift() -> None:
    schema = json.loads((review.ROOT / review.PLAN_SCHEMA_REL).read_text(encoding="utf-8"))
    plan = json.loads((review.ROOT / review.PLAN_REL).read_text(encoding="utf-8"))
    authority = copy.deepcopy(plan)
    authority["authority"]["build_execution"] = True
    argv = copy.deepcopy(plan)
    argv["future_build_contract"]["invocation"]["argv"].append("--pull=true")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(authority)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(argv)


def test_author_handoff_actual_pin_fingerprint_and_authority_are_exact(audit: dict[str, object]) -> None:
    handoff = audit["author_handoff"]
    assert handoff["receipt"] == review.AUTHOR_HANDOFF_PIN
    assert handoff["self_fingerprint"] == review.HANDOFF_FINGERPRINT
    assert handoff["authority_all_false"] is True
    assert [row["passed"] for row in handoff["test_runs"]] == [84, 84]


def test_a32_foundation_and_context_tree_lineage_are_exact(audit: dict[str, object]) -> None:
    assert audit["foundation"]["decision"] == "ACCEPT"
    assert audit["foundation"]["review_fingerprint_sha256"] == review.FOUNDATION_FINGERPRINT
    lineage = audit["context_lineage"]
    assert lineage["full_tree"] == review.EXPECTED_TREES["full"]
    assert lineage["payload_tree"] == review.EXPECTED_TREES["payload"]
    assert lineage["rootfs_tree"] == review.EXPECTED_TREES["rootfs"]
    assert lineage["symlink_closure"] == review.EXPECTED_SYMLINK_CLOSURE
    assert lineage["full_tree_replayed_now"] is False
    assert lineage["fresh_pre_stream_full_tree_replay_required"] is True


def test_toolchain_captured_versions_direct_binary_pins_and_freshness_boundary(audit: dict[str, object]) -> None:
    toolchain = audit["toolchain"]
    assert toolchain["docker_client_version"] == "29.6.2"
    assert toolchain["buildx_version"] == "v0.35.0"
    assert toolchain["buildkit_version"] == "v0.31.2"
    assert toolchain["base_image_digest"] == review.BASE_DIGEST
    assert toolchain["docker_binary"] == review.DOCKER_PIN
    assert toolchain["buildx_binary"] == review.BUILDX_PIN
    assert toolchain["reviewer_subprocess_calls"] is False
    assert toolchain["fresh_execution_time_replay_still_required"] is True


def test_future_build_stage_final_and_execution_receipt_are_absent(audit: dict[str, object]) -> None:
    assert audit["output_absence"] == {"paths": list(review.OUTPUT_ABSENCE_PATHS), "all_absent": True}


def test_author_ast_execute_path_is_fail_closed_and_observer_is_bounded(audit: dict[str, object]) -> None:
    source = audit["source_audit"]
    assert source["execute_branch_calls"] == ["require_execution_gate"]
    assert source["require_execution_gate_unconditional_raise"] is True
    assert source["build_adapter_present"] is False
    assert source["subprocess_run_count"] == 1
    assert source["subprocess_run_enclosing_function"] == "run_readonly_command"
    assert source["read_only_command_allowlist_size"] == 5
    assert source["forbidden_imports"] == []
    assert source["forbidden_calls"] == []


def test_reviewer_source_has_no_subprocess_writer_network_model_or_gpu_surface() -> None:
    source = inspect.getsource(review)
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports.intersection({"subprocess", "socket", "requests", "torch", "onnx", "onnxruntime", "tensorrt", "ultralytics"})
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls.intersection({"run", "Popen", "system", "mkdir", "write_text", "write_bytes", "rename", "replace", "unlink", "chmod"})
    assert "O_CREAT" not in source and "O_WRONLY" not in source and "O_TRUNC" not in source


def test_p1_001_author_validator_accepts_self_resigned_test_overclaim() -> None:
    handoff = json.loads((review.ROOT / review.AUTHOR_HANDOFF_REL).read_text(encoding="utf-8"))
    for row in handoff["test_replay"]["runs"]:
        row.update({"command": "true", "collected": 1, "passed": 1})
    _resign_handoff(handoff)
    assert author.validate_handoff(handoff)["valid"] is True
    with pytest.raises(review.IndependentReviewError, match="fingerprint differs"):
        review.validate_handoff(handoff)


def test_p1_001_author_validator_accepts_self_resigned_future_contract_widening() -> None:
    handoff = json.loads((review.ROOT / review.AUTHOR_HANDOFF_REL).read_text(encoding="utf-8"))
    handoff["future_exact_build"].update({"pull": True, "network": "host"})
    _resign_handoff(handoff)
    assert author.validate_handoff(handoff)["valid"] is True
    with pytest.raises(review.IndependentReviewError, match="fingerprint differs"):
        review.validate_handoff(handoff)


def test_p1_001_ast_reproduces_missing_exact_nested_checks(audit: dict[str, object]) -> None:
    source = audit["source_audit"]
    assert source["handoff_validator_missing_exact_string_checks"] == ["command", "network", "pull"]
    assert source["handoff_validator_missing_exact_test_count_84"] is True


def test_p1_002_author_reader_accepts_split_inode_pin(tmp_path: Path) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"good")
    target.chmod(0o600)
    with author.AnchoredWorkspace(tmp_path) as workspace:
        original = workspace.read_bytes

        def raced(relative: str, **kwargs: object) -> bytes:
            raw = original(relative, **kwargs)
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"evil")
            replacement.chmod(0o440)
            replacement.replace(target)
            return raw

        workspace.read_bytes = raced  # type: ignore[method-assign]
        pin = {"path": "item", "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest(), "mode": "0440"}
        assert workspace.read_pin(pin) == b"good"
    assert target.read_bytes() == b"evil"
    assert f"{stat.S_IMODE(target.stat().st_mode):04o}" == "0440"


def test_p1_002_ast_reproduces_split_read_and_stat_flow(audit: dict[str, object]) -> None:
    assert audit["source_audit"]["read_pin_content_and_mode_from_separate_operations"] is True
    assert audit["adversarial_results"]["split_inode_good_bytes_evil_metadata_accepted_by_author_reader"] is True


def test_review_schema_is_draft_2020_closed_and_binds_reject() -> None:
    schema = json.loads((review.ROOT / review.REVIEW_SCHEMA_REL).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["decision"]["const"] == "REJECT"
    assert schema["properties"]["authority"]["const"] == review.REJECT_AUTHORITY
    assert schema["properties"]["findings"]["const"] == review.FINDINGS


def test_receipt_is_strict_self_fingerprinted_terminal_reject(receipt: dict[str, object]) -> None:
    result = review.validate_review(receipt, replay_subject=True)
    assert result["valid"] is True
    assert result["decision"] == "REJECT"
    assert result["authority_all_false"] is True
    assert receipt["review_fingerprint_sha256"] == review.review_fingerprint(receipt)
    assert receipt["status"] == "TERMINAL_INDEPENDENT_R1_REJECTED_NO_BUILD_AUTHORITY"


def test_receipt_has_two_blocking_p1_findings_and_no_authority(receipt: dict[str, object]) -> None:
    assert receipt["severity_counts"] == {"P0": 0, "P1": 2, "P2": 0}
    assert [item["id"] for item in receipt["findings"]] == ["PPE-CPU-IMAGE-GATE-R1-P1-001", "PPE-CPU-IMAGE-GATE-R1-P1-002"]
    assert all(item["blocks_acceptance"] is True for item in receipt["findings"])
    assert receipt["authority"] == review.REJECT_AUTHORITY
    assert all(value is False for value in receipt["authority"].values())


def test_receipt_rejects_self_resigned_decision_authority_findings_and_subject_drift(receipt: dict[str, object]) -> None:
    variants = []
    decision = copy.deepcopy(receipt); decision["decision"] = "ACCEPT"; variants.append(decision)
    authority = copy.deepcopy(receipt); authority["authority"]["docker_authorized"] = True; variants.append(authority)
    findings = copy.deepcopy(receipt); findings["findings"] = []; variants.append(findings)
    subject = copy.deepcopy(receipt); subject["subject_replay"]["toolchain"]["buildkit_version"] = "v0.99.0"; variants.append(subject)
    for variant in variants:
        _resign_review(variant)
        with pytest.raises(review.IndependentReviewError):
            review.validate_review(variant, replay_subject=True)


def test_receipt_successor_requires_corrected_additive_package_and_fresh_review(receipt: dict[str, object]) -> None:
    successor = receipt["successor_criteria"]
    assert successor["allowed_now"] is False
    assert successor["close_every_handoff_nested_object_with_schema_or_exact_validation"] is True
    assert successor["bind_bytes_hash_mode_and_link_count_to_one_held_inode"] is True
    assert successor["add_self_resign_overclaim_and_split_inode_toctou_regressions"] is True
    assert successor["fresh_execution_authority_still_required"] is True


def test_receipt_pins_all_four_frozen_reviewer_artifacts(receipt: dict[str, object]) -> None:
    pins = receipt["review_source_pins"]
    assert {pin["path"] for pin in pins} == {review.REVIEW_SOURCE_REL, review.REVIEW_SCHEMA_REL, review.REVIEW_TEST_REL, review.REVIEW_DOC_REL}
    assert pins == review._review_source_pins()


def test_receipt_records_two_84_of_84_author_and_two_reviewer_runs(receipt: dict[str, object]) -> None:
    replay = receipt["test_replay"]
    assert [(row["passed"], row["failed"]) for row in replay["author_focused_runs"]] == [(84, 0), (84, 0)]
    assert len(replay["independent_reviewer_runs"]) == 2
    assert all(row["passed"] == row["collected"] and row["failed"] == 0 and row["gpu_visibility_disabled"] is True for row in replay["independent_reviewer_runs"])


def test_main_verify_review_replays_frozen_receipt() -> None:
    assert review.main(["verify-review", "--path", str(review.DEFAULT_REVIEW), "--replay-subject"]) == 0
