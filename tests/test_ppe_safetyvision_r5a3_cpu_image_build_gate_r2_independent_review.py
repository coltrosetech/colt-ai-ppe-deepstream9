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

from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r2 as author
from validation import ppe_safetyvision_r5a3_cpu_image_build_gate_r2_independent_review as review


STAMP = "2026-07-20T15:15:00Z"


@pytest.fixture(scope="session")
def audit() -> dict[str, object]:
    return review.audit_current_state()


@pytest.fixture(scope="session")
def handoff() -> dict[str, object]:
    with review.WorkspaceReader(review.ROOT) as reader:
        return reader.read_json(review.AUTHOR_HANDOFF_PIN)


@pytest.fixture(scope="session")
def built_review() -> dict[str, object]:
    return review.build_review(reviewed_at_utc=STAMP, independent_test_count=1)


@pytest.fixture(scope="session")
def receipt() -> dict[str, object]:
    return review.load_review()


def _resign_handoff(value: dict[str, object]) -> dict[str, object]:
    value["self_fingerprint"] = author.fingerprint(value)
    return value


def _resign_review(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def _future_replacement(key: str, value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if key == "argv_sha256":
        return "0" * 64
    return {"network": "host", "platform": "linux/arm64"}.get(key, "changed")


def test_strict_json_accepts_object() -> None:
    assert review.strict_json(b'{"x":1}', label="valid") == {"x": 1}


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(review.IndependentReviewError, match="duplicate"):
        review.strict_json(b'{"x":1,"x":2}', label="duplicate")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite(token: bytes) -> None:
    with pytest.raises(review.IndependentReviewError, match="non-finite"):
        review.strict_json(b'{"x":' + token + b"}", label="nonfinite")


@pytest.mark.parametrize("raw", [b"", b"[]", b"\xef\xbb\xbf{}", b'{"x":"\xff"}'])
def test_strict_json_rejects_bad_envelope(raw: bytes) -> None:
    with pytest.raises(review.IndependentReviewError):
        review.strict_json(raw, label="bad")


@pytest.mark.parametrize("value", ["", "/x", "../x", "a/../x", "a//b", "a/./b", "a\\b", "a\x00b"])
def test_safe_parts_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(review.IndependentReviewError, match="unsafe relative path"):
        review._parts(value)


def test_reviewer_reader_returns_one_inode_joint_pin(tmp_path: Path) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"exact")
    target.chmod(0o440)
    pin = {"path": "item", "bytes": 5, "sha256": hashlib.sha256(b"exact").hexdigest(), "mode": "0440"}
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.read_pin(pin) == b"exact"


def test_author_reader_rejects_r1_split_inode_content_mode_exploit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = b"g" * (1024 * 1024 + 1)
    evil = b"e" * len(good)
    target = tmp_path / "item"
    target.write_bytes(good)
    target.chmod(0o600)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(evil)
    replacement.chmod(0o440)
    pin = {"path": "item", "bytes": len(good), "sha256": hashlib.sha256(good).hexdigest(), "mode": "0440"}
    real_pread = author.os.pread
    replaced = False

    def raced(fd: int, length: int, offset: int) -> bytes:
        nonlocal replaced
        chunk = real_pread(fd, length, offset)
        if offset == 0 and not replaced:
            replacement.replace(target)
            replaced = True
        return chunk

    monkeypatch.setattr(author.os, "pread", raced)
    with author.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(author.GateR2Error, match="inode or final name changed"):
            workspace.read_pin(pin, limit=2 * 1024 * 1024)


def test_author_reader_rejects_link_count_not_one(tmp_path: Path) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"x")
    os.link(target, tmp_path / "hardlink")
    with author.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(author.GateR2Error, match="single-link"):
            workspace.read_regular("item")


def test_author_reader_rejects_mode_change_on_held_inode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "item"
    target.write_bytes(b"x" * (1024 * 1024 + 1))
    target.chmod(0o440)
    real_pread = author.os.pread
    changed = False

    def raced(fd: int, length: int, offset: int) -> bytes:
        nonlocal changed
        chunk = real_pread(fd, length, offset)
        if offset == 0 and not changed:
            os.chmod(target, 0o600)
            changed = True
        return chunk

    monkeypatch.setattr(author.os, "pread", raced)
    with author.AnchoredWorkspace(tmp_path) as workspace:
        with pytest.raises(author.GateR2Error, match="inode or final name changed"):
            workspace.read_regular("item", limit=2 * 1024 * 1024)


@pytest.mark.parametrize("name", sorted(review.AUTHOR_PINS))
def test_every_author_artifact_raw_pin_is_exact(name: str) -> None:
    with review.WorkspaceReader(review.ROOT) as reader:
        reader.read_pin(review.AUTHOR_PINS[name])


def test_author_handoff_raw_pin_is_exact() -> None:
    with review.WorkspaceReader(review.ROOT) as reader:
        receipt = reader.read_json(review.AUTHOR_HANDOFF_PIN)
    assert receipt["self_fingerprint"] == review.HANDOFF_FINGERPRINT


@pytest.mark.parametrize(
    "relative,field,expected",
    [
        (review.PLAN_REL, "self_fingerprint", review.PLAN_FINGERPRINT),
        (review.STATIC_CONTROLLER_REL, "self_fingerprint", review.CONTROLLER_FINGERPRINT),
        (review.AUTHOR_HANDOFF_REL, "self_fingerprint", review.HANDOFF_FINGERPRINT),
        (review.R1_REVIEW_REL, "review_fingerprint_sha256", review.R1_REVIEW_FINGERPRINT),
        (review.FOUNDATION_REL, "review_fingerprint_sha256", review.FOUNDATION_FINGERPRINT),
    ],
)
def test_all_subject_fingerprints_replay(relative: str, field: str, expected: str) -> None:
    value = json.loads((review.ROOT / relative).read_text(encoding="utf-8"))
    assert value[field] == expected
    assert review.fingerprint(value, field) == expected


def test_plan_schema_is_draft_2020_exact_object_const() -> None:
    plan = json.loads((review.ROOT / review.PLAN_REL).read_text(encoding="utf-8"))
    schema = json.loads((review.ROOT / review.PLAN_SCHEMA_REL).read_text(encoding="utf-8"))
    result = review.validate_plan(plan, schema)
    assert result["schema_exact_object_const"] is True


def test_handoff_schema_closes_every_structural_object() -> None:
    schema = json.loads((review.ROOT / review.HANDOFF_SCHEMA_REL).read_text(encoding="utf-8"))
    result = review.validate_handoff_schema(schema)
    assert result["closed_structural_objects"] >= 12
    assert result["test_replay_const_exact"] is True
    assert result["future_exact_build_const_exact"] is True


def test_handoff_test_records_are_exact_and_closed(handoff: dict[str, object]) -> None:
    assert handoff["test_replay"] == review.AUTHOR_TEST_REPLAY
    assert [row["passed"] for row in handoff["test_replay"]["runs"]] == [84, 84]
    assert all(row["gpu_visibility_disabled"] is True for row in handoff["test_replay"]["runs"])


def test_handoff_future_build_has_exact_complete_field_set(handoff: dict[str, object]) -> None:
    assert handoff["future_exact_build"] == review.FUTURE_EXACT_BUILD
    assert set(handoff["future_exact_build"]) == set(review.FUTURE_EXACT_BUILD)


def test_actual_author_handoff_validates_independently_and_by_author(handoff: dict[str, object]) -> None:
    schema = json.loads((review.ROOT / review.HANDOFF_SCHEMA_REL).read_text(encoding="utf-8"))
    assert review.validate_handoff(handoff, schema)["authority_all_false"] is True
    assert author.validate_handoff(handoff)["valid"] is True


@pytest.mark.parametrize("key", sorted(review.FUTURE_EXACT_BUILD))
def test_author_rejects_self_resigned_drift_in_every_future_build_field(handoff: dict[str, object], key: str) -> None:
    changed = copy.deepcopy(handoff)
    changed["future_exact_build"][key] = _future_replacement(key, changed["future_exact_build"][key])
    _resign_handoff(changed)
    with pytest.raises(author.GateR2Error):
        author.validate_handoff(changed)


@pytest.mark.parametrize("key,replacement", [("run", 9), ("command", "true"), ("collected", 1), ("passed", 1), ("failed", 1), ("gpu_visibility_disabled", False)])
def test_author_rejects_self_resigned_drift_in_every_test_record_field(handoff: dict[str, object], key: str, replacement: object) -> None:
    changed = copy.deepcopy(handoff)
    changed["test_replay"]["runs"][0][key] = replacement
    _resign_handoff(changed)
    with pytest.raises(author.GateR2Error):
        author.validate_handoff(changed)


@pytest.mark.parametrize("key", ["self_reported_non_authoritative", "independent_replay_required"])
def test_author_rejects_self_resigned_test_replay_boundary_drift(handoff: dict[str, object], key: str) -> None:
    changed = copy.deepcopy(handoff)
    changed["test_replay"][key] = not changed["test_replay"][key]
    _resign_handoff(changed)
    with pytest.raises(author.GateR2Error):
        author.validate_handoff(changed)


@pytest.mark.parametrize("location", ["top", "future", "test_replay", "test_row", "authority", "artifact_pin"])
def test_author_rejects_self_resigned_extra_keys_at_every_relevant_level(handoff: dict[str, object], location: str) -> None:
    changed = copy.deepcopy(handoff)
    target: dict[str, object]
    if location == "top":
        target = changed
    elif location == "future":
        target = changed["future_exact_build"]
    elif location == "test_replay":
        target = changed["test_replay"]
    elif location == "test_row":
        target = changed["test_replay"]["runs"][0]
    elif location == "authority":
        target = changed["authority"]
    else:
        target = changed["author_artifacts"]["tests"]
    target["unexpected"] = True
    _resign_handoff(changed)
    with pytest.raises(author.GateR2Error):
        author.validate_handoff(changed)


@pytest.mark.parametrize("key", sorted(review.AUTHOR_AUTHORITY))
def test_author_rejects_each_self_resigned_authority_widening(handoff: dict[str, object], key: str) -> None:
    changed = copy.deepcopy(handoff)
    changed["authority"][key] = True
    _resign_handoff(changed)
    with pytest.raises(author.GateR2Error):
        author.validate_handoff(changed)


def test_author_source_mechanically_binds_joint_inode_and_exact_handoff() -> None:
    source = (review.ROOT / review.AUTHOR_SOURCE_REL).read_bytes()
    result = review.inspect_author_source(source)
    assert result["joint_open_descriptor_content_size_hash_mode_nlink"] is True
    assert result["final_name_same_identity_replay"] is True
    assert result["exact_test_replay_validation_present"] is True
    assert result["exact_future_build_validation_present"] is True


def test_author_source_has_no_build_process_writer_model_gpu_runtime_surface() -> None:
    result = review.inspect_author_source((review.ROOT / review.AUTHOR_SOURCE_REL).read_bytes())
    assert result["forbidden_imports"] == []
    assert result["forbidden_calls"] == []
    assert result["subprocess_calls"] == 0
    assert result["build_adapter_present"] is False
    assert result["execute_branch_calls"] == ["require_execution_gate"]


def test_author_execute_gate_is_unconditional_fail_closed() -> None:
    with pytest.raises(author.GateR2Error, match="nonterminal"):
        author.require_execution_gate()


@pytest.mark.parametrize("pin", [review.DOCKER_PIN, review.BUILDX_PIN])
def test_tool_binaries_are_direct_joint_exact_pins(pin: dict[str, object]) -> None:
    assert review.read_absolute_pin(pin) == pin


def test_r1_reject_and_a32_foundation_lineage_are_exact(audit: dict[str, object]) -> None:
    assert audit["r1_rejection"]["decision"] == "REJECT"
    assert audit["r1_rejection"]["p1_findings"] == 2
    assert audit["foundation"]["decision"] == "ACCEPT"
    assert audit["foundation"]["accepted_context_only"] is True


def test_context_lineage_remains_static_and_requires_fresh_pre_stream_replay(audit: dict[str, object]) -> None:
    lineage = audit["context_lineage"]
    assert lineage["full_tree"] == review.EXPECTED_TREES["full"]
    assert lineage["symlink_closure"] == review.EXPECTED_SYMLINK_CLOSURE
    assert lineage["full_tree_replayed_now"] is False
    assert lineage["fresh_pre_stream_full_tree_replay_required"] is True


def test_future_outputs_remain_absent() -> None:
    with review.WorkspaceReader(review.ROOT) as reader:
        for relative in review.OUTPUT_ABSENCE_PATHS:
            reader.require_absent(relative)


def test_audit_projection_is_deterministic(audit: dict[str, object]) -> None:
    unsigned = dict(audit)
    digest = unsigned.pop("audit_projection_sha256")
    assert hashlib.sha256(review.canonical_bytes(unsigned)).hexdigest() == digest
    assert review.audit_current_state() == audit


def test_accept_authority_has_only_static_definition_true() -> None:
    assert review.ACCEPT_AUTHORITY["exact_build_definition_accepted"] is True
    assert sum(value is True for value in review.ACCEPT_AUTHORITY.values()) == 1
    for key in ("cpu_image_build_execution_authorized", "docker_authorized", "model_or_onnx_load_authorized", "gpu_authorized", "runtime_validated", "quality_validated", "production_ready"):
        assert review.ACCEPT_AUTHORITY[key] is False


def test_review_schema_is_draft_2020_and_top_level_closed() -> None:
    schema = json.loads((review.ROOT / review.REVIEW_SCHEMA_REL).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["authority"]["const"] == review.ACCEPT_AUTHORITY


def test_review_builds_and_validates_with_subject_replay(built_review: dict[str, object]) -> None:
    result = review.validate_review(built_review, replay_subject=True)
    assert result["decision"] == "ACCEPT"
    assert result["static_build_definition_readiness_accepted"] is True
    assert result["build_execution_authorized"] is False
    assert result["model_or_runtime_authorized"] is False


@pytest.mark.parametrize("key", ["cpu_image_build_execution_authorized", "docker_authorized", "model_or_onnx_load_authorized", "gpu_authorized", "runtime_validated", "quality_validated", "production_ready"])
def test_review_rejects_self_resigned_authority_widening(built_review: dict[str, object], key: str) -> None:
    changed = copy.deepcopy(built_review)
    changed["authority"][key] = True
    _resign_review(changed)
    with pytest.raises(review.IndependentReviewError):
        review.validate_review(changed, replay_subject=False)


def test_review_rejects_self_resigned_subject_projection_drift(built_review: dict[str, object]) -> None:
    changed = copy.deepcopy(built_review)
    changed["subject_replay"]["source_audit"]["final_name_same_identity_replay"] = False
    _resign_review(changed)
    with pytest.raises(review.IndependentReviewError, match="subject projection"):
        review.validate_review(changed, replay_subject=True)


def test_review_rejects_self_resigned_source_pin_drift(built_review: dict[str, object]) -> None:
    changed = copy.deepcopy(built_review)
    changed["review_source_pins"][0]["sha256"] = "0" * 64
    _resign_review(changed)
    with pytest.raises(review.IndependentReviewError, match="source pins"):
        review.validate_review(changed, replay_subject=False)


def test_review_rejects_top_level_extra_even_if_resigned(built_review: dict[str, object]) -> None:
    changed = copy.deepcopy(built_review)
    changed["unexpected"] = True
    _resign_review(changed)
    with pytest.raises(review.IndependentReviewError, match="schema rejected"):
        review.validate_review(changed, replay_subject=False)


def test_reviewer_source_has_no_subprocess_or_mutating_surface() -> None:
    source = inspect.getsource(review)
    tree = ast.parse(source)
    imports = {alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not imports.intersection({"subprocess", "socket", "requests", "torch", "onnx", "onnxruntime", "tensorrt", "ultralytics", "cv2", "gi"})
    attributes = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not attributes.intersection({"run", "Popen", "system", "popen", "mkdir", "write_text", "write_bytes", "rename", "replace", "unlink", "chmod"})
    os_flags = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os"}
    assert not os_flags.intersection({"O_CREAT", "O_WRONLY", "O_RDWR", "O_TRUNC", "O_APPEND"})


def test_main_audit_succeeds_without_execution() -> None:
    assert review.main(["audit"]) == 0


def test_frozen_receipt_validates_with_exact_subject_replay(receipt: dict[str, object]) -> None:
    result = review.validate_review(receipt, replay_subject=True)
    assert result["valid"] is True
    assert result["decision"] == "ACCEPT"
    assert result["static_build_definition_readiness_accepted"] is True
    assert result["build_execution_authorized"] is False
    assert result["model_or_runtime_authorized"] is False


def test_frozen_receipt_grants_only_static_readiness(receipt: dict[str, object]) -> None:
    assert receipt["accepted_claim"] == review.ACCEPTED_CLAIM
    assert receipt["authority"] == review.ACCEPT_AUTHORITY
    assert sum(value is True for value in receipt["authority"].values()) == 1
    assert receipt["next_gate"] == review.NEXT_GATE


def test_frozen_receipt_pins_all_reviewer_sources(receipt: dict[str, object]) -> None:
    assert receipt["review_source_pins"] == review._review_source_pins()
    assert {item["path"] for item in receipt["review_source_pins"]} == {review.REVIEW_SOURCE_REL, review.REVIEW_SCHEMA_REL, review.REVIEW_TEST_REL, review.REVIEW_DOC_REL}


def test_frozen_receipt_records_two_author_and_reviewer_runs(receipt: dict[str, object]) -> None:
    replay = receipt["test_replay"]
    assert [(row["collected"], row["passed"], row["failed"]) for row in replay["author_focused_runs"]] == [(84, 84, 0), (84, 84, 0)]
    assert len(replay["independent_reviewer_runs"]) == 2
    assert all(row["passed"] == row["collected"] and row["failed"] == 0 and row["gpu_visibility_disabled"] is True for row in replay["independent_reviewer_runs"])


def test_main_verify_review_replays_frozen_receipt() -> None:
    assert review.main(["verify-review", "--path", str(review.DEFAULT_REVIEW)]) == 0
