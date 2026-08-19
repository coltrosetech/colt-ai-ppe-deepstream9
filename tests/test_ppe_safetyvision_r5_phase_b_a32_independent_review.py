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

from validation import ppe_safetyvision_r5_phase_b_a32_independent_review as review


@pytest.fixture(scope="session")
def light_audit() -> dict[str, object]:
    return review.audit_current_state(replay_trees=False)


@pytest.fixture(scope="session")
def full_audit() -> dict[str, object]:
    return review.audit_current_state(replay_trees=True)


@pytest.fixture(scope="session")
def receipt() -> dict[str, object]:
    return review.load_review()


def _resign(value: dict[str, object]) -> dict[str, object]:
    value["review_fingerprint_sha256"] = review.review_fingerprint(value)
    return value


def _reader(root: Path) -> review.WorkspaceReader:
    return review.WorkspaceReader(root)


def test_acceptance_authority_is_narrow_and_build_remains_closed() -> None:
    assert review.ACCEPT_AUTHORITY["phase_b_context_accepted"] is True
    assert review.ACCEPT_AUTHORITY["separate_exact_cpu_image_build_gate_entry_allowed"] is True
    assert all(
        review.ACCEPT_AUTHORITY[field] is False
        for field in (
            "cpu_image_build_execution_authorized",
            "context_copy_authorized",
            "checkpoint_deserialization_or_export_authorized",
            "model_or_onnx_load_authorized",
            "model_acceptance",
            "cpu_parity",
            "docker_authorized",
            "gpu_authorized",
            "tensorrt_or_deepstream_authorized",
            "quality_validated",
            "production_ready",
        )
    )


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(review.PhaseBReviewError, match="duplicate JSON key"):
        review.strict_json(b'{"a":1,"a":2}', "fixture")


def test_strict_json_rejects_nonfinite_token() -> None:
    with pytest.raises(review.PhaseBReviewError, match="non-finite"):
        review.strict_json(b'{"a":NaN}', "fixture")


def test_strict_json_rejects_nonobject_bom_empty_and_bad_utf8() -> None:
    for raw in (b"[]", b"", b"\xef\xbb\xbf{}", b'{"x":"\xff"}'):
        with pytest.raises((review.PhaseBReviewError, UnicodeError)):
            review.strict_json(raw, "fixture")


def test_relative_paths_reject_escape_absolute_backslash_nul_and_double_slash() -> None:
    for value in ("../x", "a/../x", "/x", "a\\b", "a\x00b", "a//b", ".", ""):
        with pytest.raises(review.PhaseBReviewError, match="unsafe relative path"):
            review._parts(value)


def test_descriptor_reader_replays_exact_single_link_regular(tmp_path: Path) -> None:
    target = tmp_path / "exact"
    target.write_bytes(b"exact")
    target.chmod(0o440)
    assert _reader(tmp_path).pin_regular("exact") == {
        "bytes": 5,
        "mode": "0440",
        "path": "exact",
        "sha256": hashlib.sha256(b"exact").hexdigest(),
    }


def test_descriptor_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"x")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        _reader(tmp_path).pin_regular("alias/file")


def test_descriptor_reader_rejects_final_symlink(tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"x")
    (tmp_path / "alias").symlink_to("target")
    with pytest.raises(OSError):
        _reader(tmp_path).pin_regular("alias")


def test_descriptor_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.write_bytes(b"x")
    os.link(first, tmp_path / "second")
    with pytest.raises(review.PhaseBReviewError, match="single-link"):
        _reader(tmp_path).pin_regular("first")


def test_directory_snapshot_rejects_wrong_mode_or_entries(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir(mode=0o700)
    with pytest.raises(review.PhaseBReviewError, match="mode differs"):
        _reader(tmp_path).directory_snapshot("run", "0550")
    directory.chmod(0o550)
    with pytest.raises(review.PhaseBReviewError, match="entries differ"):
        _reader(tmp_path).directory_snapshot("run", "0550", ["receipt.json"])


def test_review_schema_is_strict_valid_draft_2020_12() -> None:
    schema = json.loads(review.REVIEW_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["subject"]["additionalProperties"] is False
    assert schema["properties"]["authority"]["const"] == review.ACCEPT_AUTHORITY


def test_reviewer_never_imports_or_invokes_author_modules() -> None:
    source = inspect.getsource(review)
    assert "from validation import ppe_safetyvision_r5_phase_a" not in source
    assert "import ppe_safetyvision_r5_phase_a" not in source
    assert "-m validation.ppe_safetyvision" not in source
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported.intersection({"subprocess", "torch", "onnx", "onnxruntime", "docker", "ultralytics"})


def test_reviewer_has_no_writer_network_container_or_gpu_surface() -> None:
    source = inspect.getsource(review)
    tree = ast.parse(source)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (attributes | names).intersection(
        {"Popen", "run", "system", "mkdir", "write_text", "write_bytes", "rename", "unlink", "chmod"}
    )
    assert "os.replace(" not in source and "Path.replace(" not in source
    assert "O_CREAT" not in source and "O_TRUNC" not in source and "O_WRONLY" not in source


def test_candidate_pin_schema_fingerprint_scope_and_authority_are_exact(
    light_audit: dict[str, object],
) -> None:
    assert light_audit["candidate_pin"]["sha256"] == review.EXPECTED_CANDIDATE_PIN[2]
    assert light_audit["candidate_fingerprint_sha256"] == review.EXPECTED_CANDIDATE_FINGERPRINT
    assert all(light_audit["candidate_checks"].values())


def test_terminal_r2_rejection_and_r1_lineage_are_preserved(
    light_audit: dict[str, object],
) -> None:
    terminal = light_audit["terminal_r2"]
    assert terminal["decision"] == "REJECT"
    assert terminal["severity_counts"] == {"P0": 0, "P1": 2, "P2": 1}
    assert terminal["r1_failure_visible"] is True
    assert terminal["authority_all_false"] is True
    assert len(terminal["r1_controls"]) == 3


def test_historical_a3_17_of_18_and_a31_6_of_7_remain_visible(
    light_audit: dict[str, object],
) -> None:
    lineage = light_audit["predecessor_test_lineage"]
    assert lineage["a3"] == {
        "collected": 18,
        "passed": 17,
        "failed": 1,
        "failure": "test_a3_static_gate_replays_every_pin_without_heavy_copy_or_execution",
    }
    assert lineage["a31"] == {
        "collected": 7,
        "passed": 6,
        "failed": 1,
        "failure": "test_live_recovery_audit_replays_frozen_a2_a3_and_sources",
    }


def test_source_closure_is_exactly_15_plus_24_equals_39(
    light_audit: dict[str, object],
) -> None:
    sources = light_audit["source_replay"]
    assert len(sources["source_pins"]) == 15
    assert len(sources["nested_source_pins"]) == 24
    assert sources["source_count"] == 39
    assert sources["source_modes_nonwritable"] is True


def test_six_receipts_are_exact_strict_and_nonexecuting(
    light_audit: dict[str, object],
) -> None:
    replay = light_audit["receipt_replay"]
    assert len(replay["pins"]) == 6
    assert replay["strict"] is True
    assert replay["all_execution_authorities_false"] is True
    assert replay["a31_self_fingerprint"] == "b9ed9f9c824b8f68bd0bc35ee7ca79f3d522e3f0112497b30c1bd57a713057cc"


def test_postpublication_objects_are_required_present_and_collision_is_preserved(
    light_audit: dict[str, object],
) -> None:
    present = light_audit["required_present"]
    assert all(row["required_present"] for row in present["required_present_objects"].values())
    assert present["required_present_objects"]["a31_run"]["entries"] == ["receipt.json"]
    assert present["historical_collision"] == {
        "path": review.A3_COLLISION_REL,
        "mode": "0700",
        "entries": [],
        "empty": True,
        "reused": False,
    }


def test_full_a2_a3_payload_and_rootfs_tree_hashes_are_exact(
    full_audit: dict[str, object],
) -> None:
    trees = full_audit["trees"]
    assert trees["a2_full_tree"] == review.A2_TREE
    assert trees["a3_full_tree"] == review.A3_TREE
    assert trees["a3_payload_tree"] == review.A3_PAYLOAD_TREE
    assert trees["a3_rootfs_tree"] == review.A3_ROOTFS_TREE


def test_full_mode_inventories_have_zero_writable_and_special(
    full_audit: dict[str, object],
) -> None:
    trees = full_audit["trees"]
    assert trees["a2_mode_inventory"] == review.A2_MODES
    assert trees["a3_mode_inventory"] == review.A3_MODES
    for name in ("a2_mode_inventory", "a3_mode_inventory"):
        assert trees[name]["writable_directories_or_regular_files"] == 0
        assert trees[name]["special_entries"] == 0


def test_full_symlink_closure_has_47_and_zero_failures(
    full_audit: dict[str, object],
) -> None:
    assert full_audit["trees"]["a3_symlink_closure"] == review.SYMLINK_CLOSURE


def test_independent_light_replay_is_idempotent() -> None:
    first = review.subject_projection(review.audit_current_state(replay_trees=False))
    second = review.subject_projection(review.audit_current_state(replay_trees=False))
    assert first == second


def test_execution_surface_is_read_only_without_model_build_docker_or_gpu(
    light_audit: dict[str, object],
) -> None:
    surface = light_audit["execution_surface"]
    assert all(surface["checks"].values())
    assert all(
        surface[field] is False
        for field in (
            "checkpoint_or_model_loaded",
            "onnx_loaded",
            "image_or_docker_called",
            "gpu_called",
            "tensorrt_or_deepstream_called",
        )
    )


def test_real_receipt_verifies_as_narrow_accept(receipt: dict[str, object]) -> None:
    result = review.verify_review(receipt)
    assert result["verification_status"] == "pass"
    assert result["decision"] == "ACCEPT"
    assert result["phase_b_context_accepted"] is True
    assert result["separate_exact_cpu_image_build_gate_entry_allowed"] is True
    assert result["cpu_image_build_execution_authorized"] is False
    assert result["gpu_authorized"] is False
    assert result["production_ready"] is False


def test_real_receipt_schema_fingerprint_and_zero_findings(
    receipt: dict[str, object],
) -> None:
    assert review._review_schema_errors(receipt) == []
    assert receipt["review_fingerprint_sha256"] == review.review_fingerprint(receipt)
    assert receipt["findings"] == []
    assert receipt["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}


@pytest.mark.parametrize(
    "field",
    [
        "cpu_image_build_execution_authorized",
        "context_copy_authorized",
        "checkpoint_deserialization_or_export_authorized",
        "model_or_onnx_load_authorized",
        "docker_authorized",
        "gpu_authorized",
        "tensorrt_or_deepstream_authorized",
        "production_ready",
    ],
)
def test_authority_escalation_is_rejected_after_refingerprint(
    receipt: dict[str, object], field: str
) -> None:
    value = copy.deepcopy(receipt)
    value["authority"][field] = True
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["cpu_image_build_execution_authorized"] is False
    assert result["gpu_authorized"] is False


@pytest.mark.parametrize(
    "section",
    ["author_control_pins", "source_pins", "nested_source_pins", "receipt_pins"],
)
def test_recorded_pin_drift_is_rejected_after_refingerprint(
    receipt: dict[str, object], section: str
) -> None:
    value = copy.deepcopy(receipt)
    value["subject_replay"][section][0]["sha256"] = "0" * 64
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("subject projection" in failure for failure in result["failures"])


def test_candidate_pin_and_fingerprint_drift_are_rejected(
    receipt: dict[str, object],
) -> None:
    for field, replacement in (("sha256", "f" * 64), ("bytes", 1)):
        value = copy.deepcopy(receipt)
        value["subject_replay"]["candidate_pin"][field] = replacement
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"
    value = copy.deepcopy(receipt)
    value["subject_replay"]["candidate_fingerprint_sha256"] = "e" * 64
    assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_tree_mode_and_symlink_tamper_are_rejected(
    receipt: dict[str, object],
) -> None:
    mutations = []
    tree = copy.deepcopy(receipt)
    tree["subject_replay"]["trees"]["a3_full_tree"]["tree_sha256"] = "0" * 64
    mutations.append(tree)
    mode = copy.deepcopy(receipt)
    mode["subject_replay"]["trees"]["a3_mode_inventory"]["writable_directories_or_regular_files"] = 1
    mutations.append(mode)
    symlink = copy.deepcopy(receipt)
    symlink["subject_replay"]["trees"]["a3_symlink_closure"]["dangling"] = 1
    mutations.append(symlink)
    for value in mutations:
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_malformed_extra_and_duplicate_pin_arrays_are_rejected(
    receipt: dict[str, object],
) -> None:
    extra = copy.deepcopy(receipt)
    extra["subject_replay"]["source_pins"].append(copy.deepcopy(extra["subject_replay"]["source_pins"][0]))
    malformed = copy.deepcopy(receipt)
    malformed["subject_replay"]["receipt_pins"][0]["unexpected"] = False
    for value in (extra, malformed):
        result = review.verify_review(_resign(value))
        assert result["verification_status"] == "blocked"
        assert result["schema_error_count"] >= 1


def test_control_decision_scope_and_next_gate_tamper_are_rejected(
    receipt: dict[str, object],
) -> None:
    mutations = []
    control = copy.deepcopy(receipt)
    control["controls"]["six_receipts_strict"] = False
    mutations.append(control)
    decision = copy.deepcopy(receipt)
    decision["decision"] = "REJECT"
    mutations.append(decision)
    scope = copy.deepcopy(receipt)
    scope["scope"]["gpu_called"] = True
    mutations.append(scope)
    gate = copy.deepcopy(receipt)
    gate["next_gate"]["image_build_executes_here"] = True
    mutations.append(gate)
    for value in mutations:
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_unknown_root_subject_and_authority_fields_are_rejected(
    receipt: dict[str, object],
) -> None:
    value = copy.deepcopy(receipt)
    value["unexpected"] = False
    value["subject_replay"]["unexpected"] = False
    value["authority"]["unexpected"] = False
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert result["schema_error_count"] >= 3


def test_unsigned_nested_mutation_is_rejected_by_fingerprint(
    receipt: dict[str, object],
) -> None:
    value = copy.deepcopy(receipt)
    value["idempotency_replay"]["same_projection"] = False
    result = review.verify_review(value)
    assert result["verification_status"] == "blocked"
    assert any("fingerprint" in failure for failure in result["failures"])


def test_findings_severity_and_test_replay_tamper_are_rejected(
    receipt: dict[str, object],
) -> None:
    finding = copy.deepcopy(receipt)
    finding["findings"] = [{"id": "fake"}]
    severity = copy.deepcopy(receipt)
    severity["severity_counts"]["P2"] = 1
    replay = copy.deepcopy(receipt)
    replay["test_replay"]["independent_passed"] -= 1
    for value in (finding, severity, replay):
        assert review.verify_review(_resign(value))["verification_status"] == "blocked"


def test_review_source_pin_drift_is_rejected_after_refingerprint(
    receipt: dict[str, object],
) -> None:
    value = copy.deepcopy(receipt)
    value["review_source_pins"][0]["sha256"] = "f" * 64
    result = review.verify_review(_resign(value))
    assert result["verification_status"] == "blocked"
    assert any("review source" in failure for failure in result["failures"])


def test_fresh_full_subject_replay_matches_receipt(receipt: dict[str, object]) -> None:
    result = review.verify_review(receipt, replay_subject=True)
    assert result["verification_status"] == "pass"
    assert result["subject_replay"] == {
        "performed": True,
        "status": "pass",
        "same_projection": True,
        "source_count": 39,
        "receipt_count": 6,
        "symlink_count": 47,
    }


def test_mocked_fresh_subject_drift_blocks_acceptance(
    receipt: dict[str, object],
    full_audit: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = copy.deepcopy(full_audit)
    changed["candidate_fingerprint_sha256"] = "0" * 64
    monkeypatch.setattr(review, "audit_current_state", lambda **_: copy.deepcopy(changed))
    result = review.verify_review(receipt, replay_subject=True)
    assert result["verification_status"] == "blocked"
    assert result["subject_replay"]["status"] == "blocked"


def test_tree_attestor_counts_writable_entries_and_rejects_special(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir(mode=0o700)
    (root / "file").write_bytes(b"x")
    observed = review.attest_tree(_reader(tmp_path), "tree")
    assert observed["mode_inventory"]["writable_directories_or_regular_files"] == 2
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(review.PhaseBReviewError, match="special tree entry"):
        review.attest_tree(_reader(tmp_path), "tree")


def test_virtual_symlink_resolver_accepts_root_bounded_terminal(
    tmp_path: Path,
) -> None:
    (tmp_path / "target").write_bytes(b"x")
    (tmp_path / "link").symlink_to("target")
    fd = os.open(tmp_path, review.DIR_FLAGS)
    try:
        row = review.resolve_virtual_symlink(fd, "link")
    finally:
        os.close(fd)
    assert row["terminal"] == "/target"
    assert row["terminal_kind"] == "file"


def test_virtual_symlink_resolver_rejects_dangling(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to("missing")
    fd = os.open(tmp_path, review.DIR_FLAGS)
    try:
        with pytest.raises(review.PhaseBReviewError, match="dangling"):
            review.resolve_virtual_symlink(fd, "link")
    finally:
        os.close(fd)


def test_virtual_symlink_resolver_rejects_cycle(tmp_path: Path) -> None:
    (tmp_path / "a").symlink_to("b")
    (tmp_path / "b").symlink_to("a")
    fd = os.open(tmp_path, review.DIR_FLAGS)
    try:
        with pytest.raises(review.PhaseBReviewError, match="cycle"):
            review.resolve_virtual_symlink(fd, "a")
    finally:
        os.close(fd)


def test_virtual_symlink_resolver_rejects_root_escape(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to("../outside")
    fd = os.open(tmp_path, review.DIR_FLAGS)
    try:
        with pytest.raises(review.PhaseBReviewError, match="escapes root"):
            review.resolve_virtual_symlink(fd, "link")
    finally:
        os.close(fd)


def test_receipt_declares_exact_author_and_independent_counts(
    receipt: dict[str, object],
) -> None:
    replay = receipt["test_replay"]
    assert replay["author_runs"] == [
        {"run": 1, "collected": 25, "passed": 25, "failed": 0},
        {"run": 2, "collected": 25, "passed": 25, "failed": 0},
    ]
    assert (
        replay["independent_collected"],
        replay["independent_passed"],
        replay["independent_failed"],
    ) == (review.INDEPENDENT_TEST_COUNT, review.INDEPENDENT_TEST_COUNT, 0)
