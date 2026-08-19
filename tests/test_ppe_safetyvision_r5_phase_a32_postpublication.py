from __future__ import annotations

import ast
import copy
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_safetyvision_r5_phase_a32_postpublication as a32


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def full_audit() -> dict:
    return a32.audit_current_state(replay_trees=True)


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(a32.A32Error, match="duplicate JSON key"):
        a32.strict_json(b'{"x":1,"x":2}', "fixture")


def test_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(a32.A32Error, match="non-finite"):
        a32.strict_json(b'{"x":NaN}', "fixture")


def test_strict_json_rejects_non_object_root() -> None:
    with pytest.raises(a32.A32Error, match="root differs"):
        a32.strict_json(b"[]", "fixture")


def test_relative_paths_reject_escape_absolute_backslash_and_nul() -> None:
    for value in ("../escape", "/absolute", "a\\b", "a\x00b"):
        with pytest.raises(a32.A32Error, match="unsafe relative path"):
            a32._parts(value)


def test_terminal_r2_controls_are_exactly_bound() -> None:
    reader = a32.WorkspaceReader()
    for expected in a32.R2_CONTROL_PINS:
        assert reader.pin_regular(expected["path"], expected=expected) == expected


def test_terminal_r2_receipt_has_exact_external_pin() -> None:
    reader = a32.WorkspaceReader()
    _, pin = reader.read_regular(a32.R2_RECEIPT_REL, expected=a32.R2_RECEIPT_PIN)
    assert pin == a32.R2_RECEIPT_PIN


def test_terminal_r2_receipt_replays_strict_schema_and_fingerprint() -> None:
    receipt, pin = a32._validate_terminal_r2(a32.WorkspaceReader())
    assert pin == a32.R2_RECEIPT_PIN
    assert set(receipt) == a32.R2_RECEIPT_KEYS
    assert receipt["review_fingerprint_sha256"] == a32.fingerprint(
        receipt, "review_fingerprint_sha256"
    )


def test_terminal_r2_remains_reject_with_no_authority() -> None:
    receipt, _ = a32._validate_terminal_r2(a32.WorkspaceReader())
    assert receipt["decision"] == "REJECT"
    assert receipt["severity_counts"] == {"P0": 0, "P1": 2, "P2": 1}
    assert all(value is False for value in receipt["authority"].values())


def test_postpublication_objects_are_required_present_not_absent() -> None:
    present = a32._postpublication_state(a32.WorkspaceReader())
    assert present["a3_context"] == {
        "mode": "0550",
        "path": a32.A3_CONTEXT_REL,
        "required_present": True,
    }
    assert present["a31_run"]["entries"] == ["receipt.json"]
    assert present["a31_run"]["mode"] == "0550"
    assert present["a31_receipt"]["required_present"] is True


def test_light_audit_uses_postpublication_semantics() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert report["status"] == "a32_postpublication_replay_complete_candidate_only"
    assert report["semantics"] == a32.EXPECTED_SEMANTICS
    assert report["semantics"]["published_run_directories_required_absent"] is False
    assert report["trees"] == {"replayed": False}


def test_a32_replay_never_executes_a_publication_probe() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert report["scope"]["cpu_and_disk_read_only"] is True
    assert report["scope"]["a2_a3_a31_or_phase_b_r2_mutated"] is False


def test_predecessor_failures_are_honest_lineage_not_hidden() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert report["predecessor_test_lineage"] == a32.EXPECTED_PREDECESSOR_TESTS
    assert report["predecessor_test_lineage"]["a3"] == {
        "collected": 18,
        "passed": 17,
        "failed": 1,
        "failure": "test_a3_static_gate_replays_every_pin_without_heavy_copy_or_execution",
    }
    assert report["predecessor_test_lineage"]["a31"]["passed"] == 6


def test_all_exact_source_pins_and_modes_replay() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert len(report["source_pins"]) == 15
    assert len(report["a3_nested_source_pins"]) == 24
    assert all(row["mode"] in {"0440", "0555"} for row in report["source_pins"])
    assert all(
        row["mode"] in {"0440", "0555"}
        for row in report["a3_nested_source_pins"]
    )


def test_six_strict_receipt_pins_replay() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert report["receipts_strict"] is True
    assert len(report["receipt_pins"]) == 6
    assert all(row["mode"] == "0440" for row in report["receipt_pins"])
    assert any(
        row["sha256"]
        == "8600fd0d3bec8741cc89ff1b2e9042cdb9c9f5a5c54603be79cd93343272428d"
        for row in report["receipt_pins"]
    )


def test_a31_receipt_self_fingerprint_and_noreplace_evidence_replay() -> None:
    report = a32.audit_current_state(replay_trees=False)
    assert report["a31_receipt_self_fingerprint"] == (
        "b9ed9f9c824b8f68bd0bc35ee7ca79f3d522e3f0112497b30c1bd57a713057cc"
    )
    assert report["a31_publication"]["method"] == "same_directory_renameat2_noreplace"
    assert report["a31_publication"]["historical_collision_directory_reused"] is False
    assert report["a31_publication"]["file_and_directory_fsync"] is True
    assert report["a31_publication"]["parent_directory_fsync"] is True


def test_full_a2_a3_payload_and_rootfs_trees_replay(full_audit: dict) -> None:
    trees = full_audit["trees"]
    assert trees["replayed"] is True
    assert trees["a2_full_tree"]["tree_sha256"] == (
        "cf2c1c9903eca67e92397e11d6484d7b66f939c57e6fcff991deaa8f028ef89b"
    )
    assert trees["a3_full_tree"]["tree_sha256"] == (
        "509977cd3945f62c45c6d717438e08477827103a53cadf87c3da1c0c89001293"
    )
    assert trees["a3_payload_tree"]["tree_sha256"] == (
        "9c7892870b7d9c7a1c292855b6b633fba7014c84c781577d07703f111c8e59e9"
    )
    assert trees["a3_rootfs_tree"]["tree_sha256"] == (
        "53d318710376fa8bf5fb66cf66fa4a000c8ced63c07248ab0394f52478444818"
    )


def test_full_tree_modes_have_no_writable_or_special_entries(full_audit: dict) -> None:
    for name in ("a2_mode_inventory", "a3_mode_inventory"):
        modes = full_audit["trees"][name]
        assert modes["writable_directories_or_regular_files"] == 0
        assert modes["special_entries"] == 0


def test_full_rootfs_symlink_closure_is_exact(full_audit: dict) -> None:
    assert full_audit["trees"]["a3_symlink_closure"] == {
        "verified": True,
        "symlinks": 47,
        "ledger_sha256": "5b7aaf115bdb38259361b860572ce6973ea69f75c0650dcdb7adf77130f2e268",
        "dangling": 0,
        "cycles": 0,
        "root_escapes": 0,
    }


def test_candidate_schema_is_strict_draft_2020_12() -> None:
    raw, _ = a32.WorkspaceReader().read_regular(a32.SCHEMA_REL)
    schema = a32.strict_json(raw, "schema")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["postpublication_replay"]["additionalProperties"] is False


def test_published_candidate_verifies_without_subject_mutation() -> None:
    assert a32.verify_candidate(a32.load_candidate()) == {
        "status": "verified_a32_non_authorizing_candidate",
        "decision": "CANDIDATE_NOT_INDEPENDENTLY_REVIEWED",
        "build_authorized": False,
        "replayed_subject": False,
    }


def test_candidate_self_fingerprint_is_canonical() -> None:
    candidate = a32.load_candidate()
    assert candidate["candidate_fingerprint_sha256"] == a32.fingerprint(
        candidate, "candidate_fingerprint_sha256"
    )


def test_candidate_has_no_acceptance_or_build_authority() -> None:
    candidate = a32.load_candidate()
    assert candidate["authority"] == a32.EXPECTED_AUTHORITY
    assert all(value is False for value in candidate["authority"].values())
    assert candidate["next_authority"] == {
        "independent_phase_b_review_required": True,
        "allowed_verdicts": [
            "ACCEPT_CONTEXT_FOR_SEPARATE_EXACT_CPU_IMAGE_BUILD_GATE",
            "REJECT",
        ],
        "acceptance_does_not_execute_or_self_authorize_a_build": True,
        "candidate_may_self_accept": False,
        "candidate_may_self_authorize_build": False,
    }


def test_candidate_records_two_identical_successful_suite_runs() -> None:
    candidate = a32.load_candidate()
    runs = candidate["a32_test_replay"]["runs"]
    assert [row["run"] for row in runs] == [1, 2]
    assert all(row["collected"] == 25 for row in runs)
    assert all(row["passed"] == 25 and row["failed"] == 0 for row in runs)
    assert candidate["a32_test_replay"][
        "same_published_subject_replayed_without_delete_or_republish"
    ] is True


def test_resigned_authority_overclaim_is_schema_rejected() -> None:
    candidate = copy.deepcopy(a32.load_candidate())
    candidate["authority"]["dedicated_cpu_image_build_authorized"] = True
    candidate["candidate_fingerprint_sha256"] = a32.fingerprint(
        candidate, "candidate_fingerprint_sha256"
    )
    with pytest.raises(jsonschema.ValidationError):
        a32.verify_candidate(candidate)


def test_verifier_source_has_no_writer_or_execution_surface() -> None:
    raw, _ = a32.WorkspaceReader().read_regular(a32.VERIFIER_REL)
    tree = ast.parse(raw.decode("utf-8", errors="strict"), filename=a32.VERIFIER_REL)
    imported_roots: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not (
        imported_roots
        & {"docker", "onnxruntime", "shutil", "subprocess", "tempfile", "torch", "ultralytics"}
    )
    assert not (
        calls
        & {
            "Popen",
            "chmod",
            "chown",
            "copy",
            "copytree",
            "export",
            "infer",
            "mkdir",
            "remove",
            "rename",
            "replace",
            "run",
            "system",
            "unlink",
            "write",
        }
    )
