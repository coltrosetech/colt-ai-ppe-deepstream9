from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_safetyvision_r5_phase_b_a31_independent_review_r2 as r2


ROOT = Path(__file__).resolve().parents[1]


def _resign(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("review_fingerprint_sha256", None)
    result["review_fingerprint_sha256"] = r2.fingerprint(result, "review_fingerprint_sha256")
    return result


def _cleanup(output: Path) -> None:
    if output.parent.exists():
        os.chmod(output.parent, 0o700, follow_symlinks=False)
    if output.exists():
        os.chmod(output, 0o600, follow_symlinks=False)
        output.unlink()
    if output.parent.exists():
        output.parent.rmdir()


def test_strict_json_rejects_duplicate_key() -> None:
    with pytest.raises(r2.R2Error, match="duplicate JSON key"):
        r2.strict_json(b'{"x":1,"x":2}', "fixture")


def test_strict_json_rejects_nonfinite() -> None:
    with pytest.raises(r2.R2Error, match="non-finite"):
        r2.strict_json(b'{"x":Infinity}', "fixture")


def test_strict_json_rejects_array_root() -> None:
    with pytest.raises(r2.R2Error, match="root differs"):
        r2.strict_json(b"[]", "fixture")


def test_relative_path_rejects_escape_absolute_backslash_and_nul() -> None:
    for value in ("../x", "/x", "a\\b", "a\x00b"):
        with pytest.raises(r2.R2Error, match="unsafe relative path"):
            r2._parts(value)


def test_descriptor_reader_replays_exact_regular(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"exact")
    path.chmod(0o440)
    assert r2.pin_regular("file", root=tmp_path) == {
        "path": "file",
        "bytes": 5,
        "mode": "0440",
        "sha256": hashlib.sha256(b"exact").hexdigest(),
    }


def test_descriptor_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"x")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        r2.pin_regular("alias/file", root=tmp_path)


def test_descriptor_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "alias").symlink_to(target)
    with pytest.raises(OSError):
        r2.pin_regular("alias", root=tmp_path)


def test_descriptor_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.write_bytes(b"x")
    os.link(first, tmp_path / "second")
    with pytest.raises(r2.R2Error, match="single-link"):
        r2.pin_regular("first", root=tmp_path)


def test_all_r1_controls_and_receipt_are_exactly_pinned() -> None:
    for row in r2.R1_PINS:
        assert r2.pin_regular(row["path"]) == row


def test_r1_failure_receipt_is_exact_and_self_fingerprinted() -> None:
    assert r2.pin_regular(r2.R1_FAILURE_REL) == r2.R1_FAILURE_PIN
    value = r2.strict_json(r2.read_regular(r2.R1_FAILURE_REL), "failure")
    assert value["self_fingerprint"] == r2.fingerprint(value, "self_fingerprint")
    assert value["authority"]["build_authorized"] is False


def test_r1_overclaim_is_detected_without_mutation() -> None:
    _, receipt, failure = r2.load_r1_lineage()
    assert receipt["independent_test_replay"] == {
        **receipt["independent_test_replay"],
        "collected": 36,
        "passed": 36,
        "failed": 0,
    }
    assert failure["observed_test_replay"]["passed"] == 35
    assert failure["observed_test_replay"]["failed"] == 1


def test_corrected_probe_finishes_cleanup_after_collision_assertions() -> None:
    r1, receipt, _ = r2.load_r1_lineage()
    assert r2.corrected_r1_publication_probe(r1, receipt) == {
        "publication_assertions_passed": True,
        "collision_rejected": True,
        "first_receipt_preserved": True,
        "cleanup_parent_chmod_precedes_unlink": True,
        "cleanup_completed": True,
    }


def test_r2_audit_without_trees_preserves_two_subject_p1_findings() -> None:
    result = r2.audit_current_state(replay_trees=False)
    assert result["status"] == "r2_independent_replay_complete_subject_rejected"
    assert result["subject"]["source_count"] == 39
    assert result["subject"]["authority_all_false"] is True
    assert result["r1_observed_tests"]["passed"] == 35


def test_r2_schema_is_strict_draft_2020_12() -> None:
    schema = r2.strict_json(r2.read_regular(r2.SCHEMA_REL), "schema")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False


def test_r2_full_tree_replay_is_exact() -> None:
    result = r2.audit_current_state(replay_trees=True)
    trees = result["subject"]["trees"]
    assert trees["a2_full_tree"]["tree_sha256"] == "cf2c1c9903eca67e92397e11d6484d7b66f939c57e6fcff991deaa8f028ef89b"
    assert trees["a3_full_tree"]["tree_sha256"] == "509977cd3945f62c45c6d717438e08477827103a53cadf87c3da1c0c89001293"
    assert trees["a3_symlink_closure"]["symlinks"] == 47
    assert trees["a3_symlink_closure"]["dangling"] == 0


def test_checked_r2_receipt_verifies_terminal_rejection() -> None:
    assert r2.verify_receipt(r2.load_review()) == {
        "status": "verified_r2_terminal_rejection",
        "decision": "REJECT",
        "severity_counts": {"P0": 0, "P1": 2, "P2": 1},
        "build_authorized": False,
    }


def test_resigned_build_overclaim_is_rejected() -> None:
    value = r2.load_review()
    value["authority"]["dedicated_cpu_image_build_authorized"] = True
    with pytest.raises(jsonschema.ValidationError):
        r2.verify_receipt(_resign(value))


def test_resigned_finding_removal_is_rejected() -> None:
    value = r2.load_review()
    value["findings"] = value["findings"][:-1]
    with pytest.raises(jsonschema.ValidationError):
        r2.verify_receipt(_resign(value))


def test_unsigned_status_mutation_is_rejected() -> None:
    value = r2.load_review()
    value["status"] = "mutated"
    with pytest.raises(jsonschema.ValidationError):
        r2.verify_receipt(value)


def test_r2_publication_is_noreplace_and_cleanup_order_is_correct(tmp_path: Path) -> None:
    value = r2.load_review()
    container = tmp_path / "results"
    container.mkdir()
    output = container / "r2-run/receipt.json"
    pin = r2.publish_review(output, value, expected_output=output)
    first = output.read_bytes()
    assert pin == (len(first), hashlib.sha256(first).hexdigest())
    with pytest.raises(r2.R2Error, match="run already exists"):
        r2.publish_review(output, value, expected_output=output)
    assert output.read_bytes() == first
    _cleanup(output)


def test_r2_publication_rejects_symlink_container(tmp_path: Path) -> None:
    value = r2.load_review()
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "run/receipt.json"
    with pytest.raises(r2.R2Error, match="container differs"):
        r2.publish_review(output, value, expected_output=output)


def test_r2_review_control_pins_are_current_and_read_only() -> None:
    value = r2.load_review()
    assert len(value["review_control_pins"]) == 3
    for row in value["review_control_pins"]:
        assert r2.pin_regular(row["path"]) == row
        assert row["mode"] in {"0440", "0555"}


def test_r2_authority_is_entirely_false_and_next_step_is_isolated() -> None:
    value = r2.load_review()
    assert all(flag is False for flag in value["authority"].values())
    assert value["next_authority"] == {
        "isolated_successor_required": True,
        "successor": "SafetyVision R5A3.2 post-publication verifier and tests",
        "may_mutate_a3_or_a31": False,
        "may_authorize_build_directly": False,
    }
