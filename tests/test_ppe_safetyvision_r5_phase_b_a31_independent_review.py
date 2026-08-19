from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import ppe_safetyvision_r5_phase_b_a31_independent_review as review


ROOT = Path(__file__).resolve().parents[1]


def _pin(path: Path) -> tuple[int, str, str]:
    raw = path.read_bytes()
    return len(raw), f"{stat.S_IMODE(path.stat().st_mode):04o}", hashlib.sha256(raw).hexdigest()


def _resign_review(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("review_fingerprint_sha256", None)
    result["review_fingerprint_sha256"] = review.review_fingerprint(result)
    return result


def _thaw_review_fixture(output: Path) -> None:
    if output.exists():
        os.chmod(output, 0o600, follow_symlinks=False)
        output.unlink()
    if output.parent.exists():
        os.chmod(output.parent, 0o700, follow_symlinks=False)
        output.parent.rmdir()


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(review.ReviewError, match="duplicate JSON key"):
        review.strict_json(b'{"x":1,"x":2}', "fixture")


def test_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(review.ReviewError, match="non-finite JSON token"):
        review.strict_json(b'{"x":NaN}', "fixture")


def test_strict_json_rejects_bom_and_non_object() -> None:
    with pytest.raises(review.ReviewError, match="BOM"):
        review.strict_json(b"\xef\xbb\xbf{}", "fixture")
    with pytest.raises(review.ReviewError, match="root is not an object"):
        review.strict_json(b"[]", "fixture")


def test_self_fingerprint_detects_mutation() -> None:
    value = {"status": "closed"}
    value["self_fingerprint"] = hashlib.sha256(review.canonical_bytes(value)).hexdigest()
    review.validate_fingerprint(value, "fixture")
    value["status"] = "overclaim"
    with pytest.raises(review.ReviewError, match="fingerprint differs"):
        review.validate_fingerprint(value, "fixture")


def test_relative_path_rejects_escape_absolute_backslash_and_nul() -> None:
    for candidate in ("../escape", "/absolute", "a\\b", "a\x00b"):
        with pytest.raises(review.ReviewError, match="unsafe relative path"):
            review._relative_parts(candidate)


def test_workspace_reader_reads_exact_single_link_regular(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"exact")
    artifact.chmod(0o440)
    raw, pin = review.WorkspaceReader(tmp_path).read_regular(
        "artifact.bin", expected=(5, "0440", hashlib.sha256(b"exact").hexdigest())
    )
    assert raw == b"exact"
    assert pin["path"] == "artifact.bin"


def test_workspace_reader_rejects_pin_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual")
    artifact.chmod(0o440)
    with pytest.raises(review.ReviewError, match="exact pin/mode differs"):
        review.WorkspaceReader(tmp_path).read_regular(
            "artifact.bin", expected=(6, "0440", "0" * 64)
        )


def test_workspace_reader_rejects_mode_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual")
    artifact.chmod(0o640)
    digest = hashlib.sha256(b"actual").hexdigest()
    with pytest.raises(review.ReviewError, match="exact pin/mode differs"):
        review.WorkspaceReader(tmp_path).read_regular(
            "artifact.bin", expected=(6, "0440", digest)
        )


def test_workspace_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "artifact.bin").write_bytes(b"content")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        review.WorkspaceReader(tmp_path).read_regular("alias/artifact.bin")


def test_workspace_reader_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"content")
    (tmp_path / "alias.bin").symlink_to(target)
    with pytest.raises(OSError):
        review.WorkspaceReader(tmp_path).read_regular("alias.bin")


def test_workspace_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"content")
    os.link(first, second)
    with pytest.raises(review.ReviewError, match="single-link"):
        review.WorkspaceReader(tmp_path).read_regular("first.bin")


def test_tree_attestation_has_exact_predecessor_record_order(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.txt").write_bytes(b"A")
    (tree / "dir").mkdir()
    (tree / "dir/z.txt").write_bytes(b"Z")
    (tree / "link").symlink_to("a.txt")
    tree.chmod(0o755)
    (tree / "dir").chmod(0o755)
    (tree / "a.txt").chmod(0o644)
    (tree / "dir/z.txt").chmod(0o644)
    report = review.attest_tree(review.WorkspaceReader(tmp_path), "tree")
    records = [
        {"kind": "file", "path": "a.txt", "mode": 0o644, "bytes": 1, "sha256": hashlib.sha256(b"A").hexdigest()},
        {"kind": "directory", "path": "dir", "mode": 0o755},
        {"kind": "symlink", "path": "link", "target": "a.txt"},
        {"kind": "file", "path": "dir/z.txt", "mode": 0o644, "bytes": 1, "sha256": hashlib.sha256(b"Z").hexdigest()},
    ]
    digest = hashlib.sha256(b"".join(review.canonical_bytes(row) + b"\n" for row in records)).hexdigest()
    assert report["full_tree"] == {
        "directories": 2,
        "regular_file_bytes": 2,
        "regular_files": 2,
        "symlinks": 1,
        "tree_sha256": digest,
    }


def test_tree_attestation_reports_writable_inventory(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir(mode=0o755)
    (tree / "file").write_bytes(b"x")
    report = review.attest_tree(review.WorkspaceReader(tmp_path), "tree")
    assert report["mode_inventory"]["writable_directories_or_regular_files"] == 2
    assert report["mode_inventory"]["special_entries"] == 0


def test_tree_attestation_rejects_hardlinks(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "first").write_bytes(b"x")
    os.link(tree / "first", tree / "second")
    with pytest.raises(review.ReviewError, match="hardlink"):
        review.attest_tree(review.WorkspaceReader(tmp_path), "tree")


def test_tree_attestation_rejects_special_entries(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    os.mkfifo(tree / "fifo")
    with pytest.raises(review.ReviewError, match="special tree entry"):
        review.attest_tree(review.WorkspaceReader(tmp_path), "tree")


def test_virtual_symlink_resolves_relative_target(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "links").mkdir(parents=True)
    (rootfs / "target").write_bytes(b"ok")
    (rootfs / "links/link").symlink_to("../target")
    with review.WorkspaceReader(tmp_path).directory_fd("rootfs") as (descriptor, _):
        row = review.resolve_virtual_symlink(descriptor, "links/link")
    assert row["terminal"] == "/target"
    assert row["hops"] == 1


def test_virtual_symlink_resolves_absolute_inside_root(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "target").write_bytes(b"ok")
    (rootfs / "link").symlink_to("/target")
    with review.WorkspaceReader(tmp_path).directory_fd("rootfs") as (descriptor, _):
        row = review.resolve_virtual_symlink(descriptor, "link")
    assert row["terminal"] == "/target"


def test_virtual_symlink_rejects_dangling_target(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "link").symlink_to("missing")
    with review.WorkspaceReader(tmp_path).directory_fd("rootfs") as (descriptor, _):
        with pytest.raises(review.ReviewError, match="dangling"):
            review.resolve_virtual_symlink(descriptor, "link")


def test_virtual_symlink_rejects_cycle(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "a").symlink_to("b")
    (rootfs / "b").symlink_to("a")
    with review.WorkspaceReader(tmp_path).directory_fd("rootfs") as (descriptor, _):
        with pytest.raises(review.ReviewError, match="cycle"):
            review.resolve_virtual_symlink(descriptor, "a")


def test_virtual_symlink_rejects_root_escape(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "link").symlink_to("../../outside")
    with review.WorkspaceReader(tmp_path).directory_fd("rootfs") as (descriptor, _):
        with pytest.raises(review.ReviewError, match="escapes virtual root"):
            review.resolve_virtual_symlink(descriptor, "link")


def test_virtual_symlink_ledger_is_deterministic(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    rootfs.mkdir()
    (rootfs / "target").write_bytes(b"ok")
    (rootfs / "link").symlink_to("target")
    reader = review.WorkspaceReader(tmp_path)
    first = review.validate_symlink_closure(reader, "rootfs", [{"path": "link", "target": "target"}])
    second = review.validate_symlink_closure(reader, "rootfs", [{"path": "link", "target": "target"}])
    assert first == second
    assert first["symlinks"] == 1


def test_exact_handoff_and_contract_are_externally_pinned() -> None:
    reader = review.WorkspaceReader()
    handoff, handoff_pin = review._validate_handoff(reader)
    contract, contract_pin, sources, nested_sources, _ = review._validate_contract(reader, handoff)
    assert handoff_pin == {
        "path": review.HANDOFF_REL,
        "bytes": review.HANDOFF_PIN[0],
        "mode": review.HANDOFF_PIN[1],
        "sha256": review.HANDOFF_PIN[2],
    }
    assert contract_pin["sha256"] == review.A31_CONTRACT_PIN[2]
    assert len(sources) == 15
    assert len(nested_sources) == 24
    assert contract["authorization"]["dedicated_cpu_image_build"] is False


def test_strict_current_receipts_and_false_boundaries() -> None:
    receipts, pins = review._strict_receipts(review.WorkspaceReader())
    assert len(pins) == 6
    assert receipts["recovery"]["authorization"]["image_build"] is False
    assert receipts["recovery"]["execution"]["context_copied"] is False
    assert receipts["failure"]["execution"]["gpu_used"] is False


def test_historical_collision_and_recovery_dirs_are_distinct() -> None:
    report = review._validate_collision_and_recovery_dirs(review.WorkspaceReader())
    assert report["historical_collision"]["empty"] is True
    assert report["historical_collision"]["reused"] is False
    assert report["recovery_run"]["published"] is True


def test_parent_order_mutation_probe_keeps_run_absent_and_no_replace() -> None:
    report = review.probe_parent_order_publication()
    assert report == {
        "run_absent_when_stage_created": True,
        "same_directory_renameat2_noreplace": True,
        "second_publish_collision_rejected": True,
        "first_receipt_unchanged_after_collision": True,
        "stale_stage_directories": [],
    }


def test_author_surface_has_no_execution_calls_and_detects_regression() -> None:
    reader = review.WorkspaceReader()
    program = reader.read_regular("validation/ppe_safetyvision_r5_phase_a31_recovery.py")[0]
    tests = reader.read_regular("tests/test_ppe_safetyvision_r5_phase_a31_recovery.py")[0]
    a3_program = reader.read_regular("validation/ppe_safetyvision_r5_phase_a3.py")[0]
    a3_tests = reader.read_regular("tests/test_ppe_safetyvision_r5_phase_a3.py")[0]
    result = review._analyze_author_surface(program, tests, a3_program, a3_tests)
    assert result["docker_surface"] is False
    assert result["gpu_surface"] is False
    assert result["post_publication_regressions"]["a31"]["deterministic_current_suite_result"] == {
        "collected": 7,
        "passed": 6,
        "failed": 1,
    }
    assert result["post_publication_regressions"]["a3"]["deterministic_current_suite_result"] == {
        "collected": 18,
        "passed": 17,
        "failed": 1,
    }


def test_current_audit_without_tree_or_probe_is_fail_closed_rejection() -> None:
    result = review.audit_current_state(replay_trees=False, publication_probe=False)
    assert result["status"].endswith("post_publication_verifiers_non_idempotent")
    assert result["source_count"] == 39
    assert result["a31_source_count"] == 15
    assert result["a3_nested_source_count"] == 24
    assert result["trees"] == {"replayed": False}
    assert result["authority_all_false"] is True


def test_live_full_tree_and_symlink_replay_matches_frozen_identity() -> None:
    result = review.audit_current_state(replay_trees=True, publication_probe=False)
    assert result["trees"]["a2_full_tree"] == review.A2_TREE
    assert result["trees"]["a3_full_tree"] == review.A3_TREE
    assert result["trees"]["a3_payload_tree"] == review.A3_PAYLOAD_TREE
    assert result["trees"]["a3_rootfs_tree"] == review.A3_ROOTFS
    assert result["trees"]["a3_symlink_closure"] == review.SYMLINK_CLOSURE


def test_checked_in_rejection_receipt_verifies() -> None:
    result = review.verify_receipt(review.load_review())
    assert result == {
        "status": "verified_frozen_phase_b_rejection",
        "decision": "REJECT",
        "severity_counts": {"P0": 0, "P1": 2, "P2": 0},
        "build_authorized": False,
    }


def test_resigned_build_overclaim_is_rejected_by_strict_schema() -> None:
    value = review.load_review()
    value["authority"]["dedicated_cpu_image_build_authorized"] = True
    value = _resign_review(value)
    schema = review.strict_json((ROOT / review.SCHEMA_REL).read_bytes(), "schema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=value, schema=schema)


def test_resigned_finding_removal_is_rejected_by_verifier() -> None:
    value = review.load_review()
    value["findings"] = []
    value = _resign_review(value)
    with pytest.raises((jsonschema.ValidationError, review.ReviewError)):
        review.verify_receipt(value)


def test_review_fingerprint_detects_unsigned_mutation() -> None:
    value = review.load_review()
    value["status"] = "mutated"
    with pytest.raises((jsonschema.ValidationError, review.ReviewError)):
        review.verify_receipt(value)


def test_review_publication_is_noreplace_and_collision_preserves_first(tmp_path: Path) -> None:
    value = review.load_review()
    container = tmp_path / "results"
    container.mkdir()
    output = container / "phase-b-independent-review/receipt.json"
    pin = review.publish_review(output, value, expected_output=output)
    first = output.read_bytes()
    assert pin == (len(first), hashlib.sha256(first).hexdigest())
    with pytest.raises(review.ReviewError, match="run already exists"):
        review.publish_review(output, value, expected_output=output)
    assert output.read_bytes() == first
    _thaw_review_fixture(output)


def test_review_publication_rejects_symlink_container(tmp_path: Path) -> None:
    value = review.load_review()
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = alias / "run/receipt.json"
    with pytest.raises(review.ReviewError, match="container differs"):
        review.publish_review(output, value, expected_output=output)


def test_review_source_pins_are_exact_read_only_single_link_files() -> None:
    value = review.load_review()
    assert len(value["review_source_pins"]) == 3
    reader = review.WorkspaceReader()
    for row in value["review_source_pins"]:
        assert reader.pin_regular(row["path"]) == row
        assert row["mode"] in {"0440", "0555"}


def test_rejection_never_authorizes_downstream_execution() -> None:
    value = review.load_review()
    assert value["decision"] == "REJECT"
    assert all(flag is False for flag in value["authority"].values())
    assert value["next_authority"]["isolated_successor_required"] is True
    assert value["next_authority"]["may_authorize_build_directly"] is False
