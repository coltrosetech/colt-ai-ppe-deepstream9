from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from validation import person_rtdetrv4_tensorrt_r14j_v6_independent_review as review


@pytest.fixture(scope="session")
def audit() -> dict[str, Any]:
    return review.audit_current_state()


@pytest.fixture(scope="session")
def workloads() -> dict[int, dict[str, Any]]:
    return {
        profile: json.loads(
            (
                review.ROOT
                / review.SUBJECT_PINS[f"workload_{profile}"]["path"]
            ).read_text(encoding="utf-8")
        )
        for profile in (640, 960)
    }


@pytest.fixture(scope="session")
def r14i_template() -> dict[str, Any]:
    return json.loads(
        (
            review.ROOT / review.FOUNDATION_PINS["r14i_template"]["path"]
        ).read_text(encoding="utf-8")
    )


def nested_set(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    parts = dotted.split(".")
    cursor: Any = value
    for part in parts[:-1]:
        cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
    if isinstance(cursor, list):
        cursor[int(parts[-1])] = replacement
    else:
        cursor[parts[-1]] = replacement


def test_strict_json_accepts_object() -> None:
    assert review.strict_json(b'{"x":1}', "ok") == {"x": 1}


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{"x":1,"x":2}', "duplicate JSON key"),
        (b'{"x":NaN}', "non-finite"),
        (b'{"x":Infinity}', "non-finite"),
        (b'\xef\xbb\xbf{"x":1}', "BOM"),
        (b'{"x":"\xff"}', "invalid JSON"),
        (b'[]', "root differs"),
    ],
)
def test_strict_json_rejects_ambiguous_input(raw: bytes, match: str) -> None:
    with pytest.raises(review.R14JV6IndependentReviewError, match=match):
        review.strict_json(raw, "bad")


@pytest.mark.parametrize(
    "path", ["", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "a/"]
)
def test_reader_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(review.R14JV6IndependentReviewError, match="unsafe path"):
        review._parts(path)


def test_reader_replays_exact_regular(tmp_path: Path) -> None:
    target = tmp_path / "subject"
    target.write_bytes(b"subject")
    target.chmod(0o440)
    expected = review.pin("subject", 7, hashlib.sha256(b"subject").hexdigest())
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.read_exact(expected, "subject") == b"subject"


def test_reader_rejects_hardlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.write_bytes(b"x")
    first.chmod(0o440)
    os.link(first, tmp_path / "second")
    expected = review.pin("first", 1, hashlib.sha256(b"x").hexdigest())
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.R14JV6IndependentReviewError, match="held regular"):
            reader.read_exact(expected, "hardlink")


def test_reader_rejects_final_symlink(tmp_path: Path) -> None:
    (tmp_path / "target").write_bytes(b"x")
    (tmp_path / "link").symlink_to("target")
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.read_exact(
                review.pin("link", 1, hashlib.sha256(b"x").hexdigest(), "0644"),
                "link",
            )


def test_reader_rejects_parent_symlink(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "item").write_bytes(b"x")
    (tmp_path / "link").symlink_to("real", target_is_directory=True)
    with review.WorkspaceReader(tmp_path) as reader:
        with pytest.raises(OSError):
            reader.read_exact(
                review.pin("link/item", 1, hashlib.sha256(b"x").hexdigest(), "0644"),
                "parent link",
            )


def test_reader_absence_is_no_follow(tmp_path: Path) -> None:
    with review.WorkspaceReader(tmp_path) as reader:
        assert reader.absent("missing/tree/item") is True


@pytest.mark.parametrize(
    "name", ["controller_schema", "workload_schema", "handoff_schema"]
)
def test_author_schemas_are_valid_and_recursively_closed(name: str) -> None:
    path = review.ROOT / review.SUBJECT_PINS[name]["path"]
    schema = review.strict_json(path.read_bytes(), name)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert review.schemas_recursively_closed(schema)["open_object_schemas"] == 0


def test_independent_schema_is_valid_and_recursively_closed() -> None:
    schema = review.strict_json((review.ROOT / review.SCHEMA_REL).read_bytes(), "review")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert review.schemas_recursively_closed(schema)["object_schemas"] >= 15


def test_audit_accepts_only_static_definition(audit: dict[str, Any]) -> None:
    assert audit["decision"] == "ACCEPT"
    assert audit["severity_counts"] == {"P0": 0, "P1": 0, "P2": 0}
    assert audit["accepted_scope"]["static_v6_engine_build_definition"] is True
    assert audit["accepted_scope"]["runtime_execution_authorized"] is False


@pytest.mark.parametrize("name", sorted(review.SUBJECT_PINS))
def test_every_subject_pin_rehashes_exactly(name: str) -> None:
    expected = review.SUBJECT_PINS[name]
    with review.WorkspaceReader() as reader:
        raw = reader.read_exact(expected, name)
    assert len(raw) == expected["bytes"]
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


@pytest.mark.parametrize(
    "name",
    ["r14i", "gpu_lease_v6", "driver_r7", "deepstream91_native", "onnx_inputs"],
)
def test_accepted_foundation_groups_replayed(audit: dict[str, Any], name: str) -> None:
    assert name in audit["verification"]["foundations"]


def test_handoff_exact_pin_and_fingerprint(audit: dict[str, Any]) -> None:
    handoff = audit["subject"]["handoff"]
    assert handoff["pin"] == review.HANDOFF_PIN
    assert handoff["canonical_fingerprint_sha256"] == review.HANDOFF_FINGERPRINT
    assert handoff["authority_closed"] is True


def test_author_source_is_inert_ast(audit: dict[str, Any]) -> None:
    surface = audit["subject"]["author_source_ast"]
    assert surface["parsed_as_inert_ast"] is True
    assert surface["imported"] is False and surface["executed"] is False
    assert surface["execution_adapter_present"] is False


@pytest.mark.parametrize("name", review.EXECUTION_FLAGS)
def test_every_author_authority_literal_is_false(
    audit: dict[str, Any], name: str
) -> None:
    assert audit["subject"]["author_source_ast"]["execution_flag_literals"][name] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_profiles_are_fp16_batch_1_12_12(
    audit: dict[str, Any], profile: int
) -> None:
    row = audit["verification"]["profiles"][str(profile)]
    assert row["optimization_profile"] == review.expected_profile(profile)
    assert row["precision"] == {"fp16": True, "int8": False, "tf32": False}


@pytest.mark.parametrize("profile", [640, 960])
def test_profiles_have_exact_foreground_argv(
    workloads: dict[int, dict[str, Any]], profile: int
) -> None:
    workload = workloads[profile]["future_v6_workload"]
    assert workload["argv"] == review.expected_argv(profile)
    assert workload["argv_sha256"] == review.command_argv_sha256(workload["argv"])
    assert workloads[profile]["runtime"]["foreground"] is True


def test_profiles_are_sequential_and_not_parallel(audit: dict[str, Any]) -> None:
    rows = audit["verification"]["profiles"]
    assert rows["640"]["sequence_index"] == 1
    assert rows["960"]["sequence_index"] == 2
    assert rows["960"]["required_previous_profile"] == 640
    assert all(row["parallel_execution"] is False for row in rows.values())


@pytest.mark.parametrize("profile", [640, 960])
def test_outputs_are_absent_fresh_and_no_overwrite(
    workloads: dict[int, dict[str, Any]], profile: int
) -> None:
    assert workloads[profile]["outputs"] == review.expected_outputs(profile)
    for row in workloads[profile]["outputs"].values():
        assert not (review.ROOT / row["path"]).exists()


def test_output_sets_are_disjoint(audit: dict[str, Any]) -> None:
    assert audit["verification"]["outputs"] == {
        "profile_640_count": 7,
        "profile_960_count": 7,
        "sets_disjoint": True,
        "all_absent": True,
        "fresh_mode": "0700",
        "overwrite_authorized": False,
    }


@pytest.mark.parametrize(
    "profile,path,replacement",
    [
        (640, "sequence.index", 2),
        (960, "sequence.required_previous_profile", None),
        (640, "precision.fp16", False),
        (960, "optimization_profile.images.opt.0", 11),
        (640, "runtime.foreground", False),
        (640, "future_v6_workload.argv_sha256", "0" * 64),
        (960, "future_v6_workload.missing_authority.plan_id", "live"),
        (640, "outputs.output_directory.materialization_authorized", True),
        (960, "outputs.final_engine.overwrite_authorized", True),
        (640, "claim_boundary.gpu_executed", True),
    ],
)
def test_profile_mutations_fail_closed(
    workloads: dict[int, dict[str, Any]],
    r14i_template: dict[str, Any],
    profile: int,
    path: str,
    replacement: Any,
) -> None:
    mutated = copy.deepcopy(workloads[profile])
    nested_set(mutated, path, replacement)
    with review.WorkspaceReader() as reader:
        with pytest.raises(review.R14JV6IndependentReviewError):
            review.verify_profile(profile, mutated, r14i_template, reader)


def test_build_and_verify_receipt(audit: dict[str, Any]) -> None:
    receipt = review.build_receipt(
        audit, author_test_replays=2, independent_tests=70
    )
    result = review.verify_receipt(receipt)
    assert result["decision"] == "ACCEPT"
    assert result["execution_authorized"] is False
    assert receipt["review_fingerprint_sha256"] == review.fingerprint(
        receipt, "review_fingerprint_sha256"
    )


@pytest.mark.parametrize(
    "path,replacement",
    [
        ("decision", "REJECT"),
        ("authority_boundary.execution_authority_granted", True),
        ("accepted_scope.engine_build_executed", True),
        ("permissions.call_docker", True),
        ("resource_boundary.gpu_called", True),
        ("author_test_replay.independent_replays", 1),
    ],
)
def test_receipt_authority_mutations_fail_closed(
    audit: dict[str, Any], path: str, replacement: Any
) -> None:
    receipt = review.build_receipt(
        audit, author_test_replays=2, independent_tests=70
    )
    nested_set(receipt, path, replacement)
    receipt["review_fingerprint_sha256"] = review.fingerprint(
        receipt, "review_fingerprint_sha256"
    )
    with pytest.raises(
        (review.R14JV6IndependentReviewError, jsonschema.ValidationError)
    ):
        review.verify_receipt(receipt)


def test_control_modes_are_immutable(audit: dict[str, Any]) -> None:
    modes = {item["path"]: item["mode"] for item in audit["review_control_pins"]}
    assert modes == {
        review.REVIEWER_REL: "0555",
        review.SCHEMA_REL: "0440",
        review.TEST_REL: "0440",
        review.DOC_REL: "0440",
    }
    for item in audit["review_control_pins"]:
        assert stat.S_IMODE((review.ROOT / item["path"]).stat().st_mode) == int(
            item["mode"], 8
        )
