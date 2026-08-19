from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

import jsonschema
import pytest

import validation.deepstream91_dgpu_runtime_smoke_r1_independent_review as review


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_strict_json_accepts_closed_object() -> None:
    assert review.strict_json(b'{"a":1}', "x") == {"a": 1}


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'\xef\xbb\xbf{"a":1}',
        b'[]',
        b'',
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_bom_and_nonobject(raw: bytes) -> None:
    with pytest.raises(review.ReviewError):
        review.strict_json(raw, "adversarial")


@pytest.mark.parametrize("path", ["../x", "a/../x", "/tmp/x", ".", "a/./b"])
def test_reader_rejects_path_traversal(path: str, tmp_path: Path) -> None:
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.ReviewError):
            reader.read(path)


def test_reader_accepts_exact_regular_single_link(tmp_path: Path) -> None:
    target = tmp_path / "value.json"
    target.write_bytes(b"{}")
    target.chmod(0o440)
    expected = review.pin("value.json", 2, hashlib.sha256(b"{}").hexdigest(), "0440")
    with review.HeldWorkspaceReader(tmp_path) as reader:
        assert reader.read_exact(expected, "fixture") == b"{}"


def test_reader_rejects_final_symlink(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to("real")
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises((review.ReviewError, OSError)):
            reader.read("link")


def test_reader_rejects_symlink_ancestor(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "value").write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to("real", target_is_directory=True)
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises((review.ReviewError, OSError)):
            reader.read("link/value")


def test_reader_rejects_hardlink_nlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.write_text("x", encoding="utf-8")
    os.link(first, tmp_path / "second")
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.ReviewError):
            reader.read("first")


def test_reader_rejects_mode_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "value"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o600)
    expected = review.pin("value", 1, hashlib.sha256(b"x").hexdigest(), "0440")
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.ReviewError, match="mode differs"):
            reader.read_exact(expected, "fixture")


def test_reader_detects_named_file_toctou(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "value"
    target.write_bytes(b"old")
    real_pread = review.os.pread
    swapped = False

    def racing_pread(fd: int, size: int, offset: int) -> bytes:
        nonlocal swapped
        data = real_pread(fd, size, offset)
        if not swapped:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"new")
            os.replace(replacement, target)
            swapped = True
        return data

    monkeypatch.setattr(review.os, "pread", racing_pread)
    with review.HeldWorkspaceReader(tmp_path) as reader:
        with pytest.raises(review.ReviewError, match="TOCTOU"):
            reader.read("value")


def test_author_and_worker_ast_baseline() -> None:
    controller = (ROOT / review.AUTHOR_SOURCE_REL).read_bytes()
    worker = (ROOT / review.WORKER_REL).read_bytes()
    projection = review.verify_author_and_worker_ast(controller, worker)
    assert projection["worker_subprocess_call_count"] == 1
    assert projection["worker_imported_or_executed"] is False


def test_worker_ast_rejects_network_import() -> None:
    controller = (ROOT / review.AUTHOR_SOURCE_REL).read_bytes()
    worker = b"import socket\n" + (ROOT / review.WORKER_REL).read_bytes()
    with pytest.raises(review.ReviewError, match="network/model"):
        review.verify_author_and_worker_ast(controller, worker)


def test_worker_ast_rejects_shell_keyword() -> None:
    controller = (ROOT / review.AUTHOR_SOURCE_REL).read_bytes()
    worker = (ROOT / review.WORKER_REL).read_bytes().replace(
        b"check=False,\n", b"check=False,\n            shell=True,\n", 1
    )
    with pytest.raises(review.ReviewError, match="shell"):
        review.verify_author_and_worker_ast(controller, worker)


def test_worker_ast_rejects_duplicate_probe_label() -> None:
    controller = (ROOT / review.AUTHOR_SOURCE_REL).read_bytes()
    worker = (ROOT / review.WORKER_REL).read_bytes().replace(
        b'"gst_inspect_nvstreammux"', b'"gst_inspect_nvinfer"', 1
    )
    with pytest.raises(review.ReviewError, match="duplicate probe"):
        review.verify_author_and_worker_ast(controller, worker)


def test_worker_ast_is_never_imported_by_reviewer_source() -> None:
    tree = ast.parse((ROOT / review.VERIFIER_REL).read_text(encoding="utf-8"))
    imported = review.imported_roots(tree)
    assert "deepstream91_dgpu_runtime_smoke_probe_r1" not in imported
    assert "subprocess" not in imported


def test_expected_argv_has_exact_sandbox_and_mount_boundary() -> None:
    argv = review.expected_docker_argv()
    assert argv[:2] == ["/usr/bin/docker", "run"]
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges:true" in argv
    assert "--rm" not in argv and "-d" not in argv
    assert len([item for item in argv if item.startswith("--mount=")]) == 2
    assert not any(item.startswith("--label") for item in argv)


def valid_plan_and_schema() -> tuple[dict, dict]:
    return load_json(review.PLAN_REL), load_json(review.LEASE_PINS["v5_plan_schema"]["path"])


def resign_plan(plan: dict) -> None:
    plan["workload"]["argv_sha256"] = review.command_argv_sha256(plan["workload"]["argv"])
    plan["plan_fingerprint_sha256"] = review.fingerprint(plan, "plan_fingerprint_sha256")


def test_plan_baseline_is_static_valid() -> None:
    plan, schema = valid_plan_and_schema()
    assert review.verify_plan(plan, schema)["foreground"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "label",
        "rm",
        "detach",
        "worker_mount_rw",
        "extra_output_mount",
        "output_path_drift",
        "image_index",
        "image_duplicate",
        "inner_authority_false",
    ],
)
def test_plan_rejects_argv_label_mount_output_and_authority_mutations(mutation: str) -> None:
    plan, schema = valid_plan_and_schema()
    argv = plan["workload"]["argv"]
    if mutation == "label":
        argv.insert(2, "--label=attacker=true")
        plan["workload"]["image_argv_index"] += 1
    elif mutation == "rm":
        argv.insert(2, "--rm")
        plan["workload"]["image_argv_index"] += 1
    elif mutation == "detach":
        argv.insert(2, "--detach")
        plan["workload"]["image_argv_index"] += 1
    elif mutation == "worker_mount_rw":
        argv[15] = argv[15].replace(",readonly", ",rw")
    elif mutation == "extra_output_mount":
        argv.insert(17, "--mount=type=bind,src=/tmp,dst=/extra,rw")
        plan["workload"]["image_argv_index"] += 1
    elif mutation == "output_path_drift":
        argv[16] = argv[16].replace("dst=/output", "dst=/other")
    elif mutation == "image_index":
        plan["workload"]["image_argv_index"] = 18
    elif mutation == "image_duplicate":
        argv.append(review.IMAGE_REFERENCE)
    elif mutation == "inner_authority_false":
        plan["execution_authorized"] = False
    resign_plan(plan)
    with pytest.raises((review.ReviewError, jsonschema.ValidationError)):
        review.verify_plan(plan, schema)


def test_handoff_baseline_and_output_absence() -> None:
    handoff = load_json(review.HANDOFF_REL)
    schema = load_json(review.HANDOFF_SCHEMA_REL)
    assert review.verify_handoff(handoff, schema)["authority_closed"] is True
    assert not (ROOT / review.OUTPUT_TREE_REL).exists()


def test_handoff_rejects_output_materialization_overclaim() -> None:
    handoff = load_json(review.HANDOFF_REL)
    schema = load_json(review.HANDOFF_SCHEMA_REL)
    handoff["runtime_plan"]["output_boundary"]["materialization_authorized"] = True
    with pytest.raises((review.ReviewError, jsonschema.ValidationError)):
        review.verify_handoff(handoff, schema)


def test_handoff_rejects_outer_authority_overclaim() -> None:
    handoff = load_json(review.HANDOFF_REL)
    schema = load_json(review.HANDOFF_SCHEMA_REL)
    handoff["authority"]["outer_acceptance"] = True
    with pytest.raises((review.ReviewError, jsonschema.ValidationError)):
        review.verify_handoff(handoff, schema)


def test_independent_schema_is_recursively_closed() -> None:
    schema = load_json(review.SCHEMA_REL)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert review.schemas_recursively_closed(schema)["open_object_schemas"] == 0


def test_author_result_and_handoff_schemas_are_closed() -> None:
    for path in (review.RESULT_SCHEMA_REL, review.HANDOFF_SCHEMA_REL):
        assert review.schemas_recursively_closed(load_json(path))["open_object_schemas"] == 0


def test_author_handoff_pin_is_exact() -> None:
    raw = (ROOT / review.HANDOFF_REL).read_bytes()
    assert len(raw) == review.HANDOFF_PIN["bytes"]
    assert hashlib.sha256(raw).hexdigest() == review.HANDOFF_PIN["sha256"]


def test_current_host_drift_is_exact_regular_file_evidence() -> None:
    for key, expected in review.CURRENT_HOST_TOOLS.items():
        observed = review.read_absolute_regular(expected["path"])
        assert {field: observed[field] for field in ("path", "bytes", "sha256")} == expected
    assert review.CURRENT_HOST_TOOLS["docker_cli"] != review.V5_EXPECTED_HOST_TOOLS["docker_cli"]
    assert review.CURRENT_HOST_TOOLS["nvidia_smi"] != review.V5_EXPECTED_HOST_TOOLS["nvidia_smi"]


def test_authority_schema_rejects_execution_overclaim() -> None:
    schema = load_json(review.SCHEMA_REL)["$defs"]["authority"]
    authority = {
        "subject_static_acceptance": True,
        "execution_readiness_accepted": False,
        "outer_acceptance": False,
        "v5_plan_activation_authorized": False,
        "launch_authorized": False,
        "execution_authorized": False,
        "runtime_result_accepted": False,
        "gpu_runtime_qualified": False,
        "deepstream_runtime_qualified": False,
        "production_authorized": False,
        "user_notification_required_before_any_future_gpu_workload": True,
        "independent_runtime_result_review_required": True,
    }
    jsonschema.Draft202012Validator(schema).validate(authority)
    for key in ("outer_acceptance", "launch_authorized", "execution_authorized", "runtime_result_accepted", "production_authorized"):
        mutated = copy.deepcopy(authority)
        mutated[key] = True
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(mutated)


def test_trtexec_layer_pin_and_member_paths() -> None:
    target = ROOT / review.NATIVE_PINS["trtexec_layer"]["path"]
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    assert size == review.NATIVE_PINS["trtexec_layer"]["bytes"]
    assert digest.hexdigest() == review.NATIVE_PINS["trtexec_layer"]["sha256"]
