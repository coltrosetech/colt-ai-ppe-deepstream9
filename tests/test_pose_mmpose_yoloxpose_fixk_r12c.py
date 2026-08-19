from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_fixk_r12c as r12c
from validation import pose_mmpose_yoloxpose_fixk_r12c_container as container


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "validation/results/pose/models/mmpose-yoloxpose-s-fixed-k100-plan-r12c.json"
EXPORT_PYTHON = ROOT / ".venv-export/bin/python"


def _write(path: Path, raw: bytes, mode: int = 0o440) -> None:
    path.write_bytes(raw)
    path.chmod(mode)


def _make_writable(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    if path.is_dir():
        path.chmod(0o700)
        for child in path.iterdir():
            _make_writable(child)
    else:
        path.chmod(0o600)


def test_r12_and_failed_r12b_inputs_remain_byte_exact_and_r12c_contract_is_plan_only() -> None:
    assert r12c.sha256_file(Path(r12c.r12.THIS_FILE)) == r12c.R12_HOST_SHA256
    assert r12c.sha256_file(r12c.r12.CONTAINER_RUNNER) == r12c.R12_CONTAINER_SHA256
    assert r12c.sha256_file(r12c.R12_PLAN) == r12c.R12_PLAN_FILE_SHA256
    assert r12c.sha256_file(r12c.R12_CONTRACT) == r12c.R12_CONTRACT_SHA256
    assert r12c.sha256_file(Path(r12c.r12b.THIS_FILE)) == r12c.R12B_HOST_SHA256
    assert r12c.sha256_file(r12c.r12b.CONTAINER_RUNNER) == r12c.R12B_CONTAINER_SHA256
    assert r12c.sha256_file(r12c.r12b.HOST_CHECKER) == r12c.R12B_HOST_CHECKER_SHA256
    assert r12c.sha256_file(r12c.r12b.CONTRACT) == r12c.R12B_CONTRACT_SHA256
    assert r12c.sha256_file(r12c.r12b.PROFILE_RECEIPT_SCHEMA) == r12c.R12B_PROFILE_SCHEMA_SHA256
    assert r12c.sha256_file(r12c.r12b.PLAN_DEFAULT) == r12c.R12B_PLAN_FILE_SHA256
    assert r12c.sha256_file(r12c.R12B_FAILURE_MANIFEST) == r12c.R12B_FAILURE_MANIFEST_SHA256
    contract = r12c.strict_json(r12c.CONTRACT)
    r12c.validate_contract(contract)
    assert contract["claims"]["r12c_execution_performed"] is False
    assert contract["claims"]["root_audit_completed"] is False
    assert contract["claims"]["production_ready"] is False


def test_r12c_schemas_and_rebuilt_plan_are_exact_and_gpu_free() -> None:
    for path in (
        r12c.CONTRACT_SCHEMA,
        r12c.PLAN_SCHEMA,
        r12c.PROFILE_RECEIPT_SCHEMA,
        r12c.HOST_CHECK_SCHEMA,
        r12c.RECEIPT_SCHEMA,
        r12c.FAILED_ATTEMPT_SCHEMA,
    ):
        jsonschema.Draft202012Validator.check_schema(r12c.strict_json(path))
    plan = r12c.build_plan(created_at="2026-07-18T00:00:00+00:00")
    jsonschema.Draft202012Validator(r12c.strict_json(r12c.PLAN_SCHEMA)).validate(plan)
    assert len(plan["source_boundary_sha256"]) == 18
    assert len(plan["source_symbol_attestations"]) == 14
    assert len(plan["host_runtime"]["packages"]) == 6
    assert len(plan["host_runtime"]["native_libraries"]) == 10
    assert len(plan["inputs"]) == 62
    assert plan["failed_attempt_recovery"] == r12c.strict_json(r12c.CONTRACT)["failed_attempt_recovery"]
    assert plan["failed_attempt_recovery"]["r12b_schema_actual_properties"] == 23
    assert plan["failed_attempt_recovery"]["r12b_quarantine_reused"] is False
    assert plan["failed_attempt_recovery"]["root_audit_required_before_execute"] is True
    assert plan["host_runtime"]["pinning_assessment"] == {
        "safe_within_r10_record_closure_threat_model": True,
        "new_environment_required": False,
        "fully_os_hermetic": False,
        "system_native_map_exact": True,
    }
    assert all(plan["execution_boundary"][key] is False for key in (
        "docker_queried", "container_run", "model_loaded", "onnx_exported",
        "onnx_transformed", "onnxruntime_executed", "network_used",
        "gpu_exposed", "gpu_api_queried", "tensorrt_executed",
        "deepstream_executed", "production_promoted",
    ))


def test_r12b_failure_manifest_and_quarantine_replay_are_exact() -> None:
    state = r12c.hold_r12b_failed_attempt()
    try:
        state.replay()
        assert len(state.manifest["frozen_r12b"]["pins"]) == 12
        assert len(state.manifest["profile_640_worker"]["artifacts"]) == 6
        assert state.manifest["profile_640_worker"]["worker_status"] == "passed"
        assert state.manifest["profile_960"]["attempted"] is False
        assert state.manifest["quarantine"]["canonical_destination"]["present"] is False
    finally:
        state.close()


def test_historical_r12b_receipt_fails_only_the_frozen_min_properties_typo() -> None:
    receipt_path = r12c.R12B_FAILED_STAGE / "640/profile-receipt-r12b.json"
    value = r12c.strict_json(receipt_path)
    original = r12c.strict_json(r12c.r12b.PROFILE_RECEIPT_SCHEMA)
    with pytest.raises(jsonschema.ValidationError) as caught:
        jsonschema.Draft202012Validator(original).validate(value)
    assert caught.value.validator == "minProperties"
    assert caught.value.validator_value == 25
    assert list(caught.value.absolute_path) == ["execution_boundary"]
    assert len(value["execution_boundary"]) == 23
    r12c._validate_historical_r12b_receipt(value)


def test_r12c_execution_boundary_requires_exact_23_and_rejects_missing_or_extra() -> None:
    full_schema = r12c.strict_json(r12c.PROFILE_RECEIPT_SCHEMA)
    boundary_schema = copy.deepcopy(full_schema["properties"]["execution_boundary"])
    wrapped = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": copy.deepcopy(full_schema["$defs"]),
        **boundary_schema,
    }
    historical = r12c.strict_json(
        r12c.R12B_FAILED_STAGE / "640/profile-receipt-r12b.json"
    )["execution_boundary"]
    valid = copy.deepcopy(historical)
    valid["runtime"] = "cpu_only_exact_image_fixed_k100_correction_r12c"
    validator = jsonschema.Draft202012Validator(wrapped)
    validator.validate(valid)
    assert len(boundary_schema["required"]) == 23
    assert len(boundary_schema["properties"]) == 23
    assert boundary_schema["minProperties"] == boundary_schema["maxProperties"] == 23
    assert boundary_schema["additionalProperties"] is False
    for key in boundary_schema["required"]:
        missing = copy.deepcopy(valid)
        del missing[key]
        assert not validator.is_valid(missing), key
    extra = copy.deepcopy(valid)
    extra["unattested"] = False
    assert not validator.is_valid(extra)

    candidate = r12c.strict_json(
        r12c.R12B_FAILED_STAGE / "640/profile-receipt-r12b.json"
    )
    candidate["schema_version"] = r12c.PROFILE_SCHEMA_VERSION
    candidate["run_id"] = "pose-r12c-schema-fixture"
    candidate["failed_attempt_recovery"] = copy.deepcopy(r12c.FAILED_ATTEMPT_RECOVERY)
    candidate["execution_boundary"] = valid
    for key in ("raw", "corrected", "real_seed_b12"):
        candidate["publication"][key]["path"] = candidate["publication"][key]["path"].replace(
            "r12b", "r12c"
        )
    candidate["receipt_fingerprint_sha256"] = r12c.fingerprint(
        candidate, "receipt_fingerprint_sha256"
    )
    jsonschema.Draft202012Validator(full_schema).validate(candidate)


def test_r12c_execute_rejects_failed_run_id_and_wrong_recovery_acceptance_before_plan_open() -> None:
    with pytest.raises(r12c.FixedKR12CError, match="must differ"):
        r12c.execute(Path("missing"), "a" * 64, "b" * 64, r12c.R12B_FAILED_RUN_ID, "c" * 64)
    with pytest.raises(r12c.FixedKR12CError, match="recovery acceptance differs"):
        r12c.execute(Path("missing"), "a" * 64, "b" * 64, "pose-r12c-cpu-export-001", "c" * 64)


def test_bootstrap_pins_and_privately_rebinds_all_five_mounted_inputs() -> None:
    source = r12c.R12C_BOOTSTRAP_SOURCE
    compile(source, "<r12c-bootstrap-test>", "exec")
    for label in ("runner", "checkpoint", "seed", "r12_base", "r11_onnx"):
        assert label in source
    for flag in ("--checkpoint", "--seed-image", "--r12-base", "--r11-onnx"):
        assert flag in source
    assert "replay_source" in source
    command = r12c.docker_command(
        640,
        "/tmp/output",
        "a" * 64,
        "run-r12c",
        "1000:1000",
        1,
        2,
        {"bytes": 1, "sha256": "1" * 64},
        {"bytes": 2, "sha256": "2" * 64},
        {"bytes": 3, "sha256": "3" * 64},
        {"bytes": 4, "sha256": "4" * 64},
        {"bytes": 5, "sha256": "5" * 64},
    )
    for mounted in (
        "/opt/deepsafe/r12c-runner.py", "/opt/deepsafe/checkpoint-r12c.pth",
        "/opt/deepsafe/seed-r12c.jpg", "/opt/deepsafe/r12-base.py",
        "/opt/deepsafe/r11-raw-640.onnx",
    ):
        assert mounted in command


def test_r12c_source_maps_have_exact_unique_counts_and_no_literal_duplicate_keys() -> None:
    assert len(container.SOURCE_PINS) == 18
    assert len(container.SYMBOL_ATTESTATIONS) == 14
    assert len(r12c.CONTAINER_SOURCE_HASHES) == 18
    assert len(r12c.SYMBOL_ATTESTATIONS) == 14
    for path in (r12c.THIS_FILE, r12c.CONTAINER_RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
            assert len(keys) == len(set(keys)), f"duplicate literal dict key at {path}:{node.lineno}"
    container_source = r12c.CONTAINER_RUNNER.read_text(encoding="utf-8")
    assert "base.publish_file(" not in container_source
    assert "_publish_file_held(" in container_source


def test_container_publication_holds_the_same_anonymous_inode_and_rejects_swap(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source.onnx"
    source.write_bytes(b"actual-private-model")
    directory_fd = os.open(output, container._directory_flags())
    held = None
    try:
        held, pin = container._publish_file_held(
            source, directory_fd, "model.onnx", maximum_bytes=1024
        )
        assert pin["sha256"] == hashlib.sha256(b"actual-private-model").hexdigest()
        assert os.fstat(held.fd).st_ino == (output / "model.onnx").stat().st_ino
        os.rename("model.onnx", "model.held", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        _write(output / "model.onnx", b"actual-private-model")
        with pytest.raises(container.FixedKR12CContainerError, match="inode changed"):
            container._replay_held(directory_fd, held, 1024)
        assert os.fstat(held.fd).st_ino == (output / "model.held").stat().st_ino
    finally:
        if held is not None:
            os.close(held.fd)
        os.close(directory_fd)


@pytest.mark.parametrize("name", ["receipt.json", "raw.onnx", "corrected.onnx"])
def test_held_leaf_rejects_byte_identical_name_replacement(tmp_path: Path, name: str) -> None:
    _write(tmp_path / name, b"same-bytes")
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    held = r12c.open_regular_at(parent_fd, name, maximum_bytes=1024)
    try:
        os.rename(name, name + ".old", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        _write(tmp_path / name, b"same-bytes")
        with pytest.raises(r12c.FixedKR12CError, match="inode differs"):
            r12c.replay_regular(parent_fd, held)
        assert os.fstat(held.fd).st_ino == (tmp_path / (name + ".old")).stat().st_ino
    finally:
        os.close(held.fd)
        os.close(parent_fd)


def test_profile_directory_name_replacement_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "640").mkdir()
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    profile_fd = os.open("640", r12c._directory_flags(), dir_fd=parent_fd)
    try:
        os.rename("640", "640.old", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        (tmp_path / "640").mkdir()
        with pytest.raises(r12c.FixedKR12CError, match="directory entry changed"):
            r12c.assert_directory_name_binds_fd(parent_fd, "640", profile_fd, "profile")
    finally:
        os.close(profile_fd)
        os.close(parent_fd)


def test_leaf_policy_rejects_symlink_fifo_and_hardlink_without_blocking(tmp_path: Path) -> None:
    _write(tmp_path / "target", b"x")
    (tmp_path / "link").symlink_to("target")
    os.mkfifo(tmp_path / "fifo", 0o440)
    os.link(tmp_path / "target", tmp_path / "hard")
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    try:
        with pytest.raises(OSError):
            r12c.open_regular_at(parent_fd, "link", maximum_bytes=10)
        with pytest.raises(r12c.FixedKR12CError, match="not regular"):
            r12c.open_regular_at(parent_fd, "fifo", maximum_bytes=10)
        with pytest.raises(r12c.FixedKR12CError, match="link count"):
            r12c.open_regular_at(parent_fd, "hard", maximum_bytes=10)
    finally:
        os.close(parent_fd)


def test_same_size_in_place_mutation_is_rejected_and_fd_lifecycle_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "leaf"
    path.write_bytes(b"abcdefgh")
    writer = os.open(path, os.O_RDWR)
    path.chmod(0o440)
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    held = r12c.open_regular_at(parent_fd, "leaf", maximum_bytes=100)
    descriptor = held.fd
    try:
        os.pwrite(writer, b"ABCDEFGH", 0)
        os.fsync(writer)
        with pytest.raises(r12c.FixedKR12CError, match="differ|changed"):
            r12c.replay_regular(parent_fd, held)
    finally:
        os.close(writer)
        os.close(held.fd)
        os.close(parent_fd)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_accepted_plan_inputs_stay_held_and_reject_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r12c, "ROOT", tmp_path)
    _write(tmp_path / "input.bin", b"accepted-input", mode=0o440)
    plan = {
        "inputs": {
            "input": {
                "path": "input.bin",
                "bytes": len(b"accepted-input"),
                "sha256": hashlib.sha256(b"accepted-input").hexdigest(),
            }
        }
    }
    held = r12c.hold_plan_inputs(plan)
    try:
        r12c.replay_plan_inputs(held)
        os.rename(tmp_path / "input.bin", tmp_path / "input.old")
        _write(tmp_path / "input.bin", b"accepted-input", mode=0o440)
        with pytest.raises(r12c.FixedKR12CError, match="inode differs"):
            r12c.replay_plan_inputs(held)
        assert os.fstat(held[0].file.fd).st_ino == (tmp_path / "input.old").stat().st_ino
    finally:
        for item in held:
            item.close()


@pytest.mark.parametrize("failure_point", ["stage", "640", "960", "freeze", "receipt", "fsync"])
def test_precommit_failure_points_leave_canonical_name_absent(tmp_path: Path, failure_point: str) -> None:
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    stage_fd = -1
    try:
        stage_name, stage_fd, _identity = r12c.allocate_stage_at(parent_fd, "atomic-run")
        if failure_point in {"640", "960", "freeze", "receipt", "fsync"}:
            _write(tmp_path / stage_name / "evidence", failure_point.encode())
        if failure_point in {"freeze", "receipt", "fsync"}:
            os.fchmod(stage_fd, 0o550)
        if failure_point == "fsync":
            os.fsync(stage_fd)
        assert not (tmp_path / "atomic-run").exists()
        assert (tmp_path / stage_name).is_dir()
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)
        os.close(parent_fd)
        for child in tmp_path.iterdir():
            _make_writable(child)


def test_640_failure_blocks_960_before_any_second_attempt() -> None:
    results = {"640": {"status": "failed"}, "960": {"status": "not_attempted"}}
    with pytest.raises(r12c.FixedKR12CError, match="blocked"):
        r12c.assert_profile_start_allowed(960, results)
    assert results["960"] == {"status": "not_attempted"}


@pytest.mark.parametrize("destination_kind", ["file", "directory", "symlink", "dangling"])
def test_rename_noreplace_refuses_every_existing_destination(tmp_path: Path, destination_kind: str) -> None:
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    stage_name, stage_fd, stage_identity = r12c.allocate_stage_at(parent_fd, "commit")
    try:
        target = tmp_path / "commit"
        if destination_kind == "file":
            target.write_bytes(b"owner")
        elif destination_kind == "directory":
            target.mkdir()
        elif destination_kind == "symlink":
            (tmp_path / "owner").write_bytes(b"owner")
            target.symlink_to("owner")
        else:
            target.symlink_to("missing")
        with pytest.raises(r12c.FixedKR12CError, match="already exists"):
            r12c.rename_noreplace_at(parent_fd, stage_name, "commit")
        assert (os.fstat(stage_fd).st_dev, os.fstat(stage_fd).st_ino) == stage_identity
        assert os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False).st_ino == stage_identity[1]
    finally:
        os.close(stage_fd)
        os.close(parent_fd)


def test_atomic_rename_first_visibility_is_complete_and_inode_bound(tmp_path: Path) -> None:
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    stage_name, stage_fd, identity = r12c.allocate_stage_at(parent_fd, "visible")
    try:
        _write(tmp_path / stage_name / "a", b"a")
        _write(tmp_path / stage_name / "b", b"b")
        os.fchmod(stage_fd, 0o550)
        os.fsync(stage_fd)
        assert not (tmp_path / "visible").exists()
        r12c.rename_noreplace_at(parent_fd, stage_name, "visible")
        destination_fd = os.open("visible", r12c._directory_flags(), dir_fd=parent_fd)
        try:
            assert r12c._identity(os.fstat(destination_fd)) == identity
            assert sorted(os.listdir(destination_fd)) == ["a", "b"]
        finally:
            os.close(destination_fd)
        assert not (tmp_path / stage_name).exists()
    finally:
        os.close(stage_fd)
        os.close(parent_fd)
        _make_writable(tmp_path / "visible")


def test_two_concurrent_publishers_have_exactly_one_winner(tmp_path: Path) -> None:
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    stages = [r12c.allocate_stage_at(parent_fd, "race") for _ in range(2)]
    outcomes: list[str] = []

    def publish(name: str) -> None:
        try:
            r12c.rename_noreplace_at(parent_fd, name, "race")
            outcomes.append("won")
        except r12c.FixedKR12CError:
            outcomes.append("lost")

    threads = [threading.Thread(target=publish, args=(item[0],)) for item in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert sorted(outcomes) == ["lost", "won"]
    finally:
        for _name, descriptor, _identity in stages:
            os.close(descriptor)
        os.close(parent_fd)


def test_stage_name_substitution_cannot_be_accepted_by_held_inode(tmp_path: Path) -> None:
    parent_fd = os.open(tmp_path, r12c._directory_flags())
    stage_name, stage_fd, held_identity = r12c.allocate_stage_at(parent_fd, "substitute")
    try:
        os.rename(stage_name, stage_name + ".held", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        r12c.rename_noreplace_at(parent_fd, stage_name, "substitute")
        destination_fd = os.open("substitute", r12c._directory_flags(), dir_fd=parent_fd)
        try:
            assert r12c._identity(os.fstat(destination_fd)) != held_identity
        finally:
            os.close(destination_fd)
        with pytest.raises(r12c.FixedKR12CError, match="directory entry changed"):
            r12c.assert_directory_name_binds_fd(parent_fd, "substitute", stage_fd, "post-rename")
    finally:
        os.close(stage_fd)
        os.close(parent_fd)


def test_bounded_subprocess_rejects_huge_output_and_timeout_without_orphan() -> None:
    with pytest.raises(r12c.FixedKR12CError, match="output exceeds"):
        r12c.run_bounded_process(
            [str(r12c.HOST_PYTHON), "-c", "import os; os.write(1, b'x' * 200000)"],
            timeout_seconds=5,
            output_limit=1024,
        )
    with pytest.raises(r12c.FixedKR12CError, match="exceeded"):
        r12c.run_bounded_process(
            [str(r12c.HOST_PYTHON), "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.05,
            output_limit=1024,
        )


def test_verify_run_rejects_an_alternate_self_consistent_plan_path(tmp_path: Path) -> None:
    alternate = r12c.build_plan(created_at="2026-07-18T00:00:01+00:00")
    path = tmp_path / "alternate.json"
    _write(path, (json.dumps(alternate, sort_keys=True) + "\n").encode())
    with pytest.raises(r12c.FixedKR12CError, match="canonical superseding"):
        r12c.verify_run(
            tmp_path / "none/run-receipt-r12c.json",
            "a" * 64,
            plan_path=path,
            expected_plan_fingerprint=alternate["plan_fingerprint_sha256"],
            expected_plan_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


HOSTCHECK_SYNTHETIC_SCRIPT = r'''
import copy
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from validation import pose_mmpose_yoloxpose_fixk_r12c_hostcheck as h

def vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

def raw_model(profile):
    nms = helper.make_node("NonMaxSuppression", ["boxes", "scores", "max", "iou", "threshold"], ["selected"], name="nms", domain="")
    graph = helper.make_graph([nms], "raw", [vi("input", ["batch", 3, profile, profile])], [vi("dets", ["batch", "k", 5]), vi("keypoints", ["batch", "k", 17, 3])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=7)

def initializer(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=h._w(name))

def corrected_model(raw, profile):
    initializers = [
        initializer("batch_axis_index", [0], np.int64),
        initializer("det_zero_tail", [100, 5], np.int64),
        initializer("kpt_zero_tail", [100, 17, 3], np.int64),
        initializer("score_axis_index", np.asarray(4, dtype=np.int64), np.int64),
        initializer("topk_k", [100], np.int64),
        initializer("det_gather_tail", [5], np.int64),
        initializer("kpt_gather_tail", [17, 3], np.int64),
        initializer("score_threshold", np.asarray(0.01, dtype=np.float32), np.float32),
    ]
    nodes = [copy.deepcopy(node) for node in raw.graph.node]
    for short in h.WRAPPER_NODE_ORDER:
        op, inputs, outputs, attrs = h.WRAPPER_NODE_SPECS[short]
        kwargs = {}
        for key, value in attrs.items():
            if value == "<POSITIVE_FLOAT32_ZERO_TENSOR>":
                kwargs[key] = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name=h._w("zero_value"))
            else:
                kwargs[key] = value
        nodes.append(helper.make_node(op, list(inputs), list(outputs), name=h._w(short), domain="", **kwargs))
    graph = helper.make_graph(nodes, "corrected", [vi("input", ["batch", 3, profile, profile])], [vi("dets", ["batch", 100, 5]), vi("keypoints", ["batch", 100, 17, 3])], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=raw.ir_version)
    for key, value in {
        "deepsafe.r12.wrapper": "fixed-k100-score-mask",
        "deepsafe.r12.score_predicate": "score>0.01",
        "deepsafe.r12.invalid_rows": "exact-zero",
        "deepsafe.r12.profile": str(profile),
    }.items():
        item = model.metadata_props.add(); item.key = key; item.value = value
    return model

raw = raw_model(640)
valid = corrected_model(raw, 640)
result = h.validate_model_pair(raw, valid, profile=640, onnx_module=onnx, numpy_module=np, run_onnx_checker=False)
assert result["base_graph_cross_bind"]["wrapper_nodes_appended"] == 24
assert result["corrected"]["wrapper_initializers"]["count"] == 8

mutations = {}
m = copy.deepcopy(valid); del m.graph.node[-2:]; mutations["22-node"] = m
m = copy.deepcopy(valid); next(n for n in m.graph.node if n.name == h._w("UnsqueezeKptIndices")).input[0] = h._w("split_indices"); mutations["split-gather"] = m
m = copy.deepcopy(valid); next(n for n in m.graph.node if n.name == h._w("ScoreStrictlyGreaterThanThreshold")).op_type = "GreaterOrEqual"; mutations["greater-equal"] = m
m = copy.deepcopy(valid)
for index, item in enumerate(m.graph.initializer):
    if item.name == h._w("topk_k"):
        m.graph.initializer[index].CopyFrom(numpy_helper.from_array(np.asarray([99], dtype=np.int64), name=h._w("topk_k")))
mutations["k99"] = m
m = copy.deepcopy(valid)
node = next(n for n in m.graph.node if n.name == h._w("DetZeros"))
node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.asarray([-0.0], dtype=np.float32), name=h._w("zero_value")))
mutations["negative-zero"] = m

for name, model in mutations.items():
    try:
        h.validate_model_pair(raw, model, profile=640, onnx_module=onnx, numpy_module=np, run_onnx_checker=False)
    except h.HostR12CCheckError:
        pass
    else:
        raise AssertionError(name + " unexpectedly passed")

nested = copy.deepcopy(valid)
inner = helper.make_graph([helper.make_node("Hidden", [], ["hidden"], domain="evil.nested")], "inner", [], [])
nested.graph.node.append(helper.make_node("If", ["cond"], ["unused"], name="nested-holder", then_branch=inner, else_branch=copy.deepcopy(inner)))
try:
    h.validate_model_proto(nested, profile=640, corrected=True, onnx_module=onnx, numpy_module=np, run_onnx_checker=False)
except h.HostR12CCheckError:
    pass
else:
    raise AssertionError("nested custom domain unexpectedly passed")
print("synthetic-r12c-hostcheck-ok")
'''


def test_independent_checker_rejects_all_required_structural_mutations() -> None:
    completed = subprocess.run(
        [str(EXPORT_PYTHON), "-B", "-c", HOSTCHECK_SYNTHETIC_SCRIPT],
        cwd=ROOT,
        env={"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "PYTHONPATH": str(ROOT)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "synthetic-r12c-hostcheck-ok"


def test_frozen_plan_mode_hash_and_verify_replay() -> None:
    assert PLAN.is_file()
    assert stat.S_IMODE(PLAN.stat().st_mode) == 0o440
    raw = PLAN.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    plan, held, parent_fd = r12c.open_frozen_plan(
        PLAN, value["plan_fingerprint_sha256"], file_sha
    )
    try:
        assert plan == value
        assert held.sha256 == file_sha
    finally:
        r12c.close_frozen_plan(held, parent_fd)
