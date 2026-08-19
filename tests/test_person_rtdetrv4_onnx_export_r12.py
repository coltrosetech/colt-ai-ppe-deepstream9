from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from validation import person_rtdetrv4_onnx_export_r12 as lane


@pytest.fixture()
def schema() -> dict:
    return json.loads(lane.EVIDENCE_SCHEMA.read_text(encoding="utf-8"))


@pytest.fixture()
def fake_accepted(schema: dict):
    return SimpleNamespace(
        schema=schema,
        plan={
            "training_lineage": {
                "best_checkpoint": {
                    "path": lane.CHECKPOINT_RELATIVE,
                    "bytes": lane.CHECKPOINT_BYTES,
                    "sha256": lane.CHECKPOINT_SHA256,
                }
            }
        },
        plan_pin={
            "path": lane.PLAN_RELATIVE,
            "bytes": lane.PLAN_BYTES,
            "sha256": lane.PLAN_FILE_SHA256,
            "fingerprint_sha256": lane.PLAN_FINGERPRINT,
        },
        plan_fd=31,
        schema_fd=32,
        checkpoint_fd=33,
        config_fd=34,
        executor_fd=35,
        runtime_python_fd=36,
        executor_sha256="a" * 64,
    )


def legacy_raw_bindings(profile: int = 640) -> list[dict]:
    observed_bindings = lane.expected_bindings(profile)
    observed_bindings[2]["shape"] = ["batch", "Sublabels_dim_1"]
    observed_bindings[3]["shape"] = [
        "batch",
        "GatherElementsboxes_dim_1",
        "batch",
    ]
    observed_bindings[4]["shape"] = ["batch", "GatherElementsboxes_dim_1"]
    return observed_bindings


def worker_result(profile: int = 640) -> dict:
    observed_bindings = legacy_raw_bindings(profile)
    changes = lane._validate_output_metadata_repair_candidate(
        observed_bindings,
        profile=profile,
        observed_opsets={"ai.onnx": 18},
    )
    return {
        "status": "passed",
        "profile": profile,
        "runtime": copy.deepcopy(lane.EXPECTED_RUNTIME),
        "gpu_device_nodes": [],
        "cpu_only_torch_wheel": True,
        "cuda_visible_devices": "",
        "network_interfaces": ["docker0", "lo", "wlan0"],
        "network_syscalls_seccomp_denied": True,
        "checkpoint_payload": "ema.module",
        "strict_load": True,
        "num_classes": 1,
        "export_trace_batch": 2,
        "output_metadata_repair": {
            "applied": True,
            "scope": "graph.output.type.tensor_type.shape_only",
            "observed_bindings": observed_bindings,
            "repaired_bindings": lane.expected_bindings(profile),
            "opsets": {"ai.onnx": 18},
            "changes": changes,
            "model_sha256_excluding_graph_output_shapes_before": "c" * 64,
            "model_sha256_excluding_graph_output_shapes_after": "c" * 64,
            "runtime_shape_proof_batches": [1, 12],
            "checker_reloaded_after_repair": True,
        },
        "opset": 18,
        "checker_passed": True,
        "external_data": False,
        "bindings": lane.expected_bindings(profile),
        "batch12_shape_finite": True,
        "learned_parameters_unchanged": True,
        "framework_onnx_parity": {
            "batches": [1, 12],
            "labels_class_exact": True,
            "boxes_max_abs": 0.001,
            "scores_max_abs": 0.00001,
            "passed": True,
        },
    }


def onnx_pin(profile: int = 640) -> dict:
    return {
        "path": lane.ONNX_RELATIVE[profile],
        "bytes": 123,
        "sha256": "b" * 64,
    }


def make_receipt(fake_accepted, profile: int = 640, prior=None) -> dict:
    return lane.build_stage_receipt(
        accepted=fake_accepted,
        profile=profile,
        onnx_pin=onnx_pin(profile),
        worker_result=worker_result(profile),
        prior_receipts=[] if prior is None else prior,
        created_at_utc="2026-07-18T08:00:00+00:00",
    )


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    assert lane.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(lane.PersonOnnxR12Error, match="non-finite"):
        lane.canonical_bytes({"bad": math.nan})


def test_strict_json_rejects_duplicate_keys_and_nan() -> None:
    with pytest.raises(lane.PersonOnnxR12Error, match="duplicate JSON key"):
        lane._strict_json_bytes(b'{"a":1,"a":2}', source="unit")
    with pytest.raises(lane.PersonOnnxR12Error, match="non-finite"):
        lane._strict_json_bytes(b'{"a":NaN}', source="unit")


def test_safe_relative_path_is_anchored_to_repo() -> None:
    relative, absolute = lane._normalize_repo_path("validation/example.json")
    assert relative == "validation/example.json"
    assert absolute == lane.ROOT / "validation/example.json"


@pytest.mark.parametrize(
    "value",
    ["../escape", "a/../../escape", "validation/../models/x", "/tmp/outside"],
)
def test_path_traversal_and_outside_absolute_are_rejected(value: str) -> None:
    with pytest.raises(
        lane.PersonOnnxR12Error, match="outside repository|unsafe repository"
    ):
        lane._normalize_repo_path(value)


def test_no_follow_open_rejects_leaf_symlink(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "real.bin").write_bytes(b"ok")
    (tmp_path / "link.bin").symlink_to("real.bin")
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with pytest.raises(lane.PersonOnnxR12Error, match="non-symlink"):
        lane._open_repo_file("link.bin")


def test_no_follow_open_rejects_parent_symlink(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "x").write_bytes(b"ok")
    (tmp_path / "alias").symlink_to("real", target_is_directory=True)
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with pytest.raises(lane.PersonOnnxR12Error, match="non-symlink"):
        lane._open_repo_file("alias/x")


def test_output_parent_rejects_symlink_component(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to("real", target_is_directory=True)
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    with pytest.raises(lane.PersonOnnxR12Error, match="safe output parent"):
        lane._open_or_create_parent("alias/output.onnx")


def test_atomic_publication_is_no_overwrite(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, lane._directory_flags())
    try:
        lane._atomic_bytes_at(directory_fd, "receipt.json", b"{}\n")
        assert (tmp_path / "receipt.json").read_bytes() == b"{}\n"
        assert (tmp_path / "receipt.json").stat().st_mode & 0o777 == 0o440
        with pytest.raises(lane.PersonOnnxR12Error, match="overwrite"):
            lane._atomic_bytes_at(directory_fd, "receipt.json", b"changed")
    finally:
        os.close(directory_fd)


def test_unsafe_atomic_leaf_name_is_rejected(tmp_path: Path) -> None:
    directory_fd = os.open(tmp_path, lane._directory_flags())
    try:
        with pytest.raises(lane.PersonOnnxR12Error, match="unsafe output"):
            lane._name_absent(directory_fd, "../bad")
    finally:
        os.close(directory_fd)


def test_expected_binding_contract_is_fixed_spatial_dynamic_batch() -> None:
    bindings = lane.expected_bindings(960)
    assert bindings[0] == {
        "name": "images",
        "io": "input",
        "dtype": "FLOAT32",
        "shape": ["batch", 3, 960, 960],
    }
    assert bindings[1]["shape"] == ["batch", 2]
    assert [item["name"] for item in bindings] == [
        "images",
        "orig_target_sizes",
        "labels",
        "boxes",
        "scores",
    ]


def test_legacy_symbolic_output_metadata_has_exact_bounded_repair_plan() -> None:
    observed = legacy_raw_bindings(640)
    changes = lane._validate_output_metadata_repair_candidate(
        observed,
        profile=640,
        observed_opsets={"ai.onnx": 18},
    )
    assert changes == [
        {
            "name": "labels",
            "axis": 1,
            "observed": "Sublabels_dim_1",
            "repaired": 300,
        },
        {
            "name": "boxes",
            "axis": 1,
            "observed": "GatherElementsboxes_dim_1",
            "repaired": 300,
        },
        {
            "name": "boxes",
            "axis": 2,
            "observed": "batch",
            "repaired": 4,
        },
        {
            "name": "scores",
            "axis": 1,
            "observed": "GatherElementsboxes_dim_1",
            "repaired": 300,
        },
    ]


def test_metadata_repair_rejects_contradictory_numeric_tail_with_diagnostic() -> None:
    observed = legacy_raw_bindings(640)
    observed[3]["shape"][2] = 5
    with pytest.raises(lane.PersonOnnxR12Error) as captured:
        lane._validate_output_metadata_repair_candidate(
            observed,
            profile=640,
            observed_opsets={"ai.onnx": 18},
        )
    message = str(captured.value)
    assert '"observed"' in message
    assert '"expected"' in message
    assert '"opsets":{"ai.onnx":18}' in message
    assert '"shape":["batch",300,4]' in message
    assert '"shape":["batch","GatherElementsboxes_dim_1",5]' in message


def test_valid_640_receipt_uses_exact_r11_schema(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    assert receipt["stage"] == "onnx_export_640"
    assert receipt["prior_receipts"] == []
    assert receipt["payload"]["framework_onnx_parity"]["batches"] == [1, 12]
    assert receipt["fingerprint_sha256"] == lane.fingerprint(receipt)


def test_resealed_gpu_overclaim_is_rejected(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    receipt["execution"]["gpu"] = True
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="overclaims"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=640,
            expected_prior_pins=[],
        )


def test_resealed_failed_status_is_rejected(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    receipt["status"] = "failed"
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="only passed"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=640,
            expected_prior_pins=[],
        )


def test_resealed_onnx_path_escape_is_rejected(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    receipt["payload"]["onnx"]["path"] = "../../malicious.onnx"
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="ONNX path"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=640,
            expected_prior_pins=[],
        )


def test_resealed_wrong_checkpoint_payload_is_rejected(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    receipt["payload"]["checkpoint_payload"] = "model"
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="schema mismatch"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=640,
            expected_prior_pins=[],
        )


def test_resealed_binding_overclaim_is_rejected(fake_accepted) -> None:
    receipt = make_receipt(fake_accepted)
    receipt["payload"]["bindings"][0]["shape"] = ["batch", 3, "height", "width"]
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="bindings"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=640,
            expected_prior_pins=[],
        )


def test_box_and_score_tolerance_overclaims_are_rejected(fake_accepted) -> None:
    for field, value in (("boxes_max_abs", 0.02001), ("scores_max_abs", 0.000201)):
        receipt = make_receipt(fake_accepted)
        receipt["payload"]["framework_onnx_parity"][field] = value
        receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
        with pytest.raises(lane.PersonOnnxR12Error, match="schema mismatch"):
            lane._validate_receipt_semantics(
                receipt,
                accepted=fake_accepted,
                profile=640,
                expected_prior_pins=[],
            )


def test_640_rejects_any_prior_stage_pin(fake_accepted) -> None:
    prior = {
        "path": lane.RECEIPT_RELATIVE[640],
        "bytes": 100,
        "sha256": "c" * 64,
        "fingerprint_sha256": "d" * 64,
    }
    with pytest.raises(lane.PersonOnnxR12Error, match="prior receipt count"):
        make_receipt(fake_accepted, profile=640, prior=[prior])


def test_960_requires_one_exact_prior_stage_pin(fake_accepted) -> None:
    with pytest.raises(lane.PersonOnnxR12Error, match="prior receipt count"):
        make_receipt(fake_accepted, profile=960, prior=[])
    prior = {
        "path": lane.RECEIPT_RELATIVE[640],
        "bytes": 100,
        "sha256": "c" * 64,
        "fingerprint_sha256": "d" * 64,
    }
    receipt = make_receipt(fake_accepted, profile=960, prior=[prior])
    assert receipt["prior_receipts"] == [prior]


def test_960_resealed_prior_order_or_pin_is_rejected(fake_accepted) -> None:
    prior = {
        "path": lane.RECEIPT_RELATIVE[640],
        "bytes": 100,
        "sha256": "c" * 64,
        "fingerprint_sha256": "d" * 64,
    }
    receipt = make_receipt(fake_accepted, profile=960, prior=[prior])
    changed = copy.deepcopy(prior)
    changed["fingerprint_sha256"] = "e" * 64
    receipt["prior_receipts"] = [changed]
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.PersonOnnxR12Error, match="prior-stage"):
        lane._validate_receipt_semantics(
            receipt,
            accepted=fake_accepted,
            profile=960,
            expected_prior_pins=[prior],
        )


def test_plan_reseal_cannot_redirect_fixed_output() -> None:
    plan = json.loads(lane.PLAN.read_text(encoding="utf-8"))
    plan["profiles"]["640"]["onnx_path"] = "models/person/redirect.onnx"
    plan["fingerprint_sha256"] = lane.fingerprint(plan)
    with pytest.raises(lane.PersonOnnxR12Error, match="plan fingerprint"):
        lane._validate_plan_semantics(plan)


def test_plan_semantics_accept_exact_immutable_plan() -> None:
    lane._validate_plan_semantics(json.loads(lane.PLAN.read_text(encoding="utf-8")))


def test_worker_result_rejects_wrong_runtime_and_batch12() -> None:
    result = worker_result()
    result["runtime"]["torch"] = "2.13.0+cu131"
    with pytest.raises(lane.PersonOnnxR12Error, match="runtime"):
        lane._validate_worker_result(result, profile=640)
    result = worker_result()
    result["batch12_shape_finite"] = False
    with pytest.raises(lane.PersonOnnxR12Error, match="batch 12"):
        lane._validate_worker_result(result, profile=640)
    result = worker_result()
    result["output_metadata_repair"][
        "model_sha256_excluding_graph_output_shapes_after"
    ] = "d" * 64
    with pytest.raises(lane.PersonOnnxR12Error, match="metadata-only model digest"):
        lane._validate_worker_result(result, profile=640)


def test_worker_command_is_networkless_cpu_only_and_fd_bound(fake_accepted) -> None:
    command, pass_fds = lane.worker_command(fake_accepted, profile=640, output_fd=37)
    assert command[0] == "/proc/self/fd/36"
    assert command[1:4] == ["-B", "-I", "-S"]
    assert command[4] == "/proc/self/fd/35"
    assert "internal-worker" in command
    assert "--checkpoint-fd" in command
    assert "--config-fd" in command
    assert "--output-fd" in command
    assert "docker" not in command
    assert "bwrap" not in command
    assert set(pass_fds) == {31, 33, 34, 35, 36, 37}
    environment = lane._cpu_child_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_independent_checker_command_does_not_bind_checkpoint_or_config(fake_accepted) -> None:
    command, pass_fds = lane.checker_command(fake_accepted, profile=960, onnx_fd=40)
    assert "--checkpoint-fd" not in command
    assert "--config-fd" not in command
    assert "internal-check" in command
    assert set(pass_fds) == {35, 36, 40}


def test_preflight_command_uses_only_exact_runtime_and_executor_fds(fake_accepted) -> None:
    command, pass_fds = lane.preflight_command(fake_accepted)
    assert command[:5] == [
        "/proc/self/fd/36",
        "-B",
        "-I",
        "-S",
        "/proc/self/fd/35",
    ]
    assert "internal-preflight" in command
    assert set(pass_fds) == {35, 36}


def test_libseccomp_filter_denies_raw_socket_before_heavy_imports() -> None:
    code = """
import errno, socket
from validation import person_rtdetrv4_onnx_export_r12 as lane
lane._install_network_seccomp()
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError as exc:
    raise SystemExit(0 if exc.errno == errno.EPERM else 3)
raise SystemExit(4)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=lane.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_known_batch12_graph_repair_changes_only_output_shape_metadata(
    tmp_path: Path,
) -> None:
    source = (
        lane.ROOT
        / "models/person/candidates/rtdetrv4-s/onnx/640/"
        "rtdetrv4-s-640-bdynamic-opset18.onnx"
    )
    repaired = tmp_path / "metadata-repaired.onnx"
    code = r"""
import json, os, sys
from pathlib import Path
root, source, repaired = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from validation import person_rtdetrv4_onnx_export_r12 as lane
lane._install_network_seccomp()
lane._prove_network_seccomp()
lane._activate_export_site_packages()
try:
    lane._inspect_onnx(source, profile=640)
except lane.PersonOnnxR12Error as exc:
    diagnostic = str(exc)
else:
    raise SystemExit("known raw graph unexpectedly had exact output metadata")
onnx, model, bindings, opsets = lane._load_checked_onnx(source)
evidence = lane._apply_output_metadata_repair(
    onnx,
    model,
    profile=640,
    observed_bindings=bindings,
    observed_opsets=opsets,
)
descriptor = os.open(repaired, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
try:
    lane._write_onnx_model_to_fd(model, descriptor)
finally:
    os.close(descriptor)
inspection = lane._inspect_onnx(repaired, profile=640)
print(json.dumps({"diagnostic": diagnostic, "evidence": evidence, "inspection": inspection}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            str(lane.RUNTIME_PYTHON_SOURCE),
            "-B",
            "-I",
            "-S",
            "-c",
            code,
            str(lane.ROOT),
            str(source),
            str(repaired),
        ],
        cwd=lane.ROOT,
        env=lane._cpu_child_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "ONNX tensor/opset contract differs" in result["diagnostic"]
    assert "Sublabels_dim_1" in result["diagnostic"]
    assert "GatherElementsboxes_dim_1" in result["diagnostic"]
    evidence = result["evidence"]
    assert evidence["applied"] is True
    assert evidence["scope"] == "graph.output.type.tensor_type.shape_only"
    assert (
        evidence["model_sha256_excluding_graph_output_shapes_before"]
        == evidence["model_sha256_excluding_graph_output_shapes_after"]
    )
    assert result["inspection"]["bindings"] == lane.expected_bindings(640)


def test_actual_preflight_imports_export_stack_after_seccomp_without_model() -> None:
    targets = [
        lane.ROOT / lane.ONNX_RELATIVE[profile]
        for profile in lane.PROFILES
    ] + [
        lane.ROOT / lane.RECEIPT_RELATIVE[profile]
        for profile in lane.PROFILES
    ] + [
        lane.ROOT
        / Path(lane.RECEIPT_RELATIVE[profile]).parent
        / lane.RECOVERY_NAME
        for profile in lane.PROFILES
    ]

    def snapshot(path: Path):
        try:
            value = path.lstat()
        except FileNotFoundError:
            return None
        return (value.st_mode, value.st_size, value.st_mtime_ns)

    before = {path: snapshot(path) for path in targets}
    executor_sha256 = hashlib.sha256(lane.THIS_FILE.read_bytes()).hexdigest()
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(lane.THIS_FILE),
            "preflight",
            "--plan",
            lane.PLAN_RELATIVE,
            "--accept-plan-fingerprint",
            lane.PLAN_FINGERPRINT,
            "--accept-contract-fingerprint",
            lane.CONTRACT_FINGERPRINT,
            "--accept-executor-sha256",
            executor_sha256,
            "--accept-checkpoint-sha256",
            lane.CHECKPOINT_SHA256,
        ],
        cwd=lane.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed"
    assert result["network_syscalls_seccomp_denied"] is True
    assert result["libraries_imported_after_seccomp"] == [
        "numpy",
        "onnxruntime",
        "torch",
        "torch.nn",
    ]
    assert result["socket_class_preserved"] is True
    assert result["gpu_api_queried"] is False
    assert result["model_loaded"] is False
    assert result["onnx_artifact_loaded"] is False
    assert result["onnx_exported"] is False
    assert {path: snapshot(path) for path in targets} == before


def test_subprocess_json_requires_exactly_one_line() -> None:
    assert lane._parse_exact_subprocess_json('{"ok":true}\n', label="unit") == {"ok": True}
    with pytest.raises(lane.PersonOnnxR12Error, match="exactly one"):
        lane._parse_exact_subprocess_json('{"a":1}\n{"b":2}\n', label="unit")


def test_internal_worker_is_inert_outside_sandbox(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSAFE_R12_SANDBOX", raising=False)
    args = SimpleNamespace(
        profile=640,
        accept_plan_fingerprint=lane.PLAN_FINGERPRINT,
        accept_executor_sha256="a" * 64,
        accept_checkpoint_sha256=lane.CHECKPOINT_SHA256,
    )
    with pytest.raises(lane.PersonOnnxR12Error, match="sandbox-only"):
        lane.internal_worker(args)


def test_failure_output_does_not_overclaim_model_was_not_exported(
    monkeypatch, capsys
) -> None:
    monkeypatch.delenv("DEEPSAFE_R12_SANDBOX", raising=False)
    result = lane.main(
        [
            "internal-worker",
            "--profile",
            "640",
            "--executor-fd",
            "35",
            "--plan-fd",
            "31",
            "--checkpoint-fd",
            "33",
            "--config-fd",
            "34",
            "--output-fd",
            "37",
            "--accept-plan-fingerprint",
            lane.PLAN_FINGERPRINT,
            "--accept-executor-sha256",
            "a" * 64,
            "--accept-checkpoint-sha256",
            lane.CHECKPOINT_SHA256,
        ]
    )
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert "model_exported" not in error
    assert error["execution_state"].startswith("unknown_before_or_after")


def test_bounded_bijection_accepts_permutation_and_rejects_delta() -> None:
    labels = np.zeros((300,), dtype=np.int64)
    boxes = np.arange(1200, dtype=np.float32).reshape(300, 4)
    scores = np.linspace(0.0, 1.0, 300, dtype=np.float32)
    permutation = np.arange(299, -1, -1)
    passed = lane._bounded_bijection(
        labels,
        boxes,
        scores,
        labels[permutation],
        boxes[permutation],
        scores[permutation],
    )
    assert passed["passed"] is True
    changed = boxes[permutation].copy()
    changed[0, 0] += 1.0
    failed = lane._bounded_bijection(
        labels,
        boxes,
        scores,
        labels[permutation],
        changed,
        scores[permutation],
    )
    assert failed["passed"] is False


def test_recovery_intent_reseal_is_rejected(fake_accepted, monkeypatch) -> None:
    receipt = make_receipt(fake_accepted)
    monkeypatch.setattr(lane.os, "fstat", lambda _fd: SimpleNamespace(st_size=777))
    recovery = lane._build_recovery_intent(
        accepted=fake_accepted, profile=640, receipt=receipt
    )
    recovery["onnx"]["sha256"] = "f" * 64
    recovery["fingerprint_sha256"] = lane.fingerprint(recovery)
    with pytest.raises(lane.PersonOnnxR12Error, match="ONNX binding"):
        lane._validate_recovery_intent(
            recovery, accepted=fake_accepted, profile=640
        )


def test_recovery_intent_binds_exact_executor_and_checkpoint(fake_accepted, monkeypatch) -> None:
    receipt = make_receipt(fake_accepted)
    monkeypatch.setattr(lane.os, "fstat", lambda _fd: SimpleNamespace(st_size=777))
    recovery = lane._build_recovery_intent(
        accepted=fake_accepted, profile=640, receipt=receipt
    )
    assert recovery["executor"]["sha256"] == "a" * 64
    assert recovery["checkpoint"]["sha256"] == lane.CHECKPOINT_SHA256
    assert recovery["final_receipt"]["fingerprint_sha256"] == receipt["fingerprint_sha256"]
