from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from validation import person_rtdetrv4_tensorrt_r14b as lane
from validation import person_rtdetrv4_tensorrt_r14b_container as worker


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    assert lane.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(lane.TensorRTR14BError, match="non-finite"):
        lane.canonical_bytes({"bad": math.nan})


def test_strict_json_rejects_duplicate_and_nan() -> None:
    with pytest.raises(lane.TensorRTR14BError, match="duplicate JSON key"):
        lane._strict_json_bytes(b'{"a":1,"a":2}', source="unit")
    with pytest.raises(lane.TensorRTR14BError, match="non-finite"):
        lane._strict_json_bytes(b'{"a":NaN}', source="unit")


def test_plan_projection_has_exact_r11_r12_r13_and_runtime_pins() -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    assert plan["r11"]["stage_order"] == list(lane.R11_STAGE_ORDER)
    assert plan["r11"]["plan"]["sha256"] == lane.R11_PLAN_SHA256
    assert plan["r11"]["contract"]["fingerprint_sha256"] == lane.R11_CONTRACT_FINGERPRINT
    assert plan["r12"]["onnx_receipts"]["640"] == lane.ONNX_RECEIPT_PINS[640]
    assert plan["r12"]["onnx_artifacts"]["960"] == lane.ONNX_PINS[960]
    assert plan["threshold_gate"]["r13b_plan"]["fingerprint_sha256"] == lane.R13_PLAN_FINGERPRINT
    assert plan["runtime"]["image_id"] == lane.IMAGE_ID
    assert plan["runtime"]["tensorrt"] == "10.14.1.48"
    assert plan["gpu"]["compute_capability"] == "8.6"
    assert plan["fingerprint_sha256"] == lane.fingerprint(plan)


def test_plan_selects_24_real_images_from_24_capture_groups() -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    items = plan["real_image_selection"]["items"]
    assert len(items) == 24
    assert len({item["capture_group_id"] for item in items}) == 24
    assert all("materialized-v1" in item["path"] for item in items)
    for item in items:
        path = lane.ROOT / item["path"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_plan_schema_rejects_image_and_runtime_drift() -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    changed = copy.deepcopy(plan)
    changed["runtime"]["tensorrt"] = "10.14.1.49"
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14BError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["real_image_selection"]["images"] = 23
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14BError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def test_threshold_missing_fails_before_any_subprocess_or_gpu_probe(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-threshold-receipt.json"
    monkeypatch.setattr(lane, "THRESHOLD_RECEIPT", missing)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not be reached")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    with pytest.raises(lane.TensorRTR14BError, match="before Docker/GPU probe"):
        lane.load_threshold_gate({"threshold_gate": {"receipt_path": lane.THRESHOLD_RECEIPT_RELATIVE}})


def test_r14_stage_order_is_exact_and_640_precedes_960() -> None:
    assert lane.STAGES == (
        "tensorrt_fp16_640",
        "tensorrt_fp16_960",
        "numerical_parity_640",
        "numerical_parity_960",
        "deepstream_parser_parity_640",
        "deepstream_parser_parity_960",
    )
    assert lane.STAGES.index("tensorrt_fp16_640") < lane.STAGES.index("tensorrt_fp16_960")
    assert lane.STAGES.index("numerical_parity_640") < lane.STAGES.index("numerical_parity_960")
    assert lane.STAGES.index("deepstream_parser_parity_640") < lane.STAGES.index("deepstream_parser_parity_960")


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_paths_match_immutable_r11_engine_contract(profile: int) -> None:
    paths = lane.profile_paths(profile)
    assert paths["engine_path"].endswith(
        f"engines/{profile}/rtdetrv4-s-r11-{profile}-b12-fp16-ds9-trt10.14.engine"
    )
    assert paths["engine_receipt_path"].endswith(f"tensorrt-{profile}/receipt.json")
    profile_plan = lane._profile_plan(profile)
    assert profile_plan["images"] == {
        "min": [1, 3, profile, profile],
        "opt": [12, 3, profile, profile],
        "max": [12, 3, profile, profile],
    }
    assert profile_plan["orig_target_sizes"] == {
        "min": [1, 2], "opt": [12, 2], "max": [12, 2]
    }


def test_build_docker_command_is_exact_offline_foreground_target(tmp_path: Path) -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    output = tmp_path / "output"
    output.mkdir()
    command = lane.render_docker_command(
        plan=plan,
        stage="tensorrt_fp16_640",
        output_dir=output,
        cidfile=tmp_path / "container.cid",
    )
    assert command[:5] == ["docker", "run", "--rm", "--pull=never", "--network=none"]
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--gpus=device=0" in command
    assert lane.IMAGE_ID in command
    assert lane.IMAGE_REFERENCE not in command
    assert "--fp16" not in command  # trtexec is invoked only by the pinned worker
    assert command[-1] == lane.ONNX_PINS[640]["sha256"]


def test_gpu_lease_wraps_exactly_one_docker_command(tmp_path: Path) -> None:
    docker = ["docker", "run", "--rm", lane.IMAGE_ID, "true"]
    wrapped = lane.render_lease_command(docker)
    assert wrapped[:4] == [lane.sys.executable, "-m", "validation.gpu_lease", "run"]
    assert wrapped[4:10] == [
        "--owner-kind", "legacy_validation", "--gpu-index", "0",
        "--ttl-seconds", "30",
    ]
    assert wrapped[-len(docker):] == docker
    with pytest.raises(lane.TensorRTR14BError, match="only docker run"):
        lane.render_lease_command(["nvidia-smi"])


@pytest.mark.parametrize("profile,threshold", [(640, 0.0), (960, 0.375), (640, 1.0)])
def test_deepstream_config_has_exact_no_clustering_and_calibrated_threshold(
    profile: int, threshold: float
) -> None:
    rendered = lane._render_config(profile, threshold).decode("utf-8")
    assert "cluster-mode=4" in rendered
    assert "maintain-aspect-ratio=1" in rendered
    assert "symmetric-padding=1" in rendered
    assert "parse-bbox-func-name=NvDsInferParseCustomRTDETRv4Person" in rendered
    assert "custom-lib-path=/models/runtime/rtdetrv4-parser-ds9-r1/libdeepsafe_rtdetrv4_parser.so" in rendered
    assert "output-blob-names=labels;boxes;scores" in rendered
    assert f"infer-dims=3;{profile};{profile}" in rendered
    assert f"pre-cluster-threshold={repr(float(threshold))}" in rendered
    assert "[class-attrs-0]" in rendered


def test_binding_projection_accepts_trt_int32_labels_but_preserves_exact_shape() -> None:
    observed = [
        {"name": "images", "io": "input", "dtype": "FLOAT"},
        {"name": "orig_target_sizes", "io": "input", "dtype": "INT64"},
        {"name": "labels", "io": "output", "dtype": "INT32"},
        {"name": "boxes", "io": "output", "dtype": "FLOAT"},
        {"name": "scores", "io": "output", "dtype": "HALF"},
    ]
    projected = lane._binding_projection(960, observed)
    assert projected[2] == {
        "name": "labels", "io": "output", "dtype": "INT32",
        "shape": ["batch", 300],
    }
    assert projected[3]["shape"] == ["batch", 300, 4]


def test_build_r11_receipt_is_schema_valid_and_claim_bounded() -> None:
    worker_result = {
        "validation": {
            "optimization_profiles": {
                "images": {"min": [1, 3, 640, 640], "opt": [12, 3, 640, 640], "max": [12, 3, 640, 640]},
                "orig_target_sizes": {"min": [1, 2], "opt": [12, 2], "max": [12, 2]},
            },
            "bindings": [
                {"name": "images", "io": "input", "dtype": "FLOAT"},
                {"name": "orig_target_sizes", "io": "input", "dtype": "INT64"},
                {"name": "labels", "io": "output", "dtype": "INT32"},
                {"name": "boxes", "io": "output", "dtype": "FLOAT"},
                {"name": "scores", "io": "output", "dtype": "FLOAT"},
            ],
        }
    }
    artifact = {
        "path": lane.profile_paths(640)["engine_path"],
        "bytes": 1234,
        "sha256": "a" * 64,
    }
    receipt = lane.build_r11_stage_receipt(
        stage="tensorrt_fp16_640",
        profile=640,
        prior_receipts=[lane.ONNX_RECEIPT_PINS[640], {**lane.ONNX_RECEIPT_PINS[960], "path": lane.THRESHOLD_RECEIPT_RELATIVE}],
        threshold_pin={**lane.ONNX_RECEIPT_PINS[960], "path": lane.THRESHOLD_RECEIPT_RELATIVE},
        worker=worker_result,
        published_artifact=artifact,
        numerical=None,
        config_pin=None,
        created_at_utc="2026-07-18T08:00:00+00:00",
    )
    assert receipt["stage"] == "tensorrt_fp16_640"
    assert receipt["payload"]["precision"] == "FP16"
    assert receipt["payload"]["bindings"][2]["dtype"] == "INT32"
    assert receipt["claim_boundary"]["production_ready"] is False
    assert receipt["fingerprint_sha256"] == lane.fingerprint(receipt)


def test_native_adapter_calls_production_parser_and_one_time_initializer() -> None:
    source = lane.PARSER_ADAPTER.read_text(encoding="utf-8")
    assert "NvDsInferParseCustomRTDETRv4Person" in source
    assert "NvDsInferInitializeInputLayers" in source
    assert "DeepSafeRTDETRv4ParserParityR14B" in source
    assert "DeepSafeRTDETRv4InitializerParityR14B" in source
    assert "batch_size > 12U" in source


def test_container_build_contract_has_fp16_no_int8_and_b1_b12_smoke() -> None:
    source = lane.CONTAINER_WORKER.read_text(encoding="utf-8")
    assert '"--fp16"' in source
    assert '"--noTF32"' in source
    assert "--int8" not in source
    assert 'for batch in (1, 12):' in source
    assert '"--skipInference"' in source
    assert "execute_async_v3" in source


def test_container_parser_contract_proves_no_nms_duplicates_and_initializer() -> None:
    source = lane.CONTAINER_WORKER.read_text(encoding="utf-8")
    assert '"cluster_mode": 4' in source
    assert '"clustering": "disabled"' in source
    assert '"second_nms": False' in source
    assert "duplicates_preserved" in source
    assert "once_before_first_inference" in source
    assert "batch12_height_width_rows_exact" in source
    assert "NvDsInferParseCustomRTDETRv4Person@@DEEPSAFE_RTDETRV4_PARSER_1.0" in source
    assert '["ldd", "-r", str(args.production_parser)]' in source


def test_pair_iou_treats_identical_degenerate_pair_as_exact() -> None:
    left = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 10.0, 10.0]])
    right = left.copy()
    observed = lane._pair_iou_matrix(left, right, np)
    assert observed[0, 0] == 1.0
    assert observed[1, 1] == 1.0


def test_atomic_bytes_is_immutable_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    lane.atomic_bytes(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    assert path.stat().st_mode & 0o777 == 0o440
    with pytest.raises(lane.TensorRTR14BError, match="already exists"):
        lane.atomic_bytes(path, b'{"changed":true}\n')


def test_atomic_copy_is_fd_bound_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.engine"
    source.write_bytes(b"engine-bytes")
    destination = tmp_path / "published.engine"
    lane.atomic_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o440
    with pytest.raises(lane.TensorRTR14BError, match="already exists"):
        lane.atomic_copy(source, destination)


def test_container_lease_environment_rejects_missing_lease(monkeypatch) -> None:
    for name in (
        "DEEPSAFE_GPU_LEASE_HELD", "DEEPSAFE_GPU_LEASE_ID",
        "DEEPSAFE_GPU_LEASE_GPU_INDEX", "DEEPSAFE_GPU_LEASE_GPU_UUID",
        "DEEPSAFE_GPU_LEASE_OWNER_KIND", "DEEPSAFE_GPU_LEASE_COMMAND_SHA256",
        "DEEPSAFE_GPU_LEASE_CONTRACT_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(worker.ContainerR14BError, match="not held"):
        worker.lease_environment()


def test_container_lease_environment_accepts_exact_non_secret_projection(monkeypatch) -> None:
    values = {
        "DEEPSAFE_GPU_LEASE_HELD": "1",
        "DEEPSAFE_GPU_LEASE_ID": "a" * 64,
        "DEEPSAFE_GPU_LEASE_GPU_INDEX": "0",
        "DEEPSAFE_GPU_LEASE_GPU_UUID": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
        "DEEPSAFE_GPU_LEASE_OWNER_KIND": "legacy_validation",
        "DEEPSAFE_GPU_LEASE_COMMAND_SHA256": "b" * 64,
        "DEEPSAFE_GPU_LEASE_CONTRACT_SHA256": "c" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    observed = worker.lease_environment()
    assert observed["lease_id"] == "a" * 64
    assert "capability" not in observed


def test_public_module_import_has_no_docker_gpu_or_trtexec_side_effects() -> None:
    # The constants may contain command names; there is no subprocess call at
    # module scope.  The AST-free assertion keeps this test dependency-light.
    source = lane.THIS_FILE.read_text(encoding="utf-8")
    prefix = source[: source.index("def execute_stage(")]
    assert "subprocess.run(leased_command" not in prefix
    worker_source = lane.CONTAINER_WORKER.read_text(encoding="utf-8")
    assert "if __name__ == \"__main__\":" in worker_source


def test_plan_claim_boundary_never_implies_quality_capacity_or_25m() -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    assert plan["claim_boundary"] == {
        "quality": False,
        "exact_25m": False,
        "twelve_camera_capacity": False,
        "three_module_full_stack": False,
        "production_ready": False,
    }


def _metrics(gt: int, tp: int, fp: int) -> dict[str, int | float]:
    fn = gt - tp
    selected = tp + fp
    denominator = 2 * tp + fp + fn
    return {
        "ground_truth": gt, "selected_predictions": selected,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / selected, 12) if selected else 0.0,
        "recall": round(tp / gt, 12) if gt else 0.0,
        "f1": round(2 * tp / denominator, 12) if denominator else 0.0,
    }


def test_r13b_final_and_sweeps_are_exact_external_acceptances() -> None:
    receipt, pin, sweeps = lane.replay_exact_r13b_acceptance()
    assert pin == lane.THRESHOLD_RECEIPT_PIN
    assert receipt["fingerprint_sha256"] == lane.THRESHOLD_RECEIPT_PIN["fingerprint_sha256"]
    assert {profile: sweeps[profile]["selected"]["threshold"] for profile in lane.PROFILES} == lane.THRESHOLDS
    assert sweeps[640]["fingerprint_sha256"] == lane.THRESHOLD_SWEEP_PINS[640]["fingerprint_sha256"]
    assert sweeps[960]["fingerprint_sha256"] == lane.THRESHOLD_SWEEP_PINS[960]["fingerprint_sha256"]


def test_r13b_semantic_replay_rejects_forged_metric_and_selection() -> None:
    points = [
        {"threshold": 0.0, "metrics": _metrics(10, 8, 8)},
        {"threshold": 0.5, "metrics": _metrics(10, 8, 2)},
        {"threshold": 1.0, "metrics": _metrics(10, 0, 0)},
    ]
    selected = {
        "threshold": 0.5,
        "objective": "max_f1_tie_break_higher_recall_then_lower_threshold",
        "tie_break": ["higher_exact_recall", "lower_threshold"],
        "metrics": points[1]["metrics"], "threshold_finite": True,
    }
    lane.replay_r13b_selection(points, selected, expected_ground_truth=10)
    forged = copy.deepcopy(points)
    forged[1]["metrics"]["precision"] = 0.999
    with pytest.raises(lane.TensorRTR14BError, match="count-derived metric"):
        lane.replay_r13b_selection(forged, selected, expected_ground_truth=10)
    wrong = copy.deepcopy(selected)
    wrong["threshold"] = 0.0
    wrong["metrics"] = points[0]["metrics"]
    with pytest.raises(lane.TensorRTR14BError, match="max-F1"):
        lane.replay_r13b_selection(points, wrong, expected_ground_truth=10)


def test_superseded_r14_is_exact_and_immutable() -> None:
    assert lane._verify_superseded_r14() == lane.R14_SUPERSEDED
    assert lane.R14_SUPERSEDED["plan"]["sha256"] == "72045bfa09c40a38aa60dc22f7bb3c55e290c15f210068a5804682638090d374"
    assert lane.R14_SUPERSEDED["host_executor"]["sha256"] == "1e42c5c880c07d4b0c8c2b576acc9429ece8ec484ce5c2921ce53ba294739b5d"


def test_production_parser_is_dlopen_dlvsym_dladdr_not_recompiled() -> None:
    adapter = lane.PARSER_ADAPTER.read_text(encoding="utf-8")
    container = lane.CONTAINER_WORKER.read_text(encoding="utf-8")
    assert "::dlopen(requested, RTLD_NOW | RTLD_LOCAL)" in adapter
    assert "::dlvsym" in adapter and "::dladdr" in adapter
    assert "same_regular_inode" in adapter
    assert "str(args.parser_source), str(args.adapter_source)" not in container
    assert "str(args.adapter_source)" in container and '"-ldl"' in container


def test_native_gst_nvinfer_probe_checks_exact_tensor_and_frame_meta() -> None:
    source = lane.GST_NVINFER_SMOKE.read_text(encoding="utf-8")
    for token in (
        "gstnvdsinfer.h", "gst_buffer_get_nvds_batch_meta",
        "NVDSINFER_TENSOR_OUTPUT_META", "tensor->unique_id == 1U",
        "tensor->num_output_layers == 3U", "out_buf_ptrs_host",
        'constexpr const char* names[] = {"labels", "boxes", "scores"}',
        "batch->num_frames_in_batch != 1U", "frame->batch_id != 0U",
        'element("nvinfer", "primary-person")',
    ):
        assert token in source


def test_successor_requires_exact_external_commit_acceptance_set() -> None:
    with pytest.raises(lane.TensorRTR14BError, match="acceptance set"):
        lane._require_accepted_predecessor_commits({}, "tensorrt_fp16_640", {"tensorrt_fp16_640": "a" * 64})
    with pytest.raises(lane.TensorRTR14BError, match="acceptance set"):
        lane._require_accepted_predecessor_commits({}, "numerical_parity_640", {})
    assert lane.parse_commit_acceptances(["tensorrt_fp16_640=" + "b" * 64]) == {"tensorrt_fp16_640": "b" * 64}
    with pytest.raises(lane.TensorRTR14BError, match="commit acceptance"):
        lane.parse_commit_acceptances(["tensorrt_fp16_640=" + "b" * 64, "tensorrt_fp16_640=" + "c" * 64])


def test_cli_has_no_docker_argv_or_worker_override_surface() -> None:
    with pytest.raises(SystemExit):
        lane._parser().parse_args([
            "execute", "--accept-plan-fingerprint", "a" * 64,
            "--stage", "tensorrt_fp16_640", "--run-id", "unit-run",
            "--docker-argv", "docker run forged",
        ])
    with pytest.raises(SystemExit):
        lane._parser().parse_args([
            "execute", "--accept-plan-fingerprint", "a" * 64,
            "--stage", "tensorrt_fp16_640", "--run-id", "unit-run",
            "--worker", "/tmp/forged.py",
        ])


def test_held_fd_publication_survives_source_name_swap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    source = tmp_path / "source.engine"
    source.write_bytes(b"reviewed-engine")
    descriptor, source_pin = lane.open_held_source(source)
    try:
        moved = tmp_path / "moved.engine"
        source.rename(moved)
        source.write_bytes(b"forged-engine")
        destination = tmp_path / "published.engine"
        expected = {"path": "published.engine", "bytes": source_pin["bytes"], "sha256": source_pin["sha256"]}
        assert lane.publish_held_fd(descriptor, destination, expected, allow_existing_exact=False) == expected
        assert destination.read_bytes() == b"reviewed-engine"
    finally:
        os.close(descriptor)


def test_commit_last_partial_recovery_fills_exact_members(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    roles = ("artifact", "r11_stage_receipt", "execution_receipt")
    descriptors: dict[str, int] = {}
    members: list[dict[str, object]] = []
    try:
        for role in roles:
            source = tmp_path / f"source-{role}.bin"
            source.write_bytes(role.encode())
            descriptor, source_pin = lane.open_held_source(source)
            descriptors[role] = descriptor
            published = {"path": f"final-{role}.bin", "bytes": source_pin["bytes"], "sha256": source_pin["sha256"]}
            members.append({"role": role, "source": source_pin, "destination": published["path"], "published": published})
        commit_source = tmp_path / "source-commit.json"
        commit_source.write_bytes(b'{"fingerprint_sha256":"' + b"d" * 64 + b'"}\n')
        commit_descriptor, commit_source_pin = lane.open_held_source(commit_source)
        commit_pin = {"path": "commit.json", "bytes": commit_source_pin["bytes"], "sha256": commit_source_pin["sha256"], "fingerprint_sha256": "d" * 64}
        intent = {"members": members, "commit_destination": "commit.json"}
        # Simulate a crash after only the artifact linkat.
        lane.publish_held_fd(descriptors["artifact"], tmp_path / "final-artifact.bin", members[0]["published"], allow_existing_exact=False)
        order: list[str] = []
        original = lane.publish_held_fd

        def observed(fd, destination, expected, *, allow_existing_exact):
            order.append(Path(destination).name)
            return original(fd, destination, expected, allow_existing_exact=allow_existing_exact)

        monkeypatch.setattr(lane, "publish_held_fd", observed)
        result = lane.publish_transaction_from_held_fds(
            intent=intent, member_descriptors=descriptors,
            commit_descriptor=commit_descriptor, commit_pin=commit_pin,
            recovery=True,
        )
        assert result["status"] == "committed"
        assert order[-1] == "commit.json"
        assert all((tmp_path / f"final-{role}.bin").exists() for role in roles)
        assert (tmp_path / "commit.json").exists()
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        if "commit_descriptor" in locals():
            os.close(commit_descriptor)


def test_process_group_timeout_kills_term_ignoring_child(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "PROCESS_TERM_GRACE_SECONDS", 0.1)
    command = [
        sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)",
    ]
    with pytest.raises(lane.TensorRTR14BError, match="timed out"):
        lane._run_bounded_process_group(
            command, log_path=tmp_path / "timeout.log", timeout_seconds=1,
            output_limit_bytes=4096,
        )
    assert (tmp_path / "timeout.log").read_text().startswith("ready")


def test_process_group_output_is_bounded_and_fails_closed(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stdout.write('x'*8192); sys.stdout.flush()"]
    with pytest.raises(lane.TensorRTR14BError, match="output exceeded"):
        lane._run_bounded_process_group(
            command, log_path=tmp_path / "bounded.log", timeout_seconds=10,
            output_limit_bytes=1024,
        )
    assert (tmp_path / "bounded.log").stat().st_size == 1024


def test_worker_schema_and_commit_schemas_are_strict() -> None:
    for path in (lane.WORKER_SCHEMA, lane.COMMIT_SCHEMA, lane.INTENT_SCHEMA, lane.RECEIPT_SCHEMA):
        schema = lane.load_json(path)
        assert schema["additionalProperties"] is False
    commit = {
        "schema_version": lane.COMMIT_SCHEMA_VERSION, "commit_id": "unit-commit-r14b",
        "status": "committed", "stage": "tensorrt_fp16_640", "profile": 640,
        "run_id": "unit-run", "committed_at_utc": "2026-07-18T00:00:00+00:00",
        "plan": {"path": "plan.json", "bytes": 1, "sha256": "a" * 64, "fingerprint_sha256": "b" * 64},
        "publication_intent": {"path": "intent.json", "bytes": 1, "sha256": "c" * 64, "fingerprint_sha256": "d" * 64},
        "artifact": {"path": "artifact", "bytes": 1, "sha256": "e" * 64},
        "r11_stage_receipt": {"path": "r11.json", "bytes": 1, "sha256": "f" * 64, "fingerprint_sha256": "1" * 64},
        "execution_receipt": {"path": "execution.json", "bytes": 1, "sha256": "2" * 64, "fingerprint_sha256": "3" * 64},
        "worker_result": {"path": "worker.json", "bytes": 1, "sha256": "4" * 64},
        "publication": {"held_fd_sources": True, "otmpfile_linkat_noreplace": True, "bundle_members_verified_before_commit": True, "commit_manifest_last": True, "partial_recovery_supported": True, "overwrite": False},
        "claim_boundary": {"quality": False, "exact_25m": False, "twelve_camera_capacity": False, "three_module_full_stack": False, "production_ready": False},
        "fingerprint_sha256": "5" * 64,
    }
    lane.validate_schema(commit, lane.COMMIT_SCHEMA)
    forged = copy.deepcopy(commit)
    forged["publication"]["commit_manifest_last"] = False
    with pytest.raises(lane.TensorRTR14BError, match="schema validation failed"):
        lane.validate_schema(forged, lane.COMMIT_SCHEMA)


def test_exact_worker_schema_rejects_binding_dtype_forge(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    engine = output / "engine.staging"
    engine.write_bytes(b"engine")
    output_pin = {"path": engine.name, "bytes": engine.stat().st_size, "sha256": hashlib.sha256(engine.read_bytes()).hexdigest()}
    gpu_uuid = "GPU-8cbaba1c-2629-a732-f528-66f459089ef6"
    shapes = {
        "images": {"min": [1, 3, 640, 640], "opt": [12, 3, 640, 640], "max": [12, 3, 640, 640]},
        "orig_target_sizes": {"min": [1, 2], "opt": [12, 2], "max": [12, 2]},
    }
    bindings = [
        {"name": "images", "io": "input", "dtype": "FLOAT", "profile_shapes": shapes["images"]},
        {"name": "orig_target_sizes", "io": "input", "dtype": "INT64", "profile_shapes": shapes["orig_target_sizes"]},
        {"name": "labels", "io": "output", "dtype": "INT32", "profile_shapes": {}},
        {"name": "boxes", "io": "output", "dtype": "FLOAT", "profile_shapes": {}},
        {"name": "scores", "io": "output", "dtype": "HALF", "profile_shapes": {}},
    ]
    sample = {"monotonic_seconds": 0.0, "gpu_uuid": gpu_uuid, "utilization_gpu_percent": 0.0, "memory_used_mib": 1.0, "temperature_c": 40.0, "power_w": 1.0, "graphics_clock_mhz": 1.0, "memory_clock_mhz": 1.0}
    value = {
        "schema_version": lane.WORKER_SCHEMA_VERSION, "status": "passed", "operation": "build", "profile": 640,
        "plan_fingerprint_sha256": "a" * 64, "duration_seconds": 1.0,
        "lease": {"held": "1", "lease_id": "b" * 64, "gpu_index": "0", "gpu_uuid": gpu_uuid, "owner_kind": "legacy_validation", "command_sha256": "c" * 64, "contract_sha256": "d" * 64},
        "gpu": {"name": "NVIDIA RTX A5000 Laptop GPU", "uuid": gpu_uuid, "compute_capability": "8.6"},
        "runtime": {"image_reference": lane.IMAGE_REFERENCE, "image_id": lane.IMAGE_ID, "deepstream": "9.0.0", "cuda": "13.1", "tensorrt": "10.14.1.48", "trtexec": "/usr/src/tensorrt/bin/trtexec"},
        "telemetry": {"sample_interval_seconds": 1.0, "samples": [sample, {**sample, "monotonic_seconds": 1.0}], "all_samples_same_gpu": True, "sample_errors": []},
        "outputs": [output_pin],
        "validation": {"kind": "tensorrt_engine_build", "precision": "FP16", "int8": False, "optimization_profiles": shapes, "bindings": bindings, "engine_deserialized": True, "smoke_batches": [1, 12], "smoke_shapes": {}, "trtexec_command": ["/usr/src/tensorrt/bin/trtexec", "--fp16"], "trtexec_command_sha256": "e" * 64, "trtexec_exit_code": 0, "passed": True},
    }
    value["fingerprint_sha256"] = lane.fingerprint(value)
    result_path = output / "worker-result.json"
    lane.atomic_json(result_path, value)
    plan = {"fingerprint_sha256": "a" * 64, "execution_contract": {"global_gpu_lease": {"contract": {"fingerprint_sha256": "d" * 64}}}}
    assert lane._validate_worker_result(result_path, plan=plan, stage="tensorrt_fp16_640") == value
    forged = copy.deepcopy(value)
    forged["validation"]["bindings"][2]["dtype"] = "INT8"
    forged["fingerprint_sha256"] = lane.fingerprint(forged)
    with pytest.raises(lane.TensorRTR14BError, match="schema validation failed"):
        lane.validate_schema(forged, lane.WORKER_SCHEMA)


def test_recover_publication_replays_accepted_intent_without_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    monkeypatch.setattr(lane, "RUNS_ROOT", tmp_path / "runs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b'{"fingerprint_sha256":"' + b"a" * 64 + b'"}\n')
    monkeypatch.setattr(lane, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(lane, "PLAN_RELATIVE", "plan.json")
    plan = {"fingerprint_sha256": "a" * 64}
    monkeypatch.setattr(lane, "verify_plan", lambda path, expected_fingerprint: plan)
    stage = "tensorrt_fp16_640"
    run_id = "unit-recovery"
    run_root = lane.RUNS_ROOT / stage / run_id
    control = run_root / "control"
    control.mkdir(parents=True)

    artifact_source_path = control / "engine.staging"
    artifact_source_path.write_bytes(b"engine")
    r11_source_path = control / "publish-r11.json"
    r11_value = {"fingerprint_sha256": "b" * 64}
    r11_source_path.write_bytes(lane._json_payload(r11_value))
    execution_source_path = control / "publish-execution-r14b.json"
    execution_value = {"fingerprint_sha256": "c" * 64}
    execution_source_path.write_bytes(lane._json_payload(execution_value))
    artifact_source = lane.file_pin(artifact_source_path)
    r11_source = lane.file_pin(r11_source_path)
    execution_source = lane.file_pin(execution_source_path)
    artifact_destination = lane._stage_final_artifact(stage)
    r11_destination, execution_destination = lane._stage_receipt_paths(stage)
    artifact_published = {**artifact_source, "path": lane.repo_relative(artifact_destination)}
    r11_published = {**r11_source, "path": lane.repo_relative(r11_destination), "fingerprint_sha256": "b" * 64}
    execution_published = {**execution_source, "path": lane.repo_relative(execution_destination), "fingerprint_sha256": "c" * 64}
    intent = lane.build_publication_intent(
        plan=plan, stage=stage, run_id=run_id,
        artifact_source=artifact_source, artifact_published=artifact_published,
        r11_source=r11_source, r11_published=r11_published,
        execution_source=execution_source, execution_published=execution_published,
        prepared_at_utc="2026-07-18T00:00:00+00:00",
    )
    intent_path = control / "publication-intent-r14b.json"
    lane.atomic_json(intent_path, intent)
    worker_pin = {"path": lane.repo_relative(control / "worker-result.json"), "bytes": 1, "sha256": "d" * 64}
    commit = lane.build_commit_manifest(
        plan=plan, stage=stage, run_id=run_id,
        intent_pin=lane.document_pin(intent_path), artifact_pin=artifact_published,
        r11_pin=r11_published, execution_pin=execution_published,
        worker_pin=worker_pin, committed_at_utc="2026-07-18T00:00:00+00:00",
    )
    commit_source_path = control / "publish-commit-r14b.json"
    lane.atomic_json(commit_source_path, commit)
    descriptor, _pin = lane.open_held_source(artifact_source_path)
    try:
        lane.publish_held_fd(descriptor, artifact_destination, artifact_published, allow_existing_exact=False)
    finally:
        os.close(descriptor)
    with pytest.raises(lane.TensorRTR14BError, match="external recovery intent"):
        lane.recover_partial_publication(
            plan_path=plan_path, accepted_plan_fingerprint="a" * 64,
            stage=stage, run_id=run_id, accepted_intent_fingerprint="e" * 64,
        )
    result = lane.recover_partial_publication(
        plan_path=plan_path, accepted_plan_fingerprint="a" * 64,
        stage=stage, run_id=run_id,
        accepted_intent_fingerprint=intent["fingerprint_sha256"],
    )
    assert result["recovered"] is True and result["gpu"] is False and result["docker"] is False
    assert r11_destination.exists() and execution_destination.exists()
    assert lane._stage_commit_path(stage).exists()


def test_docker_worker_hashes_are_derived_from_same_held_fds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    runner = tmp_path / "validation/runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# pinned worker\n", encoding="utf-8")
    monkeypatch.setattr(lane, "CONTAINER_WORKER", runner)
    engine = tmp_path / lane.profile_paths(640)["engine_path"]
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"reviewed-engine")
    bundle = tmp_path / "control/parity-input.npz"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"reviewed-input")
    output = tmp_path / "output"
    output.mkdir()
    descriptors: dict[str, int] = {}
    try:
        for path in (engine, bundle):
            descriptor, pin = lane.open_held_source(path)
            descriptors[pin["path"]] = descriptor
            path.rename(path.with_suffix(path.suffix + ".reviewed"))
            path.write_bytes(b"forged-name-content")
        command = lane.render_docker_command(
            plan={"fingerprint_sha256": "a" * 64},
            stage="numerical_parity_640", output_dir=output,
            cidfile=tmp_path / "container.cid", input_bundle=bundle,
            held_sources=descriptors,
        )
        assert hashlib.sha256(b"reviewed-engine").hexdigest() in command
        assert hashlib.sha256(b"reviewed-input").hexdigest() in command
        assert hashlib.sha256(b"forged-name-content").hexdigest() not in command
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_lease_event_semantic_replay_rejects_command_forge(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    command_digest = "1" * 64
    lease_id = "2" * 64
    gpu_uuid = "GPU-8cbaba1c-2629-a732-f528-66f459089ef6"
    contract = "3" * 64

    def write_event(path: Path, digest: str) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "deepsafe.gpu-lease-event-receipt/v1",
            "event_id": "4" * 64, "event_type": "acquire",
            "created_at_utc": "2026-07-18T00:00:00Z", "created_monotonic_ns": 1,
            "contract_fingerprint_sha256": contract,
            "gpu": {"index": 0, "uuid": gpu_uuid},
            "lease": {
                "lease_id": lease_id, "owner_kind": "legacy_validation",
                "owner_identity_sha256": "5" * 64,
                "command_argv_sha256": digest, "capability_sha256": "6" * 64,
            },
            "previous_state_fingerprint_sha256": None,
            "next_lease_record_sha256": "7" * 64,
            "decision": {
                "accepted": True, "reason": "uncontended",
                "stale_recovery_reason": None, "legacy_lock_held": True,
            },
        }
        value["event_fingerprint_sha256"] = lane.fingerprint(value, "event_fingerprint_sha256")
        path.parent.mkdir(parents=True, exist_ok=True)
        lane.atomic_json(path, value)
        return {**lane.file_pin(path), "fingerprint_sha256": value["event_fingerprint_sha256"]}

    accepted = write_event(tmp_path / "receipts/accepted.json", command_digest)
    lane._validate_lease_event(
        accepted, allowed_types={"acquire"}, lease_id=lease_id,
        command_digest=command_digest, gpu_uuid=gpu_uuid,
        contract_fingerprint=contract,
    )
    forged = write_event(tmp_path / "receipts/forged.json", "8" * 64)
    with pytest.raises(lane.TensorRTR14BError, match="command digest"):
        lane._validate_lease_event(
            forged, allowed_types={"acquire"}, lease_id=lease_id,
            command_digest=command_digest, gpu_uuid=gpu_uuid,
            contract_fingerprint=contract,
        )
