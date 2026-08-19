from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest

from validation import person_rtdetrv4_tensorrt_r14c as lane
from validation import person_rtdetrv4_tensorrt_r14c_container as worker


def _seal_bytes(payload: bytes) -> int:
    descriptor = os.memfd_create(
        "r14c-adversarial-test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    os.write(descriptor, payload)
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, lane.MEMFD_REQUIRED_SEALS)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return descriptor


def _tar_payload(
    manifest: dict[str, object],
    members: list[tuple[tarfile.TarInfo, bytes | None]],
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        raw = lane.canonical_bytes(manifest)
        archive.addfile(lane._tar_info("input-manifest.json", len(raw)), io.BytesIO(raw))
        for info, payload in members:
            archive.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return output.getvalue()


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    assert lane.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(lane.TensorRTR14CError, match="non-finite"):
        lane.canonical_bytes({"bad": math.nan})


def test_strict_json_rejects_duplicate_and_nan() -> None:
    with pytest.raises(lane.TensorRTR14CError, match="duplicate JSON key"):
        lane._strict_json_bytes(b'{"a":1,"a":2}', source="unit")
    with pytest.raises(lane.TensorRTR14CError, match="non-finite"):
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
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["execution_contract"]["input_transport"]["proc_fd_bind_mounts"] = 1
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["execution_contract"]["process_lifecycle"]["lease_cleanup_margin_seconds"] = 0
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["real_image_selection"]["images"] = 23
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def test_threshold_missing_fails_before_any_subprocess_or_gpu_probe(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-threshold-receipt.json"
    monkeypatch.setattr(lane, "THRESHOLD_RECEIPT", missing)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not be reached")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    with pytest.raises(lane.TensorRTR14CError, match="before Docker/GPU probe"):
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
    assert paths["worker_result_path"].endswith(f"tensorrt-{profile}/worker-result-r14c.json")
    assert paths["numerical_worker_result_path"].endswith(
        f"numerical-parity-{profile}/worker-result-r14c.json"
    )
    assert paths["parser_worker_result_path"].endswith(
        f"deepstream-parser-parity-{profile}/worker-result-r14c.json"
    )
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
    manifest = lane.build_input_manifest(
        stage="tensorrt_fp16_640",
        members=[
            ("runner.py", lane.file_pin(lane.CONTAINER_WORKER)),
            ("model.onnx", lane.ONNX_PINS[640]),
        ],
    )
    command = lane.render_docker_command(
        plan=plan,
        stage="tensorrt_fp16_640",
        output_dir=output,
        cidfile=tmp_path / "container.cid",
        input_manifest=manifest,
    )
    assert command[:6] == ["docker", "run", "--rm", "-i", "--pull=never", "--network=none"]
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--gpus=device=0" in command
    assert lane.IMAGE_ID in command
    assert lane.IMAGE_REFERENCE not in command
    assert not any("/proc/" in item and "/fd/" in item for item in command)
    assert len([item for item in command if item.startswith("--mount=")]) == 1
    assert "-v" not in command and "--volume" not in command
    assert not any(item.startswith("--volume=") for item in command)
    assert "--env=DEEPSAFE_MANAGED_DOCKER_CIDFILE" in command
    assert any(item.startswith("--tmpfs=/inputs:") and "noexec" in item for item in command)
    assert any(item.startswith("--tmpfs=/runtime:") and "exec" in item for item in command)
    assert any(item.startswith("--tmpfs=/work:") and "exec" in item for item in command)
    assert "--fp16" not in command  # trtexec is invoked only by the pinned worker
    assert command[-1] == lane.ONNX_PINS[640]["sha256"]


def test_gpu_lease_wraps_exactly_one_docker_command(tmp_path: Path) -> None:
    cidfile = tmp_path / "container.cid"
    docker = ["docker", "run", "--rm", f"--cidfile={cidfile}", lane.IMAGE_ID, "true"]
    wrapped = lane.render_lease_command(docker, stage="tensorrt_fp16_640")
    assert wrapped[:4] == [lane.sys.executable, "-m", "validation.gpu_lease", "run"]
    assert wrapped[4:10] == [
        "--owner-kind", "legacy_validation", "--gpu-index", "0",
        "--ttl-seconds", "30",
    ]
    assert wrapped[10:15] == [
        "--timeout-seconds", "14460", "--managed-docker-cidfile", str(cidfile), "--",
    ]
    assert wrapped[-len(docker):] == docker
    with pytest.raises(lane.TensorRTR14CError, match="only docker run"):
        lane.render_lease_command(["nvidia-smi"], stage="tensorrt_fp16_640")


@pytest.mark.parametrize("profile,threshold", [(640, 0.0), (960, 0.375), (640, 1.0)])
def test_deepstream_config_has_exact_no_clustering_and_calibrated_threshold(
    profile: int, threshold: float
) -> None:
    rendered = lane._render_config(profile, threshold).decode("utf-8")
    assert "cluster-mode=4" in rendered
    assert "maintain-aspect-ratio=1" in rendered
    assert "symmetric-padding=1" in rendered
    assert "parse-bbox-func-name=NvDsInferParseCustomRTDETRv4Person" in rendered
    assert "model-engine-file=/work/model.engine" in rendered
    assert "custom-lib-path=/runtime/production-parser.so" in rendered
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
    assert "DeepSafeRTDETRv4ParserParityR14C" in source
    assert "DeepSafeRTDETRv4InitializerParityR14C" in source
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
    with pytest.raises(lane.TensorRTR14CError, match="already exists"):
        lane.atomic_bytes(path, b'{"changed":true}\n')


def test_atomic_copy_is_fd_bound_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.engine"
    source.write_bytes(b"engine-bytes")
    destination = tmp_path / "published.engine"
    lane.atomic_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o440
    with pytest.raises(lane.TensorRTR14CError, match="already exists"):
        lane.atomic_copy(source, destination)


def test_container_lease_environment_rejects_missing_lease(monkeypatch) -> None:
    for name in (
        "DEEPSAFE_GPU_LEASE_HELD", "DEEPSAFE_GPU_LEASE_ID",
        "DEEPSAFE_GPU_LEASE_GPU_INDEX", "DEEPSAFE_GPU_LEASE_GPU_UUID",
        "DEEPSAFE_GPU_LEASE_OWNER_KIND", "DEEPSAFE_GPU_LEASE_COMMAND_SHA256",
        "DEEPSAFE_GPU_LEASE_CONTRACT_SHA256",
        "DEEPSAFE_MANAGED_DOCKER_CIDFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(worker.ContainerR14CError, match="not held"):
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
        "DEEPSAFE_MANAGED_DOCKER_CIDFILE": "/tmp/r14c/control/container.cid",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    observed = worker.lease_environment()
    assert observed["lease_id"] == "a" * 64
    assert observed["managed_docker_cidfile"] == "/tmp/r14c/control/container.cid"
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
    with pytest.raises(lane.TensorRTR14CError, match="count-derived metric"):
        lane.replay_r13b_selection(forged, selected, expected_ground_truth=10)
    wrong = copy.deepcopy(selected)
    wrong["threshold"] = 0.0
    wrong["metrics"] = points[0]["metrics"]
    with pytest.raises(lane.TensorRTR14CError, match="max-F1"):
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
    with pytest.raises(lane.TensorRTR14CError, match="acceptance set"):
        lane._require_accepted_predecessor_commits({}, "tensorrt_fp16_640", {"tensorrt_fp16_640": "a" * 64})
    with pytest.raises(lane.TensorRTR14CError, match="acceptance set"):
        lane._require_accepted_predecessor_commits({}, "numerical_parity_640", {})
    assert lane.parse_commit_acceptances(["tensorrt_fp16_640=" + "b" * 64]) == {"tensorrt_fp16_640": "b" * 64}
    with pytest.raises(lane.TensorRTR14CError, match="commit acceptance"):
        lane.parse_commit_acceptances(["tensorrt_fp16_640=" + "b" * 64, "tensorrt_fp16_640=" + "c" * 64])


def test_successor_commit_rejects_forged_execution_semantic_projection() -> None:
    stage = "tensorrt_fp16_640"
    run_id = "unit-projection"
    plan_pin = {
        "path": lane.PLAN_RELATIVE, "bytes": 1, "sha256": "1" * 64,
        "fingerprint_sha256": "2" * 64,
    }
    r11_pin = {
        "path": lane.profile_paths(640)["engine_receipt_path"],
        "bytes": 1, "sha256": "3" * 64, "fingerprint_sha256": "4" * 64,
    }
    worker_pin = {
        "path": lane.profile_paths(640)["worker_result_path"],
        "bytes": 1, "sha256": "5" * 64,
    }
    execution_fingerprint = "6" * 64
    commit = {
        "plan": plan_pin, "r11_stage_receipt": r11_pin,
        "worker_result": worker_pin,
        "execution_receipt": {
            "path": lane.profile_paths(640)["r14_receipt_path"],
            "bytes": 1, "sha256": "7" * 64,
            "fingerprint_sha256": execution_fingerprint,
        },
    }
    execution = {
        "receipt_id": "rtdetrv4-s-r11-tensorrt-fp16-640-r14c",
        "stage": stage, "profile": 640, "status": "passed",
        "plan": plan_pin, "r11_stage_receipt": r11_pin,
        "worker_result": worker_pin,
        "fingerprint_sha256": execution_fingerprint,
        "command": {
            "managed_docker_cidfile": str(
                lane.RUNS_ROOT / stage / run_id / "control/container.cid"
            ),
        },
    }
    lane._validate_commit_execution_projection(
        commit, execution, stage=stage, run_id=run_id,
    )
    forged_values = []
    for mutate in (
        lambda value: value.update(stage="tensorrt_fp16_960"),
        lambda value: value.update(profile=960),
        lambda value: value.update(status="failed"),
        lambda value: value.update(plan={**plan_pin, "sha256": "8" * 64}),
        lambda value: value.update(r11_stage_receipt={**r11_pin, "sha256": "9" * 64}),
        lambda value: value.update(worker_result={**worker_pin, "sha256": "a" * 64}),
        lambda value: value["command"].update(managed_docker_cidfile="/tmp/forged/container.cid"),
    ):
        forged = copy.deepcopy(execution)
        mutate(forged)
        forged_values.append(forged)
    forged_commit = copy.deepcopy(commit)
    forged_commit["execution_receipt"]["path"] = "forged/execution-r14c.json"
    with pytest.raises(lane.TensorRTR14CError, match="semantic projection"):
        lane._validate_commit_execution_projection(
            forged_commit, execution, stage=stage, run_id=run_id,
        )
    for forged in forged_values:
        with pytest.raises(lane.TensorRTR14CError, match="semantic projection"):
            lane._validate_commit_execution_projection(
                commit, forged, stage=stage, run_id=run_id,
            )


def test_cli_has_no_docker_argv_or_worker_override_surface() -> None:
    assert lane._parser().parse_args(["prepare-failed-attempt"]).command == "prepare-failed-attempt"
    verified = lane._parser().parse_args([
        "verify-failed-attempt", "--accept-receipt-fingerprint", "a" * 64,
    ])
    assert verified.command == "verify-failed-attempt"
    with pytest.raises(SystemExit):
        lane._parser().parse_args(["verify-failed-attempt"])
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


def test_open_held_source_rejects_same_size_mutation_during_hash(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * 8192)
    source_inode = source.stat().st_ino
    real_read = lane.os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        block = real_read(descriptor, count)
        if block and not mutated and os.fstat(descriptor).st_ino == source_inode:
            mutated = True
            source.write_bytes(b"b" * 8192)
        return block

    monkeypatch.setattr(lane.os, "read", mutating_read)
    with pytest.raises(lane.TensorRTR14CError, match="mutated during held-FD read"):
        lane.open_held_source(source)


def test_publish_held_fd_rejects_same_size_mutation_during_copy(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * 8192)
    descriptor, source_pin = lane.open_held_source(source)
    destination = tmp_path / "published.bin"
    expected = {**source_pin, "path": "published.bin"}
    source_inode = source.stat().st_ino
    real_read = lane.os.read
    mutated = False

    def mutating_read(current: int, count: int) -> bytes:
        nonlocal mutated
        block = real_read(current, count)
        if block and not mutated and os.fstat(current).st_ino == source_inode:
            mutated = True
            source.write_bytes(b"b" * 8192)
        return block

    monkeypatch.setattr(lane.os, "read", mutating_read)
    try:
        with pytest.raises(lane.TensorRTR14CError, match="mutated during held-FD read"):
            lane.publish_held_fd(
                descriptor, destination, expected, allow_existing_exact=False,
            )
        assert not destination.exists()
    finally:
        os.close(descriptor)


def test_read_held_json_rejects_same_size_mutation_during_pread(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    source = tmp_path / "receipt.json"
    original = b'{"status":"passed","value":"aaaa"}\n'
    replacement = b'{"status":"passed","value":"bbbb"}\n'
    assert len(original) == len(replacement)
    source.write_bytes(original)
    descriptor, pin = lane.open_held_source(source)
    real_pread = lane.os.pread
    mutated = False

    def mutating_pread(current: int, count: int, offset: int) -> bytes:
        nonlocal mutated
        block = real_pread(current, count, offset)
        if block and not mutated:
            mutated = True
            source.write_bytes(replacement)
        return block

    monkeypatch.setattr(lane.os, "pread", mutating_pread)
    try:
        with pytest.raises(lane.TensorRTR14CError, match="mutated during held-FD read"):
            lane.read_held_json(descriptor, pin, source=str(source))
    finally:
        os.close(descriptor)


def test_held_execution_snapshot_rejects_b1_to_b2_name_swap(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    execution_path = tmp_path / "execution-r14c.json"
    b1 = b'{"semantic":"B1","status":"passed"}\n'
    b2 = b'{"semantic":"B2","status":"passed"}\n'
    assert len(b1) == len(b2)
    execution_path.write_bytes(b1)
    descriptor, pin = lane.open_held_source(execution_path)
    try:
        assert lane.read_held_json(
            descriptor, pin, source=str(execution_path),
        )["semantic"] == "B1"
        execution_path.rename(tmp_path / "execution-b1-moved.json")
        execution_path.write_bytes(b2)
        with pytest.raises(lane.TensorRTR14CError, match="name/inode metadata differs"):
            lane._replay_held_source_pin(
                descriptor, execution_path, pin,
                context="published execution receipt",
            )
    finally:
        os.close(descriptor)


def test_failed_r14b_attempt_receipt_is_exact_and_does_not_invent_argv() -> None:
    receipt, pin = lane.verify_failed_attempt_receipt()
    assert receipt["failure"] == {
        "layer": "runc_oci_bind_mount_setup",
        "docker_exit_code": 125,
        "error_signature": "MS_BIND_MS_REC_invalid_argument",
        "rejected_source": "/proc/2868740/fd/5",
        "rejected_destination": "/runner.py",
        "exact_argv_status": "argv_not_persisted_before_failure",
        "container_process_started": False,
        "worker_started": False,
        "cuda_initialization_reached": False,
        "gpu_workload_started": False,
    }
    assert receipt["lease"]["released_cleanly"] is True
    assert receipt["publication"]["commit_manifest"] is False
    assert pin["fingerprint_sha256"] == receipt["fingerprint_sha256"]


def test_r14c_supersedes_immutable_r14b_and_binds_failure_receipt() -> None:
    plan = lane.build_plan(prepared_at_utc="2026-07-18T08:00:00+00:00")
    assert plan["supersedes"]["release"] == "r14b"
    assert plan["supersedes"]["immutable_artifacts"] == lane.R14B_SUPERSEDED
    assert plan["supersedes"]["execution_reused"] is False
    assert plan["supersedes"]["failed_attempt_receipt"] == lane.document_pin(
        lane.R14B_FAILED_ATTEMPT_RECEIPT
    )
    assert plan["execution_contract"]["input_transport"]["proc_fd_bind_mounts"] == 0
    assert plan["execution_contract"]["docker"]["private_work_tmpfs"] == "/work"
    assert plan["execution_contract"]["worker_output_isolation"]["host_output_role"] == (
        "exclusive_copy_only_never_loaded_executed_or_parsed"
    )
    assert plan["execution_contract"]["worker_output_isolation"]["runtime_engine_ingress"] == (
        "same_held_fd_exact_hash_copy_to_private_work_deserialize_private_only"
    )
    assert plan["execution_contract"]["worker_output_isolation"]["held_source_stability"] == (
        "device_inode_size_mtime_ctime_mode_nlink_before_after_hash_and_copy"
    )


def test_sealed_archive_survives_parent_rename_replacement_and_symlink(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    runner = reviewed / "runner.py"
    model = reviewed / "model.onnx"
    runner.write_bytes(b"reviewed-runner")
    model.write_bytes(b"reviewed-model")
    descriptors: dict[str, int] = {}
    archive_descriptor = -1
    try:
        rows = []
        for archive_name, path in (("runner.py", runner), ("model.onnx", model)):
            descriptor, pin = lane.open_held_source(path)
            descriptors[pin["path"]] = descriptor
            rows.append((archive_name, pin))
        manifest = lane.build_input_manifest(stage="tensorrt_fp16_640", members=rows)
        moved = tmp_path / "reviewed-held"
        reviewed.rename(moved)
        forged = tmp_path / "forged"
        forged.mkdir()
        (forged / "runner.py").write_bytes(b"forged-runner")
        (forged / "model.onnx").write_bytes(b"forged-model")
        reviewed.symlink_to(forged, target_is_directory=True)
        archive_descriptor, evidence = lane.build_sealed_input_archive(
            manifest=manifest, held_sources=descriptors,
        )
        lane.inspect_sealed_input_archive(
            archive_descriptor, manifest=manifest, expected_evidence=evidence,
        )
        with os.fdopen(os.dup(archive_descriptor), "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                assert archive.extractfile("runner.py").read() == b"reviewed-runner"
                assert archive.extractfile("model.onnx").read() == b"reviewed-model"
    finally:
        if archive_descriptor >= 0:
            os.close(archive_descriptor)
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_sealed_archive_rejects_in_place_mutation_after_pin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    model = tmp_path / "model.onnx"
    runner.write_bytes(b"reviewed-runner")
    model.write_bytes(b"reviewed-model")
    descriptors: dict[str, int] = {}
    try:
        rows = []
        for archive_name, path in (("runner.py", runner), ("model.onnx", model)):
            descriptor, pin = lane.open_held_source(path)
            descriptors[pin["path"]] = descriptor
            rows.append((archive_name, pin))
        manifest = lane.build_input_manifest(stage="tensorrt_fp16_640", members=rows)
        model.write_bytes(b"mutated-model!")
        with pytest.raises(lane.TensorRTR14CError, match="held container input pin differs"):
            lane.build_sealed_input_archive(manifest=manifest, held_sources=descriptors)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_sealed_memfd_cannot_be_written_or_resized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    model = tmp_path / "model.onnx"
    runner.write_bytes(b"runner")
    model.write_bytes(b"model")
    descriptors: dict[str, int] = {}
    archive_descriptor = -1
    try:
        rows = []
        for archive_name, path in (("runner.py", runner), ("model.onnx", model)):
            descriptor, pin = lane.open_held_source(path)
            descriptors[pin["path"]] = descriptor
            rows.append((archive_name, pin))
        manifest = lane.build_input_manifest(stage="tensorrt_fp16_640", members=rows)
        archive_descriptor, _evidence = lane.build_sealed_input_archive(
            manifest=manifest, held_sources=descriptors,
        )
        assert fcntl.fcntl(archive_descriptor, fcntl.F_GET_SEALS) == lane.MEMFD_REQUIRED_SEALS
        with pytest.raises(PermissionError):
            os.write(archive_descriptor, b"x")
        with pytest.raises(PermissionError):
            os.ftruncate(archive_descriptor, 0)
    finally:
        if archive_descriptor >= 0:
            os.close(archive_descriptor)
        for descriptor in descriptors.values():
            os.close(descriptor)


@pytest.mark.parametrize("attack", ["traversal", "symlink", "hardlink", "device", "duplicate", "size"])
def test_input_archive_inspector_rejects_unsafe_tar_headers(attack: str) -> None:
    runner_payload = b"runner"
    model_payload = b"model"
    manifest = lane.build_input_manifest(
        stage="tensorrt_fp16_640",
        members=[
            ("runner.py", {"path": "runner.py", "bytes": len(runner_payload), "sha256": hashlib.sha256(runner_payload).hexdigest()}),
            ("model.onnx", {"path": "model.onnx", "bytes": len(model_payload), "sha256": hashlib.sha256(model_payload).hexdigest()}),
        ],
    )
    runner_info = lane._tar_info("runner.py", len(runner_payload))
    model_info = lane._tar_info("model.onnx", len(model_payload))
    members: list[tuple[tarfile.TarInfo, bytes | None]]
    if attack == "traversal":
        runner_info.name = "../runner.py"
        members = [(runner_info, runner_payload), (model_info, model_payload)]
    elif attack in {"symlink", "hardlink", "device"}:
        runner_info.type = {
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "device": tarfile.CHRTYPE,
        }[attack]
        runner_info.size = 0
        runner_info.linkname = "model.onnx" if attack != "device" else ""
        members = [(runner_info, None), (model_info, model_payload)]
    elif attack == "duplicate":
        members = [
            (runner_info, runner_payload),
            (lane._tar_info("runner.py", len(runner_payload)), runner_payload),
            (model_info, model_payload),
        ]
    else:
        runner_info.size = len(runner_payload) + 1
        members = [(runner_info, runner_payload + b"x"), (model_info, model_payload)]
    descriptor = _seal_bytes(_tar_payload(manifest, members))
    try:
        with pytest.raises(lane.TensorRTR14CError):
            lane.inspect_sealed_input_archive(descriptor, manifest=manifest)
    finally:
        os.close(descriptor)


def test_bootstrap_is_manifest_first_strict_bounded_and_execs_only_pinned_runner() -> None:
    source = lane.CONTAINER_INPUT_BOOTSTRAP
    for token in (
        'header(first,"input-manifest.json")',
        "duplicate_json_key", "nonfinite", "member_extensions",
        "member_metadata", "duplicate_or_extra", "member_overflow",
        'seen!=set(expected)', 'os.execv(sys.executable',
    ):
        assert token in source


def test_worker_generated_code_and_logs_stay_private_until_copy() -> None:
    source = lane.CONTAINER_WORKER.read_text(encoding="utf-8")
    assert 'PRIVATE_WORK = Path("/work")' in source
    assert 'PUBLIC_OUTPUT = Path("/output")' in source
    assert "publish_private_outputs" in source
    assert source.count("PUBLIC_OUTPUT /") == 2  # copy destination and final worker receipt only
    assert 'engine = PRIVATE_WORK / "engine.staging"' in source
    assert 'library = PRIVATE_WORK / "libparser-parity-r14c.so"' in source
    assert 'smoke_binary = PRIVATE_WORK / "gst-nvinfer-smoke-r14c"' in source
    assert "materialize_private_input" in source
    assert 'PRIVATE_WORK / "model.engine"' in source
    assert '"model-engine-file": "/work/model.engine"' in source
    assert "load_engine(args.engine)" not in source
    assert "ctypes.CDLL(str(library)" in source
    assert "load_engine(PUBLIC_OUTPUT" not in source
    assert "ctypes.CDLL(str(PUBLIC_OUTPUT" not in source


def test_private_output_copy_is_exclusive_and_source_fd_bound(monkeypatch, tmp_path: Path) -> None:
    private = tmp_path / "work"
    public = tmp_path / "output"
    private.mkdir()
    public.mkdir()
    monkeypatch.setattr(worker, "PRIVATE_WORK", private)
    monkeypatch.setattr(worker, "PUBLIC_OUTPUT", public)
    generated = private / "generated.bin"
    generated.write_bytes(b"validated-private-output")
    pins = worker.publish_private_outputs([generated])
    assert pins == [{
        "path": "generated.bin", "bytes": len(b"validated-private-output"),
        "sha256": hashlib.sha256(b"validated-private-output").hexdigest(),
    }]
    assert (public / "generated.bin").read_bytes() == b"validated-private-output"
    with pytest.raises(FileExistsError):
        worker.publish_private_outputs([generated])


def test_engine_input_is_materialized_from_same_held_fd_into_private_work(
    monkeypatch, tmp_path: Path,
) -> None:
    private = tmp_path / "work"
    private.mkdir()
    monkeypatch.setattr(worker, "PRIVATE_WORK", private)
    ingress = tmp_path / "inputs"
    ingress.mkdir()
    source = ingress / "model.engine"
    source.write_bytes(b"reviewed-engine")
    expected = hashlib.sha256(b"reviewed-engine").hexdigest()
    destination = private / "model.engine"
    observed = worker.materialize_private_input(source, expected, destination)
    assert observed == destination
    assert destination.read_bytes() == b"reviewed-engine"
    assert destination.stat().st_mode & 0o777 == 0o440
    with pytest.raises(FileExistsError):
        worker.materialize_private_input(source, expected, destination)


def test_commit_last_partial_recovery_fills_exact_members(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane, "ROOT", tmp_path)
    roles = ("artifact", "worker_result", "r11_stage_receipt", "execution_receipt")
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
    with pytest.raises(lane.TensorRTR14CError, match="timed out"):
        lane._run_bounded_process_group(
            command, log_path=tmp_path / "timeout.log", timeout_seconds=1,
            output_limit_bytes=4096,
        )
    assert (tmp_path / "timeout.log").read_text().startswith("ready")


def test_process_group_output_is_bounded_and_fails_closed(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; sys.stdout.write('x'*8192); sys.stdout.flush()"]
    with pytest.raises(lane.TensorRTR14CError, match="output exceeded"):
        lane._run_bounded_process_group(
            command, log_path=tmp_path / "bounded.log", timeout_seconds=10,
            output_limit_bytes=1024,
        )
    assert (tmp_path / "bounded.log").stat().st_size == 1024


def test_host_output_drain_write_failure_terminates_child_and_fails_closed(
    monkeypatch, tmp_path: Path,
) -> None:
    log = tmp_path / "drain-error.log"
    real_write = os.write

    def fail_log_write(descriptor: int, payload: bytes | memoryview) -> int:
        if log.exists() and os.fstat(descriptor).st_ino == log.stat().st_ino:
            raise OSError("synthetic log sink failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(lane.os, "write", fail_log_write)
    started = time.monotonic()
    with pytest.raises(lane.TensorRTR14CError, match="output drain failed"):
        lane._run_bounded_process_group(
            [sys.executable, "-c", "import sys,time; print('x'*131072,flush=True); time.sleep(30)"],
            log_path=log, timeout_seconds=20, output_limit_bytes=1024 * 1024,
        )
    assert time.monotonic() - started < 10


def test_container_output_drain_write_failure_terminates_child_and_fails_closed(
    monkeypatch, tmp_path: Path,
) -> None:
    log = tmp_path / "container-drain-error.log"
    real_write = os.write

    def fail_log_write(descriptor: int, payload: bytes | memoryview) -> int:
        if log.exists() and os.fstat(descriptor).st_ino == log.stat().st_ino:
            raise OSError("synthetic container sink failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(worker.os, "write", fail_log_write)
    started = time.monotonic()
    with pytest.raises(worker.ContainerR14CError, match="output drain failed"):
        worker._run_checked(
            [sys.executable, "-c", "import sys,time; print('x'*131072,flush=True); time.sleep(30)"],
            stdout_path=log, timeout_seconds=20, output_limit_bytes=1024 * 1024,
        )
    assert time.monotonic() - started < 10


def test_worker_schema_and_commit_schemas_are_strict() -> None:
    for path in (
        lane.PLAN_SCHEMA, lane.WORKER_SCHEMA, lane.COMMIT_SCHEMA,
        lane.INTENT_SCHEMA, lane.RECEIPT_SCHEMA,
        lane.INPUT_MANIFEST_SCHEMA, lane.FAILED_ATTEMPT_SCHEMA,
    ):
        schema = lane.load_json(path)
        assert schema["additionalProperties"] is False
    commit = {
        "schema_version": lane.COMMIT_SCHEMA_VERSION, "commit_id": "unit-commit-r14c",
        "status": "committed", "stage": "tensorrt_fp16_640", "profile": 640,
        "run_id": "unit-run", "committed_at_utc": "2026-07-18T00:00:00+00:00",
        "plan": {"path": "plan.json", "bytes": 1, "sha256": "a" * 64, "fingerprint_sha256": "b" * 64},
        "publication_intent": {"path": "intent.json", "bytes": 1, "sha256": "c" * 64, "fingerprint_sha256": "d" * 64},
        "artifact": {"path": "artifact", "bytes": 1, "sha256": "e" * 64},
        "r11_stage_receipt": {"path": "r11.json", "bytes": 1, "sha256": "f" * 64, "fingerprint_sha256": "1" * 64},
        "execution_receipt": {"path": "execution.json", "bytes": 1, "sha256": "2" * 64, "fingerprint_sha256": "3" * 64},
        "worker_result": {"path": "worker.json", "bytes": 1, "sha256": "4" * 64},
        "publication": {"held_fd_sources": True, "otmpfile_linkat_noreplace": True, "bundle_members_verified_before_commit": True, "bundle_member_roles": ["artifact", "worker_result", "r11_stage_receipt", "execution_receipt"], "commit_manifest_last": True, "partial_recovery_supported": True, "overwrite": False},
        "claim_boundary": {"quality": False, "exact_25m": False, "twelve_camera_capacity": False, "three_module_full_stack": False, "production_ready": False},
        "fingerprint_sha256": "5" * 64,
    }
    lane.validate_schema(commit, lane.COMMIT_SCHEMA)
    forged = copy.deepcopy(commit)
    forged["publication"]["commit_manifest_last"] = False
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
        lane.validate_schema(forged, lane.COMMIT_SCHEMA)


def test_execution_receipt_schema_binds_stdin_work_cid_and_nested_timeouts(tmp_path: Path) -> None:
    stage = "tensorrt_fp16_640"
    plan_fingerprint = "1" * 64
    manifest = lane.build_input_manifest(
        stage=stage,
        members=[
            ("runner.py", {"path": "runner.py", "bytes": 1, "sha256": "2" * 64}),
            ("model.onnx", {"path": "model.onnx", "bytes": 1, "sha256": "3" * 64}),
        ],
    )
    output = tmp_path / "output"
    output.mkdir()
    cidfile = tmp_path / "control" / "container.cid"
    cidfile.parent.mkdir(mode=0o700)
    docker = lane.render_docker_command(
        plan={"fingerprint_sha256": plan_fingerprint}, stage=stage,
        output_dir=output, cidfile=cidfile, input_manifest=manifest,
    )
    leased = lane.render_lease_command(docker, stage=stage)
    gpu_uuid = "GPU-8cbaba1c-2629-a732-f528-66f459089ef6"
    sample = {
        "monotonic_seconds": 0.0, "gpu_uuid": gpu_uuid,
        "utilization_gpu_percent": 0.0, "memory_used_mib": 1.0,
        "temperature_c": 40.0, "power_w": 1.0,
        "graphics_clock_mhz": 1.0, "memory_clock_mhz": 1.0,
    }
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
    command_digest = lane.gpu_lease_command_sha256(docker)
    worker_value = {
        "validation": {
            "optimization_profiles": shapes, "bindings": bindings,
            "engine_deserialized": True, "trtexec_exit_code": 0,
        },
        "gpu": {"uuid": gpu_uuid, "name": "NVIDIA RTX A5000 Laptop GPU", "compute_capability": "8.6"},
        "lease": {"lease_id": "4" * 64, "command_sha256": command_digest},
        "telemetry": {
            "sample_interval_seconds": 1.0,
            "samples": [sample, {**sample, "monotonic_seconds": 1.0}],
            "all_samples_same_gpu": True, "sample_errors": [],
        },
    }
    document = lambda path, digest, fingerprint: {
        "path": path, "bytes": 1, "sha256": digest * 64,
        "fingerprint_sha256": fingerprint * 64,
    }
    plan = {
        "fingerprint_sha256": plan_fingerprint,
        "execution_contract": {"global_gpu_lease": {"contract": document("lease-contract.json", "5", "6")}},
    }
    lifecycle = {
        "start_new_session": True, "timeout_seconds": 14520,
        "output_limit_bytes": lane.HOST_PROCESS_OUTPUT_LIMIT_BYTES,
        "output_truncated": False, "signal_received": None,
        "cleanup": {"attempted": False, "container_id": None, "removed": False},
        "orphan_survived": False,
    }
    transport = {
        "kind": "deterministic_ustar_in_sealed_memfd", "bytes": 10240,
        "sha256": "7" * 64, "manifest_sha256": "8" * 64,
        "manifest_fingerprint_sha256": manifest["fingerprint_sha256"],
        "member_count": 2,
        "seals": ["F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL"],
        "docker_stdin": True, "host_path": False,
    }
    receipt = lane.build_r14_receipt(
        plan=plan, stage=stage,
        started_at_utc="2026-07-18T00:00:00+00:00",
        completed_at_utc="2026-07-18T00:00:01+00:00", duration_seconds=1.0,
        r11_pin=document("r11.json", "9", "a"),
        prior_receipts=[document("onnx-640.json", "b", "c"), document("onnx-960.json", "d", "e")],
        input_pins=[{"path": "runner.py", "bytes": 1, "sha256": "2" * 64}, {"path": "model.onnx", "bytes": 1, "sha256": "3" * 64}],
        output_pins=[{"path": "engine", "bytes": 1, "sha256": "f" * 64}],
        docker_command=docker, leased_command=leased,
        container_id="0" * 64, worker=worker_value,
        acquire_pin=document("acquire.json", "1", "2"),
        release_pin=document("release.json", "3", "4"),
        threshold_pin=document("threshold.json", "5", "6"),
        config_pin=None, numerical=None,
        input_manifest_pin={
            "path": "input-manifest.json", "bytes": 1, "sha256": "7" * 64,
            "fingerprint_sha256": manifest["fingerprint_sha256"],
        },
        input_transport=transport, process_lifecycle=lifecycle,
        worker_pin={"path": "worker.json", "bytes": 1, "sha256": "8" * 64},
        plan_file_pin={"path": "plan.json", "bytes": 1, "sha256": "9" * 64},
    )
    assert receipt["command"]["input_transport"]["work_tmpfs"] == "/work"
    assert receipt["command"]["managed_docker_cidfile"] == str(cidfile)
    assert receipt["command"]["stage_timeout_seconds"] == 14400
    assert receipt["command"]["lease_timeout_seconds"] == 14460
    assert receipt["command"]["timeout_seconds"] == 14520


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
        "lease": {"held": "1", "lease_id": "b" * 64, "gpu_index": "0", "gpu_uuid": gpu_uuid, "owner_kind": "legacy_validation", "command_sha256": "c" * 64, "contract_sha256": "d" * 64, "managed_docker_cidfile": str(tmp_path / "control/container.cid")},
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
    with pytest.raises(lane.TensorRTR14CError, match="schema validation failed"):
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
    output = run_root / "output"
    output.mkdir()

    real_validate_schema = lane.validate_schema

    def validate_recovery_fixture(value, schema_path):
        if schema_path in {lane.RECEIPT_SCHEMA, lane.R11_EVIDENCE_SCHEMA}:
            return None
        return real_validate_schema(value, schema_path)

    monkeypatch.setattr(lane, "validate_schema", validate_recovery_fixture)
    monkeypatch.setattr(lane, "_validate_worker_result", lambda *args, **kwargs: {"status": "passed"})

    artifact_source_path = control / "engine.staging"
    artifact_source_path.write_bytes(b"engine")
    worker_source_path = output / "worker-result.json"
    worker_source_path.write_bytes(b'{"status":"passed"}\n')
    r11_source_path = control / "publish-r11.json"
    r11_value = {"kind": "unit-recovery", "stage": stage, "status": "passed"}
    r11_value["fingerprint_sha256"] = lane.fingerprint(r11_value)
    r11_source_path.write_bytes(lane._json_payload(r11_value))
    artifact_source = lane.file_pin(artifact_source_path)
    worker_source = lane.file_pin(worker_source_path)
    r11_source = lane.file_pin(r11_source_path)
    artifact_destination = lane._stage_final_artifact(stage)
    worker_destination = lane._stage_worker_result_path(stage)
    r11_destination, execution_destination = lane._stage_receipt_paths(stage)
    artifact_published = {**artifact_source, "path": lane.repo_relative(artifact_destination)}
    worker_published = {**worker_source, "path": lane.repo_relative(worker_destination)}
    r11_published = {
        **r11_source, "path": lane.repo_relative(r11_destination),
        "fingerprint_sha256": r11_value["fingerprint_sha256"],
    }
    execution_source_path = control / "publish-execution-r14c.json"
    execution_value = {
        "stage": stage, "profile": 640, "status": "passed",
        "plan": {**lane.file_pin(plan_path), "fingerprint_sha256": plan["fingerprint_sha256"]},
        "r11_stage_receipt": r11_published,
        "worker_result": worker_published,
        "command": {"managed_docker_cidfile": str(control / "container.cid")},
    }
    execution_value["fingerprint_sha256"] = lane.fingerprint(execution_value)
    execution_source_path.write_bytes(lane._json_payload(execution_value))
    execution_source = lane.file_pin(execution_source_path)
    execution_published = {
        **execution_source, "path": lane.repo_relative(execution_destination),
        "fingerprint_sha256": execution_value["fingerprint_sha256"],
    }
    intent = lane.build_publication_intent(
        plan=plan, stage=stage, run_id=run_id,
        artifact_source=artifact_source, artifact_published=artifact_published,
        worker_source=worker_source, worker_published=worker_published,
        r11_source=r11_source, r11_published=r11_published,
        execution_source=execution_source, execution_published=execution_published,
        prepared_at_utc="2026-07-18T00:00:00+00:00",
    )
    intent_path = control / "publication-intent-r14c.json"
    lane.atomic_json(intent_path, intent)
    commit = lane.build_commit_manifest(
        plan=plan, stage=stage, run_id=run_id,
        intent_pin=lane.document_pin(intent_path), artifact_pin=artifact_published,
        r11_pin=r11_published, execution_pin=execution_published,
        worker_pin=worker_published, committed_at_utc="2026-07-18T00:00:00+00:00",
    )
    commit_source_path = control / "publish-commit-r14c.json"
    lane.atomic_json(commit_source_path, commit)

    def validate_recovered_commit(current_plan, current_stage, accepted_fingerprint):
        path = lane._stage_commit_path(current_stage)
        return lane.load_json(path), lane.document_pin(path)

    monkeypatch.setattr(lane, "_validate_commit_manifest", validate_recovered_commit)
    descriptor, _pin = lane.open_held_source(artifact_source_path)
    try:
        lane.publish_held_fd(descriptor, artifact_destination, artifact_published, allow_existing_exact=False)
    finally:
        os.close(descriptor)
    with pytest.raises(lane.TensorRTR14CError, match="external recovery intent"):
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
    assert worker_destination.exists() and r11_destination.exists() and execution_destination.exists()
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
    archive_descriptor = -1
    try:
        pins: dict[str, dict[str, object]] = {}
        for path in (runner, engine, bundle):
            descriptor, pin = lane.open_held_source(path)
            descriptors[pin["path"]] = descriptor
            pins[path.name] = pin
        manifest = lane.build_input_manifest(
            stage="numerical_parity_640",
            members=[
                ("runner.py", pins[runner.name]),
                ("model.engine", pins[engine.name]),
                ("parity-input.npz", pins[bundle.name]),
            ],
        )
        for path in (engine, bundle):
            path.rename(path.with_suffix(path.suffix + ".reviewed"))
            path.write_bytes(b"forged-name-content")
        archive_descriptor, evidence = lane.build_sealed_input_archive(
            manifest=manifest, held_sources=descriptors,
        )
        assert lane.inspect_sealed_input_archive(
            archive_descriptor, manifest=manifest, expected_evidence=evidence,
        ) == evidence
        command = lane.render_docker_command(
            plan={"fingerprint_sha256": "a" * 64},
            stage="numerical_parity_640", output_dir=output,
            cidfile=tmp_path / "container.cid", input_bundle=bundle,
            input_manifest=manifest,
        )
        assert hashlib.sha256(b"reviewed-engine").hexdigest() in command
        assert hashlib.sha256(b"reviewed-input").hexdigest() in command
        assert hashlib.sha256(b"forged-name-content").hexdigest() not in command
        with os.fdopen(os.dup(archive_descriptor), "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                assert archive.extractfile("model.engine").read() == b"reviewed-engine"
                assert archive.extractfile("parity-input.npz").read() == b"reviewed-input"
    finally:
        if archive_descriptor >= 0:
            os.close(archive_descriptor)
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
    with pytest.raises(lane.TensorRTR14CError, match="command digest"):
        lane._validate_lease_event(
            forged, allowed_types={"acquire"}, lease_id=lease_id,
            command_digest=command_digest, gpu_uuid=gpu_uuid,
            contract_fingerprint=contract,
        )
