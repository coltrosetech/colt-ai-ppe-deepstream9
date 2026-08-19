from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from validation import person_rtdetrv4_tensorrt_r14 as lane
from validation import person_rtdetrv4_tensorrt_r14_container as worker


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    assert lane.canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(lane.TensorRTR14Error, match="non-finite"):
        lane.canonical_bytes({"bad": math.nan})


def test_strict_json_rejects_duplicate_and_nan() -> None:
    with pytest.raises(lane.TensorRTR14Error, match="duplicate JSON key"):
        lane._strict_json_bytes(b'{"a":1,"a":2}', source="unit")
    with pytest.raises(lane.TensorRTR14Error, match="non-finite"):
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
    with pytest.raises(lane.TensorRTR14Error, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["real_image_selection"]["images"] = 23
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.TensorRTR14Error, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def test_threshold_missing_fails_before_any_subprocess_or_gpu_probe(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-threshold-receipt.json"
    monkeypatch.setattr(lane, "THRESHOLD_RECEIPT", missing)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not be reached")

    monkeypatch.setattr(lane.subprocess, "run", forbidden)
    with pytest.raises(lane.TensorRTR14Error, match="before Docker/GPU probe"):
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
    with pytest.raises(lane.TensorRTR14Error, match="only docker run"):
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
    assert "DeepSafeRTDETRv4ParserParityR14" in source
    assert "DeepSafeRTDETRv4InitializerParityR14" in source
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
    with pytest.raises(lane.TensorRTR14Error, match="already exists"):
        lane.atomic_bytes(path, b'{"changed":true}\n')


def test_atomic_copy_is_fd_bound_and_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.engine"
    source.write_bytes(b"engine-bytes")
    destination = tmp_path / "published.engine"
    lane.atomic_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o440
    with pytest.raises(lane.TensorRTR14Error, match="already exists"):
        lane.atomic_copy(source, destination)


def test_container_lease_environment_rejects_missing_lease(monkeypatch) -> None:
    for name in (
        "DEEPSAFE_GPU_LEASE_HELD", "DEEPSAFE_GPU_LEASE_ID",
        "DEEPSAFE_GPU_LEASE_GPU_INDEX", "DEEPSAFE_GPU_LEASE_GPU_UUID",
        "DEEPSAFE_GPU_LEASE_OWNER_KIND", "DEEPSAFE_GPU_LEASE_COMMAND_SHA256",
        "DEEPSAFE_GPU_LEASE_CONTRACT_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(worker.ContainerR14Error, match="not held"):
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
