from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_shape_diag_r11 as host
from validation import pose_mmpose_yoloxpose_shape_diag_r11_container as container


SHA = "0" * 64


def _dimension(kind: str, value):
    return {"kind": kind, "value": value}


def _value_info(name: str, dimensions: list[dict]):
    return {
        "name": name,
        "dtype": "FLOAT",
        "rank": len(dimensions),
        "dimensions": dimensions,
    }


def _effective_postprocess() -> dict:
    return {
        "score_threshold": 0.01,
        "score_threshold_source": "model.model.test_cfg.score_thr",
        "iou_threshold": 0.65,
        "iou_threshold_source": "model.model.test_cfg.nms_thr",
        "max_output_boxes_per_class": 200,
        "max_output_boxes_per_class_source": (
            "deploy.codebase_config.post_processing.max_output_boxes_per_class"
        ),
        "pre_top_k": 5000,
        "pre_top_k_source": "deploy.codebase_config.post_processing.pre_top_k",
        "keep_top_k": 100,
        "keep_top_k_source": (
            "deploy.codebase_config.post_processing.keep_top_k_fallback"
        ),
    }


def _classification_inputs(k1: int = 2, k12: int = 13):
    graph = {
        "checker_passed": True,
        "external_data": False,
        "raw_inputs": [
            _value_info(
                "input",
                [
                    _dimension("dim_param", "batch"),
                    _dimension("dim_value", 3),
                    _dimension("dim_value", 640),
                    _dimension("dim_value", 640),
                ],
            )
        ],
        "raw_outputs": [
            _value_info(
                "dets",
                [
                    _dimension("dim_param", "batch"),
                    _dimension("dim_param", "TopK_axis"),
                    _dimension("dim_value", 5),
                ],
            ),
            _value_info(
                "keypoints",
                [
                    _dimension("dim_param", "batch"),
                    _dimension("dim_param", "TopK_axis"),
                    _dimension("dim_value", 17),
                    _dimension("dim_value", 3),
                ],
            ),
        ],
        "topk_nodes": [
            {
                "node_index": 42,
                "k_provenance": {
                    "integer_values": [0, 1, 100],
                    "operator_types": ["Gather", "Less", "Reshape", "Shape", "Where"],
                    "truncated": False,
                },
            }
        ],
        "semantic_lineage": {"passed": True},
    }
    runtime = {
        "passed": True,
        "batches": {
            "1": {
                "outputs": [
                    {"name": "dets", "shape": [1, k1, 5], "dtype": "float32", "finite": True},
                    {"name": "keypoints", "shape": [1, k1, 17, 3], "dtype": "float32", "finite": True},
                ]
            },
            "12": {
                "outputs": [
                    {"name": "dets", "shape": [12, k12, 5], "dtype": "float32", "finite": True},
                    {"name": "keypoints", "shape": [12, k12, 17, 3], "dtype": "float32", "finite": True},
                ]
            },
        },
    }
    return graph, runtime, {"effective": _effective_postprocess()}


def test_contract_and_in_memory_plan_validate_without_docker(monkeypatch):
    monkeypatch.setattr(
        host.r10.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Docker/subprocess forbidden")),
    )
    contract = host.strict_json(host.CONTRACT)
    jsonschema.Draft202012Validator(host.strict_json(host.CONTRACT_SCHEMA)).validate(
        contract
    )
    host.validate_contract(contract)
    plan = host.build_plan(created_at="2026-07-18T00:00:00+00:00")
    jsonschema.Draft202012Validator(host.strict_json(host.PLAN_SCHEMA)).validate(plan)
    replay = host.verify_plan(plan, plan["plan_fingerprint_sha256"])
    assert replay["valid"] is True
    assert replay["docker_queried"] is False
    assert len(plan["inputs"]) == 24


def test_pinned_r10_failure_is_replayed_and_immutable():
    observed = host.validate_r10_failure()
    assert observed["failed_run_id"] == "cpu-export-001"
    assert observed["profile_640_attempted"] is True
    assert observed["profile_960_attempted"] is False
    assert observed["shape_error_marker_observed"] is True
    assert host.sha256_file(host.R10_PLAN) == host.R10_PLAN_FILE_SHA256
    assert host.sha256_file(host.R10_FAILED_RECEIPT) == host.R10_FAILED_RECEIPT_SHA256
    assert host.sha256_file(host.R10_FAILED_LOG) == host.R10_FAILED_LOG_SHA256


def test_docker_command_is_cpu_network_none_and_exact_image():
    command = host.docker_command(
        "/new/output",
        "1" * 64,
        "cpu-shape-diag-001",
        "1000:1000",
        7,
        11,
        "2" * 64,
        123,
        "3" * 64,
        456,
        "4" * 64,
        789,
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network=none" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert host.IMAGE_ID in command
    assert "--gpus" not in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "--output-device" in command and "--output-inode" in command


def test_effective_postprocess_uses_mmpose_thresholds_and_deploy_topk():
    pipeline = [{"type": "LoadImage"}, {"type": "BottomupResize", "input_size": (1, 1)}]
    model_cfg = {
        "input_size": (1, 1),
        "codec": {"input_size": (1, 1)},
        "test_dataloader": {"dataset": {"pipeline": copy.deepcopy(pipeline)}},
        "val_dataloader": {"dataset": {"pipeline": copy.deepcopy(pipeline)}},
        "model": {"test_cfg": {"score_thr": 0.01, "nms_thr": 0.65}},
    }
    deploy_cfg = {
        "onnx_config": {"input_shape": [1, 1], "save_file": "old.onnx"},
        "codebase_config": {
            "post_processing": {
                "score_threshold": 0.05,
                "iou_threshold": 0.5,
                "max_output_boxes_per_class": 200,
                "pre_top_k": 5000,
                "keep_top_k": 100,
                "background_label_id": -1,
            }
        },
    }
    profile, post = container.apply_profile_configuration(deploy_cfg, model_cfg)
    assert profile["deploy_onnx_input_shape"] == [640, 640]
    assert post["deploy_defaults"]["score_threshold"] == 0.05
    assert post["model_test_cfg"]["max_per_img_present"] is False
    assert post["effective"] == _effective_postprocess()


def test_dynamic_shape_classification_accepts_shared_bounded_variable_k():
    graph, runtime, post = _classification_inputs(k1=2, k12=13)
    result = container.classify_observation(graph, runtime, post)
    assert result["classification"] == "bounded_dynamic_topk_instance_axis_candidate"
    assert result["evidence"]["runtime_per_batch"]["1"]["dets_k"] == 2
    assert result["evidence"]["runtime_per_batch"]["12"]["dets_k"] == 13
    assert result["derived_shape_interface"]["k_formula_from_pinned_source"] == "K=min(100,M+1)"
    assert result["contract_change_authorized"] is False


def test_static_k100_requires_raw_metadata_and_both_runtime_batches():
    graph, runtime, post = _classification_inputs(k1=100, k12=100)
    for output in graph["raw_outputs"]:
        output["dimensions"][1] = _dimension("dim_value", 100)
    graph["topk_nodes"] = []
    graph["semantic_lineage"] = {"passed": False}
    result = container.classify_observation(graph, runtime, post)
    assert result["classification"] == "r10_expected_raw_metadata_observed"
    assert result["evidence"]["runtime_batch1_and_batch12_static_k100"] is True
    assert result["contract_change_authorized"] is False


@pytest.mark.parametrize("mutation", ["missing_where", "k_above_bound", "split_k", "wrong_threshold"])
def test_dynamic_shape_classification_fails_closed_on_unproven_contract(mutation):
    graph, runtime, post = _classification_inputs()
    if mutation == "missing_where":
        graph["topk_nodes"][0]["k_provenance"]["operator_types"].remove("Where")
    elif mutation == "k_above_bound":
        runtime["batches"]["12"]["outputs"][0]["shape"][1] = 101
        runtime["batches"]["12"]["outputs"][1]["shape"][1] = 101
    elif mutation == "split_k":
        runtime["batches"]["1"]["outputs"][1]["shape"][1] = 3
    else:
        post["effective"]["score_threshold"] = 0.05
    result = container.classify_observation(graph, runtime, post)
    assert result["classification"] == "other_shape_observation_fail_closed"
    assert result["unexpected_shapes_fail_closed"] is True
    assert result["contract_change_authorized"] is False


def test_topk_upstream_provenance_is_name_independent():
    def node(name, op_type, inputs, outputs):
        return SimpleNamespace(name=name, op_type=op_type, domain="", input=inputs, output=outputs)

    nodes = [
        node("shape-random", "Shape", ["topk_data"], ["shape_value"]),
        node("axis-random", "Gather", ["shape_value", "axis"], ["size"]),
        node("less-random", "Less", ["hundred", "size"], ["condition"]),
        node("where-random", "Where", ["condition", "hundred", "size"], ["bounded"]),
        node("reshape-random", "Reshape", ["bounded", "one"], ["topk_k"]),
    ]
    producers = {
        output: (index, item)
        for index, item in enumerate(nodes)
        for output in item.output
    }
    constants = {
        "hundred": {"source": "Constant", "dtype": "int64", "shape": [], "values": [100]},
        "axis": {"source": "Constant", "dtype": "int64", "shape": [], "values": [1]},
        "one": {"source": "Constant", "dtype": "int64", "shape": [1], "values": [1]},
    }
    provenance = container._input_provenance("topk_k", producers, constants)
    assert provenance["truncated"] is False
    assert 100 in provenance["integer_values"]
    assert {"Shape", "Gather", "Less", "Where", "Reshape"}.issubset(
        provenance["operator_types"]
    )


def test_nms_topk_indices_reach_both_outputs():
    def node(name, op_type, inputs, outputs):
        return SimpleNamespace(name=name, op_type=op_type, domain="", input=inputs, output=outputs)

    nodes = [
        node("n", "NonMaxSuppression", ["boxes", "scores"], ["nms_indices"]),
        node("pad", "Gather", ["nms_indices"], ["topk_data"]),
        node("tk", "TopK", ["topk_data", "topk_k"], ["values", "shared_indices"]),
        node("d", "Gather", ["shared_indices"], ["dets"]),
        node("k", "Gather", ["shared_indices"], ["keypoints"]),
    ]
    model = SimpleNamespace(graph=SimpleNamespace(node=nodes))
    topk = [{
        "node_index": 2,
        "inputs": ["topk_data", "topk_k"],
        "outputs": ["values", "shared_indices"],
        "k_provenance": {
            "integer_values": [100],
            "operator_types": ["Shape", "Gather", "Less", "Where"],
            "truncated": False,
        },
    }]
    nms = [{"node_index": 0, "outputs": ["nms_indices"]}]
    lineage = container._semantic_nms_topk_lineage(model, topk, nms)
    assert lineage["passed"] is True
    assert lineage["passing_candidate_count"] == 1


def _synthetic_profile_receipt(output: Path, plan_fp: str, run_id: str) -> dict:
    onnx_bytes = b"quarantined synthetic graph bytes"
    onnx_path = output / host.OUTPUT_NAME
    onnx_path.write_bytes(onnx_bytes)
    onnx_path.chmod(0o440)
    dimensions_input = [
        _dimension("dim_param", "batch"), _dimension("dim_value", 3),
        _dimension("dim_value", 640), _dimension("dim_value", 640),
    ]
    dimensions_dets = [
        _dimension("dim_param", "batch"), _dimension("dim_param", "K"),
        _dimension("dim_value", 5),
    ]
    dimensions_kpts = [
        _dimension("dim_param", "batch"), _dimension("dim_param", "K"),
        _dimension("dim_value", 17), _dimension("dim_value", 3),
    ]
    runtime_batches = {}
    for batch, k in ((1, 2), (12, 13)):
        runtime_batches[str(batch)] = {
            "status": "passed",
            "probe": "deterministic_blank_zero_tensor",
            "input_shape": [batch, 3, 640, 640],
            "outputs": [
                {"name": "dets", "shape": [batch, k, 5], "dtype": "float32", "finite": True, "minimum": 0.0, "maximum": 0.9},
                {"name": "keypoints", "shape": [batch, k, 17, 3], "dtype": "float32", "finite": True, "minimum": 0.0, "maximum": 0.8},
            ],
            "all_outputs_finite": True,
            "error": None,
        }
    origin_sources = {
        "mmdeploy.pytorch.functions.topk": (host.MMDEPLOY_TOPK, "/opt/src/mmdeploy/mmdeploy/pytorch/functions/topk.py"),
        "mmdeploy.mmcv.ops.nms": (host.MMDEPLOY_NMS, "/opt/src/mmdeploy/mmdeploy/mmcv/ops/nms.py"),
        "mmdeploy.codebase.mmpose.models.heads.yolox_pose_head": (host.MMDEPLOY_REWRITE, "/opt/src/mmdeploy/mmdeploy/codebase/mmpose/models/heads/yolox_pose_head.py"),
        "mmpose.models.pose_estimators.base": (host.MMPOSE_BASE_ESTIMATOR, "/opt/src/mmpose/mmpose/models/pose_estimators/base.py"),
    }
    origins = {
        name: {
            "origin": container_path,
            "pinned_source": container_path,
            "sha256": host.sha256_file(local_path),
            "matched_pinned_source": True,
        }
        for name, (local_path, container_path) in origin_sources.items()
    }
    effective = _effective_postprocess()
    receipt = {
        "schema_version": host.PROFILE_SCHEMA_VERSION,
        "status": "observed", "run_id": run_id, "profile": 640,
        "created_at": "2026-07-18T00:00:00+00:00", "plan_fingerprint_sha256": plan_fp,
        "execution_boundary": {
            "effective_uid": 1000, "effective_gid": 1000, "root_read_only": True,
            "network_interfaces": ["lo"], "gpu_device_nodes": [], "gpu_api_queried": False,
            "output_directory_device": output.stat().st_dev, "output_directory_inode": output.stat().st_ino,
            "output_directory_binding_verified": True,
            "source_sha256": {f"/opt/src/{index}.py": SHA for index in range(9)},
            "runtime": "cpu_only_exact_image_shape_diagnostic", "network": "none",
            "root_filesystem": "read_only", "gpu_exposed": False, "gpu_compute_executed": False,
            "model_loaded": True, "onnx_exported": True, "onnxruntime_executed": True,
            "tensorrt_executed": False, "deepstream_executed": False,
        },
        "profile_configuration": {
            "spatial_size": 640, "deploy_onnx_input_shape": [640, 640],
            "model_input_size": [640, 640], "codec_input_size": [640, 640],
            "test_bottomup_resize_input_size": [640, 640], "val_bottomup_resize_input_size": [640, 640],
            "training_pipeline_modified": False, "passed": True,
        },
        "post_processing_configuration": {
            "deploy_defaults": {"score_threshold": 0.05, "iou_threshold": 0.5, "max_output_boxes_per_class": 200, "pre_top_k": 5000, "keep_top_k": 100, "background_label_id": -1},
            "model_test_cfg": {"score_thr": 0.01, "nms_thr": 0.65, "max_per_img_present": False, "max_per_img": None},
            "effective": effective, "rewrite_resolution_source": "/opt/src/rewrite.py",
            "model_to_head_copy_source": "/opt/src/base.py", "passed": True,
        },
        "imported_module_origins": origins,
        "publication": {
            "export_staging": "container_private_tmpfs", "host_output_used_for_export_or_validation": False,
            "anonymous_inode": True, "linkat_empty_path": True, "no_overwrite": True,
            "source_sha256": hashlib.sha256(onnx_bytes).hexdigest(), "published_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
            "source_and_published_match": True, "quarantined_diagnostic_only": True,
        },
        "onnx": {"path": host.OUTPUT_NAME, "bytes": len(onnx_bytes), "sha256": hashlib.sha256(onnx_bytes).hexdigest()},
        "graph_observation": {
            "checker_passed": True, "external_data": False, "ir_version": 7,
            "producer_name": "pytorch", "producer_version": "2.0", "opset_imports": [{"domain": "", "version": 11}],
            "node_count": 5, "operator_histogram": {"TopK": 1},
            "raw_inputs": [_value_info("input", dimensions_input)],
            "raw_outputs": [_value_info("dets", dimensions_dets), _value_info("keypoints", dimensions_kpts)],
            "inference": {"status": "passed", "error": None, "inputs": [], "outputs": []},
            "output_producers": {}, "topk_nodes": [{}], "non_max_suppression_nodes": [{}],
            "semantic_lineage": {"claim": "lineage", "passing_candidate_count": 1, "candidates": [], "passed": True},
        },
        "runtime_observation": {
            "provider_requested": "CPUExecutionProvider", "providers_active": ["CPUExecutionProvider"],
            "input": {"name": "input", "shape": ["batch", 3, 640, 640], "type": "tensor(float)"},
            "output_metadata": [{}, {}], "output_names": ["dets", "keypoints"], "batches": runtime_batches, "passed": True, "error": None,
        },
        "classification": {
            "classification": "bounded_dynamic_topk_instance_axis_candidate", "evidence": {f"e{index}": True for index in range(10)},
            "derived_shape_interface": {f"d{index}": True for index in range(9)},
            "contract_change_authorized": False, "correction_plan_created": False, "unexpected_shapes_fail_closed": False,
        },
        "claims": {
            "shape_observed": True, "contract_changed": False, "production_model_selected": False,
            "production_ready": False, "tensorrt_verified": False, "deepstream9_verified": False,
        },
    }
    receipt["receipt_fingerprint_sha256"] = host.fingerprint(receipt, "receipt_fingerprint_sha256")
    return receipt


def test_profile_schema_and_host_semantics_accept_bounded_dynamic_observation(tmp_path):
    output = tmp_path / "diagnostic"
    output.mkdir()
    receipt = _synthetic_profile_receipt(output, "1" * 64, "cpu-shape-diag-test")
    path = output / host.PROFILE_RECEIPT_NAME
    path.write_text(json.dumps(receipt), encoding="utf-8")
    path.chmod(0o440)
    jsonschema.Draft202012Validator(host.strict_json(host.PROFILE_RECEIPT_SCHEMA)).validate(receipt)
    observed = host.validate_profile_receipt(
        path,
        plan_fingerprint="1" * 64,
        run_id="cpu-shape-diag-test",
        output_identity=(output.stat().st_dev, output.stat().st_ino),
    )
    assert observed["classification"]["classification"] == "bounded_dynamic_topk_instance_axis_candidate"


def test_planner_cli_creates_immutable_plan_without_subprocess(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        host.r10.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess forbidden")),
    )
    output = tmp_path / "plan.json"
    assert host.main(["create-plan", "--output", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    plan = host.strict_json(output)
    assert host.main([
        "verify-plan", "--plan", str(output),
        "--expected-plan-fingerprint", plan["plan_fingerprint_sha256"],
    ]) == 0
    assert "docker_queried" in capsys.readouterr().out
