from __future__ import annotations

import copy
import json
import stat
import subprocess
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from validation import pose_mmpose_yoloxpose_fixk_r12 as host
from validation import pose_mmpose_yoloxpose_fixk_r12_container as container


ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = "2026-07-18T00:00:00+00:00"
SHA = "0" * 64


def _pipeline() -> list[dict]:
    return [
        {"type": "LoadImage"},
        {"type": "BottomupResize", "input_size": (1, 1)},
    ]


def _config_pair() -> tuple[dict, dict]:
    model = {
        "input_size": (1, 1),
        "codec": {"input_size": (1, 1)},
        "test_dataloader": {"dataset": {"pipeline": copy.deepcopy(_pipeline())}},
        "val_dataloader": {"dataset": {"pipeline": copy.deepcopy(_pipeline())}},
        "model": {"test_cfg": {"score_thr": 0.01, "nms_thr": 0.65}},
    }
    deploy = {
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
    return deploy, model


def _raw_and_corrected(batch: int = 1):
    raw_dets = np.zeros((batch, 3, 5), dtype=np.float32)
    raw_dets[:, 0] = [1.0, 2.0, 3.0, 4.0, 0.9]
    raw_dets[:, 1] = [5.0, 6.0, 7.0, 8.0, 0.000031076]
    raw_dets[:, 2] = [9.0, 10.0, 11.0, 12.0, 0.01]
    raw_keypoints = np.zeros((batch, 3, 17, 3), dtype=np.float32)
    raw_keypoints[:, 0] = 10.0
    raw_keypoints[:, 1] = 20.0
    raw_keypoints[:, 2] = 30.0
    corrected_dets = np.zeros((batch, 100, 5), dtype=np.float32)
    corrected_keypoints = np.zeros((batch, 100, 17, 3), dtype=np.float32)
    corrected_dets[:, 0] = raw_dets[:, 0]
    corrected_keypoints[:, 0] = raw_keypoints[:, 0]
    return raw_dets, raw_keypoints, corrected_dets, corrected_keypoints


def _joint_match(valid: int) -> dict:
    return {
        "row_count": valid,
        "maximum_absolute_error": 0.0,
        "maximum_relative_error": 0.0,
        "atol": 0.00001,
        "rtol": 0.00001,
        "passed": True,
    }


def _image_result(index: int, *, raw_k: int, valid: int, positive_invalid: bool) -> dict:
    return {
        "image_index": index,
        "raw_k": raw_k,
        "raw_valid_count": valid,
        "corrected_valid_count": valid,
        "invalid_corrected_count": 100 - valid,
        "valid_joint_rows": _joint_match(valid),
        "invalid_dets_exact_zero": True,
        "invalid_keypoints_exact_zero": True,
        "raw_invalid_nonzero_row_observed": positive_invalid,
        "raw_positive_invalid_score_observed": positive_invalid,
        "maximum_raw_invalid_score": 0.000031076 if positive_invalid else None,
        "corrected_scores_nonincreasing": True,
    }


def _batch_result(profile: int, batch: int, probe: str) -> dict:
    valid = 0 if probe == "blank" else 1
    raw_k = 1 if probe == "blank" else 2
    positive = probe == "blank"
    return {
        "input_shape": [batch, 3, profile, profile],
        "input_sha256": SHA,
        "batch": batch,
        "raw_shapes": {
            "dets": [batch, raw_k, 5],
            "keypoints": [batch, raw_k, 17, 3],
        },
        "corrected_shapes": {
            "dets": [batch, 100, 5],
            "keypoints": [batch, 100, 17, 3],
        },
        "images": [
            _image_result(
                index,
                raw_k=raw_k,
                valid=valid,
                positive_invalid=positive,
            )
            for index in range(batch)
        ],
        "all_outputs_finite": True,
        "all_valid_rows_preserved_bidirectionally": True,
        "all_invalid_rows_exact_zero": True,
        "passed": True,
    }


def _graph(profile: int, *, corrected: bool) -> dict:
    nms = [{"name": "nms", "domain": "", "inputs": ["a"], "outputs": ["b"]}]
    outputs = (
        [
            {"name": "dets", "dtype": "FLOAT", "shape": ["batch", 100, 5]},
            {
                "name": "keypoints",
                "dtype": "FLOAT",
                "shape": ["batch", 100, 17, 3],
            },
        ]
        if corrected
        else [
            {"name": "dets", "dtype": "FLOAT", "shape": ["batch", "K", 5]},
            {
                "name": "keypoints",
                "dtype": "FLOAT",
                "shape": ["batch", "K", 17, 3],
            },
        ]
    )
    return {
        "checker_passed": True,
        "external_data": False,
        "default_opset": 11,
        "input": {
            "name": "input",
            "dtype": "FLOAT",
            "shape": ["batch", 3, profile, profile],
        },
        "outputs": outputs,
        "node_count": 42 if corrected else 20,
        "standard_non_max_suppression": nms,
        "wrapper_node_count": 22 if corrected else 0,
        "wrapper_operator_types": ["TopK", "GatherElements", "GatherElements", "Where", "Where"] if corrected else [],
        "custom_node_domains": [],
        "corrected": corrected,
        "passed": True,
    }


def _profile_receipt(tmp_path: Path, profile: int = 960) -> tuple[Path, dict]:
    raw = tmp_path / host.PROFILE_FILES[profile]["raw"]
    corrected = tmp_path / host.PROFILE_FILES[profile]["corrected"]
    raw.write_bytes(b"synthetic raw graph")
    corrected.write_bytes(b"synthetic corrected graph")
    raw.chmod(0o440)
    corrected.chmod(0o440)
    raw_pin = {"path": raw.name, "bytes": raw.stat().st_size, "sha256": host.sha256_file(raw)}
    corrected_pin = {
        "path": corrected.name,
        "bytes": corrected.stat().st_size,
        "sha256": host.sha256_file(corrected),
    }
    deploy, model = _config_pair()
    configuration, post = container.apply_profile_configuration(deploy, model, profile)
    value = {
        "schema_version": host.PROFILE_SCHEMA_VERSION,
        "status": "passed",
        "run_id": "cpu-fixed-k100-test",
        "profile": profile,
        "created_at": FIXED_TIME,
        "plan_fingerprint_sha256": "1" * 64,
        "execution_boundary": {
            "effective_uid": 1000,
            "effective_gid": 1000,
            "root_read_only": True,
            "network_interfaces": ["lo"],
            "gpu_device_nodes": [],
            "gpu_api_queried": False,
            "output_directory_device": tmp_path.stat().st_dev,
            "output_directory_inode": tmp_path.stat().st_ino,
            "output_directory_binding_verified": True,
            "source_sha256": {str(path): digest for path, digest in container.SOURCE_PINS.items()},
            "r11_onnx_sha256": container.R11_ONNX_SHA256,
            "runtime": "cpu_only_exact_image_fixed_k100_correction",
            "network": "none",
            "root_filesystem": "read_only",
            "gpu_exposed": False,
            "gpu_compute_executed": False,
            "model_loaded": True,
            "raw_model_exported": profile == 960,
            "onnxruntime_executed": True,
            "tensorrt_executed": False,
            "deepstream_executed": False,
        },
        "profile_configuration": configuration,
        "post_processing_configuration": post,
        "imported_module_origins": {
            module_name: {
                "origin": pinned_source,
                "pinned_source": pinned_source,
                "sha256": digest,
                "matched_pinned_source": True,
            }
            for module_name, (pinned_source, digest) in host.EXPECTED_MODULE_ORIGINS.items()
        },
        "raw_graph_source": {
            "kind": "fresh_exact_image_cpu_export",
            "model_exported_in_this_profile": True,
            "r11_sha256": None,
            "private_sha256": raw_pin["sha256"],
            "exact_match": None,
        },
        "wrapper": {
            "wrapper_prefix": container.WRAPPER_PREFIX,
            "append_zero_rows": 100,
            "topk_k": 100,
            "topk_axis": 1,
            "shared_indices": True,
            "score_threshold": 0.01,
            "validity_predicate": "score>0.01",
            "invalid_rows_exact_zero": True,
            "new_node_count": 22,
            "new_initializer_count": 8,
            "custom_domains_added": False,
            "passed": True,
        },
        "raw_graph_validation": _graph(profile, corrected=False),
        "corrected_graph_validation": _graph(profile, corrected=True),
        "semantic_validation": {
            "provider_requested": "CPUExecutionProvider",
            "raw_providers_active": ["CPUExecutionProvider"],
            "corrected_providers_active": ["CPUExecutionProvider"],
            "score_threshold": 0.01,
            "validity_predicate": "score>0.01",
            "probes": {
                probe: {
                    "batches": {
                        str(batch): _batch_result(profile, batch, probe)
                        for batch in (1, 12)
                    },
                    "mandatory_mask_exercised": probe == "blank",
                    "passed": True,
                }
                for probe in ("blank", "real_seed")
            },
            "passed": True,
        },
        "publication": {
            "private_staging": True,
            "anonymous_inode": True,
            "linkat_empty_path": True,
            "no_overwrite": True,
            "raw": raw_pin,
            "corrected": corrected_pin,
            "raw_status": "quarantined_semantic_reference",
            "corrected_status": "candidate_requires_tensorrt_and_deepstream_follow_on",
        },
        "claims": {
            "fixed_public_shapes_created": True,
            "cpu_onnxruntime_semantics_verified": True,
            "invalid_rows_exact_zero_verified": True,
            "profile_quality_claimed": False,
            "production_model_selected": False,
            "tensorrt_verified": False,
            "deepstream9_verified": False,
            "existing_packer_verified": False,
            "production_ready": False,
        },
    }
    value["receipt_fingerprint_sha256"] = host.fingerprint(
        value, "receipt_fingerprint_sha256"
    )
    receipt = tmp_path / "profile-receipt.json"
    receipt.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt.chmod(0o440)
    return receipt, value


def test_contract_and_plan_validate_without_subprocess_or_docker(monkeypatch):
    monkeypatch.setattr(
        host.r10.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess/Docker forbidden during planning")
        ),
    )
    contract = host.strict_json(host.CONTRACT)
    host.validate_contract(contract)
    plan = host.build_plan(created_at=FIXED_TIME)
    jsonschema.Draft202012Validator(host.strict_json(host.PLAN_SCHEMA)).validate(plan)
    replay = host.verify_plan(plan, plan["plan_fingerprint_sha256"])
    assert replay["valid"] is True
    assert replay["docker_queried"] is False
    assert len(plan["inputs"]) == 30
    assert list(plan["profiles"]) == ["640", "960"]


def test_r10_r11_prerequisites_replay_without_modification():
    value = host.validate_prerequisites()
    assert value["r10"]["status"] == "failed_as_expected"
    assert value["r11"]["blank_batch1_k"] == 1
    assert value["r11"]["blank_batch12_k"] == 1
    assert value["r11"]["effective_score_threshold"] == 0.01
    assert value["r11"]["effective_iou_threshold"] == 0.65
    assert stat.S_IMODE(host.R11_RAW_ONNX.stat().st_mode) == 0o440
    assert host.sha256_file(host.R10_PLAN) == host.R10_PLAN_FILE_SHA256
    assert host.sha256_file(host.R11_PLAN) == host.R11_PLAN_FILE_SHA256
    assert host.sha256_file(host.R11_FINAL_RECEIPT) == host.R11_FINAL_RECEIPT_SHA256
    assert host.sha256_file(host.R11_RAW_ONNX) == host.R11_RAW_ONNX_SHA256


@pytest.mark.parametrize("profile", [640, 960])
def test_docker_command_is_exact_image_cpu_network_none(profile):
    command = host.docker_command(
        profile,
        "/new/output",
        "1" * 64,
        "cpu-fixed-k100-test",
        "1000:1000",
        7,
        11,
        {"sha256": "2" * 64, "bytes": 123},
        {"sha256": "3" * 64, "bytes": 456},
        {"sha256": "4" * 64, "bytes": 789},
    )
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network=none" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert host.IMAGE_ID in command
    assert "--gpus" not in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert command[command.index("--profile") + 1] == str(profile)


def test_contract_makes_mask_and_follow_on_boundaries_explicit():
    value = host.strict_json(host.CONTRACT)
    assert value["correction_algorithm"]["append_zero_rows"] == 100
    assert value["correction_algorithm"]["topk"]["k"] == 100
    assert value["correction_algorithm"]["shared_indices"] is True
    assert value["semantic_acceptance"]["blank_probe_must_exercise_positive_invalid_source_score"] is True
    assert value["design_input_not_acceptance_evidence"]["may_not_be_used_as_execution_evidence"] is True
    assert value["runtime_policy"]["imported_export_preprocess_module_origin_sha256_required"] is True
    assert value["tensorrt_deepstream9_follow_on_plan_only"]["required_tensorrt_version"] == "10.14"
    assert value["tensorrt_deepstream9_follow_on_plan_only"]["required_deepstream_version"] == "9"
    assert value["claims"]["correction_executed"] is False
    assert value["claims"]["production_ready"] is False


@pytest.mark.parametrize("profile", [640, 960])
def test_profile_configuration_uses_effective_mmpose_threshold(profile):
    deploy, model = _config_pair()
    configuration, post = container.apply_profile_configuration(deploy, model, profile)
    assert configuration["deploy_onnx_input_shape"] == [profile, profile]
    assert configuration["codec_input_size"] == [profile, profile]
    assert post["deploy_defaults"]["score_threshold"] == 0.05
    assert post["effective"]["score_threshold"] == 0.01
    assert post["effective"]["iou_threshold"] == 0.65
    assert post["effective"]["pre_top_k"] == 5000
    assert post["effective"]["keep_top_k"] == 100


@pytest.mark.parametrize("batch", [1, 12])
def test_semantic_comparison_preserves_joint_valid_rows_and_masks_dummy(batch):
    arrays = _raw_and_corrected(batch)
    result = container.compare_semantics(*arrays)
    assert result["corrected_shapes"]["dets"] == [batch, 100, 5]
    assert result["corrected_shapes"]["keypoints"] == [batch, 100, 17, 3]
    assert all(image["raw_valid_count"] == 1 for image in result["images"])
    assert all(image["raw_positive_invalid_score_observed"] for image in result["images"])
    assert all(image["invalid_dets_exact_zero"] for image in result["images"])
    assert all(image["invalid_keypoints_exact_zero"] for image in result["images"])


@pytest.mark.parametrize("mutation", ["unmasked_dummy", "negative_zero", "missing_valid", "split_keypoints"])
def test_semantic_comparison_fails_closed(mutation):
    raw_dets, raw_keypoints, corrected_dets, corrected_keypoints = _raw_and_corrected()
    if mutation == "unmasked_dummy":
        corrected_dets[0, 1] = raw_dets[0, 1]
        corrected_keypoints[0, 1] = raw_keypoints[0, 1]
    elif mutation == "negative_zero":
        corrected_dets[0, 1, 0] = np.float32(-0.0)
    elif mutation == "missing_valid":
        corrected_dets[0, 0] = 0.0
        corrected_keypoints[0, 0] = 0.0
    else:
        corrected_keypoints[0, 0, 0, 0] += 1.0
    with pytest.raises(container.FixedKCorrectionError):
        container.compare_semantics(
            raw_dets, raw_keypoints, corrected_dets, corrected_keypoints
        )


def test_profile_receipt_schema_and_host_replay_are_fail_closed(tmp_path):
    receipt, value = _profile_receipt(tmp_path)
    schema = host.strict_json(host.PROFILE_RECEIPT_SCHEMA)
    jsonschema.Draft202012Validator(schema).validate(value)
    replay = host.validate_profile_receipt(
        receipt,
        profile=960,
        plan_fingerprint="1" * 64,
        run_id="cpu-fixed-k100-test",
        output_identity=(tmp_path.stat().st_dev, tmp_path.stat().st_ino),
    )
    assert replay["status"] == "passed"
    forged = copy.deepcopy(value)
    for batch in ("1", "12"):
        for image in forged["semantic_validation"]["probes"]["blank"]["batches"][batch]["images"]:
            image["raw_positive_invalid_score_observed"] = False
    forged["receipt_fingerprint_sha256"] = host.fingerprint(
        forged, "receipt_fingerprint_sha256"
    )
    receipt.chmod(0o640)
    receipt.write_text(json.dumps(forged, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt.chmod(0o440)
    with pytest.raises(host.FixedKR12Error, match="positive invalid source score"):
        host.validate_profile_receipt(
            receipt,
            profile=960,
            plan_fingerprint="1" * 64,
            run_id="cpu-fixed-k100-test",
            output_identity=(tmp_path.stat().st_dev, tmp_path.stat().st_ino),
        )


def test_failed_final_receipt_schema_preserves_640_then_960_gate():
    profile_results = {
        "640": {
            "status": "failed",
            "attempted": True,
            "profile": 640,
            "container_exit_code": 2,
            "receipt": None,
            "raw": None,
            "corrected": None,
            "error": "profile container exited 2",
        },
        "960": {
            "status": "not_attempted",
            "attempted": False,
            "profile": 960,
            "container_exit_code": None,
            "receipt": None,
            "raw": None,
            "corrected": None,
            "error": None,
        },
    }
    value = {
        "schema_version": host.RECEIPT_SCHEMA_VERSION,
        "status": "failed",
        "run_id": "cpu-fixed-k100-test",
        "created_at": FIXED_TIME,
        "plan_fingerprint_sha256": "1" * 64,
        "prerequisites": host.validate_prerequisites(),
        "image": {
            "required_id": host.IMAGE_ID,
            "before_id": host.IMAGE_ID,
            "after_id": None,
            "stable": False,
        },
        "run_directory": {
            "name": "cpu-fixed-k100-test",
            "device": 1,
            "inode": 2,
            "mode": "0550",
            "frozen_before_receipt": True,
            "artifact_count": 1,
        },
        "execution_boundary": {
            "container_runs_attempted": 1,
            "profile_order": [640, 960],
            "network_policy": "none",
            "root_filesystem_policy": "read_only",
            "non_root_policy": True,
            "gpu_exposed": False,
            "gpu_api_queried": False,
            "docker_pull": False,
            "docker_build": False,
            "tensorrt_executed": False,
            "deepstream_executed": False,
            "packer_used": False,
        },
        "profiles": profile_results,
        "artifacts": [
            {"path": "host-failure.json", "bytes": 1, "sha256": SHA, "mode": "0440"}
        ],
        "error": "FixedKR12Error: profile 640 container exited 2",
        "follow_on": host.strict_json(host.CONTRACT)[
            "tensorrt_deepstream9_follow_on_plan_only"
        ],
        "conclusions": {
            "both_cpu_profiles_passed": False,
            "fixed_k100_candidate_pair_created": False,
            "invalid_rows_exact_zero_verified": False,
            "profile_960_quality_claimed": False,
            "production_onnx_publishable": False,
            "production_model_selected": False,
            "tensorrt_verified": False,
            "deepstream9_verified": False,
            "existing_packer_verified": False,
            "production_ready": False,
        },
    }
    value["receipt_fingerprint_sha256"] = host.fingerprint(
        value, "receipt_fingerprint_sha256"
    )
    jsonschema.Draft202012Validator(host.strict_json(host.RECEIPT_SCHEMA)).validate(
        value
    )
    host.validate_final_receipt_semantics(value)
    forged = copy.deepcopy(value)
    forged["profiles"]["960"]["status"] = "failed"
    forged["profiles"]["960"]["attempted"] = True
    forged["profiles"]["960"]["container_exit_code"] = 2
    forged["profiles"]["960"]["error"] = "should never have run"
    forged["execution_boundary"]["container_runs_attempted"] = 2
    with pytest.raises(host.FixedKR12Error, match="without a passing profile 640"):
        host.validate_final_receipt_semantics(forged)


SYNTHETIC_WRAPPER_SCRIPT = r'''
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from validation import pose_mmpose_yoloxpose_fixk_r12_container as lane

root = Path(sys.argv[1])
raw = root / "raw.onnx"
corrected = root / "corrected.onnx"

det = np.zeros((1, 3, 5), dtype=np.float32)
det[0, 0] = [1, 2, 3, 4, .9]
det[0, 1] = [5, 6, 7, 8, .000031076]
det[0, 2] = [9, 10, 11, 12, .01]
kpt = np.zeros((1, 3, 17, 3), dtype=np.float32)
kpt[0, 0] = 10
kpt[0, 1] = 20
kpt[0, 2] = 30

initializers = [
    numpy_helper.from_array(det, "det_template"),
    numpy_helper.from_array(kpt, "kpt_template"),
    numpy_helper.from_array(np.asarray([0], dtype=np.int64), "axis0"),
    numpy_helper.from_array(np.asarray([3, 5], dtype=np.int64), "det_tail"),
    numpy_helper.from_array(np.asarray([3, 17, 3], dtype=np.int64), "kpt_tail"),
    numpy_helper.from_array(np.zeros((1, 1, 4), dtype=np.float32), "nms_boxes"),
    numpy_helper.from_array(np.zeros((1, 1, 1), dtype=np.float32), "nms_scores"),
    numpy_helper.from_array(np.asarray([1], dtype=np.int64), "nms_max"),
]
nodes = [
    helper.make_node("Shape", ["input"], ["input_shape"]),
    helper.make_node("Gather", ["input_shape", "axis0"], ["batch_dim"], axis=0),
    helper.make_node("Concat", ["batch_dim", "det_tail"], ["det_shape"], axis=0),
    helper.make_node("Concat", ["batch_dim", "kpt_tail"], ["kpt_shape"], axis=0),
    helper.make_node("Expand", ["det_template", "det_shape"], ["dets"]),
    helper.make_node("Expand", ["kpt_template", "kpt_shape"], ["keypoints"]),
    helper.make_node("NonMaxSuppression", ["nms_boxes", "nms_scores", "nms_max"], ["unused_nms"], center_point_box=0),
]
graph = helper.make_graph(
    nodes,
    "synthetic_dynamic_pose",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 1, 1])],
    [
        helper.make_tensor_value_info("dets", TensorProto.FLOAT, ["batch", "K", 5]),
        helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, ["batch", "K", 17, 3]),
    ],
    initializer=initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, raw)
raw_graph = lane.inspect_graph(raw, 1, corrected=False)
wrapper = lane.add_fixed_k_wrapper(raw, corrected, 1)
corrected_graph = lane.inspect_graph(corrected, 1, corrected=True)
fixed = onnx.load(corrected)
onnx.checker.check_model(fixed, full_check=True)
assert lane._dimension_list(fixed.graph.output[0]) == ["batch", 100, 5]
assert lane._dimension_list(fixed.graph.output[1]) == ["batch", 100, 17, 3]
assert len([n for n in fixed.graph.node if n.op_type == "NonMaxSuppression" and n.domain == ""]) == 1
assert not [n.domain for n in fixed.graph.node if n.domain]
assert raw_graph["standard_non_max_suppression"] == corrected_graph["standard_non_max_suppression"]

raw_session = ort.InferenceSession(str(raw), providers=["CPUExecutionProvider"])
fixed_session = ort.InferenceSession(str(corrected), providers=["CPUExecutionProvider"])
observations = {}
for batch in (1, 12):
    inputs = {"input": np.zeros((batch, 3, 1, 1), dtype=np.float32)}
    raw_outputs = raw_session.run(["dets", "keypoints"], inputs)
    fixed_outputs = fixed_session.run(["dets", "keypoints"], inputs)
    semantic = lane.compare_semantics(*raw_outputs, *fixed_outputs)
    assert semantic["corrected_shapes"]["dets"] == [batch, 100, 5]
    assert semantic["corrected_shapes"]["keypoints"] == [batch, 100, 17, 3]
    assert all(item["raw_positive_invalid_score_observed"] for item in semantic["images"])
    invalid = fixed_outputs[0][..., 4] <= lane.SCORE_THRESHOLD
    assert np.equal(fixed_outputs[0][invalid], 0).all()
    assert not np.signbit(fixed_outputs[0][invalid]).any()
    assert np.equal(fixed_outputs[1][invalid], 0).all()
    assert not np.signbit(fixed_outputs[1][invalid]).any()
    observations[str(batch)] = semantic["corrected_shapes"]
print(json.dumps({"wrapper": wrapper, "observations": observations}, sort_keys=True))
'''


def test_synthetic_dynamic_onnx_wrapper_checker_and_ort_b1_b12(tmp_path):
    python = ROOT / ".venv-export/bin/python"
    assert python.exists()
    completed = subprocess.run(
        [str(python), "-B", "-c", SYNTHETIC_WRAPPER_SCRIPT, str(tmp_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["wrapper"]["topk_k"] == 100
    assert value["wrapper"]["shared_indices"] is True
    assert value["wrapper"]["new_node_count"] == 24
    assert value["wrapper"]["new_initializer_count"] == 8
    assert value["observations"]["1"]["dets"] == [1, 100, 5]
    assert value["observations"]["12"]["keypoints"] == [12, 100, 17, 3]
