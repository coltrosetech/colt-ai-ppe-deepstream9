from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from validation import person_rtdetrv4_export_r11 as r11


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FINGERPRINT = (
    "baf37272b2f1d098d4735be30c2cd3c4a92e03921a6e25d3896a0a81a5d2d2f7"
)
CONTRACT_FILE_SHA256 = (
    "72492433672e801faadbe95de4b128f354288c6299f8576fffbffdcb98234148"
)
HEX_A = "a" * 64
HEX_B = "b" * 64


def _file_pin(path: str = "evidence/artifact.bin") -> dict:
    return {"path": path, "bytes": 1, "sha256": HEX_A}


def _document_pin(path: str = "evidence/receipt.json") -> dict:
    return {**_file_pin(path), "fingerprint_sha256": HEX_B}


def _schema(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


def _schema_errors(schema: dict, value: dict) -> list:
    return list(
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(value)
    )


@pytest.fixture(scope="module")
def contract() -> dict:
    return r11.load_and_verify_contract(
        expected_fingerprint=CONTRACT_FINGERPRINT
    )


def test_contract_is_schema_valid_self_hashed_and_externally_pinned(
    contract: dict,
) -> None:
    schema = _schema(r11.CONTRACT_SCHEMA)
    assert not _schema_errors(schema, contract)
    assert hashlib.sha256(r11.CONTRACT.read_bytes()).hexdigest() == CONTRACT_FILE_SHA256
    assert stat.S_IMODE(r11.CONTRACT.stat().st_mode) == 0o440
    assert contract["fingerprint_sha256"] == CONTRACT_FINGERPRINT
    assert r11.canonical_fingerprint(contract) == CONTRACT_FINGERPRINT
    assert contract["status"] == "prepared_waiting_for_passed_full_run"


def test_contract_replays_exact_r10_build_smoke_and_baseline_lineage(
    contract: dict,
) -> None:
    inputs = contract["inputs"]
    assert inputs["r10_execution_plan"] == {
        "path": "models/person/training-lanes/rtdetrv4-s-r-livit-person-r1-gpu-v1/execution-plan-r10.json",
        "bytes": 68143,
        "sha256": r11.R10_PLAN_FILE_SHA256,
        "fingerprint_sha256": r11.R10_PLAN_FINGERPRINT,
    }
    assert inputs["r10_build_receipt"]["sha256"] == r11.R10_BUILD_FILE_SHA256
    assert (
        inputs["smoke_host_receipt"]["fingerprint_sha256"]
        == r11.SMOKE_HOST_FINGERPRINT
    )
    assert (
        inputs["baseline_host_receipt"]["fingerprint_sha256"]
        == r11.BASELINE_HOST_FINGERPRINT
    )


@pytest.mark.parametrize("profile", [640, 960])
def test_profiles_are_separate_fixed_spatial_fp16_batch12_contracts(
    contract: dict, profile: int
) -> None:
    value = contract["profiles"][str(profile)]
    assert value["spatial"] == [profile, profile]
    assert value["precision"] == "FP16"
    assert value["images_profile"] == {
        "min": [1, 3, profile, profile],
        "opt": [12, 3, profile, profile],
        "max": [12, 3, profile, profile],
    }
    assert value["orig_target_sizes_profile"] == {
        "min": [1, 2],
        "opt": [12, 2],
        "max": [12, 2],
    }
    assert value["deepstream_batch_size"] == 12
    assert f"/{profile}/" in value["onnx_path"]
    assert f"/{profile}/" in value["engine_path"]
    other = contract["profiles"]["960" if profile == 640 else "640"]
    assert value["onnx_path"] != other["onnx_path"]
    assert value["engine_path"] != other["engine_path"]


def test_tensor_bindings_deepstream_runtime_and_no_nms_are_exact(
    contract: dict,
) -> None:
    tensors = contract["tensor_contract"]
    assert list(tensors["inputs"]) == ["images", "orig_target_sizes"]
    assert list(tensors["outputs"]) == ["labels", "boxes", "scores"]
    assert tensors["checkpoint_payload"] == "ema.module"
    assert tensors["batch_axis_dynamic"] is True
    assert tensors["spatial_axes_dynamic"] is False
    assert contract["runtime"] == {
        "deepstream": "9.0.0",
        "cuda": "13.1",
        "tensorrt": "10.14.1.48",
        "compute_capability": "8.6",
        "gpu": "NVIDIA RTX A5000 Laptop GPU",
        "engine_portability": "exact_runtime_and_gpu_only_rebuild_on_drift",
    }
    ds9 = contract["deepstream9_contract"]
    assert ds9["nvinfer"]["batch_size"] == 12
    assert ds9["nvinfer"]["output_blob_names"] == ["labels", "boxes", "scores"]
    assert ds9["non_image_input"]["initializer"] == "NvDsInferInitializeInputLayers"
    assert ds9["postprocess"] == {
        "nms": "disabled",
        "second_nms": False,
        "duplicate_preservation_adversarial_case_required": True,
        "invalid_or_nonfinite_boxes_rejected": True,
    }


def test_calibration_and_numerical_parity_have_bounded_claims(contract: dict) -> None:
    calibration = contract["calibration_contract"]
    assert calibration["profiles_calibrated_separately"] == [640, 960]
    assert calibration["int8_calibration"] is False
    assert calibration["official_test_opened"] is False
    assert calibration["test_unseen_opened"] is False
    assert calibration["quality_claim_from_calibration"] is False
    parity = contract["parity_contract"]
    assert parity["batches"] == [1, 12]
    assert parity["onnx_to_tensorrt_fp16"]["boxes_max_abs_px_lte"] == {
        "640": 1.0,
        "960": 1.5,
    }
    assert parity["nms_parity"] == {
        "expected_policy": "no_nms",
        "deepstream_cluster_mode": 4,
        "parser_second_nms": False,
        "adversarial_overlapping_duplicate_pair_must_be_preserved": True,
    }
    assert all(value is False for value in contract["claim_boundary"].values())


def test_plan_and_execution_fail_closed_when_full_receipt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = ROOT / "validation/results/person/export/r11-test-missing/host-receipt.json"
    isolated_plan = tmp_path / "execution-plan-r11.json"
    monkeypatch.setattr(r11, "FULL_HOST", missing)
    monkeypatch.setattr(r11, "DEFAULT_PLAN", isolated_plan)
    with pytest.raises(r11.ExportR11Error, match="required file is absent"):
        r11._validate_full_run_evidence(
            missing,
            expected_host_fingerprint=HEX_A,
            expected_best_sha256=HEX_B,
        )
    assert not isolated_plan.exists()


def test_external_acceptance_rejects_a_resealed_contract(contract: dict) -> None:
    changed = copy.deepcopy(contract)
    changed["profiles"]["640"]["deepstream_batch_size"] = 1
    changed["fingerprint_sha256"] = r11.canonical_fingerprint(changed)
    with pytest.raises(r11.ExportR11Error, match="canonical fingerprint mismatch"):
        r11._verify_document_fingerprint(
            changed,
            expected=CONTRACT_FINGERPRINT,
            label="adversarial reseal",
        )


def test_plan_builder_binds_future_receipts_and_keeps_every_gate_false(
    contract: dict,
) -> None:
    future = {
        "host_receipt": _document_pin(
            "models/person/training-runs/lane/full-60e-001/host-receipt.json"
        ),
        "container_receipt": _document_pin(
            "models/person/training-runs/lane/full-60e-001/container-receipt.json"
        ),
        "best_checkpoint": _file_pin(
            "models/person/training-runs/lane/full-60e-001/checkpoints/best.pth"
        ),
        "selection": {
            "checkpoint_payload": "ema.module",
            "selection_metric": "best_coco_ap",
            "best_coco_ap": 0.4,
            "final_epoch_completed": 59,
            "total_epochs_contract": 60,
            "official_test_opened": False,
            "test_unseen_opened": False,
        },
    }
    plan = r11.build_plan(
        contract=contract,
        full_evidence=future,
        prepared_at_utc="2026-07-18T04:00:00Z",
    )
    assert plan["fingerprint_sha256"] == r11.canonical_fingerprint(plan)
    assert plan["training_lineage"]["best_checkpoint"] == future["best_checkpoint"]
    assert plan["stage_order"] == list(r11.STAGES)
    assert all(value is False for value in plan["execution"].values())
    assert all(value is False for value in plan["acceptance"].values())

    overclaim = copy.deepcopy(plan)
    overclaim["acceptance"]["onnx_640_passed"] = True
    overclaim["fingerprint_sha256"] = r11.canonical_fingerprint(overclaim)
    with pytest.raises(r11.ExportR11Error, match="schema mismatch"):
        r11._validate_schema(overclaim, r11.PLAN_SCHEMA)


def test_plan_verifier_rejects_a_resealed_contract_projection_drift(
    contract: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    future = {
        "host_receipt": _document_pin("full-60e-001/host-receipt.json"),
        "container_receipt": _document_pin("full-60e-001/container-receipt.json"),
        "best_checkpoint": _file_pin("full-60e-001/checkpoints/best.pth"),
        "selection": {
            "checkpoint_payload": "ema.module",
            "selection_metric": "best_coco_ap",
            "best_coco_ap": 0.4,
            "final_epoch_completed": 59,
            "total_epochs_contract": 60,
            "official_test_opened": False,
            "test_unseen_opened": False,
        },
    }
    plan = r11.build_plan(
        contract=contract,
        full_evidence=future,
        prepared_at_utc="2026-07-18T04:00:00Z",
    )
    monkeypatch.setattr(r11, "load_and_verify_contract", lambda **_: contract)
    monkeypatch.setattr(r11, "_load_json", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(r11, "_validate_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(r11, "_validate_full_run_evidence", lambda *_args, **_kwargs: future)
    result = r11.verify_plan(
        Path("synthetic-plan.json"),
        expected_plan_fingerprint=plan["fingerprint_sha256"],
        contract_fingerprint=CONTRACT_FINGERPRINT,
    )
    assert result["valid"] is True

    plan["profiles"]["640"]["deepstream_batch_size"] = 1
    plan["fingerprint_sha256"] = r11.canonical_fingerprint(plan)
    with pytest.raises(r11.ExportR11Error, match="contract projection differs: profiles"):
        r11.verify_plan(
            Path("synthetic-plan.json"),
            expected_plan_fingerprint=plan["fingerprint_sha256"],
            contract_fingerprint=CONTRACT_FINGERPRINT,
        )


def _bindings() -> list[dict]:
    return [
        {"name": "images", "io": "input", "dtype": "FLOAT32", "shape": ["batch", 3, 640, 640]},
        {"name": "orig_target_sizes", "io": "input", "dtype": "INT64", "shape": ["batch", 2]},
        {"name": "labels", "io": "output", "dtype": "INT64", "shape": ["batch", 300]},
        {"name": "boxes", "io": "output", "dtype": "FLOAT32", "shape": ["batch", 300, 4]},
        {"name": "scores", "io": "output", "dtype": "FLOAT32", "shape": ["batch", 300]},
    ]


def _evidence(stage: str, payload: dict) -> dict:
    value = {
        "schema_version": "deepsafe.person-rtdetrv4-trained-export-evidence/r11",
        "receipt_id": f"rtdetrv4-r11-{stage}",
        "status": "passed",
        "stage": stage,
        "created_at_utc": "2026-07-18T04:00:00Z",
        "plan": _document_pin("execution-plan-r11.json"),
        "prior_receipts": [],
        "execution": {
            "docker": False,
            "gpu": False,
            "model_loaded": False,
            "onnx": False,
            "tensorrt": False,
            "deepstream9": False,
            "network_downloads": 0,
        },
        "payload": payload,
        "claim_boundary": {
            "quality": False,
            "exact_25m": False,
            "twelve_camera_capacity": False,
            "three_module_full_stack": False,
            "production_ready": False,
        },
        "fingerprint_sha256": HEX_A,
    }
    return value


@pytest.mark.parametrize("profile", [640, 960])
def test_evidence_schema_defines_onnx_engine_numerical_and_ds9_receipts(
    profile: int,
) -> None:
    schema = _schema(r11.EVIDENCE_SCHEMA)
    suffix = str(profile)
    bindings = _bindings()
    for row in bindings:
        row["shape"] = [profile if value == 640 and row["name"] == "images" else value for value in row["shape"]]
    onnx = _evidence(
        f"onnx_export_{suffix}",
        {
            "profile": profile,
            "checkpoint": _file_pin("best.pth"),
            "checkpoint_payload": "ema.module",
            "onnx": _file_pin(f"model-{profile}.onnx"),
            "opset": 18,
            "checker_passed": True,
            "bindings": bindings,
            "batch12_shape_finite": True,
            "framework_onnx_parity": {
                "batches": [1, 12],
                "labels_class_exact": True,
                "boxes_max_abs": 0.01,
                "scores_max_abs": 0.0001,
                "passed": True,
            },
            "passed": True,
        },
    )
    assert not _schema_errors(schema, onnx)

    optimization = {
        "images": {
            "min": [1, 3, profile, profile],
            "opt": [12, 3, profile, profile],
            "max": [12, 3, profile, profile],
        },
        "orig_target_sizes": {
            "min": [1, 2],
            "opt": [12, 2],
            "max": [12, 2],
        },
    }
    engine = _evidence(
        f"tensorrt_fp16_{suffix}",
        {
            "profile": profile,
            "onnx_receipt": _document_pin("onnx-receipt.json"),
            "onnx": _file_pin(f"model-{profile}.onnx"),
            "engine": _file_pin(f"model-{profile}.engine"),
            "precision": "FP16",
            "runtime": {
                "deepstream": "9.0.0",
                "cuda": "13.1",
                "tensorrt": "10.14.1.48",
                "compute_capability": "8.6",
                "gpu": "NVIDIA RTX A5000 Laptop GPU",
            },
            "optimization_profiles": optimization,
            "bindings": bindings,
            "engine_deserialized": True,
            "passed": True,
        },
    )
    assert not _schema_errors(schema, engine)

    box_limit = 1.0 if profile == 640 else 1.5
    numerical = _evidence(
        f"numerical_parity_{suffix}",
        {
            "profile": profile,
            "onnx_receipt": _document_pin("onnx-receipt.json"),
            "engine_receipt": _document_pin("engine-receipt.json"),
            "batches": [1, 12],
            "real_images": 24,
            "capture_groups": 10,
            "matching": "class_preserving_min_cost_bijection_on_all_300_queries",
            "tolerances": {
                "labels_class_exact": True,
                "scores_max_abs_lte": 0.005,
                "boxes_max_abs_px_lte": box_limit,
                "minimum_pair_iou_gte": 0.99,
            },
            "observed": {
                "labels_class_exact": True,
                "scores_max_abs": 0.001,
                "boxes_max_abs_px": 0.5,
                "minimum_pair_iou": 0.995,
            },
            "all_finite": True,
            "passed": True,
        },
    )
    assert not _schema_errors(schema, numerical)

    deepstream = _evidence(
        f"deepstream_parser_parity_{suffix}",
        {
            "profile": profile,
            "engine_receipt": _document_pin("engine-receipt.json"),
            "calibration_receipt": _document_pin("calibration-receipt.json"),
            "numerical_parity_receipt": _document_pin("parity-receipt.json"),
            "runtime": {
                "deepstream": "9.0.0",
                "cuda": "13.1",
                "tensorrt": "10.14.1.48",
            },
            "batch_size": 12,
            "bindings": ["images", "orig_target_sizes", "labels", "boxes", "scores"],
            "parser": {
                "library": _file_pin("parser.so"),
                "function": "NvDsInferParseCustomRTDETRv4Person",
                "input_initializer": "NvDsInferInitializeInputLayers",
            },
            "postprocess": {
                "cluster_mode": 4,
                "nms": "disabled",
                "second_nms": False,
                "duplicate_preservation_adversarial_case_required": True,
            },
            "comparison": {
                "class_ids_exact": True,
                "object_counts_exact": True,
                "scores_max_abs": 0.0005,
                "boxes_max_abs_px": 0.25,
                "adversarial_duplicates_preserved": True,
            },
            "passed": True,
        },
    )
    assert not _schema_errors(schema, deepstream)

    bad = copy.deepcopy(deepstream)
    bad["payload"]["postprocess"]["nms"] = "enabled"
    assert _schema_errors(schema, bad)


def test_evidence_schema_defines_non_int8_threshold_calibration_receipt() -> None:
    schema = _schema(r11.EVIDENCE_SCHEMA)
    calibration = _evidence(
        "threshold_calibration",
        {
            "kind": "person_score_threshold_calibration_not_int8",
            "source_role": "development_validation_seen_during_model_selection_not_independent_test",
            "images": 384,
            "capture_groups": 26,
            "profiles": {
                "640": {
                    "threshold": 0.25,
                    "objective": "max_f1_tie_break_higher_recall_then_lower_threshold",
                    "full_sweep_receipt": _file_pin("sweep-640.json"),
                    "threshold_finite": True,
                },
                "960": {
                    "threshold": 0.2,
                    "objective": "max_f1_tie_break_higher_recall_then_lower_threshold",
                    "full_sweep_receipt": _file_pin("sweep-960.json"),
                    "threshold_finite": True,
                },
            },
            "int8_calibration": False,
            "official_test_opened": False,
            "test_unseen_opened": False,
            "passed": True,
        },
    )
    assert not _schema_errors(schema, calibration)
    calibration["payload"]["int8_calibration"] = True
    assert _schema_errors(schema, calibration)


def test_preparation_runner_contains_no_workload_invocation() -> None:
    source = Path(r11.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        "import onnx",
        "import tensorrt",
        "subprocess.run",
        "subprocess.Popen",
        "os.system",
        "docker run",
        "nvidia-smi",
        "tempfile.mkstemp",
        "os.replace",
    ):
        assert forbidden not in source
    assert "O_TMPFILE" in source
    assert "AT_EMPTY_PATH" in source
    assert r11.main(
        [
            "execute",
            "--plan",
            "validation/results/person/export/nonexistent-r11-plan.json",
            "--accept-plan-fingerprint",
            HEX_A,
            "--accept-contract-fingerprint",
            CONTRACT_FINGERPRINT,
            "--stage",
            "onnx_export_640",
        ]
    ) == 2


def test_plan_publication_is_open_fd_bound_noreplace_and_mode_0440(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r11, "ROOT", tmp_path)
    output = tmp_path / "plans" / "execution-plan-r11.json"
    r11._publish_immutable(output, {"value": "bound-to-anonymous-inode"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "value": "bound-to-anonymous-inode"
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    with pytest.raises(r11.ExportR11Error, match="already exists"):
        r11._publish_immutable(output, {"value": "must-not-overwrite"})
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == (
        "bound-to-anonymous-inode"
    )
