import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import validation.site_distance_runner_v3 as runner
from validation.site_distance_attestation_v1 import (
    read_json_with_pin,
    validate_onnx_attestation,
    validate_parser_attestation,
)
from validation.site_distance_evaluation import _schema_validate


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _pin(path: Path, root: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _media_pin(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fake_attestations(root: Path) -> tuple[Path, Path, dict, dict, dict]:
    model_640 = _write_bytes(root / "runtime/640/model.onnx", b"static-640-model")
    model_960 = _write_bytes(root / "runtime/960/model.onnx", b"static-960-model")
    parser = _write_bytes(root / "runtime/parser.so", b"ds9-parser")
    models = {640: _pin(model_640, root), 960: _pin(model_960, root)}
    parser_pin = _pin(parser, root)
    onnx = {
        "schema_version": "fixture/onnx",
        "status": "pass",
        "mode": "paired_static_shared_checkpoint",
        "exports": [
            {"profiles": [640], "model": models[640]},
            {"profiles": [960], "model": models[960]},
        ],
        "lineage": {"fixture": "shared-checkpoint"},
        "receipt_fingerprint_sha256": "1" * 64,
    }
    parser_receipt = {
        "schema_version": "fixture/parser",
        "status": "pass",
        "parser": parser_pin,
        "contract": {
            "expected_parse_function": "NvDsInferParseYoloCuda",
            "expected_engine_create_function": "NvDsInferYoloCudaEngineGet",
        },
        "abi_runtime_proof": {
            "deepstream_version": "9.0.0",
            "cuda_version": "13.1",
            "tensorrt_version": "10.14.1.48",
            "resolved_image_id": "sha256:" + "a" * 64,
        },
        "kernel_runtime_proof": {
            "compute_capability": "8.6",
            "resolved_image_id": "sha256:" + "a" * 64,
        },
        "receipt_fingerprint_sha256": "2" * 64,
    }
    onnx_path = _write_json(root / "evidence/onnx.json", onnx)
    parser_path = _write_json(root / "evidence/parser.json", parser_receipt)
    return onnx_path, parser_path, onnx, parser_receipt, {
        "models": models,
        "parser": parser_pin,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path.resolve()
    source = _write_bytes(root / "inputs/source.mp4", b"bounded-site-video")
    camera = _write_json(
        root / "inputs/camera.json",
        {
            "calibration_coordinate_space": {
                "coordinate_space_id": "source-1920x1080",
                "width": 1920,
                "height": 1080,
                "transform_chain": [
                    {"order": 0, "operation": "identity", "parameters": "none"}
                ],
            },
            "source_stream": {
                "width": 1920,
                "height": 1080,
                "crop_pixel_xywh": [0, 0, 1920, 1080],
                "rotation_degrees_clockwise": 0,
            },
            "optics": {"dewarp_enabled": False},
        },
    )
    media = _write_json(
        root / "inputs/media.json",
        {
            "sequence_id": "site-sequence",
            "source_asset": _media_pin(source),
            "source_probe": {"width": 1920, "height": 1080},
        },
    )
    acceptance = _write_json(
        root / "inputs/acceptance.json",
        {"evaluation_config": {"serialization_confidence_floor": 0.0}},
    )
    calibration = _write_json(root / "inputs/calibration.json", {"fixture": True})
    ground_truth = _write_json(root / "inputs/ground-truth.json", {"fixture": True})
    identity = {
        "dataset_id": "site-dataset",
        "site_id": "site-a",
        "camera_id": "camera-a",
        "camera_configuration_id": "camera-config-a",
        "calibration_id": "calibration-a",
        "sequence_id": "site-sequence",
        "coordinate_space_id": "source-1920x1080",
    }
    preflight_value = {"schema_version": "fixture/preflight", "identity": identity}
    preflight = _write_json(root / "inputs/preflight.json", preflight_value)
    monkeypatch.setattr(runner, "validate_preflight", lambda **_: deepcopy(preflight_value))
    replayed = {"onnx": 0, "parser": 0}

    def replay_onnx(*args, **kwargs):
        replayed["onnx"] += 1

    def replay_parser(*args, **kwargs):
        replayed["parser"] += 1

    monkeypatch.setattr(runner, "validate_onnx_attestation", replay_onnx)
    monkeypatch.setattr(runner, "validate_parser_attestation", replay_parser)
    onnx_path, parser_path, onnx, parser_receipt, attested = _fake_attestations(root)
    labels = _write_bytes(root / "runtime/labels.txt", b"person\n")
    app = _write_bytes(root / "runtime/deepstream-app.identity", b"deepstream-app-9")
    binding_value = {
        "schema_version": "deepsafe.site-distance-runtime-binding/v3",
        "status": "locked",
        "binding_id": "site-person-paired-static-v3",
        "base_model_id": "site-person-yolo11s",
        "profile_models": {
            "640": attested["models"][640],
            "960": attested["models"][960],
        },
        "parser_library": attested["parser"],
        "label_file": _pin(labels, root),
        "attestations": {
            "onnx": {
                "receipt": _pin(onnx_path, root),
                "receipt_fingerprint_sha256": onnx["receipt_fingerprint_sha256"],
            },
            "parser": {
                "receipt": _pin(parser_path, root),
                "receipt_fingerprint_sha256": parser_receipt[
                    "receipt_fingerprint_sha256"
                ],
            },
        },
        "runtime": {
            "deepstream_version": "9.0.0",
            "container_image_reference": "deepsafe-deepstream:9.0",
            "resolved_image_id": "sha256:" + "a" * 64,
            "deepstream_app": _pin(app, root),
            "cuda_version": "13.1",
            "tensorrt_version": "10.14.1.48",
            "driver_version": "590.48.01",
            "gpu_uuid": "GPU-8cbaba1c-2629-a732-f528-66f459089ef6",
            "compute_capability": "8.6",
        },
        "preprocessing": {
            "color_format": "RGB",
            "maintain_aspect_ratio": True,
            "symmetric_padding": True,
            "normalization_factor": 1 / 255,
            "offsets": [0, 0, 0],
        },
        "postprocessing": {
            "person_class_name": "person",
            "person_class_id": 0,
            "cluster_mode": "NMS",
            "nms_iou_threshold": 0.45,
            "serialization_confidence_floor": 0.0,
            "bbox_coordinate_space": "declared_calibration_coordinate_space",
        },
        "inference_backend": {
            "parse_bbox_func_name": "NvDsInferParseYoloCuda",
            "engine_create_func_name": "NvDsInferYoloCudaEngineGet",
            "num_detected_classes": 1,
            "topk": 300,
        },
        "execution": {
            "container_runtime": "docker",
            "host_gpu_index": 0,
            "container_gpu_ordinal": 0,
            "workspace_container_root": "/workspace",
            "validation_gpu_lock_path_template": (
                "/tmp/deepsafe-caviar-gpu{host_gpu_index}.lock"
            ),
            "active_endurance_status_files": [
                "validation/results/endurance/current/status.json",
                "validation/results/endurance/current/live.json",
            ],
            "endurance_supervisor_lock": (
                "validation/results/endurance/current/supervisor.lock"
            ),
            "global_gpu_lease_contract": runner.gpu_lease_contract_projection(),
            "gpu_lease_contract_status": "verified",
            "gpu_lease_owner_kind": "site_distance_25m",
            "gpu_lease_required_for_launch": True,
            "launch_requested": False,
            "launch_authorized": False,
            "launch_blockers": [],
            "engine_load_evidence": {
                "stage": "during_execution",
                "schema_version": (
                    "deepsafe.site-distance-tensorrt-engine-load-attestation/v1"
                ),
                "required_profiles": [640, 960],
                "pair_validator": "validate_engine_load_attestation_pair",
                "plan_time_receipts_allowed": False,
                "required_before_campaign_finalization": True,
            },
        },
        "permitted_profile_differences": runner.PERMITTED_PROFILE_DIFFERENCES,
    }
    binding = _write_json(root / "runtime-binding-v3.json", binding_value)
    return {
        "root": root,
        "camera": camera,
        "media": media,
        "acceptance": acceptance,
        "calibration": calibration,
        "ground_truth": ground_truth,
        "preflight": preflight,
        "binding": binding,
        "binding_value": binding_value,
        "output": root / "generated/site-distance-v3",
        "replayed": replayed,
    }


def _generate(paths: dict):
    return runner.generate_run_plan(
        preflight_receipt_path=paths["preflight"],
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
        runtime_binding_path=paths["binding"],
        campaign_evidence_id="campaign-site-a-2026-07-18-r1",
        output_root=paths["output"],
        workspace_root=paths["root"],
    )


def test_checked_in_parent_attestations_still_validate_without_live_tool_replay():
    root = Path(__file__).resolve().parents[1]
    onnx, _ = read_json_with_pin(
        root / "validation/results/site-distance-25m/attestations/onnx-v1.json",
        project_root=root,
    )
    parser, _ = read_json_with_pin(
        root / "validation/results/site-distance-25m/attestations/parser-v1.json",
        project_root=root,
    )
    validate_onnx_attestation(onnx, project_root=root, verify_live=False)
    validate_parser_attestation(parser, project_root=root, verify_live=False)
    assert onnx["receipt_fingerprint_sha256"] == (
        "fb377d7ed9bc1195ff879dafd1238b050e3a9aa0c1f475d83ba3d0d70e97e912"
    )
    assert parser["receipt_fingerprint_sha256"] == (
        "f7dea8e37484d7fd2bdaed8a143f9afb52addbd51f65a3410108ffb9987d2283"
    )


def test_runtime_binding_v3_template_pins_current_parent_baseline_exactly():
    root = Path(__file__).resolve().parents[1]
    template = json.loads(
        (
            root
            / "validation/inputs/distance-25m/templates/runtime-binding-v3.json.example"
        ).read_text(encoding="utf-8")
    )
    _schema_validate(
        template,
        "site-distance-runtime-binding-v3.schema.json",
        "v3 runtime template",
    )
    records = [
        template["profile_models"]["640"],
        template["profile_models"]["960"],
        template["parser_library"],
        template["label_file"],
        template["attestations"]["onnx"]["receipt"],
        template["attestations"]["parser"]["receipt"],
        template["execution"]["global_gpu_lease_contract"]["artifact"],
    ]
    for record in records:
        assert _pin(root / record["path"], root) == record
    assert template["runtime"]["deepstream_app"]["sha256"] == "0" * 64
    assert not (root / template["runtime"]["deepstream_app"]["path"]).exists()


def test_generate_is_deterministic_and_engine_evidence_is_not_plan_prerequisite(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    first, plan_path, first_preparation, preparation_path = _generate(paths)
    plan_bytes = plan_path.read_bytes()
    preparation_bytes = preparation_path.read_bytes()
    second, _, second_preparation, _ = _generate(paths)
    assert first == second
    assert first_preparation == second_preparation
    assert plan_path.read_bytes() == plan_bytes
    assert preparation_path.read_bytes() == preparation_bytes
    assert first["pre_execution_gate"] == {
        "static_attestations": "verified",
        "gpu_lease_contract": "verified",
        "launch_contract_ready": True,
        "launch_requested": False,
        "launch_authorized": False,
        "launch_blockers": [],
        "engine_load_evidence_is_plan_prerequisite": False,
    }
    assert first["completion_gate"]["evidence_stage"] == "during_execution"
    assert len(first["completion_gate"]["required_pair_id"]) == 64
    assert first["completion_gate"]["required_profiles"] == [640, 960]
    assert first["completion_gate"]["required_before_campaign_finalization"] is True
    for job in first["jobs"]:
        assert job["command"][:11] == [
            "python3",
            "-m",
            "validation.gpu_lease",
            "run",
            "--owner-kind",
            "site_distance_25m",
            "--gpu-index",
            "0",
            "--ttl-seconds",
            "30",
            "--",
        ]
    assert first_preparation["completion_evidence_status"] == (
        "pending_paired_640_960_engine_load_receipts"
    )
    assert first_preparation["safety"] == {
        "cpu_inference_executed": False,
        "docker_called": False,
        "deepstream_called": False,
        "tensorrt_called": False,
        "gpu_accessed": False,
    }
    assert paths["replayed"]["onnx"] >= 2
    assert paths["replayed"]["parser"] >= 2
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)


def test_plan_projects_exact_parent_pins_fingerprints_and_profile_models(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    plan, _, _, _ = _generate(paths)
    binding = paths["binding_value"]
    assert plan["parent_attestations"] == binding["attestations"]
    assert plan["jobs"][0]["profile_model"] == binding["profile_models"]["640"]
    assert plan["jobs"][1]["profile_model"] == binding["profile_models"]["960"]
    infer_640 = (paths["output"] / "640/infer.txt").read_text()
    infer_960 = (paths["output"] / "960/infer.txt").read_text()
    assert "onnx-file=/workspace/runtime/640/model.onnx" in infer_640
    assert "onnx-file=/workspace/runtime/960/model.onnx" in infer_960
    assert plan["semantic_equivalence"]["observed_config_differences"] == (
        runner.ALLOWED_CONFIG_DIFFERENCES
    )


def test_runtime_binding_rejects_parent_fingerprint_and_model_projection_tamper(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    value = deepcopy(paths["binding_value"])
    value["attestations"]["onnx"]["receipt_fingerprint_sha256"] = "f" * 64
    _write_json(paths["binding"], value)
    with pytest.raises(ValueError, match="ONNX attestation fingerprint differs"):
        _generate(paths)
    value = deepcopy(paths["binding_value"])
    value["profile_models"]["640"] = value["profile_models"]["960"]
    _write_json(paths["binding"], value)
    with pytest.raises(ValueError, match="profile 640 model differs"):
        _generate(paths)
    value = deepcopy(paths["binding_value"])
    value["execution"]["global_gpu_lease_contract"][
        "contract_fingerprint_sha256"
    ] = "f" * 64
    _write_json(paths["binding"], value)
    with pytest.raises(ValueError, match="global GPU lease contract projection differs"):
        _generate(paths)


def test_plan_replay_rejects_profile_config_and_output_path_tamper(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    plan, plan_path, _, _ = _generate(paths)
    infer = paths["output"] / "960/infer.txt"
    infer.write_text(
        infer.read_text().replace(
            "nms-iou-threshold=0.45000000000000001",
            "nms-iou-threshold=0.4",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="live bytes/SHA-256 differs"):
        runner.verify_run_plan(plan_path, workspace_root=paths["root"])
    _generate(paths)
    tampered = deepcopy(plan)
    tampered["jobs"][0]["engine_path"] = "/workspace/elsewhere/model.engine"
    body = {key: value for key, value in tampered.items() if key != "plan_fingerprint_sha256"}
    tampered["plan_fingerprint_sha256"] = runner._sha256_json(body)
    _write_json(plan_path, tampered)
    with pytest.raises(ValueError, match="output path projection differs"):
        runner.verify_run_plan(plan_path, workspace_root=paths["root"])


def test_execute_is_fail_closed_without_touching_a_process(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    _, plan_path, _, _ = _generate(paths)
    with pytest.raises(RuntimeError, match="no in-process executor"):
        runner.execute_plan(plan_path=plan_path, confirmation=runner.CONFIRMATION)


def _engine_receipt(paths: dict, plan: dict, profile: int, *, pair_id: str) -> Path:
    job = next(item for item in plan["jobs"] if item["profile"] == profile)
    directory = paths["output"] / str(profile)
    engine = _write_bytes(directory / "model.engine", f"engine-{profile}".encode())
    runtime_log = _write_bytes(directory / "deepstream.log", f"log-{profile}".encode())
    invariant = plan["cross_profile_invariants"]
    runtime = invariant["runtime"]
    backend = invariant["inference_backend"]
    receipt = {
        "parents": {
            "onnx_attestation": plan["parent_attestations"]["onnx"]["receipt"],
            "onnx_receipt_fingerprint_sha256": plan["parent_attestations"]["onnx"][
                "receipt_fingerprint_sha256"
            ],
            "parser_attestation": plan["parent_attestations"]["parser"]["receipt"],
            "parser_receipt_fingerprint_sha256": plan["parent_attestations"]["parser"][
                "receipt_fingerprint_sha256"
            ],
        },
        "contract": {
            "profile": profile,
            "model_sha256": job["profile_model"]["sha256"],
            "parser_sha256": invariant["parser_library"]["sha256"],
            "parse_function": backend["parse_bbox_func_name"],
            "engine_create_function": backend["engine_create_func_name"],
        },
        "run": {
            "profile": profile,
            "pair_id": pair_id,
            "run_id": hashlib.sha256(f"run-{profile}".encode()).hexdigest(),
        },
        "runtime": {
            "deepstream_version": runtime["deepstream_version"],
            "tensorrt_version": runtime["tensorrt_version"],
            "cuda_version": runtime["cuda_version"],
            "requested_image": runtime["container_image_reference"],
            "resolved_image_id": runtime["resolved_image_id"],
            "gpu": {
                "host_index": 0,
                "uuid": runtime["gpu_uuid"],
                "compute_capability": runtime["compute_capability"],
                "container_device_ordinal": 0,
                "docker_device_request": {
                    "driver": "nvidia",
                    "count": 1,
                    "device_ids": ["0"],
                    "capabilities": [["gpu"]],
                },
            },
        },
        "artifacts": {
            "engine": _pin(engine, paths["root"]),
            "inference_config": job["inference_config"],
            "runtime_log": _pin(runtime_log, paths["root"]),
        },
        "load": {
            "onnx_path_in_container": "/workspace/" + job["profile_model"]["path"],
            "engine_path_in_container": job["engine_path"],
            "parser_path_in_container": (
                "/workspace/" + invariant["parser_library"]["path"]
            ),
            "inference_config_path_in_container": (
                "/workspace/" + job["inference_config"]["path"]
            ),
        },
        "receipt_fingerprint_sha256": hashlib.sha256(
            f"receipt-{profile}".encode()
        ).hexdigest(),
    }
    return _write_json(directory / "engine-load-attestation.json", receipt)


def test_pair_gate_requires_exact_640_960_receipts_and_creates_no_quality_claim(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    plan, plan_path, _, _ = _generate(paths)
    observed = []

    def pair_validator(receipts, **kwargs):
        observed.append(([item["contract"]["profile"] for item in receipts], kwargs))

    monkeypatch.setattr(runner, "validate_engine_load_attestation_pair", pair_validator)
    pair_id = plan["completion_gate"]["required_pair_id"]
    receipt_640 = _engine_receipt(paths, plan, 640, pair_id=pair_id)
    receipt_960 = _engine_receipt(paths, plan, 960, pair_id=pair_id)
    output = paths["output"] / "engine-pair-gate-receipt.json"
    result = runner.finalize_engine_pair_gate(
        plan_path=plan_path,
        receipt_640_path=receipt_640,
        receipt_960_path=receipt_960,
        output_path=output,
        workspace_root=paths["root"],
    )
    assert observed[0][0] == [640, 960]
    assert observed[0][1]["verify_live"] is True
    assert result["status"] == "paired_engine_load_evidence_verified"
    assert [item["profile"] for item in result["profiles"]] == [640, 960]
    assert result["campaign_finalized"] is False
    assert result["inference_executed_by_finalizer"] is False
    assert result["quality_claims_created"] is False
    _schema_validate(
        result,
        "site-distance-engine-pair-gate-receipt-v3.schema.json",
        "test pair gate",
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.finalize_engine_pair_gate(
            plan_path=plan_path,
            receipt_640_path=receipt_640,
            receipt_960_path=receipt_960,
            output_path=output,
            workspace_root=paths["root"],
        )


def test_pair_gate_rejects_missing_or_cross_parent_receipt(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    plan, plan_path, _, _ = _generate(paths)
    monkeypatch.setattr(
        runner, "validate_engine_load_attestation_pair", lambda *args, **kwargs: None
    )
    pair_id = plan["completion_gate"]["required_pair_id"]
    receipt_640 = _engine_receipt(paths, plan, 640, pair_id=pair_id)
    missing = paths["output"] / "960/engine-load-attestation.json"
    with pytest.raises(Exception, match="missing"):
        runner.finalize_engine_pair_gate(
            plan_path=plan_path,
            receipt_640_path=receipt_640,
            receipt_960_path=missing,
            output_path=paths["output"] / "gate.json",
            workspace_root=paths["root"],
        )
    receipt_960 = _engine_receipt(paths, plan, 960, pair_id=pair_id)
    wrong_pair = json.loads(receipt_960.read_text())
    wrong_pair["run"]["pair_id"] = "e" * 64
    _write_json(receipt_960, wrong_pair)
    with pytest.raises(ValueError, match="do not share one pair ID"):
        runner.finalize_engine_pair_gate(
            plan_path=plan_path,
            receipt_640_path=receipt_640,
            receipt_960_path=receipt_960,
            output_path=paths["output"] / "gate.json",
            workspace_root=paths["root"],
        )
    receipt_960 = _engine_receipt(paths, plan, 960, pair_id=pair_id)
    value = json.loads(receipt_960.read_text())
    value["parents"]["onnx_receipt_fingerprint_sha256"] = "f" * 64
    _write_json(receipt_960, value)
    with pytest.raises(ValueError, match="parent projection differs"):
        runner.finalize_engine_pair_gate(
            plan_path=plan_path,
            receipt_640_path=receipt_640,
            receipt_960_path=receipt_960,
            output_path=paths["output"] / "gate.json",
            workspace_root=paths["root"],
        )
