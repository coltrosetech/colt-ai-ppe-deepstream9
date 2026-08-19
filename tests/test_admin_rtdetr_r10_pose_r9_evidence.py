from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app


ROOT = Path(__file__).resolve().parents[1]


def _raw(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _pin(path: str, raw: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _fingerprint(value: dict, field: str = "fingerprint_sha256") -> str:
    unsigned = deepcopy(value)
    unsigned.pop(field, None)
    result = admin_validation._canonical_sha256(unsigned)
    assert result is not None
    return result


def _copy_pins(workspace: Path, pins: dict[str, dict]) -> None:
    for pin in pins.values():
        source = ROOT / str(pin["path"])
        destination = workspace / str(pin["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination.chmod(0o640)


def _reader(workspace: Path) -> admin_validation.ArtifactReader:
    return admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )


def test_checked_in_r10_gpu_and_distance_proxy_projection_is_strictly_bounded() -> None:
    projected = admin_validation._person_rtdetr_gpu_r10(_reader(ROOT))

    assert projected["available"] is True
    assert projected["evidence_version"] == "r10"
    assert projected["state"] == (
        "full_training_completed_internal_validation_only"
    )
    assert all(projected["integrity"].values())
    assert projected["image_build"]["status"] == "passed"
    assert projected["image_build"]["exact_immutable_identity_verified"] is True
    assert projected["image_build"]["gpu_exposed_during_build"] is False
    assert projected["plan"]["full_training_started"] is True
    assert (
        projected["plan"]["full_training_completion_evidence_present"]
        is True
    )
    assert projected["plan"]["full_training_complete"] is True
    smoke = projected["smoke_one_step"]
    assert smoke["run_id"] == "smoke-one-step-006"
    assert smoke["amp_attempts"] == 9
    assert smoke["overflow_backoffs"] == 8
    assert smoke["accepted_scale"] == 256.0
    assert smoke["finite_gradient_tensors"] == 488
    assert smoke["nonfinite_gradient_tensors"] == 0
    assert smoke["optimizer_steps"] == 1
    assert smoke["ema_updates"] == 1
    assert smoke["quality_measured"] is False
    baseline = projected["internal_baseline"]
    assert baseline["run_id"] == "baseline-eval-002"
    assert baseline["images"] == 384
    assert baseline["annotations"] == 3256
    assert baseline["official_test_opened"] is False
    assert baseline["test_unseen_opened"] is False
    assert baseline["coco"]["ap_50_95"] == 0.25337032843844887
    assert baseline["operating_point_score_0_5_iou_0_5"] == {
        "tp": 850,
        "fp": 80,
        "fn": 2406,
        "precision": 0.9139784946236559,
        "recall": 0.26105651105651106,
        "f1": 0.40611562350692787,
    }
    full = projected["full_training_r10"]
    assert full["available"] is True
    assert all(full["integrity"].values())
    assert full["full_training_started"] is True
    assert full["completion_evidence_present"] is True
    assert full["full_training_complete"] is True
    assert full["epochs"] == {
        "completed": 60,
        "contract": 60,
        "validation_runs": 60,
        "best_zero_based_epoch": 53,
        "final_zero_based_epoch": 59,
    }
    assert full["coco"]["best"]["ap_50_95"] == 0.43638015456949064
    assert full["coco"]["final"]["ap_50_95"] == 0.43601712785856694
    assert full["dataset_boundary"]["official_test_opened"] is False
    assert full["dataset_boundary"]["test_unseen_opened"] is False
    assert full["checkpoints"]["published_checkpoint_artifact_count"] == 14
    assert full["checkpoints"]["best_is_final"] is False
    assert full["gpu"]["max_cuda_memory_allocated_bytes"] == 4021828608
    export = projected["export_plan_r11"]
    assert export["available"] is True
    assert all(export["integrity"].values())
    assert export["plan_ready"] is True
    assert export["execution_authorized"] is True
    assert export["export_executed"] is False
    assert export["onnx_exported"] is False
    assert export["tensorrt_executed"] is False
    assert export["deepstream9_executed"] is False
    assert projected["export_plan_ready"] is True
    distance = projected["distance_proxy_r1"]
    assert distance["available"] is True
    assert all(distance["integrity"].values())
    assert distance["metric_distance_established"] is False
    assert distance["twenty_m_established"] is False
    assert distance["twenty_five_m_established"] is False
    assert distance["twenty_to_twenty_five_m_status"] == (
        "blocked_missing_per_camera_metric_calibration"
    )
    assert distance["operating_point_score_0_25"]["precision"] == 0.576804916
    assert distance["operating_point_score_0_25"]["recall"] == 0.461302211
    assert distance["operating_point_score_0_25"]["f1"] == 0.512627986
    assert distance["pixel_height_proxy"]["le_16_px_recall"] == 0.010810811
    assert distance["pixel_height_proxy"]["gt_96_px_recall"] == 0.91283293
    assert projected["full_training_complete"] is True
    for gate in (
        "export_complete",
        "tensorrt_complete",
        "deepstream9_complete",
        "twelve_camera_capacity_complete",
        "production_ready",
    ):
        assert projected[gate] is False
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "sha256:" not in serialized
    assert "/home/" not in serialized
    assert "/workspace/" not in serialized


def test_checked_in_pose_r9_projection_is_environment_only() -> None:
    projected = admin_validation._pose_permission_probe_r9(_reader(ROOT))

    assert projected["available"] is True
    assert projected["evidence_version"] == "r9"
    assert projected["state"] == (
        "export_environment_runtime_ready_model_not_exported"
    )
    assert all(projected["integrity"].values())
    assert projected["export_environment_runtime_ready"] is True
    assert projected["exact_image"]["binding_stable_before_after_probe"] is True
    assert projected["exact_image"]["exact_immutable_identity_verified"] is True
    assert projected["exact_image"]["rebuilt_during_r9"] is False
    assert projected["runtime"]["compiled_mmcv_ops_ready"] is True
    assert projected["runtime"]["mmdeploy_yoloxpose_rewrite_ready"] is True
    assert projected["runtime"]["onnxruntime_cpu_probe_passed"] is True
    assert projected["isolation"] == {
        "network": "none",
        "root_filesystem": "read_only",
        "non_root_uid": 1000,
        "gpu_exposed": False,
        "gpu_api_queried": False,
    }
    for gate in (
        "model_loaded",
        "model_exported",
        "onnx_640_exported",
        "onnx_960_exported",
        "dynamic_batch12_verified",
        "tensorrt_executed",
        "deepstream9_executed",
        "quality_measured",
        "capacity_measured",
        "production_ready",
    ):
        assert projected[gate] is False
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "sha256:" not in serialized
    assert "/home/" not in serialized
    assert "/workspace/" not in serialized


def test_r10_rejects_rehashed_official_test_overclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = deepcopy(admin_validation.PERSON_RTDETR_GPU_R10_PINS)
    _copy_pins(workspace, pins)

    container_key = "baseline_container_receipt"
    container_path = workspace / str(pins[container_key]["path"])
    container = json.loads(container_path.read_text(encoding="utf-8"))
    container["official_test_opened"] = True
    container["fingerprint_sha256"] = _fingerprint(container)
    container_raw = _raw(container)
    container_path.write_bytes(container_raw)
    pins[container_key] = _pin(str(pins[container_key]["path"]), container_raw)

    host_key = "baseline_host_receipt"
    host_path = workspace / str(pins[host_key]["path"])
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["container_receipt"] = {
        "bytes": len(container_raw),
        "fingerprint_sha256": container["fingerprint_sha256"],
        "path": "container-receipt.json",
        "sha256": hashlib.sha256(container_raw).hexdigest(),
    }
    for artifact in host["artifacts"]:
        if artifact["path"] == "container-receipt.json":
            artifact.update(
                {
                    "bytes": len(container_raw),
                    "sha256": hashlib.sha256(container_raw).hexdigest(),
                }
            )
    host["fingerprint_sha256"] = _fingerprint(host)
    host_raw = _raw(host)
    host_path.write_bytes(host_raw)
    pins[host_key] = _pin(str(pins[host_key]["path"]), host_raw)
    monkeypatch.setattr(admin_validation, "PERSON_RTDETR_GPU_R10_PINS", pins)

    projected = admin_validation._person_rtdetr_gpu_r10(_reader(workspace))
    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["reason"] == "r10_cross_artifact_contract_invalid"
    assert projected["integrity"]["baseline_semantics_verified"] is False
    assert projected["production_ready"] is False


def test_r10_preserves_historical_fallback_when_full_receipt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = deepcopy(admin_validation.PERSON_RTDETR_FULL_TRAINING_R10_PINS)
    pins["host_receipt"] = {
        "path": "models/person/training-runs/missing-full/host-receipt.json",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    monkeypatch.setattr(
        admin_validation, "PERSON_RTDETR_FULL_TRAINING_R10_PINS", pins
    )

    projected = admin_validation._person_rtdetr_gpu_r10(_reader(ROOT))
    assert projected["available"] is True
    assert projected["state"] == (
        "smoke_and_internal_baseline_passed_"
        "full_training_result_not_available"
    )
    assert projected["plan"]["full_training_started"] is None
    assert projected["plan"]["full_training_completion_evidence_present"] is False
    assert projected["full_training_complete"] is False
    assert projected["export_plan_ready"] is False
    assert projected["export_complete"] is False


def test_full_training_rejects_rehashed_official_test_overclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = deepcopy(admin_validation.PERSON_RTDETR_FULL_TRAINING_R10_PINS)
    metadata_pins = {
        key: pins[key]
        for key in ("host_receipt", "container_receipt", "events")
    }
    _copy_pins(workspace, metadata_pins)

    container_path = workspace / str(pins["container_receipt"]["path"])
    container = json.loads(container_path.read_text(encoding="utf-8"))
    container["official_test_opened"] = True
    container["fingerprint_sha256"] = _fingerprint(container)
    container_raw = _raw(container)
    container_path.write_bytes(container_raw)
    pins["container_receipt"] = _pin(
        str(pins["container_receipt"]["path"]), container_raw
    )

    host_path = workspace / str(pins["host_receipt"]["path"])
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["container_receipt"] = {
        "bytes": len(container_raw),
        "fingerprint_sha256": container["fingerprint_sha256"],
        "path": "container-receipt.json",
        "sha256": hashlib.sha256(container_raw).hexdigest(),
    }
    for artifact in host["artifacts"]:
        if artifact["path"] == "container-receipt.json":
            artifact.update(
                {
                    "bytes": len(container_raw),
                    "sha256": hashlib.sha256(container_raw).hexdigest(),
                }
            )
    host["fingerprint_sha256"] = _fingerprint(host)
    host_raw = _raw(host)
    host_path.write_bytes(host_raw)
    pins["host_receipt"] = _pin(
        str(pins["host_receipt"]["path"]), host_raw
    )
    monkeypatch.setattr(
        admin_validation, "PERSON_RTDETR_FULL_TRAINING_R10_PINS", pins
    )

    original_read = admin_validation._read_workspace_pin

    def checkpoint_stub(reader, pin, *, expected_path, maximum_bytes, collect):
        if expected_path == str(pins["best_checkpoint"]["path"]):
            return admin_validation.WorkspacePinRead("ok")
        return original_read(
            reader,
            pin,
            expected_path=expected_path,
            maximum_bytes=maximum_bytes,
            collect=collect,
        )

    monkeypatch.setattr(
        admin_validation, "_read_workspace_pin", checkpoint_stub
    )
    projected = admin_validation._person_full_training_r10(_reader(workspace))
    assert projected["available"] is False
    assert projected["reason"] == "full_training_cross_artifact_contract_invalid"
    assert projected["integrity"]["host_fingerprint_replayed"] is True
    assert projected["integrity"]["container_fingerprint_replayed"] is True
    assert projected["integrity"]["cross_artifact_bindings_verified"] is True
    assert projected["integrity"]["container_semantics_verified"] is False
    assert projected["full_training_complete"] is False
    assert projected["export_complete"] is False
    assert projected["production_ready"] is False


def test_r11_rejects_rehashed_export_execution_overclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = deepcopy(admin_validation.PERSON_RTDETR_EXPORT_R11_PINS)
    _copy_pins(workspace, pins)

    plan_path = workspace / str(pins["plan"]["path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["execution"]["model_loaded"] = True
    plan["execution"]["onnx_export"] = True
    plan["fingerprint_sha256"] = _fingerprint(plan)
    plan_raw = _raw(plan)
    plan_path.write_bytes(plan_raw)
    pins["plan"] = _pin(str(pins["plan"]["path"]), plan_raw)
    monkeypatch.setattr(admin_validation, "PERSON_RTDETR_EXPORT_R11_PINS", pins)
    monkeypatch.setattr(
        admin_validation,
        "PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT",
        plan["fingerprint_sha256"],
    )

    projected = admin_validation._person_export_plan_r11(
        _reader(workspace), full_training_verified=True
    )
    assert projected["available"] is False
    assert projected["reason"] == "export_r11_cross_artifact_contract_invalid"
    assert projected["integrity"]["execution_boundary_verified"] is False
    assert projected["plan_ready"] is False
    assert projected["export_executed"] is False
    assert projected["onnx_exported"] is False
    assert projected["production_ready"] is False


def test_distance_proxy_rejects_rehashed_metric_distance_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = deepcopy(admin_validation.PERSON_RTDETR_DISTANCE_PROXY_R1_PINS)
    _copy_pins(workspace, pins)

    report_path = workspace / str(pins["report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["interpretation"]["detection_at_25m_established"] = True
    report["fingerprint_sha256"] = _fingerprint(report)
    report_raw = _raw(report)
    report_path.write_bytes(report_raw)
    pins["report"] = _pin(str(pins["report"]["path"]), report_raw)

    receipt_path = workspace / str(pins["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["report"] = deepcopy(pins["report"])
    receipt["fingerprint_sha256"] = _fingerprint(receipt)
    receipt_raw = _raw(receipt)
    receipt_path.write_bytes(receipt_raw)
    pins["receipt"] = _pin(str(pins["receipt"]["path"]), receipt_raw)
    monkeypatch.setattr(
        admin_validation, "PERSON_RTDETR_DISTANCE_PROXY_R1_PINS", pins
    )

    projected = admin_validation._person_distance_proxy_r1(_reader(workspace))
    assert projected["available"] is False
    assert projected["metric_distance_established"] is False
    assert projected["twenty_to_twenty_five_m_status"] == (
        "blocked_missing_per_camera_metric_calibration"
    )
    assert projected["integrity"]["proxy_only_semantics_verified"] is False


def test_pose_r9_rejects_rehashed_model_export_overclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = deepcopy(admin_validation.POSE_MMPOSE_PERMISSION_PROBE_R9_PINS)
    _copy_pins(workspace, pins)

    probe_path = workspace / str(pins["probe_receipt"]["path"])
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["conclusions"]["model_exported"] = True
    probe["execution_boundary"]["model_exported"] = True
    probe["receipt_payload_sha256"] = _fingerprint(
        probe, "receipt_payload_sha256"
    )
    probe_raw = _raw(probe)
    probe_path.write_bytes(probe_raw)
    pins["probe_receipt"] = _pin(
        str(pins["probe_receipt"]["path"]), probe_raw
    )

    attempt_path = workspace / str(pins["attempt_receipt"]["path"])
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["controls"]["probe_receipt"].update(
        {
            "bytes": len(probe_raw),
            "sha256": hashlib.sha256(probe_raw).hexdigest(),
        }
    )
    attempt["receipt_payload_sha256"] = _fingerprint(
        attempt, "receipt_payload_sha256"
    )
    attempt_raw = _raw(attempt)
    attempt_path.write_bytes(attempt_raw)
    pins["attempt_receipt"] = _pin(
        str(pins["attempt_receipt"]["path"]), attempt_raw
    )
    monkeypatch.setattr(
        admin_validation, "POSE_MMPOSE_PERMISSION_PROBE_R9_PINS", pins
    )
    monkeypatch.setattr(
        admin_validation,
        "POSE_MMPOSE_PERMISSION_PROBE_R9_ATTEMPT_PAYLOAD_SHA256",
        attempt["receipt_payload_sha256"],
    )
    monkeypatch.setattr(
        admin_validation,
        "POSE_MMPOSE_PERMISSION_PROBE_R9_PROBE_PAYLOAD_SHA256",
        probe["receipt_payload_sha256"],
    )

    projected = admin_validation._pose_permission_probe_r9(_reader(workspace))
    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["integrity"]["runtime_probe_semantics_verified"] is False
    assert projected["model_exported"] is False
    assert projected["production_ready"] is False


def test_api_and_ui_expose_versioned_evidence_without_promoting_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(ROOT))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    person = response.json()["campaigns"]["person_model_upgrade"]
    pose = response.json()["campaigns"]["pose_model_readiness"]
    assert person["state"] == "prepared_not_evaluated"
    assert person["selection"]["primary_candidate"] == "YOLO26s"
    assert person["gpu_execution_r10"]["available"] is True
    assert person["gpu_execution_r10"]["full_training_complete"] is True
    assert person["gpu_execution_r10"]["full_training_r10"]["epochs"][
        "completed"
    ] == 60
    assert person["gpu_execution_r10"]["export_plan_r11"]["plan_ready"] is True
    assert person["gpu_execution_r10"]["export_plan_r11"][
        "export_executed"
    ] is False
    assert person["gpu_execution_r10"]["distance_proxy_r1"][
        "metric_distance_established"
    ] is False
    assert person["gates"]["production_ready"] is False
    assert pose["state"] == "planned_license_required_not_exported"
    assert pose["selection"]["candidate"] == "YOLO26s-pose"
    export_r9 = pose["permissive_challenger"]["export_environment_r9"]
    assert export_r9["available"] is True
    assert export_r9["model_exported"] is False
    assert pose["gates"]["production_ready"] is False
    assert page.status_code == 200
    assert "RT-DETRv4 GPU zinciri R10" in page.text
    assert "R10 full-60e-001" in page.text
    assert "R11 export planı" in page.text
    assert "R10 mesafe proxy R1" in page.text
    assert "20–25 m kabulü" in page.text
    assert "Export ortamı R9" in page.text
    assert "R9 kanıt sınırı" in page.text


def test_dashboard_docs_explain_versioned_fail_closed_boundaries() -> None:
    docs = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )
    assert "`gpu_execution_r10`" in docs
    assert "`gpu_execution_r10.distance_proxy_r1`" in docs
    assert "`blocked_missing_per_camera_metric_calibration`" in docs
    assert "`export_environment_r9`" in docs
    assert "did not load the checkpoint" in docs
    assert "`full_training_started=true`" in docs
    assert "`full_training_started=null`" in docs
    assert "`plan_ready=true`" in docs
    assert "`export_executed=false`" in docs
