from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from admin import validation as admin_validation


ROOT = Path(__file__).resolve().parents[1]


def _reader(workspace: Path) -> admin_validation.ArtifactReader:
    return admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )


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


def _write(workspace: Path, relative: str, raw: bytes) -> None:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _copy_pin(workspace: Path, descriptor: dict) -> None:
    relative = str(descriptor["path"])
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


def test_checked_in_person_r12_pair_is_exact_but_deployment_stays_closed() -> None:
    projected = admin_validation._person_onnx_r12(_reader(ROOT))
    parent = admin_validation._person_rtdetr_gpu_r10(_reader(ROOT))

    assert projected["available"] is True
    assert projected["state"] == "onnx_640_960_passed_claims_still_closed"
    assert all(projected["integrity"].values())
    assert projected["onnx_640_exported"] is True
    assert projected["onnx_960_exported"] is True
    assert projected["both_profiles_exported"] is True
    for profile in ("640", "960"):
        row = projected["profiles"][profile]
        assert row["receipt_verified"] is True
        assert row["onnx_exported"] is True
        assert row["dynamic_batch"] == [1, 12]
        assert row["fixed_spatial"] == int(profile)
        assert row["opset"] == 18
        assert row["checker_passed"] is True
        assert row["batch12_shape_finite"] is True
        assert row["framework_onnx_parity_passed"] is True
    assert projected["gpu_executed"] is False
    assert projected["tensorrt_executed"] is False
    assert projected["deepstream9_executed"] is False
    assert projected["quality_passed"] is False
    assert projected["production_ready"] is False
    assert parent["onnx_640_exported"] is True
    assert parent["onnx_960_exported"] is True
    assert parent["onnx_export_complete"] is True
    threshold = parent["threshold_calibration_r13b"]
    assert threshold["available"] is True
    assert threshold["state"] == (
        "threshold_calibration_640_960_passed_internal_validation_only"
    )
    assert all(threshold["integrity"].values())
    assert threshold["profiles"]["640"]["threshold"] == 0.4116777181625366
    assert threshold["profiles"]["960"]["threshold"] == 0.430930495262146
    assert threshold["cpu_only"] is True
    assert threshold["gpu_executed"] is False
    assert threshold["quality_passed"] is False
    assert threshold["metric_distance_passed"] is False
    assert threshold["tensorrt_executed"] is False
    assert threshold["deepstream9_executed"] is False
    assert threshold["production_ready"] is False
    assert parent["threshold_calibration_complete"] is True
    assert parent["export_complete"] is False
    assert parent["tensorrt_complete"] is False
    assert parent["deepstream9_complete"] is False
    assert parent["production_ready"] is False


def test_person_r13b_rejects_a_drifted_independent_plan_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pins = copy.deepcopy(admin_validation.PERSON_RTDETR_THRESHOLD_R13B_PINS)
    pins["plan"]["sha256"] = "0" * 64
    monkeypatch.setattr(admin_validation, "PERSON_RTDETR_THRESHOLD_R13B_PINS", pins)

    projected = admin_validation._person_threshold_r13b(_reader(ROOT))

    assert projected["available"] is False
    assert projected["reason"] == "r13b_plan_pin_mismatch"
    assert projected["threshold_calibration_executed"] is False
    assert projected["quality_passed"] is False
    assert projected["tensorrt_executed"] is False
    assert projected["production_ready"] is False


def test_checked_in_pose_r10_failure_and_r11_shape_facts_are_bounded() -> None:
    reader = _reader(ROOT)
    failed = admin_validation._pose_export_r10_failure(reader)
    diagnostic = admin_validation._pose_shape_diagnostic_r11(reader)

    assert failed["available"] is True
    assert all(failed["integrity"].values())
    assert failed["run_status"] == "failed"
    assert failed["failure_cause"] == "ProfileExportError: dets shape differs"
    assert failed["profiles"] == {
        "640": {"attempted": True, "status": "failed"},
        "960": {"attempted": False, "status": "not_attempted"},
    }
    assert failed["onnx_640_published"] is False
    assert failed["onnx_960_published"] is False
    assert failed["production_ready"] is False

    assert diagnostic["available"] is True
    assert all(diagnostic["integrity"].values())
    assert diagnostic["derived_interface"] == {
        "input": ["B", 3, 640, 640],
        "dets": ["B", "K", 5],
        "keypoints": ["B", "K", 17, 3],
        "shared_k": True,
        "k_formula_from_pinned_source": "K=min(100,M+1)",
        "k_min": 1,
        "k_max": 100,
        "k_data_and_batch_dependent": True,
    }
    assert diagnostic["runtime_blank_probe"]["batch_1"]["k"] == 1
    assert diagnostic["runtime_blank_probe"]["batch_12"]["k"] == 1
    assert (
        diagnostic["runtime_blank_probe"]["runtime_k_variation_proven"]
        is False
    )
    assert diagnostic["diagnostic_onnx_quarantined"] is True
    assert diagnostic["production_onnx_publishable"] is False
    assert diagnostic["existing_fixed_k_packer_compatible"] is False
    assert diagnostic["contract_change_authorized"] is False
    serialized = json.dumps(
        {"failed": failed, "diagnostic": diagnostic}, ensure_ascii=False
    )
    assert "/home/" not in serialized
    assert "sha256:" not in serialized


def test_repinning_pose_failure_log_without_shape_marker_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = copy.deepcopy(admin_validation.POSE_MMPOSE_EXPORT_R10_FAILURE_PINS)
    for descriptor in pins.values():
        _copy_pin(workspace, descriptor)
    log_path = workspace / str(pins["failure_log"]["path"])
    raw = log_path.read_bytes().replace(
        b"ProfileExportError: dets shape differs",
        b"ProfileExportError: score shape differs",
    )
    log_path.chmod(0o640)
    log_path.write_bytes(raw)
    pins["failure_log"] = _pin(str(pins["failure_log"]["path"]), raw)
    monkeypatch.setattr(
        admin_validation, "POSE_MMPOSE_EXPORT_R10_FAILURE_PINS", pins
    )

    projected = admin_validation._pose_export_r10_failure(_reader(workspace))

    assert projected["available"] is False
    assert projected["reason"] == "pose_r10_failure_contract_invalid"
    assert projected["integrity"]["failure_log_shape_marker_verified"] is False
    assert projected["onnx_640_published"] is False
    assert projected["production_ready"] is False


def test_checked_in_ppe_r5_projects_counts_and_zero_holdout_payload() -> None:
    projected = admin_validation._ppe_four_class_r5(_reader(ROOT))

    assert projected["available"] is True
    assert all(projected["integrity"].values())
    assert projected["state"] == (
        "semantic_remediation_prepared_not_training_authorized"
    )
    assert projected["candidate"]["images"] == 2327
    assert projected["candidate"]["groups"] == 220
    assert projected["candidate"]["retained_bbox_rows"] == 13059
    assert projected["candidate"]["quarantined_bbox_rows"] == 2999
    assert projected["candidate"]["retained_class_counts"] == {
        "helmet_worn_candidate": 4365,
        "hi_vis_worn_candidate": 2403,
        "no_helmet_explicit": 925,
        "person": 5366,
    }
    assert projected["holdout_payload_access"] == {
        "image_files_opened": 0,
        "label_files_opened": 0,
        "metadata_rows_seen": 259,
    }
    assert projected["runtime_no_hi_vis_detector_class_created"] is False
    for gate in (
        "training_authorized",
        "evaluation_authorized",
        "export_authorized",
        "human_qa_complete",
        "production_ready",
    ):
        assert projected[gate] is False
    assert projected["human_qa_packet_r6"]["available"] is True
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "/home/" not in serialized
    assert "sha256" not in serialized


def test_checked_in_ppe_r6_projects_packet_without_human_approval() -> None:
    projected = admin_validation._ppe_human_qa_r6(_reader(ROOT))

    assert projected["available"] is True
    assert all(projected["integrity"].values())
    assert projected["state"] == "human_qa_packet_prepared_not_adjudicated"
    assert projected["packet_prepared"] is True
    assert projected["samples"] == 718
    assert projected["contact_sheets"] == 45
    assert projected["groups"] == 213
    assert projected["role_samples"] == {"train": 386, "calibration": 332}
    assert projected["holdout_payload_access"] == {
        "image_files_opened": 0,
        "label_files_opened": 0,
    }
    for gate in (
        "human_qa_complete",
        "permanent_review_record_present",
        "permanent_approval_present",
        "training_authorized",
        "evaluation_authorized",
        "export_authorized",
        "production_ready",
    ):
        assert projected[gate] is False
    assert projected["integrity"]["samples_not_collected"] is True
    assert projected["integrity"]["artifact_manifest_not_collected"] is True
    assert projected["integrity"]["payload_access_not_collected"] is True
    serialized = json.dumps(projected, ensure_ascii=False)
    assert "/home/" not in serialized
    assert "sha256" not in serialized


def test_missing_ppe_r6_packet_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = copy.deepcopy(admin_validation.PPE_HUMAN_QA_R6_PINS)
    for key, descriptor in pins.items():
        if key != "samples":
            _copy_pin(workspace, descriptor)
    monkeypatch.setattr(admin_validation, "PPE_HUMAN_QA_R6_PINS", pins)

    projected = admin_validation._ppe_human_qa_r6(_reader(workspace))

    assert projected["available"] is False
    assert projected["reason"] == "r6_samples_missing"
    assert projected["packet_prepared"] is False
    assert projected["human_qa_complete"] is False
    assert projected["training_authorized"] is False
    assert projected["production_ready"] is False


def test_tampered_ppe_r6_samples_fail_exact_pin_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = copy.deepcopy(admin_validation.PPE_HUMAN_QA_R6_PINS)
    for descriptor in pins.values():
        _copy_pin(workspace, descriptor)
    samples_path = workspace / str(pins["samples"]["path"])
    raw = bytearray(samples_path.read_bytes())
    raw[0] ^= 1
    samples_path.chmod(0o640)
    samples_path.write_bytes(raw)
    monkeypatch.setattr(admin_validation, "PPE_HUMAN_QA_R6_PINS", pins)

    projected = admin_validation._ppe_human_qa_r6(_reader(workspace))

    assert projected["available"] is False
    assert projected["reason"] == "r6_samples_pin_mismatch"
    assert projected["integrity"]["samples_exact_pin_verified"] is False
    assert projected["human_qa_complete"] is False
    assert projected["training_authorized"] is False
    assert projected["production_ready"] is False


def test_resealed_ppe_r6_human_qa_overclaim_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = copy.deepcopy(admin_validation.PPE_HUMAN_QA_R6_PINS)
    for descriptor in pins.values():
        _copy_pin(workspace, descriptor)
    receipt_path = workspace / str(pins["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["readiness"]["human_qa_complete"] = True
    receipt["readiness"]["training_authorized"] = True
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(unsigned)
    raw = _raw(receipt)
    receipt_path.chmod(0o640)
    receipt_path.write_bytes(raw)
    pins["receipt"] = {
        **_pin(str(pins["receipt"]["path"]), raw),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    monkeypatch.setattr(admin_validation, "PPE_HUMAN_QA_R6_PINS", pins)

    projected = admin_validation._ppe_human_qa_r6(_reader(workspace))

    assert projected["available"] is False
    assert projected["reason"] == "r6_cross_artifact_contract_invalid"
    assert projected["integrity"]["receipt_self_hash_verified"] is True
    assert projected["integrity"]["receipt_schema_replayed"] is False
    assert projected["integrity"]["cross_artifact_semantics_verified"] is False
    assert projected["human_qa_complete"] is False
    assert projected["training_authorized"] is False
    assert projected["production_ready"] is False


def test_resealed_ppe_r5_training_overclaim_fails_semantic_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    pins = copy.deepcopy(admin_validation.PPE_FOUR_CLASS_SEMANTIC_R5_PINS)
    for descriptor in pins.values():
        _copy_pin(workspace, descriptor)
    receipt_path = workspace / str(pins["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["readiness"]["training_authorized"] = True
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = admin_validation._canonical_sha256(unsigned)
    raw = _raw(receipt)
    receipt_path.chmod(0o640)
    receipt_path.write_bytes(raw)
    pins["receipt"] = {
        **_pin(str(pins["receipt"]["path"]), raw),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    monkeypatch.setattr(
        admin_validation, "PPE_FOUR_CLASS_SEMANTIC_R5_PINS", pins
    )

    projected = admin_validation._ppe_four_class_r5(_reader(workspace))

    assert projected["available"] is False
    assert projected["reason"] == "r5_cross_artifact_contract_invalid"
    assert projected["integrity"]["receipt_self_hash_verified"] is True
    assert projected["integrity"]["cross_artifact_semantics_verified"] is False
    assert projected["training_authorized"] is False
    assert projected["production_ready"] is False


def _person_receipt(
    *, profile: int, onnx_pin: dict, prior_receipts: list[dict]
) -> dict:
    receipt = {
        "schema_version": (
            "deepsafe.person-rtdetrv4-trained-export-evidence/r11"
        ),
        "receipt_id": f"rtdetrv4-s-r11-onnx-export-{profile}",
        "status": "passed",
        "stage": f"onnx_export_{profile}",
        "created_at_utc": "2026-07-18T06:30:00+00:00",
        "plan": {
            **admin_validation.PERSON_RTDETR_EXPORT_R11_PINS["plan"],
            "fingerprint_sha256": (
                admin_validation.PERSON_RTDETR_EXPORT_R11_PLAN_FINGERPRINT
            ),
        },
        "prior_receipts": prior_receipts,
        "execution": {
            "docker": False,
            "gpu": False,
            "model_loaded": True,
            "onnx": True,
            "tensorrt": False,
            "deepstream9": False,
            "network_downloads": 0,
        },
        "payload": {
            "profile": profile,
            "checkpoint": admin_validation.PERSON_RTDETR_FULL_TRAINING_R10_PINS[
                "best_checkpoint"
            ],
            "checkpoint_payload": "ema.module",
            "onnx": onnx_pin,
            "opset": 18,
            "checker_passed": True,
            "bindings": admin_validation._person_onnx_r12_expected_bindings(
                profile
            ),
            "batch12_shape_finite": True,
            "framework_onnx_parity": {
                "batches": [1, 12],
                "labels_class_exact": True,
                "boxes_max_abs": 0.001,
                "scores_max_abs": 0.00001,
                "passed": True,
            },
            "passed": True,
        },
        "claim_boundary": {
            "quality": False,
            "exact_25m": False,
            "twelve_camera_capacity": False,
            "three_module_full_stack": False,
            "production_ready": False,
        },
    }
    receipt["fingerprint_sha256"] = admin_validation._canonical_sha256(receipt)
    return receipt


def _stage_person_receipt(
    workspace: Path,
    *,
    profile: int,
    prior_receipts: list[dict],
) -> dict:
    onnx_path = (
        "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/"
        f"onnx/{profile}/rtdetrv4-s-r11-{profile}-bdynamic-opset18.onnx"
    )
    onnx_raw = f"test-onnx-{profile}".encode("ascii")
    _write(workspace, onnx_path, onnx_raw)
    onnx_pin = _pin(onnx_path, onnx_raw)
    receipt = _person_receipt(
        profile=profile, onnx_pin=onnx_pin, prior_receipts=prior_receipts
    )
    raw = _raw(receipt)
    receipt_path = admin_validation.PERSON_RTDETR_ONNX_R12_RECEIPT_PATHS[
        profile
    ]
    _write(workspace, receipt_path, raw)
    return {
        **_pin(receipt_path, raw),
        "fingerprint_sha256": receipt["fingerprint_sha256"],
    }


def test_person_r12_requires_pinned_receipts_and_preserves_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    _copy_pin(workspace, admin_validation.PERSON_RTDETR_ONNX_R12_SCHEMA_PIN)

    # A pre-publication ONNX/recovery intent at an expected lane is not
    # discovered when no independent receipt pin exists.
    orphan_onnx = (
        "models/person/export-lanes/rtdetrv4-s-r-livit-person-r11/onnx/640/"
        "rtdetrv4-s-r11-640-bdynamic-opset18.onnx"
    )
    _write(workspace, orphan_onnx, b"orphan-pre-publication-onnx")
    recovery = (
        "validation/results/person/export/"
        "rtdetrv4-s-r-livit-person-r11/onnx-640/recovery-intent-r12.json"
    )
    _write(workspace, recovery, b'{"status":"ready"}\n')
    monkeypatch.setattr(
        admin_validation, "PERSON_RTDETR_ONNX_R12_RECEIPT_PINS", {}
    )
    unpinned = admin_validation._person_onnx_r12(_reader(workspace))
    assert unpinned["available"] is False
    assert unpinned["onnx_640_exported"] is False
    assert unpinned["model_loaded"] is None

    # Recreate the lane in a fresh workspace because publication is
    # no-overwrite in production and the reader requires exact live bytes.
    workspace = tmp_path / "accepted-workspace"
    _copy_pin(workspace, admin_validation.PERSON_RTDETR_ONNX_R12_SCHEMA_PIN)
    pin640 = _stage_person_receipt(
        workspace, profile=640, prior_receipts=[]
    )
    monkeypatch.setattr(
        admin_validation,
        "PERSON_RTDETR_ONNX_R12_RECEIPT_PINS",
        {640: pin640},
    )
    partial = admin_validation._person_onnx_r12(_reader(workspace))
    assert partial["available"] is True
    assert all(partial["integrity"].values())
    assert partial["onnx_640_exported"] is True
    assert partial["onnx_960_exported"] is False
    assert partial["both_profiles_exported"] is False
    assert partial["production_ready"] is False

    prior = [{**pin640}]
    pin960 = _stage_person_receipt(
        workspace, profile=960, prior_receipts=prior
    )
    monkeypatch.setattr(
        admin_validation,
        "PERSON_RTDETR_ONNX_R12_RECEIPT_PINS",
        {640: pin640, 960: pin960},
    )
    complete = admin_validation._person_onnx_r12(_reader(workspace))
    assert complete["available"] is True
    assert all(complete["integrity"].values())
    assert complete["both_profiles_exported"] is True
    assert complete["onnx_640_exported"] is True
    assert complete["onnx_960_exported"] is True
    assert complete["tensorrt_executed"] is False
    assert complete["deepstream9_executed"] is False
    assert complete["quality_passed"] is False
    assert complete["production_ready"] is False


def test_person_r12_rejects_independently_repinned_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    _copy_pin(workspace, admin_validation.PERSON_RTDETR_ONNX_R12_SCHEMA_PIN)
    pin640 = _stage_person_receipt(
        workspace, profile=640, prior_receipts=[]
    )
    receipt_path = workspace / str(pin640["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "failed"
    receipt.pop("fingerprint_sha256", None)
    receipt["fingerprint_sha256"] = admin_validation._canonical_sha256(receipt)
    raw = _raw(receipt)
    receipt_path.write_bytes(raw)
    pin640 = {
        **_pin(str(pin640["path"]), raw),
        "fingerprint_sha256": receipt["fingerprint_sha256"],
    }
    monkeypatch.setattr(
        admin_validation,
        "PERSON_RTDETR_ONNX_R12_RECEIPT_PINS",
        {640: pin640},
    )

    projected = admin_validation._person_onnx_r12(_reader(workspace))

    assert projected["available"] is False
    assert projected["onnx_640_exported"] is False
    assert projected["production_ready"] is False


def test_admin_ui_and_docs_name_latest_evidence_without_execution_actions() -> None:
    page = (ROOT / "admin/static/index.html").read_text(encoding="utf-8")
    docs = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )

    for label in (
        "R12 trained-person ONNX",
        "Pose export R10",
        "Pose shape diagnostic R11",
        "R5 semantik remediation",
        "R5 development holdout payload",
        "R6 insan-QA paket durumu",
        "R6 kalıcı inceleme / onay",
    ):
        assert label in page
    assert "dets [B,K,5]" in page
    assert "TensorRT/DS9/PCK/FPS kapalı" in page
    assert "The R12 reader never scans export directories" in docs
    assert "A lone ONNX, recovery\nintent, failed receipt" in docs
    assert "`semantic_remediation_r5`" in docs
    assert "`human_qa_packet_r6`" in docs
    assert "0 holdout\nimages and 0 holdout labels" in docs
    assert "no permanent review record or\napproval" in docs
    assert "onclick" not in page[page.index("R12 trained-person ONNX") - 100 : page.index("R12 trained-person ONNX") + 100]
