from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATHS = (
    "validation/inputs/product-acceptance/three-module.approved.json",
    "validation/schemas/product-acceptance-policy-v1.schema.json",
)


def _pin(path: str, raw: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json(path: Path, value: dict) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


@pytest.fixture
def pose_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    relative_paths = {
        admin_validation.POSE_EXPORT_PROVENANCE_PIN["path"],
        *admin_validation.POSE_EXPORT_PATHS.values(),
        *admin_validation.POSE_PERMISSIVE_CHALLENGER_PATHS.values(),
        *(
            pin["path"]
            for pin in admin_validation.POSE_MMPOSE_ONNX_PREFLIGHT_PINS.values()
        ),
        *(
            pin["path"]
            for pin in admin_validation.POSE_GT_EVIDENCE_PINS.values()
        ),
        *(
            pin["path"]
            for pin in admin_validation.POSE_PCK_EVIDENCE_PINS.values()
        ),
        *POLICY_PATHS,
    }
    for relative in sorted(relative_paths):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return workspace


def _project(workspace: Path) -> dict:
    reader = admin_validation.ArtifactReader(
        workspace / "missing-results",
        workspace_root=workspace,
        schema_root=ROOT / "validation/schemas",
    )
    return admin_validation._pose_readiness(reader)


def test_checked_in_pose_chain_matches_admin_trust_anchor() -> None:
    projected = _project(ROOT)

    assert projected["available"] is True
    assert projected["state"] == "planned_license_required_not_exported"
    assert all(projected["integrity"].values())
    assert projected["license"]["decision"] is None
    assert all(value is False for value in projected["gates"].values())


def test_projection_separates_preparation_from_execution_and_is_pathless(
    pose_workspace: Path,
) -> None:
    projected = _project(pose_workspace)

    assert projected["available"] is True
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["license"] == {
        "decision": None,
        "selected": False,
        "download_authorized": False,
        "export_authorized": False,
        "allowed_bases": [
            "AGPL-3.0 compatible",
            "Ultralytics Enterprise",
        ],
    }
    challenger = projected["permissive_challenger"]
    assert challenger["candidate"] == "MMPose YOLOX-Pose-S"
    assert challenger["license"] == "Apache-2.0"
    assert challenger["production_model_selected"] is False
    assert challenger["replaces_yolo26_selection"] is False
    assert challenger["checkpoint"] == {
        "acquired": True,
        "integrity_verified": True,
        "immutable_read_only": True,
    }
    assert challenger["cpu_structural_evidence"][
        "strict_state_load_verified"
    ] is True
    assert challenger["cpu_structural_evidence"][
        "raw_profiles_verified"
    ] == [640, 960]
    assert challenger["profiles"]["960"]["feasibility_only"] is True
    assert challenger["profiles"]["960"]["quality_verified"] is False
    assert challenger["deployment"]["dynamic_batch12_verified"] is False
    assert challenger["deployment"]["custom_parser_required"] is True
    assert challenger["deployment"]["custom_parser_implemented"] is False
    preflight = challenger["onnx_preflight"]
    assert preflight["current"] == {
        "run_id": "r2",
        "snapshot_kind": "current_immutable_preflight",
        "state": "blocked",
        "status": "blocked_preflight_no_export_attempted",
        "blocker_count": 2,
        "blocker_codes": [
            "mmdeploy_distribution_missing_or_wrong",
            "compiled_mmcv_ops_missing",
        ],
        "mmdeploy_checkout_verified": True,
        "export_attempted": False,
        "onnxruntime_executed": False,
        "batch12_executed": False,
        "deepstream9_executed": False,
        "production_ready": False,
    }
    assert preflight["historical"] == [
        {
            "run_id": "r1",
            "snapshot_kind": "historical_immutable_preflight",
            "state": "blocked",
            "status": "blocked_preflight_no_export_attempted",
            "blocker_count": 3,
            "blocker_codes": [
                "mmdeploy_checkout_missing",
                "mmdeploy_distribution_missing_or_wrong",
                "compiled_mmcv_ops_missing",
            ],
            "mmdeploy_checkout_verified": False,
            "export_attempted": False,
            "onnxruntime_executed": False,
            "batch12_executed": False,
            "deepstream9_executed": False,
            "production_ready": False,
        }
    ]
    assert preflight["progress"] == {
        "resolved_blocker_codes": ["mmdeploy_checkout_missing"],
        "remaining_blocker_codes": [
            "mmdeploy_distribution_missing_or_wrong",
            "compiled_mmcv_ops_missing",
        ],
    }
    assert preflight[
        "historical_snapshot_is_not_live_environment_state"
    ] is True
    assert preflight["production_ready"] is False
    assert challenger["quality"]["pck_640_passed"] is False
    assert challenger["quality"]["pck_960_passed"] is False
    assert challenger["production_ready"] is False
    assert all(projected["preparation"].values())
    assert projected["model_contract"]["layout"] == "COCO17"
    assert projected["model_contract"]["profiles"] == [640, 960]
    assert projected["model_contract"]["batch_opt"] == 12
    assert all(value is False for value in projected["artifacts"].values())
    assert projected["pck"]["evaluator_contract_verified"] is True
    for key in (
        "evaluation_plan_pin_declared",
        "ground_truth_pin_declared",
        "predictions_pin_declared",
        "receipt_pin_declared",
        "result_available",
    ):
        assert projected["pck"][key] is False
    assert projected["source_readiness"]["diagnostic_images_planned"] == 3
    assert projected["source_readiness"]["video_sequences"] == 0
    assert projected["source_readiness"]["overhead_or_security_views"] == 0
    assert projected["source_readiness"]["eligible_for_product_pck"] is False
    assert all(value is False for value in projected["gates"].values())

    serialized = json.dumps(projected, ensure_ascii=False)
    for private_fragment in (
        "/workspace",
        "models/pose/",
        "validation/",
        "data/manifests/",
        "https://",
        "docker.io/",
        "sha256:",
        ".pt",
        ".onnx",
        ".engine",
        "run_export.sh",
        "29c973ea",
        "14f7a83c",
        "b4fb5c98",
    ):
        assert private_fragment not in serialized


def test_unpinned_weight_exports_and_receipt_cannot_promote_pose(
    pose_workspace: Path,
) -> None:
    planted = {
        "models/pose/candidates/yolo26s-pose/yolo26s-pose.pt": b"fake-weight",
        "models/pose/artifacts/yolo26s-pose/640/yolo26s-pose-640.onnx": b"fake-onnx",
        "models/pose/artifacts/yolo26s-pose/960/fake.engine": b"fake-engine",
        "validation/results/pose/pck/current/receipt.json": b'{"quality_gate":{"status":"pass"}}',
    }
    for relative, raw in planted.items():
        path = pose_workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    projected = _project(pose_workspace)

    assert projected["available"] is True
    assert all(value is False for value in projected["artifacts"].values())
    assert projected["pck"]["receipt_pin_declared"] is False
    assert projected["pck"]["result_available"] is False
    assert all(value is False for value in projected["gates"].values())


def test_provenance_hash_drift_fails_every_pose_gate_closed(
    pose_workspace: Path,
) -> None:
    path = pose_workspace / admin_validation.POSE_EXPORT_PROVENANCE_PIN["path"]
    path.write_bytes(path.read_bytes() + b" ")

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["integrity"]["provenance_plan_verified"] is False
    assert all(value is False for value in projected["gates"].values())
    assert "pin_mismatch" in projected["reason"]


def test_export_plan_symlink_is_rejected(
    pose_workspace: Path,
) -> None:
    relative = admin_validation.POSE_EXPORT_PATHS["plan_640"]
    plan = pose_workspace / relative
    target = pose_workspace / "same-pose-plan-elsewhere.json"
    target.write_bytes(plan.read_bytes())
    plan.unlink()
    plan.symlink_to(target)

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["integrity"]["export_plan_640_verified"] is False
    assert projected["reason"] == "export_plan_640_unsafe_path"
    assert all(value is False for value in projected["gates"].values())


def test_exact_repin_cannot_turn_license_decision_into_readiness(
    pose_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = admin_validation.POSE_EXPORT_PROVENANCE_PIN["path"]
    path = pose_workspace / relative
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance["license"]["decision"] = "enterprise"
    provenance["license"]["download_authorized"] = True
    provenance["license"]["export_authorized"] = True
    raw = _write_json(path, provenance)
    monkeypatch.setattr(
        admin_validation,
        "POSE_EXPORT_PROVENANCE_PIN",
        _pin(relative, raw),
    )

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "pose_cross_artifact_contract_invalid"
    assert projected["license"]["decision"] is None
    assert all(value is False for value in projected["gates"].values())


def test_repinning_a_changed_export_plan_does_not_bypass_self_fingerprint(
    pose_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_relative = admin_validation.POSE_EXPORT_PATHS["plan_960"]
    plan_path = pose_workspace / plan_relative
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["license_gate"]["decision"] = "enterprise"
    plan_raw = _write_json(plan_path, plan)
    new_plan_pin = _pin(plan_relative, plan_raw)

    provenance_relative = admin_validation.POSE_EXPORT_PROVENANCE_PIN["path"]
    provenance_path = pose_workspace / provenance_relative
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["frozen_export_plans"]["960"].update(new_plan_pin)
    provenance_raw = _write_json(provenance_path, provenance)
    monkeypatch.setattr(
        admin_validation,
        "POSE_EXPORT_PROVENANCE_PIN",
        _pin(provenance_relative, provenance_raw),
    )

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "pose_cross_artifact_contract_invalid"
    assert projected["integrity"]["export_plan_960_verified"] is True
    assert all(value is False for value in projected["gates"].values())


def test_pose_gt_source_manifest_tamper_fails_closed(
    pose_workspace: Path,
) -> None:
    relative = admin_validation.POSE_GT_EVIDENCE_PINS["source_manifest"][
        "path"
    ]
    path = pose_workspace / relative
    path.write_bytes(path.read_bytes() + b" ")

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["integrity"]["gt_source_manifest_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_mmpose_challenger_receipt_tamper_fails_pose_projection_closed(
    pose_workspace: Path,
) -> None:
    relative = admin_validation.POSE_PERMISSIVE_CHALLENGER_PATHS["receipt"]
    path = pose_workspace / relative
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b" ")

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["integrity"][
        "permissive_challenger_receipt_verified"
    ] is False
    assert projected["permissive_challenger"] == {}
    assert all(value is False for value in projected["gates"].values())


def test_mmpose_current_onnx_preflight_tamper_fails_projection_closed(
    pose_workspace: Path,
) -> None:
    relative = admin_validation.POSE_MMPOSE_ONNX_PREFLIGHT_PINS[
        "current_r2"
    ]["path"]
    path = pose_workspace / relative
    path.chmod(0o640)
    path.write_bytes(path.read_bytes() + b" ")

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["integrity"][
        "mmpose_onnx_preflight_current_r2_verified"
    ] is False
    assert projected["permissive_challenger"] == {}
    assert all(value is False for value in projected["gates"].values())


def test_missing_owner_policy_keeps_export_and_pck_claims_closed(
    pose_workspace: Path,
) -> None:
    (pose_workspace / POLICY_PATHS[0]).unlink()

    projected = _project(pose_workspace)

    assert projected["available"] is False
    assert projected["reason"] == "owner_acceptance_policy_invalid"
    assert projected["integrity"]["owner_acceptance_policy_verified"] is False
    assert all(value is False for value in projected["gates"].values())


def test_validation_api_and_ui_expose_compact_pose_card(
    pose_workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(pose_workspace)
    )
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["pose_model_readiness"]
    assert projected["state"] == "planned_license_required_not_exported"
    assert projected["license"]["decision"] is None
    assert projected["permissive_challenger"]["license"] == "Apache-2.0"
    assert projected["permissive_challenger"]["production_ready"] is False
    assert projected["permissive_challenger"]["onnx_preflight"]["current"][
        "blocker_count"
    ] == 2
    assert projected["permissive_challenger"]["onnx_preflight"][
        "historical"
    ][0]["blocker_count"] == 3
    assert projected["artifacts"]["weights_acquired"] is False
    assert projected["gates"]["pck_640_passed"] is False
    assert projected["gates"]["production_ready"] is False
    assert page.status_code == 200
    assert "Pose modeli hazırlığı" in page.text
    assert "Açık lisans challenger" in page.text
    assert "ONNX preflight R2" in page.text
    assert "R1 tarihsel preflight" in page.text
    assert "960 yalnız fizibilite" in page.text
    assert "evaluation plan / GT / prediction / receipt pini yok" in page.text
    assert "Export/parity/PCK/kapasite/full-stack/kabul" in page.text


def test_compose_and_docs_define_pose_read_only_boundary() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    docs = (ROOT / "docs/admin-validation-dashboard.md").read_text(
        encoding="utf-8"
    )

    assert "./models:/workspace/models:ro" in compose
    assert "./data:/workspace/data:ro" in compose
    assert "./validation:/workspace/validation:ro" in compose
    assert "pose-model readiness card uses its own immutable trust boundary" in docs
    assert "does not scan model, artifact or result directories" in docs
    assert "`planned_license_required_not_exported`" in docs
