from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from validation import product_acceptance_policy as policy


ROOT = Path(__file__).resolve().parents[1]


def _policy_workspace(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for relative in (policy.POLICY_RELATIVE_PATH, policy.SCHEMA_RELATIVE_PATH):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return project


def test_admin_projects_exact_approved_policy_without_claiming_acceptance(
    tmp_path: Path,
) -> None:
    project = _policy_workspace(tmp_path)
    reader = admin_validation.ArtifactReader(
        tmp_path / "missing-results", workspace_root=project
    )

    projected = admin_validation._product_acceptance_policy(reader)

    assert projected["available"] is True
    assert projected["state"] == "approved_not_evaluated"
    assert projected["acceptance_state"] == "not_evaluated"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["does_not_imply_product_readiness"] is True
    assert projected["read_only"] is True
    assert projected["execution_actions_available"] is False
    assert projected["policy"]["pre_run_fingerprint_sha256"] == (
        policy.APPROVED_POLICY_FINGERPRINT_SHA256
    )
    assert projected["scope"]["required_modules"] == ["person", "pose", "ppe"]
    assert projected["scope"]["model_input_sizes"] == [640, 960]
    assert projected["scope"]["simulated_streams"] == 12
    assert projected["quality_thresholds"]["pose_pck_at_0_2"] == 0.80
    assert projected["capacity"]["minimum_output_fps_per_camera"] == 25.0
    assert projected["endurance"]["total_seconds"] == 604800


def test_admin_policy_projection_fails_closed_on_resigned_threshold_tamper(
    tmp_path: Path,
) -> None:
    project = _policy_workspace(tmp_path)
    policy_path = project / policy.POLICY_RELATIVE_PATH
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    value["quality_thresholds"]["person"]["recall"]["threshold"] = 0.01
    value["pre_run_fingerprint_sha256"] = policy.policy_fingerprint_sha256(value)
    policy_path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    reader = admin_validation.ArtifactReader(
        tmp_path / "missing-results", workspace_root=project
    )

    projected = admin_validation._product_acceptance_policy(reader)

    assert projected["available"] is False
    assert projected["state"] == "artifact_error"
    assert projected["ready"] is False
    assert projected["final_claim_allowed"] is False
    assert projected["quality_thresholds"] == {}


def test_validation_api_exposes_policy_projection_not_raw_document(
    tmp_path: Path, monkeypatch
) -> None:
    project = _policy_workspace(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(project))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT", str(ROOT / "validation/schemas")
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["product_acceptance_policy"]
    assert projected["state"] == "approved_not_evaluated"
    assert projected["final_claim_allowed"] is False
    assert "approval_statement_excerpt" not in json.dumps(projected)
    assert page.status_code == 200
    assert "approved_not_evaluated" in page.text
    assert "Politika tek başına kabul" in page.text


def test_admin_image_contains_policy_verifier() -> None:
    dockerfile = (ROOT / "admin/Dockerfile").read_text(encoding="utf-8")
    assert (
        "COPY validation/product_acceptance_policy.py "
        "/app/validation/product_acceptance_policy.py"
    ) in dockerfile
