from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from models.person import train_yolo26 as training


ROOT = Path(__file__).resolve().parents[1]


def test_live_prepared_dataset_is_verified_and_test_is_excluded() -> None:
    value = training.verify_prepared_dataset(ROOT)

    assert value["train_frames"] == 1524
    assert value["calibration_frames"] == 384
    assert value["person_instances"] == 16652
    assert value["official_test_output_frames"] == 0


def test_dry_run_training_plan_is_pinned_and_not_an_acceptance_result(tmp_path: Path) -> None:
    output = ROOT / "models/person/training/test-dry-run-does-not-exist"
    assert not output.exists()
    value = training.build_plan(output_dir=output, project_root=ROOT)

    assert value["status"] == "planned_license_required_not_executed"
    assert value["candidate"]["sha256"] == training.MODEL_SPEC["sha256"]
    assert "@sha256:" in value["runtime"]["container_image"]
    assert value["runtime"]["implementation"]["harness"]["sha256"]
    assert value["runtime"]["implementation"]["wrapper"]["sha256"]
    assert value["training_arguments"]["imgsz"] == 960
    assert value["training_arguments"]["batch"] == 4
    assert value["training_arguments"]["device"] == 0
    assert value["training_arguments"]["data"] == (
        "/workspace/data/derived/r-livit/person-finetune-v1/dataset.yaml"
    )
    assert value["held_out_guardrails"]["r_livit_official_test_used"] is False
    assert value["held_out_guardrails"]["loaf_20_to_25m_used"] is False
    assert value["license_gate"]["decision_recorded_in_plan"] is False
    assert value["acceptance_effect"].startswith("none_until")
    assert value["outputs"]["directory"] == (
        "/workspace/models/person/training/test-dry-run-does-not-exist"
    )
    assert not output.exists()
    json.dumps(value, allow_nan=False)


def test_execute_license_gate_precedes_network_and_gpu() -> None:
    with pytest.raises(training.TrainingContractError, match="Refusing Docker"):
        training.license_gate(execute=True, accepted=False, basis=None)
    with pytest.raises(training.TrainingContractError, match="Refusing Docker"):
        training.license_gate(execute=True, accepted=True, basis=None)
    training.license_gate(execute=True, accepted=True, basis="enterprise")
    training.license_gate(execute=True, accepted=True, basis="agpl-3.0")


def test_shell_wrapper_refuses_before_docker_when_license_flags_are_missing() -> None:
    result = subprocess.run(
        ["bash", "models/person/run_yolo26_training.sh", "--execute"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing Docker pull/GPU execution" in result.stderr


def test_shell_wrapper_requires_frozen_plan_before_docker() -> None:
    result = subprocess.run(
        [
            "bash",
            "models/person/run_yolo26_training.sh",
            "--execute",
            "--accept-ultralytics-license",
            "--license-basis",
            "enterprise",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "exact frozen plan fingerprint" in result.stderr


def test_manifest_tamper_is_rejected_before_training(tmp_path: Path) -> None:
    plan = json.loads(training.UPGRADE_PLAN.read_text(encoding="utf-8"))
    pin = plan["training_data"]["primary"]["prepared_dataset_manifest"]
    fixture = tmp_path / pin["path"]
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"status":"prepared_cpu_only"}\n', encoding="utf-8")
    plan_path = tmp_path / training.UPGRADE_PLAN.relative_to(training.PROJECT_ROOT)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(training.TrainingContractError, match="byte count differs"):
        training.verify_prepared_dataset(tmp_path)
