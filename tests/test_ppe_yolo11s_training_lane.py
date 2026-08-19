from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from validation.ppe_yolo11s_training_lane import (
    PpeTrainingLaneError,
    launch,
    verify_plan,
)


ROOT = Path(__file__).resolve().parents[1]
LANE = (
    ROOT
    / "models/ppe/training-lanes/yolo11s-mendeley-five-class-development-r1"
)
PLAN = LANE / "training-plan-r1.json"
SCHEMA = LANE / "training-plan-v1.schema.json"
DOCKERFILE = LANE / "Dockerfile"
ENTRYPOINT = LANE / "train_entrypoint.py"
CONTAINER_YAML = LANE / "dataset-container.yaml"
LAUNCHER = ROOT / "validation/ppe_yolo11s_training_lane.py"
PLAN_SHA256 = "cab02a9e3588b264bda7880109d7eccd446753c45e7c437361145d7763b1e91d"
IMAGE = (
    "docker.io/ultralytics/ultralytics@sha256:"
    "36e457c94f0c6fed5e99b109124567ef36f1d3b58435860771636a677fbaed8a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def verified() -> dict:
    return verify_plan(
        PLAN,
        expected_plan_sha256=PLAN_SHA256,
        root=ROOT,
        schema_path=SCHEMA,
    )


def test_training_plan_is_schema_valid_externally_pinned_and_immutable() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(plan)
    assert PLAN.stat().st_size == 7150
    assert _sha256(PLAN) == PLAN_SHA256
    assert plan["runtime"]["container_image"] == IMAGE
    assert plan["runtime"]["ultralytics_version"] == "8.4.99"
    assert plan["model"]["checkpoint"]["sha256"] == (
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
    )


def test_verifier_replays_dataset_and_never_invokes_gpu_or_docker(verified: dict) -> None:
    assert verified["valid"] is True
    assert verified["images"] == 2586
    assert verified["bbox_rows"] == 17827
    assert verified["leakage_zero"] is True
    assert verified["docker_invoked"] is False
    assert verified["gpu_queried"] is False
    assert verified["training_executed"] is False
    assert verified["training_authorized"] is False
    assert verified["production_eligible"] is False


def test_dockerfile_is_digest_pinned_and_has_no_dependency_install() -> None:
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert content.splitlines()[0] == f"FROM {IMAGE}"
    assert "8.4.99" in content
    assert "pip install" not in content
    assert "apt-get" not in content
    assert "curl " not in content
    assert "wget " not in content
    assert _sha256(DOCKERFILE) == (
        "597e61e839e1a61a75e9d7a7767b5e8f3a47c7bd4ae16db00182328a7addde89"
    )


def test_entrypoint_checks_all_authorization_gates_before_ultralytics_import() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    import_offset = source.index("import ultralytics")
    for gate in (
        'execution["container_build_authorized"]',
        'execution["gpu_query_authorized"]',
        'execution["training_authorized"]',
        'plan["license_gate"]["decision_recorded"]',
        'plan["data_rights_gate"]["development_use_approved"]',
    ):
        assert source.index(gate) < import_offset
    assert _sha256(ENTRYPOINT) == (
        "d7534475a428165fd1b72cd9505eda0b11fb7b58fa12b336fbb0eeb34976198f"
    )


def test_container_yaml_uses_an_absolute_workspace_root() -> None:
    content = CONTAINER_YAML.read_text(encoding="utf-8")
    assert (
        "path: /workspace/data/derived/ppe/"
        "mendeley-ppe-five-class-v1-development-r3\n"
    ) in content
    assert "train: images/train\n" in content
    assert "val: images/development_holdout\n" in content
    assert "\ntest:" not in "\n" + content
    assert _sha256(CONTAINER_YAML) == (
        "9778f8559e25ad1629e5d3dfc255c504bf26a61054de8f4d53b5db0f622d2fac"
    )


def test_launch_refuses_before_any_subprocess_or_output() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "docker run" not in source
    with pytest.raises(PpeTrainingLaneError, match="not authorized"):
        launch(PLAN, expected_plan_sha256=PLAN_SHA256, root=ROOT)
    output = (
        ROOT
        / "models/ppe/training/yolo11s-mendeley-five-class-development-r1"
    )
    assert not output.exists()


def test_wrong_external_plan_pin_fails_closed() -> None:
    with pytest.raises(PpeTrainingLaneError, match="external training plan pin"):
        verify_plan(
            PLAN,
            expected_plan_sha256="0" * 64,
            root=ROOT,
            schema_path=SCHEMA,
        )


def test_dataset_roles_are_development_only_and_no_final_test_is_claimed() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["dataset"]["roles"] == {
        "train": "development_only_model_fitting_candidate",
        "calibration": "development_only_calibration_candidate",
        "development_holdout": "development_only_not_final_test",
    }
    assert plan["dataset"]["final_test_role_present"] is False
    assert plan["heldout_guardrails"]["development_holdout_is_final_test"] is False
    assert plan["heldout_guardrails"]["independent_final_test_available"] is False


def test_rights_authorization_and_production_acceptance_are_distinct() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert all(value is False for value in plan["execution_gate"].values())
    assert plan["license_gate"]["decision_recorded"] is False
    assert plan["data_rights_gate"]["development_use_approved"] is False
    development = plan["blocker_partitions"]["development_training_authorization"]
    production = plan["blocker_partitions"]["production_acceptance"]
    assert "ultralytics_license_basis_selection_required" in development
    assert "new_explicit_training_authorization_plan_required" in development
    assert "independent_final_test_dataset_required" in production
    assert "twelve_stream_640_and_960_endurance_required" in production
