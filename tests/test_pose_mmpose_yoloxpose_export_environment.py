from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_export_environment as lane


SCHEMA = json.loads(lane.SCHEMA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(created_at="2026-07-18T00:32:15+00:00")


def test_plan_is_honestly_blocked_without_materialization(plan: dict) -> None:
    assert plan["status"] == "blocked_materialization_not_attempted"
    assert {item["code"] for item in plan["blockers"]} == {
        "full_wheel_hash_lock_missing",
        "deterministic_source_archives_missing",
        "child_image_not_built",
        "environment_probe_not_executed",
    }
    assert plan["conclusions"]["export_environment_ready"] is False
    assert plan["conclusions"]["production_ready"] is False


def test_schema_accepts_exact_plan(plan: dict) -> None:
    jsonschema.Draft202012Validator(SCHEMA).validate(plan)


def test_plan_self_hash_and_replay_are_exact(plan: dict) -> None:
    assert plan["plan_sha256"] == lane.plan_sha256(plan)
    result = lane.verify_plan(plan, expected_plan_sha256=plan["plan_sha256"])
    assert result["valid"] is True
    assert result["image_built"] is False
    assert result["export_executed"] is False


def test_external_hash_tamper_fails(plan: dict) -> None:
    with pytest.raises(lane.ExportEnvironmentError, match="external plan pin"):
        lane.verify_plan(plan, expected_plan_sha256="0" * 64)


def test_semantic_tamper_fails_even_if_resealed(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["blockers"][0]["detail"] = "forged"
    tampered["plan_sha256"] = lane.plan_sha256(tampered)
    with pytest.raises(lane.ExportEnvironmentError, match="plan replay differs"):
        lane.verify_plan(tampered, expected_plan_sha256=tampered["plan_sha256"])


def test_schema_rejects_execution_overclaim(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["execution_boundary"]["docker_image_built"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(SCHEMA).validate(tampered)


def test_schema_rejects_unknown_top_level_key(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(SCHEMA).validate(tampered)


def test_official_base_is_manifest_digest_pinned(plan: dict) -> None:
    base = plan["base_image"]
    assert base["tag_reference"] == lane.BASE_TAG
    assert base["immutable_reference"] == lane.BASE_IMMUTABLE
    assert base["manifest_digest"] == lane.BASE_DIGEST
    assert base["config_digest"] == lane.BASE_CONFIG_DIGEST
    assert base["platform"] == {"os": "linux", "architecture": "amd64"}
    assert base["registry_observation"]["read_only"] is True
    assert base["registry_observation"]["image_pulled"] is False


def test_exact_critical_versions_and_compiled_mmcv_gate(plan: dict) -> None:
    assert lane._critical_versions(plan) == lane.EXPECTED_CRITICAL_PINS
    mmcv = plan["critical_compatibility_pins"]["mmcv"]
    assert mmcv["distribution"] == "mmcv"
    assert mmcv["version"] == "2.0.1"
    assert mmcv["variant"] == "compiled_full_not_mmcv_lite"
    assert mmcv["required_module"] == "mmcv._ext"
    assert plan["compatibility_assessment"]["mmcv_compiled_variant_observed"] is False


def test_host_python_and_torch_are_not_reused(plan: dict) -> None:
    assessment = plan["compatibility_assessment"]
    assert assessment["python_3_12_host_environment_reused"] is False
    assert assessment["torch_2_13_host_environment_reused"] is False
    assert assessment["child_environment_isolated"] is True
    assert assessment["child_environment_materialized"] is False


def test_exact_official_source_checkouts_are_clean(plan: dict) -> None:
    assert plan["source_checkouts"] == {
        "mmdeploy": {
            "path": "third_party/mmdeploy",
            "commit": lane.MMDEPLOY_COMMIT,
            "git_tree": lane.MMDEPLOY_TREE,
            "clean": True,
        },
        "mmpose": {
            "path": "third_party/mmpose",
            "commit": lane.MMPOSE_COMMIT,
            "git_tree": lane.MMPOSE_TREE,
            "clean": True,
        },
    }


def test_critical_requirements_cannot_masquerade_as_hash_lock(plan: dict) -> None:
    lane._validate_requirements()
    text = lane.CRITICAL_REQUIREMENTS.read_text(encoding="utf-8")
    assert "NOT an install lock" in text
    assert "--hash=" not in text
    assert plan["conclusions"]["full_wheel_lock_ready"] is False


def test_build_and_export_templates_are_offline_and_digest_gated(plan: dict) -> None:
    commands = plan["planned_commands_not_executed"]
    build = commands["offline_child_build"]
    assert "--network=none" in build
    assert "--pull=false" in build
    assert f"BASE_IMAGE={lane.BASE_IMMUTABLE}" in build
    for profile in ("export_640", "export_960"):
        command = commands[profile]
        assert "--network=none" in command
        assert "--read-only" in command
        assert "CUDA_VISIBLE_DEVICES=" in command
        assert "<CHILD_IMAGE_IMMUTABLE_REFERENCE>" in command
        assert command[-2:] == ["--device", "cpu"]


def test_tensorrt_and_deepstream_are_strictly_separate(plan: dict) -> None:
    phase = plan["phase_separation"]
    assert phase["phase_1_exporter"]["may_build_tensorrt"] is False
    assert phase["phase_1_exporter"]["may_run_deepstream"] is False
    ds9 = phase["phase_3_deepstream9_tensorrt"]
    assert ds9["deepstream"] == "9.0.0"
    assert ds9["cuda"] == "13.1"
    assert ds9["tensorrt"] == "10.14.1.48"
    assert ds9["exporter_tensorrt_8_6_engine_reusable"] is False
    assert ds9["onnx_to_tensorrt_10_14_compatibility_claimed"] is False
    assert ds9["engine_build_executed"] is False


def test_license_boundary_does_not_overclaim_redistribution(plan: dict) -> None:
    licenses = plan["licenses"]
    assert {item["component"] for item in licenses["locally_byte_verified"]} == {
        "MMDeploy source",
        "MMPose source and checkpoint codebase",
    }
    assert "NVIDIA proprietary" in licenses["mixed_base_notice"]
    assert "SBOM" in licenses["distribution_gate"]
    assert plan["required_future_evidence"]["license_sbom_review"] is True


def test_environment_probe_source_has_fail_closed_runtime_gates() -> None:
    lane._validate_probe_source()
    source = lane.ENVIRONMENT_PROBE.read_text(encoding="utf-8")
    assert '"mmcv._ext"' in source
    assert "torch.cuda.is_available()" in source
    assert "CPUExecutionProvider" in source
    assert "mmcv-lite must not coexist" in source


def test_atomic_publication_refuses_overwrite(tmp_path: Path, plan: dict) -> None:
    output = tmp_path / "plan.json"
    lane.atomic_write_no_overwrite(output, plan)
    before = output.read_bytes()
    assert oct(output.stat().st_mode & 0o777) == "0o440"
    with pytest.raises(lane.ExportEnvironmentError, match="refusing to overwrite"):
        lane.atomic_write_no_overwrite(output, plan)
    assert output.read_bytes() == before


def test_atomic_publication_rejects_symlinked_parent(tmp_path: Path, plan: dict) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(lane.ExportEnvironmentError, match="symlink"):
        lane.atomic_write_no_overwrite(link / "plan.json", plan)


def test_contract_rejects_trt_compatibility_overclaim() -> None:
    contract = lane._strict_json(lane.CONTRACT)
    contract["phase_separation"]["phase_3_deepstream9_tensorrt"][
        "onnx_to_tensorrt_10_14_compatibility_claimed"
    ] = True
    with pytest.raises(lane.ExportEnvironmentError, match="TRT compatibility overclaimed"):
        lane._validate_contract(contract)
