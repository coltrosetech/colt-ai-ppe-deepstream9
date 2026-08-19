from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_child_image as lane


@pytest.fixture(scope="module")
def plan() -> dict:
    return lane.build_plan(
        created_at="2026-07-18T02:30:00+00:00",
        base_local_observed=False,
        docker_version="29.4.0",
        buildx_version=(
            "github.com/docker/buildx v0.33.0 "
            "f7897eba028583e0071642db3c011e860444f8cf"
        ),
    )


def test_plan_is_hash_bound_and_replayable(plan: dict) -> None:
    assert plan["plan_sha256"] == lane.payload_sha256(plan, "plan_sha256")
    result = lane.verify_plan(plan, plan["plan_sha256"])
    assert result["valid"] is True
    assert result["image_built"] is False
    assert result["gpu_exposed"] is False
    assert result["model_exported"] is False


def test_plan_records_explicit_base_pull_network_boundary(plan: dict) -> None:
    assert plan["status"] == "planned_base_pull_required"
    assert plan["base_image"] == {
        "immutable_reference": lane.BASE_REFERENCE,
        "manifest_digest": lane.BASE_DIGEST,
        "config_digest": lane.BASE_CONFIG_DIGEST,
        "platform": {"os": "linux", "architecture": "amd64"},
        "local_at_plan_creation": False,
        "explicit_pull_required": True,
        "pull_network": "required_if_absent",
    }
    assert plan["commands"]["base_pull_if_absent"] == [
        "docker",
        "pull",
        lane.BASE_REFERENCE,
    ]


def test_wheelhouse_and_source_pins_are_exact(plan: dict) -> None:
    verification = plan["wheelhouse_verification"]
    assert verification["valid"] is True
    assert verification["manifest_sha256"] == lane.MANIFEST_PAYLOAD_SHA256
    assert verification["receipt_payload_sha256"] == lane.WHEELHOUSE_RECEIPT_PAYLOAD_SHA256
    assert verification["wheel_count"] == 69
    assert verification["source_archive_count"] == 3
    assert plan["source_pins"]["mmdeploy"]["commit"] == lane.MMDEPLOY_COMMIT
    assert plan["source_pins"]["mmpose"]["commit"] == lane.MMPOSE_COMMIT


def test_build_is_external_context_offline_and_digest_pinned(plan: dict) -> None:
    command = plan["commands"]["build"]
    assert command[:3] == ["docker", "buildx", "build"]
    assert "--load" in command
    assert "--network=none" in command
    assert "--pull=false" in command
    assert f"wheelbundle={lane.WHEELHOUSE}" in command
    assert f"BASE_IMAGE={lane.BASE_REFERENCE}" in command
    assert "--gpus" not in command
    assert command[-1] == str(lane.CHILD_ROOT)


def test_runtime_probe_command_is_read_only_non_gpu(plan: dict) -> None:
    command = plan["commands"]["runtime_probe_template"]
    assert command[:2] == ["docker", "run"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "NVIDIA_VISIBLE_DEVICES=void" in command
    assert "--gpus" not in command
    assert command[command.index("--entrypoint") + 1] == "/opt/deepsafe-export/bin/python"


def test_dockerfile_has_two_stage_offline_hash_install() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM ${BASE_IMAGE} AS builder" in source
    assert "FROM ${BASE_IMAGE} AS runtime" in source
    assert source.count("--require-hashes") >= 3
    assert source.count("--no-index") >= 4
    assert "--no-build-isolation" in source
    assert "SOURCE_DATE_EPOCH=1581638400" in source
    assert "--from=wheelbundle" in source
    assert "--gpus" not in source


def test_image_labels_bind_base_bundle_and_sources() -> None:
    source = lane.DOCKERFILE.read_text(encoding="utf-8")
    for token in (
        "deepsafe.base.manifest",
        "deepsafe.wheelhouse.manifest.file.sha256",
        "deepsafe.wheelhouse.manifest.payload.sha256",
        "deepsafe.mmdeploy.commit",
        "deepsafe.mmpose.commit",
        'deepsafe.export.device="cpu"',
        'deepsafe.gpu.exposed="false"',
    ):
        assert token in source


def test_probe_source_never_queries_gpu_api() -> None:
    source = lane.ENVIRONMENT_PROBE.read_text(encoding="utf-8")
    assert "torch.cuda." not in source
    assert "torch.cuda(" not in source
    assert "nvidia-smi" not in source
    assert "torch.version.cuda" in source
    assert 'glob.glob("/dev/nvidia*"' in source
    assert '"gpu_api_query_executed": False' in source


def test_probe_has_required_compatibility_and_cpu_gates() -> None:
    source = lane.ENVIRONMENT_PROBE.read_text(encoding="utf-8")
    for token in (
        "import mmcv._ext",
        'required = ["nms", "batched_nms", "roi_align", "deform_conv2d"]',
        "ops.nms",
        "CPUExecutionProvider",
        "onnx.checker.check_model",
        "mmpose.models.heads.hybrid_heads.yoloxpose_head",
        "mmdeploy.codebase.mmpose.models.heads.yolox_pose_head",
        'command = [sys.executable, "-m", "pip", "check"]',
    ):
        assert token in source


def test_container_helpers_parse_as_python38() -> None:
    for path in (lane.BUNDLE_VERIFY, lane.MAKE_LOCAL_LOCK, lane.ENVIRONMENT_PROBE):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))


def test_bootstrap_lock_is_exact_and_hashed() -> None:
    text = lane.BOOTSTRAP_LOCK.read_text(encoding="utf-8")
    assert text.count("--hash=sha256:") == 3
    assert "pip==25.0.1" in text
    assert "setuptools==75.3.4" in text
    assert "wheel==0.45.1" in text


def test_plan_tamper_fails_even_if_original_external_hash_is_used(plan: dict) -> None:
    tampered = copy.deepcopy(plan)
    tampered["commands"]["build"].remove("--network=none")
    tampered["plan_sha256"] = lane.payload_sha256(tampered, "plan_sha256")
    with pytest.raises(lane.ChildImageError, match="external plan pin"):
        lane.verify_plan(tampered, plan["plan_sha256"])


def test_atomic_plan_publication_refuses_overwrite(tmp_path: Path, plan: dict) -> None:
    output = tmp_path / "plan.json"
    lane.atomic_write_no_overwrite(output, plan)
    before = output.read_bytes()
    assert oct(output.stat().st_mode & 0o777) == "0o440"
    with pytest.raises(lane.ChildImageError, match="refusing to overwrite"):
        lane.atomic_write_no_overwrite(output, plan)
    assert output.read_bytes() == before


def test_attempt_id_is_bounded() -> None:
    assert lane.ATTEMPT_ID_RE.fullmatch("child-v1-r1-001")
    assert lane.ATTEMPT_ID_RE.fullmatch("../escape") is None
    assert lane.ATTEMPT_ID_RE.fullmatch("UPPERCASE") is None


def test_synthetic_image_projection_requires_exact_lineage() -> None:
    base_layers = ["sha256:a", "sha256:b"]
    labels = {
        "deepsafe.base.manifest": lane.BASE_DIGEST,
        "deepsafe.wheelhouse.manifest.file.sha256": lane.MANIFEST_FILE_SHA256,
        "deepsafe.wheelhouse.manifest.payload.sha256": lane.MANIFEST_PAYLOAD_SHA256,
        "deepsafe.mmdeploy.commit": lane.MMDEPLOY_COMMIT,
        "deepsafe.mmpose.commit": lane.MMPOSE_COMMIT,
        "deepsafe.export.device": "cpu",
        "deepsafe.gpu.exposed": "false",
    }
    image = {
        "Id": "sha256:" + "1" * 64,
        "Architecture": "amd64",
        "Os": "linux",
        "Created": "2026-07-18T00:00:00Z",
        "Size": 1,
        "VirtualSize": 1,
        "Config": {"Labels": labels},
        "RootFS": {"Layers": base_layers + ["sha256:c"]},
    }
    projection = lane._inspect_projection(image, {"RootFS": {"Layers": base_layers}})
    assert projection["base_layer_prefix_match"] is True
    broken = copy.deepcopy(image)
    broken["RootFS"]["Layers"][0] = "sha256:wrong"
    with pytest.raises(lane.ChildImageError, match="base layer prefix"):
        lane._inspect_projection(broken, {"RootFS": {"Layers": base_layers}})


def test_contract_does_not_overclaim_execution() -> None:
    contract = json.loads(lane.CONTRACT.read_text(encoding="utf-8"))
    assert all(value is False for value in contract["observed"].values())
    assert contract["build_policy"]["gpu_devices_exposed"] is False
    assert contract["build_policy"]["gpu_api_query_allowed"] is False
    assert contract["build_policy"]["model_export_allowed"] is False
