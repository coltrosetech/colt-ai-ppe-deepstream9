from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_wheelhouse as lane


BUNDLE = lane.DEFAULT_OUTPUT


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads((BUNDLE / "receipt.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verification() -> dict:
    return lane.verify_bundle(BUNDLE)


def test_published_bundle_replays_exactly(verification: dict) -> None:
    assert verification == {
        "valid": True,
        "bundle_id": "mmpose-yoloxpose-s-wheelhouse-r1",
        "manifest_sha256": "459d23b9026e1bff31a442f4d717916118e22a5f3f92316ea1b7be3d6c60d522",
        "receipt_payload_sha256": "fc829ef77a380a72c9330a4969c04d0570ff8f3b13b974190a7be6b8a3f2d8e1",
        "wheel_count": 69,
        "source_archive_count": 3,
        "offline_resolver_passed": True,
        "mmcv_static_abi_gate_passed": True,
        "mmcv_runtime_import_verified": False,
        "production_ready": False,
    }


def test_exact_target_and_critical_pins(manifest: dict) -> None:
    assert manifest["target"]["python"] == "CPython 3.8.10"
    assert manifest["target"]["abi"] == "cp38"
    assert manifest["target"]["architecture"] == "x86_64"
    assert manifest["target"]["cuda_wheel_abi"] == "11.8"
    assert manifest["critical_pins"] == lane.CRITICAL_PINS
    assert manifest["build_root_pins"] == lane.BUILD_ROOT_PINS


def test_complete_marker_aware_dependency_closure(manifest: dict) -> None:
    closure = manifest["dependency_closure"]
    assert closure["wheel_distributions"] == 69
    assert closure["active_dependency_edges"] == 107
    assert closure["missing"] == []
    assert closure["version_mismatches"] == []
    assert closure["unreachable"] == []
    names = {item["distribution"] for item in manifest["wheel_artifacts"]}
    # pip running on Python 3.12 did not select these Python<3.10 marker
    # dependencies automatically; the lane must keep the target replay exact.
    assert {"importlib-resources", "tomli", "zipp"} <= names


def test_full_mmcv_compiled_extension_is_statically_abi_paired(manifest: dict) -> None:
    probe = manifest["mmcv_compiled_extension"]
    assert probe["distribution"] == "mmcv"
    assert probe["variant"] == "compiled_full_not_mmcv_lite"
    assert probe["version"] == "2.0.1"
    assert probe["official_index_abi"] == "cu118/torch2.0.0"
    assert probe["extension_path"] == "mmcv/_ext.cpython-38-x86_64-linux-gnu.so"
    assert probe["elf"] == {"class": 64, "endianness": "little", "machine": "x86-64"}
    assert {
        "libc10.so",
        "libc10_cuda.so",
        "libcudart.so.11.0",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
    } <= set(probe["dynamic_needed"])
    assert probe["torch_wheel_version"] == "2.0.0+cu118"
    assert probe["torch_cuda_build"] == "11.8"
    assert probe["static_abi_gate_passed"] is True
    assert probe["runtime_import_executed"] is False


def test_every_artifact_has_hash_bytes_origin_and_license(manifest: dict) -> None:
    assert manifest["acquisition_authorities"]["unapproved_sources"] == []
    assert manifest["license_boundary"]["missing_wheel_declarations"] == []
    for artifact in manifest["wheel_artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["bytes"] > 0
        assert artifact["origin"]["index_metadata_verified"] is True
        assert artifact["license"]["status"] != "missing_upstream_wheel_declaration"
        assert artifact["license"]["legal_approval"] is False
    for artifact in manifest["source_artifacts"]:
        assert len(artifact["sha256"]) == 64
        assert artifact["bytes"] > 0
        assert artifact["source_url"].startswith("https://")
        assert artifact["license"]["legal_approval"] is False


def test_local_openmmlab_sources_are_exact(manifest: dict) -> None:
    sources = {item["component"]: item for item in manifest["source_artifacts"]}
    assert sources["mmdeploy"]["commit"] == lane.MMDEPLOY_COMMIT
    assert sources["mmdeploy"]["git_tree"] == lane.MMDEPLOY_TREE
    assert sources["mmpose"]["commit"] == lane.MMPOSE_COMMIT
    assert sources["mmpose"]["git_tree"] == lane.MMPOSE_TREE
    assert sources["mmdeploy"]["license"]["spdx"] == "Apache-2.0"
    assert sources["mmpose"]["license"]["spdx"] == "Apache-2.0"


def test_chumpy_wheel_is_reproducible_unmodified_pypi_derivation(manifest: dict) -> None:
    wheels = {item["distribution"]: item for item in manifest["wheel_artifacts"]}
    chumpy = wheels["chumpy"]
    assert chumpy["sha256"] == lane.CHUMPY_WHEEL_SHA256
    assert chumpy["wheel_tags"] == ["py3-none-any"]
    assert chumpy["origin"]["source_sha256"] == lane.CHUMPY_SOURCE_SHA256
    derivation = chumpy["origin"]["derivation"]
    assert derivation["source_modified"] is False
    assert derivation["repeat_builds_byte_identical"] == 2
    assert derivation["source_date_epoch"] == "1581638400"


def test_hash_locked_offline_resolver_passed_without_install(receipt: dict) -> None:
    resolver = receipt["offline_resolver"]
    assert resolver["passed"] is True
    assert resolver["exit_code"] == 0
    assert resolver["network"] == "disabled_by_no_index"
    assert resolver["installation_executed"] is False
    assert resolver["resolver_mode"] == "download_only_to_disposable_directory"
    assert "--no-index" in resolver["command"]
    assert "--require-hashes" in resolver["command"]
    lock = (BUNDLE / "requirements-wheels.lock.txt").read_text(encoding="utf-8")
    assert lock.count("--hash=sha256:") == 69


def test_execution_boundary_and_remaining_runtime_gates_are_honest(receipt: dict) -> None:
    boundary = receipt["execution_boundary"]
    assert boundary["network_used_for_official_artifact_acquisition_and_origin_verification"] is True
    for key in (
        "package_installed",
        "host_venv_mutated",
        "docker_image_pulled",
        "docker_image_built",
        "container_run",
        "gpu_touched",
        "onnx_exported",
        "tensorrt_executed",
        "deepstream_executed",
    ):
        assert boundary[key] is False
    conclusions = receipt["conclusions"]
    assert conclusions["offline_binary_resolver_passed"] is True
    assert conclusions["mmcv_ext_runtime_import_verified"] is False
    assert conclusions["child_image_built"] is False
    assert conclusions["export_environment_runtime_ready"] is False
    assert conclusions["production_ready"] is False


def test_receipt_payload_tamper_is_detectable(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["conclusions"]["production_ready"] = True
    assert tampered["receipt_payload_sha256"] != lane.payload_sha256(
        tampered, "receipt_payload_sha256"
    )


def test_materializer_refuses_existing_output_before_network(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(lane.WheelhouseError, match="refusing to overwrite"):
        lane.materialize(
            staged_wheelhouse=tmp_path / "does-not-matter",
            chumpy_source=tmp_path / "does-not-matter.tar.gz",
            output=output,
        )


def test_published_files_are_read_only() -> None:
    assert oct(BUNDLE.stat().st_mode & 0o777) == "0o550"
    assert oct((BUNDLE / "wheelhouse").stat().st_mode & 0o777) == "0o550"
    assert oct((BUNDLE / "sources").stat().st_mode & 0o777) == "0o550"
    for path in BUNDLE.rglob("*"):
        if path.is_file():
            assert oct(path.stat().st_mode & 0o777) == "0o440"
