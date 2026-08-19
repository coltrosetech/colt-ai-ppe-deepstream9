import copy
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from deepstream import full_stack, parallel_runtime


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "deepstream/parallel-runtime-build-plan.json"
PUBLICATION = ROOT / "models/runtime/deepstream-parallel-infer-ds9-9946965e-r2"


def test_frozen_plan_hash_source_and_image_identity_are_exact():
    raw = PLAN.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == parallel_runtime.FROZEN_PLAN_SHA256
    plan, digest = parallel_runtime.load_frozen_plan()
    assert digest == "ea22142ac35a10c84f89f5d452153b95379495f258a8533ff432bd267f4310d1"
    assert plan["source"]["commit"] == "9946965e8adb1aa93b1b66983ec4196351c9190c"
    assert plan["source"]["tree"] == "4c4382ed32fad08767d01cd7bfbf464bd6be0e37"
    assert plan["container"]["image_reference"] == parallel_runtime.EXPECTED_IMAGE
    assert plan["source"]["license"]["spdx"] == "Apache-2.0"


def test_every_upstream_build_input_is_read_from_the_frozen_git_object():
    plan, _ = parallel_runtime.load_frozen_plan()
    result = parallel_runtime.verify_source(plan)
    assert result["commit"] == parallel_runtime.EXPECTED_COMMIT
    assert result["tree"] == parallel_runtime.EXPECTED_TREE
    assert result["build_input_count"] == 8
    assert result["license_files_verified"] is True
    evidence = result["static_feature_evidence"]
    assert evidence["nvdsmetamux_factory"] == 2
    assert evidence["deepsafe_fusion_config"] == 0
    assert evidence["helmet"] == 0
    assert evidence["hi_vis"] == 0


def test_build_and_probe_argv_are_networkless_and_do_not_request_a_gpu():
    plan, _ = parallel_runtime.load_frozen_plan()
    templates = [
        plan["build"]["docker_argv_template"],
        plan["probes"]["docker_common_argv_template"],
    ]
    for argv in templates:
        assert "--pull=never" in argv
        assert "--network=none" in argv
        assert "--runtime=runc" in argv
        assert "--env=NVIDIA_VISIBLE_DEVICES=void" in argv
        assert "--env=NVIDIA_DRIVER_CAPABILITIES=none" in argv
        assert "--env=CUDA_VISIBLE_DEVICES=-1" in argv
        assert not any(token.startswith("--gpus") for token in argv)
        assert not any(token.startswith("--device") for token in argv)
        assert "--runtime=nvidia" not in argv
    assert "--read-only" in templates[1]
    assert plan["build"]["uses_gpu"] is False
    assert plan["build"]["network"] == "none"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["source"].__setitem__("tree", "0" * 40), "source commit/tree"),
        (lambda value: value["container"].__setitem__("image_id", "sha256:" + "0" * 64), "container identity"),
        (lambda value: value["build"].__setitem__("uses_gpu", True), "networkless and GPU-free"),
        (lambda value: value["build"]["docker_argv_template"].append("--gpus=all"), "device or NVIDIA"),
    ],
)
def test_plan_validation_fails_closed_on_identity_or_gpu_drift(mutator, message):
    plan, _ = parallel_runtime.load_frozen_plan()
    changed = copy.deepcopy(plan)
    mutator(changed)
    with pytest.raises(parallel_runtime.ParallelRuntimeError, match=message):
        parallel_runtime.validate_plan(changed)


def test_published_binary_and_receipts_are_single_link_read_only_snapshots():
    expected = {
        "deepstream-parallel-infer": (0o550, "f71188ef37e323c9294b5af766ba689dbce81f3b8cb36fa03f30fa4a09dfcd4e"),
        "build-provenance.json": (0o440, "84eef9dab07324ec2d5f850e8a3fa32518c371a0e61c3446e076a146b531188f"),
        "capability-manifest.json": (0o440, "0527c9e5002c7ec13dce0c7721775585ec985b7fabac15dbf500f387c4304dca"),
    }
    assert stat.S_IMODE(PUBLICATION.stat().st_mode) == 0o550
    for name, (mode, digest) in expected.items():
        path = PUBLICATION / name
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode)
        assert not stat.S_ISLNK(info.st_mode)
        assert info.st_nlink == 1
        assert stat.S_IMODE(info.st_mode) == mode
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_published_provenance_has_static_elf_and_container_abi_evidence():
    provenance = json.loads((PUBLICATION / "build-provenance.json").read_text())
    assert provenance["schema_version"] == parallel_runtime.PROVENANCE_SCHEMA_VERSION
    assert provenance["status"] == "build_and_static_abi_probe_passed"
    assert provenance["builder"] == {
        "path": "deepstream/parallel_runtime.py",
        "sha256": "8ea77fa12b4075c990b1a8cdb6e8ab9a838a6b58583a59960320a0652d2bb20e",
    }
    assert provenance["artifact"] == {
        "name": "deepstream-parallel-infer",
        "sha256": "f71188ef37e323c9294b5af766ba689dbce81f3b8cb36fa03f30fa4a09dfcd4e",
        "size_bytes": 4716384,
        "elf_type": "DYN (Position-Independent Executable file)",
        "elf_machine": "Advanced Micro Devices X86-64",
    }
    assert provenance["probe"]["scope"] == "static_elf_and_container_abi_only"
    assert provenance["probe"]["elf"]["class"] == "ELF64"
    assert provenance["probe"]["elf"]["interpreter"] == "/lib64/ld-linux-x86-64.so.2"
    assert provenance["probe"]["abi"]["unresolved_sonames"] == []
    assert "libnvdsgst_meta.so" in provenance["probe"]["elf"]["needed_sonames"]
    assert provenance["safety"] == {
        "docker_network_disabled": True,
        "nvidia_runtime_used": False,
        "gpu_device_injected": False,
        "gpu_queried": False,
        "inference_executed": False,
        "model_or_engine_loaded": False,
        "endurance_executed": False,
    }


def test_capability_is_explicitly_blocked_and_never_claims_association():
    capability = json.loads((PUBLICATION / "capability-manifest.json").read_text())
    assert capability["schema_version"] == full_stack.CAPABILITY_SCHEMA_VERSION
    assert capability["status"] == "blocked"
    assert capability["runtime_ready"] is False
    assert capability["features"]["pose_tensor_track_association"] is False
    assert capability["features"]["ppe_person_association"] is False
    assert capability["static_evidence"]["deepsafe_fusion_config_consumed"] is False
    assert capability["static_evidence"]["binary_token_counts"]["deepsafe_fusion_config"] == 0
    assert {item["code"] for item in capability["blockers"]} == {
        "deepsafe_fusion_config_not_consumed",
        "pose_tensor_track_association_not_implemented",
        "ppe_person_association_not_implemented",
        "gpu_runtime_probe_not_executed",
        "full_stack_inference_not_executed",
    }


def test_publication_verifier_rechecks_all_cross_artifact_bindings():
    result = parallel_runtime.inspect_publication()
    assert result["binary_sha256"] == "f71188ef37e323c9294b5af766ba689dbce81f3b8cb36fa03f30fa4a09dfcd4e"
    assert result["provenance_sha256"] == "84eef9dab07324ec2d5f850e8a3fa32518c371a0e61c3446e076a146b531188f"
    assert result["capability_manifest_sha256"] == "0527c9e5002c7ec13dce0c7721775585ec985b7fabac15dbf500f387c4304dca"
    assert result["runtime_ready"] is False


def test_atomic_publication_refuses_to_replace_an_existing_directory(tmp_path):
    plan, _ = parallel_runtime.load_frozen_plan()
    provenance = b'{"kind":"test-provenance"}\n'
    capability = b'{"kind":"test-capability"}\n'
    first = parallel_runtime._publish_directory(
        plan,
        project_root=tmp_path,
        binary=b"ELF-test",
        provenance=provenance,
        capability=capability,
    )
    try:
        assert first.is_dir()
        with pytest.raises(parallel_runtime.ParallelRuntimeError, match="already exists"):
            parallel_runtime._publish_directory(
                plan,
                project_root=tmp_path,
                binary=b"replacement",
                provenance=provenance,
                capability=capability,
            )
        assert (first / "deepstream-parallel-infer").read_bytes() == b"ELF-test"
    finally:
        first.chmod(0o750)


def test_default_full_stack_contract_binds_binary_but_rejects_blocked_capability(tmp_path):
    plan = full_stack.build_plan(
        ROOT / "deepstream/full-stack-contract.json",
        profile_id="640",
        sources=[f"file:///fixtures/camera-{index:02d}.mp4" for index in range(12)],
        output_dir=tmp_path / "blocked",
        project_root=ROOT,
    )
    checks = {item["label"]: item for item in plan["artifact_checks"]}
    assert checks["runtime.parallel_app_binary"]["status"] == "ready"
    assert checks["runtime.capability_manifest"]["status"] == "json_status_mismatch"
    assert "artifact:runtime.capability_manifest:json_status_mismatch" in plan["readiness_blockers"]
    assert plan["execution_ready"] is False
    assert plan["launch_authorized"] is False
    assert plan["launch"] is None


def test_direct_cli_help_does_not_require_package_installation():
    result = subprocess.run(
        [sys.executable, str(ROOT / "deepstream/parallel_runtime.py"), "--help"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "without GPU or network" in result.stdout
