from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from models.pose import export_pose as exporting
from models.pose.validate_onnx import ContractError, TensorBinding, validate_bindings


ROOT = Path(__file__).resolve().parents[1]


def valid_contract(**overrides):
    inputs = overrides.get(
        "inputs", [TensorBinding("images", "FLOAT", ("batch", 3, "height", "width"))]
    )
    outputs = overrides.get(
        "outputs", [TensorBinding("output0", "FLOAT", ("batch", 300, 57))]
    )
    metadata = overrides.get("metadata", {"task": "pose", "kpt_shape": "[17, 3]"})
    return validate_bindings(inputs, outputs, metadata, imgsz=640)


def test_pose_contract_accepts_dynamic_batch_and_17_keypoints():
    report = valid_contract()
    assert report["valid"] is True
    assert report["expected"]["output"] == ["B", 300, 57]
    assert report["metadata"]["kpt_shape"] == [17, 3]


@pytest.mark.parametrize(
    "outputs,error",
    [
        ([TensorBinding("output0", "FLOAT", ("batch", 300, 56))], "57"),
        ([TensorBinding("output0", "INT32", ("batch", 300, 57))], "FLOAT/FLOAT16"),
        ([TensorBinding("output0", "FLOAT", (1, 300, 57))], "dynamic"),
    ],
)
def test_pose_contract_rejects_wrong_output(outputs, error):
    with pytest.raises(ContractError, match=error):
        valid_contract(outputs=outputs)


def test_pose_contract_rejects_wrong_keypoint_metadata():
    with pytest.raises(ContractError, match="kpt_shape"):
        valid_contract(metadata={"task": "pose", "kpt_shape": "[17, 2]"})


def test_execute_is_license_gated_before_network_docker_or_artifacts():
    with pytest.raises(exporting.PoseExportContractError, match="Refusing Docker"):
        exporting.license_gate(execute=True, accepted=False, basis=None)
    with pytest.raises(exporting.PoseExportContractError, match="Refusing Docker"):
        exporting.license_gate(execute=True, accepted=True, basis=None)
    exporting.license_gate(execute=True, accepted=True, basis="enterprise")
    exporting.license_gate(execute=True, accepted=True, basis="agpl-3.0")


def test_two_frozen_profiles_share_semantics_but_not_onnx_or_engine_paths():
    plans = {profile: exporting.verify_frozen_plan(profile=profile) for profile in (640, 960)}

    assert plans[640]["shared_semantic_contract"] == plans[960]["shared_semantic_contract"]
    assert plans[640]["artifacts"]["onnx"] != plans[960]["artifacts"]["onnx"]
    assert plans[640]["artifacts"]["tensorrt_engine"] != plans[960]["artifacts"]["tensorrt_engine"]
    for profile, plan in plans.items():
        assert plan["status"] == "planned_license_required_not_executed"
        assert plan["profile"] == {"name": str(profile), "height": profile, "width": profile}
        assert plan["export"]["imgsz"] == profile
        assert plan["export"]["batch"] == 12
        assert plan["export"]["gpu_exposed_to_container"] is False
        assert plan["runtime"]["ultralytics_version"] == "8.4.99"
        assert plan["runtime"]["execution_image"] == exporting.EXPORT_IMAGE
        assert plan["license_gate"]["decision"] is None
        assert plan["license_gate"]["download_authorized"] is False
        assert plan["license_gate"]["export_authorized"] is False
        assert all(value is False for value in plan["readiness"].values())
        assert all(value is False for value in plan["acceptance"].values())
        unsigned = dict(plan)
        fingerprint = unsigned.pop("fingerprint_sha256")
        assert exporting.canonical_sha256(unsigned) == fingerprint


def test_runtime_image_is_accepted_only_from_the_exact_person_plan_pin(tmp_path: Path):
    source = json.loads((ROOT / exporting.PERSON_UPGRADE_PLAN).read_text(encoding="utf-8"))
    destination = tmp_path / exporting.PERSON_UPGRADE_PLAN
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(source), encoding="utf-8")

    runtime = exporting.verify_runtime_source(tmp_path)
    assert runtime["execution_image"] == exporting.EXPORT_IMAGE

    source["upstream"]["training_runtime"]["gpu_training_container_digest_linux_amd64"] = "sha256:" + "0" * 64
    destination.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(exporting.PoseExportContractError, match="digest differs"):
        exporting.verify_runtime_source(tmp_path)


def test_implementation_byte_pins_match_live_regular_files():
    plan = exporting.verify_frozen_plan(profile=640)
    for pin in plan["runtime"]["implementation"].values():
        path = ROOT / pin["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_nlink == 1
        assert path.stat().st_size == pin["bytes"]
        assert exporting.sha256_file(path) == pin["sha256"]


def test_checked_in_provenance_plan_pins_both_frozen_plans_and_claims_no_results():
    provenance = json.loads((ROOT / "models/pose/provenance-plan.json").read_text(encoding="utf-8"))
    assert provenance["license"]["decision"] is None
    assert provenance["artifact_state"]["export_executed"] is False
    assert provenance["artifact_state"]["gpu_executed"] is False
    assert all(value is False for value in provenance["readiness"].values())
    assert all(value is False for value in provenance["acceptance"].values())
    for profile in (640, 960):
        pin = provenance["frozen_export_plans"][str(profile)]
        path = ROOT / pin["path"]
        plan = json.loads(path.read_text(encoding="utf-8"))
        assert path.stat().st_size == pin["bytes"]
        assert exporting.sha256_file(path) == pin["sha256"]
        assert plan["fingerprint_sha256"] == pin["fingerprint_sha256"]


def test_default_cli_is_read_only_and_returns_the_frozen_plan():
    artifact_root = ROOT / "models/pose/artifacts/yolo26s-pose"
    before = sorted(path.relative_to(artifact_root) for path in artifact_root.rglob("*")) if artifact_root.exists() else None
    result = subprocess.run(
        ["python3", "-B", "models/pose/export_pose.py", "--profile", "960"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    after = sorted(path.relative_to(artifact_root) for path in artifact_root.rglob("*")) if artifact_root.exists() else None

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["fingerprint_sha256"] == (
        "b623bcdfc1ebaca367026a8fbc153c6d1fb89eda9b3d0c23779f881ed55a5e28"
    )
    assert after == before


def test_python_execute_requires_exact_frozen_fingerprint_before_side_effects():
    artifact_root = ROOT / "models/pose/artifacts/yolo26s-pose/640"
    existed_before = artifact_root.exists() or artifact_root.is_symlink()
    result = subprocess.run(
        [
            "python3",
            "-B",
            "models/pose/export_pose.py",
            "--execute",
            "--profile",
            "640",
            "--accept-ultralytics-license",
            "--license-basis",
            "enterprise",
            "--expected-plan-fingerprint",
            "0" * 64,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "exact frozen pose export plan fingerprint" in result.stderr
    assert (artifact_root.exists() or artifact_root.is_symlink()) is existed_before


def test_shell_wrapper_refuses_before_docker_when_license_or_fingerprint_is_missing():
    no_license = subprocess.run(
        ["bash", "models/pose/run_export.sh", "--execute", "--profile", "640"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_license.returncode == 2
    assert "before an explicit license basis" in no_license.stderr

    no_fingerprint = subprocess.run(
        [
            "bash",
            "models/pose/run_export.sh",
            "--execute",
            "--profile",
            "640",
            "--accept-ultralytics-license",
            "--license-basis",
            "enterprise",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_fingerprint.returncode == 2
    assert "exact frozen plan fingerprint" in no_fingerprint.stderr


def test_shell_wrapper_rejects_wrong_exact_fingerprint_before_docker(tmp_path: Path):
    marker = tmp_path / "docker-called"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f"#!/usr/bin/env bash\ntouch {marker}\nexit 99\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "models/pose/run_export.sh",
            "--execute",
            "--profile",
            "640",
            "--accept-ultralytics-license",
            "--license-basis",
            "enterprise",
            "--expected-plan-fingerprint",
            "0" * 64,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "exact frozen pose export plan fingerprint" in result.stderr
    assert not marker.exists()


def test_read_only_exact_preflight_succeeds_without_export(capsys):
    plan = exporting.verify_frozen_plan(profile=640)
    result = exporting.main(
        [
            "--execute",
            "--preflight-only",
            "--profile",
            "640",
            "--accept-ultralytics-license",
            "--license-basis",
            "enterprise",
            "--expected-plan-fingerprint",
            plan["fingerprint_sha256"],
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == plan


def test_export_provenance_can_only_claim_exported_not_evaluated():
    plan = exporting.verify_frozen_plan(profile=640)
    receipt = exporting.build_export_provenance(
        plan=plan,
        license_basis="enterprise",
        onnx_pin={"path": plan["artifacts"]["onnx"], "bytes": 123, "sha256": "1" * 64},
        contract_pin={"path": plan["artifacts"]["onnx_contract"], "bytes": 456, "sha256": "2" * 64},
        packages={"ultralytics": "8.4.99"},
    )

    assert receipt["status"] == "exported_not_evaluated"
    assert receipt["artifacts"]["tensorrt_engine"]["status"] == "not_built"
    assert receipt["artifacts"]["tensorrt_engine"]["present"] is False
    assert all(value is False for value in receipt["readiness"].values())
    assert all(value is False for value in receipt["acceptance"].values())
    unsigned = dict(receipt)
    fingerprint = unsigned.pop("fingerprint_sha256")
    assert exporting.canonical_sha256(unsigned) == fingerprint


def test_fd_bound_publish_never_overwrites(tmp_path: Path):
    destination = tmp_path / "receipt.json"
    try:
        exporting._write_bytes_no_replace(destination, b"first\n")
    except exporting.PoseExportContractError as exc:
        if "O_TMPFILE" in str(exc) or "anonymous artifact" in str(exc):
            pytest.skip(str(exc))
        raise

    with pytest.raises(exporting.PoseExportContractError, match="overwrite"):
        exporting._write_bytes_no_replace(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_profile_directory_publish_is_atomic_and_never_overwrites(tmp_path: Path):
    first = tmp_path / "first"
    first.mkdir()
    (first / "provenance.json").write_text("first\n", encoding="utf-8")
    destination = tmp_path / "640"

    exporting._rename_directory_no_replace(first, destination)
    assert not first.exists()
    assert (destination / "provenance.json").read_text(encoding="utf-8") == "first\n"

    second = tmp_path / "second"
    second.mkdir()
    (second / "provenance.json").write_text("second\n", encoding="utf-8")
    with pytest.raises(exporting.PoseExportContractError, match="overwrite"):
        exporting._rename_directory_no_replace(second, destination)
    assert (destination / "provenance.json").read_text(encoding="utf-8") == "first\n"
    assert (second / "provenance.json").read_text(encoding="utf-8") == "second\n"


def test_repo_fd_reader_rejects_an_intermediate_directory_symlink(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "contract.json").write_text('{"trusted":false}\n', encoding="utf-8")
    (root / "contracts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(exporting.PoseExportContractError, match="symlink component"):
        exporting._read_repo_file(root, "contracts/contract.json", capture=True)


def test_repo_fd_reader_remains_bound_to_original_parent_during_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    parent = root / "contracts"
    evil = root / "evil"
    parent.mkdir(parents=True)
    evil.mkdir()
    trusted = b'{"trusted":true}\n'
    (parent / "contract.json").write_bytes(trusted)
    (evil / "contract.json").write_bytes(b'{"trusted":false}\n')
    expected_sha256 = hashlib.sha256(trusted).hexdigest()
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "contract.json" and dir_fd is not None and not swapped:
            parent.rename(root / "contracts-original")
            parent.symlink_to(evil, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(exporting.os, "open", racing_open)
    evidence = exporting._read_repo_file(
        root,
        "contracts/contract.json",
        expected_bytes=len(trusted),
        expected_sha256=expected_sha256,
        capture=True,
    )

    assert swapped is True
    assert evidence is not None
    assert evidence.data == trusted
    assert evidence.sha256 == expected_sha256


def test_repo_fd_reader_rejects_leaf_swap_between_parent_open_and_file_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    parent = root / "contracts"
    parent.mkdir(parents=True)
    target = parent / "contract.json"
    replacement = parent / "replacement.json"
    trusted = b"AAAA"
    target.write_bytes(trusted)
    replacement.write_bytes(b"BBBB")
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "contract.json" and dir_fd is not None and not swapped:
            os.replace(replacement, target)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(exporting.os, "open", racing_open)
    with pytest.raises(exporting.PoseExportContractError, match="SHA256 differs"):
        exporting._read_repo_file(
            root,
            "contracts/contract.json",
            expected_bytes=len(trusted),
            expected_sha256=hashlib.sha256(trusted).hexdigest(),
        )
    assert swapped is True


def test_repo_fd_reader_rejects_in_place_change_during_same_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    parent = root / "contracts"
    parent.mkdir(parents=True)
    target = parent / "large.bin"
    trusted = b"A" * (exporting.READ_CHUNK + 128)
    target.write_bytes(trusted)
    real_read = os.read
    changed = False

    def racing_read(fd, count):
        nonlocal changed
        chunk = real_read(fd, count)
        if chunk and not changed:
            with target.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"Z")
                stream.flush()
                os.fsync(stream.fileno())
            changed = True
        return chunk

    monkeypatch.setattr(exporting.os, "read", racing_read)
    with pytest.raises(exporting.PoseExportContractError, match="identity or metadata changed"):
        exporting._read_repo_file(
            root,
            "contracts/large.bin",
            expected_bytes=len(trusted),
            expected_sha256=hashlib.sha256(trusted).hexdigest(),
        )
    assert changed is True


def test_existing_checkpoint_is_verified_by_bound_repo_fd_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    relative = "models/pose/candidates/test/checkpoint.pt"
    destination = root / relative
    destination.parent.mkdir(parents=True)
    payload = b"pinned-checkpoint"
    destination.write_bytes(payload)
    model_spec = dict(exporting.MODEL_SPEC)
    model_spec.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(exporting, "MODEL_SPEC", model_spec)

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("network must not be reached for an existing valid checkpoint")

    monkeypatch.setattr(exporting.urllib.request, "urlopen", network_forbidden)
    exporting._download_verified(
        destination,
        project_root=root,
        repository_path=relative,
    )
    assert destination.read_bytes() == payload


def test_strict_json_rejects_symlink_and_duplicate_keys(tmp_path: Path):
    regular = tmp_path / "regular.json"
    regular.write_text('{"a":1}\n', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(regular)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")

    with pytest.raises(exporting.PoseExportContractError, match="symlink component"):
        exporting._strict_json(link)
    with pytest.raises(exporting.PoseExportContractError, match="duplicate JSON key"):
        exporting._strict_json(duplicate)
