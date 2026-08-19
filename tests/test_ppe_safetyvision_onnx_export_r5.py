from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from validation import ppe_safetyvision_onnx_export_r5 as r5


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/ppe/export-lanes/safetyvision-yolov8s-v2-cpu-export-r5"


def load_lane_module(filename: str, name: str) -> ModuleType:
    lane_text = str(LANE)
    if lane_text not in sys.path:
        sys.path.insert(0, lane_text)
    spec = importlib.util.spec_from_file_location(name, LANE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pin(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_static_source_audit_is_closed_and_makes_no_execution_claim() -> None:
    report = r5.static_audit()
    assert report["status"] == "static_sources_closed_runtime_image_not_built_phase_a2"
    assert report["phase_a_revision"] == 2
    assert report["predecessor_failure"] == {
        "bytes": 4443,
        "path": "validation/results/ppe/models/safetyvision-yolov8s-v2-cpu-image-context-r5/phase-a1-001/receipt.json",
        "sha256": "e2c35b6e19df6c02793dc0e3121855e2125a83f95154e562ef6ee427e22a4976",
        "status": "phase_a1_failed_no_context_published_no_workload_execution",
    }
    assert report["predecessor_failure_external_receipt_replayed"] is True
    assert report["runtime_distributions"] == 87
    assert report["host_usr_lib_venv_bind_mounts"] is False
    assert report["checkpoint_deserialized"] is False
    assert report["model_export_or_inference_executed"] is False
    assert report["image_built"] is False
    assert report["gpu_used"] is False


def test_source_contract_maps_every_independent_review_finding() -> None:
    contract = json.loads(r5.SOURCE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "static_sources_closed_runtime_image_not_built_phase_a2"
    assert contract["phase_a_revision"] == 2
    mapping = contract["remediation_mapping"]
    assert set(mapping) == {
        "P1_snapshot_toctou",
        "P1_runtime_shadow_mounts",
        "P1_terminal_effective_state",
        "P1_parity_evidence",
        "P2_atomic_publication",
        "P2_official_test_command",
    }
    assert contract["runtime"]["distribution_count"] == 87
    assert contract["runtime"]["host_runtime_bind_mounts"] == []
    assert contract["authorization"]["dedicated_cpu_image_build"] is False
    assert contract["authorization"]["checkpoint_deserialization_or_export"] is False
    assert contract["authorization"]["cpu_onnxruntime_parity"] is False
    assert contract["authorization"]["gpu"] is False
    assert any(row["logical_name"] == "official_test_command" for row in contract["sources"])
    predecessor = contract["predecessor_failure"]
    predecessor_source = next(
        row for row in contract["sources"] if row["logical_name"] == "phase_a1_context_failure"
    )
    assert predecessor_source == {
        "logical_name": "phase_a1_context_failure",
        "path": predecessor["path"],
        "bytes": predecessor["bytes"],
        "sha256": predecessor["sha256"],
    }
    predecessor_path = ROOT / predecessor["path"]
    assert predecessor_path.stat().st_mode & 0o222 == 0
    assert pin(predecessor_path) == (predecessor["bytes"], predecessor["sha256"])
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    assert predecessor_receipt["status"] == predecessor["status"]
    assert predecessor_receipt["cleanup"]["context_target_absent"] is True
    assert predecessor_receipt["execution"]["model_export_or_inference_executed"] is False


def test_official_one_command_environment_is_cpu_only_and_exact() -> None:
    command = ROOT / "scripts/test-ppe-safetyvision-r5"
    source = command.read_text(encoding="utf-8")
    assert command.stat().st_mode & 0o222 == 0
    assert "set -euo pipefail" in source
    assert "PYTHONHASHSEED=0" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "NVIDIA_VISIBLE_DEVICES=void" in source
    assert '"${ROOT}/.venv/bin/python" -m pytest -q' in source
    assert 'tests/test_ppe_safetyvision_onnx_export_r5.py' in source


def test_export_workload_manifest_breaks_plan_image_self_reference() -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_predecessor_replay_test")
    contract = json.loads(r5.SOURCE_CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads((LANE / "export-workload-manifest-r5.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "deepsafe.ppe-safetyvision-cpu-workload-manifest-r5/v1"
    assert manifest["phase"] == "export"
    assert manifest["status"] == "frozen_export_workload_image_not_built_phase_a2"
    assert manifest["phase_a_revision"] == 2
    assert "image" not in manifest
    assert manifest["snapshot"]["storage"] == "embedded_immutable_image_rootfs"
    assert manifest["snapshot"]["host_bind_mounts"] is False
    assert manifest["runtime_source_closure"]["distribution_count"] == 87
    failure_lineage = manifest["lineage"]["phase_a1_context_failure"]
    failure_snapshot = [
        row for row in manifest["snapshot"]["files"] if row["logical_name"] == "phase_a1_context_failure"
    ]
    assert failure_snapshot == [failure_lineage]
    assert failure_lineage["path"] == "/opt/deepsafe/inputs/phase-a1-context-failure.json"
    assert (failure_lineage["bytes"], failure_lineage["sha256"]) == (
        4443,
        "e2c35b6e19df6c02793dc0e3121855e2125a83f95154e562ef6ee427e22a4976",
    )
    sources = {row["logical_name"]: row for row in contract["sources"]}
    replay = builder._validate_phase_a2_predecessor(contract, manifest, sources)
    assert replay["external_receipt"]["read_only"] is True
    assert replay["external_receipt"]["replayed"] is True
    assert replay["embedded_row_is_authority"] is False
    unsigned = dict(manifest)
    fingerprint = unsigned.pop("self_fingerprint")
    assert hashlib.sha256(r5.canonical_bytes(unsigned)).hexdigest() == fingerprint


def test_dockerfile_has_exact_offline_copy_only_runtime() -> None:
    source = (LANE / "Dockerfile.r5").read_text(encoding="utf-8")
    assert not source.lstrip().startswith("# syntax=")
    assert "FROM ubuntu@sha256:4fbb8e6a" in source
    assert "COPY --link runtime/rootfs/ /" in source
    assert "COPY --link inputs/ /opt/deepsafe/inputs/" in source
    lowered = source.lower()
    for forbidden in ("apt-get", "apt ", "pip install", "curl ", "wget ", "git clone"):
        assert forbidden not in lowered
    assert "nvidia" in lowered  # NVIDIA_VISIBLE_DEVICES is explicitly set to void
    assert "nvidia_visible_devices=void" in lowered


def test_workers_have_no_process_network_docker_or_gpu_control_surface() -> None:
    for filename in ("export_worker.py", "parity_worker.py"):
        source = (LANE / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "socket" not in imported
        assert "subprocess" not in imported
        assert "docker" not in imported
        for forbidden in ("nvidia-smi", "trtexec", "deepstream-app", "docker run", "docker create"):
            assert forbidden not in source.lower()


def test_deterministic_parity_inputs_have_exact_hashes() -> None:
    worker = load_lane_module("parity_worker.py", "ppe_r5_parity_worker_test")
    expected = {
        (640, 1): "438fbf5ffedaa5105b3b46111281c96d4be82c9d5baef9855ad8e9362e39228c",
        (640, 2): "0570b43c3f00547ac52bad74cc89e7d97836b30f78963f832350d6c7bbff77fd",
        (960, 1): "8c13aa4b9cb7dafa47b39403dde2ba11dcf33e640e7082024013da8f1375f811",
    }
    for shape, sha256 in expected.items():
        report = worker.raw_array_report(worker.deterministic_tensor(*shape))
        assert report["sha256"] == sha256
        assert report["finite"] is True


def synthetic_plan(output_manifest: Path | None = None) -> dict:
    manifest = output_manifest or (LANE / "export-workload-manifest-r5.json")
    manifest_bytes, manifest_sha = pin(manifest)
    worker_bytes, worker_sha = pin(LANE / "export_worker.py")
    runner_bytes, runner_sha = pin(Path(r5.__file__))
    environment = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HOME": "/tmp/home",
        "MKL_NUM_THREADS": "8",
        "MPLCONFIGDIR": "/tmp/mplconfig",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OMP_NUM_THREADS": "8",
        "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TORCH_HOME": "/tmp/torch",
        "ULTRALYTICS_OFFLINE": "true",
        "XDG_CACHE_HOME": "/tmp/cache",
        "YOLO_CONFIG_DIR": "/tmp/ultralytics-config",
    }
    return {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-execution-authority-r5/v1",
        "status": "authorized_exact_immutable_image",
        "phase": "export",
        "image": {
            "id": "sha256:" + "a" * 64,
            "bytes": 2_500_000_000,
            "platform": {"os": "linux", "architecture": "amd64"},
            "rootfs_diff_ids": ["sha256:" + "b" * 64],
            "oci_closure": {
                "verified": True,
                "platform": {"os": "linux", "architecture": "amd64"},
                "manifest": {
                    "digest": "sha256:" + "c" * 64,
                    "config_digest": "sha256:" + "d" * 64,
                },
                "config": {
                    "digest": "sha256:" + "d" * 64,
                    "os": "linux",
                    "architecture": "amd64",
                },
                "descriptors": [
                    {
                        "digest": "sha256:" + "c" * 64,
                        "size": 123,
                        "media_type": "application/vnd.oci.image.manifest.v1+json",
                    },
                    {
                        "digest": "sha256:" + "d" * 64,
                        "size": 456,
                        "media_type": "application/vnd.oci.image.config.v1+json",
                    },
                ],
            },
        },
        "programs": {
            "host_runner": {"bytes": runner_bytes, "sha256": runner_sha},
            "export_worker": {
                "embedded_path": "/opt/deepsafe/lane/export_worker.py",
                "bytes": worker_bytes,
                "sha256": worker_sha,
            },
        },
        "workload_manifest": {
            "host_path": manifest.relative_to(ROOT).as_posix(),
            "embedded_path": "/opt/deepsafe/authority/workload-manifest.json",
            "bytes": manifest_bytes,
            "sha256": manifest_sha,
        },
        "snapshot": {"storage": "embedded_immutable_image_rootfs", "files": []},
        "environment": environment,
        "container": {
            "network_mode": "none",
            "read_only_root": True,
            "user": "1000:1000",
            "cap_drop": ["ALL"],
            "cap_add": [],
            "security_opt": ["no-new-privileges:true"],
            "ipc_mode": "private",
            "mounts": [{"destination": "/output", "read_only": False, "type": "bind"}],
            "gpu_device_request": False,
            "nvidia_runtime": False,
            "pids_limit": 256,
            "cpus": 8,
            "memory": "16g",
            "memory_bytes": 16 * 1024**3,
            "memory_swap": "16g",
            "memory_swap_bytes": 16 * 1024**3,
            "shm_size": "1g",
            "shm_size_bytes": 1024**3,
        },
    }


def finalize_plan(plan: dict) -> dict:
    value = dict(plan)
    value["self_fingerprint"] = hashlib.sha256(r5.canonical_bytes(value)).hexdigest()
    return value


def test_authority_plan_pins_runner_manifest_and_exact_oci(tmp_path: Path) -> None:
    plan = finalize_plan(synthetic_plan())
    path = tmp_path / "authority.json"
    raw = r5.canonical_bytes(plan) + b"\n"
    path.write_bytes(raw)
    path.chmod(0o440)
    loaded, loaded_raw = r5.load_plan(path, (len(raw), hashlib.sha256(raw).hexdigest()))
    assert loaded_raw == raw
    assert loaded["image"]["oci_closure"]["verified"] is True
    assert loaded["image"]["platform"] == {"os": "linux", "architecture": "amd64"}
    assert loaded["image"]["oci_closure"]["manifest"]["config_digest"] == "sha256:" + "d" * 64
    assert loaded["workload_manifest"]["sha256"] == pin(LANE / "export-workload-manifest-r5.json")[1]


def test_loaded_image_inspect_closes_linux_amd64_platform_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = finalize_plan(synthetic_plan())

    def inspect_result(architecture: str) -> subprocess.CompletedProcess[str]:
        value = [
            {
                "Id": plan["image"]["id"],
                "Os": "linux",
                "Architecture": architecture,
                "Size": plan["image"]["bytes"],
                "RootFS": {"Layers": plan["image"]["rootfs_diff_ids"]},
            }
        ]
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(value), stderr="")

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: inspect_result("amd64"))
    report = r5.inspect_image(plan)
    assert {"os": report["os"], "architecture": report["architecture"]} == plan["image"]["platform"]

    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: inspect_result("arm64"))
    with pytest.raises(r5.SafetyVisionR5Error, match="loaded image platform"):
        r5.inspect_image(plan)


def test_authority_plan_rejects_missing_platform_and_wrong_config_chain(tmp_path: Path) -> None:
    cases: list[tuple[str, dict, str]] = []
    missing = synthetic_plan()
    missing["image"]["oci_closure"].pop("platform")
    cases.append(("missing", missing, "OCI closure platform"))
    wrong = synthetic_plan()
    wrong["image"]["oci_closure"]["config"]["architecture"] = "arm64"
    cases.append(("wrong-config", wrong, "OCI authority config"))
    for name, plan, message in cases:
        finalized = finalize_plan(plan)
        raw = r5.canonical_bytes(finalized) + b"\n"
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        path.chmod(0o440)
        with pytest.raises(r5.SafetyVisionR5Error, match=message):
            r5.load_plan(path, (len(raw), hashlib.sha256(raw).hexdigest()))


def test_create_argv_has_only_output_mount_and_passes_embedded_manifest(tmp_path: Path) -> None:
    plan = finalize_plan(synthetic_plan())
    output = tmp_path.resolve() / "output"
    output.mkdir()
    argv = r5.build_create_argv(plan, (1234, "d" * 64), "unit-export-001", output, preflight_only=False)
    assert argv[:4] == ["docker", "create", "--pull", "never"]
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert "--read-only" in argv
    assert "--gpus" not in argv and "--device" not in argv
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert len(mounts) == 1
    assert f"src={output}" in mounts[0] and "dst=/output" in mounts[0]
    assert not any("dst=/usr" in value or "dst=/lib" in value or "dst=/opt/venv" in value for value in mounts)
    assert argv[argv.index("--manifest") + 1] == "/opt/deepsafe/authority/workload-manifest.json"
    assert argv[argv.index("--manifest-sha256") + 1] == plan["workload_manifest"]["sha256"]
    assert argv[argv.index("--label") + 1] == "deepsafe.scope=ppe-safetyvision-cpu-r5"


def fake_inspect(plan: dict, plan_pin: tuple[int, str], output: Path, argv: list[str]) -> dict:
    image_id = plan["image"]["id"]
    command = argv[argv.index(image_id) + 1 :]
    labels = {
        "deepsafe.scope": "ppe-safetyvision-cpu-r5",
        "deepsafe.phase": "export",
        "deepsafe.plan.sha256": plan_pin[1],
        "deepsafe.gpu": "false",
    }
    return {
        "Image": image_id,
        "Config": {
            "Image": image_id,
            "User": "1000:1000",
            "Entrypoint": ["/usr/bin/python3.12"],
            "Cmd": command,
            "Env": [f"{key}={value}" for key, value in plan["environment"].items()],
            "Labels": labels,
            "Volumes": None,
        },
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Privileged": False,
            "DeviceRequests": None,
            "Devices": None,
            "Runtime": "runc",
            "CapDrop": ["ALL"],
            "CapAdd": None,
            "SecurityOpt": ["no-new-privileges:true"],
            "PidsLimit": 256,
            "NanoCpus": 8_000_000_000,
            "Memory": 16 * 1024**3,
            "MemorySwap": 16 * 1024**3,
            "IpcMode": "private",
            "ShmSize": 1024**3,
            "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=4g,mode=1777"},
            "Ulimits": [{"Name": "nofile", "Hard": 4096, "Soft": 4096}],
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(output),
                "Destination": "/output",
                "RW": True,
                "Propagation": "rprivate",
            }
        ],
    }


def test_full_effective_inspect_records_argv_env_security_ipc_tmpfs_and_mount(tmp_path: Path) -> None:
    plan = finalize_plan(synthetic_plan())
    plan_pin = (1234, "d" * 64)
    output = tmp_path.resolve() / "output"
    output.mkdir()
    argv = r5.build_create_argv(plan, plan_pin, "unit-inspect-001", output, preflight_only=True)
    inspected = fake_inspect(plan, plan_pin, output, argv)
    report = r5.validate_container_inspect(inspected, plan, plan_pin, output, argv, preflight_only=True)
    assert report["requested_docker_create_argv"] == argv
    assert report["effective_process_argv"] == ["/usr/bin/python3.12", *inspected["Config"]["Cmd"]]
    assert report["environment"] == plan["environment"]
    assert report["user"] == "1000:1000"
    assert report["cap_drop"] == ["ALL"]
    assert report["security_opt"] == ["no-new-privileges:true"]
    assert report["ipc_mode"] == "private"
    assert report["tmpfs"] == {"/tmp": "rw,nosuid,nodev,noexec,size=4g,mode=1777"}
    assert report["mounts"] == [
        {
            "type": "bind",
            "source": str(output),
            "destination": "/output",
            "rw": True,
            "propagation": "rprivate",
        }
    ]


def test_inspect_rejects_extra_mount_or_environment_change(tmp_path: Path) -> None:
    plan = finalize_plan(synthetic_plan())
    plan_pin = (1234, "d" * 64)
    output = tmp_path.resolve() / "output"
    output.mkdir()
    argv = r5.build_create_argv(plan, plan_pin, "unit-reject-001", output, preflight_only=False)
    inspected = fake_inspect(plan, plan_pin, output, argv)
    inspected["Config"]["Env"].append("UNPLANNED=1")
    with pytest.raises(r5.SafetyVisionR5Error, match="environment"):
        r5.validate_container_inspect(inspected, plan, plan_pin, output, argv, preflight_only=False)
    inspected = fake_inspect(plan, plan_pin, output, argv)
    inspected["Mounts"].append(
        {"Type": "bind", "Source": "/usr", "Destination": "/usr", "RW": False, "Propagation": "rprivate"}
    )
    with pytest.raises(r5.SafetyVisionR5Error, match="mount count"):
        r5.validate_container_inspect(inspected, plan, plan_pin, output, argv, preflight_only=False)


def test_directory_atomic_rename_is_no_replace_and_preserves_bytes(tmp_path: Path) -> None:
    stage = tmp_path / ".r5-stage-unit"
    stage.mkdir()
    payload = b"exact\n"
    (stage / "receipt.json").write_bytes(payload)
    r5.fsync_tree(stage)
    r5.freeze_tree(stage)
    r5.fsync_tree(stage)
    r5.rename_noreplace(tmp_path, stage.name, "run-001")
    assert (tmp_path / "run-001/receipt.json").read_bytes() == payload
    collision = tmp_path / ".r5-stage-collision"
    collision.mkdir()
    with pytest.raises(r5.SafetyVisionR5Error, match="already exists"):
        r5.rename_noreplace(tmp_path, collision.name, "run-001")


def test_recovery_is_report_only_or_non_destructive_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r5, "RESULTS", tmp_path)
    stage = tmp_path / ".r5-stage-unit-recovery"
    stage.mkdir()
    report = r5.recover_staging(mode="report")
    assert report["deleted"] == 0
    assert stage.exists()
    quarantined = r5.recover_staging(mode="quarantine")
    assert quarantined["deleted"] == 0
    assert not stage.exists()
    assert any(path.name.startswith(".r5-quarantine-") for path in tmp_path.iterdir())


def test_context_tar_stream_is_deterministic_and_fd_stable(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_test")
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile.r5").write_text("FROM scratch\n", encoding="utf-8")
    nested = context / "inputs"
    nested.mkdir()
    (nested / "x.bin").write_bytes(b"abc")
    real_directory = nested / "real"
    real_directory.mkdir()
    (real_directory / "inside.bin").write_bytes(b"inside")
    (nested / "link-dir").symlink_to("real", target_is_directory=True)
    (nested / "link-file").symlink_to("x.bin")
    first = io.BytesIO()
    second = io.BytesIO()
    report1 = builder.stream_context(context, first)
    report2 = builder.stream_context(context, second)
    assert first.getvalue() == second.getvalue()
    assert report1 == report2
    with tarfile.open(fileobj=io.BytesIO(first.getvalue()), mode="r:") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == [
            "Dockerfile.r5",
            "inputs",
            "inputs/link-dir",
            "inputs/link-file",
            "inputs/real",
            "inputs/x.bin",
            "inputs/real/inside.bin",
        ]
        links = {member.name: member for member in members if member.issym()}
        assert links["inputs/link-dir"].linkname == "real"
        assert links["inputs/link-file"].linkname == "x.bin"
        assert links["inputs/link-dir"].mode == 0o777
        assert links["inputs/link-file"].mode == 0o777
        assert not any(member.name.startswith("inputs/link-dir/") for member in members)


def test_context_freeze_never_follows_outside_file_or_directory_symlink(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_freeze_test")
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    target = outside / "sentinel.bin"
    target.write_bytes(b"sentinel")
    target.chmod(0o600)
    context = tmp_path / "context"
    context.mkdir()
    (context / "outside-file").symlink_to(target)
    (context / "outside-directory").symlink_to(outside, target_is_directory=True)
    before = (target.stat().st_mode & 0o777, outside.stat().st_mode & 0o777)
    builder.freeze_context_tree(context)
    after = (target.stat().st_mode & 0o777, outside.stat().st_mode & 0o777)
    assert after == before == (0o600, 0o700)
    assert (context / "outside-file").is_symlink()
    assert (context / "outside-directory").is_symlink()


def test_runtime_private_copy_handles_nested_tree_and_symlinks_without_traverse(tmp_path: Path) -> None:
    runtime = load_lane_module("runtime_closure.py", "ppe_r5_runtime_private_copy_test")
    source = tmp_path / "source"
    nested = source / "a/b"
    nested.mkdir(parents=True)
    ordinary = nested / "ordinary.bin"
    ordinary.write_bytes(b"ordinary")
    executable = nested / "tool"
    executable.write_bytes(b"tool")
    executable.chmod(0o755)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"outside")
    sentinel.chmod(0o640)
    (source / "file-link").symlink_to(sentinel)
    (source / "directory-link").symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "destination"
    executable_paths = runtime._copy_tree(source, destination)
    assert executable_paths == {"a/b/tool"}
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "a/b").stat().st_mode & 0o777 == 0o700
    assert (destination / "a/b/ordinary.bin").stat().st_mode & 0o777 == 0o600
    assert (destination / "a/b/tool").stat().st_mode & 0o777 == 0o600
    assert (destination / "file-link").is_symlink()
    assert (destination / "directory-link").is_symlink()
    assert sentinel.stat().st_mode & 0o777 == 0o640


def test_runtime_mid_copy_exception_and_partial_freeze_cleanup_are_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_lane_module("runtime_closure.py", "ppe_r5_runtime_cleanup_test")
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.bin").write_bytes(b"a")
    (source / "b.bin").write_bytes(b"b")
    original_copy = runtime.copy_regular_exact
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected mid-copy failure")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(runtime, "copy_regular_exact", fail_second)
    destination = tmp_path / "copy-stage"
    with pytest.raises(OSError, match="mid-copy"):
        runtime._copy_tree(source, destination)
    runtime.cleanup_tree(destination)
    assert not destination.exists()

    freeze_stage = tmp_path / "freeze-stage"
    freeze_stage.mkdir(mode=0o700)
    (freeze_stage / "a.bin").write_bytes(b"a")
    (freeze_stage / "z-fail.bin").write_bytes(b"z")
    for path in freeze_stage.iterdir():
        path.chmod(0o600)
    original_chmod = runtime.os.chmod
    failed = False

    def fail_once(path, mode, *, follow_symlinks=True):
        nonlocal failed
        if Path(path).name == "z-fail.bin" and not failed:
            failed = True
            raise OSError("injected freeze failure")
        return original_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(runtime.os, "chmod", fail_once)
    with pytest.raises(OSError, match="freeze failure"):
        runtime.freeze_tree(freeze_stage)
    monkeypatch.setattr(runtime.os, "chmod", original_chmod)
    runtime.cleanup_tree(freeze_stage)
    assert not freeze_stage.exists()


def test_runtime_atomic_freeze_publish_is_read_only_and_no_replace(tmp_path: Path) -> None:
    runtime = load_lane_module("runtime_closure.py", "ppe_r5_runtime_atomic_publish_test")
    stage = tmp_path / ".stage"
    nested = stage / "nested"
    nested.mkdir(parents=True, mode=0o700)
    ordinary = nested / "ordinary.bin"
    metadata = nested / "receipt.json"
    executable = nested / "tool"
    ordinary.write_bytes(b"ordinary")
    metadata.write_bytes(b"{}\n")
    executable.write_bytes(b"tool")
    for path in (ordinary, metadata, executable):
        path.chmod(0o600)
    (nested / "link").symlink_to("ordinary.bin")
    executable_paths = {"nested/tool"}
    metadata_paths = {"nested/receipt.json"}
    before = runtime.tree_attestation(
        stage,
        normalized_snapshot_modes=True,
        policy_executable_paths=executable_paths,
        policy_metadata_paths=metadata_paths,
    )
    runtime.fsync_tree(stage)
    runtime.freeze_tree(
        stage,
        executable_paths=executable_paths,
        metadata_paths=metadata_paths,
        root_mode=0o550,
    )
    runtime.fsync_tree(stage)
    after = runtime.tree_attestation(stage)
    assert runtime._same_attestation(before, after)
    runtime.rename_noreplace(tmp_path, stage.name, "published")
    published = tmp_path / "published"
    assert published.stat().st_mode & 0o777 == 0o550
    assert (published / "nested").stat().st_mode & 0o777 == 0o555
    assert (published / "nested/ordinary.bin").stat().st_mode & 0o777 == 0o444
    assert (published / "nested/receipt.json").stat().st_mode & 0o777 == 0o440
    assert (published / "nested/tool").stat().st_mode & 0o777 == 0o555
    assert (published / "nested/link").is_symlink()

    collision = tmp_path / ".collision"
    collision.mkdir(mode=0o700)
    with pytest.raises(runtime.RuntimeClosureR5Error, match="already exists"):
        runtime.rename_noreplace(tmp_path, collision.name, "published")
    assert collision.exists()
    runtime.cleanup_tree(collision)
    runtime.cleanup_tree(published)


def test_runtime_snapshot_workflow_publishes_atomically_and_cleans_injected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_lane_module("runtime_closure.py", "ppe_r5_runtime_snapshot_workflow_test")
    venv = tmp_path / "venv"
    stdlib = tmp_path / "stdlib"
    (venv / "bin").mkdir(parents=True)
    (stdlib / "pkg").mkdir(parents=True)
    tool = venv / "bin/tool"
    tool.write_bytes(b"tool")
    tool.chmod(0o755)
    (venv / "bin/tool-link").symlink_to("tool")
    (stdlib / "pkg/module.py").write_bytes(b"value = 1\n")
    python = tmp_path / "python3.12"
    python.write_bytes(b"python")
    python.chmod(0o755)
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_bytes(b"# site\n")
    monkeypatch.setattr(runtime, "VENV", venv)
    monkeypatch.setattr(runtime, "STDLIB", stdlib)
    monkeypatch.setattr(runtime, "PYTHON", python)
    monkeypatch.setattr(runtime, "SITECUSTOMIZE", sitecustomize)
    live = {
        "python": {"bytes": len(python.read_bytes()), "sha256": hashlib.sha256(python.read_bytes()).hexdigest()},
        "sitecustomize": {
            "bytes": len(sitecustomize.read_bytes()),
            "sha256": hashlib.sha256(sitecustomize.read_bytes()).hexdigest(),
        },
        "native_library_closure": {"external_libraries": [], "external_aliases": []},
    }
    monkeypatch.setattr(runtime, "audit", lambda: copy.deepcopy(live))
    expected = tmp_path / "expected.json"
    expected.write_bytes(runtime.canonical_bytes(live) + b"\n")

    output = tmp_path / "snapshot"
    report = runtime.prepare_snapshot(output, expected)
    assert report["snapshot_receipt"]["bytes"] > 0
    assert output.stat().st_mode & 0o777 == 0o550
    assert (output / "rootfs").stat().st_mode & 0o777 == 0o555
    assert (output / "rootfs/opt/venv/bin/tool").stat().st_mode & 0o777 == 0o555
    assert (output / "rootfs/usr/lib/python3.12/pkg/module.py").stat().st_mode & 0o777 == 0o444
    assert (output / "snapshot-receipt.json").stat().st_mode & 0o777 == 0o440
    assert (output / "rootfs/opt/venv/bin/tool-link").is_symlink()
    assert not list(tmp_path.glob(".snapshot.stage-*"))

    with pytest.raises(runtime.RuntimeClosureR5Error, match="already exists"):
        runtime.prepare_snapshot(output, expected)

    original_copy = runtime.copy_regular_exact
    calls = 0

    def fail_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected prepare copy failure")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(runtime, "copy_regular_exact", fail_copy)
    failed_copy = tmp_path / "failed-copy"
    with pytest.raises(OSError, match="prepare copy"):
        runtime.prepare_snapshot(failed_copy, expected)
    assert not failed_copy.exists()
    assert not list(tmp_path.glob(".failed-copy.stage-*"))
    monkeypatch.setattr(runtime, "copy_regular_exact", original_copy)

    original_freeze = runtime.freeze_tree
    froze = False

    def fail_freeze(*args, **kwargs):
        nonlocal froze
        if not froze:
            froze = True
            raise OSError("injected prepare freeze failure")
        return original_freeze(*args, **kwargs)

    monkeypatch.setattr(runtime, "freeze_tree", fail_freeze)
    failed_freeze = tmp_path / "failed-freeze"
    with pytest.raises(OSError, match="prepare freeze"):
        runtime.prepare_snapshot(failed_freeze, expected)
    assert not failed_freeze.exists()
    assert not list(tmp_path.glob(".failed-freeze.stage-*"))
    monkeypatch.setattr(runtime, "freeze_tree", original_freeze)
    runtime.cleanup_tree(output)


def test_context_tar_rejects_member_escape_absolute_and_parent_link_escape(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_escape_test")
    with pytest.raises(builder.ImageBuilderR5Error, match="unsafe tar member"):
        builder._safe_tar_member_name("../escape")
    with pytest.raises(builder.ImageBuilderR5Error, match="unsafe tar member"):
        builder._safe_tar_member_name("/absolute")

    parent_escape = tmp_path / "parent-escape"
    inputs = parent_escape / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "escape").symlink_to("../../outside")
    with pytest.raises(builder.ImageBuilderR5Error, match="escapes context namespace"):
        builder.stream_context(parent_escape, io.BytesIO())

    absolute_escape = tmp_path / "absolute-escape"
    inputs = absolute_escape / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "escape").symlink_to("/etc/passwd")
    with pytest.raises(builder.ImageBuilderR5Error, match="absolute symlink target outside rootfs"):
        builder.stream_context(absolute_escape, io.BytesIO())


def test_rootfs_absolute_and_parent_links_stay_inside_virtual_image_root(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_rootfs_link_test")
    context = tmp_path / "context"
    binary = context / "runtime/rootfs/usr/bin/python3"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"python")
    venv_bin = context / "runtime/rootfs/opt/venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to("/usr/bin/python3")
    lib = context / "runtime/rootfs/usr/lib"
    lib.mkdir(parents=True)
    (lib / "libx.so").write_bytes(b"lib")
    lib64 = context / "runtime/rootfs/usr/lib64"
    lib64.mkdir()
    (lib64 / "libx.so").symlink_to("../lib/libx.so")
    output = io.BytesIO()
    builder.stream_context(context, output)
    with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode="r:") as archive:
        assert archive.getmember("runtime/rootfs/opt/venv/bin/python").linkname == "/usr/bin/python3"
        assert archive.getmember("runtime/rootfs/usr/lib64/libx.so").linkname == "../lib/libx.so"


def make_oci_layout(
    layout: Path,
    *,
    descriptor_platform: dict[str, str] | None = None,
    config_platform: dict[str, str] | None = None,
    manifest_count: int = 1,
) -> Path:
    blobs = layout / "blobs/sha256"
    blobs.mkdir(parents=True)
    (layout / "oci-layout").write_bytes(b'{"imageLayoutVersion":"1.0.0"}\n')
    descriptor_platform = descriptor_platform if descriptor_platform is not None else {"architecture": "amd64", "os": "linux"}
    config_platform = config_platform if config_platform is not None else {"architecture": "amd64", "os": "linux"}
    config_raw = json.dumps(config_platform, sort_keys=True, separators=(",", ":")).encode()
    layer_raw = b"layer"

    def descriptor(raw: bytes, media: str) -> dict:
        digest = hashlib.sha256(raw).hexdigest()
        (blobs / digest).write_bytes(raw)
        return {"mediaType": media, "digest": f"sha256:{digest}", "size": len(raw)}

    config = descriptor(config_raw, "application/vnd.oci.image.config.v1+json")
    layer = descriptor(layer_raw, "application/vnd.oci.image.layer.v1.tar")
    manifest_raw = json.dumps(
        {"schemaVersion": 2, "config": config, "layers": [layer]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest = descriptor(manifest_raw, "application/vnd.oci.image.manifest.v1+json")
    manifest["platform"] = descriptor_platform
    (layout / "index.json").write_text(
        json.dumps(
            {"schemaVersion": 2, "manifests": [copy.deepcopy(manifest) for _ in range(manifest_count)]},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return layout


def test_oci_closure_rejects_extra_blob_and_accepts_exact_graph(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_oci_test")
    layout = make_oci_layout(tmp_path / "oci")
    report = builder.verify_oci_layout(layout)
    assert report["verified"] is True
    assert report["platform"] == {"os": "linux", "architecture": "amd64"}
    assert report["manifest"]["config_digest"] == report["config"]["digest"]
    assert report["config"]["os"] == "linux"
    assert report["config"]["architecture"] == "amd64"
    assert report["descriptor_count"] == 3
    (layout / "blobs/sha256" / ("f" * 64)).write_bytes(b"extra")
    with pytest.raises(builder.ImageBuilderR5Error, match="unreferenced"):
        builder.verify_oci_layout(layout)


def test_oci_platform_chain_rejects_multi_arm64_missing_variant_and_wrong_config(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_oci_platform_test")
    multi = make_oci_layout(tmp_path / "multi", manifest_count=2)
    with pytest.raises(builder.ImageBuilderR5Error, match="one manifest"):
        builder.verify_oci_layout(multi)

    arm64 = make_oci_layout(tmp_path / "arm64", descriptor_platform={"architecture": "arm64", "os": "linux"})
    with pytest.raises(builder.ImageBuilderR5Error, match="descriptor platform"):
        builder.verify_oci_layout(arm64)

    missing = make_oci_layout(tmp_path / "missing", descriptor_platform={})
    with pytest.raises(builder.ImageBuilderR5Error, match="descriptor platform"):
        builder.verify_oci_layout(missing)

    variant = make_oci_layout(
        tmp_path / "variant",
        descriptor_platform={"architecture": "amd64", "os": "linux", "variant": "v3"},
    )
    with pytest.raises(builder.ImageBuilderR5Error, match="descriptor platform"):
        builder.verify_oci_layout(variant)

    wrong_config = make_oci_layout(
        tmp_path / "wrong-config",
        config_platform={"architecture": "arm64", "os": "linux"},
    )
    with pytest.raises(builder.ImageBuilderR5Error, match="config platform"):
        builder.verify_oci_layout(wrong_config)


def test_bundled_frontend_parser_accepts_only_exact_closed_dockerfile_grammar() -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_dockerfile_test")
    raw = (LANE / "Dockerfile.r5").read_bytes()
    report = builder.validate_dockerfile(raw)
    assert report["grammar"] == "bundled_frontend_closed_instruction_subset"
    assert report["external_frontend_directive"] is False
    assert report["copy_link_sources"] == 4

    with pytest.raises(builder.ImageBuilderR5Error, match="frontend directive"):
        builder.validate_dockerfile(b"# syntax=docker/dockerfile:1.7\n" + raw)
    with pytest.raises(builder.ImageBuilderR5Error, match="instruction grammar"):
        builder.validate_dockerfile(raw.replace(b"USER 1000:1000\n", b"RUN true\nUSER 1000:1000\n"))
    with pytest.raises(builder.ImageBuilderR5Error, match="COPY --link"):
        builder.validate_dockerfile(raw.replace(b"COPY --link runtime/rootfs/ /", b"COPY runtime/rootfs/ /"))

    assert builder.validate_buildx_version(
        b"github.com/docker/buildx v0.33.0 f7897eba028583e0071642db3c011e860444f8cf\n"
    ) == {"version": "v0.33.0", "commit": "f7897eba028583e0071642db3c011e860444f8cf"}
    inspect_fixture = b"""Name: default
Driver: docker
Nodes:
Name: default
Endpoint: default
Status: running
BuildKit version: v0.29.0
Platforms: linux/amd64, linux/amd64/v2
Devices:
 Name: nvidia.com/gpu=0
 Automatically allowed: false
"""
    inspected = builder.validate_builder_inspect(inspect_fixture)
    assert inspected["buildkit"] == "v0.29.0"
    assert inspected["gpu_devices_automatically_allowed"] is False
    with pytest.raises(builder.ImageBuilderR5Error, match="automatically allowed"):
        builder.validate_builder_inspect(inspect_fixture.replace(b"allowed: false", b"allowed: true"))


def test_buildx_argv_and_clean_environment_are_exact_and_override_closed(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_argv_test")
    layout = (tmp_path / "oci-layout").resolve()
    metadata = (tmp_path / ".oci-layout.build.metadata.json").resolve()
    argv = builder.build_argv(layout, metadata)
    assert argv == [
        "docker",
        "buildx",
        "build",
        "--builder",
        "default",
        "--file",
        "Dockerfile.r5",
        "--network=none",
        "--pull=false",
        "--no-cache",
        "--platform=linux/amd64",
        "--provenance=false",
        "--sbom=false",
        "--progress=rawjson",
        "--metadata-file",
        str(metadata),
        "--output",
        f"type=oci,dest={layout},tar=false",
        "-",
    ]
    environment = builder.clean_build_environment(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp/unit-home",
            "BUILDKIT_SYNTAX": "docker/dockerfile:labs",
            "BUILDKIT_FRONTEND": "gateway.v0",
            "BUILDX_BUILDER": "untrusted",
            "HTTP_PROXY": "http://untrusted.invalid",
        }
    )
    assert environment == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp/unit-home",
        "LC_ALL": "C.UTF-8",
        "DOCKER_CONTEXT": "default",
        "BUILDKIT_PROGRESS": "rawjson",
        "CUDA_VISIBLE_DEVICES": "-1",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }


def test_raw_progress_and_metadata_reject_remote_frontend_material() -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_material_test")
    base = builder.BASE_IMAGE_ID.removeprefix("sha256:")
    progress = json.dumps({"vertex": {"name": f"load exact ubuntu@sha256:{base}"}}).encode() + b"\n"
    report = builder.audit_build_material(
        progress,
        label="unit progress",
        require_raw_json_lines=True,
        require_base_digest=True,
    )
    assert report["base_digest_observed"] is True
    assert report["remote_frontend_material_observed"] is False
    metadata = json.dumps({"containerimage.digest": "sha256:" + "e" * 64}).encode()
    builder.audit_build_material(
        metadata,
        label="unit metadata",
        require_raw_json_lines=False,
        require_base_digest=False,
    )
    remote = json.dumps({"vertex": {"name": "resolve docker-image://docker.io/docker/dockerfile:1.7"}}).encode()
    with pytest.raises(builder.ImageBuilderR5Error, match="remote Dockerfile frontend"):
        builder.audit_build_material(
            remote,
            label="unit remote progress",
            require_raw_json_lines=True,
            require_base_digest=False,
        )


def test_context_receipt_pins_source_manifest_and_dockerfile_chain(tmp_path: Path) -> None:
    builder = load_lane_module("image_builder.py", "ppe_r5_image_builder_context_authority_test")

    def make_context(name: str, source_sha: str | None = None) -> tuple[Path, tuple[int, str]]:
        context = tmp_path / name
        context.mkdir()
        dockerfile_raw = (LANE / "Dockerfile.r5").read_bytes()
        manifest_raw = (LANE / "export-workload-manifest-r5.json").read_bytes()
        (context / "Dockerfile.r5").write_bytes(dockerfile_raw)
        (context / "workload-manifest.json").write_bytes(manifest_raw)
        contract_pin = pin(builder.SOURCE_CONTRACT)
        manifest_pin = (len(manifest_raw), hashlib.sha256(manifest_raw).hexdigest())
        receipt = {
            "schema_version": "deepsafe.ppe-safetyvision-cpu-image-context-r5/v1",
            "status": "frozen_context_ready_image_not_built",
            "source_contract": {
                "bytes": contract_pin[0],
                "sha256": source_sha if source_sha is not None else contract_pin[1],
            },
            "workload_manifest": {"bytes": manifest_pin[0], "sha256": manifest_pin[1]},
            "dockerfile": {"bytes": len(dockerfile_raw), "sha256": hashlib.sha256(dockerfile_raw).hexdigest()},
            "checkpoint_deserialized": False,
            "model_export_or_inference_executed": False,
            "image_built": False,
            "gpu_used": False,
        }
        raw = builder.canonical_bytes(receipt) + b"\n"
        path = context / "context-receipt.json"
        path.write_bytes(raw)
        for member in context.iterdir():
            member.chmod(0o440)
        context.chmod(0o550)
        return context, (len(raw), hashlib.sha256(raw).hexdigest())

    context, receipt_pin = make_context("good")
    receipt, raw = builder.load_context_authority(context, receipt_pin)
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == receipt_pin
    assert receipt["source_contract"]["sha256"] == pin(builder.SOURCE_CONTRACT)[1]

    drift, drift_pin = make_context("drift", "f" * 64)
    with pytest.raises(builder.ImageBuilderR5Error, match="source-contract authority"):
        builder.load_context_authority(drift, drift_pin)


def test_parity_worker_requires_lineage_inputs_raw_outputs_publisher_recheck_and_cpu_ep() -> None:
    source = (LANE / "parity_worker.py").read_text(encoding="utf-8")
    for required in (
        'lineage["historical_r3_terminal"]',
        'lineage["r5_export_terminal"]',
        '"raw_outputs": outputs',
        'read_exact(publisher_path',
        'providers=["CPUExecutionProvider"]',
        'session.get_providers() == ["CPUExecutionProvider"]',
        '"effective_ort_boundary"',
    ):
        assert required in source


def representative_export_terminal(worker: ModuleType) -> tuple[dict, dict]:
    artifacts = [
        {
            "profile": profile,
            "file": f"safetyvision-yolov8s-v2-{profile}-bdynamic-opset18.onnx",
            "bytes": 100_000 + profile,
            "sha256": ("a" if profile == 640 else "b") * 64,
            "input_shape": ["batch", 3, profile, profile],
            "output_shape": ["batch", 17, 8400 if profile == 640 else 18900],
            "nodes": 319,
            "initializers": 150,
            "opset": 18,
            "onnx_checker": "pass",
            "elapsed_seconds": 1.25,
        }
        for profile in (640, 960)
    ]
    workload_manifest = {"bytes": 4847, "sha256": "c" * 64}
    worker_receipt = {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-export-worker-receipt-r5/v1",
        "status": "cpu_export_completed_static_onnx_verified_not_model_accepted",
        "started_at": "2026-07-18T00:00:00+00:00",
        "finished_at": "2026-07-18T00:01:00+00:00",
        "workload_manifest": workload_manifest,
        "snapshot_pins": {},
        "runtime": {},
        "artifacts": artifacts,
        "execution": {},
        "accepted_model": False,
        "production_ready": False,
    }
    worker_raw = worker.canonical_bytes(worker_receipt) + b"\n"
    worker_pin = {"bytes": len(worker_raw), "sha256": hashlib.sha256(worker_raw).hexdigest()}
    terminal = {
        "schema_version": "deepsafe.ppe-safetyvision-cpu-export-terminal-r5/v1",
        "status": "cpu_export_completed_exact_image_evidence_not_model_accepted",
        "run_id": "unit-r5-export-001",
        "phase": "export",
        "plan": {},
        "workload_manifest": workload_manifest,
        "snapshot": {},
        "image": {},
        "container": {},
        "container_inspect": {},
        "worker_receipt": {"file": "worker-receipt.json", **worker_pin},
        "receipt_byte_equality": {
            "worker_stdout_suffix_equals_staged_receipt": True,
            "publication_method": "same_directory_atomic_rename_no_receipt_copy",
            "pre_publish_pin": worker_pin,
        },
        "worker": worker_receipt,
        "accepted_model": False,
        "production_ready": False,
    }
    graphs = {
        f"r5_{profile}": {
            "path": f"/opt/deepsafe/inputs/{artifact['file']}",
            "bytes": artifact["bytes"],
            "sha256": artifact["sha256"],
        }
        for profile, artifact in ((640, artifacts[0]), (960, artifacts[1]))
    }
    return terminal, graphs


def test_parity_round_trips_real_export_terminal_worker_artifact_shape() -> None:
    worker = load_lane_module("parity_worker.py", "ppe_r5_parity_lineage_positive_test")
    terminal, graphs = representative_export_terminal(worker)
    raw = worker.canonical_bytes(terminal) + b"\n"
    replay = worker.strict_object(raw, "representative R5 export terminal")
    pins = worker.export_artifact_pins_from_terminal(replay, graphs["r5_640"], graphs["r5_960"])
    assert pins == {
        "640": (100_640, "a" * 64),
        "960": (100_960, "b" * 64),
    }


def test_parity_export_terminal_lineage_rejects_shape_type_path_and_pin_drift() -> None:
    worker = load_lane_module("parity_worker.py", "ppe_r5_parity_lineage_negative_test")
    terminal, graphs = representative_export_terminal(worker)

    top_level_artifacts = copy.deepcopy(terminal)
    top_level_artifacts["artifacts"] = top_level_artifacts["worker"]["artifacts"]
    with pytest.raises(worker.ParityWorkerR5Error, match="terminal shape"):
        worker.export_artifact_pins_from_terminal(top_level_artifacts, graphs["r5_640"], graphs["r5_960"])

    missing_worker = copy.deepcopy(terminal)
    missing_worker.pop("worker")
    with pytest.raises(worker.ParityWorkerR5Error, match="terminal shape"):
        worker.export_artifact_pins_from_terminal(missing_worker, graphs["r5_640"], graphs["r5_960"])

    extra_worker_key = copy.deepcopy(terminal)
    extra_worker_key["worker"]["unexpected"] = True
    with pytest.raises(worker.ParityWorkerR5Error, match="worker receipt shape"):
        worker.export_artifact_pins_from_terminal(extra_worker_key, graphs["r5_640"], graphs["r5_960"])

    wrong_worker_type = copy.deepcopy(terminal)
    wrong_worker_type["worker"] = []
    with pytest.raises(worker.ParityWorkerR5Error, match="terminal worker"):
        worker.export_artifact_pins_from_terminal(wrong_worker_type, graphs["r5_640"], graphs["r5_960"])

    path_drift = copy.deepcopy(graphs)
    path_drift["r5_640"]["path"] = "/opt/deepsafe/inputs/wrong-640.onnx"
    with pytest.raises(worker.ParityWorkerR5Error, match="graph path"):
        worker.export_artifact_pins_from_terminal(terminal, path_drift["r5_640"], path_drift["r5_960"])

    pin_drift = copy.deepcopy(graphs)
    pin_drift["r5_960"]["sha256"] = "d" * 64
    with pytest.raises(worker.ParityWorkerR5Error, match="graph pin"):
        worker.export_artifact_pins_from_terminal(terminal, pin_drift["r5_640"], pin_drift["r5_960"])


def test_static_audit_has_no_network_process_write_or_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    report = r5.static_audit()
    assert report["image_built"] is False
    assert report["model_export_or_inference_executed"] is False
