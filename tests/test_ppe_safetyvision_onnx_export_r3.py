from __future__ import annotations

import ast
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_safetyvision_onnx_export_r3 as export_r3


ROOT = Path(__file__).resolve().parents[1]


def test_static_audit_is_prepared_and_does_not_claim_execution() -> None:
    report = export_r3.audit()
    assert report["status"] == "prepared_execution_closed_not_yet_executed"
    assert report["profiles"] == [640, 960]
    assert report["batch_profile"] == {"minimum": 1, "optimum": 12, "maximum": 12}
    assert report["checkpoint_deserialized"] is False
    assert report["cpu_export_trace_executed"] is False
    assert report["cpu_or_gpu_evaluation_inference_executed"] is False
    assert report["gpu_used"] is False
    assert report["accepted_model"] is False
    assert report["production_ready"] is False


def test_contract_records_exact_profiles_runtime_and_closed_boundary() -> None:
    contract = json.loads(export_r3.CONTRACT.read_text(encoding="utf-8"))
    assert len(contract["runtime_records"]) == 26
    assert contract["export"]["profiles"] == [640, 960]
    assert contract["export"]["anchors"] == {"640": 8400, "960": 18900}
    assert contract["export"]["runtime_class_mapping"] == {
        "helmet": 3,
        "no_helmet": 7,
        "hi_vis": 12,
        "no_hi_vis": 9,
    }
    assert contract["execution_boundary"]["network"] == "none"
    assert contract["execution_boundary"]["container_root"] == "read_only"
    assert contract["execution_boundary"]["workspace_mount"] is False
    assert contract["execution_boundary"]["docker_socket_mount"] is False
    assert contract["execution_boundary"]["gpu_device_request"] is False
    assert contract["authorization"]["checkpoint_deserialization_during_execute"] is True
    assert contract["authorization"]["cpu_export_trace_during_execute"] is True
    assert contract["authorization"]["cpu_evaluation_inference"] is False
    assert contract["authorization"]["gpu_export_or_inference"] is False


def test_worker_has_no_process_network_docker_or_gpu_control_surface() -> None:
    source = export_r3.ENTRYPOINT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "socket" not in imported
    assert "subprocess" not in imported
    assert "docker run" not in source.lower()
    assert "docker create" not in source.lower()
    assert "nvidia-smi" not in source.lower()
    assert "trtexec" not in source.lower()
    assert "deepstream-app" not in source.lower()


def test_docker_create_argv_is_cpu_only_and_mount_minimal(tmp_path: Path) -> None:
    staging = tmp_path / "output"
    inputs = tmp_path / "inputs"
    staging.mkdir()
    inputs.mkdir()
    (inputs / "contract.json").write_bytes(b"{}")
    (inputs / "entrypoint.py").write_bytes(b"pass\n")
    argv = export_r3.build_create_argv(
        "static-test-001",
        staging,
        inputs,
        preflight_only=False,
    )
    assert argv[:4] == ["docker", "create", "--pull", "never"]
    assert argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert argv[argv.index("--pids-limit") + 1] == "256"
    assert "--gpus" not in argv
    assert "--device" not in argv
    assert "nvidia" not in " ".join(argv).lower().replace("nvidia_visible_devices=void", "")
    mounts = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
    assert len(mounts) == 9
    assert not any("docker.sock" in value for value in mounts)
    assert not any(f"src={ROOT.resolve()}," in value for value in mounts)
    writable = [value for value in mounts if not value.endswith(",readonly")]
    assert len(writable) == 1 and "dst=/output" in writable[0]


def test_preflight_flag_is_explicit_and_execute_argv_omits_it(tmp_path: Path) -> None:
    staging = tmp_path / "output"
    inputs = tmp_path / "inputs"
    staging.mkdir()
    inputs.mkdir()
    run = export_r3.build_create_argv("test-run", staging, inputs, preflight_only=False)
    preflight = export_r3.build_create_argv("test-preflight", staging, inputs, preflight_only=True)
    assert "--preflight-only" not in run
    assert preflight[-1] == "--preflight-only"


def test_audit_performs_no_network_process_or_write(
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
    report = export_r3.audit()
    assert report["checkpoint_deserialized"] is False
    assert report["cpu_export_trace_executed"] is False


def test_contract_and_worker_pins_fail_closed_on_single_byte_mutation(
    tmp_path: Path,
) -> None:
    contract_raw = export_r3.CONTRACT.read_bytes()
    entrypoint_raw = export_r3.ENTRYPOINT.read_bytes()
    contract_mutated = bytearray(contract_raw)
    entrypoint_mutated = bytearray(entrypoint_raw)
    contract_mutated[len(contract_mutated) // 2] ^= 1
    entrypoint_mutated[len(entrypoint_mutated) // 2] ^= 1
    contract_path = tmp_path / "contract.json"
    entrypoint_path = tmp_path / "entrypoint.py"
    contract_path.write_bytes(contract_mutated)
    entrypoint_path.write_bytes(entrypoint_mutated)
    with pytest.raises(export_r3.SafetyVisionExportR3Error, match="file pin differs"):
        export_r3._read_exact(contract_path, export_r3.CONTRACT_PIN)
    with pytest.raises(export_r3.SafetyVisionExportR3Error, match="file pin differs"):
        export_r3._read_exact(entrypoint_path, export_r3.ENTRYPOINT_PIN)


def test_cli_audit_emits_strict_non_execution_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert export_r3.main(["audit"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "prepared_execution_closed_not_yet_executed"
    assert value["gpu_used"] is False
