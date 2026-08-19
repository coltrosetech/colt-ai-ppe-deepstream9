from __future__ import annotations

import ast
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_safetyvision_onnx_export_r4 as r4


ROOT = Path(__file__).resolve().parents[1]


def test_export_and_parity_evidence_replays() -> None:
    report = r4.audit()
    assert report["status"] == "cpu_export_and_functional_parity_verified_not_model_accepted"
    assert report["host_onnx_checker_passed"] == 2
    assert report["publisher_640_exact_output_parity"] is True
    assert report["dynamic_batch2_functional"] is True
    assert report["exact_960_batch1_functional"] is True
    assert report["batch12_proven"] is False
    assert report["gpu_used"] is False
    assert report["deepstream_or_tensorrt_executed"] is False
    assert report["accuracy_evaluated"] is False
    assert report["accepted_model"] is False
    assert report["production_ready"] is False


def test_graph_inventory_and_exact_pins() -> None:
    report = r4.audit()
    assert [(item["profile"], item["nodes"], item["initializers"]) for item in report["graphs"]] == [
        (640, 319, 150),
        (960, 319, 150),
    ]
    assert report["graphs"][0]["sha256"] == r4.PINS["onnx_640"][1]
    assert report["graphs"][1]["sha256"] == r4.PINS["onnx_960"][1]
    assert all(item["onnx_checker"] == "pass" for item in report["graphs"])


def test_validator_has_no_network_process_or_model_runtime_import() -> None:
    source = (ROOT / "validation/ppe_safetyvision_onnx_export_r4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "socket" not in imported
    assert "subprocess" not in imported
    assert "torch" not in imported
    assert "ultralytics" not in imported
    assert "onnxruntime" not in imported


def test_replay_does_not_use_network_process_or_write(
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
    report = r4.audit()
    assert report["host_onnx_checker_passed"] == 2


def test_all_evidence_is_read_only_regular_single_link() -> None:
    paths = [
        r4.EXPORT_RECEIPT,
        r4.CONTAINER_RECEIPT,
        r4.PARITY_RECEIPT,
        *r4.ARTIFACTS.values(),
    ]
    for path in paths:
        info = path.stat()
        assert info.st_nlink == 1
        assert info.st_mode & 0o222 == 0
        assert path.is_file()


def test_cli_emits_verified_non_acceptance_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert r4.main([]) == 0
    output = capsys.readouterr().out
    assert '"accepted_model": false' in output
    assert '"batch12_proven": false' in output
    assert '"publisher_640_exact_output_parity": true' in output
