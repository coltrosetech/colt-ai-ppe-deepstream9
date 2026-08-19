from __future__ import annotations

import ast
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_safetyvision_checkpoint_quarantine_r2 as checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_archive_replays_without_deserialization() -> None:
    report = checkpoint.audit()
    assert report["status"] == "checkpoint_acquired_static_archive_audited_never_deserialized"
    assert report["checkpoint_sha256"] == checkpoint.CHECKPOINT_PIN[1]
    assert report["zip_member_count"] == 365
    assert report["pickle_global_reference_count"] == 24
    assert report["checkpoint_deserialized"] is False
    assert report["cpu_or_gpu_export_executed"] is False
    assert report["cpu_or_gpu_inference_executed"] is False
    assert report["accepted_model"] is False
    assert report["production_ready"] is False


def test_checkpoint_auditor_imports_no_model_runtime() -> None:
    source = (
        ROOT / "validation/ppe_safetyvision_checkpoint_quarantine_r2.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
    assert "ultralytics" not in imported
    assert "onnx" not in imported
    assert "onnxruntime" not in imported
    assert "pickle" not in imported
    assert "pickletools" in imported


def test_auditor_has_no_network_process_write_or_model_execution_surface(
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
    report = checkpoint.audit()
    assert report["checkpoint_deserialized"] is False
    assert report["cpu_or_gpu_export_executed"] is False


def test_cli_emits_static_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert checkpoint.main([]) == 0
    output = capsys.readouterr().out
    assert "checkpoint_acquired_static_archive_audited_never_deserialized" in output
    assert '"checkpoint_deserialized": false' in output
