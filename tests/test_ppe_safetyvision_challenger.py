from __future__ import annotations

import copy
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_safetyvision_challenger as challenger


def _write_read_only(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.chmod(0o440)
    return path


def test_exact_challenger_files_and_four_class_projection() -> None:
    report = challenger.audit()
    assert report["status"] == "acquired_static_verified_not_executed_not_accepted"
    assert report["source_commit"] == challenger.COMMIT
    assert report["verified_artifact_count"] == 3
    assert report["onnx_artifact_count"] == 2
    assert report["recorded_onnx_checker_passed"] is True
    assert report["runtime_class_mapping"] == {
        "helmet": 3,
        "no_helmet": 7,
        "hi_vis": 12,
        "no_hi_vis": 9,
    }
    assert report["accepted_model"] is False
    assert report["cpu_inference_executed"] is False
    assert report["gpu_inference_executed"] is False
    assert report["tensorrt_or_deepstream_executed"] is False
    assert report["target_960_present"] is False
    assert report["dynamic_batch_12_present"] is False
    assert report["product_acceptance_authorized"] is False


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("authorization", "gpu_inference"),
        ("authorization", "training"),
        ("authorization", "product_acceptance"),
        ("limitations", "overhead_or_drone_camera_supported_by_author"),
        ("limitations", "target_960_onnx_present"),
        ("limitations", "batch_12_dynamic_profile_present"),
        ("selection", "accepted_model"),
    ],
)
def test_challenger_cannot_be_promoted_by_manifest_edit(
    tmp_path: Path,
    section: str,
    key: str,
) -> None:
    value, _ = challenger.load_manifest()
    tampered = copy.deepcopy(value)
    tampered[section][key] = True
    path = _write_read_only(tmp_path / f"{section}-{key}.json", tampered)
    with pytest.raises(challenger.SafetyVisionChallengerError):
        challenger.audit(path)


def test_auditor_exposes_no_network_process_write_or_inference_surface(
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
    report = challenger.audit()
    assert report["accepted_model"] is False
    assert report["tensorrt_or_deepstream_executed"] is False


def test_cli_emits_read_only_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert challenger.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["onnx_artifact_count"] == 2
    assert report["accepted_model"] is False
