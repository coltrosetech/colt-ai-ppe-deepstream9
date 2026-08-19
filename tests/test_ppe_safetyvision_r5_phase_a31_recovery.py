from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from validation import ppe_safetyvision_r5_phase_a31_recovery as a31


ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "models/ppe/export-lanes/safetyvision-yolov8s-v2-cpu-export-r5a31-recovery"


def pin(path: Path) -> tuple[int, str]:
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def synthetic_receipt() -> dict:
    receipt = {key: None for key in a31.RECOVERY_RECEIPT_KEYS}
    receipt["schema_version"] = "deepsafe.ppe-safetyvision-cpu-image-context-recovery-execution-a31/v1"
    receipt["status"] = "phase_a3_1_recovery_receipt_published_context_closed_build_not_authorized"
    unsigned = dict(receipt)
    del unsigned["self_fingerprint"]
    receipt["self_fingerprint"] = hashlib.sha256(a31.canonical_bytes(unsigned)).hexdigest()
    return receipt


def test_recovery_publish_keeps_run_absent_until_atomic_noreplace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = tmp_path / "results"
    container.mkdir()
    output = container / "phase-a3.1-001/receipt.json"
    real_mkdtemp = a31.tempfile.mkdtemp
    observed: list[bool] = []

    def checked_mkdtemp(*args, **kwargs):
        observed.append(not output.parent.exists())
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(a31.tempfile, "mkdtemp", checked_mkdtemp)
    receipt = synthetic_receipt()
    receipt_pin = a31.publish_recovery_receipt(output, receipt, expected_output=output)
    assert observed == [True]
    assert output.is_file()
    assert output.stat().st_mode & 0o777 == 0o440
    assert output.parent.stat().st_mode & 0o777 == 0o550
    assert pin(output) == receipt_pin
    assert not list(container.glob(".phase-a3.1-001.stage-*"))
    with pytest.raises(a31.PhaseA31RecoveryError, match="run already exists"):
        a31.publish_recovery_receipt(output, receipt, expected_output=output)
    assert pin(output) == receipt_pin


def test_recovery_publish_rejects_symlink_container_and_wrong_target(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "results"
    symlink.symlink_to(real, target_is_directory=True)
    output = symlink / "phase-a3.1-001/receipt.json"
    with pytest.raises(a31.PhaseA31RecoveryError, match="container differs"):
        a31.publish_recovery_receipt(output, synthetic_receipt(), expected_output=output)
    other = real / "other/receipt.json"
    with pytest.raises(a31.PhaseA31RecoveryError, match="path differs"):
        a31.publish_recovery_receipt(other, synthetic_receipt(), expected_output=real / "expected/receipt.json")


def test_historical_collision_is_failure_evidence_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision = tmp_path / "phase-a3-001"
    collision.mkdir(mode=0o700)
    monkeypatch.setattr(a31, "ROOT", tmp_path)
    monkeypatch.setattr(a31, "COLLISION_RUN", collision)
    report = a31._validate_empty_collision()
    assert report == {
        "path": "phase-a3-001",
        "mode": "0700",
        "empty": True,
        "receipt_absent": True,
        "treated_as_failure_evidence_only": True,
    }
    (collision / "receipt.json").write_bytes(b"unexpected")
    with pytest.raises(a31.PhaseA31RecoveryError, match="not empty"):
        a31._validate_empty_collision()


def test_strict_fingerprint_and_json_mutations_fail_closed() -> None:
    receipt = synthetic_receipt()
    a31.validate_fingerprint(receipt, "fixture")
    receipt["status"] = "mutated"
    with pytest.raises(a31.PhaseA31RecoveryError, match="fingerprint differs"):
        a31.validate_fingerprint(receipt, "fixture")
    with pytest.raises(a31.PhaseA31RecoveryError, match="duplicate JSON key"):
        a31.strict_object(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(a31.PhaseA31RecoveryError, match="non-finite JSON token"):
        a31.strict_object(b'{"x":NaN}', "nonfinite")


def test_recovery_program_has_no_context_copy_build_model_or_gpu_execution_surface() -> None:
    source = (ROOT / "validation/ppe_safetyvision_r5_phase_a31_recovery.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "prepare-context",
        "docker build",
        "buildx",
        "torch.load",
        "onnxruntime",
        "nvidia-smi",
        "deepstream-app",
        "trtexec",
        "subprocess.run",
        "shutil.copy",
        "copytree",
    ):
        assert forbidden not in lowered
    assert "rename_noreplace" in source
    assert "context_copied\": False" in source


def test_live_recovery_audit_replays_frozen_a2_a3_and_sources() -> None:
    report = a31.audit()
    assert report["status"] == "phase_a3_1_predecessor_context_closed_recovery_receipt_not_published"
    assert report["predecessor_a2"]["full_tree"] == a31.A2_TREE
    assert report["predecessor_a3"]["full_tree"] == a31.A3_TREE
    assert report["predecessor_a3"]["symlink_closure"] == a31.SYMLINK_CLOSURE
    assert report["historical_collision_evidence"]["treated_as_failure_evidence_only"] is True
    assert report["new_recovery_target"]["run_directory_absent"] is True
    assert report["context_copied"] is False
    assert report["checkpoint_deserialized"] is False
    assert report["model_export_or_inference_executed"] is False
    assert report["image_built"] is False
    assert report["gpu_used"] is False


def test_recovery_contract_and_official_command_are_closed() -> None:
    contract = json.loads((LANE / "source-contract-a31.json").read_text(encoding="utf-8"))
    unsigned = dict(contract)
    fingerprint = unsigned.pop("self_fingerprint")
    assert hashlib.sha256(a31.canonical_bytes(unsigned)).hexdigest() == fingerprint
    assert contract["authorization"]["context_copy"] is False
    assert contract["authorization"]["dedicated_cpu_image_build"] is False
    assert contract["authorization"]["gpu"] is False
    command = ROOT / "scripts/test-ppe-safetyvision-r5-phase-a31-recovery"
    source = command.read_text(encoding="utf-8")
    assert command.stat().st_mode & 0o111
    assert command.stat().st_mode & 0o222 == 0
    for expected in (
        "PYTHONDONTWRITEBYTECODE=1",
        "CUDA_VISIBLE_DEVICES=-1",
        "NVIDIA_VISIBLE_DEVICES=void",
        "test_ppe_safetyvision_r5_phase_a31_recovery.py",
    ):
        assert expected in source
