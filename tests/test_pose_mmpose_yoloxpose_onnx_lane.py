from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_onnx_lane as lane


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(lane.SCHEMA.read_text(encoding="utf-8"))
R1_PATH = (
    ROOT
    / "validation/results/pose/models/"
    "mmpose-yoloxpose-s-onnx-lane-r1.json"
)
R2_PATH = (
    ROOT
    / "validation/results/pose/models/"
    "mmpose-yoloxpose-s-onnx-lane-r2.json"
)
R1_FILE_SHA256 = (
    "84ed90c532b5d87391fd3e11edd2a810866f2157c0cc8fda981d974964f71bc0"
)
R1_SELF_SHA256 = (
    "bf9835d33cac2cdf7bc98e509633998fd321183a86483826055dd83e95623044"
)
R2_FILE_SHA256 = (
    "62bbd3faed88d13df3de3673f5cc88a33d6f97397e7b76213c5d4c5bc657ed13"
)
R2_SELF_SHA256 = (
    "2fd5a445d5a51a4377def5d900515e2adf5a3773d160e2b8bc5e0da06e7686e0"
)
EXPORT_PYTHON = ROOT / ".venv-export/bin/python"
R1_BLOCKERS = [
    "mmdeploy_checkout_missing",
    "mmdeploy_distribution_missing_or_wrong",
    "compiled_mmcv_ops_missing",
]
R2_BLOCKERS = [
    "mmdeploy_distribution_missing_or_wrong",
    "compiled_mmcv_ops_missing",
]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return lane._strict_json_bytes(
        R2_PATH.read_bytes(), source=R2_PATH.as_posix()
    )


def _verify_with_export_environment(
    receipt_path: Path,
    expected_receipt_sha256: str,
) -> subprocess.CompletedProcess[str]:
    assert EXPORT_PYTHON.is_file()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [
            str(EXPORT_PYTHON),
            "-m",
            "validation.pose_mmpose_yoloxpose_onnx_lane",
            "verify",
            "--receipt",
            str(receipt_path),
            "--expected-receipt-sha256",
            expected_receipt_sha256,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )


def test_current_preflight_is_honestly_blocked(receipt: dict) -> None:
    assert receipt["status"] == "blocked_preflight_no_export_attempted"
    assert [item["code"] for item in receipt["blockers"]] == R2_BLOCKERS
    checkout = receipt["tooling"]["mmdeploy_checkout"]
    assert checkout["exists"] is True
    assert checkout["observed_commit"] == lane.MMDEPLOY_COMMIT
    assert checkout["missing_required_paths"] == []
    assert receipt["official_mmdeploy"]["local_source_bytes_verified"] is True


def test_checked_in_r2_is_the_exact_current_immutable_snapshot() -> None:
    raw = R2_PATH.read_bytes()
    assert stat.S_IMODE(R2_PATH.stat().st_mode) == 0o440
    assert hashlib.sha256(raw).hexdigest() == R2_FILE_SHA256
    r2 = lane._strict_json_bytes(raw, source=R2_PATH.as_posix())
    jsonschema.Draft202012Validator(SCHEMA).validate(r2)
    assert r2["receipt_sha256"] == R2_SELF_SHA256
    assert lane.receipt_sha256(r2) == R2_SELF_SHA256
    assert [item["code"] for item in r2["blockers"]] == R2_BLOCKERS
    assert r2["official_mmdeploy"]["local_source_bytes_verified"] is True
    assert r2["tooling"]["mmdeploy_checkout"]["exists"] is True


def test_checked_in_r1_remains_exact_historical_three_blocker_snapshot() -> None:
    raw = R1_PATH.read_bytes()
    assert stat.S_IMODE(R1_PATH.stat().st_mode) == 0o440
    assert hashlib.sha256(raw).hexdigest() == R1_FILE_SHA256
    r1 = lane._strict_json_bytes(raw, source=R1_PATH.as_posix())
    jsonschema.Draft202012Validator(SCHEMA).validate(r1)
    assert r1["receipt_sha256"] == R1_SELF_SHA256
    assert lane.receipt_sha256(r1) == R1_SELF_SHA256
    assert [item["code"] for item in r1["blockers"]] == R1_BLOCKERS
    assert r1["official_mmdeploy"]["local_source_bytes_verified"] is False
    assert r1["tooling"]["mmdeploy_checkout"]["exists"] is False
    assert r1["tooling"]["mmdeploy_checkout"]["observed_commit"] is None
    assert r1["execution_boundary"]["export_attempted"] is False


def test_receipt_schema_accepts_current_preflight(receipt: dict) -> None:
    jsonschema.Draft202012Validator(SCHEMA).validate(receipt)


def test_receipt_self_hash_is_canonical(receipt: dict) -> None:
    assert receipt["receipt_sha256"] == lane.receipt_sha256(receipt)


def test_receipt_replay_is_exact(receipt: dict) -> None:
    completed = _verify_with_export_environment(
        R2_PATH,
        receipt["receipt_sha256"],
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["valid"] is True
    assert result["blocker_codes"] == R2_BLOCKERS
    assert result["export_executed"] is False
    assert result["production_ready"] is False


def test_receipt_external_hash_tamper_fails(receipt: dict) -> None:
    with pytest.raises(lane.PoseOnnxLaneError, match="external receipt pin"):
        lane.verify_receipt(receipt, expected_receipt_sha256="0" * 64)


def test_receipt_semantic_tamper_fails_even_when_resealed(
    receipt: dict,
    tmp_path: Path,
) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["blockers"][0]["detail"] = "forged"
    tampered["receipt_sha256"] = lane.receipt_sha256(tampered)
    path = tmp_path / "tampered-r2.json"
    path.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = _verify_with_export_environment(
        path,
        tampered["receipt_sha256"],
    )
    assert completed.returncode != 0
    assert "receipt replay differs" in completed.stderr


def test_schema_rejects_execution_overclaim(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["execution_boundary"]["export_attempted"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(SCHEMA).validate(tampered)


def test_schema_rejects_unknown_top_level_key(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(SCHEMA).validate(tampered)


def test_official_mmdeploy_pin_and_fix_are_exact(receipt: dict) -> None:
    official = receipt["official_mmdeploy"]
    assert official["tag"] == "v1.3.1"
    assert official["commit"] == "bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
    assert official["release_url"].endswith("/releases/tag/v1.3.1")
    assert official["core_mmpose_yoloxpose_fix_url"].endswith("/pull/2466")
    assert official["local_source_bytes_verified"] is True


def test_exact_onnx_output_contract() -> None:
    contract = lane._read_json(lane.CONTRACT)
    lane._validate_contract(contract)
    assert contract["onnx"]["input"]["shape"] == ["B", 3, "H", "W"]
    assert contract["onnx"]["input"]["required_batches"] == [1, 12]
    assert contract["onnx"]["outputs"][0]["shape"] == ["B", 100, 5]
    assert contract["onnx"]["outputs"][1]["shape"] == ["B", 100, 17, 3]
    assert contract["onnx"]["post_processing"]["keep_top_k"] == 100


def test_640_and_960_are_separate_static_spatial_profiles() -> None:
    contract = lane._read_json(lane.CONTRACT)
    assert contract["profiles"]["640"]["spatial_shape"] == [640, 640]
    assert contract["profiles"]["960"]["spatial_shape"] == [960, 960]
    assert contract["profiles"]["960"]["trained_resolution"] is False
    assert contract["profiles"]["960"]["quality_claimed"] is False
    assert contract["onnx"]["input"]["spatial_axes_dynamic"] is False


def test_overlay_configs_pin_only_spatial_shape() -> None:
    for size, path in lane.OVERLAYS.items():
        lane._validate_overlay(path, size)
        source = path.read_text(encoding="utf-8")
        assert "pose-detection_yolox-pose_onnxruntime_dynamic.py" in source
        assert "dynamic-b1-12.onnx" in source


def test_deepstream_handoff_records_direct_conflict_and_mapping() -> None:
    handoff = lane._read_json(lane.HANDOFF)
    lane._validate_handoff(handoff)
    audit = handoff["compatibility_audit"]
    assert audit["directly_compatible"] is False
    assert audit["existing_output"]["shape"] == ["B", 300, 57]
    assert audit["challenger_output"]["layers"][0]["shape"] == ["B", 100, 5]
    mapping = handoff["adapter"]["mapping"]
    assert mapping["destination_row_5"] == 0
    assert mapping["destination_rows_100_299"] == "all-zero padding rows"


def test_handoff_cannot_change_instance_association() -> None:
    handoff = lane._read_json(lane.HANDOFF)
    rules = handoff["adapter"]["semantic_rules"]
    assert rules["nms_must_not_run_again"] is True
    assert rules["dets_and_keypoints_instance_index_must_not_be_reordered_independently"] is True


def test_yolo26_license_decision_is_unchanged(receipt: dict) -> None:
    assert receipt["license_boundary"] == {
        "challenger_license_spdx": "Apache-2.0",
        "yolo26_license_decision_changed": False,
        "production_model_selected": False,
        "production_ready": False,
    }
    contract = lane._read_json(lane.CONTRACT)
    assert contract["candidate"]["replaces_yolo26_selection"] is False


def test_preflight_does_not_publish_or_claim_onnx(receipt: dict) -> None:
    assert receipt["execution_boundary"]["export_attempted"] is False
    assert receipt["execution_boundary"]["onnxruntime_executed"] is False
    for profile in receipt["profiles"].values():
        assert profile["onnx_file_published"] is False
        assert profile["onnx_checker_passed"] is False
        assert profile["onnxruntime_batch12_passed"] is False


def test_atomic_publication_refuses_overwrite(tmp_path: Path, receipt: dict) -> None:
    output = tmp_path / "receipt.json"
    lane.atomic_write_no_overwrite(output, receipt)
    before = output.read_bytes()
    assert oct(output.stat().st_mode & 0o777) == "0o440"
    with pytest.raises(lane.PoseOnnxLaneError, match="refusing to overwrite"):
        lane.atomic_write_no_overwrite(output, receipt)
    assert output.read_bytes() == before


def test_atomic_publication_rejects_symlinked_parent(tmp_path: Path, receipt: dict) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(lane.PoseOnnxLaneError, match="symlink"):
        lane.atomic_write_no_overwrite(link / "receipt.json", receipt)


def test_contract_tamper_is_rejected_before_any_export() -> None:
    contract = lane._read_json(lane.CONTRACT)
    contract["onnx"]["outputs"][0]["shape"] = ["B", 300, 5]
    with pytest.raises(lane.PoseOnnxLaneError, match="dets contract"):
        lane._validate_contract(contract)


def test_planned_commands_are_cpu_only_and_not_shell_strings(receipt: dict) -> None:
    for command in receipt["planned_commands_not_executed"].values():
        assert isinstance(command, list)
        assert command[-2:] == ["--device", "cpu"]
        assert "third_party/mmdeploy/tools/deploy.py" in command
        assert all("cuda" not in token.lower() for token in command)
