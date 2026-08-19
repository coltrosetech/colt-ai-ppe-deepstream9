from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_structural as structural


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "validation/results/pose/models/"
    "mmpose-yoloxpose-s-structural-r1.json"
)
SCHEMA = (
    ROOT
    / "validation/schemas/"
    "pose-mmpose-yoloxpose-structural-receipt-v1.schema.json"
)
PLAN = (
    ROOT
    / "models/pose/challengers/mmpose-yoloxpose-s/"
    "provenance-plan-v1.json"
)
POSE_PROVENANCE = ROOT / "models/pose/provenance-plan.json"
CHECKPOINT = (
    ROOT
    / "models/pose/candidates/mmpose-yoloxpose-s/"
    "yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth"
)
EXPECTED_RECEIPT_SELF_SHA = (
    "583270ea19bfab69da9cf7c4490502a209ef2228aaec6e8cb5d098d3341b2bd1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("receipt_sha256", None)
    return structural.seal_receipt(unsigned)


def test_checked_in_receipt_verifies_against_exact_external_pin() -> None:
    result = structural.verify_receipt(
        _load(RECEIPT),
        expected_receipt_sha256=EXPECTED_RECEIPT_SELF_SHA,
    )

    assert result == {
        "valid": True,
        "candidate_id": "mmpose-yoloxpose-s",
        "receipt_sha256": EXPECTED_RECEIPT_SELF_SHA,
        "external_pin_verified": True,
        "strict_architecture_load_verified": True,
        "cpu_raw_profiles_verified": [640, 960],
        "production_ready": False,
    }


def test_closed_schema_accepts_receipt_and_rejects_extra_fields() -> None:
    schema = _load(SCHEMA)
    receipt = _load(RECEIPT)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(receipt, schema)

    receipt["untrusted_claim"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(receipt, schema)


def test_receipt_records_only_cpu_batch1_raw_shape_evidence() -> None:
    receipt = _load(RECEIPT)

    assert receipt["checkpoint_structure"]["state_dict"]["tensor_count"] == 547
    assert receipt["architecture"]["parameter_count"] == 10_729_963
    assert receipt["execution"]["gpu_touched"] is False
    assert receipt["execution"]["raw_forward_batches"] == [1]
    assert receipt["execution"]["batch12_forward_executed"] is False
    assert receipt["raw_forward_profiles"]["640"]["input_shape"] == [
        1,
        3,
        640,
        640,
    ]
    assert receipt["raw_forward_profiles"]["960"]["input_shape"] == [
        1,
        3,
        960,
        960,
    ]
    assert receipt["conclusions"]["profile_960_upstream_quality_claimed"] is False
    assert receipt["conclusions"]["onnx_exported"] is False
    assert receipt["conclusions"]["deepstream9_parity_passed"] is False
    assert receipt["conclusions"]["production_ready"] is False


def test_unsealed_tamper_and_resealed_gpu_or_readiness_overclaim_fail() -> None:
    tampered = _load(RECEIPT)
    tampered["architecture"]["parameter_count"] += 1
    with pytest.raises(structural.PoseStructuralError, match="self-hash"):
        structural.verify_receipt(
            tampered,
            expected_receipt_sha256=EXPECTED_RECEIPT_SELF_SHA,
        )

    gpu_claim = _load(RECEIPT)
    gpu_claim["execution"]["gpu_touched"] = True
    gpu_claim = _reseal(gpu_claim)
    with pytest.raises(structural.PoseStructuralError, match="execution boundary"):
        structural.verify_receipt(
            gpu_claim,
            expected_receipt_sha256=gpu_claim["receipt_sha256"],
        )

    ready_claim = _load(RECEIPT)
    ready_claim["conclusions"]["production_ready"] = True
    ready_claim = _reseal(ready_claim)
    with pytest.raises(structural.PoseStructuralError, match="overclaims"):
        structural.verify_receipt(
            ready_claim,
            expected_receipt_sha256=ready_claim["receipt_sha256"],
        )


def test_challenger_plan_is_self_pinned_and_keeps_every_unrun_gate_closed() -> None:
    plan = _load(PLAN)
    unsigned = copy.deepcopy(plan)
    observed = unsigned.pop("fingerprint_sha256")
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == observed
    assert plan["license"]["spdx"] == "Apache-2.0"
    assert plan["candidate"]["production_model_selected"] is False
    assert plan["profiles"]["960"]["feasibility_profile_only"] is True
    assert plan["profiles"]["960"]["quality_verified"] is False
    assert all(value is False for value in plan["gates"].values())

    pose = _load(POSE_PROVENANCE)
    assert pose["selected_model"]["model_id"] == "yolo26s-pose"
    assert pose["license"]["decision"] is None
    assert pose["permissive_challenger"]["production_model_selected"] is False


def test_checkpoint_and_receipt_are_read_only_exact_pins() -> None:
    plan = _load(PLAN)
    checkpoint_pin = plan["acquisition"]["checkpoint"]
    receipt_pin = plan["structural_evidence"]["receipt"]

    assert stat.S_IMODE(CHECKPOINT.stat().st_mode) == 0o440
    assert stat.S_IMODE(RECEIPT.stat().st_mode) == 0o440
    assert CHECKPOINT.stat().st_size == checkpoint_pin["bytes"]
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == checkpoint_pin["sha256"]
    assert RECEIPT.stat().st_size == receipt_pin["bytes"]
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == receipt_pin["sha256"]

