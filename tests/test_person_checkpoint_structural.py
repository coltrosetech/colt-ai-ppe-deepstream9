from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation.person_checkpoint_structural import (
    ROOT,
    StructuralReceiptError,
    receipt_sha256,
    verify_receipt,
)


RECEIPT = (
    ROOT
    / "validation/results/person/models/rtdetrv4-s-structural-load-r1.json"
)
PIN = "498d1f7ccfb00f540d2ea44e4da7a097ffa867e149f130f08c2b6a45b8a76a06"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _reseal(value: dict) -> dict:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = receipt_sha256(value)
    return value


def test_checked_in_receipt_is_exact_pinned_and_fail_closed() -> None:
    receipt = _load()
    result = verify_receipt(receipt, expected_receipt_sha256=PIN)

    assert result == {
        "valid": True,
        "candidate_id": "rtdetrv4-s",
        "receipt_sha256": PIN,
        "external_pin_verified": True,
        "structural_load_verified": True,
        "production_ready": False,
    }
    assert receipt["checkpoint_structure"]["model"]["tensor_count"] == 796
    assert receipt["checkpoint_structure"]["ema_module"]["tensor_count"] == 796
    assert receipt["checkpoint_structure"]["model"]["tensor_value_count"] == 10589534
    assert receipt["architecture"]["parameter_count"] == 10519253
    assert receipt["execution"]["forward_pass_executed"] is False
    assert receipt["conclusions"]["production_ready"] is False


def test_receipt_tamper_is_rejected() -> None:
    receipt = _load()
    receipt["architecture"]["parameter_count"] += 1
    with pytest.raises(StructuralReceiptError, match="self-hash"):
        verify_receipt(receipt, expected_receipt_sha256=PIN)


def test_resealed_overclaim_is_rejected_even_with_matching_external_pin() -> None:
    receipt = copy.deepcopy(_load())
    receipt["conclusions"]["production_ready"] = True
    _reseal(receipt)
    with pytest.raises(StructuralReceiptError, match="overclaims"):
        verify_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_resealed_gpu_claim_is_rejected_even_with_matching_external_pin() -> None:
    receipt = copy.deepcopy(_load())
    receipt["execution"]["gpu_touched"] = True
    _reseal(receipt)
    with pytest.raises(StructuralReceiptError, match="execution boundary"):
        verify_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_schema_declares_a_closed_top_level_contract() -> None:
    schema = json.loads(
        (
            ROOT
            / "validation/schemas/person-checkpoint-structural-receipt-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"].endswith("/v1")
    assert set(schema["required"]) == set(schema["properties"])
