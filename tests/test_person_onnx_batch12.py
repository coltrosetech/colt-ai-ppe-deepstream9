from __future__ import annotations

import copy
import json

import pytest

from validation.person_onnx_batch12 import (
    Batch12ContractError,
    ROOT,
    receipt_sha256,
    verify_receipt,
)


RECEIPT = ROOT / "validation/results/person/models/rtdetrv4-s-onnx-batch12-r1.json"
PIN = "e14e5382bd2b66d60c37e9dfd0f0b2473db74757a90390f3aeb69348f9e6f499"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = receipt_sha256(value)


def test_checked_in_batch12_receipt_is_exact_pinned() -> None:
    result = verify_receipt(_load(), expected_receipt_sha256=PIN)
    assert result == {
        "valid": True,
        "profiles": [640, 960],
        "batch": 12,
        "receipt_sha256": PIN,
        "production_ready": False,
    }


def test_batch12_receipt_tamper_is_rejected() -> None:
    value = _load()
    value["observations"][0]["outputs"]["boxes"]["shape"][0] = 11
    with pytest.raises(Batch12ContractError, match="self-hash"):
        verify_receipt(value, expected_receipt_sha256=PIN)


def test_resealed_deepstream_overclaim_is_rejected() -> None:
    value = copy.deepcopy(_load())
    value["gates"]["deepstream9_batch12_verified"] = True
    _reseal(value)
    with pytest.raises(Batch12ContractError, match="overclaims"):
        verify_receipt(value, expected_receipt_sha256=value["receipt_sha256"])
