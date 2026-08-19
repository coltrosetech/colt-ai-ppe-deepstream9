from __future__ import annotations

import copy
import json

import pytest

from validation.person_checkpoint_structural import StructuralReceiptError, receipt_sha256
from validation.person_framework_profiles import ROOT, verify_receipt


RECEIPT = ROOT / "validation/results/person/models/rtdetrv4-s-framework-profiles-r1.json"
PIN = "8e44521abad4a8984859c61f4706b6fd37c614df829f2e13573bf3a17f9e1ea4"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = receipt_sha256(value)


def test_profile_receipt_is_exact_pinned_and_keeps_readiness_closed() -> None:
    receipt = _load()
    result = verify_receipt(receipt, expected_receipt_sha256=PIN)
    assert result["valid"] is True
    assert result["profiles"] == [640, 960]
    assert result["production_ready"] is False
    profiles = receipt["profile_contract"]["profiles"]
    assert profiles[0]["spatial_tensors"]["decoder.anchors"]["shape"] == [1, 8400, 4]
    assert profiles[1]["spatial_tensors"]["decoder.anchors"]["shape"] == [1, 18900, 4]
    assert profiles[0]["learned_parameter_sha256_after"] == profiles[1]["learned_parameter_sha256_after"]


def test_profile_receipt_tamper_is_rejected() -> None:
    receipt = _load()
    receipt["profile_contract"]["profiles"][1]["profile"] = 961
    with pytest.raises(StructuralReceiptError, match="self-hash"):
        verify_receipt(receipt, expected_receipt_sha256=PIN)


def test_resealed_spatial_dynamic_overclaim_is_rejected() -> None:
    receipt = copy.deepcopy(_load())
    receipt["profile_contract"]["spatial_axes_dynamic"] = True
    _reseal(receipt)
    with pytest.raises(StructuralReceiptError, match="spatial axis"):
        verify_receipt(receipt, expected_receipt_sha256=receipt["receipt_sha256"])


def test_resealed_unapproved_960_tensor_change_is_rejected() -> None:
    receipt = copy.deepcopy(_load())
    receipt["profile_contract"]["profiles"][1][
        "regenerated_nonlearned_tensor_allowlist"
    ].append("backbone.stem.stem1.conv.weight")
    _reseal(receipt)
    with pytest.raises(StructuralReceiptError, match="unexpected spatial"):
        verify_receipt(receipt, expected_receipt_sha256=receipt["receipt_sha256"])
