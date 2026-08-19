from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "models/person/export_rtdetrv4.py"
SPEC = importlib.util.spec_from_file_location("person_rtdetrv4_export", SCRIPT)
assert SPEC and SPEC.loader
exporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporting)
PLAN = ROOT / "models/person/candidates/rtdetrv4-s/export-plan-v1.json"
PINS = {
    640: "4a0f1550df7f0777a5d554b799ab29ffe6b7e68323d4e0aa1bd7e6eebc7324a1",
    960: "69c90420ee631f2d8a784c1fa042c56da19b49e49b5ca87c7366e195ec10713c",
}


def _receipt(profile: int) -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    path = ROOT / plan["profiles"][str(profile)]["receipt_path"]
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_export_plan_is_self_hashed_and_source_pinned() -> None:
    plan = exporting._load_plan()
    assert plan["profiles"]["640"]["spatial"] == [640, 640]
    assert plan["profiles"]["960"]["spatial"] == [960, 960]
    assert plan["batch_profile"] == {"min": 1, "opt": 12, "max": 12}
    assert plan["spatial_axes_dynamic"] is False
    assert all(value is False for value in plan["acceptance"].values())


@pytest.mark.parametrize("profile", [640, 960])
def test_checked_in_onnx_receipt_is_exact_pinned(profile: int) -> None:
    result = exporting.verify_export_receipt(
        _receipt(profile), expected_receipt_sha256=PINS[profile]
    )
    assert result["valid"] is True
    assert result["profile"] == profile
    assert result["synthetic_onnx_parity_passed"] is True
    assert result["production_ready"] is False


def test_resealed_spatial_dynamic_overclaim_is_rejected() -> None:
    value = copy.deepcopy(_receipt(640))
    value["export"]["spatial_axes_dynamic"] = True
    value.pop("receipt_sha256")
    value["receipt_sha256"] = exporting.receipt_sha256(value)
    with pytest.raises(exporting.ExportContractError, match="spatial axes"):
        exporting.verify_export_receipt(
            value, expected_receipt_sha256=value["receipt_sha256"]
        )


def test_resealed_production_overclaim_is_rejected() -> None:
    value = copy.deepcopy(_receipt(960))
    value["acceptance"]["production_ready"] = True
    value.pop("receipt_sha256")
    value["receipt_sha256"] = exporting.receipt_sha256(value)
    with pytest.raises(exporting.ExportContractError, match="overclaims"):
        exporting.verify_export_receipt(
            value, expected_receipt_sha256=value["receipt_sha256"]
        )
