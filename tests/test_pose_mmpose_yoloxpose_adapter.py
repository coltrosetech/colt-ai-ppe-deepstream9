from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = "2026-07-18T03:40:00+03:00"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return adapter.build_receipt(created_at=CREATED_AT)


def test_adapter_receipt_matches_closed_schema(receipt: dict) -> None:
    schema = json.loads(adapter.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(receipt)


def test_adapter_receipt_proves_only_the_standalone_cpu_boundary(
    receipt: dict,
) -> None:
    assert receipt["status"] == (
        "standalone_cpu_packer_verified_deepstream9_bridge_pending"
    )
    assert receipt["toolchain_and_tests"]["unit_case_count"] == 12
    assert receipt["verified_tensor_contract"]["batch_range"] == [1, 12]
    assert receipt["verified_tensor_contract"]["source_layers_in_order"] == [
        "dets",
        "keypoints",
    ]
    assert receipt["verified_tensor_contract"]["canonical_shape"] == [
        "B",
        300,
        57,
    ]
    assert receipt["conclusions"]["standalone_packer_implemented"] is True
    assert receipt["conclusions"][
        "actual_nvdsinfer_bridge_implemented"
    ] is False
    assert receipt["conclusions"]["deepstream9_parity_passed"] is False
    assert receipt["conclusions"]["production_ready"] is False
    assert receipt["production_baseline"] == {
        "known_hashes": adapter.EXPECTED_PRODUCTION_HASHES,
        "source_tree_stable_during_build": True,
        "production_cmake_modified_by_lane": False,
        "production_runtime_modified_by_lane": False,
    }


def test_adapter_receipt_replays_exact_build_and_live_pins(
    receipt: dict,
) -> None:
    result = adapter.verify_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert result == {
        "valid": True,
        "candidate_id": "mmpose-yoloxpose-s",
        "status": (
            "standalone_cpu_packer_verified_deepstream9_bridge_pending"
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "unit_case_count": 12,
        "standalone_packer_implemented": True,
        "actual_nvdsinfer_bridge_implemented": False,
        "deepstream9_executed": False,
        "production_ready": False,
    }


def test_adapter_receipt_tamper_is_rejected_before_replay(
    receipt: dict,
) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["conclusions"]["production_ready"] = True
    with pytest.raises(adapter.PoseAdapterError, match="self hash differs"):
        adapter.verify_receipt(
            tampered,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_adapter_receipt_publication_is_read_only_and_no_overwrite(
    tmp_path: Path,
    receipt: dict,
) -> None:
    output = tmp_path / "adapter-receipt.json"
    adapter.atomic_write_no_overwrite(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(adapter.PoseAdapterError, match="refusing to overwrite"):
        adapter.atomic_write_no_overwrite(output, receipt)


def test_production_pose_cmake_and_runtime_hashes_remain_exact() -> None:
    paths = adapter._input_paths()
    for key, expected in adapter.EXPECTED_PRODUCTION_HASHES.items():
        assert adapter._sha256(paths[key]) == expected
