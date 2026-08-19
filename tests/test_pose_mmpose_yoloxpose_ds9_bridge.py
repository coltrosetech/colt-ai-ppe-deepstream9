from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import jsonschema
import pytest

from validation import pose_mmpose_yoloxpose_ds9_bridge as bridge


CREATED_AT = "2026-07-18T04:15:00+03:00"


@pytest.fixture(scope="module")
def receipt() -> dict:
    return bridge.build_receipt(created_at=CREATED_AT)


def test_ds9_bridge_receipt_matches_closed_schema(receipt: dict) -> None:
    schema = json.loads(bridge.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(receipt)


def test_ds9_bridge_receipt_states_exact_verified_and_pending_boundaries(
    receipt: dict,
) -> None:
    assert receipt["status"] == (
        "real_ds9_header_cpu_abi_verified_runtime_tensor_parity_pending"
    )
    assert receipt["deepstream9_sdk"]["image_id"] == bridge.IMAGE_ID
    assert receipt["toolchain_and_tests"]["unit_case_count"] == 12
    assert receipt["verified_contract"] == {
        "tensor_meta_scope": "one_frame_or_object",
        "per_meta_batch": 1,
        "layers_in_order": ["dets", "keypoints"],
        "frame_shapes_without_batch": [[100, 5], [100, 17, 3]],
        "canonical_frame_shape": [1, 300, 57],
        "ordered_frame_meta_batch_range": [1, 12],
        "canonical_twelve_frame_shape": [12, 300, 57],
        "metadata_batch_axis_claimed": False,
        "float_640_b1_verified": True,
        "half_960_b1_verified": True,
        "ordered_frame_meta_b12_verified": True,
        "partial_output_on_failure": False,
    }
    assert receipt["conclusions"][
        "actual_nvdsinfer_tensor_meta_bridge_implemented"
    ] is True
    assert receipt["conclusions"][
        "live_gst_nvinfer_tensor_meta_observed"
    ] is False
    assert receipt["conclusions"][
        "deepstream9_runtime_tensor_parity_passed"
    ] is False
    assert receipt["conclusions"]["production_ready"] is False


def test_ds9_bridge_receipt_replays_real_header_build(receipt: dict) -> None:
    result = bridge.verify_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert result["valid"] is True
    assert result["unit_case_count"] == 12
    assert result[
        "actual_nvdsinfer_tensor_meta_bridge_implemented"
    ] is True
    assert result["live_gst_nvinfer_tensor_meta_observed"] is False
    assert result["production_ready"] is False


def test_ds9_bridge_receipt_tamper_fails_before_rebuild(receipt: dict) -> None:
    tampered = copy.deepcopy(receipt)
    tampered["conclusions"]["production_ready"] = True
    with pytest.raises(bridge.PoseBridgeError, match="self hash differs"):
        bridge.verify_receipt(
            tampered,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_ds9_bridge_publication_is_immutable_no_overwrite(
    tmp_path: Path,
    receipt: dict,
) -> None:
    output = tmp_path / "bridge.json"
    bridge.atomic_write_no_overwrite(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(bridge.PoseBridgeError, match="refusing to overwrite"):
        bridge.atomic_write_no_overwrite(output, receipt)


def test_adapter_r1_and_production_hashes_remain_unchanged() -> None:
    adapter_r1 = bridge._read_json(bridge.ADAPTER_R1)
    assert adapter_r1["receipt_sha256"] == bridge.ADAPTER_R1_SELF_SHA256
    paths = bridge._input_paths()
    for key, expected in bridge.PRODUCTION_HASHES.items():
        assert bridge._sha256(paths[key]) == expected
