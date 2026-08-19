from __future__ import annotations

import copy
import json
import stat

import numpy as np
import pytest

from validation.person_rtdetrv4_real_image_parity import (
    PLAN,
    ROOT,
    RealImageParityError,
    _bounded_bijection,
    _load_plan,
    receipt_sha256,
    verify_receipt,
)


RECEIPT = (
    ROOT
    / "validation/results/person/models/rtdetrv4-s-real-image-parity-r1.json"
)
PIN = "563d940aa0564961b7f3e79df50053ab894928ab3ba3cc8fc9c1c5ba3e8cd938"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = receipt_sha256(value)


def test_checked_in_failure_receipt_is_exact_pinned_and_fail_closed() -> None:
    result = verify_receipt(_load(), expected_receipt_sha256=PIN)

    assert result["valid"] is True
    assert result["profiles"] == [640, 960]
    assert result["batches"] == [1, 2]
    assert result["selected_frame_count"] == 11
    assert result["unique_primary_video_type_count"] == 11
    assert result["failure_count"] == 4
    assert result["real_image_framework_onnx_parity_passed"] is False
    assert result["quality_passed"] is False
    assert result["performance_passed"] is False
    assert result["production_ready"] is False


def test_plan_keeps_all_representative_visible_scenes_without_cherry_pick() -> None:
    plan = _load_plan()

    assert len(plan["selections"]) == 11
    assert len({item["scene_id"] for item in plan["selections"]}) == 11
    assert len({item["primary_video_type"] for item in plan["selections"]}) == 11
    assert all(
        item["segment_role"] in {"person_visible", "partial_body_only"}
        for item in plan["selections"]
    )
    assert any(item["review_flags"]["medium_close"] for item in plan["selections"])
    assert any(item["review_flags"]["high_oblique"] for item in plan["selections"])
    assert any(item["review_flags"]["top_view"] for item in plan["selections"])
    assert PLAN.is_file()


def test_class_preserving_bijection_accepts_topk_order_permutation_only() -> None:
    labels = (np.arange(300, dtype=np.int64) % 80).astype(np.int64)
    boxes = np.stack(
        [
            np.arange(300, dtype=np.float32),
            np.arange(300, dtype=np.float32) + 1,
            np.arange(300, dtype=np.float32) + 2,
            np.arange(300, dtype=np.float32) + 3,
        ],
        axis=1,
    )
    scores = np.linspace(0.9, 0.1, 300, dtype=np.float32)
    permutation = np.arange(299, -1, -1)

    result = _bounded_bijection(
        labels,
        boxes,
        scores,
        labels[permutation],
        boxes[permutation],
        scores[permutation],
    )

    assert result["passed"] is True
    assert result["matched_pair_count"] == 300
    assert result["positional_diagnostics_not_acceptance"][
        "label_mismatch_count"
    ] > 0


def test_box_tolerance_is_not_relaxed_for_same_topk_set() -> None:
    labels = np.zeros(300, dtype=np.int64)
    boxes = np.zeros((300, 4), dtype=np.float32)
    boxes[:, 0] = np.arange(300, dtype=np.float32)
    scores = np.linspace(0.9, 0.1, 300, dtype=np.float32)
    runtime_boxes = boxes.copy()
    runtime_boxes[10, 1] += np.float32(0.01001)

    result = _bounded_bijection(
        labels, boxes, scores, labels, runtime_boxes, scores
    )

    assert result["passed"] is False
    assert result["matched_pair_count"] < 300


def test_receipt_byte_tamper_is_rejected() -> None:
    value = _load()
    value["frames"][0]["frame"]["index"] += 1

    with pytest.raises(RealImageParityError, match="self-hash"):
        verify_receipt(value, expected_receipt_sha256=PIN)


def test_resealed_quality_overclaim_is_rejected() -> None:
    value = copy.deepcopy(_load())
    value["acceptance"]["independent_ground_truth_quality_passed"] = True
    _reseal(value)

    with pytest.raises(RealImageParityError, match="overclaims"):
        verify_receipt(
            value, expected_receipt_sha256=value["receipt_sha256"]
        )


def test_resealed_failed_receipt_cannot_claim_pass_status() -> None:
    value = copy.deepcopy(_load())
    value["status"] = (
        "real_image_framework_onnx_parity_passed_not_quality_not_performance"
    )
    value["acceptance"]["real_image_framework_onnx_parity_passed"] = True
    _reseal(value)

    with pytest.raises(RealImageParityError, match="status/outcome"):
        verify_receipt(
            value, expected_receipt_sha256=value["receipt_sha256"]
        )


def test_receipt_is_immutable_and_tie_diagnostics_do_not_override_failure() -> None:
    value = _load()

    assert stat.S_IMODE(RECEIPT.stat().st_mode) == 0o440
    assert value["outcome"]["tolerances_relaxed"] is False
    assert value["outcome"]["topk_tie_diagnostics_override_acceptance"] is False
    tie_failure = next(
        item
        for item in value["outcome"]["failures"]
        if item["topk_selection_difference_within_strict_score_tie_band"]
    )
    assert tie_failure["case_id"] == "us_inmate_full_body_walk.f000347"
    assert value["acceptance"][
        "real_image_framework_onnx_parity_passed"
    ] is False
