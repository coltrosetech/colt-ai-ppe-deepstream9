from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_threshold_calibration_r13 as lane


class _FakePillowImage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def resize(self, size: tuple[int, int], *, resample: object) -> "_FakePillowImage":
        assert resample == "bilinear"
        width, height = size
        # The contract test uses a constant-white image, so exact resampling is
        # irrelevant while geometry/padding/layout remain fully observable.
        return _FakePillowImage(
            np.full((height, width, 3), self.array[0, 0], dtype=np.uint8)
        )

    def paste(self, other: "_FakePillowImage", offset: tuple[int, int]) -> None:
        left, top = offset
        height, width = other.array.shape[:2]
        self.array[top : top + height, left : left + width] = other.array

    def __array__(self, dtype: object = None) -> np.ndarray:
        return np.asarray(self.array, dtype=dtype)


class _FakeImageModule:
    class Resampling:
        BILINEAR = "bilinear"

    @staticmethod
    def fromarray(array: np.ndarray, mode: str) -> _FakePillowImage:
        assert mode == "RGB"
        return _FakePillowImage(array.copy())

    @staticmethod
    def new(mode: str, size: tuple[int, int], color: tuple[int, int, int]) -> _FakePillowImage:
        assert mode == "RGB"
        width, height = size
        array = np.empty((height, width, 3), dtype=np.uint8)
        array[:] = color
        return _FakePillowImage(array)


def event(
    score: float,
    true_positive: bool,
    *,
    image_id: int = 1,
    group: str = "group-a",
    annotation_id: int | None = None,
) -> lane.DetectionEvent:
    return lane.DetectionEvent(
        score=score,
        true_positive=true_positive,
        image_id=image_id,
        capture_group_id=group,
        matched_annotation_id=annotation_id,
    )


def test_threshold_inclusion_is_score_greater_than_or_equal() -> None:
    points, selected = lane.build_sweep(
        ground_truth=1,
        events=[event(0.5, True, annotation_id=10), event(0.25, False)],
        finite_scores=[0.5, 0.25],
    )
    by_threshold = {point["threshold"]: point["metrics"] for point in points}
    assert by_threshold[0.5] == {
        "ground_truth": 1,
        "selected_predictions": 1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert by_threshold[1.0]["selected_predictions"] == 0
    assert selected["threshold"] == 0.5


def test_all_unique_finite_scores_plus_zero_and_one_are_canonical_pinned() -> None:
    score = float(np.float32(0.30000001192092896))
    points, _selected = lane.build_sweep(
        ground_truth=1,
        events=[event(score, True, annotation_id=1)],
        finite_scores=[score, score, 0.0, 1.0],
    )
    assert [point["threshold"] for point in points] == [0.0, score, 1.0]
    digest = hashlib.sha256(lane.canonical_bytes(points)).hexdigest()
    round_tripped = json.loads(lane.canonical_bytes(points))
    assert hashlib.sha256(lane.canonical_bytes(round_tripped)).hexdigest() == digest


def test_max_f1_tie_break_prefers_higher_recall_then_lower_threshold() -> None:
    # At 0.8: TP=1 FP=0 FN=1 -> F1=2/3.  At 0.4: TP=2 FP=2 FN=0
    # -> F1=2/3 but recall is higher, so 0.4 wins the exact rational tie.
    points, selected = lane.build_sweep(
        ground_truth=2,
        events=[
            event(0.8, True, annotation_id=1),
            event(0.4, True, annotation_id=2),
            event(0.4, False),
            event(0.4, False),
        ],
        finite_scores=[0.8, 0.4],
    )
    # 0.0 has the same selected detections as 0.4 in this synthetic case, so
    # the final lower-threshold rule deterministically carries the winner to 0.
    assert selected["threshold"] == 0.0
    assert selected["metrics"]["recall"] == 1.0
    lane.validate_sweep_points(points, selected, expected_ground_truth=2)


def test_final_tie_break_prefers_lower_threshold() -> None:
    # The score at 0.9 is an invalid-parser candidate and therefore has no
    # event.  Thresholds 0.0, 0.8 and 0.9 have identical exact counts; the
    # contract's final lower-threshold rule therefore selects the 0 endpoint.
    _points, selected = lane.build_sweep(
        ground_truth=1,
        events=[event(0.8, True, annotation_id=1)],
        finite_scores=[0.9, 0.8],
    )
    assert selected["threshold"] == 0.0


def test_nonfinite_or_out_of_range_candidate_fails_closed() -> None:
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="out of range"):
        lane.build_sweep(
            ground_truth=1,
            events=[],
            finite_scores=[1.1],
        )
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="out of range"):
        lane.build_sweep(
            ground_truth=1,
            events=[],
            finite_scores=[float("nan")],
        )


def test_matching_is_score_descending_query_stable_and_one_to_one() -> None:
    ground_truth = [
        {"annotation_id": 10, "box": (0.0, 0.0, 10.0, 10.0)},
        {"annotation_id": 11, "box": (20.0, 0.0, 30.0, 10.0)},
    ]
    detections = [
        {"query_index": 9, "score": 0.7, "box": (0.0, 0.0, 10.0, 10.0)},
        {"query_index": 2, "score": 0.9, "box": (0.0, 0.0, 10.0, 10.0)},
        {"query_index": 3, "score": 0.8, "box": (20.0, 0.0, 30.0, 10.0)},
    ]
    events = lane.match_detection_events(
        image_id=1,
        capture_group_id="g",
        ground_truth=ground_truth,
        detections=detections,
    )
    assert [(item.score, item.true_positive, item.matched_annotation_id) for item in events] == [
        (0.9, True, 10),
        (0.8, True, 11),
        (0.7, False, None),
    ]


def test_letterbox_matches_deepstream_symmetric_black_contract() -> None:
    rgb = np.full((720, 1280, 3), 255, dtype=np.uint8)
    tensor, transform = lane._letterbox(rgb, 640, np, _FakeImageModule)
    assert transform == {
        "ratio": 0.5,
        "resized_width": 640,
        "resized_height": 360,
        "pad_left": 0,
        "pad_top": 140,
    }
    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == np.float32
    assert np.all(tensor[:, :140, :] == 0.0)
    assert np.all(tensor[:, 140:500, :] == 1.0)
    assert np.all(tensor[:, 500:, :] == 0.0)


def test_gt_uses_the_same_letterbox_transform() -> None:
    transformed = lane._gt_profile_boxes(
        [{"id": 4, "bbox": [100.0, 20.0, 40.0, 60.0]}],
        {"ratio": 0.5, "pad_left": 0, "pad_top": 140},
    )
    assert transformed == [
        {
            "annotation_id": 4,
            "original_height": 60.0,
            "box": (50.0, 150.0, 70.0, 180.0),
        }
    ]


def test_capture_group_report_micro_totals_match_selected() -> None:
    images = [
        {"id": 1, "deepsafe_capture_group_id": "a"},
        {"id": 2, "deepsafe_capture_group_id": "b"},
    ]
    annotations = {1: [{"id": 1}], 2: [{"id": 2}]}
    events = [
        event(0.8, True, image_id=1, group="a", annotation_id=1),
        event(0.7, False, image_id=1, group="a"),
        event(0.6, True, image_id=2, group="b", annotation_id=2),
    ]
    selected = {"metrics": lane.metrics_from_counts(2, 2, 1)}
    report = lane._selected_group_report(
        threshold=0.5,
        images=images,
        annotations=annotations,
        events=events,
        selected=selected,
    )
    assert report["micro_totals_match_selected"] is True
    assert [row["capture_group_id"] for row in report["groups"]] == ["a", "b"]


def test_apparent_scale_is_explicitly_non_metric_and_covers_height_bins() -> None:
    heights = [10.0, 20.0, 30.0, 40.0, 60.0, 120.0]
    annotations = {
        1: [
            {"id": index + 1, "bbox": [0.0, 0.0, 10.0, height]}
            for index, height in enumerate(heights)
        ]
    }
    events = [
        event(0.9, True, annotation_id=1),
        event(0.9, True, annotation_id=6),
        event(0.9, False),
    ]
    report = lane._apparent_scale_report(
        threshold=0.5,
        annotations=annotations,
        events=events,
    )
    assert report["metric_distance_claim"] is False
    assert [row["ground_truth"] for row in report["bins"]] == [1, 1, 1, 1, 1, 1]
    assert [row["tp"] for row in report["bins"]] == [1, 0, 0, 0, 0, 1]
    assert report["unmatched_false_positives"] == 1


def test_actual_validation_coco_is_only_development_validation() -> None:
    raw = lane.strict_json_bytes(
        (lane.ROOT / lane.COCO_RELATIVE).read_bytes(),
        source=lane.COCO_RELATIVE,
    )
    images, annotations = lane._validate_dataset(raw)
    assert len(images) == 384
    assert sum(len(rows) for rows in annotations.values()) == 3256
    assert raw["info"]["deepsafe_role"] == "calibration_validation_not_official_test"


def test_actual_r12_receipts_form_the_required_two_stage_chain() -> None:
    for profile in lane.PROFILES:
        value = lane.strict_json_bytes(
            (lane.ROOT / lane.ONNX_RECEIPT_RELATIVE[profile]).read_bytes(),
            source=lane.ONNX_RECEIPT_RELATIVE[profile],
        )
        lane._validate_r12_receipt(value, profile=profile)


def test_worker_command_never_contains_or_inherits_checkpoint() -> None:
    accepted = SimpleNamespace(
        descriptors={
            "runtime_python": 30,
            "executor": 31,
            "plan": 32,
            "r11_plan": 33,
            "coco": 34,
            "onnx_640": 35,
        },
        plan={"fingerprint_sha256": "a" * 64},
        executor_sha256="b" * 64,
    )
    command, pass_fds = lane._worker_command(accepted, profile=640)
    assert not any("checkpoint" in item.lower() for item in command)
    assert pass_fds == (30, 31, 32, 33, 34, 35)


def test_internal_worker_is_not_a_public_entrypoint() -> None:
    args = SimpleNamespace()
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="sandbox-only"):
        lane.internal_worker(args)


def test_sweep_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads((lane.ROOT / lane.SWEEP_SCHEMA_RELATIVE).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_fingerprint_is_fail_closed_and_rejects_nonfinite() -> None:
    value = {"status": "planned", "fingerprint_sha256": "ignored"}
    first = lane.fingerprint(value)
    changed = copy.deepcopy(value)
    changed["status"] = "different"
    assert lane.fingerprint(changed) != first
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="non-finite"):
        lane.fingerprint({"value": float("inf")})


def test_metric_cross_fields_cannot_be_resealed_inconsistently() -> None:
    metrics = lane.metrics_from_counts(10, 4, 2)
    lane._metric_invariants(metrics)
    bad = dict(metrics)
    bad["f1"] = 0.999
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="count-derived"):
        lane._metric_invariants(bad)


def test_output_paths_are_profile_separate_then_one_r11_final() -> None:
    assert lane.SWEEP_RECEIPT_RELATIVE[640] != lane.SWEEP_RECEIPT_RELATIVE[960]
    assert lane.FINAL_RECEIPT_RELATIVE.endswith("threshold-calibration/receipt.json")
    assert "test" not in lane.FINAL_RECEIPT_RELATIVE


def test_profile_execution_order_requires_external_640_acceptance() -> None:
    with pytest.raises(
        lane.ThresholdCalibrationR13Error,
        match="externally accepted 640",
    ):
        lane.execute_profile(
            SimpleNamespace(),
            profile=960,
        )
    with pytest.raises(
        lane.ThresholdCalibrationR13Error,
        match="must not receive",
    ):
        lane.execute_profile(
            SimpleNamespace(),
            profile=640,
            prior_sweep_640_path=Path("unexpected.json"),
            accepted_prior_sweep_640_fingerprint="a" * 64,
        )


def test_actual_r13_plan_canonically_pins_the_reviewed_executor() -> None:
    plan_path = lane.ROOT / lane.PLAN_RELATIVE
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    executor_raw = (lane.ROOT / lane.EXECUTOR_RELATIVE).read_bytes()
    executor_pin = {
        "path": lane.EXECUTOR_RELATIVE,
        "bytes": len(executor_raw),
        "sha256": hashlib.sha256(executor_raw).hexdigest(),
    }
    lane._validate_plan(plan, executor_pin=executor_pin)
    assert plan["fingerprint_sha256"] == lane.fingerprint(plan)
    assert plan["execution"] == {
        "performed_during_preparation": False,
        "inference": False,
        "gpu": False,
        "checkpoint_opened": False,
        "profile_receipts_published": [],
        "final_receipt_published": False,
    }


def _fake_accepted() -> SimpleNamespace:
    sweep_schema = json.loads(
        (lane.ROOT / lane.SWEEP_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    r11_schema = json.loads(
        (lane.ROOT / lane.R11_EVIDENCE_SCHEMA_RELATIVE).read_text(encoding="utf-8")
    )
    return SimpleNamespace(
        plan_pin={
            "path": lane.PLAN_RELATIVE,
            "bytes": 1,
            "sha256": "a" * 64,
            "fingerprint_sha256": "b" * 64,
        },
        r11_plan_pin={
            "path": lane.R11_PLAN_RELATIVE,
            "bytes": lane.R11_PLAN_BYTES,
            "sha256": lane.R11_PLAN_SHA256,
            "fingerprint_sha256": lane.R11_PLAN_FINGERPRINT,
        },
        sweep_schema=sweep_schema,
        r11_schema=r11_schema,
    )


def _zero_detection_worker() -> dict:
    zero = lane.metrics_from_counts(lane.EXPECTED_ANNOTATIONS, 0, 0)
    points = [
        {"threshold": 0.0, "metrics": zero},
        {"threshold": 1.0, "metrics": zero},
    ]
    groups = []
    remaining_gt = lane.EXPECTED_ANNOTATIONS
    for index in range(lane.EXPECTED_CAPTURE_GROUPS):
        group_gt = 126 if index < 6 else 125
        remaining_gt -= group_gt
        groups.append(
            {
                "capture_group_id": f"group-{index:02d}",
                "images": 15 if index < 20 else 14,
                "metrics": lane.metrics_from_counts(group_gt, 0, 0),
            }
        )
    assert remaining_gt == 0
    height_counts = [543, 543, 543, 543, 542, 542]
    proxy_bins = []
    for (name, lower, upper), ground_truth in zip(lane.HEIGHT_BINS, height_counts):
        proxy_bins.append(
            {
                "id": name,
                "lower_exclusive": lower,
                "upper_inclusive": upper,
                "ground_truth": ground_truth,
                "tp": 0,
                "fn": ground_truth,
                "recall": 0.0,
            }
        )
    return {
        "runtime": lane.EXPECTED_RUNTIME,
        "batches_executed": 32,
        "finite_output_score_count": lane.EXPECTED_OUTPUT_SCORES,
        "nonfinite_output_score_count": 0,
        "source_image_manifest_sha256": "c" * 64,
        "points": points,
        "points_sha256": hashlib.sha256(lane.canonical_bytes(points)).hexdigest(),
        "selected": {
            "threshold": 0.0,
            "objective": lane.OBJECTIVE,
            "tie_break": ["higher_exact_recall", "lower_threshold"],
            "metrics": zero,
            "threshold_finite": True,
        },
        "capture_group_report": {
            "threshold": 0.0,
            "groups": groups,
            "group_count": 26,
            "micro_totals_match_selected": True,
        },
        "apparent_scale_proxy": {
            "kind": "fine_tuned_person_original_gt_bbox_height_non_metric_distance_proxy",
            "metric_distance_claim": False,
            "threshold": 0.0,
            "height_source": "original_1280x720_ground_truth_bbox_height_pixels",
            "bins": proxy_bins,
            "unmatched_false_positives": 0,
        },
    }


def test_full_sweep_receipt_matches_custom_schema_and_semantic_replay() -> None:
    accepted = _fake_accepted()
    receipt = lane.build_sweep_receipt(
        accepted,
        profile=640,
        worker=_zero_detection_worker(),
        created_at_utc="2026-07-18T00:00:00+00:00",
    )
    lane._validate_sweep_receipt_semantics(
        receipt,
        accepted=accepted,
        profile=640,
    )
    assert receipt["dataset"]["official_test_opened"] is False
    assert receipt["dataset"]["test_unseen_opened"] is False
    assert receipt["execution"]["checkpoint_opened"] is False
    assert receipt["claim_boundary"]["metric_distance"] is False


def test_resealed_capture_group_micro_total_is_rejected() -> None:
    accepted = _fake_accepted()
    receipt = lane.build_sweep_receipt(
        accepted,
        profile=960,
        worker=_zero_detection_worker(),
        created_at_utc="2026-07-18T00:00:00+00:00",
    )
    receipt["capture_group_report"]["groups"][0]["metrics"] = (
        lane.metrics_from_counts(126, 0, 1)
    )
    receipt["fingerprint_sha256"] = lane.fingerprint(receipt)
    with pytest.raises(lane.ThresholdCalibrationR13Error, match="micro total"):
        lane._validate_sweep_receipt_semantics(
            receipt,
            accepted=accepted,
            profile=960,
        )


def test_final_receipt_uses_exact_r11_threshold_payload_schema() -> None:
    accepted = _fake_accepted()
    worker = _zero_detection_worker()
    sweeps = {}
    for profile in lane.PROFILES:
        receipt = lane.build_sweep_receipt(
            accepted,
            profile=profile,
            worker=worker,
            created_at_utc="2026-07-18T00:00:00+00:00",
        )
        sweeps[profile] = (
            receipt,
            {
                "path": lane.SWEEP_RECEIPT_RELATIVE[profile],
                "bytes": 123,
                "sha256": ("d" if profile == 640 else "e") * 64,
                "fingerprint_sha256": receipt["fingerprint_sha256"],
            },
        )
    final = lane.build_final_receipt(
        accepted,
        sweeps=sweeps,
        created_at_utc="2026-07-18T00:00:01+00:00",
    )
    lane._validate_schema(final, accepted.r11_schema)
    assert final["stage"] == "threshold_calibration"
    assert set(final["payload"]["profiles"]) == {"640", "960"}
    assert final["payload"]["int8_calibration"] is False
    assert final["claim_boundary"]["quality"] is False
