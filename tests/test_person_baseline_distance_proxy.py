from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validation import person_baseline_distance_proxy as proxy


@pytest.fixture(scope="module")
def validated() -> proxy.ValidatedInputs:
    return proxy.validate_inputs()


@pytest.fixture(scope="module")
def report(validated: proxy.ValidatedInputs) -> dict[str, object]:
    value = proxy.build_report(validated)
    proxy.validate_report(value)
    return value


def _gt(annotation_id: int, bbox: tuple[float, float, float, float]) -> proxy.GroundTruthRecord:
    return proxy.GroundTruthRecord(annotation_id, 1, bbox)


def _prediction(
    row_index: int,
    score: float,
    bbox: tuple[float, float, float, float],
) -> proxy.PredictionRecord:
    return proxy.PredictionRecord(row_index, 1, score, bbox)


def test_real_inputs_are_exact_group_safe_validation_only(
    validated: proxy.ValidatedInputs,
) -> None:
    assert len(validated.images) == 384
    assert sum(len(items) for items in validated.ground_truth.values()) == 3256
    assert sum(len(items) for items in validated.predictions.values()) == 115200
    assert len({item.sequence_id for item in validated.images.values()}) == 32
    assert len({item.capture_group_id for item in validated.images.values()}) == 26
    assert len({item.scene_proxy_id for item in validated.images.values()}) == 11
    assert sum(not items for items in validated.ground_truth.values()) == 8
    container = validated.payloads["container_receipt"]
    assert container["official_test_opened"] is False
    assert container["test_unseen_opened"] is False
    assert container["execution"]["scope"] == proxy.EXPECTED_SCOPE


def test_real_lineage_pins_plan_image_receipts_and_detections(
    validated: proxy.ValidatedInputs,
) -> None:
    assert {
        name: pin["sha256"] for name, pin in validated.pins.items()
    } == proxy.PINNED_SHA256
    assert validated.payloads["execution_plan"]["fingerprint_sha256"] == proxy.PLAN_FINGERPRINT
    assert validated.payloads["host_receipt"]["resolved_image"]["id"] == proxy.IMAGE_ID
    assert validated.payloads["build_receipt"]["child_image"]["id"] == proxy.IMAGE_ID


def test_greedy_matching_is_score_descending_one_to_one_and_stable() -> None:
    ground_truth = (_gt(1, (0, 0, 20, 20)), _gt(2, (100, 100, 20, 20)))
    predictions = (
        _prediction(9, 0.90, (0, 0, 20, 20)),
        _prediction(3, 0.90, (0, 0, 20, 20)),
        _prediction(1, 0.80, (100, 100, 20, 20)),
    )
    outcome = proxy.match_frame(ground_truth, predictions, 0.25)
    assert outcome.matched == ((1, 3, 1.0), (2, 1, 1.0))
    assert outcome.unmatched_prediction_indices == (9,)
    assert outcome.selected_prediction_indices == (3, 9, 1)


def test_score_threshold_is_inclusive() -> None:
    ground_truth = (_gt(1, (0, 0, 20, 20)),)
    predictions = (_prediction(0, 0.25, (0, 0, 20, 20)),)
    outcome = proxy.match_frame(ground_truth, predictions, 0.25)
    assert len(outcome.matched) == 1
    assert outcome.unmatched_prediction_indices == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "le_16_px"),
        (16, "le_16_px"),
        (16.00001, "17_24_px"),
        (24, "17_24_px"),
        (24.00001, "25_32_px"),
        (32, "25_32_px"),
        (32.00001, "33_48_px"),
        (48, "33_48_px"),
        (48.00001, "49_96_px"),
        (96, "49_96_px"),
        (96.00001, "gt_96_px"),
    ],
)
def test_height_layer_boundaries(value: float, expected: str) -> None:
    assert proxy._layer_id(value, proxy.HEIGHT_LAYERS) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "small_lt_32_sq"),
        (1023.999, "small_lt_32_sq"),
        (1024, "medium_32_to_96_sq"),
        (9215.999, "medium_32_to_96_sq"),
        (9216, "large_ge_96_sq"),
    ],
)
def test_coco_area_layer_boundaries(value: float, expected: str) -> None:
    assert proxy._layer_id(value, proxy.COCO_AREA_LAYERS) == expected


def test_real_threshold_replay_matches_pinned_operating_point_and_closed_grid(
    report: dict[str, object],
) -> None:
    metrics = report["overall_threshold_metrics"]
    assert metrics == {
        "0.05": {
            "ground_truth": 3256,
            "selected_predictions": 23193,
            "tp": 2212,
            "fp": 20981,
            "fn": 1044,
            "precision": 0.095373604,
            "recall": 0.679361179,
            "f1": 0.167265303,
        },
        "0.10": {
            "ground_truth": 3256,
            "selected_predictions": 9720,
            "tp": 2033,
            "fp": 7687,
            "fn": 1223,
            "precision": 0.209156379,
            "recall": 0.624385749,
            "f1": 0.313347719,
        },
        "0.25": {
            "ground_truth": 3256,
            "selected_predictions": 2604,
            "tp": 1502,
            "fp": 1102,
            "fn": 1754,
            "precision": 0.576804916,
            "recall": 0.461302211,
            "f1": 0.512627986,
        },
        "0.50": {
            "ground_truth": 3256,
            "selected_predictions": 930,
            "tp": 850,
            "fp": 80,
            "fn": 2406,
            "precision": 0.913978495,
            "recall": 0.261056511,
            "f1": 0.406115624,
        },
    }


@pytest.mark.parametrize("family", ["height_px", "area_px2"])
@pytest.mark.parametrize("threshold", ["0.05", "0.10", "0.25", "0.50"])
def test_size_layers_conserve_all_counts(
    report: dict[str, object], family: str, threshold: str
) -> None:
    overall = report["overall_threshold_metrics"][threshold]
    layers = report["gt_bbox_size_layers"][family][threshold]
    for field in ("ground_truth", "selected_predictions", "tp", "fp", "fn"):
        assert sum(item["metrics"][field] for item in layers) == overall[field]


def test_small_apparent_scale_proxy_exposes_material_recall_gap(
    report: dict[str, object],
) -> None:
    height_layers = {
        item["layer_id"]: item["metrics"]
        for item in report["gt_bbox_size_layers"]["height_px"]["0.25"]
    }
    assert height_layers["le_16_px"]["ground_truth"] == 185
    assert height_layers["le_16_px"]["tp"] == 2
    assert height_layers["17_24_px"]["ground_truth"] == 405
    assert height_layers["17_24_px"]["tp"] == 20
    assert height_layers["gt_96_px"]["ground_truth"] == 413
    assert height_layers["gt_96_px"]["tp"] == 377


@pytest.mark.parametrize(
    ("breakdown_name", "expected_count"),
    [
        ("source_sequences", 32),
        ("capture_groups", 26),
        ("daytime_location_metadata_strata", 11),
    ],
)
def test_group_breakdowns_partition_overall_counts(
    report: dict[str, object], breakdown_name: str, expected_count: int
) -> None:
    groups = report["breakdowns"][breakdown_name]
    assert len(groups) == expected_count
    for threshold in ("0.05", "0.10", "0.25", "0.50"):
        overall = report["overall_threshold_metrics"][threshold]
        for field in ("ground_truth", "selected_predictions", "tp", "fp", "fn"):
            assert sum(group["threshold_metrics"][threshold][field] for group in groups) == overall[field]


def test_worst_group_ranking_is_deterministic(report: dict[str, object]) -> None:
    worst = report["worst_groups_at_score_0_25"]["source_sequences"]
    assert [item["rank"] for item in worst] == list(range(1, 11))
    ranking_keys = [
        (
            item["metrics"]["f1"],
            item["metrics"]["recall"],
            item["metrics"]["precision"],
            item["group_id"],
        )
        for item in worst
    ]
    assert ranking_keys == sorted(ranking_keys)
    assert worst[0]["group_id"] == "038"


def test_missed_person_list_is_bounded_and_smallest_first(
    report: dict[str, object],
) -> None:
    missed = report["missed_person_examples"]
    assert missed["score_threshold"] == 0.25
    assert missed["total_missed_ground_truth"] == 1754
    assert missed["listed_examples"] == 128
    assert missed["truncated"] is True
    ordering = [
        (
            item["gt_height_px"],
            item["gt_area_px2"],
            item["image_id"],
            item["annotation_id"],
        )
        for item in missed["examples"]
    ]
    assert ordering == sorted(ordering)


def test_report_never_promotes_pixel_scale_to_metric_distance(
    report: dict[str, object],
) -> None:
    interpretation = report["interpretation"]
    assert interpretation["bbox_pixel_size_is_apparent_scale_proxy_only"] is True
    assert interpretation["pixel_size_is_metric_distance"] is False
    assert interpretation["detection_at_20m_established"] is False
    assert interpretation["detection_at_25m_established"] is False
    assert interpretation["scene_metadata_strata_are_semantic_scene_types"] is False
    requirements = report["metric_20_25m_calibration_requirements"]
    assert requirements["proxy_can_establish_20m_or_25m"] is False
    assert requirements["required_acceptance_dimensions"]["distances_m"] == [20.0, 25.0]
    assert "calibration_artifact_sha256" in requirements["admin_acceptance_fields"]


def test_report_fingerprint_and_schema_are_deterministic(
    validated: proxy.ValidatedInputs, report: dict[str, object]
) -> None:
    rebuilt = proxy.build_report(validated)
    assert proxy._canonical_bytes(rebuilt) == proxy._canonical_bytes(report)
    assert report["fingerprint_sha256"] == proxy.fingerprint(report)
    proxy._schema_validate(report)


def test_schema_rejects_metric_distance_overclaim(report: dict[str, object]) -> None:
    tampered = copy.deepcopy(report)
    tampered["interpretation"]["detection_at_25m_established"] = True
    tampered["fingerprint_sha256"] = proxy.fingerprint(tampered)
    with pytest.raises(proxy.DistanceProxyError, match="schema validation failed"):
        proxy.validate_report(tampered)


def test_lineage_validation_fails_closed_on_image_or_scope_tamper(
    validated: proxy.ValidatedInputs,
) -> None:
    image_tamper = copy.deepcopy(validated.payloads)
    image_tamper["host_receipt"]["resolved_image"]["id"] = "sha256:" + "0" * 64
    with pytest.raises(proxy.DistanceProxyError, match="host image id differs"):
        proxy._validate_lineage(image_tamper)

    scope_tamper = copy.deepcopy(validated.payloads)
    scope_tamper["container_receipt"]["execution"]["scope"] = "official_test"
    with pytest.raises(proxy.DistanceProxyError, match="scope differs"):
        proxy._validate_lineage(scope_tamper)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(proxy.DistanceProxyError, match="duplicate JSON key"):
        proxy.strict_load_json(duplicate, "duplicate")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(proxy.DistanceProxyError, match="non-finite JSON constant"):
        proxy.strict_load_json(nonfinite, "nonfinite")


def test_closed_input_path_set_rejects_missing_or_extra_paths() -> None:
    missing = dict(proxy.DEFAULT_PATHS)
    missing.pop("detections")
    with pytest.raises(proxy.DistanceProxyError, match="input path set differs"):
        proxy.validate_inputs(missing)


def test_published_artifacts_verify_if_present() -> None:
    if not proxy.DEFAULT_OUTPUT_DIR.exists():
        pytest.skip("immutable report is generated after unit construction")
    summary = proxy.verify_artifacts()
    assert summary["status"] == "passed"
    assert summary["metric_distance_status"] == "blocked_missing_per_camera_metric_calibration"
