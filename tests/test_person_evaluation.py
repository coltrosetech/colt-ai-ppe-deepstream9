import json
from pathlib import Path

import pytest

from evaluation.cli import main
from evaluation.metrics import evaluate_person_predictions
from evaluation.readers import load_caviar, load_coco, load_mot, load_predictions_jsonl


FIXTURES = Path(__file__).parent / "fixtures" / "evaluation"


def test_coco_person_metrics_ap_sizes_and_sequences():
    ground_truth = load_coco(FIXTURES / "coco.json")
    predictions = load_predictions_jsonl(FIXTURES / "coco-predictions.jsonl")
    result = evaluate_person_predictions(ground_truth, predictions)

    assert result["overall"] == {
        "ground_truth": 3,
        "serialized_predictions_at_or_above_confidence": 4,
        "evaluated_predictions": 3,
        "ignored_predictions": 1,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": 0.666667,
        "recall": 0.666667,
        "f1": 0.666667,
        "ap_101_point": 0.663366,
        "ap_serialized_predictions": 4,
    }
    assert result["size_buckets"]["small"]["tp"] == 1
    assert result["size_buckets"]["medium"]["tp"] == 1
    assert result["size_buckets"]["large"]["fn"] == 1
    assert set(result["sequences"]) == {"seq-a", "seq-b"}
    assert result["sequences"]["seq-b"]["overall"]["recall"] == 0.0
    assert result["predictions"]["ignored_non_person_detections"] == 1
    assert result["ground_truth"]["metadata"]["ignored_non_person_annotations"] == 1


def test_mot_person_filter_and_ignore_regions():
    ground_truth = load_mot(FIXTURES / "mot-seq")
    predictions = load_predictions_jsonl(FIXTURES / "mot-predictions.jsonl")
    result = evaluate_person_predictions(ground_truth, predictions)

    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["overall"]["ignored_predictions"] == 2
    assert result["ground_truth"]["metadata"]["ignored_unmarked_or_non_person_rows"] == 2


def test_caviar_individuals_are_positive_and_groups_are_excluded():
    ground_truth = load_caviar(
        FIXTURES / "caviar.xml", image_width=100, image_height=100
    )
    predictions = load_predictions_jsonl(FIXTURES / "caviar-predictions.jsonl")
    result = evaluate_person_predictions(ground_truth, predictions)

    # The inactive individual is still a positive. The larger social group box
    # is not a person GT or ignore region, so a group-shaped prediction is an FP.
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 1
    assert result["overall"]["fn"] == 1
    assert result["ground_truth"]["metadata"]["ignored_group_boxes"] == 1
    policy = result["ground_truth"]["metadata"]["annotation_policy"]
    assert "including inactive/static" in policy["individual_objects"]
    assert "neither positives nor false-positive suppression" in policy["group_objects"]
    assert "may count as FP" in policy["never_moving_people"]


def test_prediction_schema_rejects_out_of_bounds_box(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "seq",
                "frame_index": 0,
                "image_width": 100,
                "image_height": 100,
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.5,
                        "bbox_norm_xywh": [0.9, 0.1, 0.2, 0.2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        load_predictions_jsonl(path)


def test_prediction_schema_accepts_valid_nvdcf_confidence_fields(tmp_path):
    path = tmp_path / "tracked.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "fence-f01",
                "frame_index": 0,
                "image_width": 1920,
                "image_height": 1080,
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.91,
                        "tracker_confidence": 0.91,
                        "tracker_confidence_raw": 0.910001,
                        "track_id": 7,
                        "bbox_norm_xywh": [0.2, 0.1, 0.2, 0.7],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_predictions_jsonl(path)

    assert len(loaded.detections) == 1
    assert loaded.detections[0].track_id == "7"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tracker_confidence", 1.1, "tracker_confidence must be in"),
        ("tracker_confidence_raw", -0.1, "tracker_confidence_raw must be"),
    ],
)
def test_prediction_schema_rejects_invalid_nvdcf_confidence_fields(
    tmp_path,
    field,
    value,
    message,
):
    path = tmp_path / f"bad-{field}.jsonl"
    detection = {
        "class_name": "person",
        "confidence": 0.5,
        "track_id": 3,
        "bbox_norm_xywh": [0.2, 0.1, 0.2, 0.7],
        field: value,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "fence-f01",
                "frame_index": 0,
                "image_width": 1920,
                "image_height": 1080,
                "detections": [detection],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_predictions_jsonl(path)


def test_cli_writes_machine_readable_result(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--predictions",
                str(FIXTURES / "coco-predictions.jsonl"),
                "--ground-truth",
                str(FIXTURES / "coco.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads(output.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["schema_version"] == "deepsafe.person-evaluation/v1"
