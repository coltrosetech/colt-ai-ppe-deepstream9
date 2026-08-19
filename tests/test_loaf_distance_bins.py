import json
from pathlib import Path

import pytest

from evaluation.metrics import evaluate_person_predictions
from evaluation.readers import load_coco, load_predictions_jsonl
from validation.loaf_distance_band import prepare_distance_band
from validation.loaf_distance_bins import (
    AGGREGATE_SCHEMA_VERSION,
    DISTANCE_BINS,
    METRIC_NAME,
    PROFILE_MERGE_SCHEMA_VERSION,
    PROFILES,
    _dataset_for_targets,
    _load_activations,
    _load_preparation,
    _pin,
    evaluate_prepared_bins,
    prepare_bin_activations,
    write_evaluation_plan,
)


def _base_payload():
    source = {
        "info": {"name": "synthetic LOAF val"},
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {
                "id": index,
                "file_name": f"0001_{index:05d}.jpg",
                "width": 100,
                "height": 100,
            }
            for index in range(5)
        ],
        "annotations": [
            {
                "id": index + 1,
                "image_id": index,
                "category_id": 1,
                "bbox": [10 + index * 10, 20, 8, 12],
                "rotated_box": [14 + index * 10, 26, 8, 12, 0],
                "world_location": [2000 + index * 100, 0],
                "area": 96,
                "iscrowd": 0,
                "ignore": 0,
                "difficult": 1,
            }
            for index in range(5)
        ],
    }
    filtered, report = prepare_distance_band(source, minimum_m=20, maximum_m=25)
    report["source"] = {
        "path": "data/gt/loaf/annotations/resolution_512/instances_val.json",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    return filtered


def _write_base(path: Path) -> None:
    path.write_text(json.dumps(_base_payload()), encoding="utf-8")


def _prediction_record(image, profile, *, include_detection=True):
    frame_id = image["frame_id"]
    x = 10 + frame_id * 10
    detections = []
    if include_detection:
        detections.append(
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox_norm_xywh": [x / 100, 0.2, 0.08, 0.12],
            }
        )
    return {
        "schema_version": "deepsafe.person-detections/v1",
        "sequence_id": image["sequence_id"],
        "frame_index": frame_id,
        "image_width": 100,
        "image_height": 100,
        "model_id": f"toy-{profile}-fp16",
        "detections": detections,
    }


def _write_predictions(path: Path, base_payload, profile: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_prediction_record(image, profile), separators=(",", ":")) + "\n"
            for image in base_payload["images"]
        ),
        encoding="utf-8",
    )


def _write_merge_manifest(
    path: Path,
    predictions: Path,
    preparation: dict,
    profile: int,
    *,
    output_sha256: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = path.parent / "source-run-manifest.json"
    run_manifest.write_text(
        json.dumps({"status": "complete", "model_input": profile}), encoding="utf-8"
    )
    evaluation = path.parent / "source-evaluation.json"
    evaluation.write_text(
        json.dumps({"status": "complete", "model_input": profile}), encoding="utf-8"
    )
    prediction_pin = _pin(predictions)
    manifest = {
        "schema_version": PROFILE_MERGE_SCHEMA_VERSION,
        "status": "complete",
        "split": "val",
        "model_input": profile,
        "sequence_count": preparation["sequence_count"],
        "frame_count": preparation["frame_count"],
        "frame_key_sha256": preparation["frame_key_sha256"],
        "plan_fingerprint": "f" * 64,
        "output": {
            **prediction_pin,
            "sha256": output_sha256 or prediction_pin["sha256"],
        },
        "sequences": [
            {
                "sequence_id": sequence_id,
                "frame_count": frame_count,
                "source_predictions": {
                    **prediction_pin,
                },
                "run_manifest": {
                    **_pin(run_manifest),
                },
            }
            for sequence_id, frame_count in preparation["sequence_frame_counts"].items()
        ],
        "evaluation": _pin(evaluation),
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _prepared(tmp_path):
    base = tmp_path / "val-ground-truth.json"
    _write_base(base)
    output = tmp_path / "distance-bins"
    preparation = prepare_bin_activations(base, output)
    preparation_path = output / "preparation-manifest.json"
    return base, output, preparation, preparation_path


def _complete_profiles(tmp_path, preparation, base_payload):
    values = {}
    for profile in PROFILES:
        predictions = tmp_path / "profiles" / str(profile) / "predictions.jsonl"
        merge = tmp_path / "profiles" / str(profile) / "merge-manifest.json"
        _write_predictions(predictions, base_payload, profile)
        _write_merge_manifest(merge, predictions, preparation, profile)
        values[profile] = (predictions, merge)
    return values


def test_prepare_partitions_targets_and_preserves_one_frame_mapping(tmp_path):
    _, output, preparation, preparation_path = _prepared(tmp_path)

    assert preparation["status"] == "prepared_not_evaluated"
    assert preparation["split"] == "val"
    assert preparation["test_unseen_opened"] is False
    assert preparation["base_target_count"] == 5
    assert preparation["partition_target_count"] == 5
    assert [item["target_annotation_count"] for item in preparation["distance_bins"]] == [
        1,
        1,
        1,
        1,
        1,
    ]
    assert all(
        json.loads(Path(item["activation"]["path"]).read_text())["frame_key_sha256"]
        == preparation["frame_key_sha256"]
        for item in preparation["distance_bins"]
    )

    profile_inputs = [
        {
            "model_input": profile,
            "predictions": output / "missing" / str(profile) / "predictions.jsonl",
            "merge_manifest": output / "missing" / str(profile) / "merge-manifest.json",
        }
        for profile in PROFILES
    ]
    plan = write_evaluation_plan(
        preparation_path,
        output / "evaluation-plan.json",
        profile_inputs=profile_inputs,
    )
    assert plan["status"] == "waiting_for_predictions"
    assert plan["expected_evaluations"] == 10
    assert plan["metric"]["name"] == METRIC_NAME
    assert "not COCO mAP" in plan["metric"]["explicitly_not"]


def test_out_of_bin_person_detection_is_ignored_not_false_positive(tmp_path):
    base, _, _, preparation_path = _prepared(tmp_path)
    preparation, base_path, base_payload = _load_preparation(preparation_path)
    bin_item, target_ids, _ = _load_activations(preparation, base_payload)[0]
    base_dataset = load_coco(base_path)
    dataset = _dataset_for_targets(
        base_dataset,
        target_ids,
        bin_id=bin_item["bin_id"],
        activation_path=bin_item["activation"]["path"],
    )

    predictions_path = tmp_path / "two-detections.jsonl"
    images = base_payload["images"]
    records = []
    for image in images:
        records.append(
            _prediction_record(
                image,
                640,
                include_detection=image["frame_id"] in (0, 1),
            )
        )
    predictions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    result = evaluate_person_predictions(
        dataset,
        load_predictions_jsonl(predictions_path),
        iou_threshold=0.5,
        confidence_threshold=0.25,
    )

    assert result["overall"]["ground_truth"] == 1
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["ignored_predictions"] == 1


def test_val_only_tool_rejects_test_unseen_before_json_open(tmp_path):
    forbidden = tmp_path / "test-unseen-ground-truth.json"
    forbidden.write_text("this is not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        prepare_bin_activations(forbidden, tmp_path / "out")


def test_requires_distinct_complete_640_and_960_pair_before_writing(tmp_path):
    _, output, _, preparation_path = _prepared(tmp_path)
    one = tmp_path / "one.jsonl"
    one.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="distinct prediction files"):
        evaluate_prepared_bins(
            preparation_path,
            predictions_640=one,
            predictions_960=one,
            merge_manifest_640=tmp_path / "640-merge.json",
            merge_manifest_960=tmp_path / "960-merge.json",
            output_root=output / "evaluations",
            aggregate_json=output / "aggregate.json",
            aggregate_markdown=output / "aggregate.md",
        )
    assert not (output / "aggregate.json").exists()


def test_profile_merge_output_hash_is_mandatory_and_pinned(tmp_path):
    base, output, preparation, preparation_path = _prepared(tmp_path)
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    profiles = _complete_profiles(tmp_path, preparation, base_payload)
    _write_merge_manifest(
        profiles[640][1],
        profiles[640][0],
        preparation,
        640,
        output_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="merged predictions SHA-256 does not match"):
        evaluate_prepared_bins(
            preparation_path,
            predictions_640=profiles[640][0],
            predictions_960=profiles[960][0],
            merge_manifest_640=profiles[640][1],
            merge_manifest_960=profiles[960][1],
            output_root=output / "evaluations",
            aggregate_json=output / "aggregate.json",
            aggregate_markdown=output / "aggregate.md",
        )
    assert not (output / "aggregate.json").exists()


def test_profile_merge_source_prediction_hash_is_mandatory_and_pinned(tmp_path):
    base, output, preparation, preparation_path = _prepared(tmp_path)
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    profiles = _complete_profiles(tmp_path, preparation, base_payload)
    merge_path = profiles[640][1]
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    merge["sequences"][0]["source_predictions"]["sha256"] = "0" * 64
    merge_path.write_text(json.dumps(merge), encoding="utf-8")

    with pytest.raises(ValueError, match="source predictions SHA-256 does not match"):
        evaluate_prepared_bins(
            preparation_path,
            predictions_640=profiles[640][0],
            predictions_960=profiles[960][0],
            merge_manifest_640=profiles[640][1],
            merge_manifest_960=profiles[960][1],
            output_root=output / "evaluations",
            aggregate_json=output / "aggregate.json",
            aggregate_markdown=output / "aggregate.md",
        )
    assert not (output / "aggregate.json").exists()


def test_complete_manifests_cannot_hide_missing_prediction_frame(tmp_path):
    base, output, preparation, preparation_path = _prepared(tmp_path)
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    profiles = _complete_profiles(tmp_path, preparation, base_payload)
    records = profiles[640][0].read_text(encoding="utf-8").splitlines()
    profiles[640][0].write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
    _write_merge_manifest(profiles[640][1], profiles[640][0], preparation, 640)

    with pytest.raises(ValueError, match="incomplete/unpaired"):
        evaluate_prepared_bins(
            preparation_path,
            predictions_640=profiles[640][0],
            predictions_960=profiles[960][0],
            merge_manifest_640=profiles[640][1],
            merge_manifest_960=profiles[960][1],
            output_root=output / "evaluations",
            aggregate_json=output / "aggregate.json",
            aggregate_markdown=output / "aggregate.md",
        )
    assert not (output / "aggregate.json").exists()


def test_complete_profile_pair_produces_exact_ten_rows_without_invented_map(tmp_path):
    base, output, preparation, preparation_path = _prepared(tmp_path)
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    profiles = _complete_profiles(tmp_path, preparation, base_payload)

    aggregate = evaluate_prepared_bins(
        preparation_path,
        predictions_640=profiles[640][0],
        predictions_960=profiles[960][0],
        merge_manifest_640=profiles[640][1],
        merge_manifest_960=profiles[960][1],
        output_root=output / "evaluations",
        aggregate_json=output / "aggregate.json",
        aggregate_markdown=output / "aggregate.md",
    )

    assert aggregate["schema_version"] == AGGREGATE_SCHEMA_VERSION
    assert aggregate["status"] == "complete"
    assert aggregate["completeness"] == {
        "expected_profiles": [640, 960],
        "expected_distance_bins": [f"{lower}-{upper}m" for lower, upper in DISTANCE_BINS],
        "expected_evaluations": 10,
        "complete_evaluations": 10,
        "is_complete": True,
    }
    assert len(aggregate["rows"]) == 10
    assert all(row["ground_truth"] == 1 for row in aggregate["rows"])
    assert all(row["ap101_iou_0_5"] == 1.0 for row in aggregate["rows"])
    assert aggregate["metric"]["name"] == "AP101@IoU0.5"
    assert "not COCO mAP" in aggregate["metric"]["explicitly_not"]
    assert "overall" not in aggregate
    assert _pin(profiles[640][0])["sha256"] == aggregate["profiles"]["640"][
        "predictions"
    ]["sha256"]
    assert _pin(profiles[960][0])["sha256"] == aggregate["profiles"]["960"][
        "predictions"
    ]["sha256"]
    assert (output / "aggregate.md").read_text(encoding="utf-8").count("not") >= 1
