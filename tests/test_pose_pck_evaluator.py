from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from validation import pose_pck_evaluator as pose


ROOT = Path(__file__).resolve().parents[1]
HEX = {
    "source": hashlib.sha256(b"source-manifest").hexdigest(),
    "annotator": hashlib.sha256(b"annotator").hexdigest(),
    "reviewer": hashlib.sha256(b"reviewer").hexdigest(),
    "protocol": hashlib.sha256(b"annotation-protocol").hexdigest(),
    "review": hashlib.sha256(b"review-receipt").hexdigest(),
    "model": hashlib.sha256(b"pose-model").hexdigest(),
    "model_contract": hashlib.sha256(b"pose-model-contract").hexdigest(),
    "config": hashlib.sha256(b"runtime-config").hexdigest(),
    "runtime": hashlib.sha256(b"runtime-receipt").hexdigest(),
}


def _gt_keypoints(bbox: list[float], *, visibility: int = 2) -> list[dict]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return [
        {
            "name": name,
            "x_px": x1 + width * (0.30 + (index % 5) * 0.08),
            "y_px": y1 + height * (0.25 + (index // 5) * 0.15),
            "visibility": visibility,
        }
        for index, name in enumerate(pose.COCO_KEYPOINTS)
    ]


def _pred_keypoints(gt_keypoints: list[dict]) -> list[dict]:
    return [
        {
            "name": item["name"],
            "present": True,
            "x_px": item["x_px"],
            "y_px": item["y_px"],
            "confidence": 0.95,
        }
        for item in gt_keypoints
    ]


def _minimums() -> dict:
    return {
        "overall_visible_keypoints_per_profile": 17,
        "by_joint_per_profile": {name: 1 for name in pose.COCO_KEYPOINTS},
        "by_bbox_height_per_profile": {name: 1 for name in pose.BBOX_HEIGHT_SLICES},
        "by_view_per_profile": {
            "medium_close": 1,
            "overhead_security_camera": 1,
        },
        "by_video_type_per_profile": {f"video-type-{index}": 1 for index in range(10)},
        "by_occlusion_per_profile": {name: 1 for name in pose.OCCLUSION_SLICES},
    }


def _documents() -> tuple[dict, dict]:
    plan = {
        "plan_id": "pose-pck-plan-v1",
        "approved_by_role": "project_owner",
        "approved_at_utc": "2026-07-17T18:13:40Z",
        "acceptance_policy_fingerprint_sha256": pose.APPROVED_POLICY_FINGERPRINT_SHA256,
        "source_manifest_sha256": HEX["source"],
        "profiles": [640, 960],
        "minimum_distinct_video_types": 10,
        "required_video_types": [f"video-type-{index}" for index in range(10)],
        "minimum_gt_counts": _minimums(),
    }
    plan["fingerprint_sha256"] = pose.canonical_fingerprint(plan)
    heights = (20.0, 40.0, 80.0, 140.0)
    frames = []
    for index in range(10):
        height = heights[index % len(heights)]
        bbox = [10.0, 10.0, 10.0 + height, 10.0 + height]
        frames.append(
            {
                "camera_id": f"cam-{index % 3}",
                "frame_id": index,
                "video_type": f"video-type-{index}",
                "view_type": "medium_close" if index % 2 == 0 else "overhead_security_camera",
                "image": {"width_px": 200, "height_px": 200},
                "persons": [
                    {
                        "person_id": f"person-{index}",
                        "track_id": None if index == 3 else f"track-{index}",
                        "bbox_xyxy": bbox,
                        "occlusion": pose.OCCLUSION_SLICES[index % 3],
                        "keypoints": _gt_keypoints(bbox),
                    }
                ],
            }
        )
    gt = {
        "schema_version": pose.GROUND_TRUTH_SCHEMA_VERSION,
        "dataset_id": "pose-gt-v1",
        "source_manifest_sha256": HEX["source"],
        "coordinate_space": pose.SOURCE_COORDINATE_SPACE,
        "keypoint_order": list(pose.COCO_KEYPOINTS),
        "review_provenance": {
            "status": "complete",
            "method": "independent_human_review",
            "annotator_identity_sha256": HEX["annotator"],
            "reviewer_identity_sha256": HEX["reviewer"],
            "reviewer_independent_of_prediction_producer": True,
            "reviewed_at_utc": "2026-07-17T18:13:45Z",
            "annotation_protocol_sha256": HEX["protocol"],
            "review_receipt_sha256": HEX["review"],
        },
        "evaluation_plan": plan,
        "frames": frames,
    }
    prediction_frames = []
    for profile in (640, 960):
        for frame in frames:
            prediction_frames.append(
                {
                    "profile": profile,
                    "camera_id": frame["camera_id"],
                    "frame_id": frame["frame_id"],
                    "image": copy.deepcopy(frame["image"]),
                    "persons": [
                        {
                            "prediction_id": f"pred-{profile}-{frame['frame_id']}",
                            "track_id": frame["persons"][0]["track_id"],
                            "bbox_xyxy": copy.deepcopy(frame["persons"][0]["bbox_xyxy"]),
                            "confidence": 0.99,
                            "keypoints": _pred_keypoints(frame["persons"][0]["keypoints"]),
                        }
                    ],
                }
            )
    predictions = {
        "schema_version": pose.PREDICTIONS_SCHEMA_VERSION,
        "prediction_set_id": "pose-predictions-v1",
        "ground_truth_dataset_id": gt["dataset_id"],
        "source_manifest_sha256": HEX["source"],
        "coordinate_space": pose.SOURCE_COORDINATE_SPACE,
        "keypoint_order": list(pose.COCO_KEYPOINTS),
        "profile_runs": [
            {
                "profile": profile,
                "run_id": f"pose-run-{profile}",
                "started_at_utc": "2026-07-17T18:13:41Z",
                "completed_at_utc": "2026-07-17T18:13:42Z",
                "acceptance_policy_fingerprint_sha256": pose.APPROVED_POLICY_FINGERPRINT_SHA256,
                "model_contract_sha256": HEX["model_contract"],
                "model_artifact_sha256": HEX["model"],
                "runtime_config_sha256": HEX["config"],
                "runtime_receipt_sha256": HEX["runtime"],
                "evidence_kind": "deepstream9_gpu_prediction_export",
            }
            for profile in (640, 960)
        ],
        "frames": prediction_frames,
    }
    return gt, predictions


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _paths(tmp_path: Path, gt: dict | None = None, predictions: dict | None = None) -> tuple[Path, Path]:
    default_gt, default_predictions = _documents()
    return (
        _write(tmp_path / "ground-truth.json", default_gt if gt is None else gt),
        _write(tmp_path / "predictions.json", default_predictions if predictions is None else predictions),
    )


def _to_jsonl(path: Path, value: dict) -> Path:
    manifest = {key: copy.deepcopy(item) for key, item in value.items() if key != "frames"}
    manifest["record_type"] = "manifest"
    records = [manifest]
    for frame in value["frames"]:
        record = copy.deepcopy(frame)
        record["record_type"] = "frame"
        records.append(record)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return path


def test_perfect_pose_scores_every_visible_keypoint_and_passes_declared_coverage(tmp_path: Path) -> None:
    gt_path, predictions_path = _paths(tmp_path)
    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)

    assert [item["profile"] for item in receipt["profiles"]] == [640, 960]
    assert all(item["overall"] == {"correct": 170, "total": 170, "pck": 1.0} for item in receipt["profiles"])
    assert receipt["all_profiles"]["overall"] == {"correct": 340, "total": 340, "pck": 1.0}
    assert receipt["profiles"][0]["by_joint"]["nose"]["pck"] == 1.0
    assert receipt["profiles"][0]["by_bbox_height"]["lt_32_px"]["total"] > 0
    assert receipt["profiles"][0]["by_view"]["overhead_security_camera"]["total"] > 0
    assert receipt["profiles"][0]["by_video_type"]["video-type-0"]["total"] > 0
    assert receipt["profiles"][0]["by_occlusion"]["heavy"]["total"] > 0
    assert receipt["quality_gate"]["status"] == "pass"
    assert receipt["product_acceptance_claimed"] is False
    assert pose.canonical_fingerprint(receipt) == receipt["fingerprint_sha256"]


def test_pck_boundary_at_exactly_point_two_is_inclusive(tmp_path: Path) -> None:
    gt, predictions = _documents()
    gt_person = gt["frames"][0]["persons"][0]
    pred_frame = next(frame for frame in predictions["frames"] if frame["profile"] == 640 and frame["frame_id"] == 0)
    normalizer = max(gt_person["bbox_xyxy"][2] - gt_person["bbox_xyxy"][0], gt_person["bbox_xyxy"][3] - gt_person["bbox_xyxy"][1])
    pred_frame["persons"][0]["keypoints"][0]["x_px"] += 0.2 * normalizer
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)

    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    assert receipt["profiles"][0]["overall"]["correct"] == 170
    assert receipt["profiles"][0]["overall"]["pck"] == 1.0


def test_missing_keypoint_and_person_are_zero_contributions_without_denominator_loss(tmp_path: Path) -> None:
    gt, predictions = _documents()
    frame_640_0 = next(frame for frame in predictions["frames"] if frame["profile"] == 640 and frame["frame_id"] == 0)
    missing = frame_640_0["persons"][0]["keypoints"][0]
    missing.update({"present": False, "x_px": None, "y_px": None, "confidence": None})
    predictions["frames"] = [frame for frame in predictions["frames"] if not (frame["profile"] == 960 and frame["frame_id"] == 1)]
    extra = copy.deepcopy(frame_640_0["persons"][0])
    extra.update({"prediction_id": "prediction-only", "track_id": "prediction-only-track", "bbox_xyxy": [160.0, 160.0, 190.0, 190.0]})
    frame_640_0["persons"].append(extra)
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)

    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    metrics_640, metrics_960 = receipt["profiles"]
    assert metrics_640["overall"] == {"correct": 169, "total": 170, "pck": 169 / 170}
    assert metrics_960["overall"] == {"correct": 153, "total": 170, "pck": 153 / 170}
    assert metrics_960["association_diagnostics"]["missing_prediction_frames"] == 1
    assert metrics_640["association_diagnostics"]["prediction_person_match_precision"] == 10 / 11
    assert metrics_640["overall"]["total"] == 170


def test_coco_visibility_one_is_not_in_visible_pck_denominator(tmp_path: Path) -> None:
    gt, predictions = _documents()
    gt["frames"][0]["persons"][0]["keypoints"][0]["visibility"] = 1
    pred = next(frame for frame in predictions["frames"] if frame["profile"] == 640 and frame["frame_id"] == 0)
    pred["persons"][0]["keypoints"][0].update({"present": False, "x_px": None, "y_px": None, "confidence": None})
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)

    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    assert receipt["profiles"][0]["overall"] == {"correct": 169, "total": 169, "pck": 1.0}
    assert receipt["profiles"][0]["by_joint"]["nose"]["total"] == 9


def test_wrong_track_association_is_not_repaired_by_bbox_iou(tmp_path: Path) -> None:
    gt, predictions = _documents()
    frame = gt["frames"][0]
    second_bbox = [150.0, 10.0, 190.0, 50.0]
    second = {
        "person_id": "second-person",
        "track_id": "second-track",
        "bbox_xyxy": second_bbox,
        "occlusion": "none",
        "keypoints": _gt_keypoints(second_bbox),
    }
    frame["persons"].append(second)
    for profile in (640, 960):
        pred_frame = next(item for item in predictions["frames"] if item["profile"] == profile and item["frame_id"] == 0)
        first_prediction = pred_frame["persons"][0]
        second_prediction = {
            "prediction_id": f"second-pred-{profile}",
            "track_id": "second-track",
            "bbox_xyxy": second_bbox,
            "confidence": 0.99,
            "keypoints": _pred_keypoints(second["keypoints"]),
        }
        first_prediction["track_id"] = "second-track"
        second_prediction["track_id"] = "track-0"
        pred_frame["persons"] = [first_prediction, second_prediction]
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)

    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    assert receipt["profiles"][0]["association_diagnostics"]["matched_by_canonical_track"] == 10
    assert receipt["profiles"][0]["overall"]["correct"] == 153
    assert receipt["profiles"][0]["overall"]["total"] == 187


@pytest.mark.parametrize("target", ["ground_truth", "predictions"])
def test_duplicate_logical_frame_is_rejected(tmp_path: Path, target: str) -> None:
    gt, predictions = _documents()
    document = gt if target == "ground_truth" else predictions
    document["frames"].append(copy.deepcopy(document["frames"][0]))
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="duplicate .*frame"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


def test_unknown_prediction_frame_is_rejected(tmp_path: Path) -> None:
    gt, predictions = _documents()
    predictions["frames"][0]["frame_id"] = 9999
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="unknown prediction frame"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gt, pred: pred["frames"][0]["image"].update({"width_px": 201}), "dimensions differ"),
        (lambda gt, pred: pred["frames"][0]["persons"][0]["keypoints"][0].update({"x_px": 500.0}), "outside the source frame"),
        (lambda gt, pred: pred["frames"][0]["persons"][0]["keypoints"].reverse(), "COCO-17 order"),
    ],
)
def test_dimensions_bounds_and_keypoint_order_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    gt, predictions = _documents()
    mutation(gt, predictions)
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match=message):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


def test_duplicate_json_key_and_nonfinite_number_are_rejected(tmp_path: Path) -> None:
    gt, predictions = _documents()
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    duplicate = gt_path.read_text(encoding="utf-8").replace('"dataset_id": "pose-gt-v1"', '"dataset_id": "pose-gt-v1", "dataset_id": "forged"', 1)
    gt_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(pose.PosePCKError, match="strict JSON"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)

    _write(gt_path, gt)
    nonfinite = predictions_path.read_text(encoding="utf-8").replace('"confidence": 0.99', '"confidence": NaN', 1)
    predictions_path.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(pose.PosePCKError, match="strict JSON"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


def test_preapproval_policy_binding_and_run_lineage_are_strict(tmp_path: Path) -> None:
    gt, predictions = _documents()
    predictions["profile_runs"][0]["started_at_utc"] = "2026-07-17T18:13:40Z"
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="did not start after evaluation-plan approval"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)

    gt, predictions = _documents()
    predictions["profile_runs"][0]["acceptance_policy_fingerprint_sha256"] = "0" * 64
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="policy fingerprint differs"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)

    gt, predictions = _documents()
    predictions["profile_runs"][1]["model_contract_sha256"] = "1" * 64
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="one frozen pose model contract"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


def test_evaluation_plan_tamper_is_rejected_even_with_valid_json(tmp_path: Path) -> None:
    gt, predictions = _documents()
    gt["evaluation_plan"]["minimum_distinct_video_types"] = 1
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="weakens the policy|fingerprint differs"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)


def test_evaluation_plan_binds_source_and_nonempty_video_type_coverage(tmp_path: Path) -> None:
    gt, predictions = _documents()
    gt["evaluation_plan"]["source_manifest_sha256"] = "0" * 64
    gt["evaluation_plan"]["fingerprint_sha256"] = pose.canonical_fingerprint(gt["evaluation_plan"])
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    with pytest.raises(pose.PosePCKError, match="source manifest differs"):
        pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)

    gt, predictions = _documents()
    gt["frames"][0]["persons"][0]["keypoints"] = [
        {**item, "visibility": 1} for item in gt["frames"][0]["persons"][0]["keypoints"]
    ]
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    assert receipt["profiles"][0]["by_video_type"]["video-type-0"]["total"] == 0
    assert receipt["profiles"][0]["minimum_gt_coverage"]["by_video_type"]["video-type-0"]["status"] == "fail"
    assert receipt["quality_gate"]["status"] == "fail"


def test_jsonl_contract_scores_identically_to_json_metrics(tmp_path: Path) -> None:
    gt, predictions = _documents()
    json_gt, json_predictions = _paths(tmp_path, gt, predictions)
    json_receipt = pose.evaluate_files(json_gt, json_predictions, project_root=ROOT)
    jsonl_gt = _to_jsonl(tmp_path / "ground-truth.jsonl", gt)
    jsonl_predictions = _to_jsonl(tmp_path / "predictions.jsonl", predictions)
    jsonl_receipt = pose.evaluate_files(jsonl_gt, jsonl_predictions, project_root=ROOT)

    assert jsonl_receipt["profiles"] == json_receipt["profiles"]
    assert jsonl_receipt["all_profiles"] == json_receipt["all_profiles"]
    assert jsonl_receipt["input_pins"]["ground_truth"]["canonical_content_fingerprint_sha256"] == json_receipt["input_pins"]["ground_truth"]["canonical_content_fingerprint_sha256"]
    assert jsonl_receipt["input_pins"]["ground_truth"]["format"] == "jsonl"


def test_atomic_no_overwrite_and_deterministic_replay_detect_tamper(tmp_path: Path) -> None:
    gt_path, predictions_path = _paths(tmp_path)
    first = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    second = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    assert first == second
    output = tmp_path / "receipt.json"
    pose.atomic_write_no_overwrite(output, first)
    assert stat_mode(output) == 0o440
    assert pose.verify_receipt(output, gt_path, predictions_path, project_root=ROOT) == first
    with pytest.raises(pose.PosePCKError, match="overwrite"):
        pose.atomic_write_no_overwrite(output, first)

    os.chmod(output, 0o600)
    forged = copy.deepcopy(first)
    forged["profiles"][0]["overall"]["correct"] -= 1
    output.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    with pytest.raises(pose.PosePCKError, match="fingerprint differs"):
        pose.verify_receipt(output, gt_path, predictions_path, project_root=ROOT)


def test_replay_rejects_prediction_input_drift(tmp_path: Path) -> None:
    gt_path, predictions_path = _paths(tmp_path)
    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    output = tmp_path / "receipt.json"
    pose.atomic_write_no_overwrite(output, receipt)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["frames"][0]["persons"][0]["keypoints"][0]["x_px"] += 1.0
    _write(predictions_path, predictions)
    with pytest.raises(pose.PosePCKError, match="semantic replay"):
        pose.verify_receipt(output, gt_path, predictions_path, project_root=ROOT)


def test_checked_in_schemas_are_draft_2020_12_and_validate_documents(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    gt, predictions = _documents()
    schemas = {
        "ground_truth": json.loads((ROOT / pose.SCHEMA_PATHS["ground_truth"]).read_text(encoding="utf-8")),
        "predictions": json.loads((ROOT / pose.SCHEMA_PATHS["predictions"]).read_text(encoding="utf-8")),
        "receipt": json.loads((ROOT / pose.SCHEMA_PATHS["receipt"]).read_text(encoding="utf-8")),
    }
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schemas["ground_truth"]).validate(gt)
    jsonschema.Draft202012Validator(schemas["predictions"]).validate(predictions)
    gt_path, predictions_path = _paths(tmp_path, gt, predictions)
    receipt = pose.evaluate_files(gt_path, predictions_path, project_root=ROOT)
    jsonschema.Draft202012Validator(schemas["receipt"]).validate(receipt)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
