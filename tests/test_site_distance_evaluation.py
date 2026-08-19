import hashlib
import json
from pathlib import Path

import pytest

from evaluation.model import FrameKey
from validation.site_distance_evaluation import (
    _frame_key_sha256,
    _schema_validate,
    evaluate_site_distance,
    execute,
    write_plan,
)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _pin(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _predictions(path: Path, profile: int, *, frame_indices=(0, 1), miss_second=False) -> Path:
    records = []
    boxes = {
        0: [18 / 40, 10 / 30, 4 / 40, 12 / 30],
        1: [23 / 40, 9 / 30, 4 / 40, 13 / 30],
    }
    for frame_index in frame_indices:
        detections = []
        if not (miss_second and frame_index == 1):
            detections.append(
                {
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_norm_xywh": boxes[frame_index],
                }
            )
        # This correct near-range detection must be excluded, not an in-band FP.
        if frame_index == 0:
            detections.append(
                {
                    "class_name": "person",
                    "confidence": 0.8,
                    "bbox_norm_xywh": [4 / 40, 2 / 30, 4 / 40, 8 / 30],
                }
            )
        records.append(
            {
                "schema_version": "deepsafe.person-detections/v1",
                "sequence_id": "site-sequence",
                "frame_index": frame_index,
                "image_width": 40,
                "image_height": 30,
                "source_uri": "file:///site/source.mp4",
                "model_id": f"site-person-{profile}",
                "detections": detections,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path, *, threshold=0.5, missing_960_frame=False, narrow_roi=False):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic-site-source")
    calibration_doc = tmp_path / "calibration-verification.md"
    calibration_doc.write_text("survey verification\n", encoding="utf-8")
    annotation_doc = tmp_path / "annotation-qa.md"
    annotation_doc.write_text("all people reviewed\n", encoding="utf-8")
    acceptance_doc = tmp_path / "acceptance-approval.md"
    acceptance_doc.write_text("criterion approved by site owner\n", encoding="utf-8")

    calibration = _write_json(
        tmp_path / "calibration.json",
        {
            "schema_version": "deepsafe.site-ground-plane-calibration/v1",
            "status": "verified",
            "calibration_id": "cal-1",
            "site_id": "site-1",
            "camera_id": "cam-1",
            "camera_configuration_id": "cam-1-pose-a-lens-a",
            "model": "planar_homography_image_to_ground",
            "distance_unit": "m",
            "image": {"width": 40, "height": 30},
            "image_to_ground_homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "camera_ground_position_m": [20, 0],
            "valid_image_polygon_px": (
                [[0, 0], [20, 0], [20, 30], [0, 30]]
                if narrow_roi
                else [[0, 0], [40, 0], [40, 30], [0, 30]]
            ),
            "verification": {
                "status": "pass",
                "method": "survey_control_points",
                "verified_by": "surveyor",
                "verified_at": "2026-07-15T10:00:00Z",
                "reference_point_count": 8,
                "maximum_ground_error_m": 0.08,
                "allowed_ground_error_m": 0.10,
                "document": _pin(calibration_doc),
            },
        },
    )
    ground_truth = _write_json(
        tmp_path / "ground-truth.json",
        {
            "schema_version": "deepsafe.site-distance-ground-truth/v1",
            "status": "complete",
            "evidence_kind": "deployment_site_calibrated_person_ground_truth",
            "dataset_id": "site-gt-1",
            "site_id": "site-1",
            "camera_id": "cam-1",
            "camera_configuration_id": "cam-1-pose-a-lens-a",
            "calibration_id": "cal-1",
            "sequence_id": "site-sequence",
            "source_asset": _pin(source),
            "image": {"width": 40, "height": 30},
            "annotation": {
                "status": "verified",
                "all_visible_people_in_calibrated_roi_annotated": True,
                "box_format": "pixel_xywh",
                "distance_point": "bbox_bottom_center_ground_contact",
                "reviewed_by": "annotator-qa",
                "reviewed_at": "2026-07-15T11:00:00Z",
                "document": _pin(annotation_doc),
            },
            "frames": [
                {
                    "frame_index": 0,
                    "persons": [
                        {"object_id": "p0", "bbox_pixel_xywh": [18, 10, 4, 12], "ignored": False},
                        {"object_id": "near", "bbox_pixel_xywh": [4, 2, 4, 8], "ignored": False},
                    ],
                },
                {
                    "frame_index": 1,
                    "persons": [
                        {"object_id": "p1", "bbox_pixel_xywh": [23, 9, 4, 13], "ignored": False}
                    ],
                },
            ],
        },
    )
    acceptance = _write_json(
        tmp_path / "acceptance.json",
        {
            "schema_version": "deepsafe.site-distance-acceptance/v1",
            "status": "approved",
            "criterion_id": "owner-distance-gate-v1",
            "task": "person_detection",
            "evidence_kind": "deployment_site_calibrated_ground_plane",
            "distance_unit": "m",
            "distance_bin_m": [20, 25],
            "boundary": "lower_inclusive_upper_exclusive",
            "profiles": [640, 960],
            "evaluation_config": {
                "iou_threshold": 0.5,
                "confidence_threshold": 0.25,
                "distance_point": "bbox_bottom_center_ground_contact",
            },
            "minimum_ground_truth_instances_per_profile": 2,
            "rules": [
                {"metric": "recall", "operator": "gte", "threshold": threshold, "applies_to": "each_profile"}
            ],
            "approval": {
                "approved_by": "site-owner",
                "approved_at": "2026-07-15T12:00:00Z",
                "document": _pin(acceptance_doc),
            },
        },
    )

    manifests = {}
    for profile in (640, 960):
        indices = (0,) if profile == 960 and missing_960_frame else (0, 1)
        predictions = _predictions(
            tmp_path / f"profile-{profile}" / "predictions.jsonl",
            profile,
            frame_indices=indices,
            miss_second=profile == 960,
        )
        keys = {FrameKey("site-sequence", value) for value in indices}
        manifest = _write_json(
            tmp_path / f"profile-{profile}" / "run-manifest.json",
            {
                "schema_version": "deepsafe.site-distance-profile-run/v1",
                "status": "complete",
                "evidence_kind": "deployment_site_profile_inference",
                "profile": profile,
                "dataset_id": "site-gt-1",
                "site_id": "site-1",
                "camera_id": "cam-1",
                "camera_configuration_id": "cam-1-pose-a-lens-a",
                "calibration_id": "cal-1",
                "sequence_id": "site-sequence",
                "source_asset_sha256": _pin(source)["sha256"],
                "ground_truth": _pin(ground_truth),
                "calibration": _pin(calibration),
                "predictions": _pin(predictions),
                "frame_contract": {
                    "status": "exact",
                    "expected_frames": 2,
                    "serialized_frames": len(indices),
                    "frame_key_sha256": _frame_key_sha256(keys),
                },
                "inference": {
                    "exit_code": 0,
                    "safety_guard_status": "pass",
                    "model_id": f"site-person-{profile}",
                    "model_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "completed_at": "2026-07-15T13:00:00Z",
                },
            },
        )
        manifests[profile] = manifest
    return calibration, ground_truth, acceptance, manifests


def test_default_plan_is_inert_and_has_no_threshold(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan = write_plan(
        calibration_path=tmp_path / "missing-calibration.json",
        ground_truth_path=tmp_path / "missing-gt.json",
        acceptance_path=tmp_path / "missing-acceptance.json",
        profile_640_manifest=tmp_path / "missing-640.json",
        profile_960_manifest=tmp_path / "missing-960.json",
        output_path=tmp_path / "evaluation.json",
        plan_path=plan_path,
    )
    assert plan["status"] == "waiting_for_inputs"
    assert plan["dry_run"] is True
    assert plan["gpu_or_docker_executed"] is False
    assert plan["final_evaluation_written"] is False
    assert "no default threshold" in plan["acceptance_policy"]
    assert not (tmp_path / "evaluation.json").exists()
    _schema_validate(plan, "site-distance-evaluation-plan-v1.schema.json", "test plan")


def test_passing_pair_writes_strict_final_evidence(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path)
    output = tmp_path / "evaluation.json"
    attempt = tmp_path / "attempt.json"
    assert execute(
        calibration_path=calibration,
        ground_truth_path=gt,
        acceptance_path=acceptance,
        profile_640_manifest=manifests[640],
        profile_960_manifest=manifests[960],
        output_path=output,
        attempt_path=attempt,
    )
    result = json.loads(output.read_text())
    assert result["schema_version"] == "deepsafe.distance-validation/v1"
    assert result["evidence_kind"] == "deployment_site_calibrated_ground_plane_person_detection"
    assert result["ground_truth"]["ground_truth_instances_20_25m"] == 2
    assert result["profiles"]["640"]["metrics"]["recall"] == 1.0
    assert result["profiles"]["960"]["metrics"]["recall"] == 0.5
    assert result["profiles"]["640"]["metrics"]["predictions_excluded_outside_calibrated_band"] == 1
    assert result["acceptance"]["status"] == "pass"
    assert result["loaf_evidence"]["used"] is False
    assert result["gpu_or_docker_executed_by_evaluator"] is False
    assert not attempt.exists()
    _schema_validate(result, "distance-validation-v1.schema.json", "test result")


def test_failed_owner_threshold_writes_attempt_but_not_final(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path, threshold=0.9)
    output = tmp_path / "evaluation.json"
    attempt = tmp_path / "attempt.json"
    assert not execute(
        calibration_path=calibration,
        ground_truth_path=gt,
        acceptance_path=acceptance,
        profile_640_manifest=manifests[640],
        profile_960_manifest=manifests[960],
        output_path=output,
        attempt_path=attempt,
    )
    assert not output.exists()
    rejected = json.loads(attempt.read_text())
    assert rejected["status"] == "acceptance_failed"
    assert rejected["acceptance"]["rules"][0]["profile_values"]["960"] == 0.5
    assert rejected["acceptance"]["rules"][0]["status"] == "fail"


def test_mismatched_profile_frame_set_is_rejected(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path, missing_960_frame=True)
    with pytest.raises(ValueError, match="profile 960 frame set mismatch"):
        evaluate_site_distance(
            calibration_path=calibration,
            ground_truth_path=gt,
            acceptance_path=acceptance,
            profile_640_manifest=manifests[640],
            profile_960_manifest=manifests[960],
        )


def test_ground_truth_outside_calibrated_roi_is_rejected(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path, narrow_roi=True)
    with pytest.raises(ValueError, match="outside the calibrated ROI"):
        evaluate_site_distance(
            calibration_path=calibration,
            ground_truth_path=gt,
            acceptance_path=acceptance,
            profile_640_manifest=manifests[640],
            profile_960_manifest=manifests[960],
        )


def test_acceptance_without_explicit_rules_is_rejected(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path)
    payload = json.loads(acceptance.read_text())
    payload.pop("rules")
    _write_json(acceptance, payload)
    with pytest.raises(ValueError, match="rules"):
        evaluate_site_distance(
            calibration_path=calibration,
            ground_truth_path=gt,
            acceptance_path=acceptance,
            profile_640_manifest=manifests[640],
            profile_960_manifest=manifests[960],
        )


def test_tampered_calibration_evidence_pin_is_rejected(tmp_path):
    calibration, gt, acceptance, manifests = _fixture(tmp_path)
    (tmp_path / "calibration-verification.md").write_text(
        "tamper verification\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        evaluate_site_distance(
            calibration_path=calibration,
            ground_truth_path=gt,
            acceptance_path=acceptance,
            profile_640_manifest=manifests[640],
            profile_960_manifest=manifests[960],
        )


def test_loaf_cannot_substitute_for_site_calibration(tmp_path):
    with pytest.raises(ValueError, match="LOAF is auxiliary evidence"):
        evaluate_site_distance(
            calibration_path=tmp_path / "loaf" / "calibration.json",
            ground_truth_path=tmp_path / "gt.json",
            acceptance_path=tmp_path / "acceptance.json",
            profile_640_manifest=tmp_path / "640.json",
            profile_960_manifest=tmp_path / "960.json",
        )
