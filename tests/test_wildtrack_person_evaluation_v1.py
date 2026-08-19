from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from evaluation.model import Box
from validation import wildtrack_person_evaluation_v1 as lane


ROOT = Path(__file__).resolve().parents[1]
REAL_PLAN = ROOT / "validation/results/wildtrack/person-evaluation-v1/readiness-plan.json"


@pytest.fixture(scope="module")
def real_gt() -> lane.GroundTruthEvidence:
    return lane.load_ground_truth()


@pytest.fixture(scope="module")
def real_plan() -> dict:
    return lane.build_readiness_plan()


def _synthetic_evidence() -> tuple[lane.GroundTruthEvidence, lane.PredictionEvidence]:
    frame = lane.GroundTruthFrame(
        camera_index=0,
        camera_id="CVLab1",
        frame_number=0,
        image_pin={"path": "Image_subsets/C1/00000000.png", "bytes": 1, "sha256": "a" * 64},
        persons=(
            lane.GroundTruthPerson(0, Box(20, 0, 4, 2), False, True, 22.0),
            lane.GroundTruthPerson(1, Box(40, 0, 4, 2), False, True, 30.0),
            lane.GroundTruthPerson(2, Box(60, 0, 4, 2), True, False, 22.0),
        ),
    )
    gt = lane.GroundTruthEvidence(
        receipt={},
        receipt_pin={},
        ground_truth_pin={},
        source_manifest_pin={},
        frames={("CVLab1", 0): frame},
        homographies={"CVLab1": ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))},
        camera_centers={"CVLab1": (0.0, 0.0, 2.0)},
    )
    detections = (
        lane.Detection("active", "CVLab1", 0, 0.95, Box(20, 0, 4, 2)),
        lane.Detection("other-distance", "CVLab1", 0, 0.90, Box(40, 0, 4, 2)),
        lane.Detection("truncated", "CVLab1", 0, 0.85, Box(60, 0, 4, 2)),
        lane.Detection("outside-projection", "CVLab1", 0, 0.80, Box(100, 50, 4, 2)),
        lane.Detection("inside-projection-fp", "CVLab1", 0, 0.75, Box(20, 50, 4, 2)),
    )
    predictions = lane.PredictionEvidence(
        profile=640,
        pin={},
        frames={("CVLab1", 0): {}},
        detections=detections,
    )
    return gt, predictions


def _manifest(plan: dict, gt: lane.GroundTruthEvidence, profile: int = 640) -> dict:
    fake = lambda path, token: {"path": path, "bytes": 123, "sha256": token * 64}
    plan_pin = fake("validation/results/wildtrack/person-evaluation-v1/readiness-plan.json", "1")
    profile_plan = plan["model"]["profiles"][str(profile)]
    value = {
        "schema_version": lane.RUN_SCHEMA_VERSION,
        "status": "complete",
        "run_id": f"wildtrack-profile-{profile}",
        "pair_id": "wildtrack-pair-001",
        "dataset_id": lane.DATASET_ID,
        "profile": profile,
        "network_input": {"width": profile, "height": profile},
        "bbox_space": lane.BBOX_SPACE,
        "model": {
            "model_id": plan["model"]["model_id"],
            "model_receipt": copy.deepcopy(plan["model"]["model_receipt"]),
            "model_receipt_fingerprint_sha256": plan["model"]["model_receipt_fingerprint_sha256"],
            "source_model_artifact": copy.deepcopy(profile_plan["source_model_artifact"]),
            "deployed_engine": fake(profile_plan["expected_engine_path"], "2"),
            "engine_receipt": fake(profile_plan["expected_engine_receipt_path"], "3"),
        },
        "thresholds": {
            "fixed_operating_confidence": profile_plan["fixed_operating_confidence"],
            "iou": 0.5,
            "serialization_confidence_floor": 0.0,
        },
        "inputs": {
            "ground_truth_receipt": copy.deepcopy(gt.receipt_pin),
            "ground_truth": copy.deepcopy(gt.ground_truth_pin),
            "source_manifest": copy.deepcopy(gt.source_manifest_pin),
        },
        "outputs": {
            "prediction_schema_version": lane.PREDICTION_SCHEMA_VERSION,
            "predictions": fake(plan["expected_runs"][str(profile)]["predictions_path"], "4"),
            "record_count": 2800,
            "detection_count": 0,
        },
        "runtime": {
            "framework": "NVIDIA DeepStream",
            "deepstream_major_version": 9,
            "gpu_inference_executed": True,
            "tensorrt_executed": True,
            "raw_predictions_exported": True,
        },
        "plan": {**plan_pin, "fingerprint_sha256": plan["fingerprint_sha256"]},
        "fingerprint_sha256": "",
    }
    value["fingerprint_sha256"] = lane._fingerprint(value)
    return value


def _manifest_contract(value: dict, plan: dict, gt: lane.GroundTruthEvidence, profile: int = 640) -> None:
    lane._validate_run_manifest_contract(
        value,
        profile=profile,
        plan=plan,
        plan_pin={key: value["plan"][key] for key in ("path", "bytes", "sha256")},
        plan_relative=value["plan"]["path"],
        gt=gt,
    )


def _refresh_fingerprint(value: dict) -> None:
    value["fingerprint_sha256"] = lane._fingerprint(value)


def _prediction_row(frame: lane.GroundTruthFrame, profile: int = 640) -> dict:
    return {
        "schema_version": lane.PREDICTION_SCHEMA_VERSION,
        "dataset_id": lane.DATASET_ID,
        "profile": profile,
        "model_id": lane.MODEL_ID,
        "camera_index": frame.camera_index,
        "camera_id": frame.camera_id,
        "frame_number": frame.frame_number,
        "source_dimensions": {"width": 1920, "height": 1080},
        "bbox_space": lane.BBOX_SPACE,
        "source_image": copy.deepcopy(frame.image_pin),
        "detections": [],
    }


def _prediction_project(tmp_path: Path) -> Path:
    schema_root = tmp_path / "validation/schemas"
    schema_root.mkdir(parents=True)
    shutil.copy2(
        ROOT / "validation/schemas" / lane.PREDICTION_SCHEMA_NAME,
        schema_root / lane.PREDICTION_SCHEMA_NAME,
    )
    return tmp_path


def _write_prediction_rows(project: Path, rows: list[dict]) -> tuple[Path, dict]:
    path = project / "predictions.jsonl"
    path.write_bytes(b"".join(lane._canonical_bytes(row) + b"\n" for row in rows))
    pin = lane._portable_pin(path, project_root=project, context="fixture predictions")
    return path, pin


def _all_rows(gt: lane.GroundTruthEvidence, profile: int = 640) -> list[dict]:
    return [
        _prediction_row(gt.frames[(camera, frame_number)], profile)
        for camera in lane.CAMERA_IDS
        for frame_number in lane.FRAME_NUMBERS
    ]


def test_schemas_are_closed_valid_draft_2020_12() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    for name in (
        lane.PLAN_SCHEMA_NAME,
        lane.PREDICTION_SCHEMA_NAME,
        lane.RUN_SCHEMA_NAME,
        lane.EVALUATION_SCHEMA_NAME,
    ):
        schema = json.loads((ROOT / "validation/schemas" / name).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        for definition in schema.get("$defs", {}).values():
            if definition.get("type") == "object":
                assert definition.get("additionalProperties") is False


def test_real_gt_external_pin_and_exact_coverage_replay(real_gt: lane.GroundTruthEvidence) -> None:
    assert real_gt.receipt_pin["sha256"] == lane.GT_RECEIPT_EXTERNAL_SHA256
    assert real_gt.receipt["source"]["archive"]["sha256"] == lane.ARCHIVE_SHA256
    assert set(real_gt.frames) == lane._expected_frame_keys()
    assert len(real_gt.frames) == 2800
    assert sum(len(frame.persons) for frame in real_gt.frames.values()) == 42721


def test_readiness_plan_contains_no_success_metrics(real_plan: dict) -> None:
    assert real_plan["status"] == "awaiting_deepstream_predictions"
    assert real_plan["readiness"]["missing_profiles"] == [640, 960]
    assert real_plan["readiness"]["metrics_present"] is False
    assert real_plan["claim_boundary"]["person_model_quality_measured"] is False
    assert '"metrics":' not in json.dumps(real_plan)
    assert real_plan["fingerprint_sha256"] == lane._fingerprint(real_plan)
    lane._schema_validate(real_plan, lane.PLAN_SCHEMA_NAME, project_root=ROOT, context="test plan")


def test_published_readiness_plan_matches_live_deterministic_replay(real_plan: dict) -> None:
    published, pin = lane.load_plan(REAL_PLAN)
    assert published == real_plan
    assert pin["sha256"] == "5c3c22d1b5f5192f0bb111ad8e5385e548f22207d69c3f00234c76d2c239d5d4"
    assert published["fingerprint_sha256"] == "9ef071aa425fcb89e91dca6375920a697e596b1f513832b76d380161c3639110"


def test_scope_metrics_keep_truncated_gt_only_in_overall_visible() -> None:
    gt, predictions = _synthetic_evidence()
    overall = lane._scope_metrics(
        gt=gt,
        predictions=predictions,
        scope="overall_visible",
        fixed_confidence=0.5,
        cameras={"CVLab1"},
    )
    eligible = lane._scope_metrics(
        gt=gt,
        predictions=predictions,
        scope="distance_evaluation_eligible",
        fixed_confidence=0.5,
        cameras={"CVLab1"},
    )
    band = lane._scope_metrics(
        gt=gt,
        predictions=predictions,
        scope="twenty_to_twenty_five_m",
        fixed_confidence=0.5,
        cameras={"CVLab1"},
    )
    assert (overall["ground_truth"], overall["tp"], overall["fp"], overall["fn"]) == (3, 3, 2, 0)
    assert (eligible["ground_truth"], eligible["tp"], eligible["fp"], eligible["fn"]) == (2, 2, 2, 0)
    assert eligible["ignored_predictions_by_reason"] == {"matched_excluded_ground_truth": 1}
    assert (band["ground_truth"], band["tp"], band["fp"], band["fn"]) == (1, 1, 1, 0)
    assert band["ignored_predictions_by_reason"] == {
        "matched_excluded_ground_truth": 2,
        "projected_prediction_outside_20_25m": 1,
    }
    assert band["ap_101_point"] is not None


def test_pair_delta_direction_is_explicit() -> None:
    left = {"ground_truth": 10, "tp": 4, "fp": 2, "fn": 6, "precision": 0.666667, "recall": 0.4, "f1": 0.5, "ap_101_point": 0.4}
    right = {"ground_truth": 10, "tp": 7, "fp": 1, "fn": 3, "precision": 0.875, "recall": 0.7, "f1": 0.777778, "ap_101_point": 0.6}
    assert lane._metric_delta(left, right) == {
        "ground_truth": 0,
        "tp": 3,
        "fp": -1,
        "fn": -3,
        "precision": 0.208333,
        "recall": 0.3,
        "f1": 0.277778,
        "ap_101_point": 0.2,
    }


def test_duplicate_json_keys_rejected_before_semantics() -> None:
    with pytest.raises(lane.WildtrackEvaluationError, match="duplicate JSON key: profile"):
        lane._strict_json_bytes(b'{"profile":640,"profile":960}', context="fixture")


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonfinite_json_constants_rejected(constant: bytes) -> None:
    with pytest.raises(lane.WildtrackEvaluationError, match="non-JSON numeric constant"):
        lane._strict_json_bytes(b'{"confidence":' + constant + b"}", context="fixture")


def test_secure_reader_rejects_intermediate_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "value.json").write_text('{"outside":true}\n', encoding="utf-8")
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(lane.WildtrackEvaluationError, match="cannot securely open"):
        lane._read_regular(
            root / "redirect/value.json",
            project_root=root,
            maximum_bytes=1024,
            context="symlink fixture",
        )


def test_secure_reader_holds_parent_across_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    inside = root / "inside"
    outside = tmp_path / "outside"
    inside.mkdir(parents=True)
    outside.mkdir()
    payload = inside / "value.json"
    payload.write_text('{"origin":"inside"}\n', encoding="utf-8")
    (outside / "value.json").write_text('{"origin":"outside"}\n', encoding="utf-8")
    expected = lane._portable_pin(payload, project_root=root, context="swap fixture")
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "value.json" and dir_fd is not None and not swapped:
            swapped = True
            inside.rename(root / "inside-original")
            inside.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {racing_open})
    raw, observed = lane._read_regular(
        payload,
        project_root=root,
        maximum_bytes=1024,
        expected=expected,
        context="swap fixture",
    )
    assert swapped is True
    assert raw == b'{"origin":"inside"}\n'
    assert observed == expected


def test_complete_empty_prediction_coverage_is_accepted(tmp_path: Path, real_gt: lane.GroundTruthEvidence) -> None:
    project = _prediction_project(tmp_path)
    path, pin = _write_prediction_rows(project, _all_rows(real_gt))
    loaded = lane._load_prediction_rows(
        path,
        profile=640,
        expected_pin=pin,
        expected_model_id=lane.MODEL_ID,
        gt=real_gt,
        project_root=project,
    )
    assert len(loaded.frames) == 2800
    assert loaded.detections == ()


def test_missing_prediction_frame_fails_closed(tmp_path: Path, real_gt: lane.GroundTruthEvidence) -> None:
    project = _prediction_project(tmp_path)
    path, pin = _write_prediction_rows(project, _all_rows(real_gt)[:-1])
    with pytest.raises(lane.WildtrackEvaluationError, match="row count differs"):
        lane._load_prediction_rows(path, profile=640, expected_pin=pin, expected_model_id=lane.MODEL_ID, gt=real_gt, project_root=project)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("profile"), "schema mismatch"),
        (lambda row: row.__setitem__("profile", 960), "profile swap"),
        (lambda row: row.__setitem__("bbox_space", "network_input_pixel_xyxy"), "schema mismatch"),
        (lambda row: row["source_image"].__setitem__("sha256", "b" * 64), "source image pin differs"),
    ],
)
def test_prediction_identity_and_bbox_space_drift_fail_closed(
    tmp_path: Path,
    real_gt: lane.GroundTruthEvidence,
    mutation,
    message: str,
) -> None:
    project = _prediction_project(tmp_path)
    rows = _all_rows(real_gt)
    mutation(rows[0])
    path, pin = _write_prediction_rows(project, rows)
    with pytest.raises(lane.WildtrackEvaluationError, match=message):
        lane._load_prediction_rows(path, profile=640, expected_pin=pin, expected_model_id=lane.MODEL_ID, gt=real_gt, project_root=project)


def test_bbox_sum_outside_normalized_source_fails_closed(tmp_path: Path, real_gt: lane.GroundTruthEvidence) -> None:
    project = _prediction_project(tmp_path)
    rows = _all_rows(real_gt)
    rows[0]["detections"] = [{
        "prediction_id": "overflow",
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox_norm_xywh": [0.9, 0.1, 0.2, 0.2],
    }]
    path, pin = _write_prediction_rows(project, rows)
    with pytest.raises(lane.WildtrackEvaluationError, match="leaves source-normalized space"):
        lane._load_prediction_rows(path, profile=640, expected_pin=pin, expected_model_id=lane.MODEL_ID, gt=real_gt, project_root=project)


def test_duplicate_prediction_ids_fail_closed(tmp_path: Path, real_gt: lane.GroundTruthEvidence) -> None:
    project = _prediction_project(tmp_path)
    rows = _all_rows(real_gt)
    detection = {
        "prediction_id": "duplicate-id",
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox_norm_xywh": [0.1, 0.1, 0.1, 0.1],
    }
    rows[0]["detections"] = [detection]
    rows[1]["detections"] = [copy.deepcopy(detection)]
    path, pin = _write_prediction_rows(project, rows)
    with pytest.raises(lane.WildtrackEvaluationError, match="duplicate prediction id"):
        lane._load_prediction_rows(path, profile=640, expected_pin=pin, expected_model_id=lane.MODEL_ID, gt=real_gt, project_root=project)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("profile", 960), "cross-profile swap"),
        (lambda value: value["thresholds"].__setitem__("fixed_operating_confidence", 0.25), "threshold drift"),
        (lambda value: value["model"]["model_receipt"].__setitem__("sha256", "f" * 64), "wrong model receipt"),
        (lambda value: value["model"].__setitem__("source_model_artifact", {**value["model"]["source_model_artifact"], "sha256": "e" * 64}), "source model pin differs"),
        (lambda value: value["inputs"]["ground_truth"].__setitem__("sha256", "d" * 64), "GT input pins drifted"),
        (lambda value: value["outputs"]["predictions"].__setitem__("path", "validation/results/wildtrack/wrong/predictions.jsonl"), "prediction output path differs"),
    ],
)
def test_run_manifest_cross_document_drift_fails_closed(
    real_plan: dict,
    real_gt: lane.GroundTruthEvidence,
    mutate,
    message: str,
) -> None:
    value = _manifest(real_plan, real_gt)
    mutate(value)
    _refresh_fingerprint(value)
    with pytest.raises(lane.WildtrackEvaluationError, match=message):
        _manifest_contract(value, real_plan, real_gt)


def test_valid_manifest_contract_binds_all_pins(real_plan: dict, real_gt: lane.GroundTruthEvidence) -> None:
    value = _manifest(real_plan, real_gt)
    lane._schema_validate(value, lane.RUN_SCHEMA_NAME, project_root=ROOT, context="test run manifest")
    _manifest_contract(value, real_plan, real_gt)


def test_metric_delta_rejects_unpaired_ground_truth() -> None:
    left = {field: 0 for field in lane.DELTA_FIELDS}
    right = dict(left)
    left["ground_truth"] = 1
    right["ground_truth"] = 2
    with pytest.raises(lane.WildtrackEvaluationError, match="GT counts differ"):
        lane._metric_delta(left, right)
