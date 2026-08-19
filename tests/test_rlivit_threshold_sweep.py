from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evaluation.rlivit import (
    Box,
    FrameGroundTruth,
    Prediction,
    SequenceGroundTruth,
    SequencePredictions,
    canonical_sha256,
    evaluate_sequences,
)
from validation import rlivit_paired_audit as paired
from validation import rlivit_threshold_sweep as sweep
from validation.rlivit_paired_audit import ReplayEvidence, RLiViTPairedAuditError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "validation/schemas/rlivit-threshold-sweep-v1.schema.json"
REAL_REPORT = (
    ROOT / "validation/results/rlivit/threshold-sweep/threshold-sweep.json"
)
REAL_RECEIPT = ROOT / "validation/results/rlivit/threshold-sweep/receipt.json"


def _pin(path: str, digest: str = "a") -> dict[str, object]:
    return {"path": path, "size_bytes": 1, "sha256": digest * 64}


def _prediction(index: int, confidence: float, box: Box) -> Prediction:
    return Prediction(
        sequence_id="002",
        video_frame_index=0,
        detection_index=index,
        confidence=confidence,
        box=box,
    )


def _fake_evidence() -> ReplayEvidence:
    frame = FrameGroundTruth(
        sequence_id="002",
        video_frame_index=0,
        source_frame_index=2,
        width=1280,
        height=720,
        daytime="day",
        location="0",
        persons=(Box(0, 0, 20, 20), Box(100, 100, 140, 160)),
    )
    ground_truth = SequenceGroundTruth(
        sequence_id="002",
        daytime="day",
        location="0",
        frames=(frame,),
        ground_truth_pin=_pin("fixture/gt.jsonl", "1"),
        frame_map_pin=_pin("fixture/map.jsonl", "2"),
        live_sources_fingerprint_sha256="3" * 64,
    )
    predictions_640 = SequencePredictions(
        sequence_id="002",
        frames=((0, 1280, 720),),
        detections=(
            _prediction(0, 0.95, Box(0, 0, 20, 20)),
            _prediction(1, 0.90, Box(1, 1, 19, 19)),
            _prediction(2, 0.85, Box(100, 100, 140, 160)),
            _prediction(3, 0.20, Box(500, 500, 520, 520)),
        ),
        predictions_pin=_pin("fixture/predictions-640.jsonl", "4"),
    )
    predictions_960 = SequencePredictions(
        sequence_id="002",
        frames=((0, 1280, 720),),
        detections=(
            _prediction(0, 0.95, Box(0, 0, 20, 20)),
            _prediction(1, 0.80, Box(100, 100, 140, 160)),
            _prediction(2, 0.10, Box(500, 500, 520, 520)),
        ),
        predictions_pin=_pin("fixture/predictions-960.jsonl", "5"),
    )
    gt_values = [ground_truth]
    by_profile = {
        640: {"002": predictions_640},
        960: {"002": predictions_960},
    }
    evaluations = {
        profile: evaluate_sequences(
            gt_values,
            [by_profile[profile]["002"]],
            iou_threshold=0.5,
            confidence_threshold=0.25,
        )
        for profile in (640, 960)
    }
    return ReplayEvidence(
        batch_receipt_pin=_pin("fixture/batch-receipt.json", "6"),
        aggregate_pin=_pin("fixture/batch-aggregate.json", "7"),
        batch_receipt_fingerprint="8" * 64,
        aggregate_fingerprint="9" * 64,
        campaign_nonce="b" * 64,
        verified_unique_files=7,
        verified_pin_references=11,
        ground_truth={"002": ground_truth},
        predictions=by_profile,
        evaluations=evaluations,
        source_pins={},
        job_receipt_pins={},
    )


def _expected_overall(evidence: ReplayEvidence, profile: int, threshold: float):
    evaluated = evaluate_sequences(
        [evidence.ground_truth["002"]],
        [evidence.predictions[profile]["002"]],
        iou_threshold=0.5,
        confidence_threshold=threshold,
    )["overall"]
    return {
        "ground_truth": evaluated["ground_truth"],
        "selected_predictions": evaluated[
            "serialized_predictions_at_or_above_confidence"
        ],
        "tp": evaluated["tp"],
        "fp": evaluated["fp"],
        "fn": evaluated["fn"],
        "precision": evaluated["precision"],
        "recall": evaluated["recall"],
        "f1": evaluated["f1"],
    }


def test_sweep_has_exact_universal_grid_and_baseline_delta_direction() -> None:
    evidence = _fake_evidence()
    points = sweep.build_sweep_points(evidence)

    assert len(points) == 101
    assert [item["threshold_hundredths"] for item in points] == list(range(101))
    assert [item["confidence_threshold"] for item in points] == [
        value / 100 for value in range(101)
    ]
    assert points[25]["profiles"]["640"] == _expected_overall(
        evidence, 640, 0.25
    )
    assert points[25]["profiles"]["960"] == _expected_overall(
        evidence, 960, 0.25
    )
    assert points[25]["delta_960_minus_640"]["fp"] == -1
    assert points[25]["delta_960_minus_640"]["precision"] > 0


@pytest.mark.parametrize("threshold_hundredths", [0, 20, 25, 80, 85, 90, 95, 100])
@pytest.mark.parametrize("profile", [640, 960])
def test_event_prefix_projection_exactly_matches_authoritative_evaluator(
    profile: int,
    threshold_hundredths: int,
) -> None:
    evidence = _fake_evidence()
    point = sweep.build_sweep_points(evidence)[threshold_hundredths]
    assert point["profiles"][str(profile)] == _expected_overall(
        evidence,
        profile,
        threshold_hundredths / 100,
    )


def test_threshold_inclusion_is_greater_than_or_equal() -> None:
    points = sweep.build_sweep_points(_fake_evidence())
    # The 0.90 duplicate detection remains selected at exactly 0.90 and drops
    # only at 0.91.
    assert points[90]["profiles"]["640"]["selected_predictions"] == 2
    assert points[90]["profiles"]["640"]["fp"] == 1
    assert points[91]["profiles"]["640"]["selected_predictions"] == 1
    assert points[91]["profiles"]["640"]["fp"] == 0


def test_diagnostic_selection_uses_exact_counts_and_never_claims_deployment() -> None:
    points = sweep.build_sweep_points(_fake_evidence())
    for profile in (640, 960):
        selected = sweep.select_diagnostic_points(points, profile)
        maximum = selected["maximum_f1"]
        assert maximum["status"] == "selected_test_set_diagnostic"
        assert maximum["point"]["role"] == sweep.DIAGNOSTIC_ROLE
        assert maximum["point"]["threshold_hundredths"] == min(
            maximum["exact_tied_threshold_hundredths"]
        )
        for constrained in selected["precision_constrained"]:
            if constrained["point"] is None:
                assert constrained["status"] == "no_grid_point_satisfies_constraint"
                continue
            metrics = constrained["point"]["metrics"]
            exact_precision = metrics["tp"] / (metrics["tp"] + metrics["fp"])
            assert exact_precision >= constrained["minimum_precision"]
            assert constrained["point"]["role"] == sweep.DIAGNOSTIC_ROLE


def test_report_keeps_ap_outside_fixed_threshold_points() -> None:
    report = sweep.build_report(_fake_evidence())
    assert report["interpretation"] == {
        "dataset_role": "R-LiViT official test split",
        "result_role": sweep.DIAGNOSTIC_ROLE,
        "deployment_threshold_selected": False,
        "calibration_performed": False,
        "acceptance_policy_decision": False,
        "leakage_warning": (
            "Do not choose or claim a production threshold from these test-set "
            "diagnostics; use an independent validation/calibration split."
        ),
    }
    assert report["ap_reference"]["threshold_independent_within_sweep"] is True
    assert report["lineage"]["raw_input_pins"]["ground_truth"]["002"][
        "path"
    ] == "data/derived/r-livit/materialized-v1/fixture/gt.jsonl"
    assert report["lineage"]["raw_input_pins"]["frame_maps"]["002"][
        "path"
    ] == "data/derived/r-livit/materialized-v1/fixture/map.jsonl"
    assert all(
        "ap_101_point" not in metric
        for point in report["sweep"]["points"]
        for metric in [
            point["profiles"]["640"],
            point["profiles"]["960"],
            point["delta_960_minus_640"],
        ]
    )


def test_output_is_byte_deterministic_for_identical_replayed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _fake_evidence()
    monkeypatch.setattr(sweep, "replay_completed_batch", lambda _path: evidence)
    monkeypatch.setattr(
        sweep,
        "validate_sweep_report",
        lambda _report, **_kwargs: None,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    sweep.build_threshold_sweep(batch_receipt_path=Path("ignored"), output_dir=first)
    sweep.build_threshold_sweep(batch_receipt_path=Path("ignored"), output_dir=second)

    assert (first / "threshold-sweep.json").read_bytes() == (
        second / "threshold-sweep.json"
    ).read_bytes()
    assert (first / "receipt.json").read_bytes() == (
        second / "receipt.json"
    ).read_bytes()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"schema_version":"x","status":"a","status":"b"}\n', "duplicate JSON key"),
        ('{"schema_version":"x","confidence":NaN}\n', "non-JSON numeric constant"),
    ],
)
def test_build_reuses_strict_batch_receipt_json_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    message: str,
) -> None:
    monkeypatch.setattr(paired, "PROJECT_ROOT", tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(payload, encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(RLiViTPairedAuditError, match=message):
        sweep.build_threshold_sweep(
            batch_receipt_path=receipt,
            output_dir=output,
        )
    assert not output.exists()


def test_build_reuses_no_symlink_openat_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "receipt.json").write_text('{"outside":true}\n', encoding="utf-8")
    (root / "receipt.json").symlink_to(outside / "receipt.json")
    monkeypatch.setattr(paired, "PROJECT_ROOT", root)
    output = root / "output"

    with pytest.raises(RLiViTPairedAuditError, match="cannot securely open"):
        sweep.build_threshold_sweep(
            batch_receipt_path=root / "receipt.json",
            output_dir=output,
        )
    assert not output.exists()


def test_batch_receipt_path_outside_verified_root_fails_closed(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema_version":"irrelevant"}\n', encoding="utf-8")
    with pytest.raises(RLiViTPairedAuditError, match="file escapes root"):
        sweep.build_threshold_sweep(
            batch_receipt_path=receipt,
            output_dir=tmp_path / "output",
        )


def test_build_rejects_output_ancestor_symlink_before_creating_parent(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(sweep.RLiViTThresholdSweepError, match="symlink component"):
        sweep.build_threshold_sweep(
            batch_receipt_path=Path("not-reached.json"),
            output_dir=linked / "new-parent" / "sweep",
        )
    assert not (outside / "new-parent").exists()


def test_schema_is_valid_draft_2020_12() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_real_output_validates_and_rejects_deployment_or_ap_point_drift() -> None:
    if not REAL_REPORT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        json.loads(SCHEMA.read_text(encoding="utf-8"))
    )
    report = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    validator.validate(report)

    deployment_claim = copy.deepcopy(report)
    deployment_claim["interpretation"]["deployment_threshold_selected"] = True
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(deployment_claim)

    false_ap = copy.deepcopy(report)
    false_ap["sweep"]["points"][25]["profiles"]["640"]["ap_101_point"] = 0.9
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(false_ap)

    traversal = copy.deepcopy(report)
    traversal["lineage"]["batch_receipt"]["path"] = "../escape.json"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(traversal)

    wrong_job = copy.deepcopy(report)
    jobs = wrong_job["lineage"]["job_receipts"]
    jobs["rlivit:999:640"] = jobs.pop(next(iter(jobs)))
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(wrong_job)


def test_real_output_and_receipt_fingerprints_are_self_consistent() -> None:
    if not REAL_REPORT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    report = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    receipt = json.loads(REAL_RECEIPT.read_text(encoding="utf-8"))
    fingerprint = report.pop("fingerprint_sha256")
    assert canonical_sha256(report) == fingerprint
    assert receipt["sweep_fingerprint_sha256"] == fingerprint
    assert receipt["sweep"]["size_bytes"] == REAL_REPORT.stat().st_size
    assert receipt["sweep"]["sha256"] == hashlib.sha256(
        REAL_REPORT.read_bytes()
    ).hexdigest()
    receipt_fingerprint = receipt.pop("fingerprint_sha256")
    assert canonical_sha256(receipt) == receipt_fingerprint


def test_real_output_replays_all_count_rate_delta_and_selection_invariants() -> None:
    if not REAL_REPORT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    report = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    points = report["sweep"]["points"]
    assert len(points) == 101
    for threshold, point in enumerate(points):
        assert point["threshold_hundredths"] == threshold
        assert point["confidence_threshold"] == threshold / 100
        for profile in (640, 960):
            metrics = point["profiles"][str(profile)]
            assert metrics["tp"] + metrics["fn"] == metrics["ground_truth"] == 4318
            assert metrics["tp"] + metrics["fp"] == metrics["selected_predictions"]
            expected_precision = (
                round(metrics["tp"] / metrics["selected_predictions"], 6)
                if metrics["selected_predictions"]
                else 0.0
            )
            expected_recall = round(metrics["tp"] / 4318, 6)
            expected_f1 = (
                round(
                    2
                    * expected_precision
                    * expected_recall
                    / (expected_precision + expected_recall),
                    6,
                )
                if expected_precision + expected_recall
                else 0.0
            )
            assert metrics["precision"] == expected_precision
            assert metrics["recall"] == expected_recall
            assert metrics["f1"] == expected_f1
        assert point["delta_960_minus_640"] == sweep._metric_delta(
            point["profiles"]["640"],
            point["profiles"]["960"],
        )
    assert report["baseline_crosscheck"]["profiles"] == points[25]["profiles"]
    for profile in (640, 960):
        metrics = [point["profiles"][str(profile)] for point in points]
        for field in ("selected_predictions", "tp", "fp"):
            assert all(
                left[field] >= right[field]
                for left, right in zip(metrics, metrics[1:])
            )
        assert all(
            left["fn"] <= right["fn"]
            for left, right in zip(metrics, metrics[1:])
        )
    assert report["sweep"]["diagnostic_selections"] == {
        str(profile): sweep.select_diagnostic_points(points, profile)
        for profile in (640, 960)
    }


def test_semantic_validator_rejects_cross_field_pocs_schema_cannot_express() -> None:
    if not REAL_REPORT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    original = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    malformed: list[dict] = []

    duplicate_grid = copy.deepcopy(original)
    duplicate_grid["sweep"]["points"][1] = copy.deepcopy(
        duplicate_grid["sweep"]["points"][0]
    )
    malformed.append(duplicate_grid)

    threshold_mismatch = copy.deepcopy(original)
    threshold_mismatch["sweep"]["points"][7]["confidence_threshold"] = 0.08
    malformed.append(threshold_mismatch)

    count_mismatch = copy.deepcopy(original)
    count_mismatch["sweep"]["points"][25]["profiles"]["640"]["fn"] += 1
    malformed.append(count_mismatch)

    nonfinite = copy.deepcopy(original)
    nonfinite["sweep"]["points"][25]["profiles"]["640"]["precision"] = float(
        "nan"
    )
    malformed.append(nonfinite)

    delta_mismatch = copy.deepcopy(original)
    delta_mismatch["sweep"]["points"][25]["delta_960_minus_640"]["tp"] += 1
    malformed.append(delta_mismatch)

    selection_mismatch = copy.deepcopy(original)
    selection_mismatch["sweep"]["diagnostic_selections"]["640"]["maximum_f1"][
        "point"
    ] = {
        "role": sweep.DIAGNOSTIC_ROLE,
        "threshold_hundredths": 25,
        "confidence_threshold": 0.25,
        "metrics": copy.deepcopy(
            selection_mismatch["sweep"]["points"][25]["profiles"]["640"]
        ),
    }
    malformed.append(selection_mismatch)

    wrong_job = copy.deepcopy(original)
    jobs = wrong_job["lineage"]["job_receipts"]
    jobs["rlivit:999:640"] = jobs.pop(next(iter(jobs)))
    malformed.append(wrong_job)

    for candidate in malformed:
        with pytest.raises(
            (sweep.RLiViTThresholdSweepError, RLiViTPairedAuditError)
        ):
            sweep.validate_sweep_report(candidate)


def test_artifact_verifier_rejects_package_tamper_and_output_symlink(
    tmp_path: Path,
) -> None:
    if not REAL_REPORT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    verified = sweep.verify_threshold_sweep_artifact(
        REAL_REPORT.parent,
        replay_raw_inputs=False,
    )
    assert verified["status"] == "complete_live_raw_replay"

    copied = tmp_path / "copied"
    shutil.copytree(REAL_REPORT.parent, copied)
    report = json.loads((copied / "threshold-sweep.json").read_text(encoding="utf-8"))
    report["interpretation"]["deployment_threshold_selected"] = True
    (copied / "threshold-sweep.json").write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RLiViTPairedAuditError, match="live pin differs"):
        sweep.verify_threshold_sweep_artifact(copied, replay_raw_inputs=False)

    target = tmp_path / "target"
    shutil.copytree(REAL_REPORT.parent, target)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(sweep.RLiViTThresholdSweepError, match="symlink component"):
        sweep.verify_threshold_sweep_artifact(linked, replay_raw_inputs=False)


@pytest.mark.parametrize("mutation", ["bytes_alias", "extra_field"])
def test_artifact_verifier_requires_exact_receipt_sweep_pin_shape(
    tmp_path: Path,
    mutation: str,
) -> None:
    if not REAL_REPORT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    copied = tmp_path / "copied"
    shutil.copytree(REAL_REPORT.parent, copied)
    receipt_path = copied / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "bytes_alias":
        receipt["sweep"]["bytes"] = receipt["sweep"].pop("size_bytes")
    else:
        receipt["sweep"]["unexpected"] = "not-closed"
    receipt.pop("fingerprint_sha256")
    receipt["fingerprint_sha256"] = canonical_sha256(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        sweep.RLiViTThresholdSweepError,
        match="receipt.sweep pin fields differ",
    ):
        sweep.verify_threshold_sweep_artifact(copied, replay_raw_inputs=False)


def test_real_artifact_verifier_replays_all_final_batch_raw_inputs() -> None:
    if not REAL_REPORT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    verified = sweep.verify_threshold_sweep_artifact(
        REAL_REPORT.parent,
        replay_raw_inputs=True,
    )
    assert verified["status"] == "complete_live_raw_replay"


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        (
            "receipt.json",
            '{"schema_version":"x","status":"a","status":"b"}\n',
            "duplicate JSON key",
        ),
        (
            "threshold-sweep.json",
            '{"schema_version":"x","confidence":NaN}\n',
            "non-JSON numeric constant",
        ),
        (
            "threshold-sweep.json",
            '{"schema_version":"x","confidence":Infinity}\n',
            "non-JSON numeric constant",
        ),
    ],
)
def test_artifact_verifier_rejects_non_strict_json(
    tmp_path: Path,
    filename: str,
    payload: str,
    message: str,
) -> None:
    if not REAL_REPORT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real threshold sweep has not been generated")
    copied = tmp_path / "copied"
    shutil.copytree(REAL_REPORT.parent, copied)
    payload_bytes = payload.encode("utf-8")
    (copied / filename).write_bytes(payload_bytes)
    if filename == "threshold-sweep.json":
        # Keep the outer byte pin valid so the strict JSON parser, rather than
        # the earlier pin check, is the layer exercised by this regression.
        receipt = json.loads((copied / "receipt.json").read_text(encoding="utf-8"))
        receipt["sweep"]["size_bytes"] = len(payload_bytes)
        receipt["sweep"]["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
        receipt.pop("fingerprint_sha256")
        receipt["fingerprint_sha256"] = canonical_sha256(receipt)
        (copied / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(RLiViTPairedAuditError, match=message):
        sweep.verify_threshold_sweep_artifact(copied, replay_raw_inputs=False)
