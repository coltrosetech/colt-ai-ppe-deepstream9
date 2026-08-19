import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

import validation.site_distance_evaluator_v2 as evaluator_v2

from tests.test_site_distance_readiness_v2 import (
    _pair,
    _pin,
    _preflight,
    _profile_pair_fixture,
    _write_json,
)
from validation.site_distance_evaluator_v2 import (
    ATTEMPT_VERSION,
    FINAL_VERSION,
    INPUT_LEDGER_NAMES,
    _expected_result_ledger,
    _receipt_hash,
    evaluate_and_write,
    verify_evaluation_receipt,
    verify_receipt_self_hash,
)
from validation.site_distance_readiness_v2 import (
    _nominal_bin,
    _schema_validate,
    _sha256_json,
)


@pytest.mark.parametrize(
    ("distance", "expected_bin"),
    [
        (19.999999, None),
        (20.0, "20-21m"),
        (20.999999, "20-21m"),
        (21.0, "21-22m"),
        (22.0, "22-23m"),
        (23.0, "23-24m"),
        (24.0, "24-25m"),
        (25.0, "24-25m"),
        (25.000001, None),
    ],
)
def test_every_inclusive_v2_bin_edge_has_one_exact_assignment(
    distance, expected_bin
):
    assigned = _nominal_bin(distance)
    assert (assigned[0] if assigned is not None else None) == expected_bin


def _add_exact_25m_and_unambiguous_endpoint(paths: dict) -> None:
    ground_truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    endpoint_frame = ground_truth["frames"][4]
    exact = endpoint_frame["persons"][0]
    exact["bbox_pixel_xywh"] = [940, 480, 40, 100]
    exact["independent_event_id"] = "event-exact-25"

    # A second spatially separate 24.8 m instance keeps the final bin's
    # unambiguous quota valid while the exact-25 sample is correctly ambiguous
    # under the calibration's nonzero uncertainty.
    center_x = 1100.0
    ground_x = (center_x - 960.0) * 0.02
    ground_y = math.sqrt(24.8**2 - ground_x**2)
    bottom_y = 1080.0 - ground_y / 0.05
    endpoint_frame["persons"].append(
        {
            "object_id": "person-final-unambiguous",
            "independent_event_id": "event-final-unambiguous",
            "bbox_pixel_xywh": [1080, bottom_y - 100, 40, 100],
            "ground_contact_uncertainty_px": 0.0,
            "additional_distance_uncertainty_m": 0.0,
            "occlusion_fraction": 0.0,
            "truncated": False,
            "ignored": False,
        }
    )
    _write_json(paths["ground_truth"], ground_truth)
    for manifest_path in paths["manifests"].values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ground_truth"] = _pin(paths["ground_truth"])
        _write_json(manifest_path, manifest)


def _write_gt_predictions(paths: dict, confidence: float) -> None:
    media = json.loads(paths["media"].read_text(encoding="utf-8"))
    ground_truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    gt_by_frame = {frame["frame_index"]: frame for frame in ground_truth["frames"]}
    for profile, manifest_path in paths["manifests"].items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = Path(manifest["predictions"]["path"])
        records = []
        for frame in media["frames"]:
            detections = []
            for person in gt_by_frame[frame["frame_index"]]["persons"]:
                if person["ignored"]:
                    continue
                x, y, width, height = person["bbox_pixel_xywh"]
                detections.append(
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": confidence,
                        "bbox_norm_xywh": [
                            x / frame["image_width"],
                            y / frame["image_height"],
                            width / frame["image_width"],
                            height / frame["image_height"],
                        ],
                    }
                )
            records.append(
                {
                    "schema_version": "deepsafe.person-detections/v1",
                    "sequence_id": ground_truth["sequence_id"],
                    "frame_index": frame["frame_index"],
                    "image_width": frame["image_width"],
                    "image_height": frame["image_height"],
                    "timestamp_ns": frame["timestamp_ns"],
                    "source_uri": "file:///validation/inputs/distance-25m/site-source.mp4",
                    "model_id": manifest["cross_profile_invariants"]["base_model_id"],
                    "detections": detections,
                }
            )
        prediction_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest["predictions"] = _pin(prediction_path)
        _write_json(manifest_path, manifest)


def _stored_receipts(paths: dict, tmp_path: Path) -> tuple[Path, Path]:
    preflight = _write_json(tmp_path / "preflight-receipt.json", _preflight(paths))
    pair = _write_json(tmp_path / "profile-pair-receipt.json", _pair(paths))
    return preflight, pair


def _evaluate(
    paths: dict,
    tmp_path: Path,
    preflight: Path,
    pair: Path,
    suffix: str = "primary",
):
    attempt = tmp_path / f"attempt-{suffix}.json"
    final = tmp_path / f"final-{suffix}.json"
    result = evaluate_and_write(
        camera_configuration_path=paths["camera"],
        calibration_path=paths["calibration"],
        media_frame_ledger_path=paths["media"],
        ground_truth_path=paths["ground_truth"],
        acceptance_path=paths["acceptance"],
        preflight_receipt_path=preflight,
        profile_pair_receipt_path=pair,
        profile_640_manifest_path=paths["manifests"][640],
        profile_960_manifest_path=paths["manifests"][960],
        attempt_output_path=attempt,
        final_output_path=final,
    )
    return result, attempt, final


def _reseal(value: dict, *, rebuild_results: bool = False) -> dict:
    value = deepcopy(value)
    value.pop("receipt_sha256", None)
    if rebuild_results:
        ledger = _expected_result_ledger(value)
        value["integrity"]["result_ledger"] = ledger
        value["integrity"]["result_ledger_sha256"] = _sha256_json(ledger)
    value["receipt_sha256"] = _receipt_hash(value)
    return value


def test_perfect_pair_includes_exact_25m_and_writes_verified_receipts(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _add_exact_25m_and_unambiguous_endpoint(paths)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)

    (attempt_value, final_value), attempt, final = _evaluate(
        paths, tmp_path, preflight, pair
    )

    assert attempt_value["schema_version"] == ATTEMPT_VERSION
    assert attempt_value["status"] == "accepted"
    assert final_value is not None
    assert final_value["schema_version"] == FINAL_VERSION
    assert final_value["paired_acceptance"]["status"] == "pass"
    assert final_value["profiles"]["640"]["overall"] == {
        "ground_truth": 6,
        "serialized_predictions_at_or_above_confidence": 6,
        "evaluated_predictions": 6,
        "ignored_predictions": 0,
        "tp": 6,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "ap_101_point": 1.0,
        "ap_serialized_predictions": 6,
    }
    final_bin = final_value["profiles"]["960"]["bins"][-1]
    assert final_bin["bin_id"] == "24-25m"
    assert final_bin["metrics"]["ground_truth"] == 2
    assert final_bin["metrics"]["tp"] == 2
    quota = final_value["quota_recomputation"]
    assert quota["exact_25m_instances"] == 1
    assert quota["bins"][-1]["instances"] == 2
    assert quota["bins"][-1]["ambiguous_instances"] == 1
    assert quota["bins"][-1]["unambiguous_independent_events"] == 1
    assert quota["matches_preflight_receipt"] is True
    assert tuple(
        row["name"] for row in final_value["integrity"]["input_ledger"]
    ) == INPUT_LEDGER_NAMES
    assert len(final_value["integrity"]["result_ledger"]) == 16
    assert verify_evaluation_receipt(attempt)["status"] == "accepted"
    assert verify_evaluation_receipt(final)["status"] == "complete"
    _schema_validate(
        final_value,
        "site-distance-evaluation-final-v2.schema.json",
        "test inclusive final",
    )


def test_ap_uses_confidence_zero_but_threshold_metrics_fail_closed(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.0)
    preflight, pair = _stored_receipts(paths, tmp_path)

    (attempt_value, final_value), attempt, final = _evaluate(
        paths, tmp_path, preflight, pair
    )

    assert final_value is None
    assert attempt_value["status"] == "acceptance_failed"
    assert attempt.exists()
    assert not final.exists()
    metrics = attempt_value["profiles"]["640"]["overall"]
    assert metrics["serialized_predictions_at_or_above_confidence"] == 0
    assert metrics["tp"] == 0
    assert metrics["recall"] == 0.0
    assert metrics["ap_serialized_predictions"] == 5
    assert metrics["ap_101_point"] == 1.0
    assert "profile_640_recall_below_0.8" in attempt_value[
        "paired_acceptance"
    ]["failure_reasons"]
    assert verify_evaluation_receipt(attempt)["status"] == "acceptance_failed"


def test_wrong_bin_prediction_is_fp_not_suppressed_by_neighbor_gt(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    # Frame 1 GT is at 21.5 m. Moving the otherwise high-IoU box 12 pixels
    # downward assigns its bottom-centre to 20.9 m (the preceding bin).
    for profile, manifest_path in paths["manifests"].items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = Path(manifest["predictions"]["path"])
        records = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
        ]
        bbox = records[1]["detections"][0]["bbox_norm_xywh"]
        bbox[1] += 12 / records[1]["image_height"]
        prediction_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest["predictions"] = _pin(prediction_path)
        _write_json(manifest_path, manifest)
    preflight, pair = _stored_receipts(paths, tmp_path)

    (_, final_value), _, _ = _evaluate(paths, tmp_path, preflight, pair)
    assert final_value is not None
    overall = final_value["profiles"]["640"]["overall"]
    assert overall["tp"] == 5
    assert overall["fp"] == 0
    bins = {
        row["bin_id"]: row["metrics"]
        for row in final_value["profiles"]["640"]["bins"]
    }
    assert bins["20-21m"]["tp"] == 1
    assert bins["20-21m"]["fp"] == 1
    assert bins["20-21m"]["ignored_predictions"] == 0
    assert bins["21-22m"]["fn"] == 1


def test_stored_profile_pair_receipt_must_equal_live_recomputation(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    preflight, pair = _stored_receipts(paths, tmp_path)
    value = json.loads(pair.read_text(encoding="utf-8"))
    value["shared_binding_id"] = "tampered-binding"
    _write_json(pair, value)

    with pytest.raises(ValueError, match="profile-pair receipt differs"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()
    assert not (tmp_path / "final-primary.json").exists()


def test_duplicate_ground_truth_object_id_is_rejected_before_attempt(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    ground_truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    ground_truth["frames"][0]["persons"].append(
        deepcopy(ground_truth["frames"][0]["persons"][0])
    )
    _write_json(paths["ground_truth"], ground_truth)
    for manifest_path in paths["manifests"].values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ground_truth"] = _pin(paths["ground_truth"])
        _write_json(manifest_path, manifest)
    preflight = _write_json(tmp_path / "preflight-receipt.json", {})
    pair = _write_json(tmp_path / "profile-pair-receipt.json", {})

    with pytest.raises(ValueError, match="repeats object_id"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


def test_boolean_cannot_substitute_for_numeric_acceptance_threshold(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    acceptance = json.loads(paths["acceptance"].read_text(encoding="utf-8"))
    acceptance["rules"][0]["threshold"] = True
    _write_json(paths["acceptance"], acceptance)
    for manifest_path in paths["manifests"].values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["acceptance"] = _pin(paths["acceptance"])
        _write_json(manifest_path, manifest)
    preflight = _write_json(tmp_path / "preflight-receipt.json", {})
    pair = _write_json(tmp_path / "profile-pair-receipt.json", {})

    with pytest.raises(ValueError, match="threshold.*number"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


def test_prediction_tamper_after_manifest_and_pair_is_rejected(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    preflight, pair = _stored_receipts(paths, tmp_path)
    manifest = json.loads(paths["manifests"][640].read_text(encoding="utf-8"))
    prediction_path = Path(manifest["predictions"]["path"])
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="(byte count|SHA-256) does not match"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


@pytest.mark.parametrize(
    "artifact_key", ["ground_truth", "calibration", "media", "acceptance"]
)
def test_core_evidence_pin_tamper_is_rejected_before_attempt(
    tmp_path, artifact_key
):
    paths = _profile_pair_fixture(tmp_path)
    preflight, pair = _stored_receipts(paths, tmp_path)
    artifact = paths[artifact_key]
    artifact.write_text(
        artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="(does not match|differs)"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


def test_self_hash_and_result_ledger_detect_semantic_tamper(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (_, final_value), _, final = _evaluate(paths, tmp_path, preflight, pair)
    assert final_value is not None

    tampered = json.loads(final.read_text(encoding="utf-8"))
    tampered["profiles"]["640"]["overall"]["tp"] -= 1
    _write_json(final, tampered)
    with pytest.raises(ValueError, match="self-hash does not match"):
        verify_evaluation_receipt(final)
    with pytest.raises(ValueError, match="self-hash does not match"):
        verify_receipt_self_hash(tampered, "tampered final")


def test_attempt_and_final_paths_are_immutable(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    _evaluate(paths, tmp_path, preflight, pair)

    with pytest.raises(FileExistsError, match="immutable attempt already exists"):
        _evaluate(paths, tmp_path, preflight, pair)


def test_rehashed_forged_attempt_is_rejected_by_live_semantic_replay(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (attempt_value, _), attempt, _ = _evaluate(paths, tmp_path, preflight, pair)

    forged = deepcopy(attempt_value)
    forged["profiles"]["640"]["overall"]["tp"] -= 1
    forged = _reseal(forged, rebuild_results=True)
    _write_json(attempt, forged)

    with pytest.raises(ValueError, match="live semantic replay"):
        verify_evaluation_receipt(attempt)


def test_rehashed_final_cannot_change_unledgered_evaluation_config(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (_, final_value), _, final = _evaluate(paths, tmp_path, preflight, pair)
    assert final_value is not None

    forged = deepcopy(final_value)
    forged["evaluation_config"]["confidence_threshold"] = 0.2
    _write_json(final, _reseal(forged))

    with pytest.raises(ValueError, match="live semantic replay"):
        verify_evaluation_receipt(final)


def test_duplicate_receipt_key_is_rejected_even_when_semantics_are_identical(
    tmp_path,
):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    _, attempt, _ = _evaluate(paths, tmp_path, preflight, pair)

    content = attempt.read_text(encoding="utf-8")
    marker = '  "status": "accepted",'
    assert content.count(marker) == 1
    attempt.write_text(content.replace(marker, f"{marker}\n{marker}"), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key 'status'"):
        verify_evaluation_receipt(attempt)


def test_nonfinite_receipt_number_is_rejected_before_schema_validation(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (attempt_value, _), attempt, _ = _evaluate(paths, tmp_path, preflight, pair)

    forged = deepcopy(attempt_value)
    forged.pop("receipt_sha256")
    forged["profiles"]["640"]["prediction_filter"][
        "minimum_serialized_person_confidence"
    ] = float("nan")
    ledger = _expected_result_ledger(forged)
    forged["integrity"]["result_ledger"] = ledger
    forged["integrity"]["result_ledger_sha256"] = _sha256_json(ledger)
    canonical = json.dumps(
        forged, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    forged["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json(attempt, forged)

    with pytest.raises(ValueError, match="non-finite JSON constant 'NaN'"):
        verify_evaluation_receipt(attempt)


def test_duplicate_prediction_json_key_is_rejected_before_evaluation(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    manifest_path = paths["manifests"][640]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["predictions"]["path"])
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    marker = f'"model_id": {json.dumps(record["model_id"])}'
    assert lines[0].count(marker) == 1
    lines[0] = lines[0].replace(marker, f"{marker}, {marker}")
    prediction_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["predictions"] = _pin(prediction_path)
    _write_json(manifest_path, manifest)
    preflight, pair = _stored_receipts(paths, tmp_path)

    with pytest.raises(ValueError, match="duplicate JSON object key 'model_id'"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


def test_dangling_symlink_cannot_redirect_immutable_output(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    attempt = tmp_path / "attempt-primary.json"
    redirected = tmp_path / "redirected-attempt.json"
    attempt.symlink_to(redirected)

    with pytest.raises(FileExistsError, match="immutable attempt already exists"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert attempt.is_symlink()
    assert not redirected.exists()


def test_receipt_rejects_symlink_alias_in_rehashed_input_ledger(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (attempt_value, _), attempt, _ = _evaluate(paths, tmp_path, preflight, pair)
    alias = tmp_path / "acceptance-alias.json"
    alias.symlink_to(paths["acceptance"])

    forged = deepcopy(attempt_value)
    row = next(
        item
        for item in forged["integrity"]["input_ledger"]
        if item["name"] == "acceptance"
    )
    row["pin"]["path"] = str(alias)
    forged["integrity"]["input_ledger_sha256"] = _sha256_json(
        forged["integrity"]["input_ledger"]
    )
    _write_json(attempt, _reseal(forged))

    with pytest.raises(ValueError, match="non-canonical path or pin"):
        verify_evaluation_receipt(attempt)


def test_prediction_change_during_metrics_is_rejected_before_write(
    tmp_path, monkeypatch
):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    original = evaluator_v2._profile_result

    def mutate_after_profile(**kwargs):
        result = original(**kwargs)
        if kwargs["profile"] == 960:
            prediction_path = kwargs["prediction_path"]
            prediction_path.write_text(
                prediction_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(evaluator_v2, "_profile_result", mutate_after_profile)
    with pytest.raises(ValueError, match="input changed during metric computation"):
        _evaluate(paths, tmp_path, preflight, pair)
    assert not (tmp_path / "attempt-primary.json").exists()


def test_receipt_change_during_verification_is_rejected(
    tmp_path, monkeypatch
):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    _, attempt, _ = _evaluate(paths, tmp_path, preflight, pair)
    original = evaluator_v2._semantic_replay

    def mutate_after_replay(payload, inputs):
        original(payload, inputs)
        attempt.write_text(
            attempt.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

    monkeypatch.setattr(evaluator_v2, "_semantic_replay", mutate_after_replay)
    with pytest.raises(ValueError, match="receipt changed during verification"):
        verify_evaluation_receipt(attempt)


def test_final_schema_rejects_unknown_nested_metric_field(tmp_path):
    paths = _profile_pair_fixture(tmp_path)
    _write_gt_predictions(paths, confidence=0.9)
    preflight, pair = _stored_receipts(paths, tmp_path)
    (_, final_value), _, _ = _evaluate(paths, tmp_path, preflight, pair)
    assert final_value is not None
    forged = deepcopy(final_value)
    forged["profiles"]["640"]["overall"]["unexpected"] = 1

    with pytest.raises(
        ValueError, match="(?i)(additional properties|unknown fields)"
    ):
        _schema_validate(
            forged,
            "site-distance-evaluation-final-v2.schema.json",
            "forged final",
        )
