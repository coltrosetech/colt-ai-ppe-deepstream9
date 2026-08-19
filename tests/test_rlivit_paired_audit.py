from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from evaluation.rlivit import Box, FrameGroundTruth, Prediction, canonical_sha256, stable_file_pin
from validation.rlivit_paired_audit import (
    FixedFrameResult,
    RLiViTPairedAuditError,
    _read_json,
    _read_jsonl_records,
    _rank_frames,
    _stable_bytes,
    _verify_fingerprint,
    match_fixed_frame,
    pair_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "validation/schemas/rlivit-paired-error-audit-v1.schema.json"
REAL_AUDIT = ROOT / "validation/results/rlivit/paired-error-audit/paired-audit.json"
REAL_RECEIPT = ROOT / "validation/results/rlivit/paired-error-audit/receipt.json"


def _metrics(**overrides):
    value = {
        "ground_truth": 10,
        "tp": 4,
        "fp": 1,
        "fn": 6,
        "precision": 0.8,
        "recall": 0.4,
        "f1": 0.533333,
        "ap_101_point": 0.5,
    }
    value.update(overrides)
    return value


def _prediction(frame_index: int, detection_index: int, confidence: float, box: Box) -> Prediction:
    return Prediction(
        sequence_id="002",
        video_frame_index=frame_index,
        detection_index=detection_index,
        confidence=confidence,
        box=box,
    )


def _result(
    sequence_id: str,
    frame_index: int,
    *,
    gt: int,
    tp: int,
    fp: int,
) -> FixedFrameResult:
    boxes = tuple(Box(float(index * 20), 0.0, float(index * 20 + 10), 20.0) for index in range(gt))
    predictions = tuple(
        Prediction(
            sequence_id=sequence_id,
            video_frame_index=frame_index,
            detection_index=index,
            confidence=0.9 - index * 0.01,
            box=Box(float(index * 20), 0.0, float(index * 20 + 10), 20.0),
        )
        for index in range(tp + fp)
    )
    return FixedFrameResult(
        sequence_id=sequence_id,
        frame_index=frame_index,
        source_frame_index=2 + frame_index * 4,
        daytime="day",
        location="0",
        ground_truth=boxes,
        predictions=predictions,
        matches=tuple((index, index, 1.0) for index in range(tp)),
        false_negative_indices=tuple(range(tp, gt)),
        false_positive_indices=tuple(range(tp, tp + fp)),
    )


def test_pair_metrics_uses_explicit_960_minus_640_direction() -> None:
    paired = pair_metrics(
        _metrics(),
        _metrics(tp=7, fp=3, fn=3, precision=0.7, recall=0.7, f1=0.7, ap_101_point=0.625001),
    )
    assert paired["delta_960_minus_640"] == {
        "ground_truth": 0,
        "tp": 3,
        "fp": 2,
        "fn": -3,
        "precision": -0.1,
        "recall": 0.3,
        "f1": 0.166667,
        "ap_101_point": 0.125001,
    }


def test_pair_metrics_rejects_unpaired_ground_truth() -> None:
    with pytest.raises(RLiViTPairedAuditError, match="GT counts differ"):
        pair_metrics(_metrics(), _metrics(ground_truth=11))


def test_fixed_frame_matching_matches_evaluator_confidence_greedy_contract() -> None:
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
    result = match_fixed_frame(
        frame,
        (
            _prediction(0, 3, 0.1, Box(500, 500, 520, 520)),  # below threshold
            _prediction(0, 2, 0.7, Box(100, 100, 140, 160)),
            _prediction(0, 1, 0.8, Box(1, 1, 19, 19)),
            _prediction(0, 0, 0.9, Box(0, 0, 20, 20)),  # wins first GT
        ),
    )
    assert (result.tp, result.fp, result.fn) == (2, 1, 0)
    assert [item.confidence for item in result.predictions] == [0.9, 0.8, 0.7]
    assert result.matches[0][:2] == (0, 0)
    assert result.false_positive_indices == (1,)


def test_rankings_are_bounded_stable_and_surface_960_regression() -> None:
    keys = [("002", 0), ("002", 1), ("007", 0)]
    results = {
        640: {
            keys[0]: _result(*keys[0], gt=3, tp=2, fp=0),
            keys[1]: _result(*keys[1], gt=3, tp=3, fp=0),
            keys[2]: _result(*keys[2], gt=3, tp=1, fp=1),
        },
        960: {
            keys[0]: _result(*keys[0], gt=3, tp=1, fp=2),
            keys[1]: _result(*keys[1], gt=3, tp=3, fp=0),
            keys[2]: _result(*keys[2], gt=3, tp=2, fp=0),
        },
    }
    ranked = _rank_frames(results, limit=1)
    assert all(len(items) <= 1 for items in ranked.values())
    assert ranked["profile_regression_960_vs_640"] == [keys[0]]
    assert ranked["fn_640"] == [keys[2]]
    assert ranked["fp_960"] == [keys[0]]


def test_fingerprint_replay_rejects_mutation() -> None:
    value = {"schema_version": "example/v1", "status": "complete"}
    value["fingerprint_sha256"] = canonical_sha256(value)
    _verify_fingerprint(value, "fixture")
    value["status"] = "changed"
    with pytest.raises(RLiViTPairedAuditError, match="fingerprint replay differs"):
        _verify_fingerprint(value, "fixture")


def test_json_reader_rejects_duplicate_keys_at_any_object_depth(tmp_path: Path) -> None:
    document = tmp_path / "receipt.json"
    document.write_text(
        '{"schema_version":"fixture/v1","nested":{"value":1,"value":2}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RLiViTPairedAuditError, match="duplicate JSON key: value"):
        _read_json(document, root=tmp_path, context="fixture receipt")


def test_jsonl_reader_rejects_duplicate_keys_before_semantic_replay(tmp_path: Path) -> None:
    document = tmp_path / "predictions.jsonl"
    document.write_text(
        '{"frame_index":0,"detections":[],"frame_index":1}\n',
        encoding="utf-8",
    )
    pin = stable_file_pin(document, root=tmp_path)
    with pytest.raises(RLiViTPairedAuditError, match="duplicate JSON key: frame_index"):
        _read_jsonl_records(
            document,
            root=tmp_path,
            expected=pin,
            context="fixture predictions",
        )


def test_json_reader_rejects_non_json_numeric_constants(tmp_path: Path) -> None:
    document = tmp_path / "receipt.json"
    document.write_text('{"confidence":NaN}\n', encoding="utf-8")
    with pytest.raises(RLiViTPairedAuditError, match="non-JSON numeric constant: NaN"):
        _read_json(document, root=tmp_path, context="fixture receipt")


def test_secure_reader_rejects_preexisting_intermediate_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "value.json").write_text('{"outside":true}\n', encoding="utf-8")
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RLiViTPairedAuditError, match="cannot securely open pinned file beneath root"):
        _stable_bytes(Path("redirect/value.json"), root=root)


def test_secure_reader_uses_open_directory_fd_across_symlink_swap(
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
    expected = stable_file_pin(payload, root=root)

    real_open = os.open
    raced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal raced
        if path == "value.json" and dir_fd is not None and not raced:
            raced = True
            inside.rename(root / "inside-original")
            inside.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {racing_open})
    raw, observed = _stable_bytes(Path("inside/value.json"), root=root, expected=expected)
    assert raced is True
    assert raw == b'{"origin":"inside"}\n'
    assert observed == expected
    assert inside.is_symlink()


def test_secure_reader_fails_closed_without_openat_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "value.json"
    document.write_text('{"ok":true}\n', encoding="utf-8")
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(RLiViTPairedAuditError, match="unavailable on this platform"):
        _stable_bytes(document, root=tmp_path)


def test_paired_audit_schema_is_valid_draft_2020_12() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_paired_audit_schema_rejects_semantically_empty_nested_poc() -> None:
    if not REAL_AUDIT.is_file():
        pytest.skip("real paired audit has not been generated in this checkout")
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    report = json.loads(REAL_AUDIT.read_text(encoding="utf-8"))
    malformed = copy.deepcopy(report)
    malformed["paired_metrics"]["sequences"] = {
        key: None for key in malformed["paired_metrics"]["sequences"]
    }
    malformed["paired_metrics"]["strata"] = {
        key: {} for key in ("daytime", "locations", "coco_area", "height_bands")
    }
    malformed["rankings"] = {key: [{}] for key in malformed["rankings"]}
    malformed["assets"]["frames"] = {"fabricated": None}
    malformed["assets"]["contact_sheets"] = {"fabricated": None}
    malformed["assets"]["rendering_contract"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(malformed)


def test_paired_audit_schema_rejects_ranking_category_and_asset_contract_drift() -> None:
    if not REAL_AUDIT.is_file():
        pytest.skip("real paired audit has not been generated in this checkout")
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        json.loads(SCHEMA.read_text(encoding="utf-8"))
    )
    report = json.loads(REAL_AUDIT.read_text(encoding="utf-8"))

    wrong_category = copy.deepcopy(report)
    wrong_category["rankings"]["fn_640"][0]["category"] = "fp_960"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(wrong_category)

    bad_asset = copy.deepcopy(report)
    first = next(iter(bad_asset["assets"]["frames"].values()))
    first["overlays"]["640"]["path"] = "../escape.png"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad_asset)


def test_paired_audit_schema_accepts_explicit_no_render_shape() -> None:
    if not REAL_AUDIT.is_file():
        pytest.skip("real paired audit has not been generated in this checkout")
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    report = json.loads(REAL_AUDIT.read_text(encoding="utf-8"))
    report["assets"] = {
        "rendered": False,
        "frame_asset_count": 0,
        "frames": {},
        "contact_sheets": {},
        "rendering_contract": {"reason": "disabled_by_explicit_no_render"},
    }
    jsonschema.Draft202012Validator(schema).validate(report)


def test_real_paired_audit_is_self_consistent_when_present() -> None:
    if not REAL_AUDIT.is_file() or not REAL_RECEIPT.is_file():
        pytest.skip("real paired audit has not been generated in this checkout")
    report = json.loads(REAL_AUDIT.read_text(encoding="utf-8"))
    receipt = json.loads(REAL_RECEIPT.read_text(encoding="utf-8"))
    report_fingerprint = report.pop("fingerprint_sha256")
    assert canonical_sha256(report) == report_fingerprint
    assert receipt["audit_fingerprint_sha256"] == report_fingerprint
    assert receipt["audit"]["size_bytes"] == REAL_AUDIT.stat().st_size
    assert receipt["audit"]["sha256"] == __import__("hashlib").sha256(REAL_AUDIT.read_bytes()).hexdigest()
    receipt_fingerprint = receipt.pop("fingerprint_sha256")
    assert canonical_sha256(receipt) == receipt_fingerprint
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(
        {**report, "fingerprint_sha256": report_fingerprint}
    )
