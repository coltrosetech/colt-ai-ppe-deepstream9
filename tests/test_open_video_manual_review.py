import copy
import hashlib
import json
from pathlib import Path

import pytest

from validation import open_video_manual_review as manual


SOURCE_RECORDS = Path("validation/open_video_review/source-frame-reviews-v1.jsonl")
AI_QUALITATIVE_AUDIT = Path(
    "validation/results/open-video-review/ai-qualitative-audit.json"
)
MANUAL_ASSET_INDEX = Path(
    "validation/results/open-video-review/manual-assets/index.json"
)


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sources():
    return _rows(SOURCE_RECORDS)


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        '{"outer":{"value":1,"value":2}}\n',
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
        '{"value":-Infinity}\n',
        '{"value":1e400}\n',
    ],
)
def test_integrity_jsonl_loader_rejects_nested_duplicate_and_nonfinite_values(
    tmp_path, payload
):
    path = tmp_path / "records.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(manual.ReviewValidationError, match="invalid JSON"):
        manual._read_jsonl(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"first","schema_version":"second"}',
        '{"schema_version":"deepsafe.open-video-review-scenes/v1",'
        '"scenes":[],"metadata":{"value":NaN}}',
    ],
)
def test_scene_control_json_rejects_duplicate_and_nonfinite_values(
    tmp_path, payload
):
    scene_manifest = tmp_path / "scenes.json"
    normalization = tmp_path / "normalization.tsv"
    scene_manifest.write_text(payload, encoding="utf-8")
    normalization.write_text("id\n", encoding="utf-8")

    # StrictJSONError remains a JSONDecodeError so the historical exception
    # contract for this low-level control-file loader is unchanged.
    with pytest.raises(json.JSONDecodeError):
        manual._load_scene_contract(scene_manifest, normalization)


def _reviewed_decision(tmp_path: Path, source):
    report = tmp_path / "review.json"
    overlay = tmp_path / "overlay.png"
    predictions = tmp_path / "predictions.jsonl"
    report.write_text('{"fixture":true}\n', encoding="utf-8")
    overlay.write_bytes(b"fixture-png")
    predictions.write_text('{"fixture":true}\n', encoding="utf-8")
    row = manual.make_overlay_template([source], profiles=(640,))[0]
    row["overlay_evidence"] = {
        "review_report_path": report.name,
        "review_report_sha256": _sha(report),
        "overlay_image_path": overlay.name,
        "overlay_image_sha256": _sha(overlay),
        "predictions_path": predictions.name,
        "predictions_sha256": _sha(predictions),
    }
    row["decision"] = {
        "status": "reviewed",
        "detection_count_reviewed": 1,
        "visible_person_count_confirmed": 1,
        "scorable_person_count_confirmed": 1,
        "true_positive_count": 1,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "ignored_detection_count": 0,
        "unscorable_visible_person_count": 0,
        "reasons": [],
    }
    row["review"] = {
        "reviewer_id": "fixture-reviewer",
        "reviewer_type": "human_with_ai_assist",
        "reviewed_at": "2026-07-16T04:00:00+03:00",
    }
    return row


def test_committed_source_records_cover_all_scenes_and_keep_sensitive_closed():
    source = _sources()
    report = manual.validate_source_records(source)

    assert report["valid"] is True
    assert report["record_count"] == 22
    assert report["scene_count"] == 12
    assert report["closed_review_records"] == 1
    sensitive = [row for row in source if row["sensitive"]]
    assert len(sensitive) == 1
    assert sensitive[0]["review"]["status"] == "closed_review"
    assert sensitive[0]["evidence"]["contact_sheet_path"] is None
    assert sensitive[0]["observation"]["visible_person_count_range"] == {
        "min": None,
        "max": None,
    }


def test_committed_ai_estimates_are_contained_by_source_review_ranges():
    sources = {row["record_id"]: row for row in _sources()}
    audit = json.loads(AI_QUALITATIVE_AUDIT.read_text(encoding="utf-8"))

    for review in audit["reviews"]:
        source_id = review["source_review_id"]
        source_observation = sources[source_id]["observation"]
        for estimate_key, source_key in (
            ("estimated_visible_persons", "visible_person_count_range"),
            ("estimated_scorable_persons", "scorable_person_count_range"),
        ):
            estimate = review[estimate_key]
            source_range = source_observation[source_key]
            assert estimate["min"] >= source_range["min"], source_id
            if source_range["max"] is not None:
                assert estimate["max"] is not None, source_id
                assert estimate["max"] <= source_range["max"], source_id


def test_committed_ai_audit_is_hash_bound_to_source_records_and_asset_index():
    source_sha256 = _sha(SOURCE_RECORDS)
    index_bytes = MANUAL_ASSET_INDEX.read_bytes()
    index = json.loads(index_bytes)
    audit = json.loads(AI_QUALITATIVE_AUDIT.read_text(encoding="utf-8"))
    bindings = audit["input_bindings"]
    manual_binding = bindings["manual_assets_index"]

    assert index["source_records_sha256"] == source_sha256
    assert bindings["source_records_sha256"] == source_sha256
    assert manual_binding["sha256"] == hashlib.sha256(index_bytes).hexdigest()
    assert manual_binding["size_bytes"] == len(index_bytes)
    assert manual_binding["bundle_id"] == index["bundle_id"]


def test_contact_sheet_midpoint_mapping_matches_committed_examples():
    assert manual.contact_sheet_frame_index(1, 932) == 116
    assert manual.contact_sheet_frame_index(8, 932) == 660
    assert manual.contact_sheet_frame_index(10, 597) == 522
    with pytest.raises(manual.ReviewValidationError):
        manual.contact_sheet_frame_index(12, 932)


def test_json_schemas_accept_source_and_pending_overlay_rows():
    jsonschema = pytest.importorskip("jsonschema")
    source_schema = json.loads(
        Path(
            "validation/open_video_review/schemas/source-frame-review-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    overlay_schema = json.loads(
        Path(
            "validation/open_video_review/schemas/overlay-decision-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    source = _sources()
    source_validator = jsonschema.Draft202012Validator(source_schema)
    overlay_validator = jsonschema.Draft202012Validator(overlay_schema)
    for row in source:
        source_validator.validate(row)
    for row in manual.make_overlay_template(source):
        overlay_validator.validate(row)


def test_template_creates_both_profiles_and_omits_sensitive_by_default():
    source = _sources()
    decisions = manual.make_overlay_template(source)

    assert len(decisions) == 42
    assert {row["model_input"] for row in decisions} == {640, 960}
    assert not any(row["scene_id"] == "pk_fall_event_distant" for row in decisions)
    assert {row["decision"]["status"] for row in decisions} == {"pending_overlay"}

    with_sensitive = manual.make_overlay_template(
        source, profiles=(640,), include_sensitive=True
    )
    closed = [row for row in with_sensitive if row["scene_id"] == "pk_fall_event_distant"]
    assert len(closed) == 1
    assert closed[0]["decision"]["status"] == "closed_review"
    assert closed[0]["review_visibility"] == "closed"
    assert all(value is None for value in closed[0]["overlay_evidence"].values())


def test_overlay_qa_accepts_reconciled_human_decision_and_bound_hashes(tmp_path):
    source = _sources()[0]
    decision = _reviewed_decision(tmp_path, source)

    report = manual.validate_overlay_decisions(
        [source], [decision], workspace_root=tmp_path, require_profiles=(640,), require_complete=True
    )
    assert report["status_counts"] == {"reviewed": 1}


def test_overlay_qa_rejects_bad_counts_wrong_frame_and_ai_only_final_review(tmp_path):
    source = _sources()[0]
    decision = _reviewed_decision(tmp_path, source)

    broken = copy.deepcopy(decision)
    broken["decision"]["false_positive_count"] = 1
    with pytest.raises(manual.ReviewValidationError, match=r"TP\+FP\+ignored"):
        manual.validate_overlay_decisions([source], [broken], workspace_root=tmp_path)

    broken = copy.deepcopy(decision)
    broken["frame_index"] += 1
    with pytest.raises(manual.ReviewValidationError, match="wrong scene/frame"):
        manual.validate_overlay_decisions([source], [broken], workspace_root=tmp_path)

    broken = copy.deepcopy(decision)
    broken["review"]["reviewer_type"] = "ai_visual_inspection"
    with pytest.raises(manual.ReviewValidationError, match="human reviewer"):
        manual.validate_overlay_decisions([source], [broken], workspace_root=tmp_path)


def test_overlay_qa_rejects_hash_mismatch_and_confirmed_count_outside_source_range(tmp_path):
    source = _sources()[0]
    decision = _reviewed_decision(tmp_path, source)

    broken = copy.deepcopy(decision)
    broken["overlay_evidence"]["overlay_image_sha256"] = "0" * 64
    with pytest.raises(manual.ReviewValidationError, match="SHA-256 mismatch"):
        manual.validate_overlay_decisions([source], [broken], workspace_root=tmp_path)

    broken = copy.deepcopy(decision)
    broken["decision"].update(
        {
            "detection_count_reviewed": 2,
            "visible_person_count_confirmed": 2,
            "scorable_person_count_confirmed": 2,
            "true_positive_count": 2,
        }
    )
    with pytest.raises(manual.ReviewValidationError, match="outside source review range"):
        manual.validate_overlay_decisions([source], [broken], workspace_root=tmp_path)


def test_require_complete_rejects_pending_rows_and_merge_reports_progress():
    source = _sources()
    decisions = manual.make_overlay_template(source)
    manual.validate_overlay_decisions(
        source, decisions, require_profiles=(640, 960), require_complete=False
    )
    with pytest.raises(manual.ReviewValidationError, match="non-terminal"):
        manual.validate_overlay_decisions(
            source, decisions, require_profiles=(640, 960), require_complete=True
        )

    merged = manual.merge_reviews(source, decisions)
    assert len(merged) == len(source)
    ordinary = [row for row in merged if not row["source_review"]["sensitive"]]
    closed = [row for row in merged if row["source_review"]["sensitive"]]
    assert {row["comparison_status"] for row in ordinary} == {"awaiting_overlays"}
    assert closed[0]["comparison_status"] == "closed_review"
    assert closed[0]["overlay_reviews"] == []
