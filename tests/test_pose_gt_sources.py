from __future__ import annotations

import copy
import hashlib
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import pose_gt_sources as sources


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/pose-gt-evaluation-sources.json"
SCHEMA_PATH = ROOT / "validation/schemas/pose-gt-evaluation-sources-v1.schema.json"


def _load(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _repin(value: dict) -> dict:
    value["fingerprint_sha256"] = sources.canonical_fingerprint(value)
    return value


def _by_id(value: dict) -> dict[str, dict]:
    return {source["id"]: source for source in value["sources"]}


def test_checked_in_package_is_valid_but_blocked_for_training_acceptance_and_25m() -> None:
    report = sources.validate_checked_in_package()
    assert report["schema_version"] == "deepsafe.pose-gt-source-validation/v1"
    assert report["status"] == "valid_diagnostic_plan_blocked_for_training_product_acceptance_and_25m"
    assert report["manifest_sha256"] == hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert report["manifest_fingerprint_sha256"] == sources.EXPECTED_MANIFEST_FINGERPRINT_SHA256
    assert report["schema_sha256"] == hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    assert report["source_count"] == 4
    assert report["selected_subset"] == {
        "id": "coco2017-val-license7-visible-pose-3-v1",
        "images": 3,
        "person_annotations": 3,
        "visible_keypoints": 38,
        "video_sequences": 0,
        "track_ids": 0,
        "overhead_or_security_views": 0,
        "media_persisted_in_repository": False,
    }
    for key, value in report.items():
        if key.startswith("eligible_for_"):
            assert value is False
    assert report["large_archive_downloaded"] is False
    assert report["validator_network_media_decode_gpu_docker_or_inference_used"] is False


def test_checked_in_manifest_matches_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        _load(SCHEMA_PATH), format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(_load()), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]


def test_schema_contract_is_exactly_size_and_hash_pinned() -> None:
    raw = SCHEMA_PATH.read_bytes()
    assert len(raw) == sources.EXPECTED_SCHEMA_SIZE_BYTES
    assert hashlib.sha256(raw).hexdigest() == sources.EXPECTED_SCHEMA_SHA256


def test_source_decisions_preserve_license_and_format_blockers() -> None:
    candidates = _by_id(_load())
    assert list(candidates) == list(sources.REQUIRED_SOURCE_IDS)

    coco = candidates["coco_2017_keypoints_val"]
    assert coco["decision"] == "selected_for_small_remote_diagnostic_plan_only"
    assert coco["annotations"]["keypoint_order"] == list(sources.COCO_KEYPOINTS)
    assert coco["media"]["video"] is False
    assert coco["media"]["track_ids"] is False
    assert coco["rights"]["annotation_license"] == "CC-BY-4.0"
    assert coco["rights"]["image_copyright_owned_by_dataset_publisher"] is False
    assert coco["rights"]["selected_image_rights_is_a_warranty"] is False

    jrdb = candidates["jrdb_pose_2022"]
    assert jrdb["media"]["video"] is True and jrdb["media"]["track_ids"] is True
    assert jrdb["annotations"]["exact_coco17_order"] is False
    assert jrdb["rights"]["dataset_license"] == "CC-BY-NC-SA-3.0"

    jta = candidates["jta"]
    assert jta["media"]["video"] is True and jta["media"]["track_ids"] is True
    assert jta["rights"]["dataset_license"] == "CC-BY-NC-4.0"

    mpii = candidates["mpii_human_pose_v1"]
    assert mpii["annotations"]["keypoint_count"] == 16
    assert mpii["media"]["track_ids"] is False
    assert mpii["rights"]["publisher_owns_image_copyright"] is False
    assert mpii["rights"]["commercial_use_allowed"] is False

    for candidate in candidates.values():
        assert candidate["eligibility"]["pose_pck_product_acceptance"] is False
        assert candidate["eligibility"]["commercial_model_training"] is False
        assert candidate["eligibility"]["exact_25m"] is False
        assert candidate["media"]["fixed_security_camera"] is False
        assert candidate["media"]["high_or_overhead_view_officially_labeled"] is False


def test_small_coco_plan_reconciles_all_remote_pins_without_local_media() -> None:
    plan = _load()["selected_subset_plan"]
    artifact = plan["official_annotation_artifact"]
    assert artifact["archive_content_length_bytes"] == 252_907_541
    assert artifact["full_archive_downloaded"] is False
    assert artifact["full_archive_sha256"] is None
    assert artifact["member_compressed_bytes"] == 2_986_607
    assert artifact["member_uncompressed_bytes"] == 10_020_657
    assert artifact["member_crc32"] == "27d86024"
    assert artifact["member_sha256"] == "788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14"

    images = plan["images"]
    assert [image["image_id"] for image in images] == [274219, 427077, 560880]
    assert sum(image["size_bytes"] for image in images) == plan["summary"]["total_remote_image_bytes"] == 361_160
    assert sum(image["visible_keypoints"] for image in images) == plan["summary"]["visible_keypoints"] == 38
    assert sum(image["occluded_labeled_keypoints"] for image in images) == 2
    assert all(image["coco_license_id"] == 7 for image in images)
    assert all(image["independent_human_review_complete"] is False for image in images)
    assert all(image["persisted_in_repository"] is False for image in images)
    assert plan["summary"]["video_sequences"] == 0
    assert plan["summary"]["track_ids"] == 0
    assert plan["summary"]["overhead_or_security_views"] == 0


@pytest.mark.parametrize("source_id", sources.REQUIRED_SOURCE_IDS)
@pytest.mark.parametrize(
    "field",
    ("pose_pck_product_acceptance", "commercial_model_training", "exact_25m"),
)
def test_promoting_any_source_eligibility_is_rejected(source_id: str, field: str) -> None:
    value = _load()
    _by_id(value)[source_id]["eligibility"][field] = True
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="forbidden"):
        sources.validate_manifest(value)


@pytest.mark.parametrize("source_id", sources.REQUIRED_SOURCE_IDS)
@pytest.mark.parametrize(
    "field",
    ("fixed_security_camera", "high_or_overhead_view_officially_labeled"),
)
def test_promoting_unproven_camera_views_is_rejected(source_id: str, field: str) -> None:
    value = _load()
    _by_id(value)[source_id]["media"][field] = True
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="claim forbidden"):
        sources.validate_manifest(value)


@pytest.mark.parametrize(
    ("source_id", "field"),
    (
        ("coco_2017_keypoints_val", "commercial_training_cleared"),
        ("jrdb_pose_2022", "commercial_use_allowed"),
        ("jta", "commercial_use_allowed"),
        ("mpii_human_pose_v1", "commercial_use_allowed"),
    ),
)
def test_license_blocked_sources_cannot_be_reclassified_by_boolean(
    source_id: str, field: str
) -> None:
    value = _load()
    _by_id(value)[source_id]["rights"][field] = True
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="forbidden|not cleared"):
        sources.validate_manifest(value)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("selection_policy", "training_allowed", True),
        ("selection_policy", "production_acceptance_allowed", True),
        ("official_annotation_artifact", "full_archive_downloaded", True),
        ("acquisition_receipt", "large_archive_downloaded", True),
        ("acquisition_receipt", "gpu_model_docker_or_inference_executed", True),
        ("readiness", "independent_project_human_review_complete", True),
        ("readiness", "security_or_overhead_coverage_complete", True),
        ("readiness", "commercial_or_closed_product_rights_cleared", True),
        ("readiness", "eligible_for_pose_pck_product_acceptance", True),
        ("readiness", "eligible_for_exact_25m", True),
    ),
)
def test_subset_cannot_be_promoted_or_claim_unperformed_work(
    section: str, field: str, replacement: object
) -> None:
    value = _load()
    value["selected_subset_plan"][section][field] = replacement
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="differs"):
        sources.validate_manifest(value)


def test_image_media_hash_review_and_repository_state_are_pinned() -> None:
    changes = (
        ("sha256", "0" * 64),
        ("annotation_record_sha256", "1" * 64),
        ("independent_human_review_complete", True),
        ("persisted_in_repository", True),
    )
    for field, replacement in changes:
        value = _load()
        value["selected_subset_plan"]["images"][0][field] = replacement
        _repin(value)
        with pytest.raises(sources.PoseGTSourceError):
            sources.validate_manifest(value)


def test_summary_tamper_and_reconciled_but_unreviewed_manifest_change_fail() -> None:
    value = _load()
    value["selected_subset_plan"]["summary"]["visible_keypoints"] = 39
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="summary differs"):
        sources.validate_manifest(value)

    # A field not otherwise interpreted still cannot change without a reviewed
    # code/schema pin update.
    value = _load()
    value["sources"][0]["name"] = "renamed source"
    _repin(value)
    with pytest.raises(sources.PoseGTSourceError, match="reviewed fingerprint differs"):
        sources.validate_manifest(value)


def test_self_fingerprint_is_required() -> None:
    value = _load()
    value["fingerprint_sha256"] = "0" * 64
    with pytest.raises(sources.PoseGTSourceError, match="self-fingerprint differs"):
        sources.validate_manifest(value)


def test_global_25m_security_and_acceptance_guardrails_cannot_be_relaxed() -> None:
    for field, current in sources.EXPECTED_GUARDRAILS.items():
        value = _load()
        value["global_guardrails"][field] = not current
        _repin(value)
        with pytest.raises(sources.PoseGTSourceError, match="guardrails differ"):
            sources.validate_manifest(value)


def test_validator_has_no_network_process_or_filesystem_write_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    before = list(tmp_path.iterdir())
    report = sources.validate_checked_in_package()
    assert list(tmp_path.iterdir()) == before == []
    assert report["source_count"] == 4


def test_validator_exposes_no_download_decode_gpu_docker_or_inference_api() -> None:
    forbidden = {
        "download",
        "fetch",
        "materialize",
        "decode",
        "run_inference",
        "load_model",
        "gpu_probe",
        "docker_run",
    }
    assert forbidden.isdisjoint(set(dir(sources)))


def test_bounded_loader_rejects_symlinks_hardlinks_and_ambiguous_json(
    tmp_path: Path,
) -> None:
    link = tmp_path / "manifest-link.json"
    link.symlink_to(MANIFEST_PATH)
    with pytest.raises(sources.PoseGTSourceError, match="cannot safely read"):
        sources.load_manifest(link)

    hardlink = tmp_path / "manifest-hardlink.json"
    hardlink.hardlink_to(MANIFEST_PATH)
    try:
        with pytest.raises(sources.PoseGTSourceError, match="exactly one hard link"):
            sources.load_manifest(hardlink)
    finally:
        hardlink.unlink()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(sources.PoseGTSourceError, match="strict UTF-8 JSON"):
        sources.load_manifest(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(sources.PoseGTSourceError, match="strict UTF-8 JSON"):
        sources.load_manifest(nonfinite)


def test_cli_prints_pathless_blocked_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert sources.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == sources.REPORT_STATUS
    assert "path" not in json.dumps(report).lower()
    assert report["eligible_for_pose_pck_product_acceptance"] is False

