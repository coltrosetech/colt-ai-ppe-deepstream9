import copy
import hashlib
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_video_sources as sources


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/manifests/ppe-video-source-candidates.json"
SCHEMA_PATH = ROOT / "validation/schemas/ppe-video-source-candidates-v1.schema.json"


def _load(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(registry: dict) -> dict[str, dict]:
    return {candidate["id"]: candidate for candidate in registry["candidates"]}


def test_checked_in_registry_is_valid_metadata_only_and_all_quantitative_candidates_blocked() -> None:
    report = sources.validate_checked_in_registry()
    assert report["schema_version"] == "deepsafe.ppe-video-source-registry-report/v1"
    assert report["status"] == "valid_metadata_only_all_quantitative_commercial_video_candidates_blocked"
    assert report["candidate_count"] == 12
    assert report["registry_sha256"] == hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    assert set(report["eligibility_counts"].values()) == {0}
    assert report["media_or_annotations_downloaded"] is False
    assert report["network_gpu_docker_or_inference_used"] is False


def test_checked_in_registry_matches_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        _load(SCHEMA_PATH), format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(_load()), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]


def test_exact_candidate_set_and_evidence_dates_are_preserved() -> None:
    registry = _load()
    assert set(_by_id(registry)) == sources.REQUIRED_CANDIDATE_IDS
    for candidate in registry["candidates"]:
        if candidate["id"] == "user_owned_authorized_site_footage":
            assert candidate["evidence"] == []
        else:
            assert candidate["evidence"]
        for evidence in candidate["evidence"]:
            assert evidence["url"].startswith("https://")
            assert evidence["observed_at"] == "2026-07-16"


def test_every_requested_right_and_ground_truth_dimension_is_explicit_tri_state() -> None:
    for candidate in _load()["candidates"]:
        media = candidate["media"]
        truth = candidate["ground_truth"]
        rights = candidate["rights"]
        assert media["public_media_available"] in (True, False, None)
        assert media["is_video"] in (True, False, None)
        for field in (
            "bbox_ground_truth",
            "frame_linked_bbox_ground_truth",
            "helmet_scope",
            "high_visibility_scope",
            "temporal_ground_truth",
            "track_ground_truth",
            "pose_ground_truth",
            "metric_distance_ground_truth",
        ):
            assert truth[field] in (True, False, None)
        assert candidate["camera"]["camera_angle_per_asset_documented"] in (True, False, None)
        for field in (
            "commercial_ml_evaluation_rights_confirmed",
            "person_release_confirmed",
            "location_release_confirmed",
            "model_release_confirmed",
        ):
            assert rights[field] in (True, False, None)


def test_null_or_false_in_every_required_right_or_gt_gate_prevents_acceptance() -> None:
    complete = copy.deepcopy(_by_id(_load())["user_owned_authorized_site_footage"])
    for section, field in sources.BASE_REQUIREMENTS:
        complete[section][field] = True
    for field in sources.ADDITIONAL_REQUIREMENTS.values():
        complete["ground_truth"][field] = True
    assert set(sources.derive_eligibility(complete).values()) == {True}

    for section, field in sources.BASE_REQUIREMENTS:
        for missing in (None, False):
            candidate = copy.deepcopy(complete)
            candidate[section][field] = missing
            result = sources.derive_eligibility(candidate)
            assert result["quantitative_commercial_video_benchmark"] is False, (section, field, missing)
            assert set(result.values()) == {False}, (section, field, missing)


def test_each_additional_claim_needs_its_own_literal_true_ground_truth() -> None:
    complete = copy.deepcopy(_by_id(_load())["user_owned_authorized_site_footage"])
    for section, field in sources.BASE_REQUIREMENTS:
        complete[section][field] = True
    for field in sources.ADDITIONAL_REQUIREMENTS.values():
        complete["ground_truth"][field] = True

    for metric, field in sources.ADDITIONAL_REQUIREMENTS.items():
        for missing in (None, False):
            candidate = copy.deepcopy(complete)
            candidate["ground_truth"][field] = missing
            result = sources.derive_eligibility(candidate)
            assert result["quantitative_commercial_video_benchmark"] is True
            assert result[metric] is False


def test_public_media_is_recorded_but_private_authorized_owner_media_is_a_valid_access_path() -> None:
    complete = copy.deepcopy(_by_id(_load())["user_owned_authorized_site_footage"])
    for section, field in sources.BASE_REQUIREMENTS:
        complete[section][field] = True
    assert complete["media"]["public_media_available"] is False
    assert sources.derive_eligibility(complete)["quantitative_commercial_video_benchmark"] is True


def test_forcing_any_checked_in_candidate_to_accepted_is_rejected() -> None:
    registry = _load()
    for index in range(len(registry["candidates"])):
        tampered = copy.deepcopy(registry)
        tampered["candidates"][index]["eligibility"]["quantitative_commercial_video_benchmark"] = True
        with pytest.raises(sources.PPEVideoSourceRegistryError, match="eligibility differs"):
            sources.validate_registry(tampered)


def test_al_azani_requires_written_dataset_and_media_license() -> None:
    candidate = _by_id(_load())["al_azani_kfupm_ppe_cctv"]
    assert candidate["media"]["public_media_available"] is False
    assert candidate["media"]["project_access_authorized"] is False
    assert candidate["rights"]["commercial_ml_evaluation_rights_confirmed"] is None
    assert candidate["disposition"] == "blocked_requires_written_dataset_and_media_license"
    assert "written_dataset_and_media_license_missing" in candidate["blocking_reasons"]


def test_mobiusi_remains_conditional_on_sample_schema_contract_and_has_hi_vis_gap() -> None:
    candidate = _by_id(_load())["mobiusi_helmet_action"]
    assert candidate["media"]["public_media_available"] is True
    assert candidate["ground_truth"]["bbox_ground_truth"] is None
    assert candidate["ground_truth"]["frame_linked_bbox_ground_truth"] is None
    assert candidate["ground_truth"]["high_visibility_scope"] is False
    assert candidate["rights"]["commercial_ml_evaluation_rights_confirmed"] is False
    assert candidate["disposition"] == "conditional_sample_schema_contract_review_with_hi_vis_gap"


def test_non_video_and_unlinked_video_labels_cannot_be_promoted() -> None:
    candidates = _by_id(_load())
    assert candidates["foundation_pit_v2"]["ground_truth"]["frame_linked_bbox_ground_truth"] is False
    assert candidates["tcrsf_sfchd"]["media"]["is_video"] is False
    assert candidates["ppe_cctv_topdown"]["media"]["is_video"] is False
    assert candidates["put_your_ppe_on"]["media"]["is_video"] is False
    assert candidates["put_your_ppe_on"]["ground_truth"]["pose_ground_truth"] is True
    assert candidates["put_your_ppe_on"]["ground_truth"]["metric_distance_ground_truth"] is True


def test_stock_candidates_are_qualitative_only_and_pexels_is_ml_restricted() -> None:
    candidates = _by_id(_load())
    for candidate_id in (
        "pixabay_construction_worker_348896",
        "mixkit_two_construction_workers_1436",
    ):
        candidate = candidates[candidate_id]
        assert candidate["ground_truth"]["bbox_ground_truth"] is False
        assert candidate["rights"]["commercial_ml_evaluation_rights_confirmed"] is None
        assert candidate["disposition"] == "qualitative_only_no_ground_truth_or_release_proof"
    for candidate_id in (
        "pexels_construction_worker_roof_16393893",
        "pexels_construction_site_7448386",
    ):
        candidate = candidates[candidate_id]
        assert candidate["rights"]["commercial_ml_evaluation_rights_confirmed"] is False
        assert candidate["disposition"] == "rejected_for_ml_use_without_written_permission"


def test_no_local_media_path_or_download_receipt_is_present() -> None:
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"local_path", "download_url", "local_sha256", "media_sha256"}
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(_load())
    assert _load()["acquisition"] == {
        "metadata_only": True,
        "media_downloaded": False,
        "annotation_downloaded": False,
        "media_decoded": False,
        "network_used_by_validator": False,
        "gpu_used": False,
        "docker_used": False,
        "inference_used": False,
    }


def test_validator_has_no_network_process_or_filesystem_write_side_effects(monkeypatch, tmp_path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    before = list(tmp_path.iterdir())
    report = sources.validate_checked_in_registry()
    after = list(tmp_path.iterdir())
    assert before == after == []
    assert report["candidate_count"] == 12


def test_validator_exposes_no_download_decode_gpu_docker_or_inference_api() -> None:
    forbidden = {
        "download",
        "decode",
        "run_inference",
        "docker_run",
        "gpu_probe",
        "materialize",
        "fetch",
    }
    assert forbidden.isdisjoint(set(dir(sources)))


def test_bounded_loader_rejects_symlinks_and_duplicate_keys(tmp_path) -> None:
    link = tmp_path / "registry.json"
    link.symlink_to(REGISTRY_PATH)
    with pytest.raises(sources.PPEVideoSourceRegistryError, match="not a regular file"):
        sources._load_bounded_json(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(sources.PPEVideoSourceRegistryError, match="duplicate JSON key"):
        sources._load_bounded_json(duplicate)


def test_cli_prints_pathless_report_without_writing(capsys) -> None:
    assert sources.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_count"] == 12
    assert "path" not in json.dumps(report).lower()
    assert set(report["eligibility_counts"].values()) == {0}
