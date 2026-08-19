import copy
import hashlib
import json
import socket
import subprocess
from pathlib import Path

import pytest

from validation import ppe_open_gt_sources as sources


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/manifests/ppe-open-gt-source-candidates-v1.json"
SCHEMA_PATH = ROOT / "validation/schemas/ppe-open-gt-source-candidates-v1.schema.json"


def _load(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(registry: dict | None = None) -> dict[str, dict]:
    value = registry or _load()
    return {candidate["id"]: candidate for candidate in value["candidates"]}


def test_checked_in_registry_is_valid_and_every_authorization_is_blocked() -> None:
    report = sources.validate_checked_in_registry()
    assert report["schema_version"] == "deepsafe.ppe-open-gt-source-candidates-report/v1"
    assert report["status"] == "valid_metadata_only_all_download_and_training_authorizations_blocked"
    assert report["candidate_count"] == 14
    assert report["registry_sha256"] == hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    assert report["gate_pass_counts"]["training_eligible"] == 0
    assert report["gate_pass_counts"]["download_authorized"] == 0
    assert report["gate_pass_counts"]["training_authorized"] == 0
    assert report["media_or_annotations_downloaded"] is False
    assert report["training_inference_or_gpu_used"] is False


def test_registry_matches_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        _load(SCHEMA_PATH), format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(_load()), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]


def test_exact_candidate_set_and_current_primary_evidence_are_preserved() -> None:
    registry = _load()
    assert set(_by_id(registry)) == sources.REQUIRED_CANDIDATE_IDS
    assert len(registry["candidates"]) >= 8
    allowed_hosts = {
        "www.mdpi.com",
        "data.mendeley.com",
        "docs.ultralytics.com",
        "github.com",
        "arxiv.org",
        "pmc.ncbi.nlm.nih.gov",
        "universe.roboflow.com",
        "zenodo.org",
        "www.nature.com",
        "ieeexplore.ieee.org",
        "ri.kfupm.edu.sa",
    }
    from urllib.parse import urlsplit

    for candidate in registry["candidates"]:
        assert candidate["evidence"]
        for evidence in candidate["evidence"]:
            assert evidence["observed_at"] == "2026-07-18"
            assert urlsplit(evidence["url"]).hostname in allowed_hosts


def test_every_requested_source_dimension_is_explicit() -> None:
    for candidate in _load()["candidates"]:
        assert candidate["evidence"][0]["url"].startswith("https://")
        assert candidate["rights"]["license_status"] in {
            "verified_permissive",
            "verified_reciprocal",
            "verified_noncommercial",
            "proprietary_contract",
            "missing_or_unknown",
        }
        assert candidate["media"]["type"] in {
            "static_images",
            "sampled_video_frames",
            "video",
            "video_and_extracted_frames",
        }
        assert candidate["ground_truth"]["types"]
        assert candidate["ground_truth"]["helmet_scope"] in {"present", "absent", "unknown"}
        assert candidate["ground_truth"]["hi_vis_scope"] in {"present", "absent", "unknown"}
        assert candidate["camera_scene"]["scene_types"]
        assert candidate["camera_scene"]["angle_types"]
        assert candidate["grouping_leakage"]["leakage_risk"] in {"low", "medium", "high", "unknown"}
        assert candidate["download"]["availability"]
        assert candidate["decision"]["reason"]
        assert candidate["decision"]["blocking_reasons"]


def test_literal_true_all_gate_derivation_and_license_fail_closed_behavior() -> None:
    complete = copy.deepcopy(_by_id()["mendeley_ppe_five_class_v1"])
    complete["rights"].update(
        {
            "embedded_media_provenance_cleared": True,
            "person_release_confirmed": True,
            "location_release_confirmed": True,
        }
    )
    complete["grouping_leakage"]["split_grouping_metadata_sufficient"] = True
    for field in sources.GATE_FACT_FIELDS:
        complete["gate_facts"][field] = True

    assert set(sources.derive_eligibility(complete).values()) == {True}

    for status in (
        "missing_or_unknown",
        "verified_reciprocal",
        "verified_noncommercial",
        "proprietary_contract",
    ):
        candidate = copy.deepcopy(complete)
        candidate["rights"]["license_status"] = status
        result = sources.derive_eligibility(candidate)
        assert result["license_gate_passed"] is False
        assert result["training_eligible"] is False
        assert result["download_authorized"] is False
        assert result["training_authorized"] is False

    for field in (
        "embedded_media_provenance_cleared",
        "person_release_confirmed",
        "location_release_confirmed",
    ):
        for missing in (None, False):
            candidate = copy.deepcopy(complete)
            candidate["rights"][field] = missing
            result = sources.derive_eligibility(candidate)
            assert result["rights_provenance_gate_passed"] is False
            assert result["training_eligible"] is False


def test_checked_in_eligibility_is_derived_not_asserted() -> None:
    registry = _load()
    for candidate in registry["candidates"]:
        assert candidate["eligibility"] == sources.derive_eligibility(candidate)
        assert candidate["eligibility"]["training_eligible"] is False
        assert candidate["eligibility"]["download_authorized"] is False
        assert candidate["eligibility"]["training_authorized"] is False

    tampered = copy.deepcopy(registry)
    tampered["candidates"][0]["eligibility"]["training_eligible"] = True
    with pytest.raises(sources.PPEOpenGTSourceError, match="eligibility differs"):
        sources.validate_registry(tampered)


def test_verified_permissive_license_is_not_treated_as_complete_rights() -> None:
    candidates = _by_id()
    expected_license_pass = {
        "mendeley_ppe_five_class_v1",
        "shel5k_v2",
        "edgevision_v1",
        "ppe_cctv_topdown",
        "zenodo_pped_v1",
    }
    actual = {
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["eligibility"]["license_gate_passed"]
    }
    assert actual == expected_license_pass
    for candidate_id in expected_license_pass:
        candidate = candidates[candidate_id]
        assert candidate["eligibility"]["rights_provenance_gate_passed"] is False
        assert candidate["eligibility"]["training_eligible"] is False


def test_joint_target_and_security_camera_shortlist_is_cautious() -> None:
    candidates = _by_id()
    target_pass = {
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["eligibility"]["target_scope_gate_passed"]
    }
    assert target_pass == {
        "lo_ppe_compliance_11k",
        "mendeley_ppe_five_class_v1",
        "ultralytics_construction_ppe_v1",
        "pictor_v3",
        "chv_2021",
        "sh17",
        "ppe_cctv_topdown",
        "al_azani_kfupm_cctv",
    }
    assert candidates["sfchd_scale"]["ground_truth"]["hi_vis_scope"] == "unknown"
    assert candidates["sfchd_scale"]["ground_truth"]["declared_classes"][2] == "Safety Clothing"

    strong_camera = {
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["camera_scene"]["security_camera_fit"] == "strong"
    }
    assert {
        "sfchd_scale",
        "chv_2021",
        "ppe_cctv_topdown",
        "r2ppe_v1",
        "al_azani_kfupm_cctv",
    }.issubset(strong_camera)


def test_video_candidates_are_not_promoted_by_video_availability_alone() -> None:
    candidates = _by_id()
    r2ppe = candidates["r2ppe_v1"]
    assert r2ppe["media"]["declared_video_count"] == 26
    assert r2ppe["camera_scene"]["angle_types"] == ["ceiling_top_down", "fixed_camera"]
    assert r2ppe["rights"]["license_status"] == "verified_noncommercial"
    assert r2ppe["ground_truth"]["helmet_scope"] == "absent"
    assert r2ppe["decision"]["go_no_go"] == "methodology_only"

    al_azani = candidates["al_azani_kfupm_cctv"]
    assert al_azani["media"]["declared_video_count"] == 10
    assert al_azani["ground_truth"]["helmet_scope"] == "present"
    assert al_azani["ground_truth"]["hi_vis_scope"] == "present"
    assert al_azani["download"]["availability"] == "no_public_package"
    assert al_azani["eligibility"]["download_gate_passed"] is False


def test_repo_or_article_license_scope_is_never_silently_widened() -> None:
    candidates = _by_id()
    assert candidates["shwd"]["rights"]["license_id"] == "MIT-repository"
    assert candidates["shwd"]["rights"]["license_status"] == "missing_or_unknown"
    assert candidates["chv_2021"]["rights"]["license_status"] == "missing_or_unknown"
    assert candidates["lo_ppe_compliance_11k"]["rights"]["license_scope"] == "paper_only"
    assert candidates["ultralytics_construction_ppe_v1"]["rights"]["license_status"] == "verified_reciprocal"
    assert candidates["sh17"]["rights"]["commercial_derivative_training_allowed"] is False


def test_no_local_media_download_or_execution_receipt_is_present() -> None:
    registry = _load()
    assert registry["acquisition"] == {
        "metadata_only": True,
        "media_downloaded": False,
        "annotations_downloaded": False,
        "archives_opened": False,
        "media_decoded": False,
        "training_run": False,
        "inference_run": False,
        "gpu_used": False,
        "artifact_receipt_created": False,
        "legal_approval_receipt_created": False,
    }

    forbidden_keys = {"local_path", "media_sha256", "annotation_sha256", "receipt_path"}

    def walk(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(registry)


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
    assert report["candidate_count"] == 14


def test_validator_exposes_no_acquisition_or_training_api() -> None:
    forbidden = {
        "download",
        "fetch",
        "materialize",
        "extract",
        "decode",
        "train",
        "run_training",
        "run_inference",
        "gpu_probe",
        "docker_run",
    }
    assert forbidden.isdisjoint(set(dir(sources)))


def test_bounded_loader_rejects_symlink_duplicate_keys_and_oversize(tmp_path) -> None:
    link = tmp_path / "registry.json"
    link.symlink_to(REGISTRY_PATH)
    with pytest.raises(sources.PPEOpenGTSourceError, match="not a regular file"):
        sources._load_bounded_json(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    with pytest.raises(sources.PPEOpenGTSourceError, match="duplicate JSON key"):
        sources._load_bounded_json(duplicate)

    oversize = tmp_path / "oversize.json"
    oversize.write_bytes(b"{} " * 100)
    with pytest.raises(sources.PPEOpenGTSourceError, match="size outside bounds"):
        sources._load_bounded_json(oversize, max_bytes=32)


def test_cli_prints_pathless_report_without_writing(capsys) -> None:
    assert sources.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_count"] == 14
    assert "path" not in json.dumps(report).lower()
    assert report["gate_pass_counts"]["training_authorized"] == 0
