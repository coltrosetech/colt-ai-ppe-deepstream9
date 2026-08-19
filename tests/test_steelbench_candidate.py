import hashlib
import json
from pathlib import Path

import pytest

from validation import steelbench_candidate as steelbench


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data/manifests/steelbench-source-contract.json"
PLAN_PATH = ROOT / "data/manifests/steelbench-sample-candidates.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_metadata_package_is_valid_and_offline() -> None:
    result = steelbench.validate_checked_in_package()
    assert result == {
        "schema_version": "deepsafe.steelbench-metadata-validation/v1",
        "status": "valid_metadata_only_blocked_for_media_and_product_use",
        "source_contract_file_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "candidate_plan_file_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "candidate_count": 12,
        "media_downloaded": False,
        "gpu_or_docker_or_inference_executed": False,
    }


def test_checked_in_documents_match_json_schemas() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    pairs = (
        (
            SOURCE_PATH,
            ROOT / "validation/schemas/steelbench-source-contract-v1.schema.json",
        ),
        (
            PLAN_PATH,
            ROOT / "validation/schemas/steelbench-sample-candidates-v1.schema.json",
        ),
    )
    for document_path, schema_path in pairs:
        document = _load(document_path)
        schema = _load(schema_path)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
        assert not errors, [error.message for error in errors]


def test_source_contract_records_exact_sample_annotation_gap_and_statuses() -> None:
    source = _load(SOURCE_PATH)
    audit = source["sample_annotation_audit"]
    assert audit["confirmed_annotation_file_count"] == 47
    assert audit["confirmed_missing_annotation_count"] == 3
    assert audit["missing_annotation_clip_ids"] == [
        "clip_CO_-_ASF_FBD-1_BAGGING_AREA_20251111_001013_0041",
        "clip_RERS_RERS-1_20251115_000117_0005",
        "clip_Sinter_Plant_Flux_Screening_20251115_001517_0042",
    ]
    assert source["observed_inventory"]["sample_annotation_status_counts"] == {
        "submitted": 45,
        "flagged": 1,
        "discarded": 1,
    }
    assert audit["media_bodies_fetched"] == 0


def test_source_contract_forbids_detection_pose_distance_and_camera_geometry_claims() -> None:
    contract = _load(SOURCE_PATH)["annotation_contract"]
    for key in (
        "person_bounding_box_ground_truth",
        "ppe_object_bounding_box_ground_truth",
        "keypoint_ground_truth",
        "track_ground_truth",
        "metric_distance_ground_truth",
        "camera_angle_ground_truth",
        "person_detection_map_supported",
        "ppe_detection_map_supported",
        "pose_metric_supported",
        "distance_accuracy_metric_supported",
    ):
        assert contract[key] is False


def test_candidate_plan_has_twelve_distinct_sites_cameras_and_all_visibility_conditions() -> None:
    plan = _load(PLAN_PATH)
    candidates = plan["candidates"]
    assert len(candidates) == 12
    assert len({item["site"] for item in candidates}) == 12
    assert len({item["camera_id"] for item in candidates}) == 12
    visibility = {
        condition
        for item in candidates
        for condition in item["annotation"]["visibility_conditions"]
    }
    assert visibility == {"clear", "steam", "dust", "smoke", "low_light", "glare"}
    assert sum(item["annotation"]["layer"] == 1 for item in candidates) == 1
    assert sum(item["annotation"]["status"] == "flagged" for item in candidates) == 1


def test_lfs_and_xet_values_are_metadata_only_not_download_receipts() -> None:
    plan = _load(PLAN_PATH)
    for candidate in plan["candidates"]:
        media = candidate["media"]
        assert len(media["lfs_sha256_oid"]) == 64
        assert len(media["xet_hash"]) == 64
        assert media["downloaded"] is False
        assert media["local_path"] is None
        assert media["local_sha256"] is None
        assert candidate["camera_view"] == {
            "top_or_high_angle": None,
            "medium_close": None,
            "status": "unverified_no_media_review",
        }


def test_validator_rejects_relaxed_noncommercial_rights() -> None:
    source = _load(SOURCE_PATH)
    source["rights"]["commercial_use_allowed"] = True
    with pytest.raises(steelbench.SteelBenchContractError, match="commercial_use_allowed"):
        steelbench.validate_source_contract(source)


def test_validator_rejects_pointer_promoted_to_downloaded_media() -> None:
    plan = _load(PLAN_PATH)
    plan["candidates"][0]["media"]["downloaded"] = True
    with pytest.raises(steelbench.SteelBenchContractError, match="not downloaded"):
        steelbench.validate_candidate_plan(plan)


def test_validator_rejects_detection_map_eligibility() -> None:
    plan = _load(PLAN_PATH)
    plan["candidates"][0]["eligibility"]["detection_map"] = True
    with pytest.raises(steelbench.SteelBenchContractError, match="detection_map"):
        steelbench.validate_candidate_plan(plan)


def test_validator_rejects_unreviewed_camera_angle_claim() -> None:
    plan = _load(PLAN_PATH)
    plan["candidates"][0]["camera_view"]["top_or_high_angle"] = True
    with pytest.raises(steelbench.SteelBenchContractError, match="camera geometry"):
        steelbench.validate_candidate_plan(plan)


def test_validator_rejects_scene_layer_promoted_to_person_labels() -> None:
    plan = _load(PLAN_PATH)
    scene = next(item for item in plan["candidates"] if item["annotation"]["layer"] == 1)
    scene["annotation"]["label_scope"] = "person"
    with pytest.raises(steelbench.SteelBenchContractError, match="Layer 1"):
        steelbench.validate_candidate_plan(plan)


def test_validator_rejects_flagged_annotation_promoted_to_future_person_scoring() -> None:
    plan = _load(PLAN_PATH)
    flagged = next(item for item in plan["candidates"] if item["annotation"]["status"] == "flagged")
    flagged["eligibility"]["future_ppe_compliance_scope"] = "person"
    with pytest.raises(steelbench.SteelBenchContractError, match="future PPE scope"):
        steelbench.validate_candidate_plan(plan)


def test_candidate_plan_is_bound_to_exact_source_contract_bytes() -> None:
    plan = _load(PLAN_PATH)
    plan["source_contract_file_sha256"] = "0" * 64
    with pytest.raises(steelbench.SteelBenchContractError, match="file pin"):
        steelbench.validate_candidate_plan(plan)


def test_rmh_annotation_pin_matches_observed_hugging_face_git_oid() -> None:
    plan = _load(PLAN_PATH)
    item = next(candidate for candidate in plan["candidates"] if candidate["site"] == "RMHP")
    assert item["annotation"]["hf_git_oid"] == "ecf95d242e09bbaacddc8f4082e937531f832861"


def test_no_network_or_execution_api_is_exposed_by_metadata_validator() -> None:
    forbidden = {
        "download",
        "decode",
        "run_inference",
        "docker_run",
        "gpu_probe",
        "materialize",
    }
    assert forbidden.isdisjoint(set(dir(steelbench)))
