from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from validation import product_acceptance_policy as policy


ROOT = Path(__file__).resolve().parents[1]


def _load_raw() -> dict:
    return json.loads(
        (ROOT / policy.POLICY_RELATIVE_PATH).read_text(encoding="utf-8")
    )


def _fixture_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for relative in (policy.POLICY_RELATIVE_PATH, policy.SCHEMA_RELATIVE_PATH):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return project


def _write_policy(project: Path, value: dict) -> None:
    (project / policy.POLICY_RELATIVE_PATH).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resign(value: dict) -> None:
    value["pre_run_fingerprint_sha256"] = policy.policy_fingerprint_sha256(value)


def test_checked_in_policy_is_exact_approved_threshold_contract() -> None:
    value = policy.load_approved_policy(ROOT)

    assert value["status"] == "approved"
    assert value["policy_role"] == "threshold_contract_only_not_an_acceptance_result"
    assert value["acceptance_state"] == "not_evaluated"
    assert value["pre_run_fingerprint_sha256"] == (
        policy.APPROVED_POLICY_FINGERPRINT_SHA256
    )
    assert policy.policy_fingerprint_sha256(value) == (
        policy.APPROVED_POLICY_FINGERPRINT_SHA256
    )

    person = value["quality_thresholds"]["person"]
    assert person["precision"] == {"operator": "gte", "threshold": 0.85}
    assert person["recall"] == {"operator": "gte", "threshold": 0.60}
    assert person["f1"] == {"operator": "gte", "threshold": 0.70}
    assert person["large_person_recall"]["threshold"] == 0.90
    assert person["large_person_recall"]["subset_definition"] == (
        "ground_truth_bbox_pixel_area_gte_96_squared_in_source_frame"
    )
    assert person["large_person_recall"]["physical_near_inference_allowed"] is False

    distance = value["quality_thresholds"]["exact_25m_person"]
    assert distance["nominal_target_distance_m"] == 25.0
    assert distance["recall"] == {"operator": "gte", "threshold": 0.70}
    assert distance["evidence_kind"] == "deployment_camera_calibrated_ground_plane"
    assert distance["ground_truth_review"] == "independent_human_reviewed"

    pose = value["quality_thresholds"]["pose"]
    assert pose == {
        "scope": "each_profile_visible_ground_truth_keypoints_only",
        "metric": "PCK",
        "threshold_radius": 0.20,
        "normalization": "ground_truth_person_bbox_max_dimension",
        "operator": "gte",
        "threshold": 0.80,
    }

    ppe = value["quality_thresholds"]["ppe"]
    assert ppe["precision"] == {"operator": "gte", "threshold": 0.85}
    assert ppe["violation_recall"] == {"operator": "gte", "threshold": 0.90}
    assert ppe["maximum_alert_latency_seconds"] == 2.0
    assert ppe["maximum_false_alarms_per_camera_hour"] == 1.0


def test_scope_cadence_capacity_and_endurance_are_literal() -> None:
    value = policy.load_approved_policy(ROOT)

    assert value["scope"] == {
        "deepstream_version": "9.0.0",
        "profiles": [640, 960],
        "camera_count": 12,
        "minimum_distinct_video_types": 10,
        "required_view_types": ["medium_close", "overhead_security_camera"],
        "required_modules": ["person", "pose", "ppe"],
        "modules_must_run_together": True,
        "performance_measurement_seconds": 300,
        "endurance_segments": 28,
        "endurance_segment_seconds": 21600,
        "endurance_total_seconds": 604800,
    }
    cadence = value["execution_cadence"]
    assert cadence["person"]["deepstream_interval"] == 0
    assert cadence["person"]["maximum_consecutive_skipped_decoded_frames"] == 0
    for name in ("pose", "ppe"):
        assert cadence[name]["maximum_deepstream_interval"] == 1
        assert cadence[name]["maximum_consecutive_skipped_decoded_frames"] == 1

    capacity = value["capacity_thresholds"]
    assert capacity["camera_count"] == 12
    assert capacity["minimum_output_fps_per_camera"] == 25.0
    assert capacity["scope"] == "each_camera_must_individually_pass"
    assert capacity["cross_camera_average_substitution_allowed"] is False

    endurance = value["endurance_thresholds"]
    assert endurance["runtime_fault_maximum_counts"] == {
        "xid": 0,
        "oom": 0,
        "fatal": 0,
        "unexpected_restart": 0,
    }
    retention = endurance["throughput_retention"]
    assert retention["statistic"] == "p05_throughput"
    assert retention["minimum_fraction_of_final_baseline"] == 0.80
    assert retention["person_only_baseline_eligible"] is False
    assert retention["baseline_kind"] == (
        "post_freeze_three_module_full_stack_same_runtime_and_sources"
    )


def test_policy_cannot_claim_acceptance_or_compensate_for_missing_evidence() -> None:
    value = policy.load_approved_policy(ROOT)
    evidence = value["evidence_rules"]

    assert evidence["policy_alone_can_pass_any_gate"] is False
    assert evidence["all_required_gates_are_conjunctive"] is True
    assert evidence["missing_invalid_unpinned_stale_or_preapproval_evidence"] == "reject"
    assert evidence["qualitative_review_counts_as_ground_truth"] is False
    assert evidence["simulated_inference_eligible"] is False
    assert evidence["real_deepstream9_gpu_execution_required"] is True


def test_threshold_tamper_is_rejected_even_if_attacker_resigns(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    forged = _load_raw()
    forged["quality_thresholds"]["person"]["recall"]["threshold"] = 0.01
    _resign(forged)
    _write_policy(project, forged)

    with pytest.raises(
        policy.AcceptancePolicyError,
        match="not the immutable owner-approved fingerprint",
    ):
        policy.load_approved_policy(project)


def test_cadence_tamper_is_rejected_even_if_attacker_resigns(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    forged = _load_raw()
    forged["execution_cadence"]["ppe"]["maximum_deepstream_interval"] = 2
    _resign(forged)
    _write_policy(project, forged)

    with pytest.raises(policy.AcceptancePolicyError, match="immutable owner-approved"):
        policy.load_approved_policy(project)


def test_added_acceptance_claim_is_rejected_even_if_resigned(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    forged = _load_raw()
    forged["acceptance_state"] = "accepted"
    forged["claimed_gate_pass"] = True
    _resign(forged)
    _write_policy(project, forged)

    with pytest.raises(policy.AcceptancePolicyError, match="immutable owner-approved"):
        policy.load_approved_policy(project)


def test_schema_tamper_is_rejected_before_it_can_relax_contract(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    schema_path = project / policy.SCHEMA_RELATIVE_PATH
    schema_path.write_text('{"type":"object"}\n', encoding="utf-8")

    with pytest.raises(policy.AcceptancePolicyError, match="schema byte count differs"):
        policy.load_approved_policy(project)


def test_policy_with_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    policy_path = project / policy.POLICY_RELATIVE_PATH
    raw = policy_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"status": "approved",',
        '"status": "approved",\n  "status": "approved",',
        1,
    )
    policy_path.write_text(raw, encoding="utf-8")

    with pytest.raises(policy.AcceptancePolicyError, match="strict JSON"):
        policy.load_approved_policy(project)


def test_policy_symlink_is_rejected(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    policy_path = project / policy.POLICY_RELATIVE_PATH
    target = project / "elsewhere.json"
    shutil.copy2(policy_path, target)
    policy_path.unlink()
    policy_path.symlink_to(target)

    with pytest.raises(policy.AcceptancePolicyError, match="cannot safely read"):
        policy.load_approved_policy(project)


def test_measurement_must_not_predate_approval(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)

    with pytest.raises(policy.AcceptancePolicyError, match="predates"):
        policy.validate_measurement_start(
            "2026-07-17T18:13:38Z", project_root=project
        )
    accepted = policy.validate_measurement_start(
        "2026-07-17T18:13:39Z", project_root=project
    )
    assert accepted["pre_run_fingerprint_sha256"] == (
        policy.APPROVED_POLICY_FINGERPRINT_SHA256
    )


def test_schema_pin_matches_exact_checked_in_bytes() -> None:
    value = _load_raw()
    schema_bytes = (ROOT / policy.SCHEMA_RELATIVE_PATH).read_bytes()
    assert len(schema_bytes) == policy.SCHEMA_BYTES
    assert hashlib.sha256(schema_bytes).hexdigest() == policy.SCHEMA_SHA256
    assert value["pre_run_immutability"]["schema_binding"] == {
        "path": policy.SCHEMA_RELATIVE_PATH.as_posix(),
        "bytes": policy.SCHEMA_BYTES,
        "sha256": policy.SCHEMA_SHA256,
    }


def test_schema_validates_policy_when_jsonschema_is_available() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / policy.SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(_load_raw())
