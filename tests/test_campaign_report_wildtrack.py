"""Focused fail-closed tests for WILDTRACK campaign-report readiness."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from validation import report_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
PLAN_RELATIVE = Path(
    "validation/results/wildtrack/person-evaluation-v1/readiness-plan.json"
)
EXPECTED_PLAN_SHA256 = (
    "5c3c22d1b5f5192f0bb111ad8e5385e548f22207d69c3f00234c76d2c239d5d4"
)


@pytest.fixture(scope="module")
def live_projection() -> tuple[dict, dict, campaign.EvidenceStore]:
    store = campaign.EvidenceStore(ROOT)
    section, facts = campaign._build_wildtrack_readiness_section(
        store,
        ROOT / "validation/results",
    )
    return section, facts, store


def test_live_wildtrack_readiness_replays_gt_plan_model_and_implementation_pins(
    live_projection,
):
    section, facts, store = live_projection

    assert facts == {
        "state": "awaiting_deepstream_predictions",
        "readiness_verified": True,
    }
    assert section["state"] == "awaiting_deepstream_predictions"
    assert section["ground_truth"] == {
        "verified": True,
        "receipt_external_sha256": (
            campaign.wildtrack_evaluation.GT_RECEIPT_EXTERNAL_SHA256
        ),
        "archive_sha256": campaign.wildtrack_evaluation.ARCHIVE_SHA256,
        "cameras": 7,
        "annotation_frames": 400,
        "camera_frames_per_profile": 2800,
        "visible_person_instances": 42721,
        "distance_evaluation_eligible_instances": 35093,
        "twenty_to_twenty_five_m_eligible_instances": 4295,
        "distance_band_boundary": "20<=d<25m",
        "preparation_geometry_quality_gate_passed": True,
    }
    assert section["evaluator"] == {
        "readiness_plan_verified": True,
        "implementation_pins_verified": True,
        "model_and_thresholds_pinned": True,
        "model_id": "rtdetrv4-s-r-livit-person-r11",
        "threshold_source_role": (
            "development_validation_seen_during_model_selection_not_independent_test"
        ),
        "expected_profiles": [640, 960],
        "expected_camera_frames_per_profile": 2800,
        "deepstream_prediction_profiles_received": [],
        "missing_prediction_profiles": [640, 960],
    }
    assert section["model_quality"] == {
        "metrics_present": False,
        "person_model_quality_measured": False,
        "paired_profile_comparison_available": False,
        "acceptance_threshold_policy_applied": False,
        "accepted": False,
    }
    assert section["acceptance_effect"] == (
        "readiness_only_no_acceptance_gate_change"
    )
    assert section["reasons"] == []
    assert len(section["evidence_ids"]) == 12
    assert all(
        store.entries[evidence_id]["state"] == "ok"
        for evidence_id in section["evidence_ids"]
    )
    assert campaign._wildtrack_readiness_projection_valid(section)
    assert hashlib.sha256((ROOT / PLAN_RELATIVE).read_bytes()).hexdigest() == (
        EXPECTED_PLAN_SHA256
    )

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            ROOT
            / "validation/schemas/validation-campaign-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(
        schema["$defs"]["wildtrack_ground_truth_readiness"]
    ).validate(section)


def test_missing_wildtrack_is_optional_incomplete_and_cannot_change_acceptance(
    tmp_path,
):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    section = report["campaigns"]["wildtrack_ground_truth_readiness"]
    requirement = next(
        item
        for item in report["requirements"]
        if item["id"] == "wildtrack_gt_evaluator_readiness"
    )

    assert section["state"] == "missing"
    assert section["ground_truth"]["verified"] is False
    assert section["evaluator"]["readiness_plan_verified"] is False
    assert section["model_quality"]["accepted"] is False
    assert requirement["state"] == "incomplete"
    assert requirement["required_for_acceptance"] is False
    assert requirement["id"] not in report["decision"]["failed_required_gates"]
    assert campaign._wildtrack_readiness_projection_valid(section)

    promoted = copy.deepcopy(report)
    promoted_requirement = next(
        item
        for item in promoted["requirements"]
        if item["id"] == "wildtrack_gt_evaluator_readiness"
    )
    promoted_requirement["state"] = "pass"
    promoted_requirement["required_for_acceptance"] = True
    with pytest.raises(ValueError, match="WILDTRACK readiness requirement"):
        campaign.validate_report_shape(promoted)


def test_readiness_plan_file_without_live_gt_model_tree_fails_closed(tmp_path):
    destination = tmp_path / PLAN_RELATIVE
    destination.parent.mkdir(parents=True)
    shutil.copy2(ROOT / PLAN_RELATIVE, destination)
    store = campaign.EvidenceStore(tmp_path)

    section, facts = campaign._build_wildtrack_readiness_section(
        store,
        tmp_path / "validation/results",
    )

    assert facts == {"state": "invalid", "readiness_verified": False}
    assert section["state"] == "invalid"
    assert section["ground_truth"]["verified"] is False
    assert section["evaluator"]["readiness_plan_verified"] is False
    assert section["model_quality"]["metrics_present"] is False
    assert section["model_quality"]["person_model_quality_measured"] is False
    assert section["model_quality"]["accepted"] is False
    assert section["reasons"] == [
        "wildtrack_ground_truth_missing_or_invalid"
    ]
    assert campaign._wildtrack_readiness_projection_valid(section)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model_quality", "metrics_present"), True),
        (("model_quality", "person_model_quality_measured"), True),
        (("model_quality", "paired_profile_comparison_available"), True),
        (("model_quality", "acceptance_threshold_policy_applied"), True),
        (("model_quality", "accepted"), True),
        (("claim_boundary", "deployment_site_acceptance"), True),
        (("claim_boundary", "commercial_deployment_authorized"), True),
        (("claim_boundary", "gpu_executed_by_reporter"), True),
    ],
)
def test_readiness_projection_rejects_every_quality_site_or_execution_overclaim(
    live_projection,
    path,
    value,
):
    section, _facts, _store = live_projection
    hostile = copy.deepcopy(section)
    hostile[path[0]][path[1]] = value

    assert not campaign._wildtrack_readiness_projection_valid(hostile)

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            ROOT
            / "validation/schemas/validation-campaign-report-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(
            schema["$defs"]["wildtrack_ground_truth_readiness"]
        ).validate(hostile)


def test_markdown_labels_wildtrack_as_readiness_not_measured_quality(
    tmp_path,
):
    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )
    markdown = campaign.render_markdown(
        report,
        tmp_path / "validation/results/campaign-report/report.md",
        tmp_path,
    )

    assert "WILDTRACK independent distance GT and evaluator readiness" in markdown
    assert "No precision, recall, F1, AP, paired profile result" in markdown
    assert (
        "becomes model-quality evidence only after both DeepStream prediction profiles"
        in markdown
    )
