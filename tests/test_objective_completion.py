from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from admin import validation as admin_validation
from admin.app import app
from tests.admin_lineage_fixtures import (
    artifact_pin,
    write_admin_endurance_lineage,
    write_json as write_admin_json,
)
import validation.objective_completion as objective_completion
import validation.finalize_validation as validation_finalizer
from validation.objective_completion import (
    CONTRACT_ID,
    GATE_ORDER,
    SCHEMA_VERSION,
    EvidenceError,
    EvidenceReader,
    _canonical_bytes,
    _canonical_fingerprint_valid,
    build_report,
    render_markdown,
    validate_report_shape,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT
    / "validation/schemas/validation-objective-completion-v1.schema.json"
)


@pytest.fixture(scope="module")
def live_report() -> dict:
    return build_report(PROJECT_ROOT)


def test_live_report_has_five_non_endurance_gates_and_fail_closed_decision(
    live_report: dict,
) -> None:
    assert live_report["schema_version"] == SCHEMA_VERSION
    assert live_report["contract_id"] == CONTRACT_ID
    gates = live_report["objective"]["gates"]
    assert [gate["id"] for gate in gates] == list(GATE_ORDER)
    assert all(gate["state"] == "pass" for gate in gates[:5])
    assert gates[5]["state"] in {"pending", "pass", "invalid"}
    expected_complete = gates[5]["state"] == "pass"
    assert live_report["objective"]["evidence_complete"] is expected_complete
    expected_state = (
        "complete"
        if expected_complete
        else "invalid"
        if gates[5]["state"] == "invalid"
        else "incomplete"
    )
    assert live_report["objective"]["state"] == expected_state


def test_live_report_validates_against_strict_json_schema(live_report: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(live_report)
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["properties"]["contract_id"]["const"] == CONTRACT_ID
    assert schema["properties"]["objective"]["additionalProperties"] is False
    assert schema["$defs"]["evidence"]["additionalProperties"] is False
    assert validate_report_shape(live_report) == []


def test_report_and_markdown_writes_are_deterministic(
    live_report: dict, tmp_path: Path
) -> None:
    write_report(live_report, tmp_path)
    first_json = (tmp_path / "report.json").read_bytes()
    first_markdown = (tmp_path / "report.md").read_bytes()
    write_report(copy.deepcopy(live_report), tmp_path)
    assert (tmp_path / "report.json").read_bytes() == first_json
    assert (tmp_path / "report.md").read_bytes() == first_markdown
    assert render_markdown(live_report).encode("utf-8") == first_markdown


def test_report_fingerprint_is_canonical_and_tamper_evident(live_report: dict) -> None:
    assert _canonical_fingerprint_valid(live_report)
    unsigned = copy.deepcopy(live_report)
    expected = unsigned.pop("fingerprint_sha256")
    assert hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() == expected
    tampered = copy.deepcopy(live_report)
    tampered["objective"]["passed_gate_count"] = 0
    assert not _canonical_fingerprint_valid(tampered)
    assert "report_fingerprint_invalid" in validate_report_shape(tampered)


def test_guardrails_preserve_objective_product_and_quality_separation(
    live_report: dict,
) -> None:
    objective = live_report["objective"]
    quality = live_report["quality_context"]
    assert objective["does_not_imply_product_readiness"] is True
    assert quality["acceptance_effect_on_objective_completion"] == "none"
    assert quality["product_readiness_decision"] == "out_of_scope_separate_truth"
    assert quality["person_quality_decision"] == "not_made_by_this_ledger"
    assert quality["calibrated_25m_decision"] == "not_made_by_this_ledger"
    assert quality["pose_decision"] == "not_made_by_this_ledger"
    assert quality["ppe_decision"] == "not_made_by_this_ledger"
    assert quality["rlivit_quality_threshold_applied"] is False
    assert set(quality["observed_limitations"]) == {
        "rlivit_recall_is_low_observation_without_owner_quality_threshold",
        "top_view_ai_audit_contains_high_severity_undercoverage_finding",
        "loaf_artifacts_do_not_satisfy_calibrated_25m_detection",
    }


def test_scene_semantics_do_not_overclaim_camera_or_unique_footage(
    live_report: dict,
) -> None:
    matrix = live_report["observations"]["video_matrix"]
    assert matrix["distinct_video_type_count"] >= 10
    assert matrix["profiles"] == [640, 960]
    assert matrix["streams_per_run"] == 12
    assert matrix["measurement_seconds_per_run"] == 300
    assert matrix["medium_close_present"] is True
    assert matrix["overhead_or_high_angle_present"] is True
    assert matrix["simulation_semantics"] == (
        "twelve_copies_of_one_scene_with_file_loop_per_run"
    )
    assert matrix["resolution_semantics"] == (
        "model_input_active_area_not_camera_capture_resolution"
    )
    assert matrix["unique_footage_claimed"] is False
    assert matrix["verified_source_media_count"] >= 10
    assert live_report["observations"]["paired_comparison"][
        "batch_aggregate_semantic_replay_verified"
    ] is True


class _VisualAuditReader:
    """Minimal reader double for semantic negative tests around the live corpus."""

    def __init__(self, audit: dict, index: dict) -> None:
        self.audit = audit
        self.index = index

    def read_json(self, evidence_id: str, _path: Path) -> dict:
        assert evidence_id == "visual_ai_audit"
        return self.audit

    def verify_pin(
        self,
        evidence_id: str,
        _pin: object,
        *,
        capture: bool = False,
    ) -> bytes | None:
        if evidence_id == "visual_manual_index_pin":
            assert capture is True
            return _canonical_bytes(self.index)
        assert evidence_id.startswith("visual_asset_")
        assert capture is False
        return None


def _live_visual_documents() -> tuple[dict, dict]:
    audit = json.loads(
        (PROJECT_ROOT / objective_completion.VISUAL_AUDIT).read_text(
            encoding="utf-8"
        )
    )
    index = json.loads(
        (PROJECT_ROOT / objective_completion.MANUAL_INDEX).read_text(
            encoding="utf-8"
        )
    )
    return audit, index


def _validated_live_visual() -> tuple[dict, dict]:
    audit, index = _live_visual_documents()
    observation, _evidence, _severity, context = (
        objective_completion._validate_visual_audit(
            _VisualAuditReader(audit, index)  # type: ignore[arg-type]
        )
    )
    return observation, context


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda audit: audit["reviewer"].update(reviewer_type="human"),
            "visual_reviewer_contract_invalid",
        ),
        (
            lambda audit: audit.update(ground_truth_available=True),
            "visual_ground_truth_contract_invalid",
        ),
        (
            lambda audit: audit["methodology"].update(
                dense_annotation_performed=True
            ),
            "visual_methodology_contract_invalid",
        ),
        (
            lambda audit: audit.update(guardrail={}),
            "visual_guardrail_contract_invalid",
        ),
        (
            lambda audit: audit["scope"].update(
                source_review_count=999,
            ),
            "visual_scope_counts_mismatch",
        ),
        (
            lambda audit: audit["scope"].update(
                profile_decision_count=999,
            ),
            "visual_scope_counts_mismatch",
        ),
        (
            lambda audit: audit.update(task="pose_visual_audit"),
            "visual_task_contract_invalid",
        ),
        (
            lambda audit: audit["input_bindings"].update(
                campaign_plan_sha256="0" * 64
            ),
            "visual_input_binding_mismatch",
        ),
        (
            lambda audit: audit["input_bindings"][
                "manual_assets_index"
            ].update(schema_version="deepsafe.open-video-manual-assets/v2"),
            "visual_input_binding_mismatch",
        ),
        (
            lambda audit: audit["input_bindings"][
                "manual_assets_index"
            ].update(bundle_id="b_forged"),
            "visual_input_binding_mismatch",
        ),
    ],
)
def test_visual_audit_semantic_claims_fail_closed(
    mutation: object,
    expected_error: str,
) -> None:
    audit, index = _live_visual_documents()
    mutation(audit)  # type: ignore[operator]

    with pytest.raises(EvidenceError, match=expected_error):
        objective_completion._validate_visual_audit(
            _VisualAuditReader(audit, index)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        (
            ("ai_qualitative_visual_audit", "source_frame_count"),
            "visual_campaign_projection_count_mismatch",
        ),
        (
            ("ai_qualitative_visual_audit", "expected_source_frame_count"),
            "visual_campaign_projection_count_mismatch",
        ),
        (
            ("ai_qualitative_visual_audit", "reviewed_scene_ids"),
            "visual_campaign_projection_count_mismatch",
        ),
        (
            (
                "automatic_candidate_generation",
                "candidate_assets",
                "asset_count",
            ),
            "visual_campaign_projection_asset_count_mismatch",
        ),
        (
            (
                "automatic_candidate_generation",
                "candidate_assets",
                "expected_asset_count",
            ),
            "visual_campaign_projection_asset_count_mismatch",
        ),
        (
            ("paired_profile_comparison", "paired_source_frame_count"),
            "visual_campaign_projection_pair_count_mismatch",
        ),
        (
            ("paired_profile_comparison", "pairs"),
            "visual_campaign_projection_pair_count_mismatch",
        ),
    ],
)
def test_visual_campaign_projection_counts_fail_closed(
    path: tuple[str, ...],
    expected_error: str,
) -> None:
    report = json.loads(
        (PROJECT_ROOT / objective_completion.CAMPAIGN_REPORT).read_text(
            encoding="utf-8"
        )
    )
    projection = report["campaigns"]["open_video_manual_review"]
    visual_observation, visual_context = _validated_live_visual()
    target = projection
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = 999

    with pytest.raises(EvidenceError, match=expected_error):
        objective_completion._validate_visual_campaign_projection(
            projection,
            visual_observation,
            visual_context,
        )


@pytest.mark.parametrize("field", ["scene_id", "frame_index", "source_observation"])
def test_visual_audit_rejects_each_cross_profile_source_identity_mismatch(
    field: str,
) -> None:
    audit, index = _live_visual_documents()
    source_id = index["decisions"][0]["source_review_id"]
    decision = next(
        item
        for item in index["decisions"]
        if item["source_review_id"] == source_id and item["model_input"] == 960
    )
    if field == "scene_id":
        decision[field] = f'{decision[field]}_different'
        for asset_id in decision["evidence"].values():
            index["assets"][asset_id][field] = decision[field]
    elif field == "frame_index":
        decision[field] += 7
        for asset_id in decision["evidence"].values():
            index["assets"][asset_id][field] = decision[field]
    else:
        decision[field] = {
            **decision[field],
            "medium_close": not decision[field]["medium_close"],
        }

    with pytest.raises(EvidenceError, match="visual_profile_pair_source_mismatch"):
        objective_completion._validate_visual_audit(
            _VisualAuditReader(audit, index)  # type: ignore[arg-type]
        )


def test_visual_audit_requires_substantive_review_records() -> None:
    audit, index = _live_visual_documents()
    audit["reviews"] = [
        {
            "source_review_id": review["source_review_id"],
            "decision_ids": review["decision_ids"],
        }
        for review in audit["reviews"]
    ]

    with pytest.raises(EvidenceError, match="visual_review_schema_invalid"):
        objective_completion._validate_visual_audit(
            _VisualAuditReader(audit, index)  # type: ignore[arg-type]
        )


def test_visual_audit_binds_profile_pair_to_same_source_image_content() -> None:
    audit, index = _live_visual_documents()
    decisions_960 = [
        decision
        for decision in index["decisions"]
        if decision["model_input"] == 960
    ]
    asset_ids = [
        decisions_960[position]["evidence"]["source_image"]
        for position in (0, 1)
    ]
    assets = [index["assets"][asset_id] for asset_id in asset_ids]
    assert assets[0]["sha256"] != assets[1]["sha256"]
    fields = ("relative_path", "size_bytes", "sha256")
    first_pin = {field: assets[0][field] for field in fields}
    second_pin = {field: assets[1][field] for field in fields}
    assets[0].update(second_pin)
    assets[1].update(first_pin)

    with pytest.raises(
        EvidenceError,
        match="visual_profile_pair_source_image_mismatch",
    ):
        objective_completion._validate_visual_audit(
            _VisualAuditReader(audit, index)  # type: ignore[arg-type]
        )


def test_build_report_marks_visual_gate_invalid_on_projection_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_projection(*_args: object) -> None:
        raise EvidenceError("forced_visual_projection_rejection")

    monkeypatch.setattr(
        objective_completion,
        "_validate_visual_campaign_projection",
        reject_projection,
    )
    report = build_report(PROJECT_ROOT)
    visual_gate = next(
        gate
        for gate in report["objective"]["gates"]
        if gate["id"] == "hash_bound_gt_free_visual_inspection"
    )

    assert visual_gate["state"] == "invalid"
    assert visual_gate["passed"] is False
    assert visual_gate["reasons"] == ["forced_visual_projection_rejection"]


def test_evidence_reader_rejects_traversal_symlink_and_pin_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe").mkdir()
    payload = b'{"ok":true}\n'
    artifact = tmp_path / "safe/artifact.json"
    artifact.write_bytes(payload)
    reader = EvidenceReader(tmp_path)
    pin = {
        "path": "safe/artifact.json",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert reader.verify_pin("good", pin, capture=True) == payload
    assert reader.verify_pin("good_streamed", pin) is None
    with pytest.raises(EvidenceError, match="pin_path_invalid"):
        reader.read_bytes("traversal", "../artifact.json")
    with pytest.raises(EvidenceError, match="pin_path_invalid"):
        reader.read_bytes("ambiguous", "safe//artifact.json")
    bad_pin = {**pin, "sha256": "0" * 64}
    with pytest.raises(EvidenceError, match="pin_mismatch"):
        reader.verify_pin("bad_hash", bad_pin)
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "safe", target_is_directory=True)
    with pytest.raises(EvidenceError, match="evidence_symlink_forbidden"):
        reader.read_bytes("symlink", "linked/artifact.json")
    hardlink = tmp_path / "safe/hardlink.json"
    os.link(artifact, hardlink)
    with pytest.raises(EvidenceError, match="evidence_hardlink_forbidden"):
        reader.read_bytes("hardlink", "safe/hardlink.json")


def test_evidence_reader_rejects_duplicate_keys_and_size_overrun(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    reader = EvidenceReader(tmp_path)
    with pytest.raises(EvidenceError, match="strict_json_invalid"):
        reader.read_json("duplicate", "duplicate.json")
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"1234")
    with pytest.raises(EvidenceError, match="evidence_size_limit_exceeded"):
        reader.read_bytes("oversized", "regular.bin", max_bytes=3)


def test_streamed_pin_verification_does_not_materialize_large_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"1234"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(payload)
    pin = {
        "path": "artifact.bin",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    reader = EvidenceReader(tmp_path)
    monkeypatch.setattr(objective_completion, "MAX_JSON_BYTES", 3)
    assert reader.verify_pin("streamed", pin) is None
    with pytest.raises(EvidenceError, match="pin_capture_size_limit_exceeded"):
        reader.verify_pin("captured", pin, capture=True)


def test_directory_fd_walk_rejects_parent_swap_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe/artifact.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "replacement").mkdir()
    (tmp_path / "replacement/artifact.json").write_text(
        '{"attacker":true}\n', encoding="utf-8"
    )
    reader = EvidenceReader(tmp_path)
    real_open = objective_completion.os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "safe" and dir_fd is not None and not swapped:
            swapped = True
            os.rename(tmp_path / "safe", tmp_path / "original")
            os.rename(tmp_path / "replacement", tmp_path / "safe")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(objective_completion.os, "open", swapping_open)
    with pytest.raises(EvidenceError, match="evidence_changed_during_path_walk"):
        reader.read_bytes("raced", "safe/artifact.json")


def test_deep_json_and_fingerprint_inputs_fail_closed(tmp_path: Path) -> None:
    deeply_nested_json = '{"value":' + "[" * 2000 + "0" + "]" * 2000 + "}\n"
    (tmp_path / "deep.json").write_text(deeply_nested_json, encoding="utf-8")
    reader = EvidenceReader(tmp_path)
    with pytest.raises(EvidenceError, match="json_nesting_limit_exceeded"):
        reader.read_json("deep", "deep.json")

    nested: object = 0
    for _ in range(2000):
        nested = [nested]
    candidate = {"nested": nested, "fingerprint_sha256": "0" * 64}
    assert _canonical_fingerprint_valid(candidate) is False


def test_schema_and_shape_reject_unknown_top_level_property(live_report: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(live_report)
    tampered["unexpected"] = True
    unsigned = copy.deepcopy(tampered)
    unsigned.pop("fingerprint_sha256")
    tampered["fingerprint_sha256"] = hashlib.sha256(
        _canonical_bytes(unsigned)
    ).hexdigest()
    assert schema["additionalProperties"] is False
    assert "report_top_level_shape_invalid" in validate_report_shape(tampered)


def _write_admin_report(root: Path, report: dict) -> None:
    path = root / "objective-completion/current/report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resign_admin_report(report: dict) -> None:
    report.pop("fingerprint_sha256", None)
    report["fingerprint_sha256"] = hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()


def _admin_reader(
    results: Path, *, workspace_root: Path = PROJECT_ROOT
) -> admin_validation.ArtifactReader:
    return admin_validation.ArtifactReader(
        results,
        workspace_root=workspace_root,
        schema_root=workspace_root / "validation/schemas",
    )


def _bind_admin_report_to_live_lineage(
    results: Path,
    report: dict,
    *,
    config_fingerprint: str = "a" * 64,
    static_input_fingerprint: str = "b" * 64,
) -> dict[str, dict]:
    lineage = write_admin_endurance_lineage(
        results,
        config_fingerprint=config_fingerprint,
        static_input_fingerprint=static_input_fingerprint,
    )
    campaign_report = {
        "schema_version": "deepsafe.validation-campaign-report/v1",
        "generated_at_utc": "2026-07-17T00:00:02+00:00",
        "decision": {
            "status": "preliminary",
            "accepted": False,
            "final_claim_allowed": False,
        },
        "requirement_summary": {"state_counts": {"incomplete": 1}},
        "requirements": [{"id": "seven_day_endurance", "state": "incomplete"}],
        "campaigns": {},
        "evidence": list(lineage.values()),
    }
    write_admin_json(
        results / "campaign-report/report.json", campaign_report
    )
    replacements = {
        "campaign_report": artifact_pin(
            results,
            "campaign-report/report.json",
            "campaign_report",
        ),
        "endurance_checkpoint": {
            key: value
            for key, value in lineage["endurance_checkpoint"].items()
            if key in {"id", "path", "size_bytes", "sha256"}
        },
        "endurance_status": {
            key: value
            for key, value in lineage["endurance_status"].items()
            if key in {"id", "path", "size_bytes", "sha256"}
        },
    }
    report["evidence"] = [
        replacements.get(row.get("id"), row)
        for row in report["evidence"]
    ]
    _resign_admin_report(report)
    return lineage


def test_admin_api_projects_objective_as_read_only_separate_truth(
    live_report: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "validation-results"
    report = copy.deepcopy(live_report)
    _bind_admin_report_to_live_lineage(results, report)
    write_report(report, results / "objective-completion/current")
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(PROJECT_ROOT / "validation/schemas"),
    )

    with TestClient(app) as client:
        response = client.get("/api/validation")
        raw_json = client.get(
            "/api/validation", params={"artifact": "objective_completion_json"}
        )
        raw_markdown = client.get(
            "/api/validation",
            params={"artifact": "objective_completion_markdown"},
        )
        page = client.get("/")

    assert response.status_code == 200
    projected = response.json()["campaigns"]["objective_completion"]
    assert projected["state"] == report["objective"]["state"]
    assert projected["evidence_complete"] is report["objective"][
        "evidence_complete"
    ]
    assert projected["read_only"] is True
    assert projected["execution_actions_available"] is False
    assert projected["does_not_imply_product_readiness"] is True
    assert projected["product_readiness_decision"] == "separate_truth"
    assert projected["progress"]["completed"] == report["objective"][
        "passed_gate_count"
    ]
    assert projected["progress"]["total"] == len(GATE_ORDER)
    assert [gate["id"] for gate in projected["required_gates"]] == list(
        GATE_ORDER
    )
    assert projected["caveats"]
    assert raw_json.status_code == 404
    assert raw_markdown.status_code == 404
    assert page.status_code == 200
    assert "campaign.evidence_complete" in page.text
    assert "campaign.required_gates" in page.text
    assert "campaign.caveats" in page.text


def test_admin_projection_rejects_schema_valid_fingerprint_tamper(
    live_report: dict, tmp_path: Path
) -> None:
    report = copy.deepcopy(live_report)
    report["objective"]["gates"][0]["reasons"] = ["tampered_reason"]
    results = tmp_path / "validation-results"
    _write_admin_report(results, report)
    reader = _admin_reader(results)

    assert reader.validates_schema(
        report, admin_validation.OBJECTIVE_COMPLETION_SCHEMA
    )
    projected = admin_validation._objective_completion(reader)
    assert projected["state"] == "artifact_error"
    assert projected["evidence_complete"] is False
    assert projected["required_gates"] == []


def test_admin_projection_rejects_resigned_count_inconsistency(
    live_report: dict, tmp_path: Path
) -> None:
    report = copy.deepcopy(live_report)
    report["objective"]["passed_gate_count"] = 0
    _resign_admin_report(report)
    results = tmp_path / "validation-results"
    _write_admin_report(results, report)
    reader = _admin_reader(results)

    assert reader.validates_schema(
        report, admin_validation.OBJECTIVE_COMPLETION_SCHEMA
    )
    assert admin_validation._canonical_fingerprint_matches(report)
    projected = admin_validation._objective_completion(reader)
    assert projected["state"] == "artifact_error"
    assert projected["progress"]["completed"] == 0
    assert projected["required_gates"] == []


def test_admin_projection_is_fail_closed_when_objective_report_is_missing(
    tmp_path: Path,
) -> None:
    results = tmp_path / "validation-results"
    results.mkdir()
    projected = admin_validation._objective_completion(_admin_reader(results))

    assert projected["state"] == "not_started"
    assert projected["available"] is False
    assert projected["evidence_complete"] is False
    assert projected["progress"] == {
        "completed": 0,
        "total": 6,
        "remaining": 6,
        "fraction": 0.0,
    }
    assert projected["required_gates"] == []


def test_admin_requires_allowlisted_draft_2020_objective_schema(
    live_report: dict, tmp_path: Path
) -> None:
    results = tmp_path / "validation-results"
    _write_admin_report(results, copy.deepcopy(live_report))
    schema_root = tmp_path / "schemas"
    schema_root.mkdir()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    (schema_root / admin_validation.OBJECTIVE_COMPLETION_SCHEMA).write_text(
        json.dumps(schema, sort_keys=True) + "\n", encoding="utf-8"
    )
    reader = admin_validation.ArtifactReader(
        results,
        workspace_root=PROJECT_ROOT,
        schema_root=schema_root,
    )

    assert reader.validates_schema(
        live_report, admin_validation.OBJECTIVE_COMPLETION_SCHEMA
    ) is False
    projected = admin_validation._objective_completion(reader)
    assert projected["state"] == "artifact_error"
    assert projected["required_gates"] == []


def test_admin_objective_reader_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    results = tmp_path / "validation-results"
    path = results / "objective-completion/current/report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":"deepsafe.validation-objective-completion/v1",'
        '"schema_version":"deepsafe.validation-objective-completion/v1"}\n',
        encoding="utf-8",
    )
    reader = _admin_reader(results)

    assert reader.read("objective_completion_json").state == "invalid_json"
    projected = admin_validation._objective_completion(reader)
    assert projected["state"] == "artifact_error"
    assert projected["evidence_complete"] is False


def _write_finalization_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    results = workspace / "validation/results"
    lineage = write_admin_endurance_lineage(
        results, state="complete", validated_seconds=604800
    )
    write_admin_json(
        results / "endurance/current/plan.json",
        {
            "schema_version": "deepsafe.endurance-plan/v1",
            "campaign_name": "deepstream9-12-camera-seven-day",
        },
    )
    for relative in admin_validation.FINALIZATION_INPUT_PATHS:
        if relative.startswith("validation/results/"):
            continue
        source = PROJECT_ROOT / relative
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    reports: dict[str, dict] = {
        "campaign_json": {
            "schema_version": "deepsafe.validation-campaign-report/v1",
            "decision": {
                "status": "blocked_by_hardware",
                "accepted": False,
                "final_claim_allowed": False,
            },
            "campaigns": {
                "endurance": {
                    "accepted": True,
                    "evidence_complete": True,
                    "target_validated_seconds": 604800,
                    "reported_validated_seconds": 604800,
                    "expected_segments": 28,
                    "healthy_checkpoint_segments": 28,
                    "verified_attempt_receipts": 28,
                }
            },
            "requirements": [{"id": "seven_day_endurance", "state": "pass"}],
            "evidence": list(lineage.values()),
        },
        "objective_json": {
            "schema_version": "deepsafe.validation-objective-completion/v1",
            "objective": {
                "state": "complete",
                "evidence_complete": True,
                "passed_gate_count": 6,
            },
        },
        "product_json": {
            "schema_version": "deepsafe.product-readiness/v1",
            "decision": {
                "status": "not_ready",
                "ready": False,
                "final_claim_allowed": False,
            },
        },
    }
    for key in ("objective_json", "product_json"):
        reports[key]["fingerprint_sha256"] = admin_validation._canonical_sha256(
            reports[key]
        )

    outputs: list[dict] = []
    for artifact_id, full_path, media_type, _artifact_key in (
        admin_validation.FINALIZATION_OUTPUTS
    ):
        relative = full_path.removeprefix("validation/results/")
        path = results / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_id in reports:
            content = (
                json.dumps(
                    reports[artifact_id],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
        else:
            content = f"# {artifact_id}\n".encode("utf-8")
        path.write_bytes(content)
        outputs.append(
            {
                "id": artifact_id,
                "path": full_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "media_type": media_type,
            }
        )

    identity = {
        "campaign_name": "deepstream9-12-camera-seven-day",
        "config_fingerprint": "a" * 64,
        "static_input_fingerprint": "b" * 64,
        "started_at_utc": "2026-07-17T00:00:00+00:00",
        "updated_at_utc": "2026-07-24T00:00:00+00:00",
        "finished_at_utc": "2026-07-24T00:00:00+00:00",
        "target_validated_seconds": 604800,
        "validated_seconds": 604800,
        "total_attempt_receipt_count": 28,
    }
    receipt_inputs: list[dict] = []
    for relative in admin_validation.FINALIZATION_INPUT_PATHS:
        content = (workspace / relative).read_bytes()
        receipt_inputs.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    receipt = {
        "schema_version": "deepsafe.validation-finalization-receipt/v1",
        "state": "complete",
        "completed_at_utc": "2026-07-24T00:00:01+00:00",
        "finalization_identity_sha256": admin_validation._canonical_sha256(
            {
                "completion_identity": identity,
                "inputs": receipt_inputs,
            }
        ),
        "completion_identity": identity,
        "inputs": receipt_inputs,
        "outputs": outputs,
        "semantics": {
            "campaign_endurance_accepted": True,
            "objective_evidence_complete": True,
            "objective_passed_gate_count": 6,
            "product_snapshot_valid": True,
            "product_status": "not_ready",
            "product_ready_required": False,
            "campaign_overall_accepted": False,
            "objective_fingerprint_sha256": reports["objective_json"][
                "fingerprint_sha256"
            ],
            "product_fingerprint_sha256": reports["product_json"][
                "fingerprint_sha256"
            ],
        },
        "generator_commands": [
            ["python", "-m", "validation.report_campaign"],
            ["python", "-m", "validation.objective_completion"],
            ["python", "-m", "validation.product_readiness"],
        ],
        "lock_contract": {
            "exclusive_nonblocking_finalizer_lock": True,
            "exclusive_nonblocking_supervisor_lock": True,
            "supervisor_lock_held_through_receipt_commit": True,
            "receipt_committed_last": True,
        },
    }
    receipt["fingerprint_sha256"] = admin_validation._canonical_sha256(receipt)
    receipt_path = results / "finalization/current/receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, receipt_path, workspace


def _set_admin_finalization_environment(
    monkeypatch: pytest.MonkeyPatch,
    results: Path,
    *,
    workspace_root: Path = PROJECT_ROOT,
) -> None:
    monkeypatch.setenv("DEEPSAFE_VALIDATION_ROOT", str(results))
    monkeypatch.setenv("DEEPSAFE_VALIDATION_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv(
        "DEEPSAFE_VALIDATION_SCHEMA_ROOT",
        str(workspace_root / "validation/schemas"),
    )


def _resign_finalization_receipt(receipt: dict) -> None:
    receipt["finalization_identity_sha256"] = admin_validation._canonical_sha256(
        {
            "completion_identity": receipt["completion_identity"],
            "inputs": receipt["inputs"],
        }
    )
    receipt.pop("fingerprint_sha256", None)
    receipt["fingerprint_sha256"] = admin_validation._canonical_sha256(receipt)


def test_admin_finalization_input_contract_matches_publisher() -> None:
    assert admin_validation.FINALIZATION_INPUT_PATHS == (
        validation_finalizer.INPUT_RELATIVE_PATHS
    )


def test_admin_finalization_bundle_is_pending_when_receipt_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "validation-results"
    results.mkdir()
    _set_admin_finalization_environment(monkeypatch, results)

    with TestClient(app) as client:
        payload = client.get("/api/validation").json()

    assert payload["finalization_bundle"] == {
        "label": "Doğrulama bundle commit",
        "available": False,
        "state": "pending",
        "committed": False,
        "reason": "receipt_missing",
        "read_only": True,
        "raw_download_allowed": False,
        "verified_output_count": 0,
        "output_count": 6,
    }
    assert payload["campaigns"]["campaign_report"]["finalized"] is False
    assert payload["campaigns"]["objective_completion"]["finalized"] is False
    assert payload["campaigns"]["product_readiness"]["finalized"] is False


def test_admin_finalization_bundle_commits_only_exact_live_input_output_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _receipt_path, workspace = _write_finalization_fixture(tmp_path)
    _set_admin_finalization_environment(
        monkeypatch, results, workspace_root=workspace
    )

    with TestClient(app) as client:
        payload = client.get("/api/validation").json()

    bundle = payload["finalization_bundle"]
    assert bundle["state"] == "complete"
    assert bundle["committed"] is True
    assert bundle["verified_output_count"] == 6
    assert bundle["product_status"] == "not_ready"
    assert payload["campaigns"]["campaign_report"]["finalized"] is True
    assert payload["campaigns"]["objective_completion"]["finalized"] is True
    assert payload["campaigns"]["product_readiness"]["finalized"] is True


def test_admin_finalization_bundle_fails_closed_on_live_output_mismatch(
    tmp_path: Path,
) -> None:
    results, _receipt_path, workspace = _write_finalization_fixture(tmp_path)
    (results / "campaign-report/report.md").write_text(
        "# changed after commit\n", encoding="utf-8"
    )

    bundle = admin_validation._finalization_bundle(
        _admin_reader(results, workspace_root=workspace)
    )

    assert bundle["state"] == "invalid"
    assert bundle["committed"] is False
    assert bundle["reason"] == "output_mismatch"


def test_admin_finalization_bundle_marks_live_input_tamper_stale_and_unfinalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _receipt_path, workspace = _write_finalization_fixture(tmp_path)
    plan_path = results / "endurance/current/plan.json"
    original = plan_path.read_bytes()
    plan_path.write_bytes(b"[" + original[1:])
    assert plan_path.stat().st_size == len(original)
    _set_admin_finalization_environment(
        monkeypatch, results, workspace_root=workspace
    )

    with TestClient(app) as client:
        payload = client.get("/api/validation").json()

    assert payload["finalization_bundle"]["state"] == "stale_lineage"
    assert payload["finalization_bundle"]["reason"] == "stale_lineage"
    assert payload["finalization_bundle"]["committed"] is False
    for campaign_id in (
        "campaign_report",
        "objective_completion",
        "product_readiness",
    ):
        assert payload["campaigns"][campaign_id]["finalized"] is False


def test_admin_finalization_bundle_marks_new_endurance_lineage_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, receipt_path, workspace = _write_finalization_fixture(tmp_path)
    write_admin_endurance_lineage(
        results,
        config_fingerprint="c" * 64,
        static_input_fingerprint="d" * 64,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for pin in receipt["inputs"][:3]:
        content = (workspace / pin["path"]).read_bytes()
        pin["size_bytes"] = len(content)
        pin["sha256"] = hashlib.sha256(content).hexdigest()
    _resign_finalization_receipt(receipt)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    _set_admin_finalization_environment(
        monkeypatch, results, workspace_root=workspace
    )

    with TestClient(app) as client:
        payload = client.get("/api/validation").json()

    assert payload["finalization_bundle"]["state"] == "stale_lineage"
    assert payload["finalization_bundle"]["reason"] == "stale_lineage"
    assert payload["finalization_bundle"]["committed"] is False
    for campaign_id in (
        "campaign_report",
        "objective_completion",
        "product_readiness",
    ):
        assert payload["campaigns"][campaign_id]["finalized"] is False


def test_admin_finalization_bundle_rejects_resigned_input_path_substitution(
    tmp_path: Path,
) -> None:
    results, receipt_path, workspace = _write_finalization_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"][-1]["path"] = (
        "validation/schemas/substituted-product-readiness-v1.schema.json"
    )
    _resign_finalization_receipt(receipt)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    bundle = admin_validation._finalization_bundle(
        _admin_reader(results, workspace_root=workspace)
    )

    assert bundle["state"] == "invalid"
    assert bundle["committed"] is False
    assert bundle["reason"] == "input_pin_contract_invalid"


def test_admin_finalization_bundle_rejects_symlinked_live_input(
    tmp_path: Path,
) -> None:
    results, _receipt_path, workspace = _write_finalization_fixture(tmp_path)
    plan_path = results / "endurance/current/plan.json"
    target = plan_path.with_name("plan-target.json")
    target.write_bytes(plan_path.read_bytes())
    plan_path.unlink()
    plan_path.symlink_to(target.name)

    bundle = admin_validation._finalization_bundle(
        _admin_reader(results, workspace_root=workspace)
    )

    assert bundle["state"] == "stale_lineage"
    assert bundle["committed"] is False
    assert bundle["reason"] == "stale_lineage"


def test_admin_finalization_bundle_rejects_oversized_input_pin_claim(
    tmp_path: Path,
) -> None:
    results, receipt_path, workspace = _write_finalization_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["inputs"][3]["size_bytes"] = (
        admin_validation.HARD_MAX_ARTIFACT_OVERRIDE_BYTES + 1
    )
    _resign_finalization_receipt(receipt)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    bundle = admin_validation._finalization_bundle(
        _admin_reader(results, workspace_root=workspace)
    )

    assert bundle["state"] == "stale_lineage"
    assert bundle["committed"] is False
    assert bundle["reason"] == "stale_lineage"


def test_admin_finalization_bundle_rejects_forged_self_fingerprint(
    tmp_path: Path,
) -> None:
    results, receipt_path, workspace = _write_finalization_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["fingerprint_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    bundle = admin_validation._finalization_bundle(
        _admin_reader(results, workspace_root=workspace)
    )

    assert bundle["state"] == "invalid"
    assert bundle["committed"] is False
    assert bundle["reason"] == "receipt_fingerprint_invalid"


def test_admin_finalization_receipt_raw_route_is_not_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _receipt_path, workspace = _write_finalization_fixture(tmp_path)
    _set_admin_finalization_environment(
        monkeypatch, results, workspace_root=workspace
    )

    with TestClient(app) as client:
        raw = client.get(
            "/api/validation", params={"artifact": "finalization_receipt"}
        )
        page = client.get("/")

    assert raw.status_code == 404
    assert 'id="finalizationBundleCard"' in page.text
    assert "renderFinalizationBundle(payload.finalization_bundle)" in page.text


def test_admin_api_marks_old_campaign_and_objective_pins_stale_but_keeps_endurance_live(
    live_report: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "validation-results"
    report = copy.deepcopy(live_report)
    _bind_admin_report_to_live_lineage(results, report)
    write_report(report, results / "objective-completion/current")

    new_config = "c" * 64
    new_static = "d" * 64
    write_admin_endurance_lineage(
        results,
        config_fingerprint=new_config,
        static_input_fingerprint=new_static,
    )
    _set_admin_finalization_environment(monkeypatch, results)
    monkeypatch.setenv(
        "DEEPSAFE_ENDURANCE_STATUS",
        str(results / "endurance/current/status.json"),
    )
    monkeypatch.setattr(
        admin_validation,
        "_finalization_bundle",
        lambda _reader: {
            "state": "complete",
            "committed": True,
            "reason": None,
        },
    )

    with TestClient(app) as client:
        validation_payload = client.get("/api/validation").json()
        endurance_response = client.get("/api/endurance")

    for campaign_id in ("campaign_report", "objective_completion"):
        projected = validation_payload["campaigns"][campaign_id]
        assert projected["state"] == "stale_lineage"
        assert projected["reason"] == "stale_lineage"
        assert projected["available"] is False
        assert projected["artifact_state"] == "stale_lineage"
        assert projected["finalized"] is False
        assert all(
            evidence["available"] is False
            and evidence["artifact_state"] == "stale_lineage"
            for evidence in projected["evidence"]
        )
    campaign = validation_payload["campaigns"]["campaign_report"]
    assert campaign["accepted"] is False
    assert campaign["final_claim_allowed"] is False
    objective = validation_payload["campaigns"]["objective_completion"]
    assert objective["evidence_complete"] is False
    assert objective["required_gates"] == []
    assert endurance_response.status_code == 200
    assert endurance_response.json()["config_fingerprint"] == new_config
    assert endurance_response.json()["static_input_fingerprint"] == new_static


@pytest.mark.parametrize(
    "lineage_overrides",
    [
        {"status_config_fingerprint": "c" * 64},
        {"status_static_input_fingerprint": "d" * 64},
    ],
    ids=("config_fingerprint", "static_input_fingerprint"),
)
def test_admin_rejects_exact_pins_when_live_endurance_identity_is_incoherent(
    live_report: dict,
    tmp_path: Path,
    lineage_overrides: dict[str, str],
) -> None:
    results = tmp_path / "validation-results"
    report = copy.deepcopy(live_report)
    _bind_admin_report_to_live_lineage(results, report)
    lineage = write_admin_endurance_lineage(results, **lineage_overrides)

    campaign_path = results / "campaign-report/report.json"
    campaign_report = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_report["evidence"] = list(lineage.values())
    write_admin_json(campaign_path, campaign_report)
    replacements = {
        "campaign_report": artifact_pin(
            results,
            "campaign-report/report.json",
            "campaign_report",
        ),
        "endurance_checkpoint": {
            key: value
            for key, value in lineage["endurance_checkpoint"].items()
            if key in {"id", "path", "size_bytes", "sha256"}
        },
        "endurance_status": {
            key: value
            for key, value in lineage["endurance_status"].items()
            if key in {"id", "path", "size_bytes", "sha256"}
        },
    }
    report["evidence"] = [
        replacements.get(row.get("id"), row) for row in report["evidence"]
    ]
    _resign_admin_report(report)
    write_report(report, results / "objective-completion/current")

    reader = _admin_reader(results)
    assert admin_validation._live_endurance_lineage(reader) is None
    for evidence_id, row in lineage.items():
        artifact = reader.read(evidence_id)
        assert admin_validation._artifact_evidence_pin_matches(
            artifact, row, campaign_evidence=True
        )
    campaign = admin_validation._campaign_report(reader)
    objective = admin_validation._objective_completion(reader)
    assert campaign["state"] == "stale_lineage"
    assert campaign["reason"] == "stale_lineage"
    assert campaign["available"] is False
    assert campaign["artifact_state"] == "stale_lineage"
    assert objective["state"] == "stale_lineage"
    assert objective["reason"] == "stale_lineage"
    assert objective["available"] is False
    assert objective["artifact_state"] == "stale_lineage"
