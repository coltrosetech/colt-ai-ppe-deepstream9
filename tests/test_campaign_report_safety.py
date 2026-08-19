"""Fail-closed regressions for campaign-report safety projections.

These tests intentionally exercise hostile but valid-JSON evidence.  They keep
the fixtures in memory so a malformed optional campaign cannot be masked by an
unrelated filesystem artifact.
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from validation import report_campaign as campaign
from validation import gpu_reentry_evidence as reentry


ROOT = Path(__file__).resolve().parents[1]
LOAF_PLAN = (
    ROOT / "validation/results/loaf/val-20-25m/deepstream/dry-run-plan.json"
)
LOAF_AGGREGATE = (
    ROOT / "validation/results/loaf/val-20-25m/deepstream/batch-aggregate.json"
)
BIN_PREPARATION = (
    ROOT
    / "validation/results/loaf/val-20-25m/distance-bins/preparation-manifest.json"
)
BIN_PLAN = (
    ROOT / "validation/results/loaf/val-20-25m/distance-bins/evaluation-plan.json"
)


def _copy_project_file(project: Path, relative: str) -> None:
    destination = project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class _MemoryStore:
    """Small EvidenceStore-compatible reader for section-level regressions."""

    def __init__(self, values: dict[str, dict[str, Any] | None]):
        self.project_root = ROOT
        self.values = values
        self.entries: dict[str, dict[str, Any]] = {}

    def read_json(
        self,
        evidence_id: str,
        path: str | Path,
        *,
        schema_prefix: str | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any] | None:
        value = copy.deepcopy(self.values.get(evidence_id))
        if value is None:
            self.entries[evidence_id] = {"id": evidence_id, "state": "missing"}
            return None
        schema = value.get("schema_version")
        if schema_prefix is not None and (
            not isinstance(schema, str) or not schema.startswith(schema_prefix)
        ):
            self.entries[evidence_id] = {
                "id": evidence_id,
                "state": "invalid_schema",
            }
            return None
        self.entries[evidence_id] = {"id": evidence_id, "state": "ok"}
        if max_bytes is not None:
            self.entries[evidence_id]["max_bytes"] = max_bytes
        return value

    def read_text(self, evidence_id: str, path: str | Path) -> str | None:
        value = self.values.get(evidence_id)
        state = "ok" if isinstance(value, dict) else "missing"
        self.entries[evidence_id] = {"id": evidence_id, "state": state}
        return "evidence" if state == "ok" else None


def _loaf_store(
    plan: dict[str, Any], aggregate: dict[str, Any]
) -> _MemoryStore:
    return _MemoryStore(
        {
            "loaf_deepstream_batch_plan": plan,
            "loaf_deepstream_batch_aggregate": aggregate,
        }
    )


def _fake_complete_loaf_aggregate(plan: dict[str, Any]) -> dict[str, Any]:
    """Counters without result/profile/guard provenance must never be complete."""

    return {
        "schema_version": "deepsafe.loaf-deepstream-batch-aggregate/v1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "completeness": {
            "expected_jobs": 16,
            "complete_jobs": 16,
            "pending_jobs": 0,
            "is_complete": True,
        },
        "aggregation_status": "complete",
    }


def test_evidence_store_keeps_the_default_limit_for_ordinary_artifacts(tmp_path):
    oversized = tmp_path / "ordinary.json"
    with oversized.open("wb") as handle:
        handle.truncate(campaign.MAX_EVIDENCE_BYTES + 1)

    store = campaign.EvidenceStore(tmp_path)

    assert store.read_json("ordinary", oversized) is None
    assert store.entries["ordinary"]["state"] == "too_large"


def test_evidence_store_override_is_per_read_and_hard_capped(tmp_path):
    payload = {
        "schema_version": "fixture/evidence/v1",
        "padding": "x" * 256,
    }
    readable = tmp_path / "readable.json"
    readable.write_text(json.dumps(payload), encoding="utf-8")
    store = campaign.EvidenceStore(tmp_path, max_bytes=64)

    assert store.read_json(
        "readable",
        readable,
        max_bytes=1024,
    ) == payload

    beyond_hard_cap = tmp_path / "beyond-hard-cap.json"
    with beyond_hard_cap.open("wb") as handle:
        handle.truncate(campaign.MAX_EVIDENCE_OVERRIDE_BYTES + 1)
    capped_store = campaign.EvidenceStore(tmp_path)

    assert capped_store.read_bytes(
        "beyond_hard_cap",
        beyond_hard_cap,
        max_bytes=campaign.MAX_EVIDENCE_OVERRIDE_BYTES + 1,
    ) is None
    assert capped_store.entries["beyond_hard_cap"]["state"] == "too_large"


def test_pinned_file_identity_accepts_path_spelling_but_not_another_file(tmp_path):
    prediction = tmp_path / "profiles/640/predictions.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"prediction\n")
    digest = campaign._sha256(prediction.read_bytes())
    relative_pin = {
        "path": "profiles/640/predictions.jsonl",
        "bytes": prediction.stat().st_size,
        "sha256": digest,
    }
    enriched_absolute_pin = {
        **relative_pin,
        "path": str(prediction),
        "model_id": "fixture-model",
    }

    assert campaign._same_pinned_project_file(
        tmp_path,
        relative_pin,
        enriched_absolute_pin,
    )

    other = tmp_path / "profiles/960/predictions.jsonl"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(prediction.read_bytes())
    different_path_pin = {**enriched_absolute_pin, "path": str(other)}

    assert not campaign._same_pinned_project_file(
        tmp_path,
        relative_pin,
        different_path_pin,
    )


def test_loaf_plan_alone_requests_the_bounded_large_artifact_limit():
    plan = _json(LOAF_PLAN)
    aggregate = _json(LOAF_AGGREGATE)
    store = _loaf_store(plan, aggregate)

    campaign._build_loaf_execution_section(
        store,
        ROOT / "validation/results",
    )

    assert store.entries["loaf_deepstream_batch_plan"]["max_bytes"] == (
        campaign.MAX_LOAF_PLAN_EVIDENCE_BYTES
    )
    assert "max_bytes" not in store.entries["loaf_deepstream_batch_aggregate"]


def _verification(
    *,
    status: str,
    gates: list[dict[str, Any]],
    failed_gate_ids: list[str],
    all_present: bool,
    collected_at: str = "2026-07-16T05:00:00+00:00",
    authorized: bool = False,
    operating_policy_id: str = reentry.WORKSTATION_MANAGED_POLICY_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    operating_policy = reentry.operating_policy_contract(operating_policy_id)
    verification = {
        "schema_version": "deepsafe.gpu-reentry-verification/v1",
        "status": status,
        "verified_at_utc": collected_at,
        "operating_policy": copy.deepcopy(operating_policy),
        "all_required_evidence_present": all_present,
        "sustained_load_authorized": authorized,
        "failed_gate_ids": failed_gate_ids,
        "gates": gates,
    }
    evidence = {
        "schema_version": "deepsafe.gpu-reentry-evidence/v1",
        "status": status,
        "collected_at_utc": collected_at,
        "operating_policy": copy.deepcopy(operating_policy),
        "collection_policy": {"operating_policy_id": operating_policy_id},
        "verification": copy.deepcopy(verification),
    }
    return evidence, verification


def _reentry_gates(
    *,
    failed_id: str | None = None,
    operating_policy_id: str = reentry.WORKSTATION_MANAGED_POLICY_ID,
) -> list[dict[str, Any]]:
    policy = reentry.operating_policy_contract(operating_policy_id)
    required_ids = set(policy["required_gate_ids"])
    return [
        {
            "id": gate_id,
            "required": gate_id in required_ids,
            "passed": gate_id != failed_id,
        }
        for gate_id in campaign.EXPECTED_GPU_REENTRY_GATE_IDS
    ]


def test_malformed_loaf_job_is_reported_invalid_instead_of_crashing():
    plan = _json(LOAF_PLAN)
    aggregate = _json(LOAF_AGGREGATE)
    plan["jobs"][0]["model_input"] = [640]

    section, _ = campaign._build_loaf_execution_section(
        _loaf_store(plan, aggregate), ROOT / "validation/results"
    )

    assert section["plan_contract_valid"] is False
    assert section["state"] == "incomplete_or_invalid"
    assert section["metrics_withheld"] is True


def test_loaf_complete_counters_without_results_profiles_or_guards_are_rejected():
    plan = _json(LOAF_PLAN)
    plan["status"] = "execution-finished"
    aggregate = _fake_complete_loaf_aggregate(plan)

    section, _ = campaign._build_loaf_execution_section(
        _loaf_store(plan, aggregate), ROOT / "validation/results"
    )

    assert section["plan_contract_valid"] is True
    assert section["aggregate_contract_valid"] is False
    assert section["state"] != "complete"
    assert section["metrics_withheld"] is True


def test_execution_finished_is_a_recognized_loaf_terminal_plan_state():
    plan = _json(LOAF_PLAN)
    plan["status"] = "execution-finished"
    aggregate = _json(LOAF_AGGREGATE)
    aggregate["completeness"].update(
        complete_jobs=0,
        pending_jobs=campaign.EXPECTED_LOAF_JOBS,
        is_complete=False,
    )
    aggregate["aggregation_status"] = "withheld_incomplete"

    section, _ = campaign._build_loaf_execution_section(
        _loaf_store(plan, aggregate), ROOT / "validation/results"
    )

    # The runner emits execution-finished after a failure-free execute_plan.
    # Aggregate/output validation, not an incompatible status spelling, must
    # decide whether the campaign is actually complete.
    assert section["plan_contract_valid"] is True
    assert section["state"] == "prepared_waiting_for_gpu"
    assert section["complete_jobs"] == 0
    assert section["metrics_withheld"] is True


def test_empty_distance_bin_rows_cannot_satisfy_complete_matrix():
    preparation = _json(BIN_PREPARATION)
    plan = _json(BIN_PLAN)
    aggregate = {
        "schema_version": "deepsafe.loaf-distance-bin-aggregate/v1",
        "status": "complete",
        "split": "val",
        "test_unseen_opened": False,
        "metric": {
            "name": "AP101@IoU0.5",
            "explicitly_not": "not COCO mAP@[.50:.95]",
        },
        "completeness": {
            "expected_profiles": [640, 960],
            "expected_distance_bins": [
                "20-21m",
                "21-22m",
                "22-23m",
                "23-24m",
                "24-25m",
            ],
            "expected_evaluations": 10,
            "complete_evaluations": 10,
            "is_complete": True,
        },
        "rows": [{} for _ in range(10)],
    }
    store = _MemoryStore(
        {
            "loaf_distance_bins_preparation": preparation,
            "loaf_distance_bins_evaluation_plan": plan,
            "loaf_distance_bins_aggregate": aggregate,
        }
    )

    section, _ = campaign._build_loaf_distance_bins_section(
        store, ROOT / "validation/results"
    )

    assert section["preparation_contract_valid"] is True
    assert section["evaluation_plan_contract_valid"] is True
    assert section["aggregate_contract_valid"] is False
    assert section["state"] != "complete"


def test_loaf_rights_review_hash_is_part_of_preparation_guardrail(tmp_path):
    for relative in (
        "data/gt/loaf/PROVENANCE.json",
        "docs/loaf-rights-review.md",
        "validation/results/loaf/val-20-25m/selection-report.json",
        "validation/results/loaf/val-20-25m/media-plan.json",
        "validation/results/loaf/test-unseen-20-25m/selection-report.json",
        "validation/results/loaf/test-unseen-20-25m/media-plan.json",
    ):
        _copy_project_file(tmp_path, relative)

    intact_store = campaign.EvidenceStore(tmp_path)
    intact, _ = campaign._build_loaf_preparation_section(
        intact_store, tmp_path / "validation/results"
    )
    assert intact["state"] == "prepared_not_evaluated"
    assert intact["dataset_rights"]["guardrail_consistent"] is True

    (tmp_path / "docs/loaf-rights-review.md").write_text(
        "tampered rights claim\n", encoding="utf-8"
    )
    tampered_store = campaign.EvidenceStore(tmp_path)
    tampered, _ = campaign._build_loaf_preparation_section(
        tampered_store, tmp_path / "validation/results"
    )
    assert tampered["state"] == "incomplete_or_invalid"
    assert tampered["dataset_rights"]["guardrail_consistent"] is False


def test_failed_required_reentry_gate_cannot_be_ready_with_empty_failed_ids():
    failed_id = reentry.WORKSTATION_REQUIRED_GATE_IDS[0]
    evidence, verification = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(failed_id=failed_id),
        failed_gate_ids=[],
        all_present=True,
        authorized=False,
    )
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, _ = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["passed_required_gate_count"] == (
        len(reentry.WORKSTATION_REQUIRED_GATE_IDS) - 1
    )
    assert section["required_gate_count"] == len(
        reentry.WORKSTATION_REQUIRED_GATE_IDS
    )
    assert section["verification_consistent"] is False
    assert section["ready_for_operator_review"] is False
    assert section["state"] == "invalid"
    assert section["sustained_load_authorized"] is False


def test_failed_informational_reentry_gate_remains_valid_and_nonblocking():
    informational_id = "bios_thermal_management"
    evidence, verification = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(failed_id=informational_id),
        failed_gate_ids=[],
        all_present=True,
        collected_at=datetime.now(timezone.utc).isoformat(),
    )
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, hardware_blockers = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["verification_consistent"] is True
    assert section["state"] == "ready_for_operator_review"
    assert section["ready_for_operator_review"] is True
    assert section["failed_gate_ids"] == []
    assert section["required_gate_count"] == len(
        reentry.WORKSTATION_REQUIRED_GATE_IDS
    )
    assert section["passed_required_gate_count"] == len(
        reentry.WORKSTATION_REQUIRED_GATE_IDS
    )
    assert hardware_blockers == []


def test_missing_reentry_operating_policy_is_rejected():
    evidence, verification = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(),
        failed_gate_ids=[],
        all_present=True,
    )
    evidence.pop("operating_policy")
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, _ = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["verification_consistent"] is False
    assert section["state"] == "invalid"
    assert section["ready_for_operator_review"] is False


def test_coherently_forged_reentry_operating_policy_is_rejected():
    evidence, verification = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(),
        failed_gate_ids=[],
        all_present=True,
    )
    forged = copy.deepcopy(evidence["operating_policy"])
    moved_gate = forged["informational_gate_ids"].pop(0)
    forged["required_gate_ids"].append(moved_gate)
    evidence["operating_policy"] = copy.deepcopy(forged)
    evidence["verification"]["operating_policy"] = copy.deepcopy(forged)
    verification["operating_policy"] = copy.deepcopy(forged)
    for bundle in (evidence["verification"], verification):
        next(gate for gate in bundle["gates"] if gate["id"] == moved_gate)[
            "required"
        ] = True
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, _ = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["verification_consistent"] is False
    assert section["state"] == "invalid"
    assert section["ready_for_operator_review"] is False


def test_stale_reentry_bundle_cannot_be_ready_for_operator_review():
    evidence, verification = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(),
        failed_gate_ids=[],
        all_present=True,
        collected_at="2000-01-01T00:00:00+00:00",
    )
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, _ = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["verification_consistent"] is True
    assert section["verification_fresh"] is False
    assert section["ready_for_operator_review"] is False
    assert section["state"] == "stale"


def test_reporter_uses_the_same_reentry_age_window_as_the_collector():
    assert campaign.GPU_REENTRY_MAX_AGE_SECONDS == 4 * 60 * 60


def test_deterministic_report_time_ignores_future_expiry_boundaries():
    timestamps = list(
        campaign._timestamps(
            {
                "completed_at_utc": "2026-07-16T16:30:21+00:00",
                "expires_at_utc": "2026-07-17T13:48:44+00:00",
                "valid_until_utc": "2026-07-18T00:00:00+00:00",
            }
        )
    )

    assert [value.isoformat() for value in timestamps] == [
        "2026-07-16T16:30:21+00:00"
    ]


def test_invalid_reentry_bundle_is_not_misclassified_as_verified_hardware():
    evidence, embedded = _verification(
        status="ready_for_operator_review",
        gates=_reentry_gates(),
        failed_gate_ids=[],
        all_present=True,
    )
    verification = copy.deepcopy(embedded)
    verification["status"] = "blocked"
    store = _MemoryStore(
        {
            "gpu_reentry_evidence": evidence,
            "gpu_reentry_verification": verification,
            "gpu_reentry_markdown": {"present": True},
        }
    )

    section, hardware_blockers = campaign._build_gpu_reentry_section(
        store, ROOT / "validation/results"
    )

    assert section["state"] == "invalid"
    assert hardware_blockers == []


def _patch_required_campaigns_as_accepted(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gpu_section: dict[str, Any],
    gpu_blockers: list[dict[str, Any]],
) -> None:
    # This unit isolates the top-level hardware veto. The full campaign-shape
    # validator has its own tests and would correctly reject these deliberately
    # compact section stubs before the veto assertion can be reached.
    monkeypatch.setattr(campaign, "validate_report_shape", lambda report: None)
    scene_facts = {
        "accepted": True,
        "planned_types": 12,
        "executed_types": 12,
        "focus_types": ["medium_close", "overhead"],
        "acceptance_safe_types": {f"type-{index}" for index in range(12)},
        "acceptance_safe_focus_types": ["medium_close", "overhead"],
        "safe_count": 24,
        "operational_count": 24,
        "pairs": [{} for _ in range(12)],
    }
    monkeypatch.setattr(
        campaign,
        "_build_scene_section",
        lambda store, root: ({"evidence_ids": []}, [], scene_facts),
    )
    monkeypatch.setattr(
        campaign,
        "_build_caviar_section",
        lambda store, root: (
            {"evidence_ids": [], "ground_truth": True},
            {
                "manifest_contract": True,
                "aggregate_contract": True,
                "accepted": True,
                "validated_count": 16,
                "pairs": [{} for _ in range(8)],
            },
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_build_open_review_section",
        lambda store, root: (
            {
                "evidence_ids": [],
                "ground_truth": False,
                "metric_guardrail": {
                    "precision_recall_ap_forbidden": True,
                    "ai_estimates_are_not_ground_truth": True,
                    "human_qa_is_separate": True,
                },
                "ai_qualitative_visual_audit": {
                    "profile_decision_count": 42,
                },
            },
            {
                "plan_contract": True,
                "review_contract": True,
                "ai_audit_accepted": True,
                "accepted": True,
                "validated_count": 24,
                "pairs": [{} for _ in range(12)],
                "distinct_reviewed_video_types": 12,
                "human_terminal_count": 42,
            },
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_build_endurance_section",
        lambda store, root: (
            {"evidence_ids": []},
            [],
            {"accepted": True, "healthy_segments": 28, "valid_days": 7},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_build_loaf_preparation_section",
        lambda store, root: ({"state": "prepared", "evidence_ids": []}, {}),
    )
    monkeypatch.setattr(
        campaign,
        "_build_loaf_execution_section",
        lambda store, root: (
            {"state": "incomplete_or_invalid", "evidence_ids": []},
            {},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_build_loaf_distance_bins_section",
        lambda store, root: (
            {"state": "incomplete_or_invalid", "evidence_ids": []},
            {},
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_build_gpu_reentry_section",
        lambda store, root: (gpu_section, gpu_blockers),
    )
    monkeypatch.setattr(
        campaign,
        "_build_distance_section",
        lambda store, root: (
            {
                "evidence_kind": "calibrated_distance_ground_truth",
                "target_maximum_distance_m": 25,
                "state": "proven",
                "accepted": True,
                "reasons": [],
                "schema_contract_valid": True,
                "pin_matrix_valid": True,
                "independent_cpu_recomputation_valid": True,
                "criterion": "fixture criterion",
                "profiles": {"640": {}, "960": {}},
                "contract": {
                    "required_schema": "deepsafe.distance-validation/v1",
                    "required_bin_m": [20, 25],
                    "required_profiles": [640, 960],
                    "requires_verified_calibration": True,
                    "requires_documented_passing_acceptance_criterion": True,
                },
                "evidence_ids": [],
            },
            {"accepted": True, "reasons": []},
        ),
    )


def test_final_acceptance_cannot_coexist_with_current_hardware_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blocker = {
        "code": "gpu_reentry_evidence_blocked",
        "detail": "Required physical GPU prerequisite is blocked.",
        "source": "gpu-reentry/current",
        "evidence_ids": [],
    }
    _patch_required_campaigns_as_accepted(
        monkeypatch,
        gpu_section={
            "state": "blocked",
            "evidence_ids": [],
            "ready_for_operator_review": False,
            "sustained_load_authorized": False,
        },
        gpu_blockers=[blocker],
    )

    report = campaign.build_campaign_report(
        project_root=tmp_path,
        results_root=tmp_path / "validation/results",
        hardware_incident_path=None,
    )

    assert report["hardware_blockers"] == [blocker]
    assert report["decision"]["accepted"] is False
    assert report["decision"]["final_claim_allowed"] is False
    assert report["decision"]["status"] == "blocked_by_hardware"
