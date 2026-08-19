from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from validation import gpu_reentry_refresh_v2 as refresh


def _blocked_training_fixture() -> dict:
    return {
        "gpu_work_blocked_until_ready": True,
        "ready": False,
        "lineage_status": "known_dependency_failures_recorded_successor_pending",
        "highest_failed_revision": 4,
        "required_successor_min_revision": 5,
        "required_successor_revision": "r5_or_later_dependency_closed",
        "known_failed_build_receipts": [
            refresh.file_pin(refresh.R2_FAILED_BUILD_RECEIPT),
            refresh.file_pin(refresh.R3_FAILED_BUILD_RECEIPT),
            refresh.file_pin(refresh.R4_FAILED_BUILD_RECEIPT),
        ],
        "current_execution_plan": refresh.file_pin(refresh.TRAINING_PLAN),
        "current_candidate_attempt_id": "effective-r5-001",
        "current_candidate_build_receipt_path": (
            "validation/results/person/training/rtdetrv4-s-r-livit-person-r1-gpu-v1/"
            "image-build-attempts/effective-r5-001/build-receipt.json"
        ),
        "current_candidate_build_receipt": None,
        "current_candidate_status": "missing",
        "image_reference": "deepsafe-rtdetrv4-person:r5",
        "resolved_image_id": None,
        "base_image_id": refresh.IMAGE_ID,
    }


@pytest.fixture
def blocked_plan(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        refresh,
        "inspect_gpu_identity",
        lambda: {
            "planning_observation_only": True,
            "observed_at_utc": "2026-07-17T23:30:00Z",
            "read_only_command": refresh.GPU_QUERY,
            "index": 0,
            "uuid": refresh.GPU_UUID,
            "name": refresh.GPU_NAME,
            "pci_bus_id": "00000000:01:00.0",
            "driver_version": refresh.GPU_DRIVER,
            "memory_total_mib": refresh.GPU_MEMORY_MIB,
        },
    )
    monkeypatch.setattr(
        refresh,
        "inspect_official_image",
        lambda: {
            "reference": refresh.IMAGE_REFERENCE,
            "resolved_image_id": refresh.IMAGE_ID,
            "repo_digest": refresh.IMAGE_REPO_DIGEST,
            "size_bytes": refresh.IMAGE_SIZE,
            "architecture": "amd64",
            "operating_system": "linux",
        },
    )
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    return refresh.build_plan()


def _publish_test_plan(plan: dict, name: str) -> Path:
    root = refresh.ROOT / "validation/results/gpu-reentry/test-r2-refresh"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    published = copy.deepcopy(plan)
    relative = path.relative_to(refresh.ROOT).as_posix()
    published["execution_contract"]["plan_path"] = relative
    template = published["execution_contract"]["root_command_template"]
    template[template.index("--plan") + 1] = relative
    published["fingerprint_sha256"] = refresh.document_fingerprint(published)
    path.write_bytes(refresh.canonical_bytes(published) + b"\n")
    return path


def test_plan_records_literal_owner_context_without_inventing_physical_proof(
    blocked_plan: dict,
) -> None:
    attestations = blocked_plan["owner_attestations"]
    assert [item["literal_quote"] for item in attestations] == [
        "Tamamdır canlandırdım ekran kartını!",
        (
            "Tamam dostum onay veriyorum! Bu işlemlerine devam edebilirsin! "
            "Goal hedefine başla/devam et"
        ),
    ]
    assert all(item["source_calendar_date"] is None for item in attestations)
    assert all(item["source_timestamp_available"] is False for item in attestations)
    assert all(item["physical_verification_performed"] is False for item in attestations)
    assert all(item["cryptographic_identity_authenticated"] is False for item in attestations)


def test_plan_preserves_legacy_evidence_and_exposes_exact_supervisor_drift(
    blocked_plan: dict,
) -> None:
    legacy = blocked_plan["legacy_current_evidence"]
    assert legacy["preserved_no_overwrite"] is True
    assert legacy["evidence"]["sha256"] == (
        "d726e8fc22413ec2190dbcdd0d9b9c6c4abe91e033f63b9946d976a7f9f65a3e"
    )
    assert legacy["replayed_status"] == "blocked"
    assert set(legacy["failed_gate_ids"]) >= {
        "fresh_idle_gpu_telemetry",
        "current_guard_code_hashes",
    }
    drift = legacy["supervisor_pin_drift"]
    assert drift["detected"] is True
    assert drift["recorded_pin"]["sha256"] != drift["current_pin"]["sha256"]
    assert drift["current_pin"] == refresh.file_pin(refresh.SUPERVISOR)


def test_historical_lease_pair_is_not_promoted_to_cuda_readiness(blocked_plan: dict) -> None:
    historical = blocked_plan["historical_minimal_cuda_smoke"]
    assert historical["status"] == "lease_lifecycle_only_cuda_result_receipt_missing"
    assert historical["cuda_result_receipt_present"] is False
    assert historical["quality_or_readiness_claim_allowed"] is False
    assert len(historical["lease_lifecycle_receipts"]) == 2
    for pin in historical["lease_lifecycle_receipts"]:
        assert refresh.file_pin(refresh.ROOT / pin["path"]) == pin


def test_failed_live_001_is_frozen_diagnosed_and_not_promoted_to_gpu_success(
    blocked_plan: dict,
) -> None:
    failed = blocked_plan["prior_failed_live_smoke"]
    assert failed["run_id"] == "gpu-reentry-r2-live-001"
    assert failed["status"] == "failed_docker_entrypoint_interception"
    assert failed["preserved_no_overwrite"] is True
    assert failed["plan"]["fingerprint_sha256"] == (
        "1e00e46829f984cebe88d69605d96e9d89802a3acca2e25b45adaf9fa0368e7f"
    )
    assert failed["docker"] == {
        "exit_status": 2,
        "previous_entrypoint_override_present": False,
        "root_cause": (
            "the derived training image ENTRYPOINT container_runner.py intercepted "
            "the intended python -c child command"
        ),
        "stderr_marker": "usage: container_runner.py",
        "corrective_control": "explicit --entrypoint=python before the image ID",
    }
    assert failed["lease"]["lifecycle_complete"] is True
    assert failed["lease"]["command_argv_sha256"] == refresh.lease_command_sha256(
        failed["lease"]["child_command"]
    )
    assert failed["gpu_result_valid"] is False
    assert not any(failed["claims"].values())
    assert failed["v1_reentry_scope"] == {
        "scope": "historical_incident_lineage_only",
        "evidence_collected_at_utc": "2026-07-18T00:16:46.353716+00:00",
        "replayed_at_recorded_verification_utc": (
            "2026-07-18T00:16:47.738510+00:00"
        ),
        "status_at_recorded_verification": "ready_for_operator_review",
        "current_freshness_evaluated": False,
        "eligible_as_current_reentry_readiness": False,
        "eligible_for_live_003_fresh_reentry_gate": False,
    }

    frozen_pins = [failed["plan"]["artifact"], *failed["artifacts"].values()]
    for pin in frozen_pins:
        path = refresh.ROOT / pin["path"]
        assert refresh.file_pin(path) == pin
        assert (path.stat().st_mode & 0o777) == 0o440

    assert failed["lease"]["acquire_receipt"] == refresh.file_pin(
        refresh.PRIOR_FAILED_ACQUIRE
    )
    assert failed["lease"]["release_receipt"] == refresh.file_pin(
        refresh.PRIOR_FAILED_RELEASE
    )


def test_live_005_uses_a_new_default_plan_and_preserves_prior_lineage() -> None:
    args = refresh.build_parser().parse_args(["create-plan"])
    assert args.output == refresh.DEFAULT_PLAN
    assert refresh.DEFAULT_PLAN != refresh.PRIOR_LIVE_PLAN
    assert refresh.DEFAULT_PLAN.name == "plan-live-005.json"
    assert refresh.RECOMMENDED_NEXT_RUN_ID == "gpu-reentry-r2-live-005"
    assert refresh._prior_successful_live_003_pin() == {
        "path": (
            "validation/results/gpu-reentry/r2-executions/"
            "gpu-reentry-r2-live-003/evidence.json"
        ),
        "bytes": 6_734,
        "sha256": (
            "599936d62a7fdcc08add01363a697f64867abc8c09a74636246428534d7c44cf"
        ),
    }
    assert refresh.file_pin(refresh.PRIOR_LIVE_PLAN) == {
        "path": "validation/results/gpu-reentry/r2-refresh-20260718/plan.json",
        "bytes": 12_317,
        "sha256": "4373703ee594b0e986a9a2464dcf3f0969ab859cde3fa2c7952a97b8fb9974e5",
    }
    assert (refresh.PRIOR_LIVE_PLAN.stat().st_mode & 0o777) == 0o440


def test_failed_live_004_is_exact_diagnosed_and_never_promoted(
    blocked_plan: dict,
) -> None:
    failed = blocked_plan["prior_failed_live_004"]
    assert failed["run_id"] == "gpu-reentry-r2-live-004"
    assert failed["status"] == "failed_fresh_v1_summary_contract_mismatch"
    assert failed["failure"]["marker"] == (
        "fresh v1 evidence changed during semantic replay"
    )
    assert failed["failure"]["stage"] == "fresh_v1_summary_binding_before_docker"
    assert failed["v1_reentry_status"] == "ready_for_operator_review"
    assert failed["v1_reentry_scope"]["summary_contract_satisfied"] is False
    assert failed["gpu_result_valid"] is False
    assert not any(failed["claims"].values())
    assert not any(
        value
        for key, value in failed["execution_observation"].items()
        if key != "host_child_exit_status"
    )
    assert failed["lease"]["renew_receipts"] == []
    assert failed["lease"]["command_argv_sha256"] == (
        refresh.lease_command_sha256(failed["lease"]["child_command"])
    )
    for pin in [failed["plan"]["artifact"], *failed["artifacts"].values()]:
        path = refresh.ROOT / pin["path"]
        assert refresh.file_pin(path, required_mode=0o440) == pin
    with pytest.raises(refresh.RefreshError, match="incident lineage only"):
        refresh.validate_plan(refresh.PRIOR_FAILED_LIVE_004_PLAN)


def test_fresh_v1_summary_contract_binds_projection_not_full_evidence() -> None:
    pin = refresh.PRIOR_FAILED_LIVE_004_EXACT_PINS["v1_evidence"]
    path, evidence = refresh.load_pinned_json(pin, required_mode=0o440)
    replay, _collected, _verified = refresh._replay_recorded_v1_evidence(
        evidence, context="summary contract test"
    )
    summary = {
        "report_path": str(path),
        "report_sha256": pin["sha256"],
        "collected_at_utc": evidence["collected_at_utc"],
        "gpu_identity": evidence["gpu_identity"],
        "operating_policy": evidence["operating_policy"],
        "verification": replay,
        "load_authority_granted_by_gate": False,
        "execution_authority": {
            "source": "explicit_user_instruction",
            "granted_by_this_evidence": False,
            "cryptographic_identity_authentication": False,
        },
    }
    assert summary != evidence
    assert refresh._validate_required_v1_summary(
        summary, v1_path=path, v1_pin=pin, v1_evidence=evidence
    ) == replay

    forged = {**summary, "report_sha256": "0" * 64}
    with pytest.raises(refresh.RefreshError, match="summary projection differs"):
        refresh._validate_required_v1_summary(
            forged, v1_path=path, v1_pin=pin, v1_evidence=evidence
        )

    forged_verification = copy.deepcopy(summary)
    forged_verification["verification"]["authorization_policy"] = "forged"
    with pytest.raises(refresh.RefreshError, match="summary projection differs"):
        refresh._validate_required_v1_summary(
            forged_verification,
            v1_path=path,
            v1_pin=pin,
            v1_evidence=evidence,
        )


def test_held_v1_summary_rejects_name_swap_after_semantic_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pin = refresh.PRIOR_FAILED_LIVE_004_EXACT_PINS["v1_evidence"]
    _, evidence = refresh.load_pinned_json(source_pin, required_mode=0o440)
    root = refresh.ROOT / "validation/results/gpu-reentry/test-r2-refresh"
    root.mkdir(parents=True, exist_ok=True)
    suffix = tmp_path.name
    path = root / f"held-summary-{suffix}.json"
    replacement_path = root / f"held-summary-{suffix}.replacement.json"
    backup_path = root / f"held-summary-{suffix}.original.json"
    pin = refresh.write_new_json(path, evidence)
    replacement = copy.deepcopy(evidence)
    replacement["status"] = "forged-path-swap"
    refresh.write_new_json(replacement_path, replacement)
    real_replay = refresh.replay_v1_evidence
    swapped = False

    def replay_then_swap(value, *, now):
        nonlocal swapped
        result = real_replay(value, now=now)
        if not swapped:
            path.rename(backup_path)
            replacement_path.rename(path)
            swapped = True
        return result

    monkeypatch.setattr(refresh, "replay_v1_evidence", replay_then_swap)
    recorded_now = datetime.fromisoformat(
        evidence["verification"]["verified_at_utc"]
    )
    try:
        with pytest.raises(
            refresh.RefreshError,
            match="publication changed after held semantic verification",
        ):
            refresh._require_held_v1_summary(
                v1_path=path,
                v1_pin=pin,
                v1_evidence=evidence,
                now=recorded_now,
            )
        assert swapped is True
        assert refresh.file_pin(path, required_mode=0o440) != pin
    finally:
        for candidate in (path, replacement_path, backup_path):
            if candidate.exists():
                candidate.unlink()


def test_aged_live_001_is_historical_lineage_not_current_readiness() -> None:
    evidence = refresh.load_json(refresh.PRIOR_FAILED_V1_EVIDENCE)
    future = datetime(2026, 7, 19, tzinfo=timezone.utc)
    current_replay = refresh.replay_v1_evidence(evidence, now=future)
    assert current_replay["status"] == "blocked"
    assert "fresh_idle_gpu_telemetry" in current_replay["failed_gate_ids"]

    historical = refresh._prior_failed_live_smoke()["v1_reentry_scope"]
    assert historical["status_at_recorded_verification"] == (
        "ready_for_operator_review"
    )
    assert historical["current_freshness_evaluated"] is False
    assert historical["eligible_as_current_reentry_readiness"] is False
    assert historical["eligible_for_live_003_fresh_reentry_gate"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda evidence: evidence.__setitem__(
                "collected_at_utc", "2026-07-18T00:17:46.353716+00:00"
            ),
            "collection timestamp differs",
        ),
        (
            lambda evidence: evidence["verification"].__setitem__(
                "verified_at_utc", "2026-07-18T00:17:47.738510+00:00"
            ),
            "verification timestamp differs",
        ),
        (
            lambda evidence: evidence["verification"]["gates"][0].__setitem__(
                "passed", False
            ),
            "historical semantic replay differs",
        ),
    ],
)
def test_failed_live_001_historical_replay_rejects_loader_tamper(
    monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    original_load_pinned_json = refresh.load_pinned_json

    def tampered_load(expected_pin: dict, **kwargs) -> tuple[Path, dict]:
        path, value = original_load_pinned_json(expected_pin, **kwargs)
        if path.resolve() == refresh.PRIOR_FAILED_V1_EVIDENCE.resolve():
            value = copy.deepcopy(value)
            mutate(value)
        return path, value

    monkeypatch.setattr(refresh, "load_pinned_json", tampered_load)
    with pytest.raises(refresh.RefreshError, match=message):
        refresh._prior_failed_live_smoke()


def test_failed_live_001_historical_pin_rejects_content_identity_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_compiled_pin = refresh._compiled_prior_failed_pin

    def tampered_pin(path: Path, *, pin_name: str) -> dict:
        pin = original_compiled_pin(path, pin_name=pin_name)
        if Path(path).resolve() == refresh.PRIOR_FAILED_V1_EVIDENCE.resolve():
            pin = {**pin, "sha256": "0" * 64}
        return pin

    monkeypatch.setattr(refresh, "_compiled_prior_failed_pin", tampered_pin)
    with pytest.raises(refresh.RefreshError, match="pinned file exact bytes differ"):
        refresh._prior_failed_live_smoke()


def test_exact_old_live_001_plan_is_incident_only_under_new_supervision_schema() -> None:
    historical = refresh.load_json(refresh.PRIOR_LIVE_PLAN)
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(historical, refresh.PLAN_SCHEMA)
    with pytest.raises(refresh.RefreshError, match="incident lineage only"):
        refresh.validate_plan(refresh.PRIOR_LIVE_PLAN)


def test_alternate_plan_output_path_is_bound_through_root_command(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    path = (
        refresh.ROOT
        / "validation/results/gpu-reentry/test-r2-refresh/alternate-live-004.json"
    )
    path.unlink(missing_ok=True)
    plan = refresh.build_plan(plan_path=path)
    refresh.write_new_json(path, plan)
    try:
        validated = refresh.validate_plan(path)
        expected_path = path.relative_to(refresh.ROOT).as_posix()
        assert validated["execution_contract"]["plan_path"] == expected_path
        assert validated["execution_contract"]["root_command_template"][5] == (
            expected_path
        )

        renderable = copy.deepcopy(validated)
        renderable["training_image_prerequisite"]["ready"] = True
        command = refresh.render_root_command(
            renderable, "gpu-reentry-r2-live-004-test"
        )
        assert command[command.index("--plan") + 1] == expected_path
        assert refresh.DEFAULT_PLAN.relative_to(refresh.ROOT).as_posix() not in command
    finally:
        path.chmod(0o640)
        path.unlink()


def test_rehashed_plan_cannot_redirect_bound_plan_path(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    tampered = copy.deepcopy(blocked_plan)
    tampered["execution_contract"]["plan_path"] = (
        refresh.PRIOR_LIVE_PLAN.relative_to(refresh.ROOT).as_posix()
    )
    tampered["fingerprint_sha256"] = refresh.document_fingerprint(tampered)
    path = (
        refresh.ROOT
        / "validation/results/gpu-reentry/test-r2-refresh/tampered-plan-path.json"
    )
    path.write_bytes(refresh.canonical_bytes(tampered) + b"\n")
    try:
        with pytest.raises(refresh.RefreshError, match="plan path binding differs"):
            refresh.validate_plan(path)
    finally:
        path.unlink()


def test_gpu_and_base_image_are_exact_pinned(blocked_plan: dict) -> None:
    gpu = blocked_plan["gpu_target"]
    assert (gpu["index"], gpu["uuid"], gpu["name"], gpu["driver_version"]) == (
        0,
        refresh.GPU_UUID,
        "NVIDIA RTX A5000 Laptop GPU",
        "590.48.01",
    )
    image = blocked_plan["official_pytorch_image"]
    assert image["resolved_image_id"] == refresh.IMAGE_ID
    assert image["repo_digest"] == refresh.IMAGE_REPO_DIGEST
    assert image["size_bytes"] == 3_037_863_937
    assert image["torch"] == "2.13.0+cu130"
    assert image["cuda"] == "13.0"


def test_r2_r3_failure_lineage_blocks_gpu_until_r4_or_later(blocked_plan: dict) -> None:
    prerequisite = blocked_plan["training_image_prerequisite"]
    assert prerequisite["ready"] is False
    assert prerequisite["required_successor_revision"] == "r5_or_later_dependency_closed"
    assert len(prerequisite["known_failed_build_receipts"]) == 3
    assert blocked_plan["status"] == "blocked_pending_successor_training_image"
    assert blocked_plan["execution_contract"]["gpu_execution_enabled"] is False
    with pytest.raises(refresh.RefreshError, match="required successor"):
        refresh.render_root_command(blocked_plan, "gpu-reentry-r2-test")


def test_validate_plan_replays_pins_and_rejects_quote_tamper(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    path = _publish_test_plan(blocked_plan, "valid.json")
    try:
        validated = refresh.validate_plan(path)
        assert validated["fingerprint_sha256"] == refresh.document_fingerprint(
            validated
        )
    finally:
        path.unlink()

    tampered = copy.deepcopy(blocked_plan)
    tampered["owner_attestations"][0]["literal_quote"] = "GPU tamam"
    tampered["fingerprint_sha256"] = refresh.document_fingerprint(tampered)
    path = _publish_test_plan(tampered, "tampered-quote.json")
    try:
        with pytest.raises(refresh.RefreshError, match="literal owner attestation"):
            refresh.validate_plan(path)
    finally:
        path.unlink()


def test_validate_plan_rejects_rehashed_source_pin_tamper(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    tampered = copy.deepcopy(blocked_plan)
    tampered["source_pins"]["validator"]["sha256"] = "0" * 64
    tampered["fingerprint_sha256"] = refresh.document_fingerprint(tampered)
    path = _publish_test_plan(tampered, "tampered-pin.json")
    try:
        with pytest.raises(refresh.RefreshError, match="source validator pin differs"):
            refresh.validate_plan(path)
    finally:
        path.unlink()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda plan: plan["prior_failed_live_smoke"]["artifacts"][
                "docker_stderr_log"
            ].__setitem__("sha256", "0" * 64),
            "prior_failed_live_docker_stderr pin differs",
        ),
        (
            lambda plan: plan["execution_contract"].__setitem__(
                "docker_entrypoint_override", "python3"
            ),
            "schema validation failed",
        ),
    ],
)
def test_validate_plan_rejects_failed_lineage_or_entrypoint_contract_tamper(
    blocked_plan: dict,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    tampered = copy.deepcopy(blocked_plan)
    mutate(tampered)
    tampered["fingerprint_sha256"] = refresh.document_fingerprint(tampered)
    path = _publish_test_plan(tampered, "tampered-entrypoint-lineage.json")
    try:
        with pytest.raises(refresh.RefreshError, match=message):
            refresh.validate_plan(path)
    finally:
        path.unlink()


def test_execute_rejects_before_creating_run_directory_when_successor_missing(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    path = _publish_test_plan(blocked_plan, "blocked-execute.json")
    output = refresh.EXECUTION_ROOT / "gpu-reentry-r2-blocked-test"
    assert not output.exists()
    try:
        published = refresh.load_json(path)
        with pytest.raises(refresh.RefreshError, match="fail-closed"):
            refresh.execute_refresh(
                plan_path=path,
                plan_fingerprint=published["fingerprint_sha256"],
                run_id=output.name,
            )
        assert not output.exists()
    finally:
        path.unlink()


def test_live_smoke_requires_held_lease_before_any_collection(
    blocked_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "_training_image_prerequisite", _blocked_training_fixture)
    path = _publish_test_plan(blocked_plan, "blocked-live.json")
    try:
        published = refresh.load_json(path)
        with pytest.raises(refresh.RefreshError, match="required successor"):
            refresh.execute_live_smoke(
                plan_path=path,
                plan_fingerprint=published["fingerprint_sha256"],
                run_id="gpu-reentry-r2-no-lease",
                output_dir=refresh.EXECUTION_ROOT / "gpu-reentry-r2-no-lease",
            )
    finally:
        path.unlink()


def test_docker_smoke_is_small_offline_read_only_and_uses_successor_image() -> None:
    successor = "sha256:" + "1" * 64
    run_id = "gpu-reentry-r2-docker-render-test"
    output = refresh.EXECUTION_ROOT / run_id
    control = output / "control"
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(mode=0o700)
    cidfile = control / "container.cid"
    try:
        command = refresh.render_docker_command(successor, cidfile=cidfile)
    finally:
        control.rmdir()
        output.rmdir()
    assert command[:2] == ["docker", "run"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--pull=never" in command
    assert f"--cidfile={cidfile}" in command
    assert "--cap-drop=ALL" in command
    assert f"--gpus=device={refresh.GPU_UUID}" in command
    assert successor in command
    assert refresh.IMAGE_ID not in command
    assert command.count("--entrypoint=python") == 1
    entrypoint_index = command.index("--entrypoint=python")
    image_index = command.index(successor)
    assert entrypoint_index < image_index
    assert command[image_index + 1] == "-c"
    assert "python" not in command[image_index + 1 :]
    code = command[-1]
    assert "torch.arange(4096" in code
    assert "torch.cuda.synchronize()" in code


def test_lease_child_command_binds_plan_run_output_and_venv() -> None:
    run_id = "gpu-reentry-r2-render-test"
    output = refresh.EXECUTION_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    control = output / "control"
    control.mkdir(mode=0o700)
    plan_path = (
        refresh.ROOT
        / "validation/results/gpu-reentry/r2-refresh-20260718/blocked-pre-r4-plan.json"
    )
    try:
        command = refresh.render_lease_child_command(
            plan_path=plan_path,
            plan_fingerprint="1" * 64,
            run_id=run_id,
            output=output,
        )
        assert command[0] == refresh.VENV_PYTHON.as_posix()
        assert command[1:4] == ["-m", "validation.gpu_reentry_refresh_v2", "live-smoke"]
        assert command[command.index("--plan-fingerprint") + 1] == "1" * 64
        assert command[command.index("--run-id") + 1] == run_id
        assert command[command.index("--output-dir") + 1] == output.relative_to(
            refresh.ROOT
        ).as_posix()
        assert refresh.lease_command_sha256(command) == refresh.lease_command_sha256(
            list(command)
        )
        launcher = refresh.render_lease_launcher_command(
            command, managed_cidfile=control / "container.cid"
        )
        assert launcher[launcher.index("--timeout-seconds") + 1] == "180"
        assert launcher[launcher.index("--managed-docker-cidfile") + 1] == str(
            control / "container.cid"
        )
        assert launcher[-len(command) :] == command
    finally:
        control.rmdir()
        output.rmdir()


def test_write_new_is_atomic_no_overwrite_and_final_fd_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The helper intentionally accepts workspace paths only; use a unique in-workspace file.
    root = refresh.ROOT / "validation/results/gpu-reentry/test-r2-refresh"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "atomic-no-overwrite.bin"
    path.unlink(missing_ok=True)
    original_link = refresh._link_anonymous_file_noreplace
    observed_before_link: list[bool] = []

    def checked_link(descriptor: int, directory_fd: int, name: str) -> None:
        observed_before_link.append(path.exists())
        if len(observed_before_link) == 1:
            assert path.exists() is False
            with pytest.raises(OSError):
                os.open(f"/proc/self/fd/{descriptor}", os.O_WRONLY)
        original_link(descriptor, directory_fd, name)

    monkeypatch.setattr(refresh, "_link_anonymous_file_noreplace", checked_link)
    try:
        refresh.write_new(path, b"first")
        assert observed_before_link == [False]
        assert path.read_bytes() == b"first"
        assert (path.stat().st_mode & 0o777) == 0o440
        with pytest.raises(refresh.RefreshError, match="already exists"):
            refresh.write_new(path, b"second")
        assert observed_before_link == [False, True]
        assert path.read_bytes() == b"first"
    finally:
        path.chmod(0o640)
        path.unlink(missing_ok=True)


def test_file_pin_rejects_a_post_publish_hardlink() -> None:
    root = refresh.ROOT / "validation/results/gpu-reentry/test-r2-refresh"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "hardlink-source.bin"
    alias = root / "hardlink-alias.bin"
    path.unlink(missing_ok=True)
    alias.unlink(missing_ok=True)
    try:
        refresh.write_new(path, b"held-fd")
        alias.hardlink_to(path)
        with pytest.raises(refresh.RefreshError, match="stable regular file identity differs"):
            refresh.file_pin(path)
    finally:
        alias.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def test_load_pinned_json_rejects_same_size_mutation_during_held_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = refresh.ROOT / "validation/results/gpu-reentry/test-r2-refresh"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "held-json-race.json"
    path.unlink(missing_ok=True)
    try:
        refresh.write_new_json(path, {"value": "aaaa"})
        expected = refresh.file_pin(path)
        original_finish = refresh._finish_stable_regular
        mutated = False

        def mutate_before_finish(
            candidate: Path, descriptor: int, before
        ):
            nonlocal mutated
            if candidate == path and not mutated:
                mutated = True
                raw = path.read_bytes()
                replacement = raw.replace(b"aaaa", b"bbbb")
                assert len(replacement) == len(raw)
                path.chmod(0o600)
                path.write_bytes(replacement)
                path.chmod(0o440)
            return original_finish(candidate, descriptor, before)

        monkeypatch.setattr(refresh, "_finish_stable_regular", mutate_before_finish)
        with pytest.raises(refresh.RefreshError, match="changed while reading"):
            refresh.load_pinned_json(expected, required_mode=0o440)
    finally:
        path.chmod(0o600) if path.exists() else None
        path.unlink(missing_ok=True)


def test_pinned_json_snapshot_does_not_reopen_through_generic_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = refresh.file_pin(refresh.PRIOR_FAILED_V1_EVIDENCE)
    monkeypatch.setattr(
        refresh,
        "load_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reopened")),
    )
    path, value = refresh.load_pinned_json(pin, required_mode=0o440)
    assert path == refresh.PRIOR_FAILED_V1_EVIDENCE
    assert value["collected_at_utc"] == refresh.PRIOR_FAILED_V1_COLLECTED_AT


def test_smoke_and_observation_envelopes_reject_extra_keys() -> None:
    smoke = {
        "torch": "2.13.0+cu130",
        "torchvision": "0.28.0+cu130",
        "cuda": "13.0",
        "cuda_available": True,
        "device_count": 1,
        "device_name": refresh.GPU_NAME,
        "compute_capability": [8, 6],
        "total_memory_bytes": refresh.GPU_MEMORY_BYTES,
        "tensor_device": "cuda:0",
        "tensor_finite": True,
        "tensor_checksum": 1.0,
        "unexpected": True,
    }
    with pytest.raises(refresh.RefreshError, match="key set differs"):
        refresh._validate_smoke_output(smoke)

    historical_observation = refresh.load_json(
        refresh.PRIOR_SUCCESSFUL_LIVE_ROOT / "live-observation.json"
    )
    historical_observation["unexpected"] = True
    historical_observation["fingerprint_sha256"] = refresh.document_fingerprint(
        historical_observation
    )
    with pytest.raises(refresh.RefreshError, match="envelope differs"):
        refresh._validate_live_observation_envelope(historical_observation)


def test_cuda_checksum_requires_the_expected_fp32_reduction() -> None:
    historical = refresh.load_json(
        refresh.PRIOR_SUCCESSFUL_LIVE_ROOT / "minimal-cuda-smoke.json"
    )
    refresh._validate_smoke_output(historical)
    for forged in (0.0, 1.0, 22_000_000_000.0):
        altered = {**historical, "tensor_checksum": forged}
        with pytest.raises(refresh.RefreshError, match="expected reduction"):
            refresh._validate_smoke_output(altered)


def test_private_run_root_and_exact_seven_artifacts_fail_closed() -> None:
    run_id = "gpu-reentry-r2-artifact-contract-test"
    run_root = refresh.EXECUTION_ROOT / run_id
    if run_root.exists():
        for item in sorted(run_root.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        run_root.rmdir()
    output = refresh._create_run_directory(run_id)
    evidence_path = output / "evidence.json"
    names = [
        "host-launch.log",
        "live-observation.json",
        "reentry-v1.json",
        "reentry-v1.md",
        "minimal-cuda-smoke.json",
        "docker.stderr.log",
    ]
    try:
        assert (refresh.EXECUTION_ROOT.stat().st_mode & 0o777) == 0o700
        assert (output.stat().st_mode & 0o777) == 0o700
        assert ((output / "control").stat().st_mode & 0o777) == 0o700
        for name in names:
            if name.endswith(".json"):
                refresh.write_new_json(output / name, {"artifact": name})
            else:
                refresh.write_new(output / name, name.encode("utf-8"))
        refresh.write_new(output / "control/container.cid", b"a" * 64)
        expected = [
            refresh.file_pin(output / "host-launch.log"),
            refresh.file_pin(output / "live-observation.json"),
            refresh.file_pin(output / "reentry-v1.json"),
            refresh.file_pin(output / "reentry-v1.md"),
            refresh.file_pin(output / "minimal-cuda-smoke.json"),
            refresh.file_pin(output / "docker.stderr.log"),
            refresh._managed_cidfile_pin(
                output / "control/container.cid", freeze=False
            ),
        ]
        assert len(expected) == 7
        observed, parsed = refresh._require_exact_run_artifacts(
            expected, evidence_path=evidence_path
        )
        assert observed == expected
        assert set(parsed) == {"observation", "v1", "smoke"}
        with pytest.raises(refresh.RefreshError, match="exact artifact path set differs"):
            refresh._require_exact_run_artifacts(
                expected[:-1], evidence_path=evidence_path
            )
        with pytest.raises(refresh.RefreshError, match="exact artifact path set differs"):
            refresh._require_exact_run_artifacts(
                [*expected[:-1], expected[0]], evidence_path=evidence_path
            )
    finally:
        for name in names:
            (output / name).unlink(missing_ok=True)
        (output / "control/container.cid").unlink(missing_ok=True)
        (output / "control").rmdir()
        output.rmdir()


def test_lease_event_replay_rejects_forged_fingerprint_and_chronology() -> None:
    prior = refresh.load_json(refresh.PRIOR_SUCCESSFUL_LIVE_EVIDENCE)
    acquire_pin = prior["gpu_lease"]["acquire_receipt"]
    release_pin = prior["gpu_lease"]["release_receipt"]
    acquire = refresh._load_verified_lease_event(
        refresh.ROOT / acquire_pin["path"], expected_pin=acquire_pin
    )
    release = refresh._load_verified_lease_event(
        refresh.ROOT / release_pin["path"], expected_pin=release_pin
    )

    forged = copy.deepcopy(acquire["value"])
    forged["event_id"] = "f" * 64
    forged_path = refresh.LEASE_RECEIPT_ROOT / ("f" * 64 + ".json")
    forged_path.unlink(missing_ok=True)
    try:
        forged_path.write_bytes(refresh.canonical_bytes(forged) + b"\n")
        forged_path.chmod(0o600)
        with pytest.raises(refresh.RefreshError, match="event replay failed"):
            refresh._load_verified_lease_event(forged_path)
    finally:
        forged_path.unlink(missing_ok=True)

    reversed_release = copy.deepcopy(release)
    reversed_release["value"]["created_monotonic_ns"] = acquire["value"][
        "created_monotonic_ns"
    ]
    with pytest.raises(refresh.RefreshError, match="monotonic chronology differs"):
        refresh._validate_lease_lifecycle(
            acquire,
            reversed_release,
            [],
            command_digest=prior["gpu_lease"]["command_argv_sha256"],
            expected_lease_id=prior["gpu_lease"]["lease_id"],
            expected_contract_fingerprint=refresh.HISTORICAL_LEASE_CONTRACT_FINGERPRINT,
        )


def test_real_acquire_renew_release_lifecycle_passes_and_advances(
    tmp_path: Path,
) -> None:
    boot_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    class Clock:
        def __init__(self) -> None:
            self.ns = 1_000_000_000
            self.wall = datetime(2026, 7, 18, tzinfo=timezone.utc)

        def monotonic_ns(self) -> int:
            return self.ns

        def wall_now(self) -> datetime:
            return self.wall

        def advance(self, seconds: int) -> None:
            self.ns += seconds * 1_000_000_000
            self.wall += timedelta(seconds=seconds)

    clock = Clock()

    def identity(pid: int, current_boot: str | None = None) -> dict:
        base = {
            "uid": os.getuid(),
            "pid": pid,
            "boot_id": current_boot or boot_id,
            "start_ticks": 12345,
        }
        return {**base, "identity_sha256": refresh.canonical_sha256(base)}

    manager = refresh.LeaseManager(
        root=tmp_path / "leases",
        gpu_uuid_resolver=lambda _index: refresh.GPU_UUID,
        boot_id_reader=lambda: boot_id,
        process_identity_reader=identity,
        monotonic_ns=clock.monotonic_ns,
        wall_now=clock.wall_now,
    )
    canonical_paths = manager._paths

    def isolated_paths(gpu_index: int):
        paths = canonical_paths(gpu_index)
        paths["legacy"] = tmp_path / f"legacy-gpu-{gpu_index}.lock"
        return paths

    manager._paths = isolated_paths  # type: ignore[method-assign]
    command = ["python", "-c", "pass"]
    credentials = manager.acquire(
        gpu_index=refresh.GPU_INDEX,
        owner_kind=refresh.LEASE_OWNER_KIND,
        command=command,
        ttl_seconds=30,
        owner_pid=os.getpid(),
    )
    clock.advance(10)
    manager.renew(
        gpu_index=refresh.GPU_INDEX,
        lease_id=credentials.lease_id,
        capability=credentials.capability,
    )
    clock.advance(5)
    manager.release(
        gpu_index=refresh.GPU_INDEX,
        lease_id=credentials.lease_id,
        capability=credentials.capability,
    )

    wrapped = []
    receipt_root = tmp_path / "leases/gpu-0/receipts"
    for path in receipt_root.glob("*.json"):
        value, raw = manager._read_receipt(path)
        wrapped.append(
            {
                "value": value,
                "pin": {
                    "path": path.as_posix(),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        )
    wrapped.sort(key=lambda item: item["value"]["created_monotonic_ns"])
    acquire, renew, release = wrapped
    ordered = refresh._validate_lease_lifecycle(
        acquire,
        release,
        [renew],
        command_digest=refresh.lease_command_sha256(command),
        expected_lease_id=credentials.lease_id,
    )
    assert ordered == [renew]

    repeated_transition = copy.deepcopy(renew)
    repeated_transition["value"]["next_lease_record_sha256"] = acquire["value"][
        "next_lease_record_sha256"
    ]
    with pytest.raises(refresh.RefreshError, match="transition hashes do not advance"):
        refresh._validate_lease_lifecycle(
            acquire,
            release,
            [repeated_transition],
            command_digest=refresh.lease_command_sha256(command),
            expected_lease_id=credentials.lease_id,
        )

    altered_owner = copy.deepcopy(renew)
    altered_owner["value"]["lease"]["owner_identity_sha256"] = "0" * 64
    with pytest.raises(refresh.RefreshError, match="semantic binding differs"):
        refresh._validate_lease_lifecycle(
            acquire,
            release,
            [altered_owner],
            command_digest=refresh.lease_command_sha256(command),
            expected_lease_id=credentials.lease_id,
        )


def test_completed_v1_replays_at_recorded_time_after_wall_clock_expiry() -> None:
    v1_path = refresh.PRIOR_SUCCESSFUL_LIVE_ROOT / "reentry-v1.json"
    evidence = refresh.load_json(v1_path)
    future = datetime(2026, 7, 20, tzinfo=timezone.utc)
    assert refresh.replay_v1_evidence(evidence, now=future)["status"] == "blocked"
    replay, collected, verified = refresh._replay_recorded_v1_evidence(
        evidence, context="test live-003 v1"
    )
    assert replay["status"] == "ready_for_operator_review"
    assert collected <= verified


def test_new_supervision_fields_are_schema_required(blocked_plan: dict) -> None:
    missing_timeout = copy.deepcopy(blocked_plan)
    del missing_timeout["gpu_lease"]["run_timeout_seconds"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_timeout, refresh.PLAN_SCHEMA)

    missing_lineage = copy.deepcopy(blocked_plan)
    del missing_lineage["source_pins"]["prior_successful_live_003_evidence"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_lineage, refresh.PLAN_SCHEMA)

    missing_incident = copy.deepcopy(blocked_plan)
    del missing_incident["prior_failed_live_smoke"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_incident, refresh.PLAN_SCHEMA)

    missing_live_004 = copy.deepcopy(blocked_plan)
    del missing_live_004["prior_failed_live_004"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_live_004, refresh.PLAN_SCHEMA)

    missing_live_004_pin = copy.deepcopy(blocked_plan)
    del missing_live_004_pin["source_pins"]["prior_failed_live_004_plan"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_live_004_pin, refresh.PLAN_SCHEMA)

    missing_scope = copy.deepcopy(blocked_plan)
    del missing_scope["prior_failed_live_smoke"]["v1_reentry_scope"]
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(missing_scope, refresh.PLAN_SCHEMA)

    old_receipt = refresh.load_json(refresh.PRIOR_SUCCESSFUL_LIVE_EVIDENCE)
    with pytest.raises(refresh.RefreshError, match="schema validation failed"):
        refresh.validate_schema(old_receipt, refresh.EVIDENCE_SCHEMA)
