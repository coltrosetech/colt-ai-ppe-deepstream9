from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from admin import validation as admin_validation
from validation import finalize_validation as finalizer


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _completion_documents(*, state: str = "complete") -> tuple[dict, dict]:
    started = "2026-07-17T00:00:00+00:00"
    finished = "2026-07-24T00:00:00+00:00" if state == "complete" else None
    updated = finished or "2026-07-17T00:01:00+00:00"
    segments = []
    for index in range(finalizer.SEGMENT_COUNT):
        profile = finalizer.PROFILES[index % 2]
        healthy = state == "complete"
        segments.append(
            {
                "index": index,
                "segment_id": f"segment-{index:03d}-{profile}",
                "profile": profile,
                "duration_seconds": finalizer.SEGMENT_SECONDS,
                "campaign_day": index // 4 + 1,
                "status": "healthy" if healthy else "pending",
                "attempts": [{"status": "healthy"}] if healthy else [],
                "attempt_receipts": (
                    [
                        {
                            "path": (
                                "validation/results/endurance/current/segments/"
                                f"segment-{index:03d}-{profile}/attempt-01/attempt-receipt.json"
                            ),
                            "size_bytes": 100 + index,
                            "sha256": "1" * 64,
                        }
                    ]
                    if healthy
                    else []
                ),
                "validated_seconds": finalizer.SEGMENT_SECONDS if healthy else 0,
            }
        )
    validated = finalizer.TARGET_SECONDS if state == "complete" else 0
    active = None
    throughput_floor = {"artifact_fingerprint": "3" * 64}
    power = {"operating_policy_mode": "workstation_managed"}
    checkpoint = {
        "schema_version": finalizer.CHECKPOINT_SCHEMA,
        "campaign_name": finalizer.CAMPAIGN_NAME,
        "config_fingerprint": "a" * 64,
        "static_input_fingerprint": "b" * 64,
        "state": state,
        "dry_run": False,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "updated_at_utc": updated,
        "target_validated_seconds": finalizer.TARGET_SECONDS,
        "validated_seconds": validated,
        "active": active,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "campaign_health_gates": [],
        "throughput_floor": throughput_floor,
        "power_safety_policy": power,
        "segments": segments,
    }
    status_counts = {"healthy": finalizer.SEGMENT_COUNT} if state == "complete" else {"pending": finalizer.SEGMENT_COUNT}
    status = {
        "schema_version": finalizer.STATUS_SCHEMA,
        "available": True,
        "campaign_name": finalizer.CAMPAIGN_NAME,
        "config_fingerprint": "a" * 64,
        "static_input_fingerprint": "b" * 64,
        "state": state,
        "dry_run": False,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "updated_at_utc": updated,
        "target_validated_seconds": finalizer.TARGET_SECONDS,
        "validated_seconds": validated,
        "progress_fraction": 1 if state == "complete" else 0,
        "active": active,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "campaign_health_gates": [],
        "throughput_floor": throughput_floor,
        "power_safety_policy": power,
        "segments": {
            "total": finalizer.SEGMENT_COUNT,
            "status_counts": status_counts,
        },
        "profiles_validated_seconds": {
            "640": finalizer.PROFILE_SECONDS if state == "complete" else 0,
            "960": finalizer.PROFILE_SECONDS if state == "complete" else 0,
        },
        "scheduled_profile_rotations_completed": finalizer.SEGMENT_COUNT - 1 if state == "complete" else 0,
        "scheduled_profile_rotations_target": finalizer.SEGMENT_COUNT - 1,
    }
    return checkpoint, status


def _project(tmp_path: Path, *, state: str = "complete") -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    results = project / "validation/results"
    current = results / "endurance/current"
    current.mkdir(parents=True)
    (current / "supervisor.lock").write_text("", encoding="utf-8")
    checkpoint, status = _completion_documents(state=state)
    _write_json(current / "checkpoint.json", checkpoint)
    _write_json(current / "status.json", status)
    _write_json(current / "campaign-resolved.json", {"schema_version": "fixture"})
    _write_json(current / "plan.json", {"schema_version": "fixture"})
    for relative in finalizer.INPUT_RELATIVE_PATHS:
        path = project / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    runtime_lock = tmp_path / "runtime/finalizer.lock"
    runtime_lock.parent.mkdir()
    return project, results, runtime_lock


def _fake_runner(command: list[str], _cwd: Path, _timeout: int) -> subprocess.CompletedProcess[str]:
    module = command[2]
    output_dir = Path(command[command.index("--output-dir") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"producer": module}
    _write_json(output_dir / "report.json", payload)
    (output_dir / "report.md").write_text(f"# {module}\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def _fake_bundle(config: finalizer.FinalizerConfig) -> dict:
    outputs = []
    for artifact_id, relative, media_type in finalizer.OUTPUT_SPECS:
        read = finalizer._secure_read(
            config.project_root,
            relative,
            max_bytes=finalizer.MAX_JSON_BYTES if relative.endswith(".json") else finalizer.MAX_TEXT_BYTES,
        )
        outputs.append(read.pin.public(artifact_id=artifact_id, media_type=media_type))
    return {
        "outputs": outputs,
        "semantics": {
            "campaign_endurance_accepted": True,
            "objective_evidence_complete": True,
            "objective_passed_gate_count": 6,
            "product_snapshot_valid": True,
            "product_status": "not_ready",
            "product_ready_required": False,
        },
    }


def _run(project: Path, results: Path, runtime_lock: Path, **kwargs: object) -> finalizer.Outcome:
    return finalizer.finalize(
        project_root=project,
        results_root=results,
        runtime_lock=runtime_lock,
        python=Path(sys.executable),
        runner=kwargs.pop("runner", _fake_runner),  # type: ignore[arg-type]
        bundle_verifier=kwargs.pop("bundle_verifier", _fake_bundle),  # type: ignore[arg-type]
        **kwargs,
    )


def test_running_campaign_is_machine_readable_waiting_and_writes_no_reports(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path, state="running")

    outcome = _run(project, results, runtime_lock)

    assert outcome.public()["status"] == "waiting"
    assert outcome.reason == "endurance_not_complete"
    assert outcome.retryable is True
    assert outcome.mutated is False
    assert not (results / "campaign-report").exists()
    assert not (results / "objective-completion").exists()
    assert not (results / "product-readiness").exists()
    assert not (results / "finalization").exists()


def test_held_supervisor_lock_is_waiting_and_no_write(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    lock_path = results / "endurance/current/supervisor.lock"
    descriptor = os.open(lock_path, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        outcome = _run(project, results, runtime_lock)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert outcome.status == "waiting"
    assert outcome.reason == "supervisor_busy"
    assert outcome.mutated is False
    assert not (results / "campaign-report").exists()
    assert not (results / "finalization").exists()


def test_stale_checkpoint_status_identity_fails_before_mutation(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    status_path = results / "endurance/current/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["updated_at_utc"] = "2026-07-24T00:00:01+00:00"
    _write_json(status_path, status)

    outcome = _run(project, results, runtime_lock)

    assert outcome.status == "failed"
    assert outcome.reason == "checkpoint_status_identity_mismatch"
    assert outcome.mutated is False
    assert outcome.exit_code == 2
    assert not (results / "campaign-report").exists()
    assert not (results / "finalization").exists()


def test_complete_checkpoint_with_running_status_is_not_treated_as_waiting(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    status_path = results / "endurance/current/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["state"] = "running"
    _write_json(status_path, status)

    outcome = _run(project, results, runtime_lock)

    assert outcome.status == "failed"
    assert outcome.reason == "checkpoint_status_state_mismatch"
    assert outcome.mutated is False
    assert not (results / "campaign-report").exists()


def test_retry_receipts_are_bound_to_checkpoint_total_not_fixed_at_28() -> None:
    checkpoint, status = _completion_documents()
    first = checkpoint["segments"][0]
    first["attempts"].append({"status": "healthy"})
    first["attempt_receipts"].append(
        {
            "path": (
                "validation/results/endurance/current/segments/"
                "segment-000-640/attempt-02/attempt-receipt.json"
            ),
            "size_bytes": 222,
            "sha256": "2" * 64,
        }
    )

    identity = finalizer._completion_identity(checkpoint, status)
    assert identity["total_attempt_receipt_count"] == 29
    endurance = {
        "accepted": True,
        "evidence_complete": True,
        "target_validated_seconds": finalizer.TARGET_SECONDS,
        "reported_validated_seconds": finalizer.TARGET_SECONDS,
        "expected_segments": 28,
        "healthy_checkpoint_segments": 28,
        "verified_latest_attempt_status_files": 28,
        "verified_attempt_receipts": 29,
        "raw_attempt_telemetry_replays_verified": 28,
        "valid_daily_reports": 7,
        "performance_quality_outcome": "passed",
        "throughput_floor": {
            "passing_endurance_attempts": 28,
            "expected_endurance_attempts": 28,
            "proven_floor_violations": 0,
        },
    }
    assert finalizer._campaign_endurance_semantics_valid(
        endurance, {"state": "pass"}, expected_attempt_receipts=29
    )
    endurance["verified_attempt_receipts"] = 28
    assert not finalizer._campaign_endurance_semantics_valid(
        endurance, {"state": "pass"}, expected_attempt_receipts=29
    )


def test_boolean_progress_fraction_is_rejected(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    status_path = results / "endurance/current/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["progress_fraction"] = True
    _write_json(status_path, status)

    outcome = _run(project, results, runtime_lock)

    assert outcome.status == "failed"
    assert outcome.reason == "status_completion_projection_invalid"
    assert outcome.mutated is False


def test_complete_success_receipt_is_last_and_product_not_ready_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletedDateTime(finalizer.datetime):
        @classmethod
        def now(cls, tz: object = None) -> "CompletedDateTime":
            value = cls.fromisoformat("2026-07-24T00:00:01+00:00")
            return value if tz is None else value.astimezone(tz)  # type: ignore[arg-type]

    monkeypatch.setattr(finalizer, "datetime", CompletedDateTime)
    project, results, runtime_lock = _project(tmp_path)

    outcome = _run(project, results, runtime_lock)

    assert outcome.status == "complete"
    assert outcome.reason == "finalized"
    assert outcome.exit_code == 0
    receipt_path = results / "finalization/current/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "complete"
    assert receipt["semantics"]["product_status"] == "not_ready"
    assert receipt["semantics"]["product_ready_required"] is False
    assert receipt["lock_contract"]["receipt_committed_last"] is True
    assert len(receipt["inputs"]) == len(finalizer.INPUT_RELATIVE_PATHS)
    assert len(receipt["outputs"]) == len(finalizer.OUTPUT_SPECS)
    assert finalizer._receipt_fingerprint_valid(receipt)
    assert admin_validation._completion_identity_valid(receipt)
    assert admin_validation._finalization_receipt_shape_valid(receipt)
    for pin in receipt["outputs"]:
        content = (project / pin["path"]).read_bytes()
        assert len(content) == pin["size_bytes"]
        assert finalizer._sha256_bytes(content) == pin["sha256"]


def test_generator_failure_never_publishes_receipt_and_reports_mutation(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)

    def failing_runner(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
        if command[2] == "validation.report_campaign":
            _fake_runner(command, cwd, timeout)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="injected")

    outcome = _run(project, results, runtime_lock, runner=failing_runner)

    assert outcome.status == "failed"
    assert outcome.reason == "generator_failed"
    assert outcome.retryable is True
    assert outcome.exit_code == 4
    assert outcome.mutated is True
    assert not (results / "finalization/current/receipt.json").exists()


def test_bundle_change_on_second_verification_prevents_receipt(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    calls = 0

    def changing_verifier(config: finalizer.FinalizerConfig) -> dict:
        nonlocal calls
        calls += 1
        bundle = _fake_bundle(config)
        if calls == 2:
            bundle = copy.deepcopy(bundle)
            bundle["semantics"]["product_status"] = "changed"
        return bundle

    outcome = _run(
        project,
        results,
        runtime_lock,
        bundle_verifier=changing_verifier,
    )

    assert calls == 2
    assert outcome.status == "failed"
    assert outcome.reason == "bundle_changed_before_receipt_commit"
    assert outcome.exit_code == 4
    assert not (results / "finalization/current/receipt.json").exists()


def test_postcommit_bundle_change_rolls_back_new_receipt(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    calls = 0

    def changing_verifier(config: finalizer.FinalizerConfig) -> dict:
        nonlocal calls
        calls += 1
        bundle = _fake_bundle(config)
        if calls == 3:
            bundle = copy.deepcopy(bundle)
            bundle["semantics"]["product_status"] = "changed"
        return bundle

    outcome = _run(
        project,
        results,
        runtime_lock,
        bundle_verifier=changing_verifier,
    )

    assert calls == 3
    assert outcome.status == "failed"
    assert outcome.reason == "bundle_changed_after_receipt_commit"
    assert not (results / "finalization/current/receipt.json").exists()


def test_identical_rerun_is_idempotent_and_conflicting_output_fails_closed(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    first = _run(project, results, runtime_lock)
    assert first.status == "complete"
    receipt_path = results / "finalization/current/receipt.json"
    receipt_before = receipt_path.read_bytes()

    def forbidden_runner(*_args: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("idempotent finalization must not rerun generators")

    second = _run(project, results, runtime_lock, runner=forbidden_runner)
    assert second.status == "complete"
    assert second.idempotent is True
    assert second.mutated is False
    assert receipt_path.read_bytes() == receipt_before

    (results / "campaign-report/report.md").write_text("tampered\n", encoding="utf-8")
    conflict = _run(project, results, runtime_lock, runner=forbidden_runner)
    assert conflict.status == "failed"
    assert conflict.reason == "finalization_receipt_conflict"
    assert conflict.mutated is False
    assert receipt_path.read_bytes() == receipt_before


def test_exact_hardlink_commit_crash_is_recovered_idempotently(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    first = _run(project, results, runtime_lock)
    assert first.status == "complete"
    receipt = results / "finalization/current/receipt.json"
    temporary = receipt.parent / ".receipt.json.tmp-crash-fixture"
    os.link(receipt, temporary)
    assert receipt.stat().st_nlink == 2

    def forbidden_runner(*_args: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("recovered committed receipt must be idempotent")

    outcome = _run(project, results, runtime_lock, runner=forbidden_runner)

    assert outcome.status == "complete"
    assert outcome.idempotent is True
    assert not temporary.exists()
    assert receipt.stat().st_nlink == 1


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_unsafe_supervisor_lock_fails_closed(tmp_path: Path, kind: str) -> None:
    project, results, runtime_lock = _project(tmp_path)
    lock_path = results / "endurance/current/supervisor.lock"
    lock_path.unlink()
    other = tmp_path / "other.lock"
    other.write_text("", encoding="utf-8")
    if kind == "symlink":
        lock_path.symlink_to(other)
    else:
        os.link(other, lock_path)

    outcome = _run(project, results, runtime_lock)

    assert outcome.status == "failed"
    assert outcome.mutated is False
    assert not (results / "finalization").exists()


def test_hardlinked_checkpoint_and_concurrent_finalizer_lock_fail_closed(tmp_path: Path) -> None:
    project, results, runtime_lock = _project(tmp_path)
    checkpoint = results / "endurance/current/checkpoint.json"
    os.link(checkpoint, tmp_path / "checkpoint-hardlink.json")
    unsafe = _run(project, results, runtime_lock)
    assert unsafe.status == "failed"
    assert unsafe.reason == "unsafe_artifact"

    (tmp_path / "checkpoint-hardlink.json").unlink()
    descriptor = os.open(runtime_lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        busy = _run(project, results, runtime_lock)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert busy.status == "waiting"
    assert busy.reason == "finalizer_busy"
    assert busy.mutated is False


def test_same_length_concurrent_rewrite_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"AAAA")
    real_read = finalizer.os.read
    raced = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        if not raced:
            raced = True
            with artifact.open("r+b") as handle:
                handle.write(b"BBBB")
                handle.flush()
                os.fsync(handle.fileno())
        return real_read(descriptor, size)

    monkeypatch.setattr(finalizer.os, "read", racing_read)
    with pytest.raises(finalizer.FinalizationError) as caught:
        finalizer._secure_read(root, "artifact.bin", max_bytes=16)
    assert caught.value.code == "artifact_identity_changed"


def test_custom_results_root_is_rejected_before_lock_or_output(tmp_path: Path) -> None:
    project, _results, runtime_lock = _project(tmp_path)
    custom = project / "custom-results"
    custom.mkdir()

    outcome = finalizer.finalize(
        project_root=project,
        results_root=custom,
        runtime_lock=runtime_lock,
        runner=_fake_runner,
        bundle_verifier=_fake_bundle,
    )

    assert outcome.status == "failed"
    assert outcome.reason == "results_root_not_canonical"
    assert outcome.mutated is False
    assert not runtime_lock.exists()


def test_campaign_hardware_blocker_participates_in_acceptance_recompute() -> None:
    report = {
        "requirements": [
            {
                "id": "seven_day_endurance",
                "state": "pass",
                "required_for_acceptance": True,
            }
        ],
        "requirement_summary": {
            "total": 1,
            "passed": 1,
            "state_counts": {"pass": 1},
        },
        "hardware_blockers": [],
        "decision": {
            "accepted": True,
            "final_claim_allowed": True,
            "failed_required_gates": [],
        },
    }
    assert finalizer._campaign_aggregate_semantics_valid(report)
    report["hardware_blockers"] = [{"code": "x"}]
    assert not finalizer._campaign_aggregate_semantics_valid(report)
    report["decision"]["accepted"] = False
    report["decision"]["final_claim_allowed"] = False
    assert finalizer._campaign_aggregate_semantics_valid(report)
