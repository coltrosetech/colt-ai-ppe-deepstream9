from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from validation.endurance.resume_condition import evaluate_resume_condition


def _checkpoint(path: Path, state: str = "running") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.endurance-checkpoint/v1",
                "state": state,
            }
        ),
        encoding="utf-8",
    )


def _complete_projection(root: Path, *, status_state: str = "complete") -> tuple[Path, Path]:
    checkpoint = root / "checkpoint.json"
    status = root / "status.json"
    segments = [
        {
            "profile": 640,
            "status": "healthy",
            "duration_seconds": 10,
            "validated_seconds": 10,
        },
        {
            "profile": 960,
            "status": "healthy",
            "duration_seconds": 10,
            "validated_seconds": 10,
        },
    ]
    common = {
        "campaign_name": "deepstream9-12-camera-seven-day",
        "config_fingerprint": "a" * 64,
        "static_input_fingerprint": "b" * 64,
        "updated_at_utc": "2026-07-24T00:00:00+00:00",
        "started_at_utc": "2026-07-17T00:00:00+00:00",
        "finished_at_utc": "2026-07-24T00:00:00+00:00",
        "target_validated_seconds": 20,
        "validated_seconds": 20,
        "active": None,
        "unexpected_restarts": 0,
        "orphan_recoveries": 0,
        "campaign_health_gates": [],
        "throughput_floor": {"artifact_fingerprint": "c" * 64},
        "power_safety_policy": {"operating_policy_mode": "workstation_managed"},
    }
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.endurance-checkpoint/v1",
                "state": "complete",
                "dry_run": False,
                "segments": segments,
                **common,
            }
        ),
        encoding="utf-8",
    )
    status.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.endurance-status/v1",
                "available": True,
                "state": status_state,
                "dry_run": False,
                "progress_fraction": 1,
                "segments": {
                    "total": 2,
                    "status_counts": {"healthy": 2},
                },
                "profiles_validated_seconds": {"640": 10, "960": 10},
                "scheduled_profile_rotations_completed": 1,
                "scheduled_profile_rotations_target": 1,
                **common,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, status


def test_missing_checkpoint_fails_closed_instead_of_starting_fresh(tmp_path: Path) -> None:
    code, message = evaluate_resume_condition(
        tmp_path / "checkpoint.json", tmp_path / "supervisor.lock"
    )
    assert code == 2
    assert "not a fresh-start path" in message


def test_incomplete_checkpoint_and_free_lock_is_ready(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    lock = tmp_path / "supervisor.lock"
    _checkpoint(checkpoint)
    lock.touch()
    assert evaluate_resume_condition(checkpoint, lock)[0] == 0


def test_complete_checkpoint_and_matching_terminal_status_is_expected_skip(
    tmp_path: Path,
) -> None:
    checkpoint, _status = _complete_projection(tmp_path)
    code, message = evaluate_resume_condition(checkpoint, tmp_path / "lock")
    assert code == 1
    assert "complete" in message


def test_complete_checkpoint_with_missing_status_runs_projection_repair(
    tmp_path: Path,
) -> None:
    checkpoint, status = _complete_projection(tmp_path)
    status.unlink()

    code, message = evaluate_resume_condition(checkpoint, tmp_path / "lock")

    assert code == 0
    assert "projection repair" in message


def test_complete_checkpoint_with_stale_status_runs_projection_repair(
    tmp_path: Path,
) -> None:
    checkpoint, _status = _complete_projection(tmp_path, status_state="running")

    code, message = evaluate_resume_condition(checkpoint, tmp_path / "lock")

    assert code == 0
    assert "projection repair" in message


def test_complete_checkpoint_rejects_boolean_terminal_progress(
    tmp_path: Path,
) -> None:
    checkpoint, status_path = _complete_projection(tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["progress_fraction"] = True
    status_path.write_text(json.dumps(status), encoding="utf-8")

    code, message = evaluate_resume_condition(checkpoint, tmp_path / "lock")

    assert code == 0
    assert "projection repair" in message


def test_owned_lock_is_expected_skip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    lock = tmp_path / "supervisor.lock"
    _checkpoint(checkpoint)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, message = evaluate_resume_condition(checkpoint, lock)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert code == 1
    assert "owned" in message


def test_malformed_or_duplicate_checkpoint_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{not-json", encoding="utf-8")
    assert evaluate_resume_condition(checkpoint, tmp_path / "lock")[0] == 2
    checkpoint.write_text(
        '{"schema_version":"deepsafe.endurance-checkpoint/v1",'
        '"state":"running","state":"complete"}',
        encoding="utf-8",
    )
    assert evaluate_resume_condition(checkpoint, tmp_path / "lock")[0] == 2


def test_checkpoint_and_lock_symlinks_fail_closed(tmp_path: Path) -> None:
    real_checkpoint = tmp_path / "real-checkpoint.json"
    _checkpoint(real_checkpoint)
    checkpoint_link = tmp_path / "checkpoint.json"
    checkpoint_link.symlink_to(real_checkpoint)
    assert evaluate_resume_condition(checkpoint_link, tmp_path / "lock")[0] == 2

    checkpoint = tmp_path / "plain-checkpoint.json"
    _checkpoint(checkpoint)
    real_lock = tmp_path / "real.lock"
    real_lock.touch()
    lock_link = tmp_path / "supervisor.lock"
    lock_link.symlink_to(real_lock)
    assert evaluate_resume_condition(checkpoint, lock_link)[0] == 2


def test_wrong_schema_or_non_string_state_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"schema_version": "wrong", "state": "running"}),
        encoding="utf-8",
    )
    assert evaluate_resume_condition(checkpoint, tmp_path / "lock")[0] == 2
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "deepsafe.endurance-checkpoint/v1",
                "state": True,
            }
        ),
        encoding="utf-8",
    )
    assert evaluate_resume_condition(checkpoint, tmp_path / "lock")[0] == 2
