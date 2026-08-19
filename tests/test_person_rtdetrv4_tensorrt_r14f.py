from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_tensorrt_r14f as lane


STAMP = "2026-07-18T18:00:00+03:00"


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".person-r14f-test-", dir=lane.ROOT)
    try:
        yield Path(context.name)
    finally:
        context.cleanup()


def _json_payload(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_immutable(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o440)
    return {
        "path": lane.r14e.repo_relative(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _memfd(payload: bytes) -> int:
    descriptor = os.memfd_create("person-r14f-test", os.MFD_CLOEXEC)
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        assert count > 0
        view = view[count:]
    os.fchmod(descriptor, 0o440)
    return descriptor


def _pin(path: Path, payload: bytes, *, fingerprint: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": lane.r14e.repo_relative(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if fingerprint is not None:
        value["fingerprint_sha256"] = fingerprint
    return value


def _minimal_plan() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "test-plan/v1",
        "plan_id": "canonical-plan",
        "prepared_at_utc": STAMP,
    }
    value["fingerprint_sha256"] = lane.r14e.fingerprint(value)
    return value


def _minimal_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "plan_id",
            "prepared_at_utc",
            "fingerprint_sha256",
        ],
        "properties": {
            "schema_version": {"const": "test-plan/v1"},
            "plan_id": {"const": "canonical-plan"},
            "prepared_at_utc": {"type": "string"},
            "fingerprint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def _plan_fixture(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    schema_path = repo_tmp_path / "schema/plan.schema.json"
    plan_path = repo_tmp_path / "plan/accepted.json"
    plan = _minimal_plan()
    _write_immutable(schema_path, _json_payload(_minimal_schema()))
    _write_immutable(plan_path, _json_payload(plan))
    monkeypatch.setattr(lane, "PLAN_SCHEMA", schema_path)
    monkeypatch.setattr(lane, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(lane.r14e, "DEFAULT_PLAN", plan_path)
    return schema_path, plan_path, plan


def _forbid_external(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("external process/run creation must not be reached")

    monkeypatch.setattr(lane.subprocess, "Popen", forbidden)
    monkeypatch.setattr(lane.r14e.HeldRunDirectories, "create", classmethod(forbidden))


def _make_run(repo_tmp_path: Path) -> lane.r14e.HeldRunDirectories:
    run = repo_tmp_path / "runs/tensorrt_fp16_640/unit"
    output = run / "output"
    control = run / "control"
    output.mkdir(parents=True, mode=0o700)
    control.mkdir(mode=0o700)
    for path in (run, output, control):
        path.chmod(0o700)
    return lane.r14e.HeldRunDirectories.open_existing(run)


def _transaction_fixture(
    repo_tmp_path: Path,
) -> tuple[
    lane.r14e.HeldRunDirectories,
    lane.HeldPublicationIntent,
    dict[str, int],
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    run_directories = _make_run(repo_tmp_path)
    published_root = repo_tmp_path / "canonical"
    published_root.mkdir(mode=0o700)
    roles = ["artifact", "worker_result", "r11_stage_receipt", "execution_receipt"]
    descriptors: dict[str, int] = {}
    destinations: dict[str, Path] = {}
    members: list[dict[str, Any]] = []
    for index, role in enumerate(roles):
        payload = f"{role}-source-{index}\n".encode()
        descriptor = _memfd(payload)
        descriptors[role] = descriptor
        source_path = repo_tmp_path / "logical-source" / f"{role}.bin"
        destination = published_root / f"{role}.bin"
        destinations[role] = destination
        members.append(
            {
                "role": role,
                "source": _pin(source_path, payload),
                "destination": lane.r14e.repo_relative(destination),
                "published": _pin(destination, payload),
            }
        )
    commit_destination = published_root / "commit.json"
    destinations["commit"] = commit_destination
    intent_value: dict[str, Any] = {
        "schema_version": "test-intent/v1",
        "status": "prepared_not_committed",
        "members": members,
        "commit_destination": lane.r14e.repo_relative(commit_destination),
    }
    intent_value["fingerprint_sha256"] = lane.r14e.fingerprint(intent_value)
    intent = lane.HeldPublicationIntent.create(run_directories, intent_value)
    commit_value: dict[str, Any] = {
        "schema_version": "test-commit/v1",
        "status": "committed",
        "publication_intent": intent.document_pin,
    }
    commit_value["fingerprint_sha256"] = lane.r14e.fingerprint(commit_value)
    commit_payload = _json_payload(commit_value)
    commit_descriptor = _memfd(commit_payload)
    commit_source_pin = _pin(
        repo_tmp_path / "logical-source/commit.json",
        commit_payload,
        fingerprint=commit_value["fingerprint_sha256"],
    )
    commit_destination_pin = _pin(
        commit_destination,
        commit_payload,
        fingerprint=commit_value["fingerprint_sha256"],
    )
    return (
        run_directories,
        intent,
        descriptors,
        commit_descriptor,
        commit_source_pin,
        commit_destination_pin,
        destinations,
    )


def _close_transaction_fixture(
    run_directories: lane.r14e.HeldRunDirectories,
    intent: lane.HeldPublicationIntent,
    descriptors: dict[str, int],
    commit_descriptor: int,
) -> None:
    for descriptor in descriptors.values():
        os.close(descriptor)
    os.close(commit_descriptor)
    intent.close_best_effort()
    run_directories.close()


def _attack_intent(intent: lane.HeldPublicationIntent, kind: str) -> None:
    path = intent.source.path
    if kind == "delete":
        path.unlink()
        return
    moved = path.with_name(path.name + ".moved")
    path.rename(moved)
    if kind == "replace":
        path.write_bytes(_json_payload(intent.value))
        path.chmod(0o440)


def test_frozen_r14d_r14e_inventory_is_exact_and_still_absent() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed["r14d"] == lane.FROZEN_R14D_LINEAGE
    assert observed["r14e"] == lane.FROZEN_R14E_LINEAGE
    assert len(observed["r14d"]) == 11
    assert len(observed["r14e"]) == 11
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.DEFAULT_CONTROLLER.exists()
    assert not lane.RUNS_ROOT.exists()


def test_controller_contract_schema_and_closed_claim_boundary() -> None:
    schema = json.loads(lane.CONTROLLER_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    Draft202012Validator(schema).validate(value)
    assert value["accepted_lineage"]["r14d"] == lane.FROZEN_R14D_LINEAGE
    assert value["accepted_lineage"]["r14e"] == lane.FROZEN_R14E_LINEAGE
    assert value["execution_gate"] == {
        "authorized": False,
        "gpu_lease_v3_audited": False,
        "block_reason": lane.EXECUTION_GATE,
    }
    assert value["claim_boundary"]["gpu_executed"] is False
    assert value["claim_boundary"]["production_ready"] is False
    assert value["fingerprint_sha256"] == lane.r14e.fingerprint(value)

    forged = copy.deepcopy(value)
    forged["accepted_lineage"]["r14e"] = {
        f"validation/forged-{index}.py": {
            "mode": "0440", "bytes": 1, "sha256": "0" * 64,
        }
        for index in range(11)
    }
    forged["fingerprint_sha256"] = lane.r14e.fingerprint(forged)
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(forged)


def test_noncanonical_exact_plan_copy_fails_before_open_popen_or_run_mkdir(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    noncanonical = repo_tmp_path / "plan/exact-copy.json"
    shutil.copyfile(plan_path, noncanonical)
    noncanonical.chmod(0o440)
    _forbid_external(monkeypatch)

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("canonical path must fail before a held open")

    monkeypatch.setattr(lane.HeldCanonicalFile, "open_existing", forbidden_open)
    with pytest.raises(lane.TensorRTR14FError, match="canonical DEFAULT_PLAN"):
        lane.verify_plan_held(
            noncanonical,
            expected_fingerprint=plan["fingerprint_sha256"],
        )


def test_held_schema_object_and_exact_projection_succeed(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_args, **_kwargs: copy.deepcopy(plan))
    held = lane.verify_plan_held(
        plan_path,
        expected_fingerprint=plan["fingerprint_sha256"],
    )
    try:
        assert held.value == plan
        assert held.schema == _minimal_schema()
        assert held.document_pin["fingerprint_sha256"] == plan["fingerprint_sha256"]
        held.assert_stable()
    finally:
        held.close()


def test_plan_schema_descriptor_is_opened_before_accepted_plan(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    schema_path, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    original = lane.HeldCanonicalFile.open_existing.__func__
    opened: list[Path] = []

    def recording(cls: object, path: Path, **kwargs: object) -> lane.HeldCanonicalFile:
        opened.append(Path(path))
        return original(cls, path, **kwargs)

    monkeypatch.setattr(
        lane.HeldCanonicalFile,
        "open_existing",
        classmethod(recording),
    )
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_args, **_kwargs: copy.deepcopy(plan))
    held = lane.verify_plan_held(
        plan_path,
        expected_fingerprint=plan["fingerprint_sha256"],
    )
    try:
        assert opened[:2] == [schema_path, plan_path]
    finally:
        held.close()


def test_schema_a_b_a_swap_is_detected_before_popen_or_run_mkdir(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    schema_path, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    _forbid_external(monkeypatch)

    def swap_and_restore(*_args: object, **_kwargs: object) -> dict[str, Any]:
        backup = schema_path.with_name("plan.schema.original")
        schema_path.rename(backup)
        replacement = {"$schema": "https://json-schema.org/draft/2020-12/schema"}
        schema_path.write_bytes(_json_payload(replacement))
        schema_path.chmod(0o440)
        schema_path.unlink()
        backup.rename(schema_path)
        return copy.deepcopy(plan)

    monkeypatch.setattr(lane.r14e, "verify_plan", swap_and_restore)
    with pytest.raises(
        lane.TensorRTR14FError,
        match="held inode metadata differs|parent namespace changed",
    ):
        lane.verify_plan_held(
            plan_path,
            expected_fingerprint=plan["fingerprint_sha256"],
        )


def test_exact_verify_plan_projection_drift_fails_before_external_action(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    _forbid_external(monkeypatch)
    drift = copy.deepcopy(plan)
    drift["plan_id"] = "different-live-projection"
    drift["fingerprint_sha256"] = lane.r14e.fingerprint(drift)
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_args, **_kwargs: drift)
    with pytest.raises(lane.TensorRTR14FError, match="projection drifted"):
        lane.verify_plan_held(
            plan_path,
            expected_fingerprint=plan["fingerprint_sha256"],
        )


def test_execute_gate_is_before_plan_open_run_mkdir_and_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_external(monkeypatch)

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed gate must precede plan open")

    monkeypatch.setattr(lane.HeldCanonicalFile, "open_existing", forbidden_open)
    with pytest.raises(lane.TensorRTR14FError, match="execution gate is closed"):
        lane.execute_stage(
            plan_path=lane.DEFAULT_PLAN,
            accepted_plan_fingerprint="0" * 64,
            stage="tensorrt_fp16_640",
            run_id="r14f-closed-gate-unit",
        )
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.RUNS_ROOT.exists()


def test_canonical_intent_transaction_success_has_exact_commit_binding(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    try:
        result = lane.execute_publication_from_held_fds(
            intent=intent,
            member_descriptors=descriptors,
            commit_descriptor=commit_fd,
            commit_source_pin=source_pin,
            commit_destination_pin=destination_pin,
        )
        assert result["status"] == "committed"
        assert result["intent"] == intent.document_pin
        assert paths["commit"].is_file()
        assert intent.source.path.is_file()
        assert os.stat(intent.source.path).st_nlink == 1
        intent.assert_canonical(boundary="test_success")
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


@pytest.mark.parametrize("attack", ["rename", "delete", "replace"])
def test_precommit_intent_attack_leaves_commit_and_members_absent(
    attack: str,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    fired = False

    def hook(boundary: str, held_intent: lane.HeldPublicationIntent) -> None:
        nonlocal fired
        if boundary == "before_commit_link" and not fired:
            fired = True
            _attack_intent(held_intent, attack)

    try:
        with pytest.raises((lane.TensorRTR14FError, FileNotFoundError)):
            lane.publish_transaction_from_held_fds(
                intent=intent,
                member_descriptors=descriptors,
                commit_descriptor=commit_fd,
                commit_source_pin=source_pin,
                commit_destination_pin=destination_pin,
                recovery=False,
                boundary_hook=hook,
            )
        assert fired is True
        assert not paths["commit"].exists()
        assert all(not paths[role].exists() for role in descriptors)
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


@pytest.mark.parametrize("attack", ["rename", "delete", "replace"])
def test_postcommit_intent_attack_rolls_back_canonical_commit(
    attack: str,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    fired = False

    def hook(boundary: str, held_intent: lane.HeldPublicationIntent) -> None:
        nonlocal fired
        if boundary == "after_commit_all_fsync" and not fired:
            fired = True
            _attack_intent(held_intent, attack)

    try:
        with pytest.raises((lane.TensorRTR14FError, FileNotFoundError)):
            lane.publish_transaction_from_held_fds(
                intent=intent,
                member_descriptors=descriptors,
                commit_descriptor=commit_fd,
                commit_source_pin=source_pin,
                commit_destination_pin=destination_pin,
                recovery=False,
                boundary_hook=hook,
            )
        assert fired is True
        assert not paths["commit"].exists()
        assert all(not paths[role].exists() for role in descriptors)
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


def test_postcommit_commit_name_replacement_is_removed_as_poison(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    fired = False

    def hook(boundary: str, _held_intent: lane.HeldPublicationIntent) -> None:
        nonlocal fired
        if boundary == "after_commit_all_fsync" and not fired:
            fired = True
            moved = paths["commit"].with_name("commit.original")
            paths["commit"].rename(moved)
            paths["commit"].write_bytes(b"poison\n")
            paths["commit"].chmod(0o440)

    try:
        with pytest.raises(
            lane.TensorRTR14FError,
            match="held inode metadata differs|canonical name/held inode",
        ):
            lane.publish_transaction_from_held_fds(
                intent=intent,
                member_descriptors=descriptors,
                commit_descriptor=commit_fd,
                commit_source_pin=source_pin,
                commit_destination_pin=destination_pin,
                recovery=False,
                boundary_hook=hook,
            )
        assert fired is True
        assert not paths["commit"].exists()
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


def test_control_directory_swap_before_commit_is_detected_and_commit_absent(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    fired = False

    def hook(boundary: str, held_intent: lane.HeldPublicationIntent) -> None:
        nonlocal fired
        if boundary == "before_commit_link" and not fired:
            fired = True
            control = held_intent.run_directories.control
            moved = control.with_name("control.original")
            control.rename(moved)
            control.mkdir(mode=0o700)
            replacement = control / lane.INTENT_NAME
            replacement.write_bytes(_json_payload(held_intent.value))
            replacement.chmod(0o440)

    try:
        with pytest.raises(lane.TensorRTR14FError, match="directory identity differs"):
            lane.publish_transaction_from_held_fds(
                intent=intent,
                member_descriptors=descriptors,
                commit_descriptor=commit_fd,
                commit_source_pin=source_pin,
                commit_destination_pin=destination_pin,
                recovery=False,
                boundary_hook=hook,
            )
        assert fired is True
        assert not paths["commit"].exists()
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


def test_recovery_reuses_same_intent_lifetime_and_exact_existing_links(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, _paths = fixture
    try:
        first = lane.publish_transaction_from_held_fds(
            intent=intent,
            member_descriptors=descriptors,
            commit_descriptor=commit_fd,
            commit_source_pin=source_pin,
            commit_destination_pin=destination_pin,
            recovery=False,
        )
        expected_intent = intent.document_pin
        intent.close()
        intent = lane.HeldPublicationIntent.open_existing(
            run_directories, expected_intent,
        )
        second = lane.recover_publication_from_held_fds(
            intent=intent,
            member_descriptors=descriptors,
            commit_descriptor=commit_fd,
            commit_source_pin=source_pin,
            commit_destination_pin=destination_pin,
        )
        assert first["status"] == "committed"
        assert second["status"] == "already_committed"
        assert second["intent"] == expected_intent
        intent.assert_canonical(boundary="recovery_success")
    finally:
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


def test_signal_restore_baseexception_rolls_back_before_handles_close(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    run_directories, intent, descriptors, commit_fd, source_pin, destination_pin, paths = fixture
    real_mask = lane.signal.pthread_sigmask
    restore_calls = 0

    def mask(how: int, signals: object) -> set[signal.Signals]:
        nonlocal restore_calls
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("pending SIGINT at restore")
        return real_mask(how, signals)

    monkeypatch.setattr(lane.signal, "pthread_sigmask", mask)
    try:
        with pytest.raises(BaseException):
            lane.publish_transaction_from_held_fds(
                intent=intent,
                member_descriptors=descriptors,
                commit_descriptor=commit_fd,
                commit_source_pin=source_pin,
                commit_destination_pin=destination_pin,
                recovery=False,
            )
        assert restore_calls >= 2
        assert not paths["commit"].exists()
        assert all(not paths[role].exists() for role in descriptors)
    finally:
        monkeypatch.setattr(lane.signal, "pthread_sigmask", real_mask)
        _close_transaction_fixture(run_directories, intent, descriptors, commit_fd)


def test_worker_descriptor_closes_when_lease_digest_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    original_open = lane.r14e.open_held_source
    opened: list[int] = []

    def recording_open(*args: object, **kwargs: object) -> tuple[int, dict[str, Any]]:
        descriptor, pin = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor, pin

    monkeypatch.setattr(lane.r14e, "open_held_source", recording_open)
    monkeypatch.setattr(
        lane.r14e,
        "lease_command_digests",
        lambda _argv: (_ for _ in ()).throw(lane.TensorRTR14FError("lease digest failure")),
    )
    with pytest.raises(lane.TensorRTR14FError, match="lease digest failure"):
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=repo_tmp_path / "missing-run",
            validate=lambda *_args: pytest.fail("validate must not run"),
        )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_worker_descriptor_closes_when_run_directory_open_fails(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    original_open = lane.r14e.open_held_source
    opened: list[int] = []

    def recording_open(*args: object, **kwargs: object) -> tuple[int, dict[str, Any]]:
        descriptor, pin = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor, pin

    def fail_open(cls: object, _path: Path) -> object:
        del cls
        raise lane.TensorRTR14FError("run directory failure")

    monkeypatch.setattr(lane.r14e, "open_held_source", recording_open)
    monkeypatch.setattr(
        lane.r14e,
        "lease_command_digests",
        lambda _argv: {"requested": "a" * 64, "effective": "a" * 64},
    )
    monkeypatch.setattr(
        lane.r14e.HeldRunDirectories,
        "open_existing",
        classmethod(fail_open),
    )
    with pytest.raises(lane.TensorRTR14FError, match="run directory failure"):
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=repo_tmp_path / "missing-run",
            validate=lambda *_args: pytest.fail("validate must not run"),
        )
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
