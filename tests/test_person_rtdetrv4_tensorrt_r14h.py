from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator

from validation import person_rtdetrv4_tensorrt_r14h as lane


STAMP = "2026-07-18T18:30:00+03:00"
ROLES = ["artifact", "worker_result", "r11_stage_receipt", "execution_receipt"]
BOUNDARIES = [
    *(item for role in ROLES for item in (f"before_member_link:{role}", f"after_member_link:{role}")),
    "before_commit_link",
    "after_commit_link",
    "after_commit_all_fsync",
    "immediately_before_success",
]


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".person-r14h-test-", dir=lane.ROOT)
    try:
        yield Path(context.name)
    finally:
        context.cleanup()


def _fds() -> set[int]:
    observed: set[int] = set()
    for descriptor in range(3, 512):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        observed.add(descriptor)
    return observed


def _json_payload(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


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
    descriptor = os.memfd_create("person-r14h-test", os.MFD_CLOEXEC)
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
        "required": ["schema_version", "plan_id", "prepared_at_utc", "fingerprint_sha256"],
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


def _make_run(repo_tmp_path: Path) -> lane.HeldRunDirectories:
    run = repo_tmp_path / "runs/tensorrt_fp16_640/unit"
    output = run / "output"
    control = run / "control"
    output.mkdir(parents=True, mode=0o700)
    control.mkdir(mode=0o700)
    for path in (run, output, control):
        path.chmod(0o700)
    return lane.HeldRunDirectories.open_existing(run)


def _transaction_fixture(
    repo_tmp_path: Path,
) -> tuple[
    lane.HeldRunDirectories,
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
    descriptors: dict[str, int] = {}
    destinations: dict[str, Path] = {}
    members: list[dict[str, Any]] = []
    for index, role in enumerate(ROLES):
        payload = f"{role}-source-{index}\n".encode()
        descriptor = _memfd(payload)
        descriptors[role] = descriptor
        destination = published_root / f"{role}.bin"
        destinations[role] = destination
        members.append(
            {
                "role": role,
                "source": _pin(repo_tmp_path / "logical-source" / f"{role}.bin", payload),
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
    source_pin = _pin(
        repo_tmp_path / "logical-source/commit.json",
        commit_payload,
        fingerprint=commit_value["fingerprint_sha256"],
    )
    destination_pin = _pin(
        commit_destination,
        commit_payload,
        fingerprint=commit_value["fingerprint_sha256"],
    )
    return (
        run_directories,
        intent,
        descriptors,
        commit_descriptor,
        source_pin,
        destination_pin,
        destinations,
    )


def _close_transaction_fixture(
    run_directories: lane.HeldRunDirectories,
    intent: lane.HeldPublicationIntent,
    descriptors: dict[str, int],
    commit_descriptor: int,
) -> None:
    cleanup: list[BaseException] = []
    for descriptor in descriptors.values():
        lane._close_descriptor(descriptor, cleanup)
    lane._close_descriptor(commit_descriptor, cleanup)
    intent.close_collect(cleanup)
    run_directories.close_collect(cleanup)
    lane._raise_failures(None, cleanup, message="test fixture cleanup failures")


def _publish(
    fixture: tuple[
        lane.HeldRunDirectories,
        lane.HeldPublicationIntent,
        dict[str, int],
        int,
        dict[str, Any],
        dict[str, Any],
        dict[str, Path],
    ],
    *,
    recovery: bool,
    hook=None,
) -> dict[str, Any]:
    _runs, intent, descriptors, commit_fd, source_pin, destination_pin, _paths = fixture
    return lane.publish_transaction_from_held_fds(
        intent=intent,
        member_descriptors=descriptors,
        commit_descriptor=commit_fd,
        commit_source_pin=source_pin,
        commit_destination_pin=destination_pin,
        recovery=recovery,
        boundary_hook=hook,
    )


def test_frozen_r14d_r14e_r14f_r14g_inventory_is_exact_and_r14h_outputs_absent() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed == {
        "r14d": lane.FROZEN_R14D_LINEAGE,
        "r14e": lane.FROZEN_R14E_LINEAGE,
        "r14f": lane.FROZEN_R14F_LINEAGE,
        "r14g": lane.FROZEN_R14G_LINEAGE,
    }
    assert [len(observed[key]) for key in ("r14d", "r14e", "r14f", "r14g")] == [11, 11, 4, 4]
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.DEFAULT_CONTROLLER.exists()
    assert not lane.RUNS_ROOT.exists()


def test_controller_contract_exact_schema_and_closed_claim_boundary() -> None:
    schema = json.loads(lane.CONTROLLER_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    Draft202012Validator(schema).validate(value)
    assert value["execution_gate"]["authorized"] is False
    assert value["execution_gate"]["gpu_lease_v3_audited"] is False
    assert value["claim_boundary"]["production_ready"] is False
    assert value["claim_boundary"]["gpu_executed"] is False
    assert value["claim_boundary"]["r14g_lineage_immutable"] is True
    assert set(value["ownership"].values()) == {True}
    forged = copy.deepcopy(value)
    forged["accepted_lineage"]["r14f"][
        "validation/person_rtdetrv4_tensorrt_r14f.py"
    ]["sha256"] = "0" * 64
    forged["fingerprint_sha256"] = lane.r14e.fingerprint(forged)
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(forged)


def test_plan_schema_is_held_before_plan_and_exact_projection(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    schema_path, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    original = lane.HeldCanonicalFile.open_existing.__func__
    opened: list[Path] = []
    schema_objects: list[dict[str, Any]] = []

    def recording(cls: object, path: Path, **kwargs: object):
        opened.append(Path(path))
        return original(cls, path, **kwargs)

    def validate(value: dict[str, Any], schema: dict[str, Any]) -> None:
        assert value == plan
        schema_objects.append(schema)

    monkeypatch.setattr(lane.HeldCanonicalFile, "open_existing", classmethod(recording))
    monkeypatch.setattr(lane.r14f, "_validate_schema_object", validate)
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_a, **_k: copy.deepcopy(plan))
    held = lane.verify_plan_held(plan_path, expected_fingerprint=plan["fingerprint_sha256"])
    try:
        assert opened[:2] == [schema_path, plan_path]
        assert schema_objects == [_minimal_schema()]
        held.assert_stable()
    finally:
        held.close()


def test_noncanonical_plan_fails_before_any_held_open(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, _plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    noncanonical = repo_tmp_path / "plan/copy.json"
    _write_immutable(noncanonical, _json_payload(plan))
    monkeypatch.setattr(
        lane.HeldCanonicalFile,
        "open_existing",
        lambda *_a, **_k: pytest.fail("held open must not run"),
    )
    with pytest.raises(lane.TensorRTR14HError, match="canonical DEFAULT_PLAN"):
        lane.verify_plan_held(
            noncanonical, expected_fingerprint=plan["fingerprint_sha256"],
        )


def test_schema_a_b_a_swap_is_detected_with_held_parent_version(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    schema_path, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)

    def swap_and_restore(*_args: object, **_kwargs: object) -> dict[str, Any]:
        backup = schema_path.with_name("plan.schema.original")
        schema_path.rename(backup)
        schema_path.write_bytes(_json_payload({"type": "object"}))
        schema_path.chmod(0o440)
        schema_path.unlink()
        backup.rename(schema_path)
        return copy.deepcopy(plan)

    monkeypatch.setattr(lane.r14e, "verify_plan", swap_and_restore)
    with pytest.raises(
        lane.TensorRTR14HError,
        match="held inode metadata differs|parent namespace changed",
    ):
        lane.verify_plan_held(
            plan_path, expected_fingerprint=plan["fingerprint_sha256"],
        )


def test_exact_r14e_projection_drift_closes_all_plan_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    drift = copy.deepcopy(plan)
    drift["plan_id"] = "drift"
    drift["fingerprint_sha256"] = lane.r14e.fingerprint(drift)
    before = _fds()
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_a, **_k: drift)
    with pytest.raises(lane.TensorRTR14HError, match="projection drifted"):
        lane.verify_plan_held(plan_path, expected_fingerprint=plan["fingerprint_sha256"])
    assert _fds() == before


def test_held_plan_close_baseexception_closes_every_owned_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _schema, plan_path, plan = _plan_fixture(monkeypatch, repo_tmp_path)
    monkeypatch.setattr(lane.r14e, "verify_plan", lambda *_a, **_k: copy.deepcopy(plan))
    before = _fds()
    held = lane.verify_plan_held(plan_path, expected_fingerprint=plan["fingerprint_sha256"])
    target = held.plan_source.descriptor
    real_close = lane.os.close
    fired = False

    def flaky_close(descriptor: int) -> None:
        nonlocal fired
        if descriptor == target and not fired:
            fired = True
            real_close(descriptor)
            raise KeyboardInterrupt("synthetic held-plan close signal")
        real_close(descriptor)

    monkeypatch.setattr(lane.os, "close", flaky_close)
    with pytest.raises(BaseException):
        held.close()
    assert fired is True
    assert _fds() == before


def test_execute_gate_precedes_plan_open_run_creation_and_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane.HeldCanonicalFile,
        "open_existing",
        lambda *_a, **_k: pytest.fail("closed gate must precede plan open"),
    )
    monkeypatch.setattr(
        lane.r14e.HeldRunDirectories,
        "create",
        classmethod(lambda *_a, **_k: pytest.fail("run creation must not run")),
    )
    monkeypatch.setattr(
        lane.r14e.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("Popen must not run"),
    )
    with pytest.raises(lane.TensorRTR14HError, match="execution gate is closed"):
        lane.execute_stage(
            plan_path=lane.DEFAULT_PLAN,
            accepted_plan_fingerprint="0" * 64,
            stage="tensorrt_fp16_640",
            run_id="r14h-gate",
        )
    assert not lane.DEFAULT_PLAN.exists()
    assert not lane.RUNS_ROOT.exists()


def test_r14h_source_exposes_no_external_process_or_accelerator_launcher() -> None:
    source = lane.THIS_FILE.read_text(encoding="utf-8")
    forbidden = (
        "import subprocess",
        "subprocess.Popen",
        "subprocess.run",
        "os.system",
        "docker.from_env",
        "nvidia-smi",
    )
    assert all(token not in source for token in forbidden)


def test_uncertain_close_does_not_consume_reused_numeric_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    victim = os.open("/dev/null", os.O_RDONLY)
    real_close = lane.os.close
    replacement: dict[str, int] = {}
    cleanup: list[BaseException] = []

    def close_then_reuse(descriptor: int) -> None:
        if descriptor == victim and "descriptor" not in replacement:
            real_close(descriptor)
            replacement["descriptor"] = os.open("/dev/null", os.O_RDONLY)
            raise KeyboardInterrupt("close completed after numeric slot release")
        real_close(descriptor)

    monkeypatch.setattr(lane.os, "close", close_then_reuse)
    assert lane._close_descriptor(victim, cleanup) is True
    reused = replacement["descriptor"]
    try:
        assert reused == victim
        os.fstat(reused)
        assert [type(item) for item in cleanup] == [KeyboardInterrupt]
    finally:
        real_close(reused)


def test_post_link_async_gap_removes_only_created_inode(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    destination = repo_tmp_path / "async-gap.bin"
    payload = b"R14H async ownership gap\n"
    source = _memfd(payload)
    expected = _pin(destination, payload)
    original = lane._link_tmpfile
    fired = False

    def link_then_interrupt(descriptor: int, parent: int, name: str) -> None:
        nonlocal fired
        original(descriptor, parent, name)
        fired = True
        raise KeyboardInterrupt("after linkat before caller bookkeeping")

    monkeypatch.setattr(lane, "_link_tmpfile", link_then_interrupt)
    try:
        with pytest.raises(BaseException):
            lane.HeldCanonicalFile.create_from_fd(
                destination,
                source,
                expected,
                allow_existing_exact=False,
            )
        assert fired is True
        assert not destination.exists()
    finally:
        os.close(source)


def test_post_link_failure_preserves_replacement_inode(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    destination = repo_tmp_path / "replacement.bin"
    payload = b"R14H owned inode\n"
    replacement_payload = b"independent replacement inode\n"
    source = _memfd(payload)
    expected = _pin(destination, payload)
    real_fsync = lane.os.fsync
    calls = 0
    replacement_identity: tuple[int, int] | None = None

    def replace_after_link(descriptor: int) -> None:
        nonlocal calls, replacement_identity
        calls += 1
        if calls == 2:
            destination.unlink()
            destination.write_bytes(replacement_payload)
            destination.chmod(0o440)
            metadata = destination.stat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            raise SystemExit("namespace replacement after link")
        real_fsync(descriptor)

    monkeypatch.setattr(lane.os, "fsync", replace_after_link)
    try:
        with pytest.raises(BaseException):
            lane.HeldCanonicalFile.create_from_fd(
                destination,
                source,
                expected,
                allow_existing_exact=False,
            )
        assert replacement_identity is not None
        assert destination.read_bytes() == replacement_payload
        metadata = destination.stat()
        assert (metadata.st_dev, metadata.st_ino) == replacement_identity
    finally:
        os.close(source)


@pytest.mark.parametrize("failure_name", ["output", "control"])
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_partial_run_open_baseexception_has_exact_fd_balance(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
    failure_name: str,
    failure_type: type[BaseException],
) -> None:
    run = _make_run(repo_tmp_path)
    run.close()
    before = _fds()
    real_open = lane.os.open
    fired = False

    def injected(path: object, *args: object, **kwargs: object) -> int:
        nonlocal fired
        if path == failure_name and not fired:
            fired = True
            raise failure_type(f"synthetic {failure_name} open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(lane.os, "open", injected)
    with pytest.raises(BaseException):
        lane.HeldRunDirectories.open_existing(run.run_root)
    assert fired is True
    assert _fds() == before


def test_repository_directory_walk_baseexception_has_exact_fd_balance(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    nested = repo_tmp_path / "walk-a/walk-b"
    nested.mkdir(parents=True)
    before = _fds()
    real_open = lane.os.open
    fired = False

    def injected(path: object, *args: object, **kwargs: object) -> int:
        nonlocal fired
        if path == "walk-b" and not fired:
            fired = True
            raise SystemExit("synthetic directory walk signal")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(lane.os, "open", injected)
    with pytest.raises(BaseException):
        lane._open_repo_directory(nested)
    assert fired is True
    assert _fds() == before


def test_worker_scope_run_close_baseexception_still_closes_worker(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    run = _make_run(repo_tmp_path)
    run_root = run.run_root
    run.close()
    before = _fds()
    target: dict[str, int] = {}
    real_close = lane.os.close
    fired = False

    def flaky_close(descriptor: int) -> None:
        nonlocal fired
        if descriptor == target.get("run") and not fired:
            fired = True
            real_close(descriptor)
            raise KeyboardInterrupt("synthetic run close signal")
        real_close(descriptor)

    def validate(_fd: int, _pin: Mapping[str, Any], _digests: Mapping[str, str], held_run):
        target["run"] = held_run.run_descriptor
        return {"ok": True}

    monkeypatch.setattr(lane.r14e, "lease_command_digests", lambda _a: {"requested": "a", "effective": "b"})
    monkeypatch.setattr(lane.os, "close", flaky_close)
    with pytest.raises(BaseException):
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=run_root,
            validate=validate,
        )
    assert fired is True
    assert _fds() == before


def test_worker_scope_output_close_baseexception_closes_all_other_owners(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    run = _make_run(repo_tmp_path)
    run_root = run.run_root
    run.close()
    before = _fds()
    target: dict[str, int] = {}
    real_close = lane.os.close
    fired = False

    def flaky_close(descriptor: int) -> None:
        nonlocal fired
        if descriptor == target.get("output") and not fired:
            fired = True
            real_close(descriptor)
            raise SystemExit("synthetic output close signal")
        real_close(descriptor)

    def validate(_fd: int, _pin: Mapping[str, Any], _digests: Mapping[str, str], _run):
        first = _memfd(b"first\n")
        second = _memfd(b"second\n")
        target["output"] = first
        return {"ok": True}, {"first": first, "second": second}

    monkeypatch.setattr(lane.r14e, "lease_command_digests", lambda _a: {"requested": "a", "effective": "b"})
    monkeypatch.setattr(lane.os, "close", flaky_close)
    with pytest.raises(BaseException):
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=run_root,
            validate=validate,
        )
    assert fired is True
    assert _fds() == before


def test_worker_scope_preserves_primary_and_cleanup_baseexceptions(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    run = _make_run(repo_tmp_path)
    run_root = run.run_root
    run.close()
    before = _fds()
    target: dict[str, int] = {}
    real_close = lane.os.close

    def flaky_close(descriptor: int) -> None:
        if descriptor == target.get("run"):
            real_close(descriptor)
            target["run"] = -1
            raise KeyboardInterrupt("cleanup signal")
        real_close(descriptor)

    def validate(_fd: int, _pin: Mapping[str, Any], _digests: Mapping[str, str], held_run):
        target["run"] = held_run.run_descriptor
        raise ValueError("primary validation failure")

    monkeypatch.setattr(lane.r14e, "lease_command_digests", lambda _a: {"requested": "a", "effective": "b"})
    monkeypatch.setattr(lane.os, "close", flaky_close)
    with pytest.raises(BaseExceptionGroup) as raised:
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=run_root,
            validate=validate,
        )
    flattened = repr(raised.value)
    assert "primary validation failure" in str(raised.value.exceptions[0])
    assert "cleanup signal" in flattened or any(
        isinstance(item, KeyboardInterrupt) for item in raised.value.exceptions
    )
    assert _fds() == before


def test_worker_scope_reclaims_predecessor_borrowed_run_file_on_baseexception(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    worker_path = repo_tmp_path / "worker.json"
    worker_pin = _write_immutable(worker_path, b"{}\n")
    run = _make_run(repo_tmp_path)
    _write_immutable(run.output / "borrowed.bin", b"borrowed\n")
    run_root = run.run_root
    run.close()
    before = _fds()

    def validate(_fd: int, _pin: Mapping[str, Any], _digests: Mapping[str, str], held_run):
        held_run.open_output("borrowed.bin")
        raise KeyboardInterrupt("frozen predecessor abandoned borrowed descriptor")

    monkeypatch.setattr(lane.r14e, "lease_command_digests", lambda _a: {"requested": "a", "effective": "b"})
    with pytest.raises(BaseException):
        lane._verify_worker_receipt_scope(
            worker_path=worker_path,
            expected_worker_pin=worker_pin,
            docker_argv=["docker", "run"],
            run_root=run_root,
            validate=validate,
        )
    assert _fds() == before


def test_success_path_cleanup_baseexception_rolls_back_every_created_name(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    before_transaction = _fds()
    original = lane.HeldCanonicalFile.close_collect
    fired = False

    def injected(self: lane.HeldCanonicalFile, cleanup: list[BaseException]) -> None:
        nonlocal fired
        original(self, cleanup)
        if self.path == paths["commit"] and not fired:
            fired = True
            cleanup.append(KeyboardInterrupt("synthetic post-commit cleanup signal"))

    monkeypatch.setattr(lane.HeldCanonicalFile, "close_collect", injected)
    try:
        with pytest.raises(BaseException):
            _publish(fixture, recovery=False)
        assert fired is True
        assert all(not path.exists() for path in paths.values())
        assert not intent.source.path.exists()
        assert _fds() == before_transaction
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_rollback_retries_one_baseexception_and_removes_total_created_set(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    original_close = lane.HeldCanonicalFile.close_collect
    original_rollback = lane.HeldCanonicalFile.force_created_absent
    cleanup_fired = False
    rollback_fired = False

    def fail_success_cleanup(
        self: lane.HeldCanonicalFile,
        cleanup: list[BaseException],
    ) -> None:
        nonlocal cleanup_fired
        original_close(self, cleanup)
        if self.path == paths["commit"] and not cleanup_fired:
            cleanup_fired = True
            cleanup.append(RuntimeError("pre-return cleanup failure"))

    def interrupt_first_commit_rollback(self: lane.HeldCanonicalFile) -> None:
        nonlocal rollback_fired
        if self.path == paths["commit"] and not rollback_fired:
            rollback_fired = True
            raise KeyboardInterrupt("first commit rollback interrupted")
        original_rollback(self)

    monkeypatch.setattr(lane.HeldCanonicalFile, "close_collect", fail_success_cleanup)
    monkeypatch.setattr(
        lane.HeldCanonicalFile,
        "force_created_absent",
        interrupt_first_commit_rollback,
    )
    try:
        with pytest.raises(BaseExceptionGroup) as raised:
            _publish(fixture, recovery=False)
        assert cleanup_fired is True
        assert rollback_fired is True
        assert "pre-return cleanup failure" in repr(raised.value)
        assert "first commit rollback interrupted" in repr(raised.value)
        assert all(not path.exists() for path in paths.values())
        assert not intent.source.path.exists()
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_transaction_entry_baseexception_rolls_back_created_intent(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    original = lane.HeldPublicationIntent.assert_canonical

    def injected(self: lane.HeldPublicationIntent, *, boundary: str):
        if boundary == "transaction_entry":
            raise KeyboardInterrupt("synthetic transaction-entry signal")
        return original(self, boundary=boundary)

    monkeypatch.setattr(lane.HeldPublicationIntent, "assert_canonical", injected)
    try:
        with pytest.raises(BaseException):
            _publish(fixture, recovery=False)
        assert all(not path.exists() for path in paths.values())
        assert not intent.source.path.exists()
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_signal_restore_baseexception_rolls_back_before_error_return(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    before_transaction = _fds()
    real_mask = lane.signal.pthread_sigmask
    restore_calls = 0

    def mask(how: int, signals: object):
        nonlocal restore_calls
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("pending signal at restore")
        return real_mask(how, signals)

    monkeypatch.setattr(lane.signal, "pthread_sigmask", mask)
    try:
        with pytest.raises(BaseException):
            _publish(fixture, recovery=False)
        assert restore_calls >= 2
        assert all(not path.exists() for path in paths.values())
        assert not intent.source.path.exists()
        assert _fds() == before_transaction
    finally:
        monkeypatch.setattr(lane.signal, "pthread_sigmask", real_mask)
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_execute_every_boundary_baseexception_rolls_back_created_names(
    repo_tmp_path: Path,
    boundary: str,
    failure_type: type[BaseException],
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    fired = False

    def hook(observed: str, _intent: lane.HeldPublicationIntent) -> None:
        nonlocal fired
        if observed == boundary and not fired:
            fired = True
            raise failure_type(f"synthetic boundary failure: {boundary}")

    try:
        with pytest.raises(BaseException):
            _publish(fixture, recovery=False, hook=hook)
        assert fired is True
        assert all(not path.exists() for path in paths.values())
        assert not intent.source.path.exists()
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


@pytest.mark.parametrize("boundary", BOUNDARIES)
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_recovery_every_boundary_preserves_preexisting_exact_names(
    repo_tmp_path: Path,
    boundary: str,
    failure_type: type[BaseException],
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    try:
        first = _publish(fixture, recovery=False)
        assert first["status"] == "committed"
        expected_intent = intent.document_pin
        intent.close()
        intent = lane.HeldPublicationIntent.open_existing(runs, expected_intent)
        fixture = (runs, intent, descriptors, commit_fd, fixture[4], fixture[5], paths)
        identities = {name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()}
        intent_identity = (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino)
        before_recovery = _fds()
        fired = False

        def hook(observed: str, _intent: lane.HeldPublicationIntent) -> None:
            nonlocal fired
            if observed == boundary and not fired:
                fired = True
                raise failure_type(f"synthetic recovery failure: {boundary}")

        with pytest.raises(BaseException):
            _publish(fixture, recovery=True, hook=hook)
        assert fired is True
        assert {
            name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()
        } == identities
        assert (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino) == intent_identity
        assert _fds() == before_recovery
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_recovery_success_records_preexisting_ownership_without_new_links(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, _paths = fixture
    try:
        first = _publish(fixture, recovery=False)
        assert set(first["transaction_start_preexisting"].values()) == {False}
        expected_intent = intent.document_pin
        intent.close()
        intent = lane.HeldPublicationIntent.open_existing(runs, expected_intent)
        fixture = (runs, intent, descriptors, commit_fd, fixture[4], fixture[5], fixture[6])
        second = _publish(fixture, recovery=True)
        assert second["status"] == "already_committed"
        assert set(second["transaction_start_preexisting"].values()) == {True}
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_recovery_cleanup_baseexception_preserves_all_preexisting_names(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    original = lane.HeldCanonicalFile.close_collect
    fired = False
    try:
        _publish(fixture, recovery=False)
        expected_intent = intent.document_pin
        intent.close()
        intent = lane.HeldPublicationIntent.open_existing(runs, expected_intent)
        fixture = (runs, intent, descriptors, commit_fd, fixture[4], fixture[5], paths)
        identities = {name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()}
        intent_identity = (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino)
        before_recovery = _fds()

        def injected(self: lane.HeldCanonicalFile, cleanup: list[BaseException]) -> None:
            nonlocal fired
            original(self, cleanup)
            if self.path == paths["commit"] and not fired:
                fired = True
                cleanup.append(KeyboardInterrupt("synthetic recovery cleanup signal"))

        monkeypatch.setattr(lane.HeldCanonicalFile, "close_collect", injected)
        with pytest.raises(BaseException):
            _publish(fixture, recovery=True)
        assert fired is True
        assert {
            name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()
        } == identities
        assert (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino) == intent_identity
        assert _fds() == before_recovery
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_recovery_same_intent_lifetime_is_preexisting_for_new_invocation(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    fired = False
    try:
        _publish(fixture, recovery=False)
        identities = {name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()}
        intent_identity = (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino)

        def hook(boundary: str, _intent: lane.HeldPublicationIntent) -> None:
            nonlocal fired
            if boundary == "before_member_link:artifact" and not fired:
                fired = True
                raise KeyboardInterrupt("same-lifetime recovery interruption")

        with pytest.raises(BaseException):
            _publish(fixture, recovery=True, hook=hook)
        assert fired is True
        assert {
            name: (path.stat().st_dev, path.stat().st_ino) for name, path in paths.items()
        } == identities
        assert (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino) == intent_identity
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)


def test_partial_recovery_rolls_back_only_member_created_by_that_invocation(
    repo_tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(repo_tmp_path)
    runs, intent, descriptors, commit_fd, _source, _destination, paths = fixture
    try:
        _publish(fixture, recovery=False)
        paths["artifact"].unlink()
        preserved = {
            name: (path.stat().st_dev, path.stat().st_ino)
            for name, path in paths.items()
            if name != "artifact"
        }
        intent_identity = (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino)

        def hook(boundary: str, _intent: lane.HeldPublicationIntent) -> None:
            if boundary == "after_member_link:artifact":
                raise SystemExit("interrupt partial recovery")

        with pytest.raises(BaseException):
            _publish(fixture, recovery=True, hook=hook)
        assert not paths["artifact"].exists()
        assert {
            name: (path.stat().st_dev, path.stat().st_ino)
            for name, path in paths.items()
            if name != "artifact"
        } == preserved
        assert (intent.source.path.stat().st_dev, intent.source.path.stat().st_ino) == intent_identity
    finally:
        _close_transaction_fixture(runs, intent, descriptors, commit_fd)
