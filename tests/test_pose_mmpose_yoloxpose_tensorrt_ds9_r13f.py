from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as r13c
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13d as r13d
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13e as r13e
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13f as lane


STAMP = "2026-07-18T18:00:00+03:00"
STAGE = "tensorrt_fp16_640"
RUN_ID = "pose-r13f-baseexception-pair-unit"


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13f-test-", dir=lane.ROOT)
    try:
        yield Path(context.name)
    finally:
        context.cleanup()


class FakeTree:
    def __init__(self) -> None:
        self.closed = False

    def assert_bound(self) -> None:
        assert self.closed is False

    def close(self) -> None:
        self.closed = True


def _valid_plan(path: Path) -> dict[str, Any]:
    return r13c.prepare_plan(path, prepared_at_utc=STAMP)


def _held_stage(plan_path: Path, plan: dict[str, Any]) -> lane.HeldVerifiedStage:
    held = r13c.verify_plan_held(
        plan_path, expected_fingerprint=str(plan["fingerprint_sha256"]),
    )
    return lane.HeldVerifiedStage(
        held_plan=held,
        plan=copy.deepcopy(held.value),
        manifest={"plan": held.document_pin},
        predecessors=[],
        stream_descriptor=-1,
        stream_pin={},
        tree=FakeTree(),
    )


def _write_json(path: Path, value: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    path.chmod(0o440)
    return r13c.open_held_source(path)


def _canonical_documents(
    repo_tmp_path: Path,
    verified: lane.HeldVerifiedStage,
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    run_root = r13c.RUNS_ROOT / STAGE / RUN_ID
    manifest = {
        "path": r13c.repo_relative(run_root / "snapshot/manifest-r13c.json"),
        "bytes": 7,
        "sha256": "a" * 64,
        "fingerprint_sha256": "b" * 64,
    }
    stream = {
        "path": r13c.repo_relative(run_root / "snapshot/snapshot-r13c.stream"),
        "bytes": 9,
        "sha256": "c" * 64,
    }
    worker = {
        "path": r13c.repo_relative(run_root / "output/worker-result.json"),
        "bytes": 11,
        "sha256": "d" * 64,
        "fingerprint_sha256": "e" * 64,
    }
    primary = {
        "path": r13c.repo_relative(r13c.primary_path(STAGE)),
        "bytes": 13,
        "sha256": "f" * 64,
    }
    lifecycle = {
        "start_new_session": True,
        "timeout_seconds": r13c.LAUNCHER_TIMEOUT[STAGE],
        "duration_seconds": 1.0,
        "stdin_held_snapshot_fd": True,
        "output_limit_bytes": r13c.MAX_HOST_LOG_BYTES,
        "output_bytes": 17,
        "orphan_survived": False,
    }
    receipt = lane.build_receipt(
        plan=verified.plan,
        plan_pin=verified.plan_pin,
        manifest_pin=manifest,
        stream_pin=stream,
        stage=STAGE,
        run_id=RUN_ID,
        worker_pin=worker,
        worker_outputs=[primary],
        primary_published=primary,
        predecessors=[],
        numerical=None,
        lifecycle=lifecycle,
        created_at_utc=STAMP,
    )
    receipt_descriptor, receipt_source = _write_json(
        repo_tmp_path / "control/execution-receipt-r13c.json", receipt,
    )
    receipt_published = {
        "path": r13c.repo_relative(r13c.receipt_path(STAGE)),
        "bytes": receipt_source["bytes"],
        "sha256": receipt_source["sha256"],
        "fingerprint_sha256": receipt["fingerprint_sha256"],
    }
    commit = lane.build_commit(
        plan=verified.plan,
        plan_pin=verified.plan_pin,
        manifest_pin=manifest,
        stage=STAGE,
        run_id=RUN_ID,
        primary_pin=primary,
        receipt_pin=receipt_published,
        worker_pin=worker,
        predecessors=[],
        committed_at_utc=STAMP,
    )
    commit_descriptor, commit_source = _write_json(
        repo_tmp_path / "control/commit-r13c.json", commit,
    )
    commit_published = {
        "path": r13c.repo_relative(r13c.commit_path(STAGE)),
        "bytes": commit_source["bytes"],
        "sha256": commit_source["sha256"],
        "fingerprint_sha256": commit["fingerprint_sha256"],
    }
    return receipt_descriptor, commit_descriptor, receipt_published, commit_published


def _publication_fixture(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> tuple[lane.HeldVerifiedStage, int, int, dict[str, Any], dict[str, Any]]:
    plan_path = repo_tmp_path / "accepted-plan-r13c.json"
    plan = _valid_plan(plan_path)
    verified = _held_stage(plan_path, plan)
    results_root = repo_tmp_path / "canonical-results"
    monkeypatch.setattr(r13c, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(r13c, "RUNS_ROOT", results_root / "runs")
    receipt_fd, commit_fd, receipt_pin, commit_pin = _canonical_documents(
        repo_tmp_path, verified,
    )
    return verified, receipt_fd, commit_fd, receipt_pin, commit_pin


def _close_fixture(
    verified: lane.HeldVerifiedStage,
    receipt_fd: int,
    commit_fd: int,
) -> None:
    os.close(receipt_fd)
    os.close(commit_fd)
    verified.close()


def _assert_pair_absent() -> None:
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


def _assert_lifetime_closed(lifetime: lane.ReceiptCommitPairLifetime) -> None:
    assert lifetime.closed is True
    assert lifetime.rollback_handles == []
    assert lifetime.residual_descriptors == []
    for item in (lifetime.receipt, lifetime.commit):
        assert item.target_descriptor == -1
        assert item.chain is None


def test_frozen_r13c_r13d_r13e_ancestry_is_exact() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed == {
        "r13c": lane.FROZEN_R13C_LINEAGE,
        "r13d": lane.FROZEN_R13D_LINEAGE,
        "r13e": lane.FROZEN_R13E_LINEAGE,
    }
    assert len(observed["r13c"]) == 11
    assert len(observed["r13d"]) == 4
    assert len(observed["r13e"]) == 4
    assert r13d.FROZEN_R13C_LINEAGE == lane.FROZEN_R13C_LINEAGE
    assert r13e.FROZEN_R13D_LINEAGE == lane.FROZEN_R13D_LINEAGE


def test_controller_schema_const_rejects_forged_11_entry_r13c_lineage() -> None:
    schema = json.loads(lane.CONTROLLER_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    Draft202012Validator(schema).validate(value)
    assert value["accepted_lineage"]["r13c"] == lane.FROZEN_R13C_LINEAGE
    forged = copy.deepcopy(value)
    forged["accepted_lineage"]["r13c"] = {
        f"validation/forged-r13c-{index}.py": {
            "mode": "0664", "bytes": 1, "sha256": "0" * 64,
        }
        for index in range(11)
    }
    forged["fingerprint_sha256"] = r13c.fingerprint(forged)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(forged)


def test_runtime_rejects_changed_imported_r13c_lineage_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = copy.deepcopy(lane.FROZEN_R13C_LINEAGE)
    first = next(iter(forged))
    forged[first] = {**forged[first], "sha256": "0" * 64}
    monkeypatch.setattr(r13d, "FROZEN_R13C_LINEAGE", forged)
    with pytest.raises(r13c.PoseR13CError, match="runtime R13C lineage constant differs"):
        lane.verify_frozen_ancestry()


def test_positive_pair_publishes_exact_canonical_receipt_and_commit(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    try:
        result = lane.publish_receipt_and_commit(
            verified=verified,
            receipt_descriptor=receipt_fd,
            commit_descriptor=commit_fd,
            receipt_published=receipt_pin,
            commit_published=commit_pin,
        )
        assert result["receipt"] == {
            key: receipt_pin[key] for key in ("path", "bytes", "sha256")
        }
        assert result["commit"] == {
            key: commit_pin[key] for key in ("path", "bytes", "sha256")
        }
        assert r13c.receipt_path(STAGE).is_file()
        assert r13c.commit_path(STAGE).is_file()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_commit_failure_then_pending_keyboardinterrupt_restores_only_after_rollback_close(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []

    def fail_commit(self: lane.ReceiptCommitPairLifetime) -> dict[str, Any]:
        captured.append(self)
        self.assert_receipt_continuity(boundary="before_injected_commit_failure")
        raise r13c.PoseR13CError("injected commit failure")

    real_mask = signal.pthread_sigmask
    restore_calls = 0

    def pending_interrupt(how: int, mask: object) -> set[signal.Signals]:
        nonlocal restore_calls
        if how == signal.SIG_SETMASK:
            assert captured
            _assert_pair_absent()
            _assert_lifetime_closed(captured[0])
        result = real_mask(how, mask)
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("pending SIGINT after commit failure")
        return result

    monkeypatch.setattr(lane.ReceiptCommitPairLifetime, "link_commit", fail_commit)
    monkeypatch.setattr(lane.signal, "pthread_sigmask", pending_interrupt)
    try:
        with pytest.raises(BaseExceptionGroup, match="publication/cleanup failures"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert restore_calls == 2
        _assert_pair_absent()
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
    finally:
        monkeypatch.setattr(lane.signal, "pthread_sigmask", real_mask)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_success_path_restore_keyboardinterrupt_reblocks_rolls_back_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    real_mask = signal.pthread_sigmask
    restore_calls = 0
    block_calls = 0
    captured: list[lane.ReceiptCommitPairLifetime] = []
    real_success_barrier = lane.ReceiptCommitPairLifetime.assert_immediately_before_success

    def capture_success_barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        real_success_barrier(self)
        captured.append(self)

    def pending_interrupt(how: int, mask: object) -> set[signal.Signals]:
        nonlocal restore_calls, block_calls
        if how == signal.SIG_SETMASK and restore_calls == 0:
            assert len(captured) == 1
            for item in (captured[0].receipt, captured[0].commit):
                assert item.target_descriptor >= 0
                assert item.chain is not None
            assert r13c.receipt_path(STAGE).is_file()
            assert r13c.commit_path(STAGE).is_file()
        result = real_mask(how, mask)
        if how == signal.SIG_BLOCK:
            block_calls += 1
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("pending SIGINT on success restore")
        return result

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_immediately_before_success",
        capture_success_barrier,
    )
    monkeypatch.setattr(lane.signal, "pthread_sigmask", pending_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="success restore"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert restore_calls == 2
        assert block_calls == 2
        _assert_pair_absent()
    finally:
        monkeypatch.setattr(lane.signal, "pthread_sigmask", real_mask)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_abort_retries_canonical_unlink_after_rollback_keyboardinterrupt(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    real_rollback = lane._PairMember.rollback
    interrupted = False

    def fail_commit(self: lane.ReceiptCommitPairLifetime) -> dict[str, Any]:
        captured.append(self)
        raise r13c.PoseR13CError("injected commit failure")

    def interrupt_receipt_rollback(self: lane._PairMember) -> None:
        nonlocal interrupted
        if self.destination == r13c.receipt_path(STAGE) and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected rollback SIGINT")
        real_rollback(self)

    monkeypatch.setattr(lane.ReceiptCommitPairLifetime, "link_commit", fail_commit)
    monkeypatch.setattr(lane._PairMember, "rollback", interrupt_receipt_rollback)
    try:
        with pytest.raises(BaseExceptionGroup, match="publication/cleanup failures"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert interrupted is True
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
        _assert_pair_absent()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_chain_close_keyboardinterrupt_attempts_every_fd_and_rolls_back_pair(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    watched: set[int] = set()
    real_close = os.close
    interrupted = False

    def capture_close_barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)
        assert self.receipt.chain is not None
        watched.update(self.receipt.chain.descriptors)

    def interrupt_one_chain_close(descriptor: int) -> None:
        nonlocal interrupted
        if descriptor in watched and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected chain close SIGINT")
        real_close(descriptor)

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "_close_barrier",
        capture_close_barrier,
    )
    monkeypatch.setattr(lane.os, "close", interrupt_one_chain_close)
    try:
        with pytest.raises(KeyboardInterrupt, match="chain close SIGINT"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert interrupted is True
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
        _assert_pair_absent()
        for descriptor in watched:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        monkeypatch.setattr(lane.os, "close", real_close)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_unexpected_finalize_baseexception_uses_persistent_rollback_handles(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    real_close_chain = lane.ReceiptCommitPairLifetime._close_chain_total
    interrupted = False

    def capture_close_barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)

    def interrupt_finalize(
        self: lane.ReceiptCommitPairLifetime,
        item: Any,
    ) -> list[BaseException]:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("unexpected finalize SIGINT")
        return real_close_chain(self, item)

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "_close_barrier",
        capture_close_barrier,
    )
    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "_close_chain_total",
        interrupt_finalize,
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="unexpected finalize SIGINT"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert interrupted is True
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
        _assert_pair_absent()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


@pytest.mark.parametrize("failure", [KeyboardInterrupt("close SIGINT"), SystemExit("close exit")])
def test_close_barrier_baseexception_rolls_back_pair_and_attempts_all_resources(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
    failure: BaseException,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []

    def fail_before_close(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)
        raise failure

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "_close_barrier",
        fail_before_close,
    )
    try:
        with pytest.raises(type(failure)):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        _assert_pair_absent()
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_rollback_attempts_receipt_after_commit_rollback_keyboardinterrupt() -> None:
    calls: list[str] = []

    class Member:
        linked = True
        target_identity = (1, 1)
        target_descriptor = -1
        chain = None

        def __init__(self, name: str, fail: bool) -> None:
            self.name = name
            self.fail = fail

        def rollback(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise KeyboardInterrupt(self.name)

    lifetime = lane.ReceiptCommitPairLifetime(
        stage=STAGE,
        verified=object(),
        receipt=Member("receipt", False),
        commit=Member("commit", True),
    )
    with pytest.raises(KeyboardInterrupt, match="commit"):
        lifetime.rollback_all()
    assert calls == ["commit", "receipt"]


def test_receipt_parent_swap_still_rolls_back_both_canonical_links(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    receipt_parent = r13c.receipt_path(STAGE).parent
    displaced = receipt_parent.with_name(receipt_parent.name + ".displaced")
    original = lane.ReceiptCommitPairLifetime.assert_receipt_continuity
    injected = False

    def assert_then_swap(
        self: lane.ReceiptCommitPairLifetime, *, boundary: str,
    ) -> dict[str, Any]:
        nonlocal injected
        result = original(self, boundary=boundary)
        if boundary == "after_commit_link" and not injected:
            os.replace(receipt_parent, displaced)
            receipt_parent.mkdir(mode=0o755)
            injected = True
        return result

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_receipt_continuity",
        assert_then_swap,
    )
    try:
        with pytest.raises(BaseException, match="directory-chain name/inode binding differs"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
        assert injected is True
        _assert_pair_absent()
        assert not (displaced / r13c.receipt_path(STAGE).name).exists()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_execution_gate_fails_before_stage_run_or_external_process(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan.json"
    plan = _valid_plan(plan_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed R13F gate reached run creation or external process")

    monkeypatch.setattr(r13c, "_build_run_tree", forbidden)
    monkeypatch.setattr(r13c.subprocess, "Popen", forbidden)
    with pytest.raises(r13c.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
            stage=STAGE,
            run_id="pose-r13f-gate-unit",
            accepted_manifest_fingerprint="1" * 64,
            accepted_commits={},
        )
    assert not lane.DEFAULT_CONTROLLER.exists()
    assert not r13c.DEFAULT_PLAN.exists()
    assert not r13c.RUNS_ROOT.exists()


def test_import_and_contract_build_launch_no_gpu_docker_or_nvidia_process() -> None:
    code = r'''
import builtins, subprocess
real_import = builtins.__import__
blocked = {"tensorrt", "cuda", "pynvml", "pycuda"}
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("GPU module import: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
def forbidden(*args, **kwargs):
    raise AssertionError("external process launch")
subprocess.Popen = forbidden
subprocess.run = forbidden
subprocess.check_output = forbidden
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13f as lane
value = lane.build_controller_contract(prepared_at_utc="2026-07-18T18:00:00+03:00")
assert value["execution_gate"]["authorized"] is False
assert value["accepted_lineage"]["r13c"] == lane.FROZEN_R13C_LINEAGE
print(value["fingerprint_sha256"])
'''
    completed = subprocess.run(
        [os.sys.executable, "-c", code],
        cwd=lane.ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(completed.stdout.strip()) == 64
