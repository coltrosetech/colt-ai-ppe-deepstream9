from __future__ import annotations

import collections
import copy
import errno
import inspect
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tests import test_pose_mmpose_yoloxpose_tensorrt_ds9_r13g as r13g_tests
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as r13c
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13g as r13g
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13h as lane


STAMP = r13g_tests.STAMP
STAGE = r13g_tests.STAGE


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13h-test-", dir=lane.ROOT)
    try:
        yield Path(context.name)
    finally:
        context.cleanup()


def _publication_fixture(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> tuple[lane.HeldVerifiedStage, int, int, dict[str, Any], dict[str, Any]]:
    return r13g_tests._publication_fixture(monkeypatch, repo_tmp_path)


def _close_fixture(
    verified: lane.HeldVerifiedStage,
    receipt_fd: int,
    commit_fd: int,
) -> None:
    r13g_tests._close_fixture(verified, receipt_fd, commit_fd)


def _assert_pair_absent() -> None:
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


def _assert_lifetime_aborted(lifetime: lane.ReceiptCommitPairLifetime) -> None:
    assert lifetime.closed is True
    assert lifetime.rollback_handles == []
    assert lifetime.pending_close_descriptors == []
    assert len(lifetime.rollback_intents) == 2
    assert all(intent.activated and intent.completed for intent in lifetime.rollback_intents)
    for item in (lifetime.receipt, lifetime.commit):
        assert item.target_descriptor == -1
        assert item.chain is None
        assert item.linked is False


def _install_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[lane.ReceiptCommitPairLifetime]:
    captured: list[lane.ReceiptCommitPairLifetime] = []
    original = lane.ReceiptCommitPairLifetime._close_barrier

    def capture(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)
        original(self)

    monkeypatch.setattr(lane.ReceiptCommitPairLifetime, "_close_barrier", capture)
    return captured


def test_frozen_r13c_through_r13g_ancestry_and_schema_are_exact() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed == {
        "r13c": lane.FROZEN_R13C_LINEAGE,
        "r13d": lane.FROZEN_R13D_LINEAGE,
        "r13e": lane.FROZEN_R13E_LINEAGE,
        "r13f": lane.FROZEN_R13F_LINEAGE,
        "r13g": lane.FROZEN_R13G_LINEAGE,
    }
    assert {key: len(value) for key, value in observed.items()} == {
        "r13c": 11, "r13d": 4, "r13e": 4, "r13f": 4, "r13g": 4,
    }
    assert r13g.FROZEN_R13C_LINEAGE == lane.FROZEN_R13C_LINEAGE
    assert r13g.FROZEN_R13D_LINEAGE == lane.FROZEN_R13D_LINEAGE
    assert r13g.FROZEN_R13E_LINEAGE == lane.FROZEN_R13E_LINEAGE
    assert r13g.FROZEN_R13F_LINEAGE == lane.FROZEN_R13F_LINEAGE

    schema = json.loads(lane.CONTROLLER_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    Draft202012Validator(schema).validate(value)
    assert value["accepted_lineage"] == observed
    forged = copy.deepcopy(value)
    forged["accepted_lineage"]["r13g"] = {
        f"validation/forged-r13g-{index}.py": {
            "mode": "0440", "bytes": 1, "sha256": "0" * 64,
        }
        for index in range(4)
    }
    forged["fingerprint_sha256"] = r13c.fingerprint(forged)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(forged)


def test_positive_pair_consumes_every_owned_fd_once(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    owned: set[int] = set()
    calls: collections.Counter[int] = collections.Counter()
    real_close = os.close

    def barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)
        values = [self.receipt.target_descriptor, self.commit.target_descriptor]
        values.extend(
            descriptor
            for item in (self.receipt, self.commit)
            for descriptor in item.chain.descriptors
        )
        values.extend(descriptor for _item, descriptor in self.rollback_handles)
        assert all(descriptor >= 0 for descriptor in values)
        assert len(values) == len(set(values))
        owned.update(values)

    def counted_close(descriptor: int) -> None:
        if descriptor in owned:
            calls[descriptor] += 1
        real_close(descriptor)

    monkeypatch.setattr(lane.ReceiptCommitPairLifetime, "_close_barrier", barrier)
    monkeypatch.setattr(lane.os, "close", counted_close)
    try:
        result = lane.publish_receipt_and_commit(
            verified=verified,
            receipt_descriptor=receipt_fd,
            commit_descriptor=commit_fd,
            receipt_published=receipt_pin,
            commit_published=commit_pin,
        )
    finally:
        monkeypatch.setattr(lane.os, "close", real_close)
    try:
        assert result["receipt"]["sha256"] == receipt_pin["sha256"]
        assert result["commit"]["sha256"] == commit_pin["sha256"]
        assert len(captured) == 1 and captured[0].closed
        assert set(calls) == owned
        assert all(count == 1 for count in calls.values())
        assert captured[0].rollback_handles == []
        assert captured[0].pending_close_descriptors == []
        assert all(intent.activated for intent in captured[0].rollback_intents)
        assert not any(intent.completed for intent in captured[0].rollback_intents)
        assert r13c.receipt_path(STAGE).is_file()
        assert r13c.commit_path(STAGE).is_file()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_restore_to_close_keyboardinterrupt_uses_durable_abort(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured = _install_capture(monkeypatch)
    source, start = inspect.getsourcelines(lane._restore_and_close)
    close_line = start + next(
        index for index, text in enumerate(source) if "lifetime.close()" in text
    )

    def interrupt(frame: Any, event: str, _argument: Any) -> Any:
        if (
            frame.f_code is lane._restore_and_close.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            raise KeyboardInterrupt("R13H restore-to-close interrupt")
        return interrupt

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="restore-to-close"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        sys.settrace(None)
    try:
        assert len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_aborted(captured[0])
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


@pytest.mark.parametrize("category", ["chain", "rollback_parent"])
def test_chain_and_rollback_parent_close_reuse_are_single_attempt_and_abort(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
    category: str,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    victim = -1
    victim_calls = 0
    replacement = -1
    real_close = os.close

    def barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        nonlocal victim
        captured.append(self)
        chains = [
            descriptor
            for item in (self.receipt, self.commit)
            for descriptor in item.chain.descriptors
        ]
        parents = [descriptor for _item, descriptor in self.rollback_handles]
        victim = {"chain": chains, "rollback_parent": parents}[category][0]

    def close_reuse_raise(descriptor: int) -> None:
        nonlocal victim_calls, replacement
        if descriptor == victim:
            victim_calls += 1
            if victim_calls == 1:
                real_close(descriptor)
                temporary = os.open("/dev/null", os.O_RDONLY)
                if temporary != descriptor:
                    os.dup2(temporary, descriptor)
                    real_close(temporary)
                replacement = descriptor
                raise KeyboardInterrupt(f"R13H {category} close/reuse")
        real_close(descriptor)

    monkeypatch.setattr(lane.ReceiptCommitPairLifetime, "_close_barrier", barrier)
    monkeypatch.setattr(lane.os, "close", close_reuse_raise)
    try:
        with pytest.raises(KeyboardInterrupt, match=f"{category} close/reuse"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        monkeypatch.setattr(lane.os, "close", real_close)
    try:
        assert victim >= 0 and replacement == victim
        assert victim_calls == 1
        os.fstat(replacement)
        assert len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_aborted(captured[0])
        # Repeated aborts operate only on completed namespace intents.
        captured[0].abort_and_close()
        captured[0].abort_and_close()
        _assert_pair_absent()
    finally:
        if replacement >= 0:
            real_close(replacement)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_parent_ownership_detachment_interval_interrupt_still_aborts(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured = _install_capture(monkeypatch)
    source, start = inspect.getsourcelines(
        lane.ReceiptCommitPairLifetime._consume_rollback_handles
    )
    drain_line = start + next(
        index
        for index, text in enumerate(source)
        if "return self._drain_registered_descriptors()" in text
    )
    fired = False

    def interrupt(frame: Any, event: str, _argument: Any) -> Any:
        nonlocal fired
        if (
            not fired
            and frame.f_code
            is lane.ReceiptCommitPairLifetime._consume_rollback_handles.__code__
            and event == "line"
            and frame.f_lineno == drain_line
        ):
            fired = True
            raise KeyboardInterrupt("R13H parent-detachment interval")
        return interrupt

    sys.settrace(interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="parent-detachment"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        sys.settrace(None)
    try:
        assert fired and len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_aborted(captured[0])
        captured[0].abort_and_close()
        _assert_pair_absent()
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_post_syscall_reblock_failure_reports_actual_blocked_cleanup_mask(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured = _install_capture(monkeypatch)
    original_rollback = lane._PairMember.rollback
    real_mask = signal.pthread_sigmask
    caller_mask = real_mask(signal.SIG_BLOCK, set())
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    masks_seen: list[set[signal.Signals]] = []
    block_calls = 0
    restore_calls = 0

    def injected_mask(how: int, mask: object) -> set[signal.Signals]:
        nonlocal block_calls, restore_calls
        result = real_mask(how, mask)
        if how == signal.SIG_BLOCK:
            block_calls += 1
            if block_calls == 3:
                raise OSError(errno.EIO, "R13H post-syscall reblock failure")
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("R13H post-syscall restore interrupt")
        return result

    def observe_rollback(self: lane._PairMember) -> None:
        masks_seen.append(real_mask(signal.SIG_BLOCK, set()))
        original_rollback(self)

    monkeypatch.setattr(lane._PairMember, "rollback", observe_rollback)
    monkeypatch.setattr(lane.signal, "pthread_sigmask", injected_mask)
    try:
        with pytest.raises(BaseExceptionGroup, match="publication/cleanup failures"):
            lane.publish_receipt_and_commit(
                verified=verified,
                receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        monkeypatch.setattr(lane.signal, "pthread_sigmask", real_mask)
    try:
        assert block_calls == 3 and restore_calls == 2
        assert masks_seen
        assert all((observed & blocked) == blocked for observed in masks_seen)
        assert real_mask(signal.SIG_BLOCK, set()) == caller_mask
        assert len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_aborted(captured[0])
    finally:
        real_mask(signal.SIG_SETMASK, caller_mask)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_execution_gate_and_publication_outputs_stay_absent(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan.json"
    plan = r13g_tests.r13f_tests._valid_plan(plan_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed R13H gate reached run creation or process launch")

    monkeypatch.setattr(r13c, "_build_run_tree", forbidden)
    monkeypatch.setattr(r13c.subprocess, "Popen", forbidden)
    with pytest.raises(r13c.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
            stage=STAGE,
            run_id="pose-r13h-gate-unit",
            accepted_manifest_fingerprint="1" * 64,
            accepted_commits={},
        )
    assert lane.EXECUTION_AUTHORIZED is False
    assert not lane.DEFAULT_CONTROLLER.exists()
    assert not r13c.DEFAULT_PLAN.exists()
    assert not r13c.RUNS_ROOT.exists()
