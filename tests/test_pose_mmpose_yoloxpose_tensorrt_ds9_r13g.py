from __future__ import annotations

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

from tests import test_pose_mmpose_yoloxpose_tensorrt_ds9_r13f as r13f_tests
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as r13c
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13f as r13f
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13g as lane


STAMP = r13f_tests.STAMP
STAGE = r13f_tests.STAGE


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13g-test-", dir=lane.ROOT)
    try:
        yield Path(context.name)
    finally:
        context.cleanup()


def _publication_fixture(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> tuple[lane.HeldVerifiedStage, int, int, dict[str, Any], dict[str, Any]]:
    return r13f_tests._publication_fixture(monkeypatch, repo_tmp_path)


def _close_fixture(
    verified: lane.HeldVerifiedStage,
    receipt_fd: int,
    commit_fd: int,
) -> None:
    r13f_tests._close_fixture(verified, receipt_fd, commit_fd)


def _assert_pair_absent() -> None:
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


def _assert_lifetime_closed(lifetime: lane.ReceiptCommitPairLifetime) -> None:
    assert lifetime.closed is True
    assert lifetime.rollback_handles == []
    assert lifetime.pending_close_descriptors == []
    for item in (lifetime.receipt, lifetime.commit):
        assert item.target_descriptor == -1
        assert item.chain is None


def test_frozen_r13c_r13d_r13e_r13f_ancestry_is_exact() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed == {
        "r13c": lane.FROZEN_R13C_LINEAGE,
        "r13d": lane.FROZEN_R13D_LINEAGE,
        "r13e": lane.FROZEN_R13E_LINEAGE,
        "r13f": lane.FROZEN_R13F_LINEAGE,
    }
    assert len(observed["r13c"]) == 11
    assert len(observed["r13d"]) == 4
    assert len(observed["r13e"]) == 4
    assert len(observed["r13f"]) == 4
    assert r13f.FROZEN_R13C_LINEAGE == lane.FROZEN_R13C_LINEAGE
    assert r13f.FROZEN_R13D_LINEAGE == lane.FROZEN_R13D_LINEAGE
    assert r13f.FROZEN_R13E_LINEAGE == lane.FROZEN_R13E_LINEAGE


def test_controller_schema_owns_exact_r13f_lineage() -> None:
    schema = json.loads(lane.CONTROLLER_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    Draft202012Validator(schema).validate(value)
    assert value["accepted_lineage"]["r13f"] == lane.FROZEN_R13F_LINEAGE
    forged = copy.deepcopy(value)
    forged["accepted_lineage"]["r13f"] = {
        f"validation/forged-r13f-{index}.py": {
            "mode": "0440", "bytes": 1, "sha256": "0" * 64,
        }
        for index in range(4)
    }
    forged["fingerprint_sha256"] = r13c.fingerprint(forged)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(forged)


def test_positive_pair_publishes_and_consumes_all_lifetime_fds(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    original = lane.ReceiptCommitPairLifetime.assert_immediately_before_success

    def capture(self: lane.ReceiptCommitPairLifetime) -> None:
        original(self)
        captured.append(self)

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_immediately_before_success",
        capture,
    )
    try:
        result = lane.publish_receipt_and_commit(
            verified=verified,
            receipt_descriptor=receipt_fd,
            commit_descriptor=commit_fd,
            receipt_published=receipt_pin,
            commit_published=commit_pin,
        )
        assert result["receipt"]["sha256"] == receipt_pin["sha256"]
        assert result["commit"]["sha256"] == commit_pin["sha256"]
        assert r13c.receipt_path(STAGE).is_file()
        assert r13c.commit_path(STAGE).is_file()
        assert len(captured) == 1
        _assert_lifetime_closed(captured[0])
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_restore_to_close_keyboardinterrupt_is_caught_and_totally_aborted(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    original = lane.ReceiptCommitPairLifetime.assert_immediately_before_success
    source, start = inspect.getsourcelines(lane._restore_and_close)
    close_line = start + next(
        index for index, text in enumerate(source) if "lifetime.close()" in text
    )

    def capture(self: lane.ReceiptCommitPairLifetime) -> None:
        original(self)
        captured.append(self)

    def interrupt_between_restore_and_close(
        frame: Any, event: str, _argument: Any,
    ) -> Any:
        if (
            frame.f_code is lane._restore_and_close.__code__
            and event == "line"
            and frame.f_lineno == close_line
        ):
            raise KeyboardInterrupt("injected restore-to-close SIGINT")
        return interrupt_between_restore_and_close

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_immediately_before_success",
        capture,
    )
    sys.settrace(interrupt_between_restore_and_close)
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
        _assert_lifetime_closed(captured[0])
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


@pytest.mark.parametrize(
    "block_failure",
    [
        KeyboardInterrupt("injected re-block BaseException"),
        OSError(errno.EIO, "injected re-block OSError"),
    ],
    ids=["baseexception", "oserror"],
)
def test_failed_reblock_reports_failure_and_cleans_without_mask_claim(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
    block_failure: BaseException,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    original_barrier = lane.ReceiptCommitPairLifetime.assert_immediately_before_success
    original_rollback = lane._PairMember.rollback
    real_mask = signal.pthread_sigmask
    caller_mask = real_mask(signal.SIG_BLOCK, set())
    blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    masks_seen: list[set[signal.Signals]] = []
    block_calls = 0
    restore_calls = 0

    def capture(self: lane.ReceiptCommitPairLifetime) -> None:
        original_barrier(self)
        captured.append(self)

    def injected_mask(how: int, mask: object) -> set[signal.Signals]:
        nonlocal block_calls, restore_calls
        if how == signal.SIG_BLOCK:
            block_calls += 1
            # Query, initial block, then failed re-block.
            if block_calls == 3:
                raise block_failure
        result = real_mask(how, mask)
        if how == signal.SIG_SETMASK:
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("pending SIGINT on restore")
        return result

    def observe_rollback(self: lane._PairMember) -> None:
        masks_seen.append(real_mask(signal.SIG_BLOCK, set()))
        original_rollback(self)

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_immediately_before_success",
        capture,
    )
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
        assert block_calls == 3
        assert restore_calls == 2
        assert masks_seen
        assert all((observed & blocked) == (caller_mask & blocked)
                   for observed in masks_seen)
        assert len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_closed(captured[0])
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_close_then_raise_never_closes_reused_numeric_fd(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    captured: list[lane.ReceiptCommitPairLifetime] = []
    target_descriptors: set[int] = set()
    replacement_descriptor = -1
    injected_descriptor = -1
    injected = False
    real_close = os.close

    def capture_close_barrier(self: lane.ReceiptCommitPairLifetime) -> None:
        captured.append(self)
        target_descriptors.update({
            self.receipt.target_descriptor,
            self.commit.target_descriptor,
        })

    def close_then_raise(descriptor: int) -> None:
        nonlocal replacement_descriptor, injected_descriptor, injected
        if descriptor in target_descriptors and not injected:
            injected = True
            injected_descriptor = descriptor
            real_close(descriptor)
            replacement_descriptor = os.open("/dev/null", os.O_RDONLY)
            assert replacement_descriptor == descriptor
            raise KeyboardInterrupt("close returned after kernel release")
        real_close(descriptor)

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "_close_barrier",
        capture_close_barrier,
    )
    monkeypatch.setattr(lane.os, "close", close_then_raise)
    try:
        with pytest.raises(KeyboardInterrupt, match="kernel release"):
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
        assert injected is True
        assert injected_descriptor == replacement_descriptor
        os.fstat(replacement_descriptor)
        assert len(captured) == 1
        _assert_pair_absent()
        _assert_lifetime_closed(captured[0])
    finally:
        if replacement_descriptor >= 0:
            real_close(replacement_descriptor)
        _close_fixture(verified, receipt_fd, commit_fd)


def test_execution_gate_stays_closed_before_run_or_external_process(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan.json"
    plan = r13f_tests._valid_plan(plan_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed R13G gate reached run creation or process launch")

    monkeypatch.setattr(r13c, "_build_run_tree", forbidden)
    monkeypatch.setattr(r13c.subprocess, "Popen", forbidden)
    with pytest.raises(r13c.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
            stage=STAGE,
            run_id="pose-r13g-gate-unit",
            accepted_manifest_fingerprint="1" * 64,
            accepted_commits={},
        )
    assert lane.EXECUTION_AUTHORIZED is False
    assert not lane.DEFAULT_CONTROLLER.exists()
    assert not r13c.DEFAULT_PLAN.exists()
    assert not r13c.RUNS_ROOT.exists()
