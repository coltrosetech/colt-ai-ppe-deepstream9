from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as r13c
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13d as r13d
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13e as lane


STAMP = "2026-07-18T18:00:00+03:00"
STAGE = "tensorrt_fp16_640"
RUN_ID = "pose-r13e-canonical-pair-unit"


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13e-test-", dir=lane.ROOT)
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


def _valid_plan(path: Path) -> dict[str, object]:
    return r13c.prepare_plan(path, prepared_at_utc=STAMP)


def _held_stage(plan_path: Path, plan: dict[str, object]) -> lane.HeldVerifiedStage:
    held = r13c.verify_plan_held(
        plan_path, expected_fingerprint=str(plan["fingerprint_sha256"]),
    )
    return lane.HeldVerifiedStage(
        held_plan=held, plan=copy.deepcopy(held.value),
        manifest={"plan": held.document_pin}, predecessors=[],
        stream_descriptor=-1, stream_pin={}, tree=FakeTree(),
    )


def _write_json(path: Path, value: dict[str, object]) -> tuple[int, dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    os.chmod(path, 0o440)
    return r13c.open_held_source(path)


def _canonical_documents(
    repo_tmp_path: Path, verified: lane.HeldVerifiedStage,
) -> tuple[int, int, dict[str, object], dict[str, object]]:
    run_root = r13c.RUNS_ROOT / STAGE / RUN_ID
    manifest = {
        "path": r13c.repo_relative(run_root / "snapshot/manifest-r13c.json"),
        "bytes": 7, "sha256": "a" * 64, "fingerprint_sha256": "b" * 64,
    }
    stream = {
        "path": r13c.repo_relative(run_root / "snapshot/snapshot-r13c.stream"),
        "bytes": 9, "sha256": "c" * 64,
    }
    worker = {
        "path": r13c.repo_relative(run_root / "output/worker-result.json"),
        "bytes": 11, "sha256": "d" * 64, "fingerprint_sha256": "e" * 64,
    }
    primary = {
        "path": r13c.repo_relative(r13c.primary_path(STAGE)),
        "bytes": 13, "sha256": "f" * 64,
    }
    lifecycle = {
        "start_new_session": True,
        "timeout_seconds": r13c.LAUNCHER_TIMEOUT[STAGE],
        "duration_seconds": 1.0, "stdin_held_snapshot_fd": True,
        "output_limit_bytes": r13c.MAX_HOST_LOG_BYTES,
        "output_bytes": 17, "orphan_survived": False,
    }
    receipt = lane.build_receipt(
        plan=verified.plan, plan_pin=verified.plan_pin,
        manifest_pin=manifest, stream_pin=stream, stage=STAGE,
        run_id=RUN_ID, worker_pin=worker, worker_outputs=[primary],
        primary_published=primary, predecessors=[], numerical=None,
        lifecycle=lifecycle, created_at_utc=STAMP,
    )
    receipt_descriptor, receipt_source = _write_json(
        repo_tmp_path / "control/execution-receipt-r13c.json", receipt,
    )
    receipt_published: dict[str, object] = {
        "path": r13c.repo_relative(r13c.receipt_path(STAGE)),
        "bytes": receipt_source["bytes"], "sha256": receipt_source["sha256"],
        "fingerprint_sha256": receipt["fingerprint_sha256"],
    }
    commit = lane.build_commit(
        plan=verified.plan, plan_pin=verified.plan_pin,
        manifest_pin=manifest, stage=STAGE, run_id=RUN_ID,
        primary_pin=primary, receipt_pin=receipt_published,
        worker_pin=worker, predecessors=[], committed_at_utc=STAMP,
    )
    commit_descriptor, commit_source = _write_json(
        repo_tmp_path / "control/commit-r13c.json", commit,
    )
    commit_published: dict[str, object] = {
        "path": r13c.repo_relative(r13c.commit_path(STAGE)),
        "bytes": commit_source["bytes"], "sha256": commit_source["sha256"],
        "fingerprint_sha256": commit["fingerprint_sha256"],
    }
    return (
        receipt_descriptor, commit_descriptor,
        receipt_published, commit_published,
    )


def _publication_fixture(
    monkeypatch, repo_tmp_path: Path,
) -> tuple[lane.HeldVerifiedStage, int, int, dict[str, object], dict[str, object]]:
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
    verified: lane.HeldVerifiedStage, receipt_fd: int, commit_fd: int,
) -> None:
    os.close(receipt_fd)
    os.close(commit_fd)
    verified.close()


def _rewrite_document(
    descriptor: int, path: Path, mutate,
) -> tuple[int, dict[str, object], dict[str, object]]:
    metadata = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    raw = os.read(descriptor, metadata.st_size)
    value = json.loads(raw)
    mutate(value)
    value["fingerprint_sha256"] = r13c.fingerprint(value)
    os.close(descriptor)
    new_descriptor, source = _write_json(path, value)
    return new_descriptor, source, value


def test_frozen_r13c_and_r13d_ancestry_is_exact() -> None:
    observed = lane.verify_frozen_ancestry()
    assert observed["r13c"] == r13d.FROZEN_R13C_LINEAGE
    assert observed["r13d"] == lane.FROZEN_R13D_LINEAGE
    assert len(observed["r13c"]) == 11 and len(observed["r13d"]) == 4


def test_controller_contract_fixes_pair_continuity_cleanup_and_closed_gate() -> None:
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    ownership = value["ownership"]
    assert ownership["stage_verify_uses_one_plan_fd_without_second_path_open"] is True
    assert ownership["receipt_commit_pair_has_one_explicit_held_lifetime"] is True
    assert ownership["receipt_continuity_after_receipt_link"] is True
    assert ownership["receipt_continuity_before_commit_link"] is True
    assert ownership["receipt_continuity_after_commit_link"] is True
    assert ownership["receipt_continuity_immediately_before_success"] is True
    assert ownership["commit_visible_requires_exact_canonical_held_receipt"] is True
    assert ownership["canonical_paths_derived_from_stage"] is True
    assert ownership["close_or_signal_restore_failure_rolls_back_pair"] is True
    assert value["execution_gate"]["authorized"] is False
    assert value["fingerprint_sha256"] == r13c.fingerprint(value)


def test_verify_stage_uses_exactly_one_plan_open_and_not_r13c_verify_stage(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-r13c.json"
    plan = _valid_plan(plan_path)
    runs_root = repo_tmp_path / "runs"
    monkeypatch.setattr(r13c, "RUNS_ROOT", runs_root)
    prepared = r13c.prepare_stage(
        plan_path=plan_path,
        accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
        stage=STAGE, run_id="pose-r13e-single-plan-open-unit",
        accepted_commits={}, prepared_at_utc=STAMP,
    )
    original = r13c.verify_plan_held
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("R13E called path-reopening R13C verify_stage")

    monkeypatch.setattr(r13c, "verify_plan_held", counted)
    monkeypatch.setattr(r13c, "verify_stage", forbidden)
    monkeypatch.setattr(r13c, "_accepted_predecessors", forbidden)
    verified = lane.verify_stage(
        plan_path=plan_path,
        accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
        stage=STAGE, run_id="pose-r13e-single-plan-open-unit",
        accepted_manifest_fingerprint=str(prepared["manifest"]["fingerprint_sha256"]),
        accepted_commits={},
    )
    try:
        assert calls == 1
        assert verified.held_plan.closed is False
        assert verified.manifest["plan"] == verified.plan_pin
    finally:
        verified.close()
        os.chmod(
            runs_root / STAGE / "pose-r13e-single-plan-open-unit/snapshot", 0o700,
        )


def test_held_stage_close_attempts_every_resource_and_aggregates(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FailingTree:
        def close(self) -> None:
            calls.append("tree")
            raise OSError("tree close")

    class FailingPlan:
        def close(self) -> None:
            calls.append("plan")
            raise OSError("plan close")

    def failing_close(_descriptor: int) -> None:
        calls.append("stream")
        raise OSError("stream close")

    monkeypatch.setattr(lane.os, "close", failing_close)
    verified = lane.HeldVerifiedStage(
        held_plan=FailingPlan(), plan={}, manifest={}, predecessors=[],
        stream_descriptor=123, stream_pin={}, tree=FailingTree(),
    )
    with pytest.raises(ExceptionGroup) as captured:
        verified.close()
    assert calls == ["stream", "tree", "plan"]
    assert len(captured.value.exceptions) == 3
    assert verified.closed is True and verified.stream_descriptor == -1


def test_positive_pair_uses_exact_stage_canonical_paths_and_held_receipt(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    try:
        result = lane.publish_receipt_and_commit(
            verified=verified, receipt_descriptor=receipt_fd,
            commit_descriptor=commit_fd, receipt_published=receipt_pin,
            commit_published=commit_pin,
        )
        assert receipt_pin["path"] == r13c.repo_relative(r13c.receipt_path(STAGE))
        assert commit_pin["path"] == r13c.repo_relative(r13c.commit_path(STAGE))
        receipt = r13c.load_json(r13c.receipt_path(STAGE))
        commit = r13c.load_json(r13c.commit_path(STAGE))
        assert result["receipt"] == {
            key: receipt_pin[key] for key in ("path", "bytes", "sha256")
        }
        assert result["commit"] == {
            key: commit_pin[key] for key in ("path", "bytes", "sha256")
        }
        assert commit["execution_receipt"] == receipt_pin
        assert receipt["primary_artifact"]["path"] == r13c.repo_relative(
            r13c.primary_path(STAGE)
        )
        assert commit["primary_artifact"] == receipt["primary_artifact"]
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


@pytest.mark.parametrize("attack_boundary", ["after_receipt_link", "after_commit_link"])
def test_receipt_parent_rename_recreate_fails_and_rolls_back_both_held_links(
    monkeypatch, repo_tmp_path: Path, attack_boundary: str,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    receipt_parent = r13c.receipt_path(STAGE).parent
    displaced = receipt_parent.with_name(receipt_parent.name + ".displaced")
    original = lane.ReceiptCommitPairLifetime.assert_receipt_continuity
    injected = False

    def assert_then_swap(self, *, boundary: str):
        nonlocal injected
        result = original(self, boundary=boundary)
        if boundary == attack_boundary and not injected:
            os.replace(receipt_parent, displaced)
            receipt_parent.mkdir(mode=0o755)
            injected = True
        return result

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime,
        "assert_receipt_continuity", assert_then_swap,
    )
    try:
        with pytest.raises(
            (r13c.PoseR13CError, ExceptionGroup),
            match="directory-chain name/inode binding differs",
        ):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd, receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)
    assert injected is True
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()
    assert not (displaced / r13c.receipt_path(STAGE).name).exists()


@pytest.mark.parametrize("redirected", ["receipt", "commit"])
def test_redirected_publication_path_is_rejected_before_canonical_parent_open(
    monkeypatch, repo_tmp_path: Path, redirected: str,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    changed_receipt = dict(receipt_pin)
    changed_commit = dict(commit_pin)
    target = repo_tmp_path / f"redirected/{redirected}.json"
    if redirected == "receipt":
        changed_receipt["path"] = r13c.repo_relative(target)
    else:
        changed_commit["path"] = r13c.repo_relative(target)
    try:
        with pytest.raises(r13c.PoseR13CError, match="canonical path differs"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd,
                receipt_published=changed_receipt,
                commit_published=changed_commit,
            )
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)
    assert not target.exists()
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


@pytest.mark.parametrize("document_kind", ["receipt", "commit"])
def test_redirected_primary_in_either_document_is_rejected(
    monkeypatch, repo_tmp_path: Path, document_kind: str,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    redirect = r13c.repo_relative(repo_tmp_path / "redirected/primary.bin")
    if document_kind == "receipt":
        receipt_fd, source, value = _rewrite_document(
            receipt_fd, repo_tmp_path / "control/forged-receipt.json",
            lambda item: item["primary_artifact"].update(path=redirect),
        )
        receipt_pin = {
            **receipt_pin, "bytes": source["bytes"], "sha256": source["sha256"],
            "fingerprint_sha256": value["fingerprint_sha256"],
        }
    else:
        commit_fd, source, value = _rewrite_document(
            commit_fd, repo_tmp_path / "control/forged-commit.json",
            lambda item: item["primary_artifact"].update(path=redirect),
        )
        commit_pin = {
            **commit_pin, "bytes": source["bytes"], "sha256": source["sha256"],
            "fingerprint_sha256": value["fingerprint_sha256"],
        }
    try:
        with pytest.raises(r13c.PoseR13CError, match="primary canonical path differs"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd, receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)


def test_signal_restore_exception_rolls_back_successfully_linked_pair(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )
    real_mask = signal.pthread_sigmask
    calls = 0

    def restore_then_fail(how, mask):
        nonlocal calls
        calls += 1
        result = real_mask(how, mask)
        if calls == 2:
            raise OSError("injected signal restore failure")
        return result

    monkeypatch.setattr(lane.signal, "pthread_sigmask", restore_then_fail)
    try:
        with pytest.raises(ExceptionGroup, match="cleanup failures"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd, receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)
    assert calls == 2
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


def test_preclose_exception_rolls_back_successfully_linked_pair(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    verified, receipt_fd, commit_fd, receipt_pin, commit_pin = _publication_fixture(
        monkeypatch, repo_tmp_path,
    )

    def fail_before_close(_self) -> None:
        raise OSError("injected close failure")

    monkeypatch.setattr(
        lane.ReceiptCommitPairLifetime, "_close_barrier", fail_before_close,
    )
    try:
        with pytest.raises(ExceptionGroup, match="cleanup failures"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_fd,
                commit_descriptor=commit_fd, receipt_published=receipt_pin,
                commit_published=commit_pin,
            )
    finally:
        _close_fixture(verified, receipt_fd, commit_fd)
    assert not r13c.receipt_path(STAGE).exists()
    assert not r13c.commit_path(STAGE).exists()


def test_execute_gate_fails_before_stage_run_or_external_runtime(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan.json"
    plan = _valid_plan(plan_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("closed R13E gate reached stage or external runtime")

    monkeypatch.setattr(lane, "verify_stage", forbidden)
    monkeypatch.setattr(r13c.subprocess, "Popen", forbidden)
    with pytest.raises(r13c.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
            stage=STAGE, run_id="pose-r13e-gate-unit",
            accepted_manifest_fingerprint="1" * 64, accepted_commits={},
        )
    assert not lane.DEFAULT_CONTROLLER.exists()


def test_import_and_contract_build_launch_no_gpu_docker_nvidia_tensorrt_or_ds() -> None:
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
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13e as lane
value = lane.build_controller_contract(prepared_at_utc="2026-07-18T18:00:00+03:00")
assert value["execution_gate"]["authorized"] is False
print(value["fingerprint_sha256"])
'''
    completed = subprocess.run(
        [os.sys.executable, "-c", code], cwd=lane.ROOT,
        check=False, capture_output=True, text=True, timeout=30,
        env={
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(completed.stdout.strip()) == 64
