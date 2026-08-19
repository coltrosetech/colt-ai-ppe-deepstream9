from __future__ import annotations

import copy
import inspect
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as r13c
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13d as lane


STAMP = "2026-07-18T16:00:00+03:00"
STAMP_B = "2026-07-18T16:01:00+03:00"


@pytest.fixture
def repo_tmp_path() -> Path:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13d-test-", dir=lane.ROOT)
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


def _valid_plan(path: Path, *, stamp: str = STAMP) -> dict[str, object]:
    return r13c.prepare_plan(path, prepared_at_utc=stamp)


def _held_stage(plan_path: Path, plan: dict[str, object]) -> lane.HeldVerifiedStage:
    held = r13c.verify_plan_held(
        plan_path, expected_fingerprint=str(plan["fingerprint_sha256"]),
    )
    return lane.HeldVerifiedStage(
        held_plan=held, plan=copy.deepcopy(held.value),
        manifest={"plan": held.document_pin}, predecessors=[],
        stream_descriptor=-1, stream_pin={}, tree=FakeTree(),
    )


def _write_held_json(path: Path, value: dict[str, object]) -> tuple[int, dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    os.chmod(path, 0o440)
    return r13c.open_held_source(path)


def _publication_documents(
    repo_tmp_path: Path, verified: lane.HeldVerifiedStage,
) -> tuple[int, int, dict[str, object], dict[str, object]]:
    document = {
        "path": "validation/results/unit-r13d.json", "bytes": 7,
        "sha256": "a" * 64, "fingerprint_sha256": "b" * 64,
    }
    artifact = {
        "path": "validation/results/unit-r13d.bin", "bytes": 11,
        "sha256": "c" * 64,
    }
    lifecycle = {
        "start_new_session": True,
        "timeout_seconds": r13c.LAUNCHER_TIMEOUT["tensorrt_fp16_640"],
        "duration_seconds": 1.0,
        "stdin_held_snapshot_fd": True,
        "output_limit_bytes": r13c.MAX_HOST_LOG_BYTES,
        "output_bytes": 12,
        "orphan_survived": False,
    }
    receipt = lane.build_receipt(
        plan=verified.plan, plan_pin=verified.plan_pin,
        manifest_pin=document, stream_pin=artifact,
        stage="tensorrt_fp16_640", run_id="pose-r13d-publication-unit",
        worker_pin=document, worker_outputs=[artifact],
        primary_published=artifact, predecessors=[], numerical=None,
        lifecycle=lifecycle, created_at_utc=STAMP,
    )
    receipt_descriptor, receipt_source = _write_held_json(
        repo_tmp_path / "control/execution-receipt-r13c.json", receipt,
    )
    receipt_path = repo_tmp_path / "canonical/receipts/tensorrt_fp16_640.json"
    receipt_published: dict[str, object] = {
        "path": r13c.repo_relative(receipt_path),
        "bytes": receipt_source["bytes"], "sha256": receipt_source["sha256"],
        "fingerprint_sha256": receipt["fingerprint_sha256"],
    }
    commit = lane.build_commit(
        plan=verified.plan, plan_pin=verified.plan_pin,
        manifest_pin=document, stage="tensorrt_fp16_640",
        run_id="pose-r13d-publication-unit", primary_pin=artifact,
        receipt_pin=receipt_published, worker_pin=document,
        predecessors=[], committed_at_utc=STAMP,
    )
    commit_descriptor, commit_source = _write_held_json(
        repo_tmp_path / "control/commit-r13c.json", commit,
    )
    commit_path = repo_tmp_path / "canonical/commits/tensorrt_fp16_640.json"
    commit_published: dict[str, object] = {
        "path": r13c.repo_relative(commit_path),
        "bytes": commit_source["bytes"], "sha256": commit_source["sha256"],
        "fingerprint_sha256": commit["fingerprint_sha256"],
    }
    return (
        receipt_descriptor, commit_descriptor,
        receipt_published, commit_published,
    )


def test_frozen_r13c_lineage_matches_exact_accepted_bytes_modes_and_hashes() -> None:
    observed = lane.verify_frozen_r13c_lineage()
    assert observed == lane.FROZEN_R13C_LINEAGE
    assert len(observed) == 11
    assert observed[
        "validation/pose_mmpose_yoloxpose_tensorrt_ds9_r13c.py"
    ] == {
        "mode": "0664", "bytes": 146433,
        "sha256": "43ad6baf8f9ef23915a0b616b48834b487da78375e06827af8d93e53d1da9eb6",
    }


def test_controller_contract_fixes_ownership_and_closed_execution_gate() -> None:
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    assert value["accepted_lineage"] == lane.FROZEN_R13C_LINEAGE
    assert value["ownership"] == {
        "verify_stage_returns_owned_held_verified_plan": True,
        "same_plan_fd_held_through_receipt_and_commit_fsync_link": True,
        "receipt_builder_accepts_exact_plan_pin_without_path_reopen": True,
        "commit_builder_accepts_exact_plan_pin_without_path_reopen": True,
        "receipt_and_commit_embed_exact_held_plan_pin": True,
        "publication_pair_rolls_back_on_plan_name_inode_drift": True,
        "canonical_parents_root_dirfd_openat_nofollow": True,
    }
    assert value["execution_gate"] == {
        "authorized": False,
        "exact_runtime_module_binding_required": True,
        "exact_runtime_module_binding_available": False,
        "lease_v1_authorized": False,
        "block_reason": lane.RUNTIME_BINDING_BLOCK_REASON,
    }
    assert value["claim_boundary"]["r13d_gpu_executed"] is False
    assert value["fingerprint_sha256"] == r13c.fingerprint(value)


def test_controller_schema_rejects_opened_gate() -> None:
    value = lane.build_controller_contract(prepared_at_utc=STAMP)
    changed = copy.deepcopy(value)
    changed["execution_gate"]["authorized"] = True
    changed["fingerprint_sha256"] = r13c.fingerprint(changed)
    with pytest.raises(r13c.PoseR13CError, match="schema validation failed"):
        r13c.validate_schema(changed, lane.CONTROLLER_SCHEMA)


def test_verify_stage_returns_first_plan_fd_as_owned_stage(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    replacement = repo_tmp_path / "valid-plan-b.json"
    displaced = repo_tmp_path / "accepted-plan-a.displaced.json"
    plan_a = _valid_plan(plan_path)
    _valid_plan(replacement, stamp=STAMP_B)
    expected_plan_pin = r13c.document_pin(plan_path)
    stream_path = repo_tmp_path / "snapshot.stream"
    stream_path.write_bytes(b"accepted-r13d-stream")
    stream_pin = r13c.file_pin(stream_path)
    fake_tree = FakeTree()

    def fake_verify_stage(**_kwargs):
        return (
            copy.deepcopy(plan_a), {"plan": expected_plan_pin}, [],
            os.open(stream_path, os.O_RDONLY), stream_pin, fake_tree,
        )

    monkeypatch.setattr(r13c, "verify_stage", fake_verify_stage)
    verified = lane.verify_stage(
        plan_path=plan_path,
        accepted_plan_fingerprint=str(plan_a["fingerprint_sha256"]),
        stage="tensorrt_fp16_640", run_id="pose-r13d-held-stage-unit",
        accepted_manifest_fingerprint="d" * 64, accepted_commits={},
    )
    try:
        assert verified.held_plan.closed is False
        assert verified.plan_pin == expected_plan_pin
        os.replace(plan_path, displaced)
        os.replace(replacement, plan_path)
        with pytest.raises(r13c.PoseR13CError, match="name/inode metadata differs"):
            verified.assert_stable()
    finally:
        verified.close()
    assert fake_tree.closed is True


def test_receipt_and_commit_builders_never_reopen_plan_path_and_embed_a_pin(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    replacement = repo_tmp_path / "valid-plan-b.json"
    displaced = repo_tmp_path / "accepted-plan-a.displaced.json"
    plan_a = _valid_plan(plan_path)
    _valid_plan(replacement, stamp=STAMP_B)
    plan_a_pin = r13c.document_pin(plan_path)
    os.replace(plan_path, displaced)
    os.replace(replacement, plan_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("receipt/commit builder reopened a plan pathname")

    monkeypatch.setattr(r13c, "file_pin", forbidden)
    document = {
        "path": "validation/results/unit.json", "bytes": 7,
        "sha256": "a" * 64, "fingerprint_sha256": "b" * 64,
    }
    artifact = {
        "path": "validation/results/unit.bin", "bytes": 11,
        "sha256": "c" * 64,
    }
    lifecycle = {
        "start_new_session": True, "timeout_seconds": 14520,
        "duration_seconds": 1.0, "stdin_held_snapshot_fd": True,
        "output_limit_bytes": r13c.MAX_HOST_LOG_BYTES, "output_bytes": 1,
        "orphan_survived": False,
    }
    receipt = lane.build_receipt(
        plan=plan_a, plan_pin=plan_a_pin, manifest_pin=document,
        stream_pin=artifact, stage="tensorrt_fp16_640",
        run_id="pose-r13d-builder-unit", worker_pin=document,
        worker_outputs=[artifact], primary_published=artifact,
        predecessors=[], numerical=None, lifecycle=lifecycle,
        created_at_utc=STAMP,
    )
    receipt_pin = {**document, "path": "validation/results/receipt.json"}
    commit = lane.build_commit(
        plan=plan_a, plan_pin=plan_a_pin, manifest_pin=document,
        stage="tensorrt_fp16_640", run_id="pose-r13d-builder-unit",
        primary_pin=artifact, receipt_pin=receipt_pin,
        worker_pin=document, predecessors=[], committed_at_utc=STAMP,
    )
    assert receipt["plan"] == plan_a_pin
    assert commit["plan"] == plan_a_pin
    assert "plan_path" not in inspect.signature(lane.build_receipt).parameters
    assert "plan_path" not in inspect.signature(lane.build_commit).parameters


def test_receipt_and_commit_publish_as_exact_held_a_pair(repo_tmp_path: Path) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    plan = _valid_plan(plan_path)
    verified = _held_stage(plan_path, plan)
    receipt_descriptor = commit_descriptor = -1
    try:
        (
            receipt_descriptor, commit_descriptor,
            receipt_expected, commit_expected,
        ) = _publication_documents(repo_tmp_path, verified)
        result = lane.publish_receipt_and_commit(
            verified=verified, receipt_descriptor=receipt_descriptor,
            commit_descriptor=commit_descriptor,
            receipt_published=receipt_expected,
            commit_published=commit_expected,
        )
        assert result["receipt"] == {
            key: receipt_expected[key] for key in ("path", "bytes", "sha256")
        }
        assert result["commit"] == {
            key: commit_expected[key] for key in ("path", "bytes", "sha256")
        }
        receipt_path = lane.ROOT / str(receipt_expected["path"])
        commit_path = lane.ROOT / str(commit_expected["path"])
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o440
        assert stat.S_IMODE(commit_path.stat().st_mode) == 0o440
        assert r13c.load_json(receipt_path)["plan"] == verified.plan_pin
        assert r13c.load_json(commit_path)["plan"] == verified.plan_pin
    finally:
        if receipt_descriptor >= 0:
            os.close(receipt_descriptor)
        if commit_descriptor >= 0:
            os.close(commit_descriptor)
        verified.close()


def test_plan_a_to_b_swap_after_receipt_link_rolls_back_both_publications(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    replacement = repo_tmp_path / "valid-plan-b.json"
    displaced = repo_tmp_path / "accepted-plan-a.displaced.json"
    plan_a = _valid_plan(plan_path)
    _valid_plan(replacement, stamp=STAMP_B)
    verified = _held_stage(plan_path, plan_a)
    (
        receipt_descriptor, commit_descriptor,
        receipt_expected, commit_expected,
    ) = _publication_documents(repo_tmp_path, verified)
    original_link = lane._PendingPublication.link
    injected = False

    def link_then_swap(self: lane._PendingPublication):
        nonlocal injected
        result = original_link(self)
        if self.expected["path"] == receipt_expected["path"] and not injected:
            os.replace(plan_path, displaced)
            os.replace(replacement, plan_path)
            injected = True
        return result

    monkeypatch.setattr(lane._PendingPublication, "link", link_then_swap)
    try:
        with pytest.raises(r13c.PoseR13CError, match="name/inode metadata differs"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_descriptor,
                commit_descriptor=commit_descriptor,
                receipt_published=receipt_expected,
                commit_published=commit_expected,
            )
    finally:
        os.close(receipt_descriptor)
        os.close(commit_descriptor)
        verified.close()
    assert injected is True
    assert not (lane.ROOT / str(receipt_expected["path"])).exists()
    assert not (lane.ROOT / str(commit_expected["path"])).exists()


def test_staged_document_with_different_plan_pin_is_rejected_before_publication(
    repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    plan = _valid_plan(plan_path)
    verified = _held_stage(plan_path, plan)
    (
        receipt_descriptor, commit_descriptor,
        receipt_expected, commit_expected,
    ) = _publication_documents(repo_tmp_path, verified)
    forged_path = repo_tmp_path / "forged-receipt.json"
    forged = r13c.read_held_json(
        receipt_descriptor,
        {
            "bytes": receipt_expected["bytes"],
            "sha256": receipt_expected["sha256"],
        },
        source="forged-receipt",
    )
    forged["plan"] = {**verified.plan_pin, "sha256": "f" * 64}
    forged["fingerprint_sha256"] = r13c.fingerprint(forged)
    os.close(receipt_descriptor)
    receipt_descriptor, forged_source = _write_held_json(forged_path, forged)
    forged_expected = {
        **receipt_expected,
        "bytes": forged_source["bytes"], "sha256": forged_source["sha256"],
        "fingerprint_sha256": forged["fingerprint_sha256"],
    }
    try:
        with pytest.raises(r13c.PoseR13CError, match="staged receipt plan pin differs"):
            lane.publish_receipt_and_commit(
                verified=verified, receipt_descriptor=receipt_descriptor,
                commit_descriptor=commit_descriptor,
                receipt_published=forged_expected,
                commit_published=commit_expected,
            )
    finally:
        os.close(receipt_descriptor)
        os.close(commit_descriptor)
        verified.close()
    assert not (lane.ROOT / str(receipt_expected["path"])).exists()
    assert not (lane.ROOT / str(commit_expected["path"])).exists()


def test_execute_gate_fails_before_stage_run_or_external_runtime(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan.json"
    plan = _valid_plan(plan_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("closed R13D gate reached stage or process launch")

    monkeypatch.setattr(lane, "verify_stage", forbidden)
    monkeypatch.setattr(r13c.subprocess, "Popen", forbidden)
    with pytest.raises(r13c.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=str(plan["fingerprint_sha256"]),
            stage="tensorrt_fp16_640", run_id="pose-r13d-gate-unit",
            accepted_manifest_fingerprint="e" * 64, accepted_commits={},
        )
    assert not (repo_tmp_path / "runs").exists()
    assert not lane.DEFAULT_CONTROLLER.exists()


def test_import_and_contract_build_do_not_launch_gpu_docker_or_nvidia() -> None:
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
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13d as lane
value = lane.build_controller_contract(prepared_at_utc="2026-07-18T16:00:00+03:00")
assert value["execution_gate"]["authorized"] is False
print(value["fingerprint_sha256"])
'''
    completed = subprocess.run(
        [os.fspath(Path(os.sys.executable)), "-c", code], cwd=lane.ROOT,
        check=False, capture_output=True, text=True, timeout=30,
        env={
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(completed.stdout.strip()) == 64
