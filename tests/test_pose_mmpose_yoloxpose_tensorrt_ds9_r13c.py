from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_snapshot_bootstrap_r13c as bootstrap
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as lane
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c_container as worker


STAMP = "2026-07-18T13:30:00+03:00"
EVIDENCE_STAMP = "2026-07-18T12:41:46+03:00"


def test_independent_audit_evidence_pins_immutable_r13b_incident_lineage() -> None:
    value = lane.build_compatibility_evidence(observed_at_utc=EVIDENCE_STAMP)
    lineage = value["incident_lineage"]
    assert lineage["plan"] == lane.R13B_PLAN_PIN
    assert lineage["manifest"] == lane.R13B_MANIFEST_PIN
    assert lineage["snapshot_stream"] == lane.R13B_STREAM_PIN
    assert lineage["output_entries"] == []
    assert lineage["r13b_artifacts_modified"] is False
    assert value["pose_r13b_independent_audit"]["audit_passed"] is False
    assert value["decision"]["r13b_direct_execute_allowed"] is False
    assert value["fingerprint_sha256"] == lane.fingerprint(value)


def test_fresh_import_and_plan_build_do_not_import_gpu_modules_or_launch_processes() -> None:
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
    raise AssertionError("process launch during import/plan build")
subprocess.Popen = forbidden
subprocess.run = forbidden
subprocess.check_output = forbidden
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13c as lane
plan = lane.build_plan(prepared_at_utc="2026-07-18T13:30:00+03:00")
assert plan["claim_boundary"]["r13c_gpu_executed"] is False
assert plan["launch_contract"]["input_bind_mounts"] is False
print(plan["fingerprint_sha256"])
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=lane.ROOT,
        check=False, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(completed.stdout.strip()) == 64


def test_plan_supersedes_frozen_r13b_and_pins_private_runtime_contract() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    assert plan["supersedes"]["plan"] == lane.R13B_PLAN_PIN
    assert plan["supersedes"]["incident_manifest"] == lane.R13B_MANIFEST_PIN
    assert plan["supersedes"]["incident_snapshot_stream"] == lane.R13B_STREAM_PIN
    assert plan["supersedes"]["r13b_artifacts_modified"] is False
    launch = plan["launch_contract"]
    assert launch["docker_interactive_stdin_exact"] is True
    assert launch["proc_fd_paths_in_docker_argv"] is False
    assert launch["container_tmpfs"] == ["/inputs", "/tmp", "/work", "/workspace"]
    assert launch["generated_runtime_artifacts_private_work_tmpfs"] is True
    assert launch["worker_runtime_args_bound_to_accepted_destinations_and_hashes"] is True
    assert launch["input_engine_copied_o_excl_rehashed_before_runtime_consumption"] is True
    assert launch["engine_deserialization_path"] == "/work/model.engine"
    assert launch["generated_nvinfer_config_path"] == "/work/nvinfer-config.txt"
    assert launch["input_engine_or_config_never_passed_to_runtime"] is True
    assert launch["held_source_stability"] == (
        "device_inode_size_mtime_ctime_mode_nlink_uid_gid_before_after_hash_read_copy"
    )
    assert launch["held_json_hash_and_parse_same_fd"] is True
    assert launch["plan_single_held_fd_verify_through_snapshot_commit"] is True
    assert launch["snapshot_plan_entry_exact_held_verified_plan"] is True
    assert launch["snapshot_frame_copy_post_stability_and_named_inode"] is True
    assert launch["snapshot_stream_pre_and_post_child_exact_revalidation"] is True
    assert launch["publication_source_post_copy_stability"] is True
    assert launch["r13b_incident_plan_manifest_stream_single_held_fd"] is True
    assert launch["r13b_incident_mode_owner_nlink_exact"] is True
    assert launch["run_tree_root_dirfd_openat_nofollow"] is True
    assert launch["run_tree_all_ancestors_held_and_identity_bound"] is True
    assert launch["canonical_publication_parent_root_dirfd_openat_nofollow"] is True
    assert launch["canonical_publication_parent_held_identity_bound"] is True
    assert launch["output_mount_and_cidfile_bound_to_held_run_tree"] is True
    assert launch["managed_signal_handlers_restored_only_after_child_cleanup_reap_and_drain"] is True
    assert launch["output_drain_error_terminates_child_in_live_loop"] is True
    assert launch["predecessor_commit_receipt_exact_semantic_projection"] is True
    assert launch["predecessor_canonical_paths_required"] is True
    assert launch["recursive_predecessor_claimed_pin_equals_observed_pin"] is True
    assert launch["worker_primary_same_held_fd_validation_and_publication"] is True
    assert launch["run_output_primary_equals_canonical_primary"] is True
    assert launch["gpu_lease_exact_runtime_module_binding_required"] is True
    assert launch["gpu_lease_exact_runtime_module_binding_available"] is False
    assert launch["gpu_lease_v1_runtime_authorized"] is False
    assert launch["gpu_execution_authorized"] is False
    assert launch["runtime_binding_block_reason"] == lane.RUNTIME_BINDING_BLOCK_REASON
    assert launch["host_output_bind_used_only_for_final_verified_no_overwrite_publication"] is True
    assert launch["gpu_lease_explicit_timeout_required"] is True
    assert launch["gpu_lease_timeout_seconds"]["tensorrt_fp16_640"] == 14460
    assert launch["launcher_timeout_seconds"]["tensorrt_fp16_640"] == 14520
    assert launch["docker_cidfile_absolute_unique_required"] is True
    assert launch["managed_docker_cidfile_path_handshake_required"] is True
    assert launch["docker_cidfile_parent_owner_uid_mode"] == "0700"
    assert launch["inline_loader_sha256"] == hashlib.sha256(lane.INLINE_LOADER.encode()).hexdigest()
    assert plan["implementation"]["container_worker"]["path"].endswith("r13c_container.py")
    assert plan["implementation"]["superseded_r13b_plan"] == lane.R13B_PLAN_PIN
    assert plan["implementation"]["r13_worker_library"]["path"].endswith("r13_container.py")
    for profile in lane.PROFILES:
        assert plan["profiles"][str(profile)]["engine_path"] == lane.repo_relative(
            lane.engine_path(profile)
        )
    assert launch["canonical_primary_outputs"] == {
        stage: lane.repo_relative(lane.primary_path(stage)) for stage in lane.STAGES
    }
    assert launch["canonical_receipts"] == {
        stage: lane.repo_relative(lane.receipt_path(stage)) for stage in lane.STAGES
    }
    assert launch["canonical_commits"] == {
        stage: lane.repo_relative(lane.commit_path(stage)) for stage in lane.STAGES
    }
    assert plan["fingerprint_sha256"] == lane.fingerprint(plan)


def test_plan_schema_rejects_unknown_or_weakened_launch_property() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    changed = copy.deepcopy(plan)
    changed["launch_contract"]["input_bind_mounts"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["launch_contract"]["gpu_lease_exact_runtime_module_binding_available"] = True
    changed["launch_contract"]["gpu_execution_authorized"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["launch_contract"]["silent_fallback"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13CError, match="Additional properties"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["profiles"]["640"]["engine_path"] = changed["profiles"]["640"]["engine_path"].replace(
        "fixed-k100-r13c", "fixed-k100-r13"
    )
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def _temporary_repo_directory() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13c-test-", dir=lane.ROOT)
    return context, Path(context.name)


@pytest.fixture
def repo_tmp_path() -> Path:
    context, temporary = _temporary_repo_directory()
    try:
        yield temporary
    finally:
        context.cleanup()


@pytest.mark.parametrize("placement", ["ancestor", "stage"])
def test_run_tree_rejects_intermediate_symlink_parent_escape(
    monkeypatch, repo_tmp_path: Path, placement: str,
) -> None:
    outside = repo_tmp_path / "outside"
    outside.mkdir(mode=0o700)
    if placement == "ancestor":
        safe = repo_tmp_path / "safe"
        safe.mkdir(mode=0o700)
        (safe / "results").symlink_to(outside, target_is_directory=True)
        runs_root = safe / "results/runs"
    else:
        runs_root = repo_tmp_path / "results/runs"
        runs_root.mkdir(parents=True, mode=0o700)
        (runs_root / "tensorrt_fp16_640").symlink_to(
            outside, target_is_directory=True,
        )
    monkeypatch.setattr(lane, "RUNS_ROOT", runs_root)
    with pytest.raises(
        lane.PoseR13CError,
        match="no-follow directory|name/inode binding differs",
    ):
        lane._new_run_tree(
            "tensorrt_fp16_640", "pose-r13c-symlink-escape-unit",
        )
    assert not (outside / "pose-r13c-symlink-escape-unit").exists()


def test_held_run_tree_rejects_stage_identity_swap_after_open(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    runs_root = repo_tmp_path / "runs"
    monkeypatch.setattr(lane, "RUNS_ROOT", runs_root)
    tree = lane._new_run_tree(
        "tensorrt_fp16_640", "pose-r13c-held-stage-unit",
    )
    try:
        stage_path = runs_root / "tensorrt_fp16_640"
        displaced = runs_root / "tensorrt_fp16_640.displaced"
        os.replace(stage_path, displaced)
        stage_path.mkdir(mode=0o700)
        with pytest.raises(
            lane.PoseR13CError,
            match="held run-tree name/inode binding differs",
        ):
            tree.assert_bound()
    finally:
        tree.close()


def test_execute_is_explicitly_blocked_until_exact_v2_runtime_binding(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "plan-r13c.json"
    plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
    monkeypatch.setattr(lane, "RUNS_ROOT", repo_tmp_path / "runs")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("blocked execution reached a launcher or stage verifier")

    monkeypatch.setattr(lane, "verify_stage", forbidden)
    monkeypatch.setattr(lane.subprocess, "Popen", forbidden)
    with pytest.raises(lane.PoseR13CError, match="GPU execution blocked"):
        lane.execute_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640",
            run_id="pose-r13c-runtime-gate-unit",
            accepted_manifest_fingerprint="a" * 64,
            accepted_commits={},
        )
    assert not (repo_tmp_path / "runs").exists()


def test_prepare_stage_rejects_plan_name_swap_after_verification_before_publication(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    plan_path = repo_tmp_path / "accepted-plan-a.json"
    replacement_path = repo_tmp_path / "valid-plan-b.json"
    displaced_path = repo_tmp_path / "accepted-plan-a.displaced.json"
    plan_a = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
    plan_b = lane.prepare_plan(
        replacement_path, prepared_at_utc="2026-07-18T13:31:00+03:00",
    )
    assert plan_a["fingerprint_sha256"] != plan_b["fingerprint_sha256"]
    runs_root = repo_tmp_path / "plan-swap-runs"
    monkeypatch.setattr(lane, "RUNS_ROOT", runs_root)
    canonical_paths = (
        lane.primary_path("tensorrt_fp16_640"),
        lane.receipt_path("tensorrt_fp16_640"),
        lane.commit_path("tensorrt_fp16_640"),
    )
    assert all(not path.exists() for path in canonical_paths)
    original_source_specs = lane._source_specs
    injected = False

    def swap_after_source_specs(*args, **kwargs):
        nonlocal injected
        specs = original_source_specs(*args, **kwargs)
        os.replace(plan_path, displaced_path)
        os.replace(replacement_path, plan_path)
        injected = True
        return specs

    monkeypatch.setattr(lane, "_source_specs", swap_after_source_specs)
    with pytest.raises(lane.PoseR13CError, match="name/inode metadata differs"):
        lane.prepare_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan_a["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13c-plan-swap-unit",
            accepted_commits={}, prepared_at_utc=STAMP,
        )
    assert injected is True
    assert lane.load_json(plan_path) == plan_b
    assert not runs_root.exists()
    assert not list(repo_tmp_path.rglob("manifest-r13c.json"))
    assert not list(repo_tmp_path.rglob("snapshot-r13c.stream"))
    assert all(not path.exists() for path in canonical_paths)


def test_worker_primary_name_swap_cannot_publish_different_run_output(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    monkeypatch.setattr(lane, "RUNS_ROOT", repo_tmp_path / "runs")
    tree = lane._new_run_tree(
        "tensorrt_fp16_640", "pose-r13c-primary-swap-unit",
    )
    held: lane.HeldWorkerValidation | None = None
    try:
        worker_path = tree.output / "worker-result.json"
        primary_path = tree.output / "engine.staging"
        for name, payload in (
            (worker_path.name, b'{"worker":"accepted"}\n'),
            (primary_path.name, b"A" * 4096),
        ):
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o440, dir_fd=tree.output_descriptor,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        worker_descriptor, worker_file_pin = lane.open_held_source_at(
            tree.output_descriptor, worker_path.name, worker_path,
        )
        primary_descriptor, primary_pin = lane.open_held_source_at(
            tree.output_descriptor, primary_path.name, primary_path,
        )
        held = lane.HeldWorkerValidation(
            value={},
            worker_pin={**worker_file_pin, "fingerprint_sha256": "a" * 64},
            output_pins=[primary_pin], primary_name=primary_path.name,
            primary_pin=primary_pin, worker_descriptor=worker_descriptor,
            primary_descriptor=primary_descriptor,
        )
        held.assert_stable(tree)

        displaced = tree.output / "engine.accepted-displaced"
        replacement = tree.output / "engine.replacement"
        replacement.write_bytes(b"B" * 4096)
        os.chmod(replacement, 0o440)
        os.replace(primary_path, displaced)
        os.replace(replacement, primary_path)
        destination = repo_tmp_path / "canonical-primary.engine"
        expected = {
            "path": lane.repo_relative(destination),
            "bytes": primary_pin["bytes"], "sha256": primary_pin["sha256"],
        }
        with pytest.raises(
            lane.PoseR13CError,
            match="worker primary changed|mutated during held-FD read|name/inode metadata differs",
        ):
            lane._publish_primary_from_worker_fd(tree, held, expected)
        assert not destination.exists()
    finally:
        if held is not None:
            held.close()
        tree.close()


def test_open_held_source_rejects_same_size_mutation_during_hash(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    source = repo_tmp_path / "hash-race.bin"
    source.write_bytes(b"A" * (4 * 1024 * 1024 + 4096))
    writer = os.open(source, os.O_WRONLY)
    real_read = lane.os.read
    injected = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal injected
        block = real_read(descriptor, size)
        if block and descriptor != writer and not injected:
            injected = True
            os.pwrite(writer, b"B" * len(block), 0)
            os.fsync(writer)
        return block

    monkeypatch.setattr(lane.os, "read", mutate_after_read)
    try:
        with pytest.raises(lane.PoseR13CError, match="mutated during held-FD read"):
            lane.open_held_source(source)
    finally:
        os.close(writer)
    assert injected is True


def test_read_held_json_rejects_same_size_mutation_during_pread(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    source = repo_tmp_path / "json-race.json"
    original = b'{"value":"AAAA"}\n'
    replacement = b'{"value":"BBBB"}\n'
    assert len(original) == len(replacement)
    source.write_bytes(original)
    descriptor, pin = lane.open_held_source(source)
    writer = os.open(source, os.O_WRONLY)
    real_pread = lane.os.pread
    injected = False

    def mutate_after_pread(fd: int, size: int, offset: int) -> bytes:
        nonlocal injected
        block = real_pread(fd, size, offset)
        if fd == descriptor and block and not injected:
            injected = True
            os.pwrite(writer, replacement, 0)
            os.fsync(writer)
        return block

    monkeypatch.setattr(lane.os, "pread", mutate_after_pread)
    try:
        with pytest.raises(lane.PoseR13CError, match="mutated during held-FD read"):
            lane.read_held_json(
                descriptor, pin, source=lane.repo_relative(source), named_path=source,
            )
    finally:
        os.close(writer)
        os.close(descriptor)
    assert injected is True


def test_publish_held_fd_rejects_same_size_mutation_without_destination(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    source = repo_tmp_path / "publication-source.bin"
    destination = repo_tmp_path / "publication-target.bin"
    payload = b"accepted-publication" * 1024
    source.write_bytes(payload)
    descriptor, source_pin = lane.open_held_source(source)
    writer = os.open(source, os.O_WRONLY)
    expected = {
        "path": lane.repo_relative(destination),
        "bytes": source_pin["bytes"],
        "sha256": source_pin["sha256"],
    }
    real_read = lane.os.read
    injected = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal injected
        block = real_read(fd, size)
        if fd == descriptor and block and not injected:
            injected = True
            os.pwrite(writer, b"X" * len(payload), 0)
            os.fsync(writer)
        return block

    monkeypatch.setattr(lane.os, "read", mutate_after_read)
    try:
        with pytest.raises(lane.PoseR13CError, match="mutated during held-FD read"):
            lane.publish_held_fd(
                descriptor, destination, expected, allow_existing_exact=False,
            )
    finally:
        os.close(writer)
        os.close(descriptor)
    assert injected is True
    assert not destination.exists()


def _canonical_publication_test_destination(root: Path, kind: str) -> Path:
    suffixes = {
        "primary": Path("engines/640/model.engine"),
        "receipt": Path("receipts/tensorrt_fp16_640-receipt.json"),
        "commit": Path("commits/tensorrt_fp16_640-commit.json"),
    }
    return root / suffixes[kind]


@pytest.mark.parametrize("kind", ["primary", "receipt", "commit"])
def test_canonical_publication_rejects_intermediate_symlink_parent(
    repo_tmp_path: Path, kind: str,
) -> None:
    source = repo_tmp_path / f"{kind}-symlink-source.bin"
    source.write_bytes((f"accepted-{kind}-symlink".encode()) * 128)
    descriptor, source_pin = lane.open_held_source(source)
    anchor = repo_tmp_path / f"{kind}-symlink-anchor"
    outside = repo_tmp_path / f"{kind}-symlink-outside"
    anchor.mkdir()
    outside.mkdir()
    (anchor / "canonical").symlink_to(outside, target_is_directory=True)
    destination = _canonical_publication_test_destination(
        anchor / "canonical", kind,
    )
    expected = {
        "path": lane.repo_relative(destination),
        "bytes": source_pin["bytes"], "sha256": source_pin["sha256"],
    }
    try:
        with pytest.raises(lane.PoseR13CError, match="no-follow directory"):
            lane.publish_held_fd(
                descriptor, destination, expected, allow_existing_exact=False,
            )
    finally:
        os.close(descriptor)
    assert not any(outside.rglob("*"))


@pytest.mark.parametrize("kind", ["primary", "receipt", "commit"])
def test_canonical_publication_rejects_held_parent_inode_swap_before_link(
    monkeypatch, repo_tmp_path: Path, kind: str,
) -> None:
    source = repo_tmp_path / f"{kind}-inode-source.bin"
    source.write_bytes((f"accepted-{kind}-inode".encode()) * 128)
    descriptor, source_pin = lane.open_held_source(source)
    anchor = repo_tmp_path / f"{kind}-inode-anchor"
    canonical = anchor / "canonical"
    destination = _canonical_publication_test_destination(canonical, kind)
    destination.parent.mkdir(parents=True)
    displaced = anchor / "canonical.displaced"
    expected = {
        "path": lane.repo_relative(destination),
        "bytes": source_pin["bytes"], "sha256": source_pin["sha256"],
    }
    injected = False
    assertion_calls = 0
    original_assert_bound = lane.HeldDirectoryChain.assert_bound

    def swap_parent_after_final_prelink_check(
        self: lane.HeldDirectoryChain,
    ) -> None:
        nonlocal assertion_calls, injected
        original_assert_bound(self)
        if self.path == destination.parent:
            assertion_calls += 1
        if assertion_calls == 3 and not injected:
            injected = True
            os.replace(canonical, displaced)
            destination.parent.mkdir(parents=True)

    monkeypatch.setattr(
        lane.HeldDirectoryChain, "assert_bound",
        swap_parent_after_final_prelink_check,
    )
    try:
        with pytest.raises(
            lane.PoseR13CError,
            match="held directory-chain name/inode binding differs",
        ):
            lane.publish_held_fd(
                descriptor, destination, expected, allow_existing_exact=False,
            )
    finally:
        os.close(descriptor)
    assert injected is True
    assert not destination.exists()
    assert not _canonical_publication_test_destination(displaced, kind).exists()


def test_snapshot_frame_copy_rejects_named_inode_swap(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    source = repo_tmp_path / "frame-source.bin"
    replacement = repo_tmp_path / "frame-replacement.bin"
    displaced = repo_tmp_path / "frame-displaced.bin"
    target = repo_tmp_path / "frame-target.bin"
    payload = b"accepted-frame" * 1024
    source.write_bytes(payload)
    replacement.write_bytes(b"forged---frame" * 1024)
    assert source.stat().st_size == replacement.stat().st_size
    descriptor, pin = lane.open_held_source(source)
    target_descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    real_read = lane.os.read
    injected = False

    def swap_name_after_read(fd: int, size: int) -> bytes:
        nonlocal injected
        block = real_read(fd, size)
        if fd == descriptor and block and not injected:
            injected = True
            os.replace(source, displaced)
            os.replace(replacement, source)
        return block

    monkeypatch.setattr(lane.os, "read", swap_name_after_read)
    try:
        with pytest.raises(
            lane.PoseR13CError,
            match="mutated during held-FD read|name/inode metadata differs",
        ):
            lane._copy_held_to_stream(
                descriptor, target_descriptor, pin, hashlib.sha256(),
            )
    finally:
        os.close(target_descriptor)
        os.close(descriptor)
    assert injected is True


def test_incident_document_requires_exact_frozen_metadata(repo_tmp_path: Path) -> None:
    path = repo_tmp_path / "incident.json"
    value = {"kind": "unit-incident"}
    value["fingerprint_sha256"] = lane.fingerprint(value)
    path.write_bytes(lane.canonical_bytes(value))
    os.chmod(path, 0o440)
    expected = lane.document_pin(path)
    assert lane._read_exact_incident_document(
        path, expected, context="unit frozen incident",
    ) == value

    os.chmod(path, 0o640)
    with pytest.raises(lane.PoseR13CError, match="immutability metadata differs"):
        lane._read_exact_incident_document(
            path, expected, context="unit frozen incident",
        )

    os.chmod(path, 0o440)
    os.link(path, repo_tmp_path / "incident-hardlink.json")
    with pytest.raises(lane.PoseR13CError, match="singleton"):
        lane._read_exact_incident_document(
            path, expected, context="unit frozen incident",
        )


@pytest.mark.parametrize(
    ("target", "field", "forged"),
    [
        ("receipt", "status", "failed"),
        ("receipt", "stage", "tensorrt_fp16_960"),
        ("receipt", "profile", 960),
        ("receipt", "run_id", "pose-r13c-forged"),
        ("receipt", "plan", {"path": "forged-plan"}),
        ("receipt", "snapshot_manifest", {"path": "forged-manifest"}),
        ("receipt", "worker_result", {"path": "forged-worker"}),
        ("receipt", "primary_artifact", {"path": "forged-primary"}),
        ("receipt", "predecessor_commits", [{"path": "forged-predecessor"}]),
        ("commit", "snapshot_manifest", {"path": "forged-manifest"}),
        ("commit", "worker_result", {"path": "forged-worker"}),
        ("commit", "primary_artifact", {"path": "forged-primary"}),
        ("commit", "predecessor_commits", [{"path": "forged-predecessor"}]),
    ],
)
def test_commit_receipt_projection_rejects_each_forged_semantic_field(
    target: str, field: str, forged: object,
) -> None:
    stage = "tensorrt_fp16_640"
    run_id = "pose-r13c-projection-unit"
    plan = {
        "path": "models/pose/challengers/plan-r13c.json",
        "bytes": 100,
        "sha256": "a" * 64,
        "fingerprint_sha256": "b" * 64,
    }
    manifest = {
        "path": "validation/results/pose/runs/manifest-r13c.json",
        "bytes": 101,
        "sha256": "c" * 64,
        "fingerprint_sha256": "d" * 64,
    }
    worker = {
        "path": "validation/results/pose/runs/worker-result.json",
        "bytes": 102,
        "sha256": "e" * 64,
        "fingerprint_sha256": "f" * 64,
    }
    primary = {
        "path": "validation/results/pose/engine.bin",
        "bytes": 103,
        "sha256": "1" * 64,
    }
    commit = {
        "snapshot_manifest": manifest,
        "worker_result": worker,
        "primary_artifact": primary,
        "predecessor_commits": [],
    }
    receipt = {
        "status": "passed",
        "stage": stage,
        "profile": 640,
        "run_id": run_id,
        "plan": plan,
        "snapshot_manifest": manifest,
        "worker_result": worker,
        "primary_artifact": primary,
        "predecessor_commits": [],
    }
    lane._validate_commit_receipt_projection(
        commit, receipt, stage=stage, profile=640, run_id=run_id,
        expected_plan=plan,
    )
    changed_commit = copy.deepcopy(commit)
    changed_receipt = copy.deepcopy(receipt)
    changed = changed_commit if target == "commit" else changed_receipt
    changed[field] = forged
    with pytest.raises(lane.PoseR13CError, match="exact semantic projection differs"):
        lane._validate_commit_receipt_projection(
            changed_commit, changed_receipt, stage=stage, profile=640,
            run_id=run_id, expected_plan=plan,
        )


@pytest.mark.parametrize(
    "field",
    ["snapshot_manifest", "execution_receipt", "worker_result", "primary_artifact"],
)
def test_commit_canonical_projection_rejects_each_redirected_path(field: str) -> None:
    stage = "tensorrt_fp16_640"
    run_id = "pose-r13c-canonical-unit"
    run_root = lane.RUNS_ROOT / stage / run_id
    commit = {
        "snapshot_manifest": {
            "path": lane.repo_relative(run_root / "snapshot/manifest-r13c.json"),
        },
        "execution_receipt": {"path": lane.repo_relative(lane.receipt_path(stage))},
        "worker_result": {
            "path": lane.repo_relative(run_root / "output/worker-result.json"),
        },
        "primary_artifact": {"path": lane.repo_relative(lane.primary_path(stage))},
    }
    assert lane._validate_commit_canonical_paths(
        commit, stage=stage, run_id=run_id,
    ) == (
        run_root,
        run_root / "snapshot/manifest-r13c.json",
        run_root / "output/worker-result.json",
    )
    changed = copy.deepcopy(commit)
    changed[field]["path"] = "validation/results/pose/redirected"
    with pytest.raises(lane.PoseR13CError, match="canonical path differs"):
        lane._validate_commit_canonical_paths(
            changed, stage=stage, run_id=run_id,
        )


def test_commit_chain_rejects_noncanonical_top_level_commit_before_open(
    repo_tmp_path: Path,
) -> None:
    with pytest.raises(lane.PoseR13CError, match="commit canonical path differs"):
        lane._validate_commit_chain(
            repo_tmp_path / "redirected-commit.json", "a" * 64, {},
            repo_tmp_path / "plan.json",
            {
                "path": lane.repo_relative(repo_tmp_path / "plan.json"),
                "bytes": 1, "sha256": "b" * 64,
                "fingerprint_sha256": "c" * 64,
            },
            "tensorrt_fp16_640",
        )


@pytest.mark.parametrize("forged_field", ["bytes", "sha256"])
def test_recursive_commit_rejects_forged_claimed_pin_against_observed_pin(
    monkeypatch, repo_tmp_path: Path, forged_field: str,
) -> None:
    runs_root = repo_tmp_path / "chain-runs"
    commits_root = repo_tmp_path / "chain-commits"
    receipts_root = repo_tmp_path / "chain-receipts"
    primaries_root = repo_tmp_path / "chain-primaries"
    plan_path = repo_tmp_path / "chain-plan.json"
    plan_path.write_bytes(b'{"plan":"accepted"}\n')
    plan = {"fingerprint_sha256": "9" * 64}
    expected_plan = {
        **lane.file_pin(plan_path),
        "fingerprint_sha256": plan["fingerprint_sha256"],
    }
    monkeypatch.setattr(lane, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(
        lane, "commit_path", lambda stage: commits_root / f"{stage}.json",
    )
    monkeypatch.setattr(
        lane, "receipt_path", lambda stage: receipts_root / f"{stage}.json",
    )
    monkeypatch.setattr(
        lane, "primary_path", lambda stage: primaries_root / f"{stage}.bin",
    )
    monkeypatch.setattr(lane, "validate_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lane, "_verify_file_pin", lambda *_args, **_kwargs: None)

    documents: dict[str, tuple[Path, dict[str, object]]] = {
        expected_plan["path"]: (plan_path, plan),
    }
    counter = 1

    def document_pin(path: Path) -> dict[str, object]:
        nonlocal counter
        token = format(counter, "x")[-1]
        counter += 1
        return {
            "path": lane.repo_relative(path), "bytes": counter,
            "sha256": token * 64,
            "fingerprint_sha256": format(counter, "x")[-1] * 64,
        }

    def file_pin(path: Path) -> dict[str, object]:
        nonlocal counter
        token = format(counter, "x")[-1]
        counter += 1
        return {
            "path": lane.repo_relative(path), "bytes": counter,
            "sha256": token * 64,
        }

    def commit_value(
        stage: str, run_id: str, predecessors: list[dict[str, object]],
    ) -> dict[str, object]:
        profile = lane.stage_profile(stage)
        run_root = runs_root / stage / run_id
        manifest_path = run_root / "snapshot/manifest-r13c.json"
        worker_path = run_root / "output/worker-result.json"
        manifest_pin = document_pin(manifest_path)
        receipt_pin = document_pin(lane.receipt_path(stage))
        worker_pin = document_pin(worker_path)
        primary_pin = file_pin(lane.primary_path(stage))
        stream_pin = file_pin(run_root / "snapshot/snapshot-r13c.stream")
        manifest = {
            "stage": stage, "profile": profile, "run_id": run_id,
            "plan": expected_plan,
        }
        operation = "build" if stage.startswith("tensorrt_fp16_") else "infer"
        worker = {
            "status": "passed", "profile": profile, "operation": operation,
            "plan_fingerprint_sha256": plan["fingerprint_sha256"],
            "outputs": [],
        }
        receipt = {
            "status": "passed", "stage": stage, "profile": profile,
            "run_id": run_id, "plan": expected_plan,
            "snapshot_manifest": manifest_pin, "snapshot_stream": stream_pin,
            "worker_result": worker_pin, "primary_artifact": primary_pin,
            "worker_outputs": [], "predecessor_commits": predecessors,
        }
        documents[str(manifest_pin["path"])] = (manifest_path, manifest)
        documents[str(receipt_pin["path"])] = (lane.receipt_path(stage), receipt)
        documents[str(worker_pin["path"])] = (worker_path, worker)
        return {
            "status": "committed", "stage": stage, "profile": profile,
            "run_id": run_id, "plan": expected_plan,
            "snapshot_manifest": manifest_pin, "execution_receipt": receipt_pin,
            "worker_result": worker_pin, "primary_artifact": primary_pin,
            "predecessor_commits": predecessors,
        }

    def write_commit(stage: str, value: dict[str, object]) -> dict[str, object]:
        value["fingerprint_sha256"] = lane.fingerprint(value)
        path = lane.commit_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(lane.canonical_bytes(value))
        return lane.document_pin(path)

    nested_stage = "tensorrt_fp16_640"
    nested = commit_value(
        nested_stage, "pose-r13c-recursive-nested-unit", [],
    )
    observed_nested_pin = write_commit(nested_stage, nested)
    claimed_nested_pin = copy.deepcopy(observed_nested_pin)
    if forged_field == "bytes":
        claimed_nested_pin["bytes"] = int(claimed_nested_pin["bytes"]) + 1
    else:
        claimed_nested_pin["sha256"] = "0" * 64

    top_stage = "numerical_parity_640"
    top = commit_value(
        top_stage, "pose-r13c-recursive-top-unit", [claimed_nested_pin],
    )
    top_pin = write_commit(top_stage, top)

    for stage, run_id in (
        (nested_stage, "pose-r13c-recursive-nested-unit"),
        (top_stage, "pose-r13c-recursive-top-unit"),
    ):
        tree = lane._new_run_tree(stage, run_id)
        os.fchmod(tree.snapshot_descriptor, 0o550)
        tree.capture()
        tree.close()

    def read_document(
        pin: dict[str, object], _schema: Path,
    ) -> tuple[Path, dict[str, object]]:
        return documents[str(pin["path"])]

    monkeypatch.setattr(lane, "_read_pinned_document", read_document)

    def read_document_at(
        _directory_descriptor: int, _name: str, _logical_path: Path,
        pin: dict[str, object], _schema: Path,
    ) -> dict[str, object]:
        return documents[str(pin["path"])][1]

    monkeypatch.setattr(lane, "_read_pinned_document_at", read_document_at)
    monkeypatch.setattr(lane, "_verify_file_pin_at", lambda *_args, **_kwargs: {})
    with pytest.raises(
        lane.PoseR13CError,
        match="recursive predecessor claimed/observed pin differs",
    ):
        lane._validate_commit_chain(
            lane.commit_path(top_stage), str(top_pin["fingerprint_sha256"]),
            plan, plan_path, expected_plan, top_stage,
        )


def test_build_stage_snapshot_stream_is_replayable_and_command_has_no_input_bind(monkeypatch) -> None:
    context, temporary = _temporary_repo_directory()
    old_runs = lane.RUNS_ROOT
    try:
        monkeypatch.setattr(lane, "RUNS_ROOT", temporary / "results/runs")
        plan_path = temporary / "plan.json"
        plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
        expected_plan_pin = lane.document_pin(plan_path)
        result = lane.prepare_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13c-unit-build-640",
            accepted_commits={}, prepared_at_utc=STAMP,
        )
        assert result["executed"] is False and result["gpu"] is False and result["docker"] is False
        assert result["execution_authorized"] is False
        assert result["runtime_binding_block_reason"] == lane.RUNTIME_BINDING_BLOCK_REASON
        assert result["stdin_stream"]["bytes"] > lane.r13.UPSTREAM[640]["onnx"][1]
        command = result["docker_argv"]
        assert command[:6] == ["docker", "run", "--rm", "-i", "--pull=never", "--network=none"]
        assert "--entrypoint=python3" in command
        assert not any("/proc/" in item for item in command)
        mounts = [item for item in command if item.startswith("--mount=")]
        assert len(mounts) == 1 and "dst=/output" in mounts[0]
        assert any(item.startswith("--tmpfs=/inputs:") for item in command)
        assert any(item.startswith("--tmpfs=/work:") for item in command)
        assert any(item.startswith("--tmpfs=/workspace:") for item in command)
        assert result["lease_argv"] is None
        cidfiles = [item for item in command if item.startswith("--cidfile=")]
        assert len(cidfiles) == 1 and Path(cidfiles[0].split("=", 1)[1]).is_absolute()
        assert stat.S_IMODE(Path(cidfiles[0].split("=", 1)[1]).parent.lstat().st_mode) == 0o700
        with pytest.raises(lane.PoseR13CError, match="legacy v1 lease command is not authorized"):
            lane.render_lease_command(command, timeout_seconds=14460)
        manifest_fp = result["manifest"]["fingerprint_sha256"]
        verified_plan, manifest, predecessors, stream_fd, stream_pin, tree = lane.verify_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13c-unit-build-640",
            accepted_manifest_fingerprint=manifest_fp, accepted_commits={},
        )
        try:
            assert verified_plan == plan
            assert predecessors == []
            assert manifest["worker_argv"][2].endswith("r13c_container.py")
            assert manifest["stream_contract"][
                "host_stream_fd_metadata_stable_before_after_validation"
            ] is True
            assert manifest["stream_contract"][
                "host_plan_fd_held_through_stream_commit"
            ] is True
            assert manifest["stream_contract"][
                "plan_entry_exact_held_verified_plan"
            ] is True
            assert manifest["stream_contract"][
                "host_run_tree_root_dirfd_openat_nofollow"
            ] is True
            assert manifest["stream_contract"][
                "host_run_tree_ancestors_held_identity_bound"
            ] is True
            assert manifest["stream_contract"]["execution_authorized"] is False
            assert manifest["stream_contract"][
                "runtime_binding_block_reason"
            ] == lane.RUNTIME_BINDING_BLOCK_REASON
            assert manifest["copy_contract"][
                "source_metadata_stable_before_after_frame_copy"
            ] is True
            assert manifest["copy_contract"][
                "source_named_inode_stable_before_after_frame_copy"
            ] is True
            assert {item["role"] for item in manifest["entries"]} >= {
                "plan", "container_worker", "r13_worker_library",
                "shared_runtime_worker", "validation_init", "onnx",
            }
            assert manifest["plan"] == expected_plan_pin
            plan_entries = [
                item for item in manifest["entries"] if item["role"] == "plan"
            ]
            assert len(plan_entries) == 1
            assert plan_entries[0]["source"] == {
                key: expected_plan_pin[key] for key in ("path", "bytes", "sha256")
            }
            assert stream_pin == result["stream"]
        finally:
            os.close(stream_fd)
            tree.close()
    finally:
        monkeypatch.setattr(lane, "RUNS_ROOT", old_runs)
        snapshot = temporary / "results/runs/tensorrt_fp16_640/pose-r13c-unit-build-640/snapshot"
        if snapshot.exists():
            os.chmod(snapshot, 0o700)
        context.cleanup()


def test_stream_validation_rejects_one_byte_corruption_before_any_launcher(monkeypatch) -> None:
    context, temporary = _temporary_repo_directory()
    old_runs = lane.RUNS_ROOT
    try:
        monkeypatch.setattr(lane, "RUNS_ROOT", temporary / "results/runs")
        plan_path = temporary / "plan.json"
        plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
        result = lane.prepare_stage(
            plan_path=plan_path, accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13c-corrupt-build-640",
            accepted_commits={}, prepared_at_utc=STAMP,
        )
        stream_path = lane.ROOT / result["stream"]["path"]
        os.chmod(stream_path, 0o640)
        with stream_path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            last = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([last[0] ^ 1]))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(stream_path, 0o440)
        with pytest.raises(Exception, match="stream|pin|digest|drift"):
            lane.verify_stage(
                plan_path=plan_path, accepted_plan_fingerprint=plan["fingerprint_sha256"],
                stage="tensorrt_fp16_640", run_id="pose-r13c-corrupt-build-640",
                accepted_manifest_fingerprint=result["manifest"]["fingerprint_sha256"],
                accepted_commits={},
            )
    finally:
        monkeypatch.setattr(lane, "RUNS_ROOT", old_runs)
        snapshot = temporary / "results/runs/tensorrt_fp16_640/pose-r13c-corrupt-build-640/snapshot"
        if snapshot.exists():
            os.chmod(snapshot, 0o700)
        context.cleanup()


def test_inline_loader_is_syntax_valid_and_hashes_whole_stream_before_compile() -> None:
    compile(lane.INLINE_LOADER, "<pose-r13c-inline-loader-test>", "exec")
    source = lane.INLINE_LOADER
    assert source.index("digest.hexdigest()!=expected_sha") < source.index("compile(source")
    assert "sys.stdin.buffer.read" in source
    assert "whole snapshot stream pin differs" in source
    assert "/proc/" not in source


def test_bootstrap_manual_contract_rejects_unknown_field_before_copy(repo_tmp_path: Path) -> None:
    plan_path = repo_tmp_path / "plan-r13c.json"
    plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
    plan_pin = lane.document_pin(plan_path)
    specs = lane._source_specs(
        plan=plan, plan_path=plan_path, stage="tensorrt_fp16_640", predecessors=[],
        plan_pin=plan_pin,
    )
    manifest = lane.build_snapshot_manifest(
        plan=plan, plan_path=plan_path,
        stage="tensorrt_fp16_640", run_id="pose-r13c-bootstrap-unit",
        specs=specs, prepared_at_utc=STAMP, plan_pin=plan_pin,
    )
    manifest["silent_fallback"] = True
    manifest["fingerprint_sha256"] = bootstrap.fingerprint(manifest)
    with pytest.raises(bootstrap.SnapshotBootstrapR13CError, match="top-level fields"):
        bootstrap.validate_manifest(
            manifest, expected_fingerprint=manifest["fingerprint_sha256"],
            expected_plan_fingerprint=plan["fingerprint_sha256"],
            expected_bootstrap_bytes=plan["implementation"]["snapshot_bootstrap"]["bytes"],
            expected_bootstrap_sha256=plan["implementation"]["snapshot_bootstrap"]["sha256"],
        )


def test_private_worker_never_generates_or_executes_from_output_bind() -> None:
    source = lane.R13C_WORKER.read_text(encoding="utf-8")
    forbidden = [
        '/output/engine.staging', '/output/trtexec-build.log',
        '/output/trt-output.npz', '/output/pose-gst',
        'build_dir = Path("/output', 'probe = Path("/output',
    ]
    assert not any(item in source for item in forbidden)
    assert 'WORK_ROOT = Path("/work")' in source
    assert "os.O_EXCL" in source and 'getattr(os, "O_NOFOLLOW", 0)' in source
    assert source.index("outputs = _publish_all") < source.index('legacy.atomic_json(OUTPUT_ROOT / "worker-result.json"')


def test_private_final_copy_is_no_overwrite_and_rehashes(tmp_path: Path) -> None:
    source = tmp_path / "private.engine"
    destination = tmp_path / "output.engine"
    source.write_bytes(b"verified-private-engine")
    os.chmod(source, 0o440)
    published = worker._copy_verified(source, destination)
    assert published["sha256"] == hashlib.sha256(b"verified-private-engine").hexdigest()
    assert destination.read_bytes() == b"verified-private-engine"
    with pytest.raises(FileExistsError):
        worker._copy_verified(source, destination)


def test_input_engine_is_exact_copied_to_private_work_before_runtime(
    monkeypatch, tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    work = tmp_path / "work"
    inputs.mkdir()
    work.mkdir()
    source = inputs / "model.engine"
    payload = b"accepted-engine-serialization"
    source.write_bytes(payload)
    os.chmod(source, 0o440)
    monkeypatch.setattr(worker, "WORK_ROOT", work)
    destination = work / "model.engine"
    pin = worker._private_copy_exact(source, hashlib.sha256(payload).hexdigest(), destination)
    assert pin["sha256"] == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload
    assert stat.S_IMODE(destination.lstat().st_mode) == 0o440
    with pytest.raises(FileExistsError):
        worker._private_copy_exact(source, hashlib.sha256(payload).hexdigest(), destination)


def test_generated_nvinfer_config_pins_only_private_engine(
    monkeypatch, tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    work = tmp_path / "work"
    inputs.mkdir()
    work.mkdir()
    source = inputs / "nvinfer-config.txt"
    raw = b"[property]\nmodel-engine-file=/inputs/model.engine\nbatch-size=12\n"
    source.write_bytes(raw)
    os.chmod(source, 0o440)
    monkeypatch.setattr(worker, "WORK_ROOT", work)
    generated, pin = worker._private_nvinfer_config(source, hashlib.sha256(raw).hexdigest())
    payload = generated.read_bytes()
    assert b"model-engine-file=/work/model.engine" in payload
    assert b"/inputs/" not in payload
    assert pin["sha256"] == hashlib.sha256(payload).hexdigest()
    assert stat.S_IMODE(generated.lstat().st_mode) == 0o440


def test_generated_nvinfer_config_rejects_ambiguous_engine_directives(
    monkeypatch, tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    work = tmp_path / "work"
    inputs.mkdir()
    work.mkdir()
    source = inputs / "nvinfer-config.txt"
    raw = b"model-engine-file=/inputs/model.engine\nmodel-engine-file=/inputs/model.engine\n"
    source.write_bytes(raw)
    os.chmod(source, 0o440)
    monkeypatch.setattr(worker, "WORK_ROOT", work)
    with pytest.raises(worker.PoseWorkerR13CError, match="engine path contract"):
        worker._private_nvinfer_config(source, hashlib.sha256(raw).hexdigest())
    assert not (work / "nvinfer-config.txt").exists()


def test_infer_and_probe_runtime_calls_never_receive_input_engine_or_config() -> None:
    source = lane.R13C_WORKER.read_text(encoding="utf-8")
    infer = source[source.index("def infer_operation"):source.index("def bridge_operation")]
    bridge = source[source.index("def bridge_operation"):source.index("def parser")]
    assert "runtime.load_engine(args.engine)" not in infer
    assert "runtime.load_engine(private_engine)" in infer
    assert "_private_copy_exact(args.engine, args.engine_sha256, private_engine)" in infer
    assert "[str(probe), str(args.profile), str(batch), str(args.config)]" not in bridge
    assert "[str(probe), str(args.profile), str(batch), str(private_config)]" in bridge
    assert "_private_copy_exact(args.engine, args.engine_sha256, private_engine)" in bridge
    assert "_private_nvinfer_config(" in bridge


def test_launcher_drain_failure_kills_and_reaps_before_handler_restore(
    monkeypatch, repo_tmp_path: Path,
) -> None:
    stream = repo_tmp_path / "snapshot.stream"
    stream.write_bytes(b"sealed-input")
    stream_pin = lane.file_pin(stream)
    stream_fd = os.open(stream, os.O_RDONLY)
    log = repo_tmp_path / "launcher.log"
    managed = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {item: signal.getsignal(item) for item in managed}
    previous_mask = set(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
    events: list[str] = []
    children: list[subprocess.Popen[bytes]] = []
    real_popen = lane.subprocess.Popen
    real_signal = lane.signal.signal
    real_write = lane.os.write
    real_terminate = lane._terminate

    def capture_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def capture_signal(signum, handler):
        if signum in previous_handlers and handler == previous_handlers[signum]:
            events.append("handler_restore")
        return real_signal(signum, handler)

    def fail_log_write(descriptor: int, payload: bytes) -> int:
        if payload:
            events.append("drain_error")
            raise OSError("injected launcher log sink failure")
        return real_write(descriptor, payload)

    def capture_terminate(child: subprocess.Popen[bytes]) -> None:
        events.append("terminate_enter")
        real_terminate(child)
        events.append("terminate_done")

    monkeypatch.setattr(lane.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(lane.signal, "signal", capture_signal)
    monkeypatch.setattr(lane.os, "write", fail_log_write)
    monkeypatch.setattr(lane, "_terminate", capture_terminate)
    try:
        with pytest.raises(lane.PoseR13CError, match="output reader failed"):
            lane._run_with_snapshot_stdin(
                [sys.executable, "-c", "import os,time;os.write(1,b'x'*65536);time.sleep(30)"],
                stream_fd=stream_fd, log_path=log, timeout_seconds=5,
                stream_pin=stream_pin, stream_path=stream,
            )
    finally:
        os.close(stream_fd)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert not lane._process_group_exists(children[0].pid)
    assert "drain_error" in events and "terminate_done" in events and "handler_restore" in events
    assert events.index("terminate_done") < events.index("handler_restore")
    assert {item: signal.getsignal(item) for item in managed} == previous_handlers
    assert set(signal.pthread_sigmask(signal.SIG_BLOCK, set())) == previous_mask


def test_private_worker_rejects_direct_entry_without_bootstrap_acceptance(monkeypatch) -> None:
    for key in (
        "DEEPSAFE_POSE_R13C_SNAPSHOT_ACCEPTED",
        "DEEPSAFE_POSE_R13C_SNAPSHOT_RECEIPT",
        "DEEPSAFE_POSE_R13C_SNAPSHOT_RECEIPT_SHA256",
        "DEEPSAFE_POSE_R13C_SNAPSHOT_FINGERPRINT",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(worker.PoseWorkerR13CError, match="acceptance is missing"):
        worker.validate_snapshot_acceptance("a" * 64)


def test_worker_runtime_arguments_must_match_exact_accepted_input_destinations() -> None:
    engine_sha = "a" * 64
    seed_sha = "b" * 64
    args = worker.parser().parse_args([
        "infer", "--profile", "640", "--plan-fingerprint", "c" * 64,
        "--engine", "/inputs/model.engine", "--engine-sha256", engine_sha,
        "--seed", "/inputs/seed.npy", "--seed-sha256", seed_sha,
    ])
    acceptance = {
        "entries": [
            {"role": "engine", "destination": "/inputs/model.engine", "sha256": engine_sha},
            {"role": "seed", "destination": "/inputs/seed.npy", "sha256": seed_sha},
        ]
    }
    worker.validate_accepted_runtime_arguments(args, acceptance)
    changed = copy.copy(args)
    changed.engine = Path("/work/model.engine")
    with pytest.raises(worker.PoseWorkerR13CError, match="destination differs"):
        worker.validate_accepted_runtime_arguments(changed, acceptance)


def test_private_worker_accepts_exact_bootstrap_receipt_with_negative_boundary_flags(
    monkeypatch, tmp_path: Path,
) -> None:
    plan_fingerprint = "a" * 64
    manifest_fingerprint = "b" * 64
    value = {
        "schema_version": "deepsafe.pose-mmpose-yoloxpose-snapshot-acceptance/r13c",
        "status": "accepted_before_worker_execution",
        "accepted_at_utc": STAMP,
        "stream": {"bytes": 10, "sha256": "c" * 64},
        "manifest_fingerprint_sha256": manifest_fingerprint,
        "plan_fingerprint_sha256": plan_fingerprint,
        "bootstrap": {"bytes": 5, "sha256": "d" * 64},
        "entries": [
            {"role": role, "destination": f"/workspace/{role}", "bytes": 1, "sha256": "e" * 64}
            for role in ("container_worker", "r13_worker_library", "shared_runtime_worker", "validation_init")
        ],
        "checks": {
            "whole_stream_preverified_by_inline_loader": True,
            "whole_stream_reopened_and_rehashed": True,
            "strict_manifest_json": True,
            "manifest_self_fingerprint": True,
            "bootstrap_frame_reverified": True,
            "all_frames_exact": True,
            "tmpfs_copy": True,
            "exclusive_destination_create": True,
            "post_copy_reopen_hash": True,
            "trailing_data_absent": True,
            "gpu_modules_imported": False,
            "subprocess_executed": False,
        },
    }
    value["fingerprint_sha256"] = worker.legacy.fingerprint(value)
    payload = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    path = tmp_path / "snapshot-acceptance-r13c.json"
    path.write_bytes(payload)
    os.chmod(path, 0o440)
    monkeypatch.setattr(worker, "ACCEPTANCE_PATH", path)
    monkeypatch.setenv("DEEPSAFE_POSE_R13C_SNAPSHOT_ACCEPTED", "1")
    monkeypatch.setenv("DEEPSAFE_POSE_R13C_SNAPSHOT_RECEIPT", str(path))
    monkeypatch.setenv("DEEPSAFE_POSE_R13C_SNAPSHOT_RECEIPT_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("DEEPSAFE_POSE_R13C_SNAPSHOT_FINGERPRINT", manifest_fingerprint)
    observed = worker.validate_snapshot_acceptance(plan_fingerprint)
    assert observed["fingerprint_sha256"] == value["fingerprint_sha256"]


def test_snapshot_schema_rejects_proc_transport_and_worker_fallback(repo_tmp_path: Path) -> None:
    plan_path = repo_tmp_path / "plan-r13c.json"
    plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
    plan_pin = lane.document_pin(plan_path)
    specs = lane._source_specs(
        plan=plan, plan_path=plan_path, stage="tensorrt_fp16_640", predecessors=[],
        plan_pin=plan_pin,
    )
    manifest = lane.build_snapshot_manifest(
        plan=plan, plan_path=plan_path,
        stage="tensorrt_fp16_640", run_id="pose-r13c-schema-unit",
        specs=specs, prepared_at_utc=STAMP, plan_pin=plan_pin,
    )
    changed = copy.deepcopy(manifest)
    changed["stream_contract"]["proc_fd_paths_in_argv"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13CError, match="schema validation failed"):
        lane.validate_schema(changed, lane.SNAPSHOT_SCHEMA)


def test_receipt_and_commit_builders_bind_snapshot_private_runtime_and_margins(repo_tmp_path: Path) -> None:
    plan_path = repo_tmp_path / "plan-r13c.json"
    plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
    document = {
        "path": "validation/results/unit.json", "bytes": 7,
        "sha256": "a" * 64, "fingerprint_sha256": "b" * 64,
    }
    artifact = {
        "path": "validation/results/unit.bin", "bytes": 11,
        "sha256": "c" * 64,
    }
    lifecycle = {
        "start_new_session": True,
        "timeout_seconds": lane.LAUNCHER_TIMEOUT["tensorrt_fp16_640"],
        "duration_seconds": 1.0,
        "stdin_held_snapshot_fd": True,
        "output_limit_bytes": lane.MAX_HOST_LOG_BYTES,
        "output_bytes": 12,
        "orphan_survived": False,
    }
    receipt = lane.build_receipt(
        plan=plan, plan_path=plan_path, manifest_pin=document,
        stream_pin=artifact, stage="tensorrt_fp16_640",
        run_id="pose-r13c-builder-unit", worker_pin=document,
        worker_outputs=[artifact], primary_published=artifact,
        predecessors=[], numerical=None, lifecycle=lifecycle,
        created_at_utc=STAMP,
    )
    assert receipt["execution"]["gpu_lease_timeout_seconds"] == 14460
    assert receipt["execution"]["launcher_timeout_seconds"] == 14520
    assert receipt["execution"]["private_work_tmpfs_for_build_deserialize_compile_execute_and_parse"] is True
    assert receipt["execution"]["input_engine_o_excl_copied_and_rehashed_to_private_work"] is False
    assert receipt["execution"]["runtime_engine_path"] is None
    assert receipt["execution"]["generated_nvinfer_config_path"] is None
    assert receipt["execution"]["input_engine_or_config_passed_to_runtime"] is False
    assert receipt["execution"]["managed_signal_handlers_cover_cleanup_reap_and_drain"] is True
    assert receipt["execution"]["output_drain_failure_live_loop_fail_closed"] is True
    assert receipt["execution"]["output_bind_final_verified_no_overwrite_only"] is True
    assert receipt["execution"]["managed_docker_cidfile_path_handshake"] is True
    assert receipt["execution"]["docker_cidfile_parent_owner_uid_mode_0700"] is True
    commit = lane.build_commit(
        plan=plan, plan_path=plan_path, manifest_pin=document,
        stage="tensorrt_fp16_640", run_id="pose-r13c-builder-unit",
        primary_pin=artifact, receipt_pin=document, worker_pin=document,
        predecessors=[], committed_at_utc=STAMP,
    )
    assert receipt["fingerprint_sha256"] == lane.fingerprint(receipt)
    assert commit["fingerprint_sha256"] == lane.fingerprint(commit)


def test_no_docker_gpu_or_deepstream_was_invoked_by_test_module(monkeypatch) -> None:
    # A final guard for accidental future expansion of the CPU-only suite.
    invoked: list[list[str]] = []

    def forbidden(argv, *args, **kwargs):
        invoked.append(list(argv))
        raise AssertionError("external runtime invocation forbidden")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    plan = lane.build_plan(prepared_at_utc=STAMP)
    assert plan["claim_boundary"]["r13c_gpu_executed"] is False
    assert invoked == []
