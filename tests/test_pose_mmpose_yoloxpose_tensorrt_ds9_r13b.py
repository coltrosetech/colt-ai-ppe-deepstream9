from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from validation import pose_mmpose_yoloxpose_snapshot_bootstrap_r13b as bootstrap
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13b as lane
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13b_container as worker


STAMP = "2026-07-18T13:30:00+03:00"
EVIDENCE_STAMP = "2026-07-18T12:41:46+03:00"


def test_compatibility_evidence_is_exact_cross_lane_pre_container_failure() -> None:
    value = lane.build_compatibility_evidence(observed_at_utc=EVIDENCE_STAMP)
    assert value["person_r14b_observation"]["docker_log"] == lane.PERSON_FAILURE_LOG_PIN
    assert value["person_r14b_observation"]["container_process_started"] is False
    assert value["person_r14b_observation"]["cuda_or_worker_code_reached"] is False
    assert value["pose_r13_static_audit"]["same_failed_primitive_as_person_r14b"] is True
    assert value["pose_r13_static_audit"]["pose_r13_docker_or_gpu_attempted"] is False
    assert value["decision"]["r13_frozen_artifacts_modified"] is False
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
from validation import pose_mmpose_yoloxpose_tensorrt_ds9_r13b as lane
plan = lane.build_plan(prepared_at_utc="2026-07-18T13:30:00+03:00")
assert plan["claim_boundary"]["r13b_gpu_executed"] is False
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


def test_plan_supersedes_frozen_r13_and_pins_stream_private_work_contract() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    assert plan["supersedes"]["plan"] == lane.R13_PLAN_PIN
    assert plan["supersedes"]["r13_artifacts_modified"] is False
    launch = plan["launch_contract"]
    assert launch["docker_interactive_stdin_exact"] is True
    assert launch["proc_fd_paths_in_docker_argv"] is False
    assert launch["container_tmpfs"] == ["/inputs", "/tmp", "/work", "/workspace"]
    assert launch["generated_runtime_artifacts_private_work_tmpfs"] is True
    assert launch["host_output_bind_used_only_for_final_verified_no_overwrite_publication"] is True
    assert launch["gpu_lease_explicit_timeout_required"] is True
    assert launch["gpu_lease_timeout_seconds"]["tensorrt_fp16_640"] == 14460
    assert launch["launcher_timeout_seconds"]["tensorrt_fp16_640"] == 14520
    assert launch["docker_cidfile_absolute_unique_required"] is True
    assert launch["managed_docker_cidfile_path_handshake_required"] is True
    assert launch["docker_cidfile_parent_owner_uid_mode"] == "0700"
    assert launch["inline_loader_sha256"] == hashlib.sha256(lane.INLINE_LOADER.encode()).hexdigest()
    assert plan["implementation"]["container_worker"]["path"].endswith("r13b_container.py")
    assert plan["implementation"]["r13_worker_library"]["path"].endswith("r13_container.py")
    assert plan["fingerprint_sha256"] == lane.fingerprint(plan)


def test_plan_schema_rejects_unknown_or_weakened_launch_property() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    changed = copy.deepcopy(plan)
    changed["launch_contract"]["input_bind_mounts"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13BError, match="schema validation failed"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)
    changed = copy.deepcopy(plan)
    changed["launch_contract"]["silent_fallback"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13BError, match="Additional properties"):
        lane.validate_schema(changed, lane.PLAN_SCHEMA)


def _temporary_repo_directory() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    context = tempfile.TemporaryDirectory(prefix=".pose-r13b-test-", dir=lane.ROOT)
    return context, Path(context.name)


def test_build_stage_snapshot_stream_is_replayable_and_command_has_no_input_bind(monkeypatch) -> None:
    context, temporary = _temporary_repo_directory()
    old_runs = lane.RUNS_ROOT
    old_results = lane.RESULTS_ROOT
    try:
        monkeypatch.setattr(lane, "RESULTS_ROOT", temporary / "results")
        monkeypatch.setattr(lane, "RUNS_ROOT", temporary / "results/runs")
        plan_path = temporary / "plan.json"
        plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
        result = lane.prepare_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13b-unit-build-640",
            accepted_commits={}, prepared_at_utc=STAMP,
        )
        assert result["executed"] is False and result["gpu"] is False and result["docker"] is False
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
        lease = result["lease_argv"]
        assert "--timeout-seconds" in lease
        assert lease[lease.index("--timeout-seconds") + 1] == "14460"
        cidfiles = [item for item in command if item.startswith("--cidfile=")]
        assert len(cidfiles) == 1 and Path(cidfiles[0].split("=", 1)[1]).is_absolute()
        assert stat.S_IMODE(Path(cidfiles[0].split("=", 1)[1]).parent.lstat().st_mode) == 0o700
        assert "--managed-docker-cidfile" in lease
        assert lease[lease.index("--managed-docker-cidfile") + 1] == cidfiles[0].split("=", 1)[1]
        manifest_fp = result["manifest"]["fingerprint_sha256"]
        verified_plan, manifest, predecessors, stream_fd, stream_pin = lane.verify_stage(
            plan_path=plan_path,
            accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13b-unit-build-640",
            accepted_manifest_fingerprint=manifest_fp, accepted_commits={},
        )
        try:
            assert verified_plan == plan
            assert predecessors == []
            assert manifest["worker_argv"][2].endswith("r13b_container.py")
            assert {item["role"] for item in manifest["entries"]} >= {
                "plan", "container_worker", "r13_worker_library",
                "shared_runtime_worker", "validation_init", "onnx",
            }
            assert stream_pin == result["stream"]
        finally:
            os.close(stream_fd)
    finally:
        monkeypatch.setattr(lane, "RUNS_ROOT", old_runs)
        monkeypatch.setattr(lane, "RESULTS_ROOT", old_results)
        snapshot = temporary / "results/runs/tensorrt_fp16_640/pose-r13b-unit-build-640/snapshot"
        if snapshot.exists():
            os.chmod(snapshot, 0o700)
        context.cleanup()


def test_stream_validation_rejects_one_byte_corruption_before_any_launcher(monkeypatch) -> None:
    context, temporary = _temporary_repo_directory()
    old_runs = lane.RUNS_ROOT
    old_results = lane.RESULTS_ROOT
    try:
        monkeypatch.setattr(lane, "RESULTS_ROOT", temporary / "results")
        monkeypatch.setattr(lane, "RUNS_ROOT", temporary / "results/runs")
        plan_path = temporary / "plan.json"
        plan = lane.prepare_plan(plan_path, prepared_at_utc=STAMP)
        result = lane.prepare_stage(
            plan_path=plan_path, accepted_plan_fingerprint=plan["fingerprint_sha256"],
            stage="tensorrt_fp16_640", run_id="pose-r13b-corrupt-build-640",
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
                stage="tensorrt_fp16_640", run_id="pose-r13b-corrupt-build-640",
                accepted_manifest_fingerprint=result["manifest"]["fingerprint_sha256"],
                accepted_commits={},
            )
    finally:
        monkeypatch.setattr(lane, "RUNS_ROOT", old_runs)
        monkeypatch.setattr(lane, "RESULTS_ROOT", old_results)
        snapshot = temporary / "results/runs/tensorrt_fp16_640/pose-r13b-corrupt-build-640/snapshot"
        if snapshot.exists():
            os.chmod(snapshot, 0o700)
        context.cleanup()


def test_inline_loader_is_syntax_valid_and_hashes_whole_stream_before_compile() -> None:
    compile(lane.INLINE_LOADER, "<pose-r13b-inline-loader-test>", "exec")
    source = lane.INLINE_LOADER
    assert source.index("digest.hexdigest()!=expected_sha") < source.index("compile(source")
    assert "sys.stdin.buffer.read" in source
    assert "whole snapshot stream pin differs" in source
    assert "/proc/" not in source


def test_bootstrap_manual_contract_rejects_unknown_field_before_copy() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    plan_pin = {key: lane.R13_PLAN_PIN[key] for key in ("path", "bytes", "sha256")}
    specs = [
        {"role": "plan", "path": lane.R13_PLAN, "pin": plan_pin, "destination": "/tmp/accepted/plan-r13b.json"},
        {"role": "container_worker", "path": lane.R13B_WORKER,
         "pin": lane._source_pin_from_plan(plan, "container_worker"),
         "destination": "/workspace/validation/pose_mmpose_yoloxpose_tensorrt_ds9_r13b_container.py"},
        {"role": "r13_worker_library", "path": lane.r13.WORKER,
         "pin": lane._source_pin_from_plan(plan, "r13_worker_library"),
         "destination": "/workspace/validation/pose_mmpose_yoloxpose_tensorrt_ds9_r13_container.py"},
        {"role": "shared_runtime_worker", "path": lane.r13.SHARED_WORKER,
         "pin": lane._source_pin_from_plan(plan, "shared_runtime_worker"),
         "destination": "/workspace/validation/person_rtdetrv4_tensorrt_r14b_container.py"},
        {"role": "validation_init", "path": lane.VALIDATION_INIT,
         "pin": lane._source_pin_from_plan(plan, "validation_init"),
         "destination": "/workspace/validation/__init__.py"},
        {"role": "onnx", "path": lane.ROOT / plan["upstream_r12c"]["profiles"]["640"]["onnx"]["path"],
         "pin": plan["upstream_r12c"]["profiles"]["640"]["onnx"], "destination": "/inputs/model.onnx"},
    ]
    manifest = lane.build_snapshot_manifest(
        plan=plan, plan_path=lane.R13_PLAN,
        stage="tensorrt_fp16_640", run_id="pose-r13b-bootstrap-unit",
        specs=specs, prepared_at_utc=STAMP,
    )
    manifest["silent_fallback"] = True
    manifest["fingerprint_sha256"] = bootstrap.fingerprint(manifest)
    with pytest.raises(bootstrap.SnapshotBootstrapR13BError, match="top-level fields"):
        bootstrap.validate_manifest(
            manifest, expected_fingerprint=manifest["fingerprint_sha256"],
            expected_plan_fingerprint=plan["fingerprint_sha256"],
            expected_bootstrap_bytes=plan["implementation"]["snapshot_bootstrap"]["bytes"],
            expected_bootstrap_sha256=plan["implementation"]["snapshot_bootstrap"]["sha256"],
        )


def test_private_worker_never_generates_or_executes_from_output_bind() -> None:
    source = lane.R13B_WORKER.read_text(encoding="utf-8")
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


def test_private_worker_rejects_direct_entry_without_bootstrap_acceptance(monkeypatch) -> None:
    for key in (
        "DEEPSAFE_POSE_R13B_SNAPSHOT_ACCEPTED",
        "DEEPSAFE_POSE_R13B_SNAPSHOT_RECEIPT",
        "DEEPSAFE_POSE_R13B_SNAPSHOT_RECEIPT_SHA256",
        "DEEPSAFE_POSE_R13B_SNAPSHOT_FINGERPRINT",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(worker.PoseWorkerR13BError, match="acceptance is missing"):
        worker.validate_snapshot_acceptance("a" * 64)


def test_private_worker_accepts_exact_bootstrap_receipt_with_negative_boundary_flags(
    monkeypatch, tmp_path: Path,
) -> None:
    plan_fingerprint = "a" * 64
    manifest_fingerprint = "b" * 64
    value = {
        "schema_version": "deepsafe.pose-mmpose-yoloxpose-snapshot-acceptance/r13b",
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
    path = tmp_path / "snapshot-acceptance-r13b.json"
    path.write_bytes(payload)
    os.chmod(path, 0o440)
    monkeypatch.setattr(worker, "ACCEPTANCE_PATH", path)
    monkeypatch.setenv("DEEPSAFE_POSE_R13B_SNAPSHOT_ACCEPTED", "1")
    monkeypatch.setenv("DEEPSAFE_POSE_R13B_SNAPSHOT_RECEIPT", str(path))
    monkeypatch.setenv("DEEPSAFE_POSE_R13B_SNAPSHOT_RECEIPT_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setenv("DEEPSAFE_POSE_R13B_SNAPSHOT_FINGERPRINT", manifest_fingerprint)
    observed = worker.validate_snapshot_acceptance(plan_fingerprint)
    assert observed["fingerprint_sha256"] == value["fingerprint_sha256"]


def test_snapshot_schema_rejects_proc_transport_and_worker_fallback() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
    plan_pin = {key: lane.R13_PLAN_PIN[key] for key in ("path", "bytes", "sha256")}
    specs = [
        {"role": "plan", "path": lane.R13_PLAN, "pin": plan_pin, "destination": "/tmp/accepted/plan-r13b.json"},
        {"role": "container_worker", "path": lane.R13B_WORKER,
         "pin": lane._source_pin_from_plan(plan, "container_worker"),
         "destination": "/workspace/validation/pose_mmpose_yoloxpose_tensorrt_ds9_r13b_container.py"},
        {"role": "r13_worker_library", "path": lane.r13.WORKER,
         "pin": lane._source_pin_from_plan(plan, "r13_worker_library"),
         "destination": "/workspace/validation/pose_mmpose_yoloxpose_tensorrt_ds9_r13_container.py"},
        {"role": "shared_runtime_worker", "path": lane.r13.SHARED_WORKER,
         "pin": lane._source_pin_from_plan(plan, "shared_runtime_worker"),
         "destination": "/workspace/validation/person_rtdetrv4_tensorrt_r14b_container.py"},
        {"role": "validation_init", "path": lane.VALIDATION_INIT,
         "pin": lane._source_pin_from_plan(plan, "validation_init"),
         "destination": "/workspace/validation/__init__.py"},
        {"role": "onnx", "path": lane.ROOT / plan["upstream_r12c"]["profiles"]["640"]["onnx"]["path"],
         "pin": plan["upstream_r12c"]["profiles"]["640"]["onnx"], "destination": "/inputs/model.onnx"},
    ]
    manifest = lane.build_snapshot_manifest(
        plan=plan, plan_path=lane.R13_PLAN,
        stage="tensorrt_fp16_640", run_id="pose-r13b-schema-unit",
        specs=specs, prepared_at_utc=STAMP,
    )
    changed = copy.deepcopy(manifest)
    changed["stream_contract"]["proc_fd_paths_in_argv"] = True
    changed["fingerprint_sha256"] = lane.fingerprint(changed)
    with pytest.raises(lane.PoseR13BError, match="schema validation failed"):
        lane.validate_schema(changed, lane.SNAPSHOT_SCHEMA)


def test_receipt_and_commit_builders_bind_snapshot_private_runtime_and_margins() -> None:
    plan = lane.build_plan(prepared_at_utc=STAMP)
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
        plan=plan, plan_path=lane.R13_PLAN, manifest_pin=document,
        stream_pin=artifact, stage="tensorrt_fp16_640",
        run_id="pose-r13b-builder-unit", worker_pin=document,
        worker_outputs=[artifact], primary_published=artifact,
        predecessors=[], numerical=None, lifecycle=lifecycle,
        created_at_utc=STAMP,
    )
    assert receipt["execution"]["gpu_lease_timeout_seconds"] == 14460
    assert receipt["execution"]["launcher_timeout_seconds"] == 14520
    assert receipt["execution"]["private_work_tmpfs_for_build_deserialize_compile_execute_and_parse"] is True
    assert receipt["execution"]["output_bind_final_verified_no_overwrite_only"] is True
    assert receipt["execution"]["managed_docker_cidfile_path_handshake"] is True
    assert receipt["execution"]["docker_cidfile_parent_owner_uid_mode_0700"] is True
    commit = lane.build_commit(
        plan=plan, plan_path=lane.R13_PLAN, manifest_pin=document,
        stage="tensorrt_fp16_640", run_id="pose-r13b-builder-unit",
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
    assert plan["claim_boundary"]["r13b_gpu_executed"] is False
    assert invoked == []
